import sys
sys.dont_write_bytecode = True
import os
import torch
import torch.nn.parallel
import torch.optim as optim
import numpy as np
from tensorboardX import SummaryWriter
import Utils.opts as opts
from Utils.dataset_thumos import VideoDataSet as VideoDataSet_thumos
from Utils.dataset_activitynet import VideoDataSet as VideoDataSet_anet
from Utils.dataset_hacs import VideoDataSet as VideoDataSet_hacs
from Models.SegTAD import SegTAD
from Utils.loss_function import  detect_loss_fuction, get_mask, boundary_loss_function, bi_loss
import datetime
import torch.nn as nn
import matplotlib.pyplot as plt

torch.manual_seed(21)

class BatchCollator(object):
    """
    From a list of samples from the dataset,
    returns the batched images and targets.
    This should be passed to the DataLoader
    """

    def __init__(self, size_divisible=0):
        self.size_divisible = size_divisible

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        video_data = torch.stack(transposed_batch[0])
        match_score_action = torch.stack(transposed_batch[1])
        match_score_start = torch.stack(transposed_batch[2])
        match_score_end = torch.stack(transposed_batch[3])
        gt_iou_map = torch.stack(transposed_batch[4])
        gt_bbox = transposed_batch[5]
        return video_data, match_score_action, match_score_start, match_score_end, gt_iou_map, gt_bbox

def Train_SegTAD(opt):
    path_appendix = '_'.join(string for string in opt['checkpoint_path'].split('_')[1:])
    writer = SummaryWriter(logdir='runs/' + path_appendix)
    model = SegTAD(opt)
    device = "cuda"
    model = torch.nn.DataParallel(model)
    model.to(device)

    if opt['dif_lr'] == 'true':
        ig_params1 = list(map(id, model.module.FeatureEnhancer.parameters()))
        ig_params2 = list(map(id, model.module.last_conv_logits.parameters()))
        ig_params = ig_params1 + ig_params2
        base_params = filter(lambda p: id(p) not in ig_params, model.parameters())

        optimizer = optim.Adam([
            {'params': base_params},
            {'params': model.module.FeatureEnhancer.parameters(), 'lr': opt["FeatureEnhancer_lr"]},
            {'params': model.module.last_conv_logits.parameters(), 'lr': opt["segmentor_lr"]},
        ], lr=opt["train_lr"], weight_decay=opt["weight_decay"])

    else:
        optimizer = optim.Adam(model.parameters(), lr=opt["train_lr"], weight_decay=opt["weight_decay"])

    if opt['resume']:
        if os.path.exists(opt["checkpoint_path"] + "/checkpoint.pth.tar"): #CHANGE
            checkpoint = torch.load(opt["checkpoint_path"] + "/checkpoint.pth.tar")
            base_dict = {}
            for k, v in list(checkpoint['state_dict'].items()):
                # base_dict = {'.'.join(k.split('.')[1:]): v }
                base_dict.setdefault('.'.join(k.split('.')[1:]), v)
            model.load_state_dict(base_dict, strict=False)
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            print('Load pretrained model successfully!')
        else:
            start_epoch = 0
    else:
        start_epoch = 0



    if opt['dataset'] == 'activitynet':
        VideoDataSet = VideoDataSet_anet
    elif opt['dataset'] == 'thumos':
        VideoDataSet = VideoDataSet_thumos
    elif opt['dataset'] == 'hacs':
        VideoDataSet = VideoDataSet_hacs

    # device_id = 0,1
    # torch.cuda.set_device(torch.device("cuda:" + str(device_id) if torch.cuda.is_available() else "cpu"))
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    kwargs = {'num_workers': 16, 'pin_memory': True, 'drop_last': True}

    train_loader = torch.utils.data.DataLoader(VideoDataSet(opt, subset="train"),
                                               batch_size=opt["batch_size"], shuffle=True,
                                               **kwargs)

    test_loader = torch.utils.data.DataLoader(VideoDataSet(opt, subset="validation"),
                                              batch_size=opt["batch_size"], shuffle=False,
                                              **kwargs)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt["step_size"], gamma=opt["step_gamma"])

    bm_mask = get_mask(opt["prop_temporal_scale"])
    for epoch in range(start_epoch, opt["num_epoch"]):

        train_SegTAD_epoch(train_loader, model, optimizer, epoch, writer, opt, bm_mask)
        epoch_loss = test_SegTAD_epoch(test_loader, model, epoch, writer, opt, bm_mask)

        print((datetime.datetime.now()))
        state = {'epoch': epoch + 1,
                 'state_dict': model.state_dict(),
                 'optimizer': optimizer.state_dict()}
        torch.save(state, opt["checkpoint_path"] + "/checkpoint.pth.tar")
        if epoch_loss < model.module.tem_best_loss:
            print((datetime.datetime.now()))
            print('The best model up to now is from Epoch {}'.format(epoch))
            model.module.tem_best_loss = np.mean(epoch_loss)
            torch.save(state, opt["checkpoint_path"] + "/best.pth.tar")

        scheduler.step()
    writer.close()

# by Catherine
def get_pos_area(pos, all_area):
    area = 4 * (pos[:, 2] - pos[:, 1])
    all_area.extend(area)

# by Catherine
def get_targets_area(gt_bbox, num_gt, all_area):
    for i, gt in enumerate(gt_bbox):
        area = 1000 * (gt[:num_gt[i], 1] - gt[:num_gt[i], 0])
        all_area.extend(area)

# by Catherine
def plot_dist_targets(all_area):
    # plt.ion()
    plt.hist(all_area, bins=100, range=(0,1000))
    plt.show(block=False)
    # plt.close('all')


def train_SegTAD_epoch(data_loader, model, optimizer, epoch, writer, opt, bm_mask):
    model.train()

    epoch_loss = 0
    epoch_loss_stage0_cls = 0
    epoch_loss_stage0_reg = 0
    epoch_loss_stage1_cls = 0
    epoch_loss_stage1_reg = 0
    epoch_loss_stage2_act = 0
    epoch_loss_stage2_bd = 0
    epoch_loss_stage2_reg = 0
    r1, r2, r3 = opt['loss_ratios']

    # all_gt_area = []  # by Catherine
    all_pos_area = []  # by Catherine
    for n_iter, (input_data, gt_action, gt_start, gt_end, gt_bbox, num_gt, num_frms) in enumerate(data_loader):

        # get_targets_area(gt_bbox, num_gt, all_gt_area) # by Catherine
        # if n_iter % 100 == 0:
        #     plot_dist_targets(all_gt_area) # by Catherine

        losses, pred_action, pred_start, pred_end = model(input_data, num_frms, gt_bbox, num_gt)

        # get_pos_area(pos_idx_st_end.detach().cpu(), all_pos_area) # by Catherine
        # if n_iter % 100 == 0:
        #     plot_dist_targets(all_pos_area) # by Catherine

        # Loss2a: actionness
        if opt['binary_actionness'] == 'true':
            cost_actionness = bi_loss(pred_action[:,0,:], gt_action.cuda())
        else:
            criterion = nn.CrossEntropyLoss()
            cost_segmentation = criterion(pred_action, gt_action.cuda().long())

        # Loss2b: start/end loss
        cost_boundary = boundary_loss_function(pred_start.cuda(), pred_end, gt_start.cuda(), gt_end.cuda())
        # cost_boundary = torch.zeros((1,), device=pred_bd_map.device)

        # Overall loss
        loss_stage0_cls = torch.mean(losses['loss_cls_enc'])
        loss_stage0_reg = torch.mean(losses['loss_reg_enc'])
        loss_stage1_cls = torch.mean(losses['loss_cls_dec'])
        loss_stage1_reg = torch.mean(losses['loss_reg_dec'])
        loss_stage2_act = cost_actionness
        loss_stage2_bd = cost_boundary
        loss_stage2_reg = torch.mean(losses['loss_reg_st2'])

        loss = loss_stage1_cls + loss_stage1_reg  \
               + loss_stage2_act + loss_stage2_bd + loss_stage2_reg

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.cpu().detach().numpy()
        epoch_loss_stage0_cls += loss_stage0_cls.cpu().detach().numpy()
        epoch_loss_stage0_reg += loss_stage0_reg.cpu().detach().numpy()
        epoch_loss_stage1_cls += loss_stage1_cls.cpu().detach().numpy()
        epoch_loss_stage1_reg += loss_stage1_reg.cpu().detach().numpy()
        epoch_loss_stage2_act += loss_stage2_act.cpu().detach().numpy()
        epoch_loss_stage2_bd += loss_stage2_bd.cpu().detach().numpy()
        epoch_loss_stage2_reg += loss_stage2_reg.cpu().detach().numpy()


    # plot_dist_targets(all_gt_area) # by Catherine

    writer.add_scalars('data/loss', {'train': epoch_loss / (n_iter + 1)}, epoch)
    writer.add_scalars('data/loss_stage0_cls', {'train': epoch_loss_stage0_cls / (n_iter + 1)}, epoch)
    writer.add_scalars('data/loss_stage0_reg', {'train': epoch_loss_stage0_reg / (n_iter + 1)}, epoch)
    writer.add_scalars('data/loss_stage1_cls', {'train': epoch_loss_stage1_cls / (n_iter + 1)}, epoch)
    writer.add_scalars('data/loss_stage1_reg', {'train': epoch_loss_stage1_reg / (n_iter + 1)}, epoch)
    writer.add_scalars('data/loss_stage2_act', {'train': epoch_loss_stage2_act / (n_iter + 1)}, epoch)
    writer.add_scalars('data/loss_stage2_bd', {'train': epoch_loss_stage2_bd / (n_iter + 1)}, epoch)
    writer.add_scalars('data/loss_stage2_reg', {'train': epoch_loss_stage2_reg / (n_iter + 1)}, epoch)

    print("Training loss (epoch %d): "
          "total %.04f, "
          "0 cls: %.04f, "
          "0 reg: %.04f, "
          "1 cls: %.04f, "
          "1 reg: %.04f, "
          "2 act: %.04f, "
          "2 bd: %.04f, "
          "2 reg: %.04f, " % (
              epoch,
              epoch_loss / (n_iter + 1),
              epoch_loss_stage0_cls/(n_iter + 1),
              epoch_loss_stage0_reg/(n_iter + 1),
              epoch_loss_stage1_cls/(n_iter + 1),
              epoch_loss_stage1_reg/(n_iter + 1),
              epoch_loss_stage2_act/(n_iter + 1),
              epoch_loss_stage2_bd/(n_iter + 1),
              epoch_loss_stage2_reg/(n_iter + 1),
          ))
    # print((datetime.datetime.now()))
    # state = {'epoch': epoch + 1,
    #          'state_dict': model.state_dict()}
    # torch.save(state, opt["checkpoint_path"] + "/checkpoint" + str(epoch) + ".pth.tar")


def test_SegTAD_epoch(data_loader, model, epoch, writer, opt, bm_mask):
    model.eval()

    epoch_loss = 0
    epoch_loss_stage0_cls = 0
    epoch_loss_stage0_reg = 0
    epoch_loss_stage1_cls = 0
    epoch_loss_stage1_reg = 0
    epoch_loss_stage2_act = 0
    epoch_loss_stage2_bd = 0
    epoch_loss_stage2_reg = 0
    r1, r2, r3 = opt['loss_ratios']

    for n_iter, (input_data, gt_action, gt_start, gt_end, gt_bbox, num_gt, num_frms) in enumerate(data_loader):
        losses, pred_action, pred_start, pred_end = model(input_data, num_frms, gt_bbox, num_gt)

        # Loss2a: actionness
        if opt['binary_actionness'] == 'true':
            cost_actionness = bi_loss(pred_action[:,0,:], gt_action.cuda())
        else:
            criterion = nn.CrossEntropyLoss()
            cost_segmentation = criterion(pred_action, gt_action.cuda().long())

        # Loss2b: start/end loss
        cost_boundary = boundary_loss_function(pred_start.cuda(), pred_end, gt_start.cuda(), gt_end.cuda())
        # cost_boundary = torch.zeros((1,), device=pred_bd_map.device)

        # Overall loss
        loss_stage0_cls = torch.mean(losses['loss_cls_enc'])
        loss_stage0_reg = torch.mean(losses['loss_reg_enc'])
        loss_stage1_cls = torch.mean(losses['loss_cls_dec'])
        loss_stage1_reg = torch.mean(losses['loss_reg_dec'])
        loss_stage2_act = cost_actionness
        loss_stage2_bd = cost_boundary
        loss_stage2_reg = torch.mean(losses['loss_reg_st2'])

        loss = loss_stage1_cls + loss_stage1_reg \
               + loss_stage2_act + loss_stage2_bd + loss_stage2_reg

        epoch_loss += loss.cpu().detach().numpy()
        epoch_loss_stage0_cls += loss_stage0_cls.cpu().detach().numpy()
        epoch_loss_stage0_reg += loss_stage0_reg.cpu().detach().numpy()
        epoch_loss_stage1_cls += loss_stage1_cls.cpu().detach().numpy()
        epoch_loss_stage1_reg += loss_stage1_reg.cpu().detach().numpy()
        epoch_loss_stage2_act += loss_stage2_act.cpu().detach().numpy()
        epoch_loss_stage2_bd += loss_stage2_bd.cpu().detach().numpy()
        epoch_loss_stage2_reg += loss_stage2_reg.cpu().detach().numpy()


    writer.add_scalars('data/loss', {'test': epoch_loss / (n_iter + 1)}, epoch)
    writer.add_scalars('data/loss_stage0_cls', {'test': epoch_loss_stage0_cls / (n_iter + 1)}, epoch)
    writer.add_scalars('data/loss_stage0_reg', {'test': epoch_loss_stage0_reg / (n_iter + 1)}, epoch)
    writer.add_scalars('data/loss_stage1_cls', {'test': epoch_loss_stage1_cls / (n_iter + 1)}, epoch)
    writer.add_scalars('data/loss_stage1_reg', {'test': epoch_loss_stage1_reg / (n_iter + 1)}, epoch)
    writer.add_scalars('data/loss_stage2_act', {'test': epoch_loss_stage2_act / (n_iter + 1)}, epoch)
    writer.add_scalars('data/loss_stage2_bd', {'test': epoch_loss_stage2_bd / (n_iter + 1)}, epoch)
    writer.add_scalars('data/loss_stage2_reg', {'test': epoch_loss_stage2_reg / (n_iter + 1)}, epoch)

    print("Testting loss (epoch %d): "
          "total %.04f, "
          "0 cls: %.04f, "
          "0 reg: %.04f, "
          "1 cls: %.04f, "
          "1 reg: %.04f, "
          "2 act: %.04f, "
          "2 bd: %.04f, "
          "2 reg: %.04f, " % (
              epoch,
              epoch_loss / (n_iter + 1),
              epoch_loss_stage0_cls/(n_iter + 1),
              epoch_loss_stage0_reg/(n_iter + 1),
              epoch_loss_stage1_cls/(n_iter + 1),
              epoch_loss_stage1_reg/(n_iter + 1),
              epoch_loss_stage2_act/(n_iter + 1),
              epoch_loss_stage2_bd/(n_iter + 1),
              epoch_loss_stage2_reg/(n_iter + 1),
          ))

    return epoch_loss




if __name__ == '__main__':

    print(datetime.datetime.now())

    opt = opts.parse_opt()
    opt = vars(opt)

    if not os.path.exists(opt["checkpoint_path"]):
        os.makedirs(opt["checkpoint_path"])

    print(opt)

    print("---------------------------------------------------------------------------------------------")
    print("Training starts!")
    print("---------------------------------------------------------------------------------------------")
    Train_SegTAD(opt)
    print("Training finishes!")

    print(datetime.datetime.now())

