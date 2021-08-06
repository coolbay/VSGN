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
from Models.SegTAD import SegTAD
from Utils.loss_function import boundary_loss_function, bi_loss
import datetime
from collections import defaultdict

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

    optimizer = optim.Adam(model.parameters(), lr=opt["train_lr"], weight_decay=opt["weight_decay"])

    start_epoch = 0

    kwargs = {'num_workers': 16, 'pin_memory': True, 'drop_last': True}

    train_loader = torch.utils.data.DataLoader(VideoDataSet_thumos(opt, subset="train"),
                                               batch_size=opt["batch_size"], shuffle=True,
                                               **kwargs)

    test_loader = torch.utils.data.DataLoader(VideoDataSet_thumos(opt, subset="validation"),
                                              batch_size=opt["batch_size"], shuffle=False,
                                              **kwargs)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt["step_size"], gamma=opt["step_gamma"])

    for epoch in range(start_epoch, opt["num_epoch"]):

        train_SegTAD_epoch(train_loader, model, optimizer, epoch, writer, opt)
        epoch_loss = test_SegTAD_epoch(test_loader, model, epoch, writer, opt)

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


def train_SegTAD_epoch(data_loader, model, optimizer, epoch, writer, opt, is_train=True):

    if is_train:
        model.train()
    else:
        model.eval()

    epoch_losses = defaultdict(float)
    for n_iter, (input_data, gt_action, gt_start, gt_end, gt_bbox, num_gt, num_frms) in enumerate(data_loader):
        with torch.set_grad_enabled(is_train):
            losses, pred_action, pred_start, pred_end = model(input_data, num_frms, gt_bbox, num_gt)

            # Loss2a: actionness
            cost_actionness = bi_loss(pred_action[:,0,:], gt_action.cuda())

            # Loss2b: start/end loss
            cost_boundary = boundary_loss_function(pred_start.cuda(), pred_end, gt_start.cuda(), gt_end.cuda())

        # Overall loss
        loss_stage1_cls = torch.mean(losses['loss_cls_dec'])
        loss_stage1_reg = torch.mean(losses['loss_reg_dec'])
        loss_stage2_act = cost_actionness
        loss_stage2_bd = cost_boundary
        loss_stage2_reg = torch.mean(losses['loss_reg_st2'])

        loss = loss_stage1_cls + loss_stage1_reg + 0.2*loss_stage2_act + 0.2*loss_stage2_bd + loss_stage2_reg

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_losses['loss'] += loss.cpu().detach().numpy()
        epoch_losses['loss_stage1_cls'] += loss_stage1_cls.cpu().detach().numpy()
        epoch_losses['loss_stage1_reg'] += loss_stage1_reg.cpu().detach().numpy()
        epoch_losses['loss_stage2_act'] += loss_stage2_act.cpu().detach().numpy()
        epoch_losses['loss_stage2_bd'] += loss_stage2_bd.cpu().detach().numpy()
        epoch_losses['loss_stage2_reg'] += loss_stage2_reg.cpu().detach().numpy()

    for k, v in epoch_losses.items():
        epoch_losses[k] = v / (n_iter + 1)

    to_print = ["%s loss (epoch %d): " % ('Train' if is_train else 'Val', epoch)]
    for k, v in epoch_losses.items():
        writer.add_scalar('%s/%s' % ('train' if is_train else 'val', k), v, epoch)
        writer.flush()
        to_print.append('%s: %.04f' % (k, v))
    print(' '.join(to_print))

    return epoch_losses['loss']



def test_SegTAD_epoch(data_loader, model, epoch, writer, opt):
    return train_SegTAD_epoch(data_loader, model, None, epoch, writer, opt, is_train=False)




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

