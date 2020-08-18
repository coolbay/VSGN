import sys
sys.dont_write_bytecode = True
import torch
import torch.nn.parallel
from Utils.dataset_thumos import VideoDataSet as VideoDataSet_thumos
from Utils.dataset_activitynet import VideoDataSet as VideoDataSet_anet
from Utils.dataset_hacs import VideoDataSet as VideoDataSet_hacs
from Models.SegTAD import SegTAD
import pandas as pd
from joblib import Parallel, delayed
import pickle
import sys
sys.dont_write_bytecode = True
import torch.nn.parallel
import os
import Utils.opts as opts

import datetime
import numpy as np

torch.manual_seed(21)


# Infer all data
def Infer_SegTAD(opt):
    model = SegTAD(opt)
    model = torch.nn.DataParallel(model).cuda()
    if not os.path.exists(opt["checkpoint_path"] + "/best.pth.tar"):
        print("There is no checkpoint. Please train first!!!")
    else:
        checkpoint = torch.load(opt["checkpoint_path"] + "/best.pth.tar")
        model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    proposal_path = os.path.join(opt["output_path"], opt["prop_path"])
    actionness_path = os.path.join(opt["output_path"], opt["actionness_path"])
    start_end_path = os.path.join(opt["output_path"], opt["start_end_path"])
    prop_map_path = os.path.join(opt["output_path"], opt["prop_map_path"])


    if not os.path.exists(proposal_path):
        os.mkdir(proposal_path)
    if not os.path.exists(actionness_path):
        os.mkdir(actionness_path)
    if not os.path.exists(start_end_path):
        os.mkdir(start_end_path)
    if not os.path.exists(prop_map_path):
        os.mkdir(prop_map_path)

    if opt['dataset'] == 'activitynet':
        VideoDataSet = VideoDataSet_anet
    elif opt['dataset'] == 'thumos':
        VideoDataSet = VideoDataSet_thumos
    elif opt['dataset'] == 'hacs':
        VideoDataSet = VideoDataSet_hacs

    test_loader = torch.utils.data.DataLoader(VideoDataSet(opt, subset="validation", mode="inference"),
                                              batch_size=opt["batch_size"], shuffle=False,
                                              num_workers=10, pin_memory=True, drop_last=False)

    with torch.no_grad():
        # for index_list, input_data, gt_actionness, gt_act_map in test_loader:
        for index_list, input_data in test_loader:
            infer_batch_selectprop(model, index_list, input_data, test_loader, proposal_path,
                                   actionness_path, start_end_path, prop_map_path)


# Infer one batch of data, for the model with proposal selection
def infer_batch_selectprop(model,
                           index_list,
                           input_data,
                           test_loader,
                           proposal_path,
                           actionness_path,
                           start_end_path,
                           prop_map_path,):

    gt_act_map = torch.zeros((input_data.shape[0],), device=input_data.device)
    loc_pred, score_pred, pred_action, pred_start, pred_end = model(input_data.cuda(), gt_act_map)


    # Move variables to output to CPU
    loc_pred_batch = loc_pred.detach().cpu().numpy()
    score_pred_batch = score_pred.detach().cpu().numpy()
    pred_action_batch = pred_action.detach().cpu().numpy()
    pred_start_batch = pred_start.detach().cpu().numpy()
    pred_end_batch = pred_end.detach().cpu().numpy()



    Parallel(n_jobs=len(index_list))(
        delayed(infer_v_asis)(
            opt,
            video=list(test_loader.dataset.video_list)[full_idx],
            score_pred_v = score_pred_batch[batch_idx],
            loc_pred_v = loc_pred_batch[batch_idx],
            pred_action_v = pred_action_batch[batch_idx],
            pred_start_v = pred_start_batch[batch_idx],
            pred_end_v = pred_end_batch[batch_idx],
            proposal_path = proposal_path,
            actionness_path = actionness_path,
            start_end_path = start_end_path,
            prop_map_path = prop_map_path

        ) for batch_idx, full_idx in enumerate(index_list))



    # # For debug: one process
    # for batch_idx, full_idx in enumerate(index_list):
    #     infer_v_asis(
    #             opt,
    #         video=list(test_loader.dataset.video_list)[full_idx],
    #         score_pred_v = score_pred_batch[batch_idx],
    #         loc_pred_v = loc_pred_batch[batch_idx],
    #         pred_action_v = pred_action_batch[batch_idx],
    #         pred_start_v = pred_start_batch[batch_idx],
    #         pred_end_v = pred_end_batch[batch_idx],
    #         proposal_path = proposal_path,
    #         actionness_path = actionness_path,
    #         start_end_path = start_end_path,
    #         prop_map_path = prop_map_path
    #     )



def infer_v_asis(*args, **kwargs):


    prop_tscale = args[0]["temporal_scale"]
    loc_pred_v = kwargs['loc_pred_v']
    score = kwargs['score_pred_v']
    proposal_path = kwargs['proposal_path']

    if opt['dataset'] == 'activitynet' or opt['dataset'] == 'hacs':
        video_name = kwargs['video']
    elif opt['dataset'] == 'thumos':
        video_name = kwargs['video']['rgb'].split('/')[-1]
        win_start, win_end = kwargs['video']['frames'][0][1:3]
        offset = kwargs['video']['win_idx']


    if False:
        loc_pred_v[:, -1] = loc_pred_v[:, -1] - 1
    elif False:
        correct_idx = ((loc_pred_v[:, -1] < prop_tscale) * (loc_pred_v[:, 0] >=0)).nonzero()
        loc_pred_v = loc_pred_v[correct_idx]
        score = score[correct_idx]
    else:
        loc_pred_v[:,0] = loc_pred_v[:,0].clip(min=0)
        loc_pred_v[:,1] = loc_pred_v[:,1].clip(max=prop_tscale)

    if opt['dataset'] == 'activitynet' or opt['dataset'] == 'hacs':
        new_props = np.concatenate((loc_pred_v/prop_tscale, score[:, None], score[:, None], score[:, None]), axis=1)

    col_name = ["xmin", "xmax", "clr_score", "reg_socre", "score"]
    new_df = pd.DataFrame(new_props, columns=col_name)

    if opt['dataset'] == 'activitynet' or opt['dataset'] == 'hacs':
        path = proposal_path + "/" + video_name + ".csv"
    elif opt['dataset'] == 'thumos':
        path = proposal_path + "/" + video_name + '_' + str(offset) + ".csv"
    new_df.to_csv(path, index=False)





if __name__ == '__main__':

    opt = opts.parse_opt()
    opt = vars(opt)

    print(opt)

    if not os.path.exists(opt["output_path"]):
        os.makedirs(opt["output_path"])

    # 1. Run inference
    print(datetime.datetime.now())
    print("---------------------------------------------------------------------------------------------")
    print("1. Inference starts!")
    print("---------------------------------------------------------------------------------------------")

    Infer_SegTAD(opt)

    print("Inference finishes! \n")

