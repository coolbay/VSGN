import sys
import numpy as np
import pandas as pd
import json
import os
from joblib import Parallel, delayed
from .eval_detection import ANETdetection

from Utils import opts_thumos

thumos_class = {
    7 : 'BaseballPitch',
    9 : 'BasketballDunk',
    12: 'Billiards',
    21: 'CleanAndJerk',
    22: 'CliffDiving',
    23: 'CricketBowling',
    24: 'CricketShot',
    26: 'Diving',
    31: 'FrisbeeCatch',
    33: 'GolfSwing',
    36: 'HammerThrow',
    40: 'HighJump',
    45: 'JavelinThrow',
    51: 'LongJump',
    68: 'PoleVault',
    79: 'Shotput',
    85: 'SoccerPenalty',
    92: 'TennisSwing',
    93: 'ThrowDiscus',
    97: 'VolleyballSpiking',
}
# thumos_class = {
#     1: 'BaseballPitch',
#     2: 'BasketballDunk',
#     3: 'Billiards',
#     4: 'CleanAndJerk',
#     5: 'CliffDiving',
#     6: 'CricketBowling',
#     7: 'CricketShot',
#     8: 'Diving',
#     9: 'FrisbeeCatch',
#     10: 'GolfSwing',
#     11: 'HammerThrow',
#     12: 'HighJump',
#     13: 'JavelinThrow',
#     14: 'LongJump',
#     15: 'PoleVault',
#     16: 'Shotput',
#     17: 'SoccerPenalty',
#     18: 'TennisSwing',
#     19: 'ThrowDiscus',
#     20: 'VolleyballSpiking'
# }
def IOU(s1, e1, s2, e2):
    if (s2 > e1) or (s1 > e2):
        return 0
    Aor = max(e1, e2) - min(s1, s2)
    Aand = min(e1, e2) - max(s1, s2)
    return float(Aand) / Aor

def Soft_NMS(df, nms_threshold=1e-5, num_prop=200):
    '''
    From BSN code
    :param df:
    :param nms_threshold:
    :return:
    '''
    df = df.sort_values(by="score", ascending=False)

    tstart = list(df.xmin.values[:])
    tend = list(df.xmax.values[:])
    tscore = list(df.score.values[:])
    tlabel = list(df.label.values[:])

    rstart = []
    rend = []
    rscore = []
    rlabel = []

    # I use a trick here, remove the detection XDD
    # which is longer than 300
    for idx in range(0, len(tscore)):
        if tend[idx] - tstart[idx] >= 300:
            tscore[idx] = 0

    while len(tscore) > 1 and len(rscore) < num_prop and max(tscore)>0:
        max_index = tscore.index(max(tscore))
        for idx in range(0, len(tscore)):
            if idx != max_index:
                tmp_iou = IOU(tstart[max_index], tend[max_index], tstart[idx], tend[idx])
                if tmp_iou > 0:
                    tscore[idx] = tscore[idx] * np.exp(-np.square(tmp_iou) / nms_threshold)

        rstart.append(tstart[max_index])
        rend.append(tend[max_index])
        rscore.append(tscore[max_index])
        rlabel.append(tlabel[max_index])

        tstart.pop(max_index)
        tend.pop(max_index)
        tscore.pop(max_index)
        tlabel.pop(max_index)

    newDf = pd.DataFrame()
    newDf['score'] = rscore
    newDf['xmin'] = rstart
    newDf['xmax'] = rend
    newDf['label'] = rlabel
    return newDf

def _gen_detection_video(video_name, thu_label_id, result, idx_classes, opt, num_prop=200, topk = 2):
    path = os.path.join(opt["output_path"], opt["prop_path"]) + "/"
    files = [path + f for f in os.listdir(path) if  video_name in f]
    if len(files) == 0:
        print('Missing result for video {}'.format(video_name)) # video_test_0001292, video_test_0000504
        return {video_name:[]}

    dfs = []  # merge pieces of video
    for snippet_file in files:
        snippet_df = pd.read_csv(snippet_file)
        snippet_df = Soft_NMS(snippet_df, nms_threshold=opt['nms_alpha_detect'])
        dfs.append(snippet_df)
    df = pd.concat(dfs)
    if len(df) > 1:
        df = Soft_NMS(df, nms_threshold=opt['nms_alpha_detect'])
    df = df.sort_values(by="score", ascending=False)

    num_frames = result['num_frames']
    proposal_list = []
    for j in range(min(num_prop, len(df))):
        for k in range(topk):
            tmp_proposal = {}
            tmp_proposal["label"] = idx_classes[int(df.label.values[j])]
            tmp_proposal["score"] = float(round(df.score.values[j], 6))
            tmp_proposal["segment"] = [float(round(max(0, df.xmin.values[j]), 1)),
                                       float(round(min(num_frames, df.xmax.values[j]), 1))]
            proposal_list.append(tmp_proposal)
    return {video_name:proposal_list}

def gen_detection_multicore(opt):
    # get video list
    thumos_test_anno = pd.read_csv(opt['video_info'].split('thumos14')[:-1][0] + "test_Annotation.csv")
    tmp = thumos_test_anno.video[thumos_test_anno.video!='video_test_0001292']
    tmp = tmp[tmp!='video_test_0000504']
    video_list = tmp.unique()
    thu_label_id = np.sort(thumos_test_anno.type_idx.unique())[1:] - 1  # get thumos class id
    thu_video_id = np.array([int(i[-4:]) - 1 for i in video_list])  # -1 is to match python index

    # # load video level classification
    # cls_data = np.load(opt['vlevel_cls_res'])
    # cls_data = cls_data[thu_video_id,:][:, thu_label_id]  # order by video list, output 213x20

    # load all categories
    if os.path.exists(opt["thumos_classes"]):
        with open(opt["thumos_classes"], 'r') as f:
            classes = json.load(f)

    idx_classes = {}
    for key, value in classes.items():
        idx_classes[value] = key

    # detection_result
    thumos_gt = pd.read_csv(opt['video_info'])

    result = {
        video:
            {
                # 'fps': thumos_gt.loc[thumos_gt['video-name'] == video]['frame-rate'].values[0],
                'num_frames': thumos_gt.loc[thumos_gt['video-name'] == video]['video-frames'].values[0]
            }
        for video in video_list
    }

    parallel = Parallel(n_jobs=20, prefer="processes")
    detection = parallel(delayed(_gen_detection_video)(video_name, thu_label_id, result[video_name], idx_classes, opt)
                        for video_name in video_list)
    detection_dict = {}
    [detection_dict.update(d) for d in detection]
    output_dict = {"version": "THUMOS14", "results": detection_dict, "external_data": {}}

    with open(os.path.join(opt["output_path"], opt["detect_result_file"]), "w") as out:
        json.dump(output_dict, out)


