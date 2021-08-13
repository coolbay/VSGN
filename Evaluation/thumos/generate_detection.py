
import numpy as np
import pandas as pd
import json
import os
from joblib import Parallel, delayed
from scipy.io import loadmat

def IOU(s1, e1, s2, e2):
    if (s2 > e1) or (s1 > e2):
        return 0
    Aor = max(e1, e2) - min(s1, s2)
    Aand = min(e1, e2) - max(s1, s2)
    return float(Aand) / (Aor+0.0000001)

def Soft_NMS(df, nms_threshold=1e-5, num_prop=200):
    '''
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

    # remove the detection longer than 50
    for idx in range(0, len(tscore)):
        if tend[idx] - tstart[idx] >= 50:
            tscore[idx] = 0

    while len(tscore) > 1 and len(rscore) < num_prop and max(tscore)>0:
        max_index = tscore.index(max(tscore))
        if tlabel[max_index] == 0:
            tstart.pop(max_index)
            tend.pop(max_index)
            tscore.pop(max_index)
            tlabel.pop(max_index)
            continue

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

def _gen_detection_video(video_name, result, idx_classes, opt, num_prop=200, topk = 2):
    path = os.path.join(opt["output_path"], opt["prop_path"]) + "/"
    files = [path + f for f in os.listdir(path) if  video_name in f]
    if len(files) == 0:
        print('Missing result for video {}'.format(video_name)) # video_test_0001292, video_test_0000504
        return {video_name:[]}

    dfs = []  # merge pieces of video
    for snippet_file in files:
        snippet_df = pd.read_csv(snippet_file)
        dfs.append(snippet_df)
    df = pd.concat(dfs)
    df = Soft_NMS(df, nms_threshold=opt['nms_alpha_detect'])
    df = df.sort_values(by="score", ascending=False)

    duration = result['duration']
    proposal_list = []
    for j in range(min(num_prop, len(df))):
        tmp_proposal = {}
        tmp_proposal["label"] = idx_classes[int(df.label.values[j])]
        tmp_proposal["score"] = float(round(df.score.values[j], 6))
        tmp_proposal["segment"] = [float(round(max(0, df.xmin.values[j]), 1)),
                                   float(round(min(duration, df.xmax.values[j]), 1))]
        proposal_list.append(tmp_proposal)

    return {video_name:proposal_list}

def gen_detection_multicore(opt):
    with open(opt["video_anno"], 'r') as fobj:
        data = json.load(fobj)
    video_list = []
    for vid, v in data['database'].items():
        if v['subset'] == 'test':
            video_list.append(vid)

    # load all categories
    if os.path.exists(opt["thumos_classes"]):
        with open(opt["thumos_classes"], 'r') as f:
            classes = json.load(f)

    idx_classes = {}
    for key, value in classes.items():
        idx_classes[value] = key

    info_file = opt['test_video_info']
    anno_df = loadmat(info_file)['videos'][0]
    result = {
        vid:
            {
                'duration': anno_df[int(vid[-4:]) - 1][5][0][0]
            }
        for vid in video_list
    }
    parallel = Parallel(n_jobs=16, prefer="processes")
    detection = parallel(delayed(_gen_detection_video)(video_name, result[video_name], idx_classes, opt)
                        for video_name in video_list)
    detection_dict = {}
    [detection_dict.update(d) for d in detection]
    output_dict = {"version": "THUMOS14", "results": detection_dict, "external_data": {}}

    with open(os.path.join(opt["output_path"], opt["detect_result_file"]), "w") as out:
        json.dump(output_dict, out)


