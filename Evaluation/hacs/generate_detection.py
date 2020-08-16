import numpy as np
import pandas as pd
import json
import os
import multiprocessing as mp
from joblib import Parallel, delayed


def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data


def getDatasetDict(opt):
    # df = pd.read_csv(opt["video_info"])
    json_data = load_json(opt["video_anno"])
    # database = json_data
    train_dict = {}
    val_dict = {}
    test_dict = {}

    anno_database = json_data['database']
    for vid in anno_database.keys():
        video_name = vid  # no v_ here!!!!!
        if "QO3DMx5Omt8" in vid:
            continue
        video_info = anno_database[vid]
        video_subset = anno_database[vid]['subset']
        video_new_info = {}
        video_new_info['duration_second'] = float(video_info['duration'])
        video_new_info['annotations'] = video_info['annotations']
        if video_subset == "training":
            train_dict[video_name] = video_new_info
        elif video_subset == "validation":
            val_dict[video_name] = video_new_info
        elif video_subset == "testing":
            test_dict[video_name] = video_new_info


    return train_dict, val_dict, test_dict


def IOU(s1, e1, s2, e2):
    if (s2 > e1) or (s1 > e2):
        return 0
    Aor = max(e1, e2) - min(s1, s2)
    Aand = min(e1, e2) - max(s1, s2)
    return float(Aand) / Aor


def NMS(df, nms_threshold):
    df = df.sort(columns="score", ascending=False)

    tstart = list(df.xmin.values[:])
    tend = list(df.xmax.values[:])
    tscore = list(df.score.values[:])
    rstart = []
    rend = []
    rscore = []
    while len(tstart) > 1 and len(rscore) < 10:
        idx = 1
        while idx < len(tstart):
            if IOU(tstart[0], tend[0], tstart[idx], tend[idx]) > nms_threshold:
                tstart.pop(idx)
                tend.pop(idx)
                tscore.pop(idx)
            else:
                idx += 1
        rstart.append(tstart[0])
        rend.append(tend[0])
        rscore.append(tscore[0])
        tstart.pop(0)
        tend.pop(0)
        tscore.pop(0)
    newDf = pd.DataFrame()
    newDf['score'] = rscore
    newDf['xmin'] = rstart
    newDf['xmax'] = rend
    return newDf


def soft_nms(df, nms_alpha):
    df = df.sort_values(by="score", ascending=False)

    tstart = list(df.xmin.values[:])
    tend = list(df.xmax.values[:])
    tscore = list(df.score.values[:])

    rstart = []
    rend = []
    rscore = []

    while len(tscore) > 1 and len(rscore) < 101:
        max_index = tscore.index(max(tscore))
        for idx in range(0, len(tscore)):
            if idx != max_index:
                tmp_iou = IOU(tstart[max_index], tend[max_index], tstart[idx], tend[idx])
                if tmp_iou > 0.:
                    tscore[idx] = tscore[idx] * np.exp(-np.square(tmp_iou) / nms_alpha)

        rstart.append(tstart[max_index])
        rend.append(tend[max_index])
        rscore.append(tscore[max_index])
        tstart.pop(max_index)
        tend.pop(max_index)
        tscore.pop(max_index)

    newDf = pd.DataFrame()
    newDf['score'] = rscore
    newDf['xmin'] = rstart
    newDf['xmax'] = rend
    return newDf


def min_max(x):
    x = (x - min(x)) / (max(x) - min(x))
    return x


def gen_detections_video(opt, idx, video_name, video_info, score, class_1):
    exp_score = np.exp(score)
    score_1=np.max(exp_score/(1E-6+np.sum(exp_score)))

    df = pd.read_csv(os.path.join(opt["output_path"], opt["prop_path"]) + "/v_" + video_name + ".csv")
    df['score'] = df.score.values[:]
    if len(df) > 1:
        df = soft_nms(df, opt["nms_alpha_detect"])
    else:
        df = df
    df = df.sort_values(by="score", ascending=False)
    # video_duration=float(video_info["duration_frame"]/16*16)/video_info["duration_frame"]*video_info["duration_second"]
    video_duration = video_info["duration_second"]
    proposal_list = []

    for j in range(min(100, len(df))):
        tmp_proposal = {}
        tmp_proposal["label"] = str(class_1)
        tmp_proposal["score"] = float(df.score.values[j] * score_1)
        tmp_proposal["segment"] = [max(0, df.xmin.values[j]) * video_duration,
                                   min(1, df.xmax.values[j]) * video_duration]
        proposal_list.append(tmp_proposal)

    # print('The {}-th video {} is finished'.format(idx, video_name))

    return {video_name: proposal_list}


# =======================================================================================================
# ===============================Option 1 for multiproc==================================================
# =======================================================================================================
def gen_detections_multiproc(opt):
    # Load videl-level classification labels
    cls_data = np.load(opt["vlevel_cls_res"], allow_pickle=True)
    cls_data_score = {}
    cls_data_max = {}
    _, val_dict, test_dict = getDatasetDict(opt)
    if 'val' in opt['infer']:
        infer_dict = val_dict
    else:
        infer_dict = test_dict

    video_list = sorted(list(infer_dict.keys()))

    with open(opt['hacs_names'], 'r') as f:
        classes = f.read().splitlines()
    classes.pop(0) # remove title
    with open(opt['{}_vids'.format(opt['infer'])], 'r') as f:
        vids = f.read().splitlines()

    for idx, vid in enumerate(vids):
        vid = vid[2:]
        cls_data_score[vid], _ = cls_data[idx, :]
        cls_data_max[vid] = np.argmax(cls_data_score[vid]) # find the max class


    detection = Parallel(n_jobs=opt["post_process_thread"])(
        delayed(gen_detections_video)(
            opt,
            idx,
            video_name,
            infer_dict[video_name],
            cls_data_score[video_name],
            classes[cls_data_max[video_name]]
        ) for idx, video_name in enumerate(video_list))
    result_dict = {}
    [result_dict.update(d) for d in detection]

    output_dict = {"version": "HACS 1.1.1", "results": result_dict, "external_data": {}}
    outfile = open(os.path.join(opt["output_path"], opt["detect_result_file"]), "w")
    json.dump(output_dict, outfile)
    outfile.close()

