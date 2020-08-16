import matplotlib.pyplot as plt
import numpy as np
import json
import pandas as pd
import os
from scipy.special import softmax
import math
import Utils.opts as opts
from scipy.interpolate import interp1d

v_length = 1000
# video_name = 'v_lZ6zN5Q447M'


opt = opts.parse_opt()
opt = vars(opt)

action_path = "output_2020-05-08_15-28_org_selAct/actionness/"
start_end_path = "output_2020-05-08_15-28_org_selAct/start_end/"
save_path = "output_2020-05-08_15-28_org_selAct/class_spec_action_curves1/"
annot_path = "Data/activitynet_annotations/anet_anno_action.json"
v_info_path = "Data/activitynet_annotations/video_info_new.csv"
anet_classes = "Data/activitynet_annotations/anet_classes_idx.json"
per_video_map_fpath = 'output_2020-05-08_15-28_org_selAct/detect_mAP_v.json'
prediction_filename = 'output_2020-05-08_15-28_org_selAct/detections_postNMS.json'


# Per-video mAP
with open(per_video_map_fpath, 'r') as fp:
    mAPs = json.load(fp)

# Prediction
with open(prediction_filename, 'r') as fobj:
    predictions = json.load(fobj)

if not os.path.exists(save_path):
    os.makedirs(save_path)

anno_df = pd.read_csv(v_info_path)
subset = 'val'

with open(annot_path) as json_file:
    data_base = json.load(json_file)

with open(anet_classes, 'r') as f:
    classes = json.load(f)

for i in range(len(anno_df)):
    video_name = anno_df.video.values[i]
    video_subset = anno_df.subset.values[i]
    if subset not in video_subset:
        continue
    video_info = data_base[video_name]
    video_labels = video_info['annotations']

    if len(video_labels) == 0:
        continue

    video_second = video_info['duration_second']

    # Plot actionness curves
    plt.figure(figsize=(30,10))

    # Actionness score: red
    data_path = action_path + video_name + ".csv"
    if not os.path.exists(data_path):
        continue
    data = np.genfromtxt(data_path, delimiter=',')
    # data_norm = softmax(data, axis=0)
    # gt_label = classes[video_labels[0]['label']]
    # plot_data = plt.plot(range(v_length), data_norm[gt_label],  'ro-', markersize = 3, markevery=1)
    plot_data = plt.plot(range(v_length), data,  'ro-', markersize = 3, markevery=1)


    # Groundtruth: blue
    gt_bbox = []
    gt_curve = np.zeros((v_length,), dtype=int)
    for j in range(len(video_labels)):
        tmp_info = video_labels[j]
        tmp_start = int(max(min(1, tmp_info['segment'][0] / video_second), 0) * v_length)
        tmp_end = int(max(min(1, tmp_info['segment'][1] / video_second), 0) * v_length)
        gt_curve[tmp_start:tmp_end] = 1
    plt.plot(range(v_length), gt_curve)

    # Predicted actions: green
    # pred_actions = []
    # pred_actions_score = []
    pred = predictions['results'][video_name[2:]]
    level = [.9,.8,.7,.6,.5,.4,.3,.2,.1]
    cnt = 0
    dur = video_info['duration_second']
    for p in pred:
        if cnt < 9:
            start = max(0, int(p['segment'][0] / dur * v_length))
            end = min(v_length-1, int(p['segment'][1]/ dur * v_length))
            x = range(start, end+1)
            y = [p['score']] * len(x)
            # pred_actions.append((x, y))
            # pred_actions_score.append(p['score'])
            cnt += 1
            plt.plot(x, y, c='g', markersize = 3)




    if video_name[2:] not in mAPs.keys():
        continue
    mAP = mAPs[video_name[2:]]

    plt.xlabel('Frame index')
    plt.ylabel('Actionness probability')
    plt.title('Actionness | mAP is {}'.format(mAP))
    # plt.show()
    plt.savefig(save_path + video_name + '_Action.png')
    plt.close()

    # Starting / ending
    data_path = start_end_path + video_name + ".csv"
    if not os.path.exists(data_path):
        continue

    data = pd.read_csv(data_path)

    # start
    plt.figure(figsize=(30,10))
    x = range(len(data['start']))

    plot_data = plt.plot(x, data['start'],  'ro-', markersize = 3, markevery=1)
    plt.plot(x, gt_curve)

    plt.xlabel('Time')
    plt.ylabel('Starting probability')
    plt.title('Starting')
    plt.savefig(save_path + video_name + '_Start.png')
    plt.close()

    # end
    plt.figure(figsize=(30,10))
    x = range(len(data['end']))

    plot_data = plt.plot(x, data['end'],  'ro-', markersize = 3, markevery=1)
    plt.plot(x, gt_curve)

    plt.xlabel('Time')
    plt.ylabel('Ending probability')
    plt.title('Ending')
    plt.savefig(save_path + video_name + '_End.png')
    plt.close()
