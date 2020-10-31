#!/usr/bin/env bash
import json
import pandas as pd
import Utils.opts as opts
import random

def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data


opt = opts.parse_opt()
opt = vars(opt)

annot_file = '../D.Start_end_input1280_pad0_rf/Evaluation/activitynet/annot/anet_anno_action.json'
info_file = '../D.Start_end_input1280_pad0_rf/Evaluation/activitynet/annot/video_info_new.csv'

anno_database = load_json(annot_file)
anno_df = pd.read_csv(info_file)

num_long_video = 0
num_longv_w_short_action = 0

fps = 5
thr_video_sec = opt['temporal_scale'] * opt['short_ratio'] / fps
thr_action_sec = opt['temporal_scale'] * opt['clip_win_size']  / fps
clip_win_size = opt['clip_win_size'] * opt['temporal_scale'] / fps


v_clip = []

for v_name, v_info in anno_database.items():
    if 'v_V90aT-d_FKo' in v_name:
        a = 1

    if 'train' not in anno_df[anno_df.video.values == v_name].subset.values[0]:
        continue

    # For long videos
    if v_info['duration_second'] > thr_video_sec:

        # For short actions in long videos
        cnt = 0
        dict_w = {}
        pre_end = 0
        for annot in v_info['annotations']:
            if annot['segment'][0] < pre_end:
                continue

            if bool(dict_w) and cnt>0 and annot['segment'][1] <= dict_w['w_end']:
                pre_end = annot['segment'][1]
                continue
            elif bool(dict_w) and cnt>0 and annot['segment'][0] < dict_w['w_end']:
                dict_w['w_end'] = annot['segment'][0]
                v_clip.append(dict_w)
                dict_w = {}
            elif bool(dict_w) and cnt>0 and annot['segment'][0] >= dict_w['w_end']:
                v_clip.append(dict_w)
                dict_w = {}

            duration_action = annot['segment'][1] -annot['segment'][0]
            if duration_action <= thr_action_sec:

                flexible = thr_action_sec - duration_action
                dict_w['v_name'] = v_name
                dict_w['w_start'] = max(0, random.uniform(annot['segment'][0] - flexible, annot['segment'][0]), pre_end)
                dict_w['w_end'] = min(dict_w['w_start'] + clip_win_size, v_info['duration_second'])
                cnt += 1
            pre_end = annot['segment'][1]

        if bool(dict_w):
            v_clip.append(dict_w)
            dict_w = {}

    # For all videos
    dict_v = {}
    dict_v['v_name'] = v_name
    dict_v['w_start'] = 0
    dict_v['w_end'] = v_info['duration_second']
    v_clip.append(dict_v)

with open('video_win_train.json', 'w') as fout:
    json.dump(v_clip, fout)
