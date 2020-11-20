#!/usr/bin/env bash
import json
import pandas as pd
import Utils.opts as opts
import random
import h5py
import os
from scipy.io import loadmat

def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data

flag = 'test'

opt = opts.parse_opt()
opt = vars(opt)

annot_file = "./Evaluation/thumos/annot/thumos_gt.json"
info_file = '/media/zhaoc/Data/Datasets/Thumos14/annotation/' + flag + '_info.mat'

# info_file = "Evaluation/thumos/annot/test_Annotation.csv"
rgb_path = "/media/zhaoc/Data/Datasets/Thumos14/thumos_feature_exp/TSN_pretrain_avepool_allfrms_hdf5/"
flow_path = "/media/zhaoc/Data/Datasets/Thumos14/thumos_feature_exp/TSN_pretrain_avepool_allfrms_hdf5/"



flag_db = flag if flag =='test' else 'train'

duration_idx = 5 if flag == 'test' else 7

step =  int(opt['temporal_scale'] /  4)
num_long_video = 0
num_longv_w_short_action = 0

anno_database = load_json(annot_file)
anno_df = loadmat(info_file)['videos'][0]
rgb_file = h5py.File(os.path.join(rgb_path, 'rgb_' + flag + '.h5'), 'r')
flow_file = h5py.File(os.path.join(flow_path, 'rgb_' + flag + '.h5'), 'r')

thr_video = opt['temporal_scale'] * opt['short_ratio']
thr_action = round(opt['temporal_scale'] * opt['clip_win_size'])
clip_win_size = round(opt['clip_win_size'] * opt['temporal_scale'])


v_clip = []

# For each video
for v_name, v_info in anno_database['database'].items():
    if flag_db not in v_info['subset']:
        continue

    num_frms = rgb_file[v_name][:].shape[0]
    v_df = anno_df[int(v_name[-4:])-1]
    fps = num_frms / v_df[duration_idx][0][0]

    cnt = 0
    # For each size=1280 window
    for w_start in range(0, num_frms, step):
        w_end = min(w_start + opt['temporal_scale'], num_frms) - 1

        dict_w = {}
        dict_w['v_name'] = v_name
        dict_w['w_start'] = w_start
        dict_w['w_end'] = w_end
        dict_w['fps'] = fps
        dict_w['v_duration'] = v_df[duration_idx][0][0]
        dict_w['w_index'] = cnt
        v_clip.append(dict_w)
        cnt += 1


with open('./Utils/video_win_infer.json', 'w') as fout:
    json.dump(v_clip, fout)
