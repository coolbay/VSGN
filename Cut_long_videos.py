#!/usr/bin/env bash
import json
import Utils.opts as opts
import random
import h5py
import os
from scipy.io import loadmat

opt = opts.parse_opt()
opt = vars(opt)

############################################################
##############    Data and annotation files  ###############
############################################################
annot_file = "./Evaluation/thumos/annot/thumos_gt.json"
info_file_prefix = './Evaluation/thumos/annot/'
rgb_path = "/mnt/sdb1/Datasets/Thumos14/thumos_feature_exp/TSN_pretrain_avepool_allfrms_hdf5/"
flow_path = "/mnt/sdb1/Datasets/Thumos14/thumos_feature_exp/TSN_pretrain_avepool_allfrms_hdf5/"

############################################################
##############       Hyper-parameters        ###############
############################################################
step = int(opt['temporal_scale'] / 4)
thr_video = opt['temporal_scale'] * opt['short_ratio']
thr_action = round(opt['temporal_scale'] * opt['clip_win_size'])
clip_win_size = round(opt['clip_win_size'] * opt['temporal_scale'])

with open(annot_file) as json_file:
    anno_database = json.load(json_file)

############################################################
######## Deal with val and test sets  for training  ########
############################################################
for flag in ['val', 'test']:

    info_file = info_file_prefix + flag + '_info.mat'
    flag_db = flag if flag =='test' else 'train'         #train: 'val'; test: 'test'
    duration_idx = 5 if flag == 'test' else 7

    anno_df = loadmat(info_file)['videos'][0]
    rgb_file = h5py.File(os.path.join(rgb_path, 'rgb_' + flag + '.h5'), 'r')

    v_clip = []
    # For each video
    for v_name, v_info in anno_database['database'].items():
        if flag_db not in v_info['subset']:
            continue

        num_frms = rgb_file[v_name][:].shape[0]
        v_df = anno_df[int(v_name[-4:])-1]
        fps = num_frms / v_df[duration_idx][0][0]

        cnt = 0
        # We slide a window to generate multiple sub-sequences since most videos are longer than VSGN input length
        for w_start in range(0, num_frms, step):
            w_end = min(w_start + opt['temporal_scale'], num_frms) - 1

            # Use all windows without cutting that contain at least one action instance
            use_window = False
            for annot in v_info['annotations']:
                seg_start = round(float(annot['segment'][0])* fps)
                seg_end = round(float(annot['segment'][1])* fps)
                if  (seg_start >=w_start and seg_end<=w_end):
                    use_window = True
                    break
            if use_window:
                dict_w = {}
                dict_w['v_name'] = v_name
                dict_w['w_start'] = w_start
                dict_w['w_end'] = w_end
                dict_w['fps'] = fps
                dict_w['v_duration'] = v_df[duration_idx][0][0]
                v_clip.append(dict_w)

            # Cut long videos with short actions into short clips
            if w_end - w_start + 1 > thr_video:

                # For short actions in long videos
                cnt = 0
                dict_w = {}
                pre_end = 0
                for annot in v_info['annotations']:
                    seg_start = round(float(annot['segment'][0])* fps)
                    seg_end = round(float(annot['segment'][1])* fps)

                    if not (seg_start >=w_start  and seg_end<=w_end):
                        continue

                    if seg_start < pre_end:
                        continue

                    if bool(dict_w) and cnt>0 and seg_end <= dict_w['w_end']:
                        pre_end = seg_end
                        continue

                    elif bool(dict_w) and cnt>0 and seg_start < dict_w['w_end']:
                        dict_w['w_end'] = seg_start
                        v_clip.append(dict_w)
                        dict_w = {}

                    elif bool(dict_w) and cnt>0 and seg_start >= dict_w['w_end']:
                        v_clip.append(dict_w)
                        dict_w = {}

                    duration_action = seg_end - seg_start + 1

                    if duration_action <= thr_action:
                        flexible = thr_action - duration_action
                        dict_w['v_name'] = v_name
                        dict_w['w_start'] = round(max(w_start, random.uniform(seg_start - flexible, seg_start), pre_end))
                        dict_w['w_end'] = round(min(dict_w['w_start'] + clip_win_size, w_end))
                        dict_w['fps'] = fps
                        dict_w['v_duration'] = v_df[duration_idx][0][0]
                        cnt += 1
                    pre_end = seg_end

                if bool(dict_w):
                    v_clip.append(dict_w)
                    dict_w = {}

    with open('./Utils/video_win_'+ flag + '.json', 'w') as fout:
        json.dump(v_clip, fout)


############################################################
############# Deal with test set for inference #############
############################################################
flag = 'test'
info_file = info_file_prefix + flag + '_info.mat'
flag_db = flag if flag =='test' else 'train'
duration_idx = 5 if flag == 'test' else 7
rgb_file = h5py.File(os.path.join(rgb_path, 'rgb_' + flag + '.h5'), 'r')

v_clip = []
# For each video
for v_name, v_info in anno_database['database'].items():
    if flag_db not in v_info['subset']:
        continue

    num_frms = rgb_file[v_name][:].shape[0]
    v_df = anno_df[int(v_name[-4:])-1]
    fps = num_frms / v_df[duration_idx][0][0]

    cnt = 0
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