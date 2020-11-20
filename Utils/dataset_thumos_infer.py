# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pandas
import numpy
import json
import torch.utils.data as data
import os
import torch
import h5py
import pickle
import torch.nn.functional as F
from scipy.io import loadmat

def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data


class VideoDataSet(data.Dataset):
    def __init__(self, opt, subset="train", mode="train"):
        self.temporal_scale = opt["temporal_scale"]
        self.prop_temporal_scale = opt["prop_temporal_scale"]
        self.feat_dim = opt['feat_dim']
        self.temporal_gap = 1. / self.temporal_scale
        self.subset = 'val' if ('train' in subset) else 'test'
        self.mode = mode
        self.feature_path = opt["feature_path"]
        self.boundary_ratio = opt["boundary_ratio"]
        self.binary_actionness = opt['binary_actionness']
        self.gap = opt['stitch_gap']
        self.short_ratio = opt['short_ratio']
        self.video_anno = opt['video_anno']
        self.thumos_classes = opt["thumos_classes"]

        if self.mode == 'train':
            if 'val' in self.subset:
                self.video_windows = load_json('./Utils/video_win_val.json')
            elif 'test' in  self.subset:
                self.video_windows = load_json('./Utils/video_win_test.json')
        elif self.mode == 'inference':
            self.video_windows = load_json('./Utils/video_win_infer.json')

        self._getDatasetDict()

        self.anchor_xmin = [self.temporal_gap * i for i in range(self.temporal_scale)]
        self.anchor_xmax = [self.temporal_gap * i for i in range(1, self.temporal_scale + 1)]

    def _getDatasetDict(self):

        anno_database = load_json(self.video_anno)['database']


        self.video_dict = {}
        class_list = []

        for video_name, video_info in anno_database.items():

            video_subset = 'val' if video_info['subset'] == 'train' else 'test'

            for item in video_info['annotations']:
                class_list.append(item['label'])
                item['segment'][0] = float(item['segment'][0])
                item['segment'][1] = float(item['segment'][1])

            if self.subset in video_subset:

                self.video_dict[video_name] = video_info

        # self.video_list = self.video_dict.keys()

        self.video_list = [win['v_name'] for win in self.video_windows]

        if os.path.exists(self.thumos_classes):
            with open(self.thumos_classes, 'r') as f:
                self.classes = json.load(f)
        else:
            class_list = list(set(class_list))
            class_list = sorted(class_list)
            self.classes = {'Background': 0}
            for i,cls in enumerate(class_list):
                self.classes[cls] = i + 1
            with open(self.thumos_classes, 'w') as f:
                f.write(json.dumps(self.classes))

    def __getitem__(self, index):
        video_data, num_frms = self._get_train_data_label(index)

        return index, video_data, num_frms



    def _get_train_data_label(self, index):

        # General data
        video_name = self.video_windows[index]['v_name']
        w_start = int(self.video_windows[index]['w_start'])
        w_end = int(self.video_windows[index]['w_end'])
        fps_org = self.video_windows[index]['fps']
        video_second_org = self.video_windows[index]['v_duration']
        num_frms_win = w_end - w_start + 1

        # Get video feature
        rgb_features = h5py.File(os.path.join(self.feature_path, 'rgb_' + self.subset + '.h5'), 'r')
        rgb_data = rgb_features[video_name][:]
        rgb_data = torch.Tensor(rgb_data)
        rgb_data = torch.transpose(rgb_data, 0, 1)

        flow_features = h5py.File(os.path.join(self.feature_path, 'flow_' + self.subset + '.h5'), 'r')
        flow_data = flow_features[video_name][:]
        flow_data = torch.Tensor(flow_data)
        flow_data = torch.transpose(flow_data, 0, 1)


        if num_frms_win > self.temporal_scale * self.short_ratio:
            return self._get_train_data_label_org(rgb_data, flow_data, video_name,  w_start, w_end, fps_org, video_second_org)
        else:
            return self._get_train_data_label_stitch(rgb_data, flow_data, video_name, w_start, w_end, fps_org, video_second_org)



    def _get_train_data_label_stitch(self, rgb_data, flow_data, video_name, w_start, w_end, fps_org, video_second_org):

        num_frms1 = w_end - w_start + 1
        video_data = torch.zeros(self.feat_dim, self.temporal_scale)

        # Left part: original length
        rgb_data1 = rgb_data[:, w_start: w_end+1]
        flow_data1 = flow_data[:, w_start: w_end+1]
        video_data[:, :num_frms1] = torch.cat((rgb_data1, flow_data1), dim=0)

        # Right part: rescaled length
        num_frms2 = self.temporal_scale - num_frms1 - self.gap
        rgb_data2 = F.interpolate(rgb_data1[None,:,:], size=num_frms2, mode='linear', align_corners=True).squeeze(0)
        flow_data2 = F.interpolate(flow_data1[None,:,:], size=num_frms2, mode='linear', align_corners=True).squeeze(0)
        video_data[:, -num_frms2:] = torch.cat((rgb_data2, flow_data2), dim=0)

        return video_data, num_frms1



    def _get_train_data_label_org(self, rgb_data, flow_data, video_name,  w_start, w_end,  fps, video_second):


        video_data = torch.zeros(self.feat_dim, self.temporal_scale)


        rgb_data = rgb_data[:, w_start: w_end+1]
        flow_data = flow_data[:, w_start: w_end+1]

        num_frms = min(rgb_data.shape[-1], self.temporal_scale)
        video_data[:, :num_frms] = torch.cat((rgb_data[:, :num_frms], flow_data[:,:num_frms]), dim=0)

        return video_data, num_frms



    def _ioa_with_anchors(self, anchors_min, anchors_max, box_min, box_max):
        len_anchors = anchors_max - anchors_min
        int_xmin = np.maximum(anchors_min, box_min)
        int_xmax = np.minimum(anchors_max, box_max)
        inter_len = np.maximum(int_xmax - int_xmin, 0.)
        scores = np.divide(inter_len, len_anchors)
        return scores

    def __len__(self):
        return len(self.video_windows)


def iou_with_anchors(anchors_min, anchors_max, box_min, box_max):
    """Compute jaccard score between a box and the anchors.
    """
    len_anchors = anchors_max - anchors_min
    int_xmin = np.maximum(anchors_min, box_min)
    int_xmax = np.minimum(anchors_max, box_max)
    inter_len = np.maximum(int_xmax - int_xmin, 0.)
    union_len = len_anchors - inter_len + box_max - box_min
    # print inter_len,union_len
    jaccard = np.divide(inter_len, union_len)
    return jaccard

