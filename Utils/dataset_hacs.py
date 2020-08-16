# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pandas
import numpy
import math
import json
import torch.utils.data as data
import os
import torch
import h5py
import scipy.interpolate



def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data


def poolData(data, videoAnno, num_prop=100, num_bin=1, num_sample_bin=3, feat_dim=1024, pool_type="mean"):
    feat_frame,feat=data.shape
    assert feat==feat_dim
    fps = 2 # for hacs
    corrected_second = feat_frame/fps

    st = 1/fps  # ONLY FOR HACS

    if len(data) == 1:
        video_feature = np.stack([data] * num_prop)
        video_feature = np.reshape(video_feature, [num_prop, feat_dim])
        return video_feature

    x = [st / 2 + ii * st for ii in range(len(data))]
    f = scipy.interpolate.interp1d(x, data, axis=0)

    video_feature = []
    zero_sample = np.zeros(num_bin * feat_dim)
    tmp_anchor_xmin = [1.0 / num_prop * i for i in range(num_prop)]
    tmp_anchor_xmax = [1.0 / num_prop * i for i in range(1, num_prop + 1)]

    num_sample = num_bin * num_sample_bin
    for idx in range(num_prop):
        xmin = max(x[0] + 0.0001, tmp_anchor_xmin[idx] * corrected_second)
        xmax = min(x[-1] - 0.0001, tmp_anchor_xmax[idx] * corrected_second)
        if xmax < x[0]:
            # print "fuck"
            video_feature.append(zero_sample)
            continue
        if xmin > x[-1]:
            video_feature.append(zero_sample)
            continue

        plen = (xmax - xmin) / (num_sample - 1)
        x_new = [xmin + plen * ii for ii in range(num_sample)]
        y_new = f(x_new)
        y_new_pool = []
        for b in range(num_bin):
            tmp_y_new = y_new[num_sample_bin * b:num_sample_bin * (b + 1)]
            if pool_type == "mean":
                tmp_y_new = np.mean(y_new, axis=0)
            elif pool_type == "max":
                tmp_y_new = np.max(y_new, axis=0)
            y_new_pool.append(tmp_y_new)
        y_new_pool = np.stack(y_new_pool)
        y_new_pool = np.reshape(y_new_pool, [-1])
        video_feature.append(y_new_pool)
    video_feature = np.stack(video_feature)
    return video_feature


class VideoDataSet(data.Dataset):
    def __init__(self, opt, subset="train", mode="train"):
        self.temporal_scale = opt["temporal_scale"]
        self.prop_temporal_scale = opt["prop_temporal_scale"]
        self.temporal_gap = 1. / self.temporal_scale
        self.subset = subset
        self.mode = mode
        self.feature_path = opt["feature_path"]
        self.boundary_ratio = opt["boundary_ratio"]
        self.video_anno_path = opt["video_anno"]
        self.hacs_classes = opt["hacs_classes"]
        self.binary_actionness = opt['binary_actionness']
        self._getDatasetDict()
        self._get_match_map()

    def _generate_classes(self, data):
        class_list = []
        for vid, vinfo in data['database'].items():
            for item in vinfo['annotations']:
                class_list.append(item['label'])

        class_list = list(set(class_list))
        class_list = sorted(class_list)
        classes = {'Background': 0}
        for i,cls in enumerate(class_list):
            classes[cls] = i + 1
        return classes

    def _getDatasetDict(self):
        anno_database = load_json(self.video_anno_path)
        self.video_dict = {}
        class_list = []

        anno_database = anno_database['database']
        for vid in anno_database.keys():
            video_name = 'v_'+vid
            video_info = anno_database[vid]
            video_subset = anno_database[vid]['subset']
            if self.subset == "full":
                self.video_dict[video_name] = video_info
            if self.subset in video_subset:
                self.video_dict[video_name] = video_info
            for item in video_info['annotations']:
                class_list.append(item['label'])

        self.video_list = self.video_dict.keys()

        if os.path.exists(self.hacs_classes):
            with open(self.hacs_classes, 'r') as f:
                self.classes = json.load(f)
        else:
            class_list = list(set(class_list))
            class_list = sorted(class_list)
            self.classes = {'Background': 0}
            for i,cls in enumerate(class_list):
                self.classes[cls] = i + 1
            with open(self.hacs_classes, 'w') as f:
                f.write(json.dumps(self.classes))

        "%s subset video numbers: %d" % (self.subset, len(self.video_list))

    def __getitem__(self, index):
        video_data, anchor_xmin, anchor_xmax = self._get_base_data(index)
        if self.binary_actionness == 'true':
            match_score_action, gt_iou_map, match_score_start, match_score_end = self._get_train_label_binary(index, anchor_xmin, anchor_xmax)
        else:
            match_score_action, gt_iou_map, match_score_start, match_score_end = self._get_train_label(index, anchor_xmin, anchor_xmax)
        if self.mode == "train":
            return video_data, match_score_action, match_score_start, match_score_end, gt_iou_map
        else:
            return index, video_data #, match_score_action, gt_iou_map

    def _get_match_map(self):
        match_map = []
        temporal_gap = 1. / self.prop_temporal_scale
        for idx in range(self.prop_temporal_scale):  # start
            tmp_match_window = []
            xmin = temporal_gap * idx
            for jdx in range(1, self.prop_temporal_scale + 1 ):  # duration
                xmax = xmin + temporal_gap * jdx
                tmp_match_window.append([xmin, xmax])
            match_map.append(tmp_match_window)
        match_map = np.array(match_map)  # 100x100x2  # start x duration
        match_map = np.transpose(match_map, [1, 0, 2])  #   # duration x start
        match_map = np.reshape(match_map, [-1, 2])  # [0,1] [1,2] [2,3].....[99,199]   # duration x start
        self.match_map = match_map  # duration is same in row, start is same in col


    def _get_base_data(self, index):
        t_scale=self.temporal_scale
        video_name = list(self.video_list)[index]
        anchor_xmin = [self.temporal_gap * i for i in range(self.temporal_scale)]
        anchor_xmax = [self.temporal_gap * i for i in range(1, self.temporal_scale + 1)]


        subset = self.video_dict[video_name]['subset']
        scaled_h5 = self.feature_path + "/" + subset + "/" + video_name + '_' +str(t_scale) + ".h5"
        if os.path.exists(scaled_h5): # read scaled file
            video_data = h5py.File(scaled_h5, 'r')[video_name + '_' +str(t_scale)][:]

        video_data = torch.Tensor(video_data)
        video_data = torch.transpose(video_data, 0, 1)
        video_data.float()

        return video_data, anchor_xmin, anchor_xmax

    def _get_train_label(self, index, anchor_xmin, anchor_xmax):
        video_name = list(self.video_list)[index]
        video_info = self.video_dict[video_name]
        corrected_second = float(video_info['duration'])
        video_labels = video_info['annotations']


        # Get gt_iou_map
        gt_bbox = []
        gt_iou_map = []
        for j in range(len(video_labels)):
            tmp_info = video_labels[j]
            tmp_start = max(min(1, tmp_info['segment'][0] / corrected_second), 0)
            tmp_end = max(min(1, tmp_info['segment'][1] / corrected_second), 0)
            tmp_class = self.classes[tmp_info['label']]
            gt_bbox.append([tmp_start, tmp_end, tmp_class])

            tmp_gt_iou_map = iou_with_anchors(
                self.match_map[:, 0], self.match_map[:, 1], tmp_start, tmp_end)
            tmp_gt_iou_map = np.reshape(tmp_gt_iou_map,
                                        [self.prop_temporal_scale, self.prop_temporal_scale])
            gt_iou_map.append(tmp_gt_iou_map)

        if len(gt_iou_map)==0: # test set
            tmp_info = 'test'
            tmp_start = 0
            tmp_end = 1
            tmp_class = 0
            gt_bbox.append([tmp_start, tmp_end, tmp_class]) 
            tmp_gt_iou_map = iou_with_anchors(self.match_map[:, 0], self.match_map[:, 1], tmp_start, tmp_end)
            tmp_gt_iou_map = np.reshape(tmp_gt_iou_map,[self.prop_temporal_scale, self.prop_temporal_scale])
            gt_iou_map.append(tmp_gt_iou_map)

        gt_iou_map = np.array(gt_iou_map)
        gt_iou_map = np.max(gt_iou_map, axis=0)
        gt_iou_map = torch.Tensor(gt_iou_map)

        # Get actionness scores
        match_score_action = [0] * self.temporal_scale
        for bbox in gt_bbox:
            left_frm = max(int(bbox[0] * self.temporal_scale), 0)
            right_frm = min(int(bbox[1] * self.temporal_scale), self.temporal_scale)
            match_score_action[left_frm:right_frm] = [bbox[2]] * (right_frm - left_frm)

        match_score_action = torch.Tensor(match_score_action)

        ####################################################################################################
        # generate R_s and R_e
        gt_bbox = np.array(gt_bbox)
        gt_xmins = gt_bbox[:, 0]
        gt_xmaxs = gt_bbox[:, 1]
        gt_len_small = 3 * self.temporal_gap  # np.maximum(self.temporal_gap, self.boundary_ratio * gt_lens)
        gt_start_bboxs = np.stack((gt_xmins - gt_len_small / 2, gt_xmins + gt_len_small / 2), axis=1)
        gt_end_bboxs = np.stack((gt_xmaxs - gt_len_small / 2, gt_xmaxs + gt_len_small / 2), axis=1)
        #####################################################################################################

        ##########################################################################################################
        # calculate the ioa for all timestamp
        match_score_start = []
        for jdx in range(len(anchor_xmin)):
            match_score_start.append(np.max(
                self._ioa_with_anchors(anchor_xmin[jdx], anchor_xmax[jdx], gt_start_bboxs[:, 0], gt_start_bboxs[:, 1])))
        match_score_end = []
        for jdx in range(len(anchor_xmin)):
            match_score_end.append(np.max(
                self._ioa_with_anchors(anchor_xmin[jdx], anchor_xmax[jdx], gt_end_bboxs[:, 0], gt_end_bboxs[:, 1])))
        match_score_start = torch.Tensor(match_score_start)
        match_score_end = torch.Tensor(match_score_end)
        ############################################################################################################

        return match_score_action, gt_iou_map, match_score_start, match_score_end


    def _get_train_label_binary(self, index, anchor_xmin, anchor_xmax):
        video_name = list(self.video_list)[index]
        video_info = self.video_dict[video_name]
        video_frame = video_info['duration_frame']
        video_second = video_info['duration_second']
        feature_frame = video_info['feature_frame']
        corrected_second = float(feature_frame) / video_frame * video_second
        video_labels = video_info['annotations']


        # Get gt_iou_map
        gt_bbox = []
        gt_iou_map = []
        for j in range(len(video_labels)):
            tmp_info = video_labels[j]
            tmp_start = max(min(1, tmp_info['segment'][0] / corrected_second), 0)
            tmp_end = max(min(1, tmp_info['segment'][1] / corrected_second), 0)
            tmp_class = self.classes[tmp_info['label']]
            gt_bbox.append([tmp_start, tmp_end, tmp_class])

            tmp_gt_iou_map = iou_with_anchors(
                self.match_map[:, 0], self.match_map[:, 1], tmp_start, tmp_end)
            tmp_gt_iou_map = np.reshape(tmp_gt_iou_map,
                                        [self.prop_temporal_scale, self.prop_temporal_scale])
            gt_iou_map.append(tmp_gt_iou_map)

        gt_iou_map = np.array(gt_iou_map)
        gt_iou_map = np.max(gt_iou_map, axis=0)
        gt_iou_map = torch.Tensor(gt_iou_map)


        # Get start/end/action
        gt_bbox = np.array(gt_bbox)
        gt_xmins = gt_bbox[:, 0]
        gt_xmaxs = gt_bbox[:, 1]
        # gt_lens = gt_xmaxs - gt_xmins
        gt_len_small = 3 * self.temporal_gap #np.maximum(self.temporal_gap, self.boundary_ratio * gt_lens)
        gt_start_bboxs = np.stack((gt_xmins - gt_len_small / 2, gt_xmins + gt_len_small / 2), axis=1)
        gt_end_bboxs = np.stack((gt_xmaxs - gt_len_small / 2, gt_xmaxs + gt_len_small / 2), axis=1)


        match_score_action = []
        for jdx in range(len(anchor_xmin)):
            match_score_action.append(
                np.max(self._ioa_with_anchors(anchor_xmin[jdx], anchor_xmax[jdx], gt_xmins, gt_xmaxs)))
        match_score_start = []
        for jdx in range(len(anchor_xmin)):
            match_score_start.append(np.max(
                self._ioa_with_anchors(anchor_xmin[jdx], anchor_xmax[jdx], gt_start_bboxs[:, 0], gt_start_bboxs[:, 1])))
        match_score_end = []
        for jdx in range(len(anchor_xmin)):
            match_score_end.append(np.max(
                self._ioa_with_anchors(anchor_xmin[jdx], anchor_xmax[jdx], gt_end_bboxs[:, 0], gt_end_bboxs[:, 1])))
        match_score_action = torch.Tensor(match_score_action)
        match_score_start = torch.Tensor(match_score_start)
        match_score_end = torch.Tensor(match_score_end)
        return match_score_action, gt_iou_map, match_score_start, match_score_end


    def _ioa_with_anchors(self, anchors_min, anchors_max, box_min, box_max):
        len_anchors = anchors_max - anchors_min
        int_xmin = np.maximum(anchors_min, box_min)
        int_xmax = np.minimum(anchors_max, box_max)
        inter_len = np.maximum(int_xmax - int_xmin, 0.)
        scores = np.divide(inter_len, len_anchors)
        return scores

    def __len__(self):
        return len(self.video_list)


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

