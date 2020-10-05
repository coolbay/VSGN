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
import torch.nn.functional as F



def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data


class VideoDataSet(data.Dataset):
    def __init__(self, opt, subset="train", mode="train"):
        self.temporal_scale = opt["temporal_scale"]
        self.prop_temporal_scale = opt["prop_temporal_scale"]
        self.temporal_gap = 1. / self.temporal_scale
        self.subset = subset
        self.mode = mode
        self.feature_path = opt["feature_path"]
        self.boundary_ratio = opt["boundary_ratio"]
        self.video_info_path = opt["video_info"]
        self.video_anno_path = opt["video_anno"]
        self.anet_classes = opt["anet_classes"]
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
        anno_df = pd.read_csv(self.video_info_path)
        anno_database = load_json(self.video_anno_path)
        self.video_dict = {}
        class_list = []

        # for key, value in anno_database.items():
        #     video_name = key
        #     video_info = value
        #     video_subset = value['subset']
        #     if self.subset == "full" or self.subset in video_subset:
        #         self.video_dict[video_name] = video_info
        #
        #     for item in video_info['annotations']:
        #         class_list.append(item['label'])

        for i in range(len(anno_df)):
            video_name = anno_df.video.values[i]
            if video_name == 'v_Bg-0ibLZrgg' or video_name == 'v_0dkIbKXXFzI' or video_name == 'v_fmtW5lcdT_0' or video_name == 'v_x0PE_98UO3s':
                continue

            # if video_name == 'v_6kgJx6ahgq0' or video_name == 'v_RuBLAghI1vE':  # the annotation is not correct
            #     continue
            video_info = anno_database[video_name]
            video_subset = anno_df.subset.values[i]
            if self.subset == "full":
                self.video_dict[video_name] = video_info
            if self.subset in video_subset:
                self.video_dict[video_name] = video_info
            for item in video_info['annotations']:
                class_list.append(item['label'])

        self.video_list = self.video_dict.keys()

        if os.path.exists(self.anet_classes):
            with open(self.anet_classes, 'r') as f:
                self.classes = json.load(f)
        else:
            class_list = list(set(class_list))
            class_list = sorted(class_list)
            self.classes = {'Background': 0}
            for i,cls in enumerate(class_list):
                self.classes[cls] = i + 1
            with open(self.anet_classes, 'w') as f:
                f.write(json.dumps(self.classes))

        "%s subset video numbers: %d" % (self.subset, len(self.video_list))

    def __getitem__(self, index):
        video_data, anchor_xmin, anchor_xmax = self._get_base_data(index)

        match_score_action, gt_iou_map, match_score_start, match_score_end, gt_bbox, num_gt  = self._get_train_label(index, anchor_xmin, anchor_xmax)

        if self.mode == "train":
            return video_data, match_score_action, match_score_start, match_score_end, gt_iou_map, gt_bbox, num_gt
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

        video_name = list(self.video_list)[index]
        anchor_xmin = [self.temporal_gap * i for i in range(self.temporal_scale)]
        anchor_xmax = [self.temporal_gap * i for i in range(1, self.temporal_scale + 1)]

        rgb_features = h5py.File(os.path.join(self.feature_path, 'rgb.h5'), 'r')
        rgb_data = rgb_features[video_name][:]
        rgb_data = torch.Tensor(rgb_data)
        rgb_data = torch.transpose(rgb_data, 0, 1)
        rgb_data = F.interpolate(rgb_data[None,:,:], size=self.temporal_scale, mode='linear', align_corners=True).squeeze(0)

        # flow_features = h5py.File(os.path.join(self.feature_path, 'flow.h5'), 'r')
        # flow_data = flow_features[video_name][:]
        # flow_data = torch.Tensor(flow_data)
        # flow_data = torch.transpose(flow_data, 0, 1)
        # flow_data = F.interpolate(flow_data[None,:,:], size=self.temporal_scale, mode='linear', align_corners=True).squeeze(0)
        #
        # video_data = torch.cat((rgb_data, flow_data), dim=0)
        video_data = rgb_data

        return video_data, anchor_xmin, anchor_xmax

    def _get_train_label(self, index, anchor_xmin, anchor_xmax):
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

            # tmp_start = max(min(1, tmp_info['segment'][0] / corrected_second), 0)  / 4. # by Catherine
            # tmp_end = max(min(1, tmp_info['segment'][1] / corrected_second), 0) / 4. # by Catherine
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

        if torch.sum(gt_iou_map) == 0:
            a = 1

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

        max_num_box = 30
        gt_bbox = torch.tensor(gt_bbox, dtype=torch.float32)
        gt_bbox_padding = gt_bbox.new(max_num_box, gt_bbox.size(1)).zero_()
        num_gt = min(gt_bbox.size(0), max_num_box)
        gt_bbox_padding[:num_gt, :] = gt_bbox[:num_gt]
        # labels = BoxList(torch.Tensor(gt_bbox))

        return match_score_action, gt_iou_map, match_score_start, match_score_end, gt_bbox_padding, num_gt



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



class BoxList(object):
    def __init__(self, bbox):
        self.bbox = bbox
