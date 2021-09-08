# -*- coding: utf-8 -*-
import numpy as np
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
        self.input_feat_dim = opt['input_feat_dim']
        self.temporal_gap = 1. / self.temporal_scale
        self.subset = 'val' if ('train' in subset) else 'test'
        self.mode = mode
        self.feature_path = opt["feature_path"]
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

        if self.mode == "train":
            video_data, match_score_action, match_score_start, match_score_end, gt_bbox, num_gt, num_frms = self._get_train_data_label(index)
            return video_data, match_score_action, match_score_start, match_score_end, gt_bbox, num_gt, num_frms
        else:
            video_data, num_frms = self._get_train_data_label(index)
            return index, video_data, num_frms #, match_score_action, gt_iou_map



    def _get_train_data_label(self, index):

        # General data
        video_name = self.video_windows[index]['v_name']
        w_start = int(self.video_windows[index]['w_start'])
        w_end = int(self.video_windows[index]['w_end'])
        fps_org = self.video_windows[index]['fps']
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
            return self._get_train_data_label_org(rgb_data, flow_data, video_name,  w_start, w_end, fps_org)
        else:
            return self._get_train_data_label_stitch(rgb_data, flow_data, video_name, w_start, w_end, fps_org)



    def _get_train_data_label_stitch(self, rgb_data, flow_data, video_name, w_start, w_end, fps_org):

        num_frms1 = w_end - w_start + 1
        video_data = torch.zeros(self.input_feat_dim, self.temporal_scale)

        # Left part: original length
        rgb_data1 = rgb_data[:, w_start: w_end+1]
        flow_data1 = flow_data[:, w_start: w_end+1]
        video_data[:, :num_frms1] = torch.cat((rgb_data1, flow_data1), dim=0)

        # Right part: rescaled length
        num_frms2 = self.temporal_scale - num_frms1 - self.gap
        rgb_data2 = F.interpolate(rgb_data1[None,:,:], size=num_frms2, mode='linear', align_corners=True).squeeze(0)
        flow_data2 = F.interpolate(flow_data1[None,:,:], size=num_frms2, mode='linear', align_corners=True).squeeze(0)
        video_data[:, -num_frms2:] = torch.cat((rgb_data2, flow_data2), dim=0)

        # Get annotations
        video_info = self.video_dict[video_name]
        video_labels_org = video_info['annotations']

        if self.mode == 'train':
            video_labels_frm = []
            for label in video_labels_org:
                label_start_frm =label['segment'][0] * fps_org
                label_end_frm = label['segment'][1] * fps_org
                if round(label_start_frm) >= w_start and round(label_end_frm) <= w_end:
                    label_frm = {}
                    label_frm['segment'] = []
                    label_frm['segment'].append(label_start_frm - w_start)
                    label_frm['segment'].append(label_end_frm - w_start)
                    label_frm['label'] = label['label']
                    video_labels_frm.append(label_frm)

            # Get gt_iou_map
            gt_bbox = []
            for j in range(len(video_labels_frm)):
                tmp_info = video_labels_frm[j]
                tmp_start_f = max(min(num_frms1-1, round(tmp_info['segment'][0] )), 0)
                tmp_end_f = max(min(num_frms1-1, round(tmp_info['segment'][1] )), 0)

                tmp_start = tmp_start_f / self.temporal_scale
                tmp_end = tmp_end_f / self.temporal_scale

                tmp_class = self.classes[tmp_info['label']]
                gt_bbox.append([tmp_start, tmp_end, tmp_class])

            for j in range(len(video_labels_frm)):
                tmp_info = video_labels_frm[j]
                tmp_start_f = max(min(num_frms2-1, round(tmp_info['segment'][0]  / num_frms1 * num_frms2)), 0) + num_frms1 + self.gap
                tmp_end_f = max(min(num_frms2-1, round(tmp_info['segment'][1]  / num_frms1 * num_frms2)), 0) + num_frms1 + self.gap

                tmp_start = tmp_start_f / self.temporal_scale
                tmp_end = tmp_end_f / self.temporal_scale

                tmp_class = self.classes[tmp_info['label']]
                gt_bbox.append([tmp_start, tmp_end, tmp_class])

            # Get actionness scores
            match_score_action = [0] * self.temporal_scale
            for bbox in gt_bbox:
                left_frm = max(round(bbox[0] * self.temporal_scale), 0)
                right_frm = min(round(bbox[1] * self.temporal_scale), self.temporal_scale-1)
                match_score_action[left_frm:right_frm+1] = [bbox[2]] * (right_frm + 1 - left_frm)

            match_score_action = torch.Tensor(match_score_action)

            ####################################################################################################
            # generate R_s and R_e
            gt_bbox = np.array(gt_bbox)
            if gt_bbox.shape[0] == 0:
                print(gt_bbox.shape)
            gt_xmins = gt_bbox[:, 0]
            gt_xmaxs = gt_bbox[:, 1]
            gt_len_small = 3 * self.temporal_gap
            gt_start_bboxs = np.stack((gt_xmins - gt_len_small / 2, gt_xmins + gt_len_small / 2), axis=1)
            gt_end_bboxs = np.stack((gt_xmaxs - gt_len_small / 2, gt_xmaxs + gt_len_small / 2), axis=1)
            #####################################################################################################

            ##########################################################################################################
            # calculate the ioa for all timestamp
            match_score_start = []
            for jdx in range(len(self.anchor_xmin)):
                match_score_start.append(np.max(
                    self._ioa_with_anchors(self.anchor_xmin[jdx], self.anchor_xmax[jdx], gt_start_bboxs[:, 0], gt_start_bboxs[:, 1])))
            match_score_end = []
            for jdx in range(len(self.anchor_xmin)):
                match_score_end.append(np.max(
                    self._ioa_with_anchors(self.anchor_xmin[jdx], self.anchor_xmax[jdx], gt_end_bboxs[:, 0], gt_end_bboxs[:, 1])))
            match_score_start = torch.Tensor(match_score_start)
            match_score_end = torch.Tensor(match_score_end)
            ############################################################################################################

            max_num_box = 50
            gt_bbox = torch.tensor(gt_bbox, dtype=torch.float32)
            gt_bbox_padding = gt_bbox.new(max_num_box, gt_bbox.size(1)).zero_()
            num_gt = min(gt_bbox.size(0), max_num_box)
            gt_bbox_padding[:num_gt, :] = gt_bbox[:num_gt]

            return video_data, match_score_action, match_score_start, match_score_end, gt_bbox_padding, num_gt, num_frms1

        else:
            return video_data, num_frms1



    def _get_train_data_label_org(self, rgb_data_org, flow_data_org, video_name,  w_start, w_end,  fps):

        video_data = torch.zeros(self.input_feat_dim, self.temporal_scale)

        rgb_data = rgb_data_org[:, w_start: w_end+1]
        flow_data = flow_data_org[:, w_start: w_end+1]

        num_frms = min(rgb_data.shape[-1], self.temporal_scale)
        video_data[:, :num_frms] = torch.cat((rgb_data[:, :num_frms], flow_data[:,:num_frms]), dim=0)

        if self.mode == 'train':
            # Get annotations
            video_info = self.video_dict[video_name]
            video_labels = video_info['annotations']
            num_frms_org = rgb_data_org.shape[-1]
            video_labels_frm = []
            for label in video_labels:
                label_start_frm = max(0, round(label['segment'][0] * fps))
                label_end_frm = min(round(label['segment'][1] * fps), num_frms_org - 1)
                if label_start_frm >= w_start and label_end_frm <= w_end:
                    label_frm = {}
                    label_frm['segment'] = []
                    label_frm['segment'].append(label_start_frm - w_start)
                    label_frm['segment'].append(label_end_frm - w_start)
                    label_frm['label'] = label['label']
                    video_labels_frm.append(label_frm)

            # Get gt_iou_map
            gt_bbox = []
            for j in range(len(video_labels_frm)):
                tmp_info = video_labels_frm[j]

                tmp_start_f = max(min(num_frms-1, tmp_info['segment'][0]), 0)
                tmp_end_f = max(min(num_frms-1, tmp_info['segment'][1]), 0)

                tmp_start = tmp_start_f / self.temporal_scale
                tmp_end = tmp_end_f / self.temporal_scale

                tmp_class = self.classes[tmp_info['label']]
                gt_bbox.append([tmp_start, tmp_end, tmp_class])


            # Get actionness scores
            match_score_action = [0] * self.temporal_scale
            for bbox in gt_bbox:
                left_frm = max(round(bbox[0] * self.temporal_scale), 0)
                right_frm = min(round(bbox[1] * self.temporal_scale), self.temporal_scale-1)
                match_score_action[left_frm:right_frm+1] = [bbox[2]] * (right_frm + 1 - left_frm)

            match_score_action = torch.Tensor(match_score_action)

            ####################################################################################################
            # generate R_s and R_e
            gt_bbox = np.array(gt_bbox)
            if gt_bbox.shape[0] == 0:
                print(gt_bbox.shape)

            gt_xmins = gt_bbox[:, 0]
            gt_xmaxs = gt_bbox[:, 1]
            gt_len_small = 3 * self.temporal_gap
            gt_start_bboxs = np.stack((gt_xmins - gt_len_small / 2, gt_xmins + gt_len_small / 2), axis=1)
            gt_end_bboxs = np.stack((gt_xmaxs - gt_len_small / 2, gt_xmaxs + gt_len_small / 2), axis=1)
            #####################################################################################################

            ##########################################################################################################
            # calculate the ioa for all timestamp
            match_score_start = []
            for jdx in range(len(self.anchor_xmin)):
                match_score_start.append(np.max(
                    self._ioa_with_anchors(self.anchor_xmin[jdx], self.anchor_xmax[jdx], gt_start_bboxs[:, 0], gt_start_bboxs[:, 1])))
            match_score_end = []
            for jdx in range(len(self.anchor_xmin)):
                match_score_end.append(np.max(
                    self._ioa_with_anchors(self.anchor_xmin[jdx], self.anchor_xmax[jdx], gt_end_bboxs[:, 0], gt_end_bboxs[:, 1])))
            match_score_start = torch.Tensor(match_score_start)
            match_score_end = torch.Tensor(match_score_end)
            ############################################################################################################

            max_num_box = 50
            gt_bbox = torch.tensor(gt_bbox, dtype=torch.float32)
            gt_bbox_padding = gt_bbox.new(max_num_box, gt_bbox.size(1)).zero_()
            num_gt = min(gt_bbox.size(0), max_num_box)
            gt_bbox_padding[:num_gt, :] = gt_bbox[:num_gt]

            return video_data, match_score_action, match_score_start, match_score_end, gt_bbox_padding, num_gt, num_frms
        else:
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

