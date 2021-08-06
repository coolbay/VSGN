
import torch
import torch.nn.functional as F
from .matcher import Matcher
import math
import sys
sys.path.append("..")

from Utils.FocalLoss.sigmoid_focal_loss import SigmoidFocalLoss
from .BoxCoder import  BoxCoder


class LossComputation(object):

    def __init__(self, opt):

        self.tscale = opt["temporal_scale"]
        self.gamma = 2.0
        self.alpha = opt['focal_alpha']
        self.num_classes = 1 if opt['dataset'] == 'activitynet' else opt['decoder_num_classes']
        self.iou_thresholds = opt['iou_thr']
        self.matcher = Matcher(True)

        self.box_coder = BoxCoder(opt)


    def _iou_anchors_gts(self, anchor, gt):

        anchors_min = anchor[:, 0]
        anchors_max = anchor[:, 1]
        box_min = gt[:, 0]
        box_max = gt[:, 1]
        len_anchors = anchors_max - anchors_min + 1
        int_xmin = torch.max(anchors_min[:, None], box_min)
        int_xmax = torch.min(anchors_max[:, None], box_max)
        inter_len = torch.clamp(int_xmax - int_xmin, min=0)
        union_len = torch.clamp(len_anchors[:, None] + box_max - box_min - inter_len, min=0)
        # print inter_len,union_len
        jaccard = inter_len / union_len
        return jaccard

    def __call__(self, cls_pred_enc, reg_pred_enc, cls_pred_dec, reg_pred_dec, anchors, gt_bbox, num_gt):
        bs = cls_pred_enc[0].shape[0]

        # Loss for Stage 0 --- encoder
        anchors = [anchor.unsqueeze(0).repeat(bs, 1, 1).to(device=gt_bbox.device) for anchor in anchors]
        cls_loss0, reg_loss0 = self._loss_one_stage(cls_pred_enc, reg_pred_enc, gt_bbox, num_gt, anchors, stage=0)

        # Loss for Stage 1 --- decoder
        anchors_update = []
        for pred, anchor in zip(reg_pred_enc, anchors):
            pred = pred.permute(0, 2, 1).reshape(-1, 2)
            anchor = anchor.view(-1, 2)
            anchors_update.append(self.box_coder.decode(pred, anchor).view(bs, -1,2))
        cls_pred_dec = [cls_pred_dec[i] for i in range(len(cls_pred_dec)-1, -1, -1)]
        reg_pred_dec = [reg_pred_dec[i] for i in range(len(reg_pred_dec)-1, -1, -1)]

        loc_dec = []
        for pred, anchor in zip(reg_pred_dec, anchors):
            pred = pred.permute(0, 2, 1).reshape(-1, 2)
            anchor = anchor.view(-1, 2)
            loc_dec.append(self.box_coder.decode(pred, anchor).view(bs, -1,2))

        _, reg_loss1 = self._loss_one_stage(cls_pred_dec, reg_pred_dec, gt_bbox, num_gt, anchors, stage=1)
        cls_loss1, _ = self._loss_one_stage(cls_pred_dec, reg_pred_dec, gt_bbox, num_gt, loc_dec, stage=1)


        losses = {
            "loss_cls_enc": cls_loss0,
            "loss_reg_enc": reg_loss0,
            "loss_cls_dec": cls_loss1,
            "loss_reg_dec": reg_loss1,
        }
        return losses, loc_dec


    def _loss_one_stage(self, cls_pred, reg_pred, gt_bbox, num_gt, anchors, stage=0):

        num_cls = self.num_classes

        cls_labels, reg_targets = self.prepare_targets(gt_bbox, num_gt, anchors, stage=stage) # bs, levels*positions*scales, num_cls/left-right

        cls_pred = torch.cat(cls_pred, dim=2).permute(0, 2, 1).reshape(-1, num_cls) # bs, levels*positions, scales*cls --> bs*levels*positions*scales, num_cls
        reg_pred = torch.cat(reg_pred, dim=2).permute(0, 2, 1).reshape(-1, 2) # bs, levels*positions, scales*left-right --> bs*levels*positions*scales, left-right

        cls_labels = torch.cat(cls_labels, dim=0)  # bs*levels*positions*scales, num_cls
        reg_targets = torch.cat(reg_targets, dim=0)  # bs*levels*positions*scales, left-right

        all_anchors = torch.cat(anchors, dim=1).view(-1, 2) # bs*levels*positions*scales, 2

        pos_inds = torch.nonzero(cls_labels > 0).squeeze(1)
        cls_loss = self.cls_loss_func(cls_pred, cls_labels)
        reg_loss = self.reg_loss_func(reg_pred[pos_inds], reg_targets[pos_inds], all_anchors[pos_inds]) / pos_inds.numel()

        return  cls_loss, reg_loss

    def cls_loss_func(self, cls_pred, cls_labels):

        pmask = (cls_labels>0).float()
        nmask = (cls_labels==0).float()
        num_pos = torch.sum(pmask)
        num_neg = torch.sum(nmask)

        CE_loss = torch.nn.CrossEntropyLoss(reduction='none')

        loss = CE_loss(cls_pred, cls_labels.to(torch.long))

        pos_loss = torch.sum(loss * pmask) / num_pos
        neg_loss = torch.sum(loss * nmask) / num_neg

        total_loss = pos_loss + neg_loss

        return total_loss

    def reg_loss_func(self, pred, target, anchor, pred_boxes=None, weight=None):
        if type(pred_boxes) == type(None):
            pred_boxes = self.box_coder.decode(pred, anchor)
        pred_x1 = torch.min(pred_boxes[:, 0], pred_boxes[:, 1])
        pred_x2 = torch.max(pred_boxes[:, 0], pred_boxes[:, 1])
        pred_area = (pred_x2 - pred_x1)

        gt_boxes = self.box_coder.decode(target, anchor)
        target_x1 = gt_boxes[:, 0]
        target_x2 = gt_boxes[:, 1]
        target_area = (target_x2 - target_x1)

        x1_intersect = torch.max(pred_x1, target_x1)
        x2_intersect = torch.min(pred_x2, target_x2)
        area_intersect = torch.zeros(pred_x1.size()).to(target)
        mask = (x2_intersect > x1_intersect)
        area_intersect[mask] = (x2_intersect[mask] - x1_intersect[mask])

        x1_enclosing = torch.min(pred_x1, target_x1)
        x2_enclosing = torch.max(pred_x2, target_x2)
        area_enclosing = (x2_enclosing - x1_enclosing)  + 1e-7

        area_union = pred_area + target_area - area_intersect + 1e-7
        ious = area_intersect / area_union
        gious = ious - (area_enclosing - area_union) / area_enclosing

        losses = 1 - gious

        if weight is not None and weight.sum() > 0:
            return (losses * weight).sum()
        else:
            assert losses.numel() != 0
            return losses.sum()

    # TODO: 1. low threshold
    # TODO: 2. Centerness: positive center should be inside the gt action
    def prepare_targets(self, gt_bbox, num_gt, anchors, stage=0):

        cls_targets = []
        reg_targets = []
        all_anchors = torch.cat(anchors, dim=1) # bs, levels, positions, scales, left-right

        for i in range(len(gt_bbox)):

            gt_cur_im = gt_bbox[i, :num_gt[i], :-1] * self.tscale
            gt_label = gt_bbox[i, :num_gt[i], -1]
            anchor_cur_im = all_anchors[i]
            iou_matrix = self._iou_anchors_gts(anchor_cur_im, gt_cur_im)

            # Find the corresponding gt for each pred
            matched_idxs = self.matcher(iou_matrix.transpose(0, 1), self.iou_thresholds[stage])

            # Use the label of the corresponding gt as the classification target for the pred
            cls_labels_cur_im = torch.zeros_like(matched_idxs)

            cls_labels_cur_im[:] = gt_label[matched_idxs]

            cls_labels_cur_im[matched_idxs <0] = 0

            # Record the boundary offset as the regression target
            matched_gts = gt_cur_im[matched_idxs.clamp(min=0)]
            reg_targets_cur_im = self.box_coder.encode(matched_gts, anchor_cur_im)

            cls_targets.append(cls_labels_cur_im.to(dtype=torch.int32))
            reg_targets.append(reg_targets_cur_im)

        return cls_targets, reg_targets