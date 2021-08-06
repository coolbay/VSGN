# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
from .FPN import FPN
# from .GCNs import GraphNet_prop
import torch.nn.functional as F
# from .crop_props import BoundaryMatch, GraphAlign
from .Head import Head
from .AnchorGenerator import AnchorGenerator
from .loss import LossComputation
from .ActionGenerator import ActionGenerator
from .BoundaryRefine import BoundaryRefine

class SegTAD(nn.Module):
    def __init__(self, opt):
        super(SegTAD, self).__init__()

        self.bb_hidden_dim = opt['bb_hidden_dim']
        self.bs = opt["batch_size"]
        self.is_train = opt['is_train']
        self.tem_best_loss = 10000000
        self.hidden_dim_2d = 128
        self.hidden_dim_3d = 512
        self.input_feat_dim = opt['input_feat_dim']

        self.fpn = FPN(opt)

        self.head_enc = Head(opt)
        self.head_dec = Head(opt)

        self.anchors = AnchorGenerator(opt).anchors
        self.rpn_loss_compute = LossComputation(opt)
        self.bd_refine = BoundaryRefine(opt)

        self.gen_predictions = ActionGenerator(opt)


        # Generate action/start/end scores
        self.head_actionness = nn.Sequential(
            nn.Conv1d(self.bb_hidden_dim, self.bb_hidden_dim, kernel_size=3, padding=1, groups=1),
            # nn.GroupNorm(32, self.bb_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.bb_hidden_dim, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.head_startness = nn.Sequential(
            nn.Conv1d(self.bb_hidden_dim, self.bb_hidden_dim, kernel_size=3, padding=1, groups=1),
            # nn.GroupNorm(32, self.bb_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.bb_hidden_dim, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.head_endness = nn.Sequential(
            nn.Conv1d(self.bb_hidden_dim, self.bb_hidden_dim, kernel_size=3, padding=1, groups=1),
            # nn.GroupNorm(32, self.bb_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.bb_hidden_dim, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def _forward_test(self, cls_pred_enc, reg_pred_enc, cls_pred_dec, reg_pred_dec):

        loc_enc, score_enc, loc_dec, score_dec = self.gen_predictions(cls_pred_enc, reg_pred_enc, cls_pred_dec, reg_pred_dec, self.anchors)

        return loc_enc, score_enc, loc_dec, score_dec



    def forward(self, input, num_frms, gt_bbox = None, num_gt = None):

        feats_enc, feats_dec = self.fpn(input, num_frms)

        # Stage 0, Stage 1
        cls_pred_enc, reg_pred_enc = self.head_enc(feats_enc)
        cls_pred_dec, reg_pred_dec = self.head_dec(feats_dec)

        if self.is_train == 'true':
            losses_rpn, loc_dec = self.rpn_loss_compute(
                cls_pred_enc,
                reg_pred_enc,
                cls_pred_dec,
                reg_pred_dec,
                self.anchors,
                gt_bbox,
                num_gt)
        else:
            score_enc, loc_enc, score_dec, loc_dec = self.gen_predictions(
                cls_pred_enc,
                reg_pred_enc,
                cls_pred_dec,
                reg_pred_dec,
                self.anchors)

        # Stage 2
        feat_frmlvl = feats_dec[-1]

        # Stage 2: Action/start/end scores
        actionness = self.head_actionness(feat_frmlvl)
        start = self.head_startness(feat_frmlvl).squeeze(1)
        end = self.head_endness(feat_frmlvl).squeeze(1)

        actionness = F.interpolate(actionness, size=input.size()[2:], mode='linear', align_corners=True)
        start = F.interpolate(start[:, None, :], size=input.size()[2:], mode='linear', align_corners=True).squeeze(1)
        end = F.interpolate(end[:, None, :], size=input.size()[2:], mode='linear', align_corners=True).squeeze(1)

        # Stage 2: Boundary refinement
        start_offsets, end_offsets  = self.bd_refine(loc_dec, feat_frmlvl)

        if self.is_train == 'true':
            loss_reg_st2 = self.bd_refine.cal_loss(start_offsets,
                                                  end_offsets,
                                                  loc_dec,
                                                  gt_bbox,
                                                  num_gt)
            losses_rpn['loss_reg_st2'] = loss_reg_st2

            return losses_rpn, actionness, start, end
        else:
            loc_st2 = self.bd_refine.update_bd(loc_dec, start_offsets, end_offsets)
            return loc_enc, score_enc, loc_dec, score_dec, loc_st2, actionness, start, end






