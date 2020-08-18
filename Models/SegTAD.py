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

class SegTAD(nn.Module):
    def __init__(self, opt):
        super(SegTAD, self).__init__()

        self.hidden_dim_1d = opt['decoder_out_dim']
        self.bs = opt["batch_size"]
        self.prop_scale = opt["prop_temporal_scale"]
        self.num_sample = opt["num_sample"]
        self.roi_method = opt['RoI_method']
        self.pretrain_model = opt['pretrain_model']
        self.binary_actionness = opt['binary_actionness']
        self.stage2 = opt['stage2']
        self.edge_type = opt['edge_type']
        self.is_train = opt['is_train']
        self.split_gcn = opt['split_gcn']
        self.splits = opt['splits']
        self.num_samp_prop = sum(opt['num_samp_prop'])
        self.tem_best_loss = 10000000
        self.hidden_dim_2d = 128
        self.hidden_dim_3d = 512
        self.PBR_actionness = opt['PBR_actionness']
        self.feat_dim = opt['feat_dim']
        # decoder_num_classes = 1 if opt['binary_actionness'] == 'true' else opt['decoder_num_classes']

        # Segmentor part
        self.fpn = FPN(opt)
        if self.pretrain_model == 'FeatureEnhancer':
            for param in self.FeatureEnhancer.parameters():
                param.requires_grad = False

        self.head_enc = Head(opt)
        self.head_dec = Head(opt)

        self.anchors = AnchorGenerator(opt).anchors
        self.rpn_loss_compute = LossComputation(opt)

        self.gen_predictions = ActionGenerator(opt)

        if self.PBR_actionness:
            self.FBv2_last = nn.Sequential(
                nn.ConvTranspose1d(in_channels=self.hidden_dim_1d, out_channels=self.hidden_dim_1d,kernel_size=3,stride=2,padding=1, output_padding=1, groups=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose1d(in_channels=self.hidden_dim_1d, out_channels=self.hidden_dim_1d,kernel_size=3,stride=2,padding=1, output_padding=1, groups=1),
                nn.ReLU(inplace=True),
            )
            self.FBV2_input = nn.Sequential(
                nn.Conv1d(in_channels=self.feat_dim, out_channels=self.hidden_dim_1d,kernel_size=3,stride=1,padding=1, groups=1),
                nn.ReLU(inplace=True),
                nn.Conv1d(in_channels=self.hidden_dim_1d, out_channels=self.hidden_dim_1d,kernel_size=3,stride=1,padding=1, groups=1),
                nn.ReLU(inplace=True),
            )
            self.FBV2_final = nn.Sequential(
                nn.Conv1d(in_channels=self.hidden_dim_1d * 2, out_channels=self.hidden_dim_1d,kernel_size=3,stride=1,padding=1, groups=1),
                nn.ReLU(inplace=True),
            )

        # Generate action/start/end scores
        self.head_actionness = nn.Sequential(
            nn.Conv1d(self.hidden_dim_1d, self.hidden_dim_1d, kernel_size=3, padding=1, groups=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.hidden_dim_1d, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.head_startness = nn.Sequential(
            nn.Conv1d(self.hidden_dim_1d, self.hidden_dim_1d, kernel_size=3, padding=1, groups=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.hidden_dim_1d, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.head_endness = nn.Sequential(
            nn.Conv1d(self.hidden_dim_1d, self.hidden_dim_1d, kernel_size=3, padding=1, groups=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.hidden_dim_1d, 1, kernel_size=1),
            nn.Sigmoid()
        )


    def _forward_train(self, cls_pred_enc, reg_pred_enc, cls_pred_dec, reg_pred_dec, gt_bbox, num_gt):

        loss_box_cls0, loss_box_reg0, loss_box_cls1, loss_box_reg1 = self.rpn_loss_compute(cls_pred_enc, reg_pred_enc, cls_pred_dec, reg_pred_dec, gt_bbox, num_gt, self.anchors)
        losses = {
            "loss_cls_enc": loss_box_cls0,
            "loss_reg_enc": loss_box_reg0,
            "loss_cls_dec": loss_box_cls1,
            "loss_reg_dec": loss_box_reg1,
        }

        return losses


    def _forward_test(self, cls_pred_enc, reg_pred_enc, cls_pred_dec, reg_pred_dec):

        loc_pred, score_pred = self.gen_predictions(cls_pred_enc, reg_pred_enc, cls_pred_dec, reg_pred_dec, self.anchors)

        return loc_pred, score_pred


    def FBv2(self, feat_last, input):
        feat1 = self.FBv2_last(feat_last)
        feat2 = self.FBV2_input(input)

        return self.FBV2_final(torch.cat((feat1, feat2), dim=1))

    def forward(self, input, gt_iou_map, gt_bbox = None, num_gt = None):
        # B = input.shape[0]

        feats_enc, feats_dec = self.fpn(input)

        cls_pred_enc, reg_pred_enc = self.head_enc(feats_enc)
        cls_pred_dec, reg_pred_dec = self.head_dec(feats_dec)

        if self.PBR_actionness:
            feat2 = self.FBv2(feats_dec[-1], input)
        else:
            feat2 = feats_dec[-1]

        # Action/start/end scores
        actionness = self.head_actionness(feat2)
        start = self.head_startness(feat2).squeeze(1)
        end = self.head_endness(feat2).squeeze(1)

        if not self.PBR_actionness:
            actionness = F.interpolate(actionness, size=input.size()[2:], mode='linear', align_corners=True)
            start = F.interpolate(start[:, None, :], size=input.size()[2:], mode='linear', align_corners=True).squeeze(1)
            end = F.interpolate(end[:, None, :], size=input.size()[2:], mode='linear', align_corners=True).squeeze(1)

        if self.is_train == 'true':
            losses_rpn = self._forward_train(cls_pred_enc, reg_pred_enc, cls_pred_dec, reg_pred_dec, gt_bbox , num_gt)
            return losses_rpn, actionness, start, end
        else:
            loc_pred, score_pred = self._forward_test(cls_pred_enc, reg_pred_enc, cls_pred_dec, reg_pred_dec)
            return loc_pred, score_pred, actionness, start, end






