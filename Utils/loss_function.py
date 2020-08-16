# -*- coding: utf-8 -*-
import torch
import numpy as np
import torch.nn.functional as F

def bi_loss(prediction, groundtruth, reduction='mean'):
    gt = groundtruth.view(-1)
    pred = prediction.contiguous().view(-1)

    pmask = (gt>0.5).float()
    num_positive = torch.sum(pmask)
    num_entries = len(gt)
    ratio=num_entries/num_positive

    coef_0=0.5*(ratio)/(ratio-1)
    coef_1=coef_0*(ratio-1)
    loss = coef_1*pmask*torch.log(pred+0.00001) + coef_0*(1.0-pmask)*torch.log(1.0-pred+0.00001)

    if reduction == 'mean':
        loss = -torch.mean(loss)
    elif reduction == 'none':
        loss = -torch.mean(loss.view(groundtruth.shape), dim=1)
    return loss



################################################################
# new losses
################################################################

def detect_loss_fuction(opt, pred_conf_map, gt_conf_map, bm_mask, anchor_idx, anchor_num, samp_thr=None):

    pred_map_reg = pred_conf_map[:, 0].contiguous()
    pred_map_cls = pred_conf_map[:, 1].contiguous()
    B = gt_conf_map.shape[0]

    def get_prop_target(in_map):
        a_idx = anchor_idx[:anchor_num[0]]
        start = a_idx[:, 1].long()
        duration = (a_idx[:, 2] - a_idx[:, 1]).long() - 1
        out = in_map[0, duration, start]
        pre = anchor_num[0]
        for i in range(1, B):
            upper = pre + anchor_num[i]
            a_idx = anchor_idx[pre:upper]
            pre = upper

            start = a_idx[:, 1].long()
            duration = (a_idx[:, 2] - a_idx[:, 1]).long() - 1
            out = torch.cat((out, in_map[i, duration, start]), 0)
        return out

    if True:
        gt_conf_map = get_prop_target(gt_conf_map)
    if not samp_thr is None:
        samp_thr[0] = get_prop_target(samp_thr[0])
        samp_thr[1] = get_prop_target(samp_thr[1])

    if opt['det_reg_loss'] == 'true':
        det_reg_loss = det_reg_loss_func(opt, pred_map_reg, gt_conf_map, bm_mask)
    else:
        det_reg_loss = torch.zeros((1,), device=gt_conf_map.device)

    if opt['det_cls_loss'] == 'true':
        det_cls_loss = det_cls_loss_func(opt, pred_map_cls, gt_conf_map, bm_mask, samp_thr)
    else:
        det_cls_loss = torch.zeros((1,), device=gt_conf_map.device)
    return det_reg_loss, det_cls_loss

def det_reg_loss_func(opt, pred_score, gt_iou_map, mask):

    if opt['samp_prop'] != 'none':
        loss = 0.5 * F.mse_loss(pred_score, gt_iou_map)

    else:
        u_hmask = (gt_iou_map > 0.7).float()
        u_mmask = ((gt_iou_map <= 0.7) & (gt_iou_map > 0.3)).float()
        u_lmask = ((gt_iou_map <= 0.3) & (gt_iou_map > 0.)).float()

        if opt['gen_prop'] == 'all':
            u_lmask = u_lmask * mask

        num_h = torch.sum(u_hmask)
        num_m = torch.sum(u_mmask)
        num_l = torch.sum(u_lmask)

        r_m = num_h / num_m
        u_smmask = torch.Tensor(np.random.rand(*gt_iou_map.shape)).cuda()
        u_smmask = u_mmask * u_smmask
        u_smmask = (u_smmask > (1. - r_m)).float()

        r_l = num_h / num_l
        u_slmask = torch.Tensor(np.random.rand(*gt_iou_map.shape)).cuda()
        u_slmask = u_lmask * u_slmask
        u_slmask = (u_slmask > (1. - r_l)).float()

        weights = u_hmask + u_smmask + u_slmask

        loss = F.mse_loss(pred_score* weights, gt_iou_map* weights)

        loss = 0.5 * torch.sum(loss*torch.ones(*weights.shape).cuda()) / torch.sum(weights)

    return loss

#TODO
# 1. IoU threshold
# 2. sampling ratio of positive and negative samples
# 3. Choose right ratio before generating proposals

def det_cls_loss_func(opt, pred_score, gt_iou_map, mask, samp_thr = None):

    # if samp_thr is None:
    #     samp_thr = opt['samp_thr']
    #     samp_thr0 = samp_thr[0]
    #     if len(samp_thr) == 1:
    #         samp_thr1 = samp_thr[0]
    #     else:
    #         samp_thr1 = samp_thr[1]
    #
    # elif isinstance(samp_thr, torch.Tensor):
    if samp_thr == None:
        samp_thr0 = opt['samp_thr'][0]
        if len(opt['samp_thr']) == 1:
            samp_thr1 = opt['samp_thr'][0]
        else:
            samp_thr1 = opt['samp_thr'][1]
    else:
        samp_thr0 = samp_thr[0]
        samp_thr1 = samp_thr[1]

    if opt['samp_prop'] != 'none':
        gt_iou_map_pos = (gt_iou_map > samp_thr0).float()
        gt_iou_map_neg = (gt_iou_map <= samp_thr1).float()
        weight= torch.zeros_like(gt_iou_map_pos)

        positive = gt_iou_map_pos.nonzero()
        negative = (gt_iou_map_neg).nonzero()
        num_pos = len(positive)
        num_neg = len(negative)
        num_total = len(gt_iou_map)
        weight[gt_iou_map_pos==1] = 0.5 * num_total / num_pos
        weight[gt_iou_map_neg==1] = 0.5 * num_total / num_neg

        ind = ((gt_iou_map > samp_thr0) + (gt_iou_map <= samp_thr1)) > 0

        loss = F.binary_cross_entropy(pred_score[ind], gt_iou_map_pos[ind], weight=weight[ind])
    else:
        pmask = (gt_iou_map > 0.9).float()
        nmask = (gt_iou_map <= 0.9).float()
        if opt['gen_prop'] == 'all':
            nmask = nmask * mask

        num_positive = torch.sum(pmask)
        num_entries = num_positive + torch.sum(nmask)
        ratio = num_entries / num_positive
        coef_0 = 0.5 * ratio / (ratio - 1)
        coef_1 = 0.5 * ratio
        epsilon = 0.000001
        loss_pos = coef_1 * torch.log(pred_score + epsilon) * pmask
        loss_neg = coef_0 * torch.log(1.0 - pred_score + epsilon) * nmask
        loss = -1 * torch.sum(loss_pos + loss_neg) / num_entries

    return loss

def get_mask(tscale):
    bm_mask = []
    for idx in range(tscale):
        mask_vector = [1 for i in range(tscale - idx)
                       ] + [0 for i in range(idx)]
        bm_mask.append(mask_vector)
    bm_mask = np.array(bm_mask, dtype=np.float32)
    return torch.Tensor(bm_mask)

def boundary_loss_function(pred_start, pred_end, gt_start, gt_end):
    def bi_loss(pred_score, gt_label):
        # pred_score = F.interpolate(pred_score[None, :, :], size=gt_label.size()[-1], mode='linear', align_corners=True).squeeze(0)

        pred_score = pred_score.view(-1)
        gt_label = gt_label.view(-1)
        pmask = (gt_label > 0.5).float()
        num_entries = len(pmask)
        num_positive = torch.sum(pmask)
        ratio = num_entries / num_positive
        coef_0 = 0.5 * ratio / (ratio - 1)
        coef_1 = 0.5 * ratio
        epsilon = 0.000001
        loss_pos = coef_1 * torch.log(pred_score + epsilon) * pmask
        loss_neg = coef_0 * torch.log(1.0 - pred_score + epsilon)*(1.0 - pmask)
        loss = -1 * torch.mean(loss_pos + loss_neg)
        return loss

    loss_start = bi_loss(pred_start, gt_start)
    loss_end = bi_loss(pred_end, gt_end)
    loss = loss_start + loss_end
    return loss