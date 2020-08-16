import math
import numpy as np
import torch
import torch.nn as nn
from Utils.RoIAlign.align import Align1DLayer
from .gen_props import _Prop_Generator
from .samp_props import _Prop_Sampler


class BoundaryMatch(nn.Module):
    def __init__(self, opt):
        super(BoundaryMatch, self).__init__()
        self.tscale = opt["prop_temporal_scale"]
        self.num_sample = opt["num_sample"]
        self.prop_boundary_ratio = opt["prop_boundary_ratio"]
        self.num_sample_perbin = opt["num_sample_perbin"]
        self._get_interp1d_mask()

    def forward(self, x):
        input_size = x.size()
        out = torch.matmul(x, self.sample_mask).reshape(input_size[0],input_size[1],self.num_sample,self.tscale,self.tscale)

        return out

    def _get_interp1d_mask(self):
        # generate sample mask for each point in Boundary-Matching Map
        mask_mat = []
        for start_index in range(self.tscale):
            mask_mat_vector = []
            for duration_index in range(self.tscale):
                if start_index + duration_index < self.tscale:
                    p_xmin = start_index
                    p_xmax = start_index + duration_index
                    center_len = float(p_xmax - p_xmin) + 1
                    sample_xmin = p_xmin - center_len * self.prop_boundary_ratio
                    sample_xmax = p_xmax + center_len * self.prop_boundary_ratio
                    p_mask = self._get_interp1d_bin_mask(
                        sample_xmin, sample_xmax, self.tscale, self.num_sample,
                        self.num_sample_perbin)
                else:
                    p_mask = np.zeros([self.tscale, self.num_sample])
                mask_mat_vector.append(p_mask)
            mask_mat_vector = np.stack(mask_mat_vector, axis=2)
            mask_mat.append(mask_mat_vector)
        mask_mat = np.stack(mask_mat, axis=3)
        mask_mat = mask_mat.astype(np.float32)
        self.sample_mask = nn.Parameter(torch.Tensor(mask_mat).view(self.tscale, -1), requires_grad=False)

    def _get_interp1d_bin_mask(self, seg_xmin, seg_xmax, tscale, num_sample, num_sample_perbin):
        # generate sample mask for a boundary-matching pair
        plen = float(seg_xmax - seg_xmin)
        plen_sample = plen / (num_sample * num_sample_perbin - 1.0)
        total_samples = [
            seg_xmin + plen_sample * ii
            for ii in range(num_sample * num_sample_perbin)
        ]
        p_mask = []
        for idx in range(num_sample):
            bin_samples = total_samples[idx * num_sample_perbin:(idx + 1) * num_sample_perbin] # t_n
            bin_vector = np.zeros([tscale])
            for sample in bin_samples: # t_n
                sample_upper = math.ceil(sample)
                sample_decimal, sample_down = math.modf(sample)
                if int(sample_down) <= (tscale - 1) and int(sample_down) >= 0:
                    bin_vector[int(sample_down)] += 1 - sample_decimal
                if int(sample_upper) <= (tscale - 1) and int(sample_upper) >= 0:
                    bin_vector[int(sample_upper)] += sample_decimal
            bin_vector = 1.0 / num_sample_perbin * bin_vector
            p_mask.append(bin_vector)
        p_mask = np.stack(p_mask, axis=1)
        return p_mask


class GraphAlign(nn.Module):
    def __init__(self, opt, k=3, t=100, d=100, bs=64, samp=0, style=0):
        super(GraphAlign, self).__init__()
        self.k = k
        self.t = t
        self.d = d
        self.bs = bs
        self.style = style
        self.expand_ratio = 0.5
        self.resolution = 32

        self.binary_actionness = opt['binary_actionness']

        self.prop_gen = _Prop_Generator(opt)
        self.prop_samp = _Prop_Sampler(opt)
        self.align_inner = Align1DLayer(self.resolution, samp)

    def forward(self, x, start, end, actionnes, gt_iou_map, gt_bbox = None, num_gt = None):

        # Generate proposals
        anchors, anchor_coord, anchor_num, samp_thr, pos_idx_st_end = self._get_anchors(start, end, actionnes, gt_iou_map, gt_bbox, num_gt) # anchor: (bs*tscale*tscal, 3); (num_anchors, 3)

        # RoI pooling
        feat = self.align_inner(x, anchors) #  (bs*tscale*tscal, ch, resolution);  (num_anchors, ch, resolution)

        if False:
            bs, ch, t = x.shape
            feat = feat.view(bs, t, t, ch, -1).permute(0, 3, 4, 2, 1)  # (bs,ch,32,t,t)  duration, start

        return feat, anchor_coord, anchor_num, samp_thr, pos_idx_st_end



    def _get_anchors(self, start, end, actionness, gt_iou_map, gt_bbox = None, num_gt = None):
        B, _ = start.size()

        # Generate proposals
        all_idx_dur_st = self.prop_gen(start, end, actionness)

        # Sample proposals if training
        anchor_idx_st_end, samp_thr, pos_idx_st_end = self.prop_samp(gt_iou_map, all_idx_dur_st, gt_bbox, num_gt)

        # Extend proposal boundaries to include context
        anchor_idx_st_end_org = anchor_idx_st_end.clone()
        context_length = (anchor_idx_st_end[:, 2] - anchor_idx_st_end[:, 1]) * self.expand_ratio  # batch idx, start-extension, end+extension
        anchor_idx_st_end[:, 1] = anchor_idx_st_end[:, 1] - context_length
        anchor_idx_st_end[:, 2] = anchor_idx_st_end[:, 2] + context_length

        anchor_num_out = torch.tensor([torch.sum(anchor_idx_st_end[:, 0] ==i) for i in range(B)], device=start.device) # number of anchors per sequence

        return anchor_idx_st_end, anchor_idx_st_end_org, anchor_num_out, samp_thr, pos_idx_st_end





