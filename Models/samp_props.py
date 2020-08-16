import torch
import torch.nn as nn
from .GCNs import segment_tiou_mat, segment_dist_mat
import numpy as np


def iou_anchors_gts(anchor, gt):
    """Compute jaccard score between a box and the anchors.
    """
    anchors_min = anchor[:, 0]
    anchors_max = anchor[:, 1]
    box_min = gt[:, 0]
    box_max = gt[:, 1]
    len_anchors = anchors_max - anchors_min
    int_xmin = torch.max(anchors_min[:, None], box_min)
    int_xmax = torch.min(anchors_max[:, None], box_max)
    inter_len = torch.clamp(int_xmax - int_xmin, min=0)
    union_len = torch.clamp(len_anchors[:, None] + box_max - box_min - inter_len, min=0)
    # print inter_len,union_len
    jaccard = inter_len / union_len
    return jaccard


class _Prop_Sampler(nn.Module):

    def __init__(self, opt):
        super(_Prop_Sampler, self).__init__()

        self.num_samp_prop = opt['num_samp_prop']
        self.samp_prop = opt['samp_prop']
        self.iou_thr_method = opt['iou_thr_method']
        self.samp_thr = opt['samp_thr']
        self.low, self.high = opt['iou_thr_bound']
        self.samp_gcn = opt['samp_gcn']
        self.stage2 = opt['stage2']
        self.is_train = opt['is_train']
        self.num_neigh = opt['num_neigh']
        self.edge_type = opt['edge_type']
        self.prop_temp_scale = opt['prop_temporal_scale']
        self.topk = 50

        self.num_fg_v = self.num_samp_prop[0]
        self.num_mid_v = self.num_samp_prop[1]
        self.num_bg_v = self.num_samp_prop[2]

    def forward(self, gt_iou_map, all_idx_dur_st, gt_bbox = None, num_gt = None):

        if self.is_train == 'true':
            anchor_idx_st_end, samp_thr, pos_idx_st_end = self.samp_props(gt_iou_map, all_idx_dur_st, gt_bbox, num_gt)
            return anchor_idx_st_end, samp_thr, pos_idx_st_end

        elif self.is_train == 'false':
            anchor_idx_st_end = self._sparsedurStr_to_denseStrEnd(all_idx_dur_st)
            return anchor_idx_st_end, [], []

    def _get_samp_thr_ATSS(self, gt_bbox, all_idx_dur_st, num_gts):
        num_img = len(gt_bbox)

        dense_durStr = (all_idx_dur_st >= 1).nonzero().float()                     # batch idx, duration, start
        dense_durStr[:, 1] = dense_durStr[:, 1] + 1

        all_thr_sparse = torch.zeros((num_img, self.prop_temp_scale, self.prop_temp_scale), device=all_idx_dur_st.device, requires_grad=False)
        for im_i in range(num_img):
            num_gt = num_gts[im_i]
            bbox_i = (gt_bbox[im_i])[:num_gt, :-1]

            idx_dur_st_i = dense_durStr[dense_durStr[:,0]==im_i, 1:] / self.prop_temp_scale
            idx_str_end_i = idx_dur_st_i.clone()
            idx_str_end_i[:, 0] = idx_dur_st_i[:, 1]
            idx_str_end_i[:, 1] = idx_dur_st_i[:, 1] + idx_dur_st_i[:, 0]
            ious = iou_anchors_gts(idx_str_end_i, bbox_i)

            max_iou, idx_max_iou = torch.max(ious, dim=1)

            gt_ctr = (bbox_i[:, 0] + bbox_i[:, 1]) / 2.
            anchor_ctr = idx_dur_st_i[:, 1] + idx_dur_st_i[:, 0] / 2.
            distances = torch.abs(anchor_ctr[:, None] - gt_ctr[None, :])

            _, candidate_idxs = distances.topk(self.topk, dim=0, largest=False)
            candidate_ious = ious[candidate_idxs, torch.arange(num_gt)]
            iou_mean_per_gt = candidate_ious.mean(0)
            iou_std_per_gt = candidate_ious.std(0)
            iou_thresh_per_gt = iou_mean_per_gt + iou_std_per_gt

            iou_thr_anchor = iou_thresh_per_gt[idx_max_iou]
            all_thr_sparse[im_i, dense_durStr[dense_durStr[:,0]==im_i, 1].to(dtype=torch.long) - 1, dense_durStr[dense_durStr[:,0]==im_i, 2].to(dtype=torch.long)] = iou_thr_anchor

        return  all_thr_sparse


    def _get_samp_thr_linear(self, gt_iou_map):
        B = gt_iou_map.size(0)
        # y = (High - Low) * x + Low
        threholds = (self.high - self.low)  *torch.tensor(range(self.prop_temp_scale), dtype=torch.float32, device=gt_iou_map.device) / self.prop_temp_scale + self.low

        all_thr = threholds[None, :, None].repeat(B, 1, self.prop_temp_scale)

        return all_thr


    def samp_props(self, gt_iou_map, all_idx_dur_st, gt_bbox, num_gt):

        # Sample proposals for training, balancing positive and negative samples
        if self.samp_prop == 'rand3':
            u_hmask = (gt_iou_map > 0.7).float() * all_idx_dur_st
            u_mmask = ((gt_iou_map <= 0.7) & (gt_iou_map > 0.3)).float() * all_idx_dur_st
            u_lmask = ((gt_iou_map <= 0.3) & (gt_iou_map > 0.)).float() * all_idx_dur_st

            num_h = torch.sum(u_hmask, dim=(1,2))
            num_m = torch.sum(u_mmask, dim=(1,2))
            num_l = torch.sum(u_lmask, dim=(1,2))

            r_h = self.num_fg_v / num_h
            r_h[r_h>1] = 1
            u_shmask = torch.Tensor(np.random.rand(*gt_iou_map.shape)).to(device=gt_iou_map.device)
            u_shmask = u_hmask * u_shmask
            u_shmask = (u_shmask > (1. - r_h)[:,None, None]).float()

            r_m = self.num_mid_v / num_m
            r_m[r_m>1] = 1
            u_smmask = torch.Tensor(np.random.rand(*gt_iou_map.shape)).to(device=gt_iou_map.device)
            u_smmask = u_mmask * u_smmask
            u_smmask = (u_smmask > (1. - r_m)[:,None, None]).float()

            r_l = self.num_bg_v / num_l
            r_l[r_l>1] = 1
            u_slmask = torch.Tensor(np.random.rand(*gt_iou_map.shape)).to(device=gt_iou_map.device)
            u_slmask = u_lmask * u_slmask
            u_slmask = (u_slmask > (1. - r_l)[:,None, None]).float()

            all_idx_dur_st_sampl = u_shmask + u_smmask + u_slmask

            if (self.stage2 == 'gcn') and (self.samp_gcn == 'sage'):
                anchor_idx_st_end = self._gen_gcn_props_sage(all_idx_dur_st_sampl, all_idx_dur_st)

            else:
                anchor_idx_st_end = self._sparsedurStr_to_denseStrEnd(all_idx_dur_st_sampl)

        elif self.samp_prop == 'rand2':

            if self.iou_thr_method == 'fixed':
                samp_thr0 = self.samp_thr[0]
                if len(self.samp_thr) == 1:
                    samp_thr1 = self.samp_thr[0]
                else:
                    samp_thr1 = self.samp_thr[1]
            elif self.iou_thr_method == 'atss':
                samp_thr0 = self._get_samp_thr_ATSS(gt_bbox, all_idx_dur_st, num_gt)
                samp_thr1 = samp_thr0
            elif self.iou_thr_method == 'anchor_linear':
                samp_thr0 = self._get_samp_thr_linear(gt_iou_map)
                samp_thr1 = samp_thr0

            fg_mask = (gt_iou_map > samp_thr0).float() * all_idx_dur_st
            bg_mask = (gt_iou_map <= samp_thr1).float() * all_idx_dur_st

            num_fg = torch.sum(fg_mask, dim=(1,2))
            num_bg = torch.sum(bg_mask, dim=(1,2))

            num_samp_fg = num_fg.clone()
            num_samp_fg[num_fg>self.num_fg_v] = self.num_fg_v
            num_samp_bg = self.num_fg_v + self.num_bg_v - num_samp_fg

            # Sample positive proposals
            fg_sampl_mask = torch.zeros_like(fg_mask)
            for i, num_fg_vid in enumerate(num_fg):
                fg_idx = fg_mask[i].nonzero().to(torch.long)
                rand_nums = (torch.randperm(int(num_fg_vid))).to(dtype=torch.long, device=fg_mask.device)
                fg_sampl_idx = fg_idx[rand_nums[:int(num_samp_fg[i])]]
                fg_sampl_mask[i][tuple(fg_sampl_idx.transpose(0,1))] = 1

            # Sample negative proposals
            bg_sampl_mask = torch.zeros_like(bg_mask)
            for i, num_bg_vid in enumerate(num_bg):
                bg_idx = bg_mask[i].nonzero().to(torch.long)
                rand_nums = (torch.randperm(int(num_bg_vid))).to(dtype=torch.long, device=bg_mask.device)
                bg_sampl_idx = bg_idx[rand_nums[:int(num_samp_bg[i])]]
                bg_sampl_mask[i][tuple(bg_sampl_idx.transpose(0,1))] = 1

            all_idx_dur_st_sampl = fg_sampl_mask + bg_sampl_mask

            if (self.stage2 == 'gcn') and (self.samp_gcn == 'sage'):
                anchor_idx_st_end = self._gen_gcn_props_sage(all_idx_dur_st_sampl, all_idx_dur_st)

            else:
                anchor_idx_st_end = self._sparsedurStr_to_denseStrEnd(all_idx_dur_st_sampl)

            # pos_idx_st_end_padding = all_idx_dur_st.self._sparsedurStr_to_denseStrEnd(all_idx_dur_st).view(4, -1, 3)
            pos_idx_st_end = self._sparsedurStr_to_denseStrEnd(fg_mask)  # by Catherine
            neg_idx_st_end = self._sparsedurStr_to_denseStrEnd(bg_mask)  # by Catherine

        elif self.samp_prop == 'none':
            anchor_idx_st_end = self._sparsedurStr_to_denseStrEnd(all_idx_dur_st)

        # sampled_gt_iou = self.test(anchor_idx_st_end, gt_iou_map)

        if self.iou_thr_method == 'fixed':
            samp_thr_ret = None
        else:
            samp_thr_ret = [samp_thr0, samp_thr1]

        return anchor_idx_st_end, samp_thr_ret, pos_idx_st_end

    def test(self, anchor_idx, in_map):

        #########################################################################
        # Solution 1
        #########################################################################
        out1 = in_map[anchor_idx>0]


        #########################################################################
        # Solution 2
        #########################################################################
        B = 4

        # STEP 1
        dense_durStr = (anchor_idx >= 1).nonzero().float()                     # batch idx, duration, start
        dense_durStr[:, 1] = dense_durStr[:, 1] + 1

        # Convert the format of all proposals to (batch_idx, start_idx, end_idx) from (batch idx, duration, start)
        dense_StrEnd = dense_durStr.clone()     # batch idx, start, end
        dense_StrEnd[:, 1] = dense_durStr[:, 2]     # batch idx, start, end
        dense_StrEnd[:, 2] = dense_durStr[:, 1] + dense_durStr[:, 2]     # batch idx, start, end

        anchor_num = torch.tensor([torch.sum(dense_StrEnd[:, 0] ==i) for i in range(B)], device=anchor_idx.device) # number of anchors per sequence

        # STEP 2
        a_idx = dense_StrEnd[:anchor_num[0]]
        start = a_idx[:, 1].long()
        duration = (a_idx[:, 2] - a_idx[:, 1]).long() - 1
        out2 = in_map[0, duration, start]
        pre = anchor_num[0]
        for i in range(1, B):
            upper = pre + anchor_num[i]
            a_idx = dense_StrEnd[pre:upper]
            pre = upper

            start = a_idx[:, 1].long()
            duration = (a_idx[:, 2] - a_idx[:, 1]).long() - 1
            out2 = torch.cat((out2, in_map[i, duration, start]), 0)

        return out2



    def _gen_gcn_props_sage(self, sampl_idx_dur_st, all_idx_dur_st):
        B = sampl_idx_dur_st.shape[0]

        samp_idx_str_end = self._sparsedurStr_to_denseStrEnd(sampl_idx_dur_st).view(B, -1, 3)
        all_idx_str_end = self._sparsedurStr_to_denseStrEnd(all_idx_dur_st).view(B, -1, 3)

        num_prop_v = samp_idx_str_end.shape[1]

        bt_indx = []
        for i in range(B):
            bt_indx += [i]*self.num_neigh*num_prop_v

        if self.edge_type == 'pgcn_iou':
            iou_array, _ = segment_tiou_mat(samp_idx_str_end[:, :, 1:], all_idx_str_end[:, :, 1:])
            iou_array[iou_array==1] = 0
            neigh_idx = torch.topk(iou_array, self.num_neigh, dim=2)[1]

            iou_array, _ = segment_tiou_mat(all_idx_str_end[bt_indx, neigh_idx.view(-1), 1:].view(B, -1, 2), all_idx_str_end[:, :, 1:])
            iou_array[iou_array==1] = 0
            iou_array = iou_array.view(B, num_prop_v, self.num_neigh, -1)
            for i in range(self.num_neigh):
                neigh_idx = torch.cat((neigh_idx, torch.topk(iou_array[:, :, i, : ], self.num_neigh, dim=2)[1]), dim=-1)

        elif self.edge_type == 'pgcn_dist':
            iou_array = segment_dist_mat(samp_idx_str_end[:, :, 1:], all_idx_str_end[:, :, 1:])
            neigh_idx = torch.topk(iou_array, self.num_neigh, dim=2, largest=False)[1]

            iou_array = segment_dist_mat(all_idx_str_end[bt_indx, neigh_idx.view(-1), 1:].view(B, -1, 2), all_idx_str_end[:, :, 1:])
            iou_array = iou_array.view(B, num_prop_v, self.num_neigh, -1)
            for i in range(self.num_neigh):
                neigh_idx = torch.cat((neigh_idx, torch.topk(iou_array[:, :, i, : ], self.num_neigh, dim=2, largest=False)[1]), dim=-1)

        bt_indx = []
        for i in range(B):
            bt_indx += [i]*self.num_neigh*num_prop_v*(self.num_neigh+1)

        gcn_props_str_end = all_idx_str_end[bt_indx, neigh_idx.view(-1), :].view(B, num_prop_v, -1, 3)
        gcn_props_str_end = torch.cat((samp_idx_str_end.unsqueeze(2), gcn_props_str_end), dim=2)

        return gcn_props_str_end.view(-1, 3)


    def _sparsedurStr_to_denseStrEnd(self, sparse_durStr):

        dense_durStr = (sparse_durStr >= 1).nonzero().float()                     # batch idx, duration, start
        dense_durStr[:, 1] = dense_durStr[:, 1] + 1

        # Convert the format of all proposals to (batch_idx, start_idx, end_idx) from (batch idx, duration, start)
        dense_StrEnd = dense_durStr.clone()     # batch idx, start, end
        dense_StrEnd[:, 1] = dense_durStr[:, 2]     # batch idx, start, end
        dense_StrEnd[:, 2] = dense_durStr[:, 1] + dense_durStr[:, 2]     # batch idx, start, end

        return dense_StrEnd
