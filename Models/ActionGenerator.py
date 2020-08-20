
import torch
import torch.nn as nn
import math
from .BoxCoder import  BoxCoder

class ActionGenerator(object):
    def __init__(self, opt):
        super(ActionGenerator, self).__init__()

        self.pre_nms_thresh = 0.00
        self.pre_nms_top_n = 10000
        self.num_classes = 1 if opt['dataset'] == 'activitynet' else opt['decoder_num_classes']

        self.box_coder = BoxCoder(opt)


    def __call__(self, cls_pred_enc, reg_pred_enc, cls_pred_dec, reg_pred_dec, anchors):
        bs = cls_pred_enc[0].shape[0]

        # First stage: encoder
        anchors = [anchor.unsqueeze(0).repeat(bs, 1, 1).to(device=cls_pred_enc[0].device) for anchor in anchors]
        all_anchors = torch.cat(anchors, dim=1)          # bs, levels*positions*scales, left-right
        loc_enc, score_enc, label_enc = self._call_one_stage(cls_pred_enc, reg_pred_enc, all_anchors)

        # Second stage: decoder
        anchors_update = torch.stack(loc_enc, dim=0)
        loc_dec, score_dec, label_dec = self._call_one_stage(cls_pred_enc, reg_pred_enc, anchors_update)

        return torch.stack(score_enc, dim=0), torch.stack(loc_enc, dim=0), torch.stack(score_dec, dim=0), torch.stack(loc_dec, dim=0)

    def _call_one_stage(self, cls_pred, reg_pred, all_anchors):

        N = cls_pred[0].shape[0]

        cls_pred = torch.cat(cls_pred, dim=2).permute(0, 2, 1).reshape(N, -1, self.num_classes).sigmoid()  # bs, levels*positions*scales, num_cls
        reg_pred = torch.cat(reg_pred, dim=2).permute(0, 2, 1).reshape(N, -1, 2)   # bs, levels*positions*scales, 2


        candidate_inds = cls_pred > self.pre_nms_thresh
        pre_nms_top_n = candidate_inds.view(N, -1).sum(1)
        pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_top_n)

        loc_res = []
        score_res = []
        label_res = []
        for cls_seq, reg_seq, anchor_seq, pre_nms_top_n_seq, candidate_inds_seq in zip(cls_pred, reg_pred, all_anchors, pre_nms_top_n, candidate_inds):

            cls_seq1 = cls_seq[candidate_inds_seq]
            cls_seq1, top_k_indices = cls_seq1.topk(pre_nms_top_n_seq, sorted=False)

            per_candidate_nonzeros = candidate_inds_seq.nonzero()[top_k_indices, :]

            per_box_loc = per_candidate_nonzeros[:, 0]

            loc_pred = self.box_coder.decode(
                reg_seq[per_box_loc, :].view(-1, 2),        # levels*positions*scales, 2
                anchor_seq[per_box_loc, :].view(-1, 2)      # levels*positions*scales, 2
            )

            score_pred = cls_seq[per_box_loc, 0]
            label_pred = score_pred > 0.5

            loc_res.append(loc_pred)
            score_res.append(score_pred)
            label_res.append(label_pred)

        return loc_res, score_res, label_res


    def cat_boxlist(self, bboxes):
        """
        Concatenates a list of BoxList (having the same image size) into a
        single BoxList

        Arguments:
            bboxes (list[BoxList])
        """
        assert isinstance(bboxes, (list, tuple))
        assert all(isinstance(bbox, dict) for bbox in bboxes)

        res = {}
        res['loc'] = torch.ones([0, 2])
        for bb in bboxes:
            res['loc'] = torch.cat(res['loc'], bb['loc'], dim=0)

        return res
