import torch
import torch.nn as nn
from Utils.loss_function import get_mask
import torch.nn.functional as F
import math
from Utils.RoIAlign.align import Align1DLayer

class _Prop_Generator(nn.Module):

    def __init__(self, opt):
        super(_Prop_Generator, self).__init__()

        self.tscale = opt["prop_temporal_scale"]
        self.gen_prop = opt['gen_prop']
        self.total_num_prop_gp = opt['total_num_prop_gp']
        self.num_thr_action_gp = opt['num_thr_action_gp']
        self.step_alter_win_gp = opt['step_alter_win_gp']
        self.thr_start_end_gp = opt['thr_start_end_gp']
        self.resolution = 32

        self.get_prop_score = Align1DLayer(self.resolution, 0)
        self.get_prop_score_inflate = Align1DLayer(int(self.resolution * 1.5), 0)

    def forward(self, start, end, actionness):
        all_idx_dur_st = self.gen_props(start, end, actionness)

        return all_idx_dur_st


    def _cal_IOC(self, str_end_v, actionness):
        alpha = 0.25
        num_props = str_end_v.shape[0]
        scores = torch.zeros((num_props), dtype=torch.float32)
        for i, str_end in enumerate(str_end_v):
            in_score = torch.mean(actionness[str_end[0]: str_end[1]])

            prop_length = str_end[1] - str_end[0]
            start_inflate = max(0, math.floor(str_end[0] - alpha * prop_length))
            end_inflate = max(num_props, math.ceil(str_end[1] + alpha * prop_length))
            out_score = (torch.sum(actionness[start_inflate:str_end[0]]) + torch.sum(actionness[str_end[ 1]:end_inflate])) / ((str_end[0] - start_inflate)+(end_inflate-str_end[1]))
            scores[i] = in_score - out_score

        return scores


    def _cal_IOC_ralign(self, str_end_v, actionness):
        alpha = 0.25
        str_end_v = str_end_v.to(torch.float32)

        T = actionness.shape[1]
        inflate_length = torch.ceil((str_end_v[:,2] - str_end_v[:,1]) * alpha)
        str_end_v_inflate = str_end_v.clone()
        str_end_v_inflate [:, 1] = torch.max(str_end_v[:, 1] - inflate_length, torch.zeros_like(str_end_v[:, 0]))
        str_end_v_inflate [:, 2] = torch.min(str_end_v[:, 2] + inflate_length, torch.ones_like(str_end_v[:, 0])*T)


        f_in = self.get_prop_score(actionness[:, None, :], str_end_v)
        f_in = torch.sum(f_in, dim=2).squeeze(1)
        f_all = self.get_prop_score(actionness[:, None, :], str_end_v_inflate)
        f_all = torch.sum(f_all, dim=2).squeeze(1)
        f_out = f_all - f_in

        scores = f_in - f_out

        return scores

    def gen_props(self, start, end, actionness):

        B, T = start.size()
        mask = get_mask(self.tscale).to(device=actionness.device)

        if self.gen_prop == 'start_end':

            step = int(T / self.tscale)
            pred_start_v = start[:, ::step]
            pred_end_v = end[:, ::step]

            start_sel_idx1 = torch.cat((torch.tensor(1, dtype=torch.uint8).expand(B).unsqueeze(1).to(device=start.device), (pred_start_v[:,1:] - pred_start_v[:,:-1] > 0).to(torch.uint8)), 1)
            start_sel_idx2 = torch.cat(((pred_start_v[:,:-1] - pred_start_v[:,1:] > 0).to(torch.uint8), torch.tensor(1, dtype=torch.uint8).expand(B).unsqueeze(1).to(device=start.device)), 1)
            start_sel_idx3 = (pred_start_v > (self.thr_start_end_gp * torch.max(pred_start_v, dim=1, keepdim=True)[0])).to(torch.uint8)
            start_sel_idx = start_sel_idx1*start_sel_idx2 + start_sel_idx3


            end_sel_idx1 = torch.cat((torch.tensor(1, dtype=torch.uint8).expand(B).unsqueeze(1).to(device=start.device), (pred_end_v[:,1:] - pred_end_v[:,:-1] > 0).to(torch.uint8)), 1)
            end_sel_idx2 = torch.cat(((pred_end_v[:,:-1] - pred_end_v[:,1:] > 0).to(torch.uint8), torch.tensor(1, dtype=torch.uint8).expand(B).unsqueeze(1).to(device=start.device)), 1)
            end_sel_idx3 = (pred_end_v > (self.thr_start_end_gp * torch.max(pred_end_v, dim=1, keepdim=True)[0])).to(torch.uint8)
            end_sel_idx = end_sel_idx1*end_sel_idx2 + end_sel_idx3

            all_idx = start_sel_idx.unsqueeze(2) * end_sel_idx.unsqueeze(1) # [start, end]
            anchor_idx_st_end = all_idx.nonzero()
            anchor_idx_st_end = anchor_idx_st_end[anchor_idx_st_end[:, 2] >= anchor_idx_st_end[:, 1]]

            score_v = start[anchor_idx_st_end[:,0].to(torch.long), anchor_idx_st_end[:,1].to(torch.long)] * end[anchor_idx_st_end[:,0].to(torch.long), anchor_idx_st_end[:,2].to(torch.long)]
            all_idx_dur_st = torch.zeros((B, self.tscale, self.tscale), device=start.device, requires_grad=False)
            for i in range(B):
                scores = score_v[anchor_idx_st_end[:,0]==i]
                str_end = anchor_idx_st_end[anchor_idx_st_end[:,0]==i, 1:]
                order = scores.argsort(descending=True)
                if self.total_num_prop_gp <= str_end.shape[0]:
                    str_end = str_end[order[:self.total_num_prop_gp]]
                else:
                    num_remain = self.total_num_prop_gp - str_end.shape[0]
                    str_end_mat = str_end.new_zeros((T, T))
                    str_end[:, 1] = str_end[:, 1]- 1
                    str_end_mat[tuple(str_end.transpose(0,1).to(torch.long))] = 1
                    temp_list = (str_end_mat==0).nonzero()
                    temp_list = temp_list[temp_list[:,0]<=temp_list[:, 1]]
                    rand_nums = torch.randperm(len(temp_list))
                    str_end = torch.cat((str_end, temp_list[rand_nums[:num_remain]]), dim=0)
                starts = str_end[:, 0]
                durations_minus1 = str_end[:, 1] - str_end[:, 0] #  should be str_end_v[:, 1] + 1 - str_end_v[:, 0]
                all_idx_dur_st[i, durations_minus1, starts] = 1

                # Use proposals at 125*125 locations
        elif self.gen_prop == 'all':

            all_idx_dur_st = torch.ones((B, self.tscale, self.tscale), device=start.device, requires_grad=False) # batch idx, duration, start

            all_idx_dur_st = all_idx_dur_st * mask

        # Use proposals at 125*125 locations
        elif self.gen_prop == 'alter_window':
            # all_idx_dur_st = torch.ones((B, self.tscale, self.tscale), device=start.device, requires_grad=False) # batch idx, duration, start
            #
            # all_idx_dur_st[:, ::self.step_alter_win, :] = 0  # Remove the length-1 proposals
            # all_idx_dur_st[:, :, 1::self.step_alter_win] = 0   # Remove the proposals that start from an odd number

            all_idx_dur_st = torch.zeros((B, self.tscale, self.tscale), device=start.device, requires_grad=False) # batch idx, duration, start
            all_idx_dur_st[:, 1::self.step_alter_win_gp, ::self.step_alter_win_gp] = 1

            all_idx_dur_st = all_idx_dur_st * mask

        elif self.gen_prop == 'sliding_window':
            all_idx_dur_st = torch.zeros((B, self.tscale, self.tscale), device=start.device, requires_grad=False) # batch idx, duration, start

            all_idx_dur_st[:, :int(self.tscale/4), :] = 1
            all_idx_dur_st[:, int(self.tscale/4):int(self.tscale/2), ::2] = 1
            all_idx_dur_st[:, int(self.tscale/2):, ::4] = 1

            all_idx_dur_st = all_idx_dur_st * mask

        elif self.gen_prop == 'thr_action':

            num_thr = self.num_thr_action_gp


            if self.binary_actionness == 'true':
                actionness_bin = actionness.squeeze(1)

            else:
                actionness = F.softmax(actionness, dim=1)
                cls = torch.max(torch.mean(actionness, dim=2)[:,1:], dim=1)[1] + 1
                actionness_bin = actionness[range(len(cls)),cls,:]

            # threshold1 = torch.tensor([0, .05, .1, .15, .2, .25, .3, .35, .4, .45, .5, .55, .6, .65, .7, .75, .8, .85, .9, .95]).to(device=actionness_bin.device)
            threshold1 = torch.tensor(range(num_thr), device=actionness.device)/float(num_thr)
            threshold2 = (torch.max(actionness_bin, dim=1, keepdim=True)[0] - torch.min(actionness_bin, dim=1, keepdim=True)[0]) * threshold1 + torch.min(actionness_bin, dim=1, keepdim=True)[0]
            threshold = torch.cat((threshold1.unsqueeze(0).expand(threshold2.shape), threshold2), dim=1)

            # binary_action = self.MyBinarize(actionness.squeeze(1))
            binary_action = (actionness_bin.unsqueeze(2) >= threshold.unsqueeze(1)).to(torch.float32)

            diff = binary_action.new_zeros((binary_action.shape[0], binary_action.shape[1] + 1, binary_action.shape[2]), device=binary_action.device)
            diff[:, 1:-1, :] = binary_action[:, 1:, :] - binary_action[:, :-1, :]
            diff[:, 0, :] = binary_action[:, 0, :]
            diff[:, binary_action.shape[1], :] = 0 - binary_action[:, -1, :]
            start_idx = (diff == 1).nonzero().to(torch.long)
            ends_idx = (diff == -1).nonzero().to(torch.long)

            all_idx_dur_st = torch.zeros((B, self.tscale, self.tscale), device=start.device, requires_grad=False)

            str_end_list = start_idx.new_zeros((0, 3))
            for i in range(B):
                start_v = torch.unique(start_idx[start_idx[:, 0] == i][:,1])
                end_v = torch.unique(start_idx[ends_idx[:, 0] == i][:,1])
                str_end_v = torch.cat((start_v[:, None].repeat(1, end_v.shape[0]).unsqueeze(2), end_v[None, :].repeat(start_v.shape[0], 1).unsqueeze(2)), dim=2).view(-1, 2)
                str_end_v = str_end_v[str_end_v[:,0]<str_end_v[:,1]]
                str_end_list = torch.cat((str_end_list, torch.cat((torch.ones_like(str_end_v[:,0]).unsqueeze(1)*i, str_end_v), dim=1)), dim = 0)

            score_v = self._cal_IOC_ralign(str_end_list, actionness_bin)

            for i in range(B):
                scores = score_v[str_end_list[:,0]==i]
                str_end = str_end_list[str_end_list[:,0]==i, 1:]
                order = scores.argsort(descending=True)
                if self.total_num_prop_gp <= str_end.shape[0]:
                    str_end = str_end[order[:self.total_num_prop_gp]]
                else:
                    num_remain = self.total_num_prop_gp - str_end.shape[0]
                    str_end_mat = str_end.new_zeros((T, T))
                    str_end[:, 1] = str_end[:, 1]- 1
                    str_end_mat[tuple(str_end.transpose(0,1).to(torch.long))] = 1
                    temp_list = (str_end_mat==0).nonzero()
                    temp_list = temp_list[temp_list[:,0]<=temp_list[:, 1]]
                    rand_nums = torch.randperm(len(temp_list))
                    str_end = torch.cat((str_end, temp_list[rand_nums[:num_remain]]), dim=0)
                starts = str_end[:, 0]
                durations_minus1 = str_end[:, 1] - str_end[:, 0] #  should be str_end_v[:, 1] + 1 - str_end_v[:, 0]
                all_idx_dur_st[i, durations_minus1, starts] = 1

        return all_idx_dur_st