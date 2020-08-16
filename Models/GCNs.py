import math
import torch
import torch.nn as nn
import torch.nn.functional as F



def segment_tiou_mat(target_segments, test_segments):
    target_num = target_segments.shape[1]
    test_num = test_segments.shape[1]
    target_ext = target_segments.unsqueeze(2).repeat(1, 1, test_num, 1)
    test_ext = test_segments.unsqueeze(1).repeat(1, target_num, 1, 1)
    maxl = torch.max(target_ext[:,:,:,0], test_ext[:,:,:,0])
    minr = torch.min(target_ext[:,:,:,1], test_ext[:,:,:,1])
    intersection = (minr - maxl).clamp(0)
    union = ((test_ext[:, :, :, 1] - test_ext[:, :, :, 0]) +
             (target_ext[:, :, :, 1] - target_ext[:, :, :, 0]) -
             intersection)
    tiou = intersection / union
    # tiou[:, range(target_num), range(test_num)] = 0
    # intersection[:, range(target_num), range(test_num)] = 0

    # tiou[tiou==1] = 0
    # intersection[intersection==1] = 0

    return tiou, intersection

def segment_dist_mat(target_segments, test_segments):
    target_num = target_segments.shape[1]
    test_num = test_segments.shape[1]

    target_ctr = (target_segments[:,:,1] - target_segments[:,:,0])/2 + target_segments[:,:,0]
    test_ctr = (test_segments[:,:,1] - test_segments[:,:,0])/2 + test_segments[:,:,0]
    target_ctr_ext = target_ctr.unsqueeze(2).repeat(1, 1, test_num)
    test_ctr_ext = test_ctr.unsqueeze(1).repeat(1, target_num, 1)

    distance = torch.abs(test_ctr_ext - target_ctr_ext)

    target_ext = target_segments.unsqueeze(2).repeat(1, 1, test_num, 1)
    test_ext = test_segments.unsqueeze(1).repeat(1, target_num, 1, 1)

    minl = torch.min(target_ext[:,:,:,0], test_ext[:,:,:,0])
    maxr = torch.max(target_ext[:,:,:,1], test_ext[:,:,:,1])
    union = (maxr - minl).clamp(0)

    dist_mat = distance / union
    dist_mat[(dist_mat==0) & (target_ext[:,:,:,0]==test_ext[:,:,:,0]) & (target_ext[:,:,:,1]==test_ext[:,:,:,1])] = 1

    return dist_mat


# dynamic graph from knn
def knn(x, y=None, k=10):

    if y is None:
        y = x

    dif = torch.sum((x.unsqueeze(2) - y.unsqueeze(1))** 2, dim=-1)
    idx = dif.topk(k=k, dim=-1, largest=False)[1]

    return idx

def get_neigh_idx_semantic(x, n_neigh):

    B, num_prop_v = x.shape[:2]
    neigh_idx = knn(x, k=n_neigh).to(dtype=torch.float32)
    shift = (torch.tensor(range(B), dtype=torch.float32, device=x.device) * num_prop_v)[:, None, None].repeat(1, num_prop_v, n_neigh)
    neigh_idx = (neigh_idx + shift).view(-1)
    return neigh_idx


class NeighConv(nn.Module):
    def __init__(self, in_features, out_features, opt, n_neigh=0, edge_weight=None, nfeat_mode=None, agg_type=None, bias=True):
        super(NeighConv, self).__init__()
        self.num_neigh = opt['num_neigh'] if n_neigh == 0 else n_neigh
        self.nfeat_mode = opt['nfeat_mode'] if nfeat_mode == None else nfeat_mode
        self.agg_type = opt['agg_mode'] if agg_type == None else agg_type
        self.edge_weight = opt['edge_weight'] if edge_weight== None else edge_weight

        self.split_gcn = opt['split_gcn']
        self.num_neigh_split = opt['num_neigh_split']
        self.split_temp_edge = opt['split_temp_edge']
        self.mlp = nn.Linear(in_features*2, out_features)

    def forward(self, feat_prop, neigh_idx):
        if self.split_gcn == 'false':
            num_neigh = self.num_neigh
        elif self.split_gcn == 'true' and self.split_temp_edge == 'in_graph':
            num_neigh = self.num_neigh_split + 2
        elif self.split_gcn == 'true':
            num_neigh = self.num_neigh_split

        feat_neigh = feat_prop[neigh_idx.to(torch.long)]
        f_neigh_temp = feat_neigh.view(-1, num_neigh, feat_neigh.shape[-1])
        weight = torch.matmul(f_neigh_temp, feat_prop.unsqueeze(2))
        weight_denom1 = torch.sqrt(torch.sum(f_neigh_temp*f_neigh_temp, dim=2, keepdim=True))
        weight_denom2 = torch.sqrt(torch.sum(feat_prop.unsqueeze(2)*feat_prop.unsqueeze(2), dim=1, keepdim=True))
        weight = (weight / torch.matmul(weight_denom1, weight_denom2)).squeeze(2)
        if self.nfeat_mode == 'feat_ctr':

            feat_neigh = torch.cat((feat_neigh.view(-1, num_neigh, feat_prop.size(-1)), feat_prop.view(-1, 1, feat_prop.size(-1)).repeat(1, num_neigh,1)), dim=-1)
        elif self.nfeat_mode == 'dif_ctr':
            feat_prop = feat_prop.view(-1, 1, feat_prop.size(-1)).repeat(1, num_neigh,1)
            diff = feat_neigh.view(-1, num_neigh, feat_prop.size(-1)) - feat_prop
            feat_neigh = torch.cat((diff, feat_prop), dim=-1)
        elif self.nfeat_mode == 'feat':
            feat_neigh = feat_neigh.view(-1, num_neigh, feat_prop.size(-1))

        feat_neigh_out = self.mlp(feat_neigh)
        if self.edge_weight == 'true':
            feat_neigh_out = feat_neigh_out * weight.unsqueeze(2)
        if self.agg_type == 'max':
            feat_neigh_out = feat_neigh_out.max(dim=1, keepdim=False)[0]
        elif self.agg_type == 'mean':
            feat_neigh_out = feat_neigh_out.mean(dim=1, keepdim=False)
        return feat_neigh_out

class GraphNet_seq(nn.Module):
    def __init__(self, in_feat, out_feat, opt, bias=True):
        super(GraphNet_seq, self).__init__()
        self.n_neigh = opt['n_neigh_seq'] # 4
        self.edge_weight = opt['edge_weight_seq']  #'false'
        self.nfeat_mode = opt['nfeat_mode_seq'] #'feat_ctr'
        self.agg_type = opt['agg_type_seq']  #'max'
        self.nconv1 = NeighConv(in_feat, out_feat, opt, self.n_neigh, self.edge_weight, self.nfeat_mode, self.agg_type)

    def forward(self, x):
        B, num_frm, C = x.shape
        neigh_idx = get_neigh_idx_semantic(x, self.n_neigh)
        x = self.nconv1(x.view(-1, C), neigh_idx)
        x = F.relu(x)
        return x.view(B, num_frm, x.shape[-1])

class GraphNet_prop(nn.Module):
    def __init__(self, hidden_dim_3d, hidden_dim_2d, opt, bias=True):
        super(GraphNet_prop, self).__init__()
        self.tscale = opt['prop_temporal_scale']
        self.num_neigh = opt['num_neigh']
        self.edge_type = opt['edge_type']
        self.num_prop_v = sum(opt['num_samp_prop'])
        self.samp_gcn = opt['samp_gcn']
        self.is_train = opt['is_train']
        self.split_gcn = opt['split_gcn']
        self.splits = opt['splits']
        self.num_neigh_split = opt['num_neigh_split']
        self.split_temp_edge = opt['split_temp_edge']

        if 'pgcn' in self.edge_type:
            self.neigh_idx_pcn_sage = self._get_neigh_idx_pgcn_sage()

        self.nconv1 = NeighConv(hidden_dim_3d, hidden_dim_2d, opt)
        self.nconv2 = NeighConv(hidden_dim_2d, hidden_dim_2d, opt)
        self.nconv3 = NeighConv(hidden_dim_2d, hidden_dim_2d, opt)

        if self.split_gcn == 'true' and self.split_temp_edge == 'after_graph':
            self.tempconv1 = nn.Conv1d(hidden_dim_2d, hidden_dim_2d, kernel_size=3, padding=1, padding_mode='replicate')
            self.tempconv2 = nn.Conv1d(hidden_dim_2d, hidden_dim_2d, kernel_size=3, padding=1, padding_mode='replicate')

        if self.split_gcn == 'true':
            self.merge = nn.Conv1d(hidden_dim_2d, hidden_dim_2d, kernel_size=self.splits)

    def _split_feat_coord(self, x, anchor_st_ed):
        total_num_prop, C, splits = x.shape
        x = x.permute(0, 2, 1).contiguous().view(-1, C)

        anchor_st_ed_new = anchor_st_ed.new_zeros((total_num_prop, splits, 3))

        interval = (anchor_st_ed[:,2] - anchor_st_ed[:,1]) / splits

        for i in range(splits):
            anchor_st_ed_new[:, i, 0] = anchor_st_ed[:, 0]
            anchor_st_ed_new[:, i, 1] = torch.min(anchor_st_ed[:,1] + interval * i, anchor_st_ed[:,2]-1./splits)
            anchor_st_ed_new[:, i, 2] = torch.min(anchor_st_ed[:,1] + interval * (i+1), anchor_st_ed[:,2])
            anchor_st_ed_new[anchor_st_ed_new[:,i,2] <= anchor_st_ed_new[:,i,1],i,2] = anchor_st_ed_new[anchor_st_ed_new[:,i,2] <= anchor_st_ed_new[:,i,1],i,1] + 1./splits

        # anchor_st_ed_new = anchor_st_ed_new.view(-1,3)

        return x, anchor_st_ed_new

    def _record_duplicate_prop(self, B, num_prop_v, adj_num, anchor_st_ed):
        anchor_st_ed = anchor_st_ed.view(B*num_prop_v, -1, 3)
        dup = (anchor_st_ed[:, :, None, 1] == anchor_st_ed[:, None, :, 1]) * (anchor_st_ed[:, :, None, 2] == anchor_st_ed[:, None, :, 2])
        dup_ext = dup[:, :, None, :, None].repeat(1, 1, self.splits, 1, self.splits).view(B*num_prop_v, adj_num*self.splits, adj_num*self.splits)

        return dup_ext

    def _get_neigh_idx_sage_split(self, anchor_st_ed, anchor_st_ed_new):
        B = int(anchor_st_ed_new[-1, -1, 0] + 1)
        adj_num = self.num_neigh * (self.num_neigh + 1) + 1
        num_prop_v = int(len(anchor_st_ed_new) / (B*adj_num))
        anchor_st_ed_new = anchor_st_ed_new.view(B*num_prop_v, adj_num*self.splits, 3)
        num_neigh = self.num_neigh_split + 2 if self.split_temp_edge == 'in_graph' else self.num_neigh_split

        if self.split_temp_edge == 'in_graph':
            # splits * splits
            mask = torch.stack((torch.tensor(range(-1, self.splits-1)), torch.tensor(range(1, self.splits+1))), dim=1).to(dtype=torch.long, device=anchor_st_ed.device)
            mask[mask<0]=0
            mask[mask>=self.splits] = self.splits - 1

            # adj_num * adj_num
            mask = mask.unsqueeze(0).repeat(adj_num, 1, 1)
            shift_in = (torch.tensor(range(adj_num), dtype=torch.long, device=anchor_st_ed_new.device) * self.splits)[:, None, None]
            mask = mask + shift_in

        # record duplicate proposals
        dup_mat = self._record_duplicate_prop(B, num_prop_v, adj_num, anchor_st_ed)

        if self.edge_type == 'pgcn_iou':
            iou_array, _ = segment_tiou_mat(anchor_st_ed_new[:, :, 1:], anchor_st_ed_new[:, :, 1:])
            iou_array[dup_mat==1] = 0
            neigh_idx = (torch.topk(iou_array, self.num_neigh_split, dim=2)[1])
        elif self.edge_type == 'pgcn_dist':
            iou_array = segment_dist_mat(anchor_st_ed_new[:, :, 1:], anchor_st_ed_new[:, :, 1:])
            neigh_idx = (torch.topk(iou_array, self.num_neigh, dim=2, largest=False)[1])

        if self.split_temp_edge == 'in_graph':
            neigh_idx = torch.cat((neigh_idx, mask.view(-1, 2).unsqueeze(0).repeat(B*num_prop_v, 1, 1)), dim=-1)

        shift = (torch.tensor(range(B*num_prop_v), dtype=torch.long, device=neigh_idx.device)*adj_num*self.splits)[:, None, None].repeat(1, adj_num*self.splits, num_neigh)
        neigh_idx = (neigh_idx + shift).view(-1)

        return neigh_idx



    def _get_neigh_idx_infer_split(self, anchor_st_ed_new):

        B = int(anchor_st_ed_new[-1, -1, 0] + 1)
        anchor_st_ed_new = anchor_st_ed_new.view(B, -1, self.splits, 3)
        num_prop_v = anchor_st_ed_new.shape[1]
        # anchor_st_ed = anchor_st_ed.view(B, num_prop_v*splits, 3)

        # splits * splits
        mask = torch.stack((torch.tensor(range(-1, self.splits-1)), torch.tensor(range(1, self.splits+1))), dim=1).to(dtype=torch.long, device=anchor_st_ed_new.device)
        mask[mask<0]=0
        mask[mask>=self.splits] = self.splits - 1

        # adj_num * adj_num
        mask = mask.unsqueeze(0).repeat(num_prop_v, 1, 1)
        shift_in = (torch.tensor(range(num_prop_v), dtype=torch.long, device=anchor_st_ed_new.device) * self.splits)[:, None, None]
        mask = mask + shift_in

        neigh_idx = anchor_st_ed_new.new_zeros((B, num_prop_v, 0, self.num_neigh_split))
        for i in range(self.splits):
            if self.edge_type == 'pgcn_iou':
                iou_array, _ = segment_tiou_mat(anchor_st_ed_new[:, :, i, 1:], anchor_st_ed_new.view(B, num_prop_v*self.splits, 3)[:, :, 1:])
                for i in range(self.splits):
                    iou_array[:, range(num_prop_v), range(i, iou_array.shape[-1], self.splits)] = 0
                neigh_idx = torch.cat((neigh_idx, torch.topk(iou_array, self.num_neigh_split, dim=2)[1].unsqueeze(2).to(torch.float32)), dim=2)
            elif self.edge_type == 'pgcn_dist':
                iou_array = segment_dist_mat(anchor_st_ed_new[:, :, i, 1:], anchor_st_ed_new.view(B, num_prop_v*self.splits, 3)[:, :, i, 1:])
                for i in range(self.splits):
                    iou_array[:, range(num_prop_v), range(i, iou_array.shape[-1], self.splits)] = 0
                neigh_idx = torch.cat((neigh_idx, torch.topk(iou_array, self.num_neigh_split, dim=2)[1].unsqueeze(2).to(torch.float32)), dim=2)

        neigh_idx = torch.cat((neigh_idx, mask.unsqueeze(0).repeat(B, 1, 1, 1).to(torch.float32)), dim=-1)
        shift = (torch.tensor(range(B), dtype=torch.float32, device=neigh_idx.device)*num_prop_v*self.splits)[:, None, None, None].repeat(1, num_prop_v, self.splits, self.num_neigh_split + 2)
        neigh_idx = (neigh_idx[None, :, :, :] + shift).view(-1)

        return neigh_idx


    def forward(self, x, anchor_st_ed):
        B = int(anchor_st_ed[-1, 0] + 1)

        if self.edge_type == 'grid':
            neigh_idx = self._get_neigh_idx_grid(anchor_st_ed)
        elif 'pgcn' in self.edge_type:
            if self.is_train == 'true' and self.samp_gcn == 'sage':
                if self.split_gcn == 'true':
                    x, anchor_st_ed_new = self._split_feat_coord(x, anchor_st_ed)
                    neigh_idx = self._get_neigh_idx_sage_split(anchor_st_ed, anchor_st_ed_new)
                else:
                    neigh_idx = self.neigh_idx_pcn_sage
                    adj_num = self.num_neigh * (self.num_neigh + 1) + 1
                    shift = (torch.tensor(range(B), dtype=torch.float32)*self.num_prop_v*adj_num)[:, None, None, None].repeat(1, self.num_prop_v, adj_num, self.num_neigh)
                    neigh_idx = (neigh_idx[None, :, :, :] + shift).view(-1)
            else:
                if self.split_gcn == 'true':
                    x, anchor_st_ed_new = self._split_feat_coord(x, anchor_st_ed)
                    neigh_idx = self._get_neigh_idx_infer_split(anchor_st_ed_new)
                else:
                    neigh_idx = self._get_neigh_idx_pgcn(anchor_st_ed)

        elif 'semantic' in self.edge_type:
            neigh_idx = get_neigh_idx_semantic(x.view(B, -1, x.shape[-1]))

        x = self.nconv1(x, neigh_idx)
        x = F.relu(x)
        if self.split_gcn == 'true' and self.split_temp_edge == 'after_graph':
            x_channel = x.shape[-1]
            x = self.tempconv1(x.view(-1, self.splits, x_channel).permute(0, 2, 1))
            x = x.permute(0, 2, 1).reshape(-1, x_channel)
            x = F.relu(x)


        if 'semantic' in self.edge_type:
            neigh_idx = get_neigh_idx_semantic(x.view(B, -1, x.shape[-1]))
        x = self.nconv2(x, neigh_idx)
        x = F.relu(x)
        if self.split_gcn == 'true' and self.split_temp_edge == 'after_graph':
            x_channel = x.shape[-1]
            x = self.tempconv1(x.view(-1, self.splits, x_channel).permute(0, 2, 1))
            x = x.permute(0, 2, 1).reshape(-1, x_channel)
            x = F.relu(x)

        if 'semantic' in self.edge_type:
            neigh_idx = get_neigh_idx_semantic(x.view(B, -1, x.shape[-1]))
        x = self.nconv3(x, neigh_idx)
        x = F.relu(x)

        if self.split_gcn == 'true':
            x = x.view(-1, self.splits, x.shape[-1])
            x = self.merge(x.permute(0, 2, 1)).squeeze(-1)
        return x

    def _get_neigh_idx_grid(self, anchor_st_ed):
        kernel = []
        ksize = int((math.sqrt(self.num_neigh + 1)-1)/2)
        B = int(anchor_st_ed[-1, 0] + 1)
        anchor_st_ed[:, 2] = anchor_st_ed[:, 2] - 1

        for i in range(-ksize*2, ksize*2 + 1, 2):
            for j in range(-ksize*2, ksize*2 + 1, 2):
                if i==0 and j==0:
                    continue
                kernel.append([0, i, j])
        kernel = torch.tensor(kernel, dtype=torch.float32, device=anchor_st_ed.device)
        # start/end indices of all the neighbors of all proposals
        neigh_start_end = anchor_st_ed.unsqueeze(1) + kernel.unsqueeze(0)
        neigh_start_end = neigh_start_end.view(-1, 3)
        neigh_start_end[neigh_start_end<0] = 0
        neigh_start_end[neigh_start_end>=self.tscale] = self.tscale - 1

        # An intermediate 2d matrix of representing all proposals before selection and sampling. It is the proposal indices where the proposal is selected, and is -1 where the the proposal is not selected.
        idx_mat = anchor_st_ed.new_zeros(B, self.tscale, self.tscale)
        idx_mat[:,:,:] = -1
        idx_mat[tuple(anchor_st_ed.to(torch.long).transpose(0,1))] = torch.tensor(range(len(anchor_st_ed)), dtype=torch.float32, device=anchor_st_ed.device)

        # Get proposal indices for all the neighbors. IF some neighbors don't belong to the selected proposals (idx is -1), they will be made the center of the neighborhood --- the proposal itself
        neigh_idx = idx_mat[tuple(neigh_start_end.to(torch.long).transpose(0,1))]
        neigh_idx[neigh_idx<0] = (neigh_idx<0).nonzero().view(-1).to(torch.float32) / self.num_neigh # If some neighbors are not among the selected proposals, just use the proposal itself

        anchor_st_ed[:, 2] = anchor_st_ed[:, 2] + 1

        return neigh_idx

    def _get_neigh_idx_pgcn(self, anchor_st_ed):
        B = int(anchor_st_ed[-1, 0] + 1)

        num_prop_v = int(len(anchor_st_ed) / B)
        anchor_st_ed = anchor_st_ed.view(B, num_prop_v, 3)

        if self.edge_type == 'pgcn_iou':
            iou_array, _ = segment_tiou_mat(anchor_st_ed[:, :, 1:], anchor_st_ed[:, :, 1:])
            iou_array[iou_array==1] = 0
            neigh_idx = (torch.topk(iou_array, self.num_neigh, dim=2)[1]).to(dtype=torch.float32)
        elif self.edge_type == 'pgcn_dist':
            iou_array = segment_dist_mat(anchor_st_ed[:, :, 1:], anchor_st_ed[:, :, 1:])
            neigh_idx = (torch.topk(iou_array, self.num_neigh, dim=2, largest=False)[1]).to(dtype=torch.float32)

        shift = (torch.tensor(range(B), dtype=torch.float32, device=anchor_st_ed.device) * num_prop_v)[:, None, None].repeat(1, num_prop_v, self.num_neigh)
        neigh_idx = (neigh_idx + shift).view(-1)

        return neigh_idx

    def _get_neigh_idx_pgcn_sage(self):
        adj_num = self.num_neigh * (self.num_neigh + 1) + 1

        mask = torch.zeros((adj_num, self.num_neigh))
        mask[range(self.num_neigh + 1)] = torch.tensor(list(range(1, adj_num)), dtype=torch.float32).view(self.num_neigh + 1, -1)
        mask[range(self.num_neigh + 1, adj_num), :] = torch.tensor(range(self.num_neigh + 1, adj_num), dtype=torch.float32).unsqueeze(1)
        mask = mask.unsqueeze(0).repeat(self.num_prop_v, 1, 1)
        shift = (torch.tensor(range(self.num_prop_v), dtype=torch.float32)*adj_num)[:, None, None].repeat(1, adj_num, self.num_neigh)
        neigh_idx = mask + shift

        return neigh_idx
