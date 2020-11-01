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
def knn(x, num_frms, opt, y=None, k=10):
    bs, _, length = x.shape
    ratio = opt['temporal_scale'] / length
    half1_k = int(k / 2)
    half2_k = k - half1_k
    if y is None:
        y = x

    # Original neighbors
    dif = torch.sum((x.unsqueeze(2) - y.unsqueeze(3))** 2, dim=1)
    idx_org = dif.topk(k=k, dim=-1, largest=False)[1]

    idx_new = idx_org.clone()

    max_dif = torch.max(dif)

    for i in range(bs):
        if num_frms[i] <= (opt['short_ratio'] * opt['temporal_scale']):
            thr = (num_frms[i] + opt['stitch_gap']) / ratio
            dif[i, thr:, thr:] = max_dif + 1

            loc1 = torch.tensor(range(length), dtype=torch.long, device=x.device)[:, None].repeat(1, half1_k).view(-1)
            loc2 = idx_org[i, :, :half1_k].reshape(-1)
            dif[i, loc1, loc2] = max_dif + 1

            idx_new[i, :, half1_k:] = dif[i].topk(k=half2_k, dim=-1, largest=False)[1]

    return idx_new


def get_neigh_idx_semantic(x, n_neigh, num_frms, opt):

    B, _, num_prop_v = x.shape
    neigh_idx = knn(x, num_frms, opt, k=n_neigh).to(dtype=torch.float32)
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

        num_neigh = self.num_neigh

        feat_neigh = feat_prop[neigh_idx.to(torch.long)]
        f_neigh_temp = feat_neigh.view(-1, num_neigh, feat_neigh.shape[-1])

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

            weight = torch.matmul(f_neigh_temp, feat_prop.unsqueeze(2))
            weight_denom1 = torch.sqrt(torch.sum(f_neigh_temp*f_neigh_temp, dim=2, keepdim=True))
            weight_denom2 = torch.sqrt(torch.sum(feat_prop.unsqueeze(2)*feat_prop.unsqueeze(2), dim=1, keepdim=True))
            weight = (weight / torch.matmul(weight_denom1, weight_denom2)).squeeze(2)
            feat_neigh_out = feat_neigh_out * weight.unsqueeze(2)

        if self.agg_type == 'max':
            feat_neigh_out = feat_neigh_out.max(dim=1, keepdim=False)[0]
        elif self.agg_type == 'mean':
            feat_neigh_out = feat_neigh_out.mean(dim=1, keepdim=False)
        return feat_neigh_out

class Graph_Layer(nn.Module):
    def __init__(self,  opt, in_channels, out_channels, bias=True):
        super(Graph_Layer, self).__init__()
        self.n_neigh = opt['n_neigh_seq'] # 4
        self.edge_weight = opt['edge_weight_seq']  #'false'
        self.nfeat_mode = opt['nfeat_mode_seq'] #'feat_ctr'
        self.agg_type = opt['agg_type_seq']  #'max'
        self.gcn_insert = opt['gcn_insert']
        self.opt = opt
        self.tconv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, groups=1)
        self.nconv1 = NeighConv(in_channels, out_channels, opt, self.n_neigh, self.edge_weight, self.nfeat_mode, self.agg_type)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x, num_frms):
        bs, C, num_frm = x.shape

        # CNN
        c_out = self.tconv1(x)

        # GCN
        if self.gcn_insert == 'par':

            neigh_idx = get_neigh_idx_semantic(x, self.n_neigh, num_frms, self.opt)
            g_out = self.nconv1(x.permute(0, 2, 1).reshape(-1, C), neigh_idx)


        elif self.gcn_insert == 'seq':

            neigh_idx = get_neigh_idx_semantic(c_out, self.n_neigh, num_frms, self.opt)
            g_out = self.nconv1(c_out.permute(0, 2, 1).reshape(-1, C), neigh_idx)

        g_out = g_out.view(bs, num_frm, -1).permute(0, 2, 1)
        out = c_out + g_out

        out = self.relu(out)
        out = self.maxpool(out)

        return out
