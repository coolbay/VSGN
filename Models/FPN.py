import torch
import torch.nn as nn
from Utils.Sync_batchnorm.batchnorm import SynchronizedBatchNorm1d
from .GCNs import Graph_Layer



class FPN(nn.Module):
    def __init__(self, opt, sync_bn=False, freeze_bn=False):
        super(FPN, self).__init__()
        self.feat_dim = opt["feat_dim"]
        self.c_hidden = opt['bb_hidden_dim']  # 512
        self.batch_size = opt["batch_size"]
        self.tem_best_loss = 10000000
        self.num_levels = opt['num_levels'] # 5

        # self.conv0 = self._make_levels_enc(opt, in_channels=self.feat_dim, out_channels=self.c_hidden)

        self.conv0 = nn.Sequential(
            nn.Conv1d(in_channels=self.feat_dim, out_channels=self.c_hidden,kernel_size=3,stride=2,padding=1,groups=1),
            nn.ReLU(inplace=True),
        )

        self.levels_enc = nn.ModuleList()
        for i in range(self.num_levels):
            self.levels_enc.append(self._make_levels_enc(opt, in_channels=self.c_hidden, out_channels=self.c_hidden))

        self.levels_dec = nn.ModuleList()
        for i in range(self.num_levels - 1):
            output_padding = 1
            self.levels_dec.append(self._make_levels_dec(in_channels=self.c_hidden, out_channels=self.c_hidden, output_padding = output_padding))

        self.levels1 = nn.ModuleList()
        for i in range(self.num_levels):
            self.levels1.append(self._make_levels(in_channels=self.c_hidden, out_channels=self.c_hidden))

        self.levels2 = nn.ModuleList()
        for i in range(self.num_levels - 1):
            self.levels2.append(self._make_levels(in_channels=self.c_hidden, out_channels=self.c_hidden))


        self.freeze_bn = freeze_bn

    def _make_levels_enc(self, opt, in_channels, out_channels):
        return  Graph_Layer(opt, in_channels=in_channels, out_channels=out_channels)

    def _make_levels_dec(self, in_channels, out_channels, output_padding = 1):

        return nn.Sequential(
            nn.ConvTranspose1d(in_channels=in_channels, out_channels=out_channels,kernel_size=3,stride=2,padding=1, output_padding=output_padding, groups=1),
            nn.ReLU(inplace=True),
        )

    def _make_levels(self, in_channels, out_channels):

        return nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels,kernel_size=3,stride=1,padding=1,groups=1),
            nn.ReLU(inplace=True),
        )

    def _encoder(self, input, num_frms):

        feats = []
        x = self.conv0(input)
        for i in range(0, self.num_levels):
            x = self.levels_enc[i](x, num_frms)
            feats.append(x)

        return feats

    def _decoder(self, input):

        feats = []
        x = self.levels1[0](input[self.num_levels - 1])
        feats.append(x)
        for i in range(self.num_levels - 1):
            ii = self.num_levels - i - 2
            feat_enc = self.levels2[i](input[ii])
            feat_dec = self.levels_dec[i](x)
            x = self.levels1[i+1](feat_enc + feat_dec)
            feats.append(x)

        return feats

    def forward(self, input, num_frms):

        feats_enc = self._encoder(input, num_frms)
        feats_dec = self._decoder(feats_enc)

        return feats_enc, feats_dec

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm1d):
                m.eval()
            elif isinstance(m, nn.BatchNorm1d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv1d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv1d) or isinstance(m[1], SynchronizedBatchNorm1d) \
                            or isinstance(m[1], nn.BatchNorm1d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv1d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv1d) or isinstance(m[1], SynchronizedBatchNorm1d) \
                            or isinstance(m[1], nn.BatchNorm1d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
