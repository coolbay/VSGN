
import torch
import torch.nn as nn
import math

class AnchorGenerator(nn.Module):
    def __init__(self, opt):
        super(AnchorGenerator, self).__init__()
        self.num_levels = opt['num_levels']
        self.tscale = opt["temporal_scale"]
        self.scale = 8
        self.base_stride = 4
        self.strides = []            # distance between anchors with respect to the sequence

        self.base_anchors = []
        for l in range(self.num_levels):
            stride = self.base_stride * pow(2, l)
            self.strides.append(stride)
            self.base_anchors.append( self.get_base_anchors(stride, self.scale))

        self.anchors = self.gen_anchors()

    def gen_anchors(self):
        feat_sizes = [ math.ceil(self.tscale / (self.scale * pow(2, l))) for l in range(self.num_levels)]
        anchors = []
        for size, stride, base_anchors in zip(feat_sizes, self.strides, self.base_anchors):
            shifts = torch.arange(0, self.tscale , step=stride, dtype=torch.float32)[:, None].repeat(1, 2)
            anchors.append((shifts.view(-1, 1, 2) + base_anchors.view(1, -1, 2)).reshape(-1, 2))

        return anchors

    def get_base_anchors(self, stride, scale):
        anchors = torch.tensor([1, stride], dtype=torch.float) - 0.5
        anchors = self._scale_enum(anchors, scale)
        return anchors

    def _scale_enum(self, anchor, scale):
        """Enumerate a set of anchors for each scale wrt an anchor."""
        length, center = self._whctrs(anchor)
        ws = length * scale
        anchors = self._mkanchors(ws, center)
        return anchors

    def _mkanchors(self, ws, ctr):

        # ws = ws[:, None]
        anchors = torch.stack(
            (
                ctr - 0.5 * (ws - 1),
                ctr + 0.5 * (ws - 1),
            )
        )
        return anchors

    def _whctrs(self, anchor):

        length = anchor[1] - anchor[0] + 1
        center = anchor[0] + 0.5 * (length - 1)
        return length, center

