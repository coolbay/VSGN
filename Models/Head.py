import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    def __init__(self, opt):
        super(Head, self).__init__()

        self.num_convs_head = 1
        in_channels = opt['bb_hidden_dim']
        num_anchors = 1
        num_classes = 1 if opt['dataset'] == 'activitynet' else opt['decoder_num_classes']

        cls_tower = []
        bbox_tower = []
        conv_func = nn.Conv1d
        for i in range(self.num_convs_head):

            cls_tower.append(
                conv_func(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True
                )
            )
            cls_tower.append(nn.GroupNorm(32, in_channels))
            cls_tower.append(nn.ReLU())

            bbox_tower.append(
                conv_func(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True
                )
            )
            bbox_tower.append(nn.GroupNorm(32, in_channels))
            bbox_tower.append(nn.ReLU())

        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('bbox_tower', nn.Sequential(*bbox_tower))
        self.cls_logits = nn.Conv1d(
            in_channels, num_anchors * num_classes, kernel_size=3, stride=1,
            padding=1
        )
        self.bbox_pred = nn.Conv1d(
            in_channels, num_anchors * 2, kernel_size=3, stride=1,
            padding=1
        )

    def forward(self, x):
        logits = []
        bbox_reg = []
        for l, feature in enumerate(x):
            cls_tower = self.cls_tower(feature)
            box_tower = self.bbox_tower(feature)

            logits.append(self.cls_logits(cls_tower))

            bbox_pred = self.bbox_pred(box_tower)
            # if self.cfg.MODEL.ATSS.REGRESSION_TYPE == 'POINT':
            #     bbox_pred = F.relu(bbox_pred)
            bbox_reg.append(bbox_pred)

        return logits, bbox_reg
