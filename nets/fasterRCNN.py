# -*- coding:utf-8 -*-
"""
Date     : 2022-03-10
Auther   : AbaKir
Email    : 425065513@qq.com
Software : PyCharm
Filename : fasterRCNN.py
"""
import torch.nn as nn
from nets.RoIHead import Resnet50RoIHead
from nets.resnext50 import resnext50
from nets.rpn import RegionProposalNetwork


class FasterRCNN(nn.Module):
    def __init__(self, num_classes,  # 分类种类
                 mode="training",  # 训练模式和预测模式
                 feat_stride=16,
                 anchor_scales=(4, 8, 16),
                 ratios=(0.5, 1, 2),
                 pretrained=False,
                 phi=3):
        super(FasterRCNN, self).__init__()
        self.feat_stride = feat_stride
        self.extractor, classifier = resnext50(pretrained, phi)  # 全卷积的特征提取层 全连接层
        self.rpn = RegionProposalNetwork(
            1024, 512,
            ratios=ratios,
            anchor_scales=anchor_scales,
            feat_stride=self.feat_stride,
            mode=mode)
        self.head = Resnet50RoIHead(
            n_class=num_classes + 1,
            roi_size=14,
            classifier=classifier)

    def forward(self, x, scale=1.):
        # ---------------------------------#
        #   计算输入图片的大小
        # ---------------------------------#
        img_size = x.shape[2:]
        # ---------------------------------#
        #   利用主干网络提取特征
        # ---------------------------------#
        base_feature = self.extractor.forward(x)  # (2, 1024, 16, 16)
        # ---------------------------------#
        #   获得建议框
        # ---------------------------------#
        _, _, rois, roi_indices, _ = self.rpn.forward(base_feature, img_size, scale)
        # ---------------------------------------#
        #   获得classifier的分类结果和回归结果
        # ---------------------------------------#
        roi_cls_locs, roi_scores = self.head.forward(base_feature, rois, roi_indices, img_size)
        return roi_cls_locs, roi_scores, rois, roi_indices

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


def normal_init(m, mean, stddev, truncated=False):
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()
