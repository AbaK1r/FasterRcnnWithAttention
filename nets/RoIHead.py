# -*- coding:utf-8 -*-
"""
Date     : 2022-03-10
Auther   : AbaKir
Email    : 425065513@qq.com
Software : PyCharm
Filename : RoIHead.py
"""
import warnings
import torch
from torch import nn
from torchvision.ops import RoIAlign

warnings.filterwarnings("ignore")


class Resnet50RoIHead(nn.Module):
    def __init__(self, n_class, roi_size, classifier):
        super(Resnet50RoIHead, self).__init__()
        self.classifier = classifier
        # --------------------------------------#
        #   对ROIPooling后的的结果进行回归预测
        # --------------------------------------#
        self.cls_loc = nn.Linear(2048, n_class * 4)
        # -----------------------------------#
        #   对ROIPooling后的的结果进行分类
        # -----------------------------------#
        self.score = nn.Linear(2048, n_class)
        # -----------------------------------#
        #   权值初始化
        # -----------------------------------#
        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)

        self.roi = RoIAlign((roi_size, roi_size), spatial_scale=1.0, sampling_ratio=-1)

    def forward(self, x, rois, roi_indices, img_size):
        n = x.shape[0]
        if x.is_cuda:
            roi_indices = roi_indices.cuda()
            rois = rois.cuda()

        rois_feature_map = torch.zeros_like(rois)
        rois_feature_map[:, [0, 2]] = rois[:, [0, 2]] / img_size[1] * x.size()[3]
        rois_feature_map[:, [1, 3]] = rois[:, [1, 3]] / img_size[0] * x.size()[2]

        indices_and_rois = torch.cat([roi_indices[:, None], rois_feature_map], dim=1)
        # -----------------------------------#
        #   利用建议框对公用特征层进行截取
        # -----------------------------------#
        pool = self.roi(x, indices_and_rois)  # (600, 1024, 14, 14)
        # -----------------------------------#
        #   利用classifier网络进行特征提取
        # -----------------------------------#
        fc7 = self.classifier(pool)  # (600, 2048, 1, 1)
        # --------------------------------------------------------------#
        #   当输入为一张图片的时候，这里获得的f7的shape为[300, 2048]
        # --------------------------------------------------------------#
        fc7 = fc7.view(fc7.size(0), -1)

        roi_cls_locs = self.cls_loc(fc7)
        roi_scores = self.score(fc7)
        roi_cls_locs = roi_cls_locs.view(n, -1, roi_cls_locs.size(1))  # (2, 215, 8)
        roi_scores = roi_scores.view(n, -1, roi_scores.size(1))  # (2, 215, 2)
        return roi_cls_locs, roi_scores


def normal_init(m, mean, stddev, truncated=False):
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()
