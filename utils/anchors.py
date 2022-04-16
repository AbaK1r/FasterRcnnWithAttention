# -*- coding:utf-8 -*-
"""
Date     : 2022-03-10
Auther   : AbaKir
Email    : 425065513@qq.com
Software : PyCharm
Filename : anchors.py
"""
import numpy as np


# --------------------------------------------#
#   生成基础的先验框
# --------------------------------------------#
def generate_anchor_base(base_size=16, ratios=(0.5, 1, 2), anchor_scales=(4, 8, 16)):
    anchor_base = np.zeros((len(ratios) * len(anchor_scales), 4), dtype=np.float32)
    for i in range(len(ratios)):
        for j in range(len(anchor_scales)):
            h = base_size * anchor_scales[j] * np.sqrt(ratios[i])
            w = base_size * anchor_scales[j] * np.sqrt(1. / ratios[i])

            index = i * len(anchor_scales) + j
            anchor_base[index, 0] = - h / 2.
            anchor_base[index, 1] = - w / 2.
            anchor_base[index, 2] = h / 2.
            anchor_base[index, 3] = w / 2.
    return anchor_base


# --------------------------------------------#
#   对基础先验框进行拓展对应到所有特征点上
# --------------------------------------------#
def enumerate_shifted_anchor(anchor_base, feat_stride, height, width):
    # ---------------------------------#
    #   计算网格中心点
    # ---------------------------------#
    shift_x = np.arange(0, width * feat_stride, feat_stride)
    shift_y = np.arange(0, height * feat_stride, feat_stride)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shift = np.stack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel(),), axis=1)

    # ---------------------------------#
    #   每个网格点上的9个先验框
    # ---------------------------------#
    A = anchor_base.shape[0]
    K = shift.shape[0]
    anchor = anchor_base.reshape((1, A, 4)) + shift.reshape((K, 1, 4))
    # ---------------------------------#
    #   所有的先验框
    # ---------------------------------#
    anchor = anchor.reshape((K * A, 4)).astype(np.float32)
    return anchor
