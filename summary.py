# -*- coding:utf-8 -*-
"""
Date     : 2022-03-26
Auther   : AbaKir
Email    : 425065513@qq.com
Software : PyCharm
Filename : summary.py
"""
from nets.fasterRCNN import FasterRCNN
import torch

if __name__ == '__main__':
    model = FasterRCNN(1, "predict").to('cuda')
    out = FasterRCNN(torch.randn([1, 3, 256, 256]).to('cuda'))
