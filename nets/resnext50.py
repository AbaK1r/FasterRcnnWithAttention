# -*- coding:utf-8 -*-
"""
Date     : 2022-03-26
Auther   : AbaKir
Email    : 425065513@qq.com
Software : PyCharm
Filename : resnext50.py
"""
from torchvision.models import resnext50_32x4d
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url
from nets.attention import se_block, cbam_block, eca_block

attention_blocks = [se_block, cbam_block, eca_block]


def resnext50(pretrained=False, phi=3):
    model = resnext50_32x4d(pretrained=False)
    if pretrained:
        state_dict = load_state_dict_from_url('https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
                                              model_dir="../model_data")
        model.load_state_dict(state_dict)

    if 1 <= phi <= 3:
        attention1 = attention_blocks[phi - 1](1024)
        attention2 = attention_blocks[phi - 1](2048)
    else:
        attention1 = attention2 = lambda x: x
    # ----------------------------------------------------------------------------#
    #   获取特征提取部分，从conv1到model.layer3，最终获得一个38,38,1024的特征层
    # ----------------------------------------------------------------------------#
    features = list([model.conv1, model.bn1, model.relu, model.maxpool,
                     model.layer1, model.layer2, model.layer3, attention1])
    # ----------------------------------------------------------------------------#
    #   获取分类部分，从model.layer4到model.avgpool
    # ----------------------------------------------------------------------------#
    classifier = list([model.layer4, model.avgpool, attention2])

    features = nn.Sequential(*features)  # [-1, 3, 600, 600] --> [-1, 1024, 38, 38]
    classifier = nn.Sequential(*classifier)  # [-1, 1024, 38, 38] --> [-1, 2048, 2, 2]
    return features, classifier
