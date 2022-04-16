# -*- coding:utf-8 -*-
"""
Date     : 2022-03-10
Auther   : AbaKir
Email    : 425065513@qq.com
Software : PyCharm
Filename : train.py
"""
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from nets.fasterRCNN import FasterRCNN
from train.frcnn_training import FasterRCNNTrainer, weights_init
from train.call_backs import LossHistory
from train.data_loader import FRCNNDataset, frcnn_dataset_collate
from utils.utils import get_classes
from train.utils_fit import fit_one_epoch


def training(model_path = '',
             pretrained=True,
             Init_Epoch=0,
             Freeze_Epoch=10,
             Freeze_batch_size=10,
             Freeze_lr=1e-5,
             UnFreeze_Epoch=50,
             Unfreeze_batch_size=5,
             Unfreeze_lr=3e-6):
    Cuda = True
    classes_path = './model_data/thyroid.txt'
    input_shape = (256, 256)
    anchors_size = [4, 8, 16]
    Freeze_Train = True
    num_workers = 4
    train_annotation_path = './train/train.txt'
    val_annotation_path = './train/val.txt'
    class_names, num_classes = get_classes(classes_path)

    model = FasterRCNN(num_classes, anchor_scales=anchors_size, pretrained=pretrained)
    if not pretrained:
        weights_init(model)
    if model_path != '':
        print('Load weights {}.'.format(model_path))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    if Cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()
    else:
        model_train = model.train()

    loss_history = LossHistory("./train/logs/")

    # ---------------------------#
    #   读取数据集对应的txt
    # ---------------------------#
    with open(train_annotation_path) as f:
        train_lines = f.readlines()
    with open(val_annotation_path) as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    batch_size = Freeze_batch_size
    lr = Freeze_lr
    start_epoch = Init_Epoch
    end_epoch = Freeze_Epoch

    epoch_step = num_train // batch_size
    epoch_step_val = num_val // batch_size

    if epoch_step == 0 or epoch_step_val == 0:
        raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

    optimizer = optim.Adam(model_train.parameters(), lr, weight_decay=5e-4)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.96)

    train_dataset = FRCNNDataset(train_lines, input_shape, train=True)
    val_dataset = FRCNNDataset(val_lines, input_shape, train=False)
    gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                     drop_last=True, collate_fn=frcnn_dataset_collate)
    gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                         drop_last=True, collate_fn=frcnn_dataset_collate)

    # -------------------------------------------------------------------------------------------------------#
    # 冻结训练
    # -------------------------------------------------------------------------------------------------------#
    if Freeze_Train:
        for param in model.extractor.parameters():
            param.requires_grad = False

    model.freeze_bn()

    train_util = FasterRCNNTrainer(model, optimizer)

    for epoch in range(start_epoch, end_epoch):
        fit_one_epoch(model, train_util, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val,
                      end_epoch, Cuda)
        lr_scheduler.step()

    # -------------------------------------------------------------------------------------------------------#
    # 不冻结训练
    # -------------------------------------------------------------------------------------------------------#
    batch_size = Unfreeze_batch_size
    lr = Unfreeze_lr
    start_epoch = Freeze_Epoch
    end_epoch = UnFreeze_Epoch

    epoch_step = num_train // batch_size
    epoch_step_val = num_val // batch_size

    if epoch_step == 0 or epoch_step_val == 0:
        raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

    optimizer = optim.Adam(model_train.parameters(), lr, weight_decay=5e-4)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.96)

    train_dataset = FRCNNDataset(train_lines, input_shape, train=True)
    val_dataset = FRCNNDataset(val_lines, input_shape, train=False)
    gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                     drop_last=True, collate_fn=frcnn_dataset_collate)
    gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                         drop_last=True, collate_fn=frcnn_dataset_collate)

    if Freeze_Train:
        for param in model.extractor.parameters():
            param.requires_grad = True

    model.freeze_bn()

    train_util = FasterRCNNTrainer(model, optimizer)

    for epoch in range(start_epoch, end_epoch):
        fit_one_epoch(model, train_util, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val,
                      end_epoch, Cuda)
        lr_scheduler.step()
