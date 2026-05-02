from pathlib import Path
import argparse
import json
import math
import os
import random
import signal
import subprocess
import sys
import pandas as pd
import time
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, precision_recall_curve
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler
from torch import nn, optim
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from finetune_dataset import ECGDataset
from models.resnet1d import ResNet18, ResNet34, ResNet50, ResNet101
from models.vit1d import vit_small
import warnings
warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser(description='MERL Finetuning')
parser.add_argument('--dataset', default='ptbxl-super',
                    type=str, help='dataset name')
parser.add_argument('--ratio', default='100',
                    type=int, help='training data ratio')
parser.add_argument('--workers', default=12, type=int, metavar='N',
                    help='number of data loader workers')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=256, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--test-batch-size', default=256, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--learning-rate', default=0.3, type=float, metavar='LR',
                    help='base learning rate for weights')
parser.add_argument('--weight-decay', default=1e-4, type=float, metavar='W',
                    help='weight decay')
parser.add_argument('--pretrain_path', default='your_pretrained_encoder.pth', type=str,
                    help='path to pretrain weight directory')
parser.add_argument('--checkpoint-dir', default='./checkpoint_finetune/', type=Path,
                    metavar='DIR', help='path to checkpoint directory')
parser.add_argument('--data-root', default=Path('/public/home/hs_mmcd_5/project/jasonwei/datasets/finetune'), type=Path,
                    help='root path of raw datasets')
parser.add_argument('--ptbxl-root', default=None, type=Path,
                    help='ptbxl dataset root (contains records500)')
parser.add_argument('--icbeb-root', default=None, type=Path,
                    help='icbeb dataset root (contains .dat/.hea)')
parser.add_argument('--chapman-root', default=None, type=Path,
                    help='chapman dataset root (contains chapman/WFDBRecords)')
parser.add_argument('--backbone', default='resnet18', type=str, metavar='B',
                    help='backbone name')
parser.add_argument('--num_leads', default=12, type=int, metavar='B',
                    help='number of leads')
parser.add_argument('--name', default='linear', type=str, metavar='B',
                    help='exp name')

def main():
    args = parser.parse_args()
    args.ngpus_per_node = torch.cuda.device_count()
    batch_size = int(args.batch_size)
    test_batch_size = int(args.test_batch_size)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.empty_cache()
    device_id = torch.cuda.device_count()
    torch.manual_seed(42)
    random.seed(0)
    np.random.seed(0)
    torch.backends.cudnn.benchmark = True
    print(f'this task use {args.dataset} dataset')
    data_split_root = Path('data_split')
    data_root = args.data_root
    ptbxl_root = args.ptbxl_root if args.ptbxl_root is not None else data_root / 'ptbxl'
    icbeb_root = args.icbeb_root if args.icbeb_root is not None else data_root / 'icbeb'
    chapman_root = args.chapman_root if args.chapman_root is not None else data_root
    
    if 'ptbxl-super' in args.dataset:
        data_split_path = data_split_root / 'ptbxl' / 'super_class'
        
        train_csv_path = data_split_path / 'ptbxl_super_class_train.csv'
        val_csv_path = data_split_path / 'ptbxl_super_class_val.csv'
        test_csv_path = data_split_path / 'ptbxl_super_class_test.csv'
        
        train_dataset = ECGDataset(ptbxl_root, csv_file=train_csv_path, mode='train', dataset_name='ptbxl-super', ratio=args.ratio,
                                   )
        val_dataset = ECGDataset(ptbxl_root, csv_file=val_csv_path, mode='val', dataset_name='ptbxl-super',
                                   )
        test_dataset = ECGDataset(ptbxl_root, csv_file=test_csv_path, mode='test', dataset_name='ptbxl-super',
                                   )

        args.labels_name = train_dataset.labels_name
        num_classes = train_dataset.num_classes
    elif 'ptbxl-sub' in args.dataset:
        data_split_path = data_split_root / 'ptbxl' / 'sub_class'
        
        train_csv_path = data_split_path / 'ptbxl_sub_class_train.csv'
        val_csv_path = data_split_path / 'ptbxl_sub_class_val.csv'
        test_csv_path = data_split_path / 'ptbxl_sub_class_test.csv'
        
        train_dataset = ECGDataset(ptbxl_root, csv_file=train_csv_path, mode='train', dataset_name='ptbxl-sub', ratio=args.ratio,
                                   )
        val_dataset = ECGDataset(ptbxl_root, csv_file=val_csv_path, mode='val', dataset_name='ptbxl-sub',
                                   )
        test_dataset = ECGDataset(ptbxl_root, csv_file=test_csv_path, mode='test', dataset_name='ptbxl-sub',
                                   )

        args.labels_name = train_dataset.labels_name
        num_classes = train_dataset.num_classes

    elif 'ptbxl-form' in args.dataset:
        data_split_path = data_split_root / 'ptbxl' / 'form'
        
        train_csv_path = data_split_path / 'ptbxl_form_train.csv'
        val_csv_path = data_split_path / 'ptbxl_form_val.csv'
        test_csv_path = data_split_path / 'ptbxl_form_test.csv'
        
        train_dataset = ECGDataset(ptbxl_root, csv_file=train_csv_path, mode='train', dataset_name='ptbxl-form', ratio=args.ratio,
                                   )
        val_dataset = ECGDataset(ptbxl_root, csv_file=val_csv_path, mode='val', dataset_name='ptbxl-form',
                                   )
        test_dataset = ECGDataset(ptbxl_root, csv_file=test_csv_path, mode='test', dataset_name='ptbxl-form',
                                   )

        args.labels_name = train_dataset.labels_name
        num_classes = train_dataset.num_classes

    elif 'ptbxl-rhythm' in args.dataset:
        data_split_path = data_split_root / 'ptbxl' / 'rhythm'
        
        train_csv_path = data_split_path / 'ptbxl_rhythm_train.csv'
        val_csv_path = data_split_path / 'ptbxl_rhythm_val.csv'
        test_csv_path = data_split_path / 'ptbxl_rhythm_test.csv'
        
        train_dataset = ECGDataset(ptbxl_root, csv_file=train_csv_path, mode='train', dataset_name='ptbxl-rhythm', ratio=args.ratio,
                                   )
        val_dataset = ECGDataset(ptbxl_root, csv_file=val_csv_path, mode='val', dataset_name='ptbxl-rhythm',
                                   )
        test_dataset = ECGDataset(ptbxl_root, csv_file=test_csv_path, mode='test', dataset_name='ptbxl-rhythm',
                                   )

        args.labels_name = train_dataset.labels_name
        num_classes = train_dataset.num_classes

    elif args.dataset == 'icbeb':
        # set the path where you store the CPSC2018 dataset, the CPSC2018 dataset folder should be icbeb2018/records500/...
        data_split_path = data_split_root / 'icbeb'
        
        train_csv_path = data_split_path / 'icbeb_train.csv'
        val_csv_path = data_split_path / 'icbeb_val.csv'
        test_csv_path = data_split_path / 'icbeb_test.csv'
        
        train_dataset = ECGDataset(icbeb_root, csv_file=train_csv_path, mode='train', dataset_name='icbeb', ratio=args.ratio,
                                   )
        val_dataset = ECGDataset(icbeb_root, csv_file=val_csv_path, mode='val', dataset_name='icbeb',
                                   )
        test_dataset = ECGDataset(icbeb_root, csv_file=test_csv_path, mode='test', dataset_name='icbeb',
                                   )

        args.labels_name = train_dataset.labels_name
        num_classes = train_dataset.num_classes

    elif args.dataset == 'chapman':
        # set the path where you store the CSN dataset, the CSN dataset folder should be chapman/...
        data_split_path = data_split_root / 'chapman'
        
        train_csv_path = data_split_path / 'chapman_train.csv'
        val_csv_path = data_split_path / 'chapman_val.csv'
        test_csv_path = data_split_path / 'chapman_test.csv'
        
        train_dataset = ECGDataset(chapman_root, csv_file=train_csv_path, mode='train', dataset_name='chapman', ratio=args.ratio,
                                   )
        val_dataset = ECGDataset(chapman_root, csv_file=val_csv_path, mode='val', dataset_name='chapman',
                                   )
        test_dataset = ECGDataset(chapman_root, csv_file=test_csv_path, mode='test', dataset_name='chapman',
                                   )

        args.labels_name = train_dataset.labels_name
        num_classes = train_dataset.num_classes

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                  num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=test_batch_size, shuffle=False,
                                                num_workers=args.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False,
                                                    num_workers=args.workers, pin_memory=True)
    
    ckpt_path = args.pretrain_path
    ckpt = torch.load(ckpt_path, map_location='cpu')
    # 只使用 encoder 权重，丢弃分类头参数以避免 num_classes 不同导致的 shape mismatch
    if isinstance(ckpt, dict):
        # 如果是完整 checkpoint，尝试取出 state_dict
        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        elif 'model_state_dict' in ckpt:
            ckpt = ckpt['model_state_dict']
    # 删除 ResNet 最后一层分类头的参数（线性层），保留 backbone
    if isinstance(ckpt, dict):
        keys_to_delete = [k for k in ckpt.keys() if k.startswith('linear.')]
        for k in keys_to_delete:
            del ckpt[k]

    if 'resnet' in args.backbone:
        if args.backbone == 'resnet18':
            model = ResNet18(num_classes=num_classes)
        elif args.backbone == 'resnet50':
            model = ResNet50(num_classes=num_classes)
        elif args.backbone == 'resnet101':
            model = ResNet101(num_classes=num_classes)
        
        model.load_state_dict(ckpt, strict=False)
        print(f'load pretrained model from {args.pretrain_path}, the backbone is {args.backbone}, using {args.num_leads} leads')
        if 'linear' in args.name:
            for param in model.parameters():
                param.requires_grad = False
            print(f'freeze backbone for {args.name} with {args.backbone}')
            
        for param in model.linear.parameters():
            param.requires_grad = True

    if 'vit' in args.backbone:
        if args.backbone == 'vit_small':
            model = vit_small(num_classes=num_classes)
        
        model.load_state_dict(ckpt, strict=False)
        print(f'load pretrained model from {args.pretrain_path}, the backbone is {args.backbone}, using {args.num_leads} leads')
        if 'linear' in args.name:
            for param in model.parameters():
                param.requires_grad = False
            print(f'freeze backbone for {args.name} with {args.backbone}')

        model.reset_head(num_classes=num_classes)
        model.head.weight.requires_grad = True
        model.head.bias.requires_grad = True


    model = model.to('cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                       milestones=[40],
                       gamma=0.1,
                       last_epoch=-1)
    criterion = nn.BCEWithLogitsLoss()

    # automatically resume from checkpoint if it exists
    if (args.checkpoint_dir / (args.backbone+'-checkpoint-'+'B-'+str(batch_size)+args.dataset+'.pth')).is_file():
        ckpt = torch.load(args.checkpoint_dir / (args.backbone+'-checkpoint-'+'B-'+str(batch_size)+args.dataset+'R-'+str(args.ratio)+'.pth'),
                          map_location='cpu')
        start_epoch = ckpt['epoch']
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
    else:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        start_epoch = 0

    global_step = 0

    log = {
        'epoch': [],
        'val_acc': [],
        'val_f1': [],
        'val_precision': [],
        'val_recall': [],
        'val_auc': [],
        'test_acc': [],
        'test_f1': [],
        'test_precision': [],
        'test_recall': [],
        'test_auc': []
    }
    class_log = {
        'val_log': [],
        'test_log': []
    }
    
    scaler = GradScaler()
    for epoch in tqdm(range(start_epoch, args.epochs)):
        model.train()
        for step, (ecg, target) in enumerate(train_loader, start=epoch * len(train_loader)):
            optimizer.zero_grad()
            with autocast():
                output = model(ecg.to('cuda'))
                loss = criterion(output, target.to('cuda'))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        val_acc, val_f1, val_precision, val_recall, val_auc, val_metric_class = infer(model, val_loader, args)
        test_acc, test_f1, test_precision, test_recall, test_auc, test_metric_class = infer(model, test_loader, args)

        log['epoch'].append(epoch)
        log['val_acc'].append(val_acc)
        log['val_f1'].append(val_f1)
        log['val_precision'].append(val_precision)
        log['val_recall'].append(val_recall)
        log['val_auc'].append(val_auc)
        log['test_acc'].append(test_acc)
        log['test_f1'].append(test_f1)
        log['test_precision'].append(test_precision)
        log['test_recall'].append(test_recall)
        log['test_auc'].append(test_auc)

        class_log['val_log'].append(val_metric_class)
        class_log['test_log'].append(test_metric_class)

        scheduler.step()
    
    csv = pd.DataFrame(log)
    csv.columns = ['epoch', 'val_acc',
                    'val_f1', 'val_precision',
                      'val_recall', 'val_auc', 
                      'test_acc',
                        'test_f1', 'test_precision',
                          'test_recall', 'test_auc']
    
    val_class_csv = pd.concat(class_log['val_log'], axis=0)
    test_class_csv = pd.concat(class_log['test_log'], axis=0)
    val_class_csv.to_csv(f'{args.checkpoint_dir}/'+args.name+'-'+args.backbone+'-B-'+str(batch_size)+args.dataset+'R-'+str(args.ratio)+'-val-class.csv', index=False)
    test_class_csv.to_csv(f'{args.checkpoint_dir}/'+args.name+'-'+args.backbone+'-B-'+str(batch_size)+args.dataset+'R-'+str(args.ratio)+'-test-class.csv', index=False)

    csv.to_csv(f'{args.checkpoint_dir}/'+args.name+'-'+args.backbone+'-B-'+str(batch_size)+args.dataset+'R-'+str(args.ratio)+'.csv', index=False)

    # 训练完成后，将关键指标同时打印到终端并写入 txt 文件
    max_val_acc = max(log["val_acc"])
    max_val_f1 = max(log["val_f1"])
    max_val_precision = max(log["val_precision"])
    max_val_recall = max(log["val_recall"])
    max_val_auc = max(log["val_auc"])
    max_test_acc = max(log["test_acc"])
    max_test_f1 = max(log["test_f1"])
    max_test_precision = max(log["test_precision"])
    max_test_recall = max(log["test_recall"])
    max_test_auc = max(log["test_auc"])

    summary_str = (
        f'max val acc: {max_val_acc}\n'
        f'max val f1: {max_val_f1}\n'
        f'max val precision: {max_val_precision}\n'
        f'max val recall: {max_val_recall}\n'
        f'max val auc: {max_val_auc}\n'
        f'max test acc: {max_test_acc}\n'
        f'max test f1: {max_test_f1}\n'
        f'max test precision: {max_test_precision}\n'
        f'max test recall: {max_test_recall}\n'
        f'max test auc: {max_test_auc}\n'
    )

    print(summary_str)

    summary_path = args.checkpoint_dir / (args.name+'-'+args.backbone+'-B-'+str(batch_size)+args.dataset+'R-'+str(args.ratio)+'-summary.txt')
    with open(summary_path, 'w') as f:
        f.write(summary_str)
    # plot each metric in one subplot
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 3, 1)
    plt.plot(log['epoch'], log['val_acc'], label='val_acc')
    plt.plot(log['epoch'], log['test_acc'], label='test_acc')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(log['epoch'], log['val_f1'], label='val_f1')
    plt.plot(log['epoch'], log['test_f1'], label='test_f1')
    plt.legend()
    plt.subplot(2, 2, 3)
    # since we donot compute precision and recall in there. so this figure is not useful.
    # plt.plot(log['epoch'], log['val_precision'], label='val_precision')
    # plt.plot(log['epoch'], log['test_precision'], label='test_precision')
    # plt.plot(log['epoch'], log['val_ecall'], label='val_recall')
    # plt.plot(log['epoch'], log['test_recall'], label='test_recall')
    # plt.legend()
    plt.subplot(1, 3, 3)
    plt.plot(log['epoch'], log['val_auc'], label='val_auc')
    plt.plot(log['epoch'], log['test_auc'], label='test_auc')
    plt.legend()
    plt.savefig(f'{args.checkpoint_dir}/'+args.name+'-'+args.backbone+'-B-'+str(batch_size)+args.dataset+'R-'+str(args.ratio)+'.png')
    plt.close()

@torch.no_grad()
def infer(model, loader, args):
    # evaluate

    model.eval()
    
    y_pred = []

    y_true = []

    for step, (ecg, target) in tqdm(enumerate(loader)):

        input_label_list = target.to('cuda')

        predictions = model(ecg.to('cuda'))
        y_true.append(input_label_list.cpu().detach().numpy())

        for index, val in enumerate(predictions):
            y_pred.append(val.cpu().detach().numpy().reshape(1, -1))

    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    auc = roc_auc_score(y_true, y_pred, average='macro')
    
    max_f1s = []
    accs = []
    
    for i in range(y_pred.shape[1]):   
        gt_np = y_true[:, i]
        pred_np = y_pred[:, i]
        precision, recall, thresholds = precision_recall_curve(gt_np, pred_np)
        numerator = 2 * recall * precision
        denom = recall + precision
        f1_scores = np.divide(numerator, denom, out=np.zeros_like(denom), where=(denom!=0))
        max_f1 = np.max(f1_scores)
        max_f1_thresh = thresholds[np.argmax(f1_scores)]
        max_f1s.append(max_f1)
        accs.append(accuracy_score(gt_np, pred_np>max_f1_thresh))
    
    
    max_f1s = [i*100 for i in max_f1s]
    accs = [i*100 for i in accs]
    f1 = np.array(max_f1s).mean()    
    acc = np.array(accs).mean()

    # we donot compute precision and recall in there.
    precision, recall = 0, 0
    
    class_name = args.labels_name

    metric_dict = {element: [] for element in class_name}
    
    for i in range(len(list(metric_dict.keys()))):
        key = list(metric_dict.keys())[i]
        metric_dict[key].append(roc_auc_score(y_true[:, i], y_pred[:, i]))
    metric_class = pd.DataFrame(metric_dict)

    return acc, f1, precision, recall, auc, metric_class


if __name__ == '__main__':
    main()
