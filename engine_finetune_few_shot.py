import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Iterable, Optional
from timm.data import Mixup
from timm.utils import accuracy
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, average_precision_score,
    hamming_loss, jaccard_score, recall_score, precision_score, cohen_kappa_score, precision_recall_curve, auc, matthews_corrcoef
)
from pycm import ConfusionMatrix
import utils.misc as misc
import utils.lr_sched as lr_sched

import codecs
from sklearn import metrics
from sklearn.utils import resample
from sklearn.preprocessing import label_binarize
import numpy as np
from collections import Counter

def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0,
    mixup_fn: Optional[Mixup] = None,
    log_writer=None,
    args=None
):
    """Train the model for one epoch."""
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    print_freq, accum_iter = 20, args.accum_iter
    optimizer.zero_grad()
    
    if log_writer:
        print(f'log_dir: {log_writer.log_dir}')
    
    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, f'Epoch: [{epoch}]')):
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        
        samples, targets = samples.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        if mixup_fn:
            samples, targets = mixup_fn(samples, targets)
        
        with torch.cuda.amp.autocast():
            outputs = model(samples)
            targets_onehot = F.one_hot(targets, num_classes=2).float()
            # print(outputs.shape)
            # print(targets.shape)
            # return
            loss = criterion(outputs, targets_onehot)
        loss_value = loss.item()
        loss /= accum_iter
        
        loss_scaler(loss, optimizer, clip_grad=max_norm, parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()
        
        torch.cuda.synchronize()
        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss/train', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)
    
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, args, epoch, mode, num_class, log_writer):
    """Evaluate the model."""
    criterion = nn.CrossEntropyLoss()
    metric_logger = misc.MetricLogger(delimiter="  ")
    # os.makedirs(os.path.join(args.output_dir, args.task), exist_ok=True)
    
    model.eval()
    preds, labels, scores = [], [], []
    
    for batch in metric_logger.log_every(data_loader, 10, f'{mode}:'):
        images, target = batch[0].to(device, non_blocking=True), batch[-1].to(device, non_blocking=True)
        target_onehot = F.one_hot(target.to(torch.int64), num_classes=num_class)
        
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)
        # output_ = nn.Softmax(dim=1)(output)
        # output_ = torch.softmax(output, dim=1)
        output_ = torch.sigmoid(output)
        
        metric_logger.update(loss=loss.item())
        scores.append(output_)
        labels.append(target)
    
    true_labels = torch.cat(labels, dim=0).cpu().numpy()
    scores = torch.cat(scores, dim=0).cpu().numpy()

    if num_class > 2:
        # 将真实标签进行one-hot编码
        true_labels_onehot = label_binarize(true_labels, classes=np.arange(num_class))
    else:
        true_labels_onehot = np.eye(num_class)[true_labels]
    
    aupr_scores = []
    if num_class > 2:
        roc_auc = roc_auc_score(true_labels_onehot, scores, average='macro', multi_class='ovr')
        # roc_auc = roc_auc_score(true_labels_onehot, scores, average='macro')
        for i in range(num_class):
            precision, recall, _ = precision_recall_curve(true_labels_onehot[:, i], scores[:, i])
            aupr = auc(recall, precision)
            aupr_scores.append(aupr)
        # print(aupr_scores)
        macro_aupr = np.mean(list(aupr_scores))
    else:
        #2分类roc，只计算疾病类，默认index为0
        roc_auc = roc_auc_score(true_labels_onehot[:, 0], scores[:, 0])
        precision, recall, _ = precision_recall_curve(true_labels_onehot[:, 0], scores[:, 0])
        macro_aupr = auc(recall, precision)

        pred = (scores[:, 0] >= 0.5).astype(int)
        acc = (pred == true_labels_onehot[:, 0]).astype(float).mean().item()

    score = roc_auc

    print(f'ROC AUC: {roc_auc:.4f}, AUPR: {macro_aupr:.4f}, ACC:{acc:.4f}, Score: {score:.4f}')
    
    metric_logger.synchronize_between_processes()
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, score, roc_auc, macro_aupr, acc
