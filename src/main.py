#!/usr/bin/env python3
#
# Nathan Tsoi © 2020

import time
import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

import sklearn.metrics as skmetrics

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from ignite.engine import Engine
from ignite.metrics import Accuracy, Precision, Recall, MetricsLambda
from ignite.contrib.metrics import ROC_AUC

from ap_perf import PerformanceMetric, MetricLayer
from ap_perf.metric import CM_Value

from btg import mean_fbeta_approx_loss_on, mean_accuracy_approx_loss_on, mean_auroc_approx_loss_on
from confusion import sig, linear_approx, fbeta, kl, bce

import copy
import logging
import io

from constants import *

def convert_labels_to_binary(classes_of_interest,labels):
    binary_labels = []
    for label in labels:
        if label in classes_of_interest:
            binary_labels.append(1)
        else:
            binary_labels.append(0)
    return binary_labels

class Dataset(torch.utils.data.Dataset):
    def __init__(self, ds_split):
        self.X = torch.from_numpy(ds_split['X']).float()
        self.y = torch.from_numpy(ds_split['y']).float()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X[index, :], self.y[index]

def wmw_loss(gamma=0.2, p=3):
    def loss(pt, gt):
        """ ROC AUC Score.
        Approximates the Area Under Curve score, using approximation based on
        the Wilcoxon-Mann-Whitney U statistic.
        Yan, L., Dodier, R., Mozer, M. C., & Wolniewicz, R. (2003).
        Optimizing Classifier Performance via an Approximation to the Wilcoxon-Mann-Whitney Statistic.
        Measures overall performance for a full range of threshold levels.
        Arguments:
            gt: `Tensor` . Targets (labels), a probability distribution.
            pt: `Tensor`. Predicted values.
        """
        pos = torch.masked_select(pt, gt>0)
        neg = torch.masked_select(pt, gt>0)
        pos = torch.unsqueeze(pos, 0)
        neg = torch.unsqueeze(neg, 1)
        difference = torch.zeros_like(pos * neg) + pos - neg - gamma
        masked = torch.masked_select(difference, difference < 0.0)
        return torch.sum(torch.pow(-masked, p))
    return loss


def dice_loss(p=2, eps=1e-7):
    def loss(gt, pt):
        """Sørensen–Dice loss
        References:
        - https://github.com/kevinzakka/pytorch-goodies/blob/master/losses.py#L54
        - https://github.com/hubutui/DiceLoss-PyTorch/blob/master/loss.py#L28
        """
        pt = pt.contiguous().view(pt.shape[0], -1)
        gt = gt.contiguous().view(gt.shape[0], -1)
        num = torch.sum(torch.mul(pt, gt), dim=1)
        den = torch.sum(pt.pow(p) + gt.pow(p), dim=1) + eps
        return 1 - (num / den).mean()
    return loss


class Net(nn.Module):
    def __init__(self, input_dim, sigmoid_out):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
        self.sigmoid_out = sigmoid_out

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        if self.sigmoid_out:
            x = self.sigmoid(x)
        return x.squeeze()

# metric definition
class Fbeta(PerformanceMetric):
    def __init__(self, beta):
        self.beta = beta

    def define(self, C):
        return ((1 + self.beta ** 2) * C.tp) / ((self.beta ** 2) * C.ap + C.pp)

# metric definition
class AccuracyMetric(PerformanceMetric):
    def define(self, C):
        return (C.tp + C.tn) / C.all

def threshold_pred(y_pred, t):
    return (y_pred > t).float()

# creating trainer,evaluator
def thresholded_output_transform(threshold, device):
    def transform(output):
        y_pred, y = output
        return threshold_pred(y_pred, t=torch.tensor([threshold]).to(device)), y
    return transform

def inference_engine(model, device, threshold=0.5):
    def inference_update_with_tta(engine, batch):
        model.eval()
        with torch.no_grad():
            x, y = batch
            y_pred = threshold_pred(model(x.to(device)).to(device), t=torch.tensor([threshold]).to(device))
            return y_pred, y
    engine = Engine(inference_update_with_tta)
    for name, metric in getmetrics(threshold, device).items():
        metric.attach(engine, name)
    return engine

def inference_over_range(model, device, data_loader, thresholds=np.arange(0.1,1,0.1)):
    res = []
    for threshold in thresholds:
        inferencer = inference_engine(model, device, threshold=threshold)
        res.append(inferencer.run(data_loader))
    return res

def getmetrics(threshold, device):
    precision = Precision(thresholded_output_transform(threshold, device), average=False)
    recall = Recall(thresholded_output_transform(threshold, device), average=False)
    def Fbetaf(r, p, beta):
        return torch.mean((1 + beta ** 2) * p * r / (beta ** 2 * p + r + 1e-20)).item()
    return {
        'precision': precision,
        'recall': recall,
        'f1': MetricsLambda(Fbetaf, recall, precision, 1),
        'f2': MetricsLambda(Fbetaf, recall, precision, 2),
        'f3': MetricsLambda(Fbetaf, recall, precision, 3),
        'auroc': ROC_AUC(thresholded_output_transform(threshold, device)),
        'accuracy': Accuracy(thresholded_output_transform(threshold, device)),
    #    'cm': ConfusionMatrix(num_classes=1)
    }

## combine 2 confusion matrices
def add_cm_val(cmv1, cmv2):
    res = CM_Value(np.array([]),np.array([]))
    res.all = cmv1.all + cmv2.all
    res.tp = cmv1.tp + cmv2.tp
    res.ap = cmv1.ap + cmv2.ap
    res.pp = cmv1.pp + cmv2.pp

    res.an = cmv1.an + cmv2.an
    res.pn = cmv1.pn + cmv2.pn

    res.fp = cmv1.fp + cmv2.fp
    res.fn = cmv1.fn + cmv2.fn
    res.tn = cmv1.tn + cmv2.tn

    return res


# compute metric value from cunfusion matrix
def compute_metric_from_cm(metric, C_val):
    # check for special cases
    if metric.special_case_positive:
        if C_val.ap == 0 and C_val.pp == 0:
            return 1.0
        elif C_val.ap == 0:
            return 0.0
        elif C_val.pp == 0:
            return 0.0

    if metric.special_case_negative:
        if C_val.an == 0 and C_val.pn == 0:
            return 1.0
        elif C_val.an == 0:
            return 0.0
        elif C_val.pn == 0:
            return 0.0

    val = metric.metric_expr.compute_value(C_val)
    return val

def plot_histograms(device, writer, p, q, epoch, name, split):
    p = p.to(device)
    q = q.to(device)
    p[torch.isnan(p)] = 0
    p[torch.isinf(p)] = 0
    q[torch.isnan(q)] = 0
    q[torch.isinf(q)] = 0
    combined = torch.hstack((p, q))
    _kl = kl(p, q)
    #writer.add_histogram(f"{split}/kl_{name}", _kl, epoch)
    writer.add_histogram(f"{split}/{name}", combined, epoch)
    writer.add_scalar(f"{split}/kl_{name}_mean", _kl.mean(), epoch)
    writer.add_scalar(f"{split}/kl_{name}_std", _kl.std(), epoch)

def train(args):
    if torch.cuda.is_available():
        print("device = cuda")
        device = f"cuda:{args.gpu}"
    else:
        print("device = cpu")
        device = "cpu"

    # ap-perf only works on cpu
    if args.loss == 'ap-perf-f1':
        print("device = cpu")
        device = 'cpu'

    dataparams = {'batch_size': args.batch_size,
                'shuffle': True,
                'num_workers': 1}

    def oversampler(ds_split):
        y = torch.tensor(ds_split['y'])
        class_count = torch.bincount(y.squeeze())
        class_weighting = 1. / class_count
        sample_weight = torch.tensor([class_weighting[t] for t in y])
        sampler = WeightedRandomSampler(sample_weight, len(y))
        return sampler

    if args.dataset.startswith('cifar'):
        import torchvision
        import torchvision.transforms as transforms
        transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transform)
        valset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transform)
        old_classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        train_size = len(trainset)
        indices = np.arange(train_size)
        np.random.shuffle(indices)
        split = int(train_size*.8)
        train_indices = indices[:split]
        val_indices = indices[split:]

        valset.data = trainset.data[val_indices]
        valset.targets = list(np.asarray(trainset.targets)[val_indices])

        trainset.data = trainset.data[train_indices]
        trainset.targets = list(np.asarray(trainset.targets)[train_indices])

        #import pdb; pdb.set_trace(); 1
        train_targets_original = trainset.targets.copy()
        val_targets_original = valset.targets.copy()
        test_targets_original = testset.targets.copy()

        if args.dataset == 'cifar-t':
            coi = [0,1,8,9]
        elif args.dataset == 'cifar-f':
            coi = [6]

        train_targets_binary = convert_labels_to_binary(coi,train_targets_original)
        val_targets_binary = convert_labels_to_binary(coi,val_targets_original)
        test_targets_binary = convert_labels_to_binary(coi,test_targets_original)

        trainset.targets = torch.tensor(train_targets_binary, dtype=torch.float)
        valset.targets = torch.tensor(val_targets_binary, dtype=torch.float)
        testset.targets = torch.tensor(test_targets_binary, dtype=torch.float)

        train_loader = DataLoader(trainset, **dataparams)
        val_loader = DataLoader(valset, **dataparams)
        test_loader = DataLoader(testset, **dataparams)
    else:
        ds = dataset_from_name(args.dataset)()

        validationset = Dataset(ds['val'])
        testset = Dataset(ds['test'])
        trainset = Dataset(ds['train'])

        val_loader = DataLoader(validationset, **dataparams)
        test_loader = DataLoader(testset, **dataparams)
        if args.oversample:
            del dataparams['shuffle']
            dataparams['sampler'] = oversampler(ds['train'])
        train_loader = DataLoader(trainset, **dataparams)

    # initialize metric
    f1_score = Fbeta(1)
    f1_score.initialize()
    f1_score.enforce_special_case_positive()

    # accuracy metric
    accm = AccuracyMetric()
    accm.initialize()

    threshold = 0.5

    # create a model and criterion layer
    sigmoid_out = False
    if args.loss.startswith('sig-') or args.loss.startswith('approx-'):
        sigmoid_out = True

    if args.model == 'ff':
        input_dim = ds['train']['X'][0].shape[0]
        model = Net(input_dim, sigmoid_out).to(device)
    elif args.model == 'img':
        from models_image import TinyDarknet
        model = TinyDarknet(sigmoid_out).to(device).float()
    else: 
        raise RuntimeError(f'Unknown model {args.model}')

    # set run timestamp or load from args
    now = int(time.time())
    if args.initial_weights:
        now = args.initial_weights

    Path(model_path(args)).mkdir(parents=True, exist_ok=True)
    run_name = f"{args.dataset}-{args.loss}-batch_{args.batch_size}-lr_{args.lr}_{now}"
    log_path = '/'.join([model_path(args), f"{run_name}.log"])
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fm = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh.setFormatter(fm)
    ch.setFormatter(fm)
    logger.addHandler(ch)
    logger.addHandler(fh)
    logging.info(f"Configured logging to output to: {log_path} and terminal")

    initial_model_file_path = '/'.join([model_path(args), '{}_initial.pth'.format(now)])

    # load or save initial weights
    if args.initial_weights:
        logging.info(f"[{now}] loading {initial_model_file_path}")
        model.load_state_dict(torch.load(initial_model_file_path))
    else:
        # persist the initial weights for future use
        torch.save(model.state_dict(), initial_model_file_path)

    class_weight = None
    if args.weight_loss:
        class_weight = DS_WEIGHTS[args.dataset]

    thresholds = torch.range(start=0.1, end=0.9, step=0.1).to(device)
    criterion_has_cm = False
    if args.loss == 'bce':
        criterion = nn.BCEWithLogitsLoss()
        if args.weight_loss:
            # $$$ Assumes only 1 class
            pos_weight = torch.tensor(DS_WEIGHTS[args.dataset][1] / DS_WEIGHTS[args.dataset][0])
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif args.loss in ['dice', 'sig-dice']:
        criterion = dice_loss()
    elif args.loss == 'mse':
        criterion = nn.MSELoss()
    elif args.loss == 'ap-perf-f1':
        criterion = MetricLayer(f1_score).to(device)
    elif args.loss in ['approx-f1', 'approx-f2', 'approx-f3'] :
        beta = int(args.loss.split('-')[-1][-1])
        criterion = mean_fbeta_approx_loss_on(device, thresholds=thresholds, beta=beta, class_weight=class_weight)
        criterion_has_cm = True
    elif args.loss == 'approx-accuracy':
        criterion = mean_accuracy_approx_loss_on(device, thresholds=thresholds, class_weight=class_weight)
        criterion_has_cm = True
    elif args.loss in ['approx-f1-sig', 'approx-f2-sig', 'approx-f3-sig'] :
        beta = int(args.loss.split('-')[1][-1])
        criterion = mean_fbeta_approx_loss_on(device, thresholds=thresholds, beta=beta, approx=sig(k=10), class_weight=class_weight)
        criterion_has_cm = True
    elif args.loss == 'approx-accuracy-sig':
        criterion = mean_accuracy_approx_loss_on(device, thresholds=thresholds, approx=sig(k=10), class_weight=class_weight)
        criterion_has_cm = True
    elif args.loss == 'approx-auroc':
        criterion = mean_auroc_approx_loss_on(device, thresholds=thresholds, class_weight=class_weight)
        criterion_has_cm = True
    elif args.loss == 'approx-auroc-sig':
        criterion = mean_auroc_approx_loss_on(device, thresholds=thresholds, approx=sig(k=10), class_weight=class_weight)
        criterion_has_cm = True
    elif args.loss == 'wmw':
        criterion = wmw_loss()
    else:
        raise RuntimeError("Unknown loss {}".format(args.loss))

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    patience = args.early_stopping_patience


    tensorboard_path = '/'.join([args.output, 'running', args.experiment, 'tensorboard', run_name])
    writer = SummaryWriter(tensorboard_path)

    # early stopping
    early_stopping = False
    best_val_loss = None

    best_test = {
        'now': now,
        'loss': args.loss,
        'accuracy_05_score':0,
        'f1_05_score':0,
        'ap_05_score':0,
        'auroc_05_score':0,
        'accuracy_mean_score': 0,
        'f1_mean_score': 0,
        'ap_mean_score':0,
        'auroc_mean_score':0,
    }


    # for early stopping
    model_file_path = None
    best_model_in_mem = None

    for epoch in range(args.epochs):
        if early_stopping:
            logging.info("[{}] Early Stopping at Epoch {}/{}".format(now, epoch, args.epochs))
            break

        data_cm = CM_Value(np.array([]),np.array([]))

        losses = []
        rocs = []
        tps = []
        fns = []
        fps = []
        tns = []

        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            #if labels.sum() < 1:
            #    print("no positive examples")
            #    continue

            model.train()
            optimizer.zero_grad()
            output = model(inputs)
            if criterion_has_cm:
                loss, tp, fn, fp, tn = criterion(output, labels)
                tps.append(tp)
                fns.append(fn)
                fps.append(fp)
                tns.append(tn)
            else:
                loss = criterion(output, labels)

            losses.append(loss)

            if loss.requires_grad:
                loss.backward()
                optimizer.step()

            ## check prediction
            model.eval()    # switch to evaluation
            y_pred = model(inputs)
            y_pred_thresh = (y_pred >= threshold).float()
            np_pred = y_pred_thresh.cpu().numpy()
            np_labels = labels.cpu().numpy()
            batch_cm = CM_Value(np_pred, np_labels)
            data_cm = add_cm_val(data_cm, batch_cm)
            # undefined if predicting only 1 value
            # https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/metrics/_ranking.py#L224
            if len(np.unique(np_labels)) == 2:
                rocs.append(skmetrics.roc_auc_score(np_labels,np_pred))

        acc_val = compute_metric_from_cm(accm, data_cm)
        f1_val = compute_metric_from_cm(f1_score, data_cm)

        mloss = torch.stack(losses).mean()
        writer.add_scalar('loss', mloss, epoch)
        if criterion_has_cm:
            #import pdb; pdb.set_trace(); 1
            writer.add_scalar('train/tps', torch.stack(tps).sum(), epoch)
            writer.add_scalar('train/fns', torch.stack(fns).sum(), epoch)
            writer.add_scalar('train/fps', torch.stack(fps).sum(), epoch)
            writer.add_scalar('train/tns', torch.stack(tns).sum(), epoch)
        writer.add_scalar('train/accuracy', np.array(acc_val).mean(), epoch)
        writer.add_scalar('train/f1', np.array(f1_val).mean(), epoch)
        writer.add_scalar('train/auroc', np.array(rocs).mean(), epoch)

        logging.info("Train - Epoch ({}): Loss: {:.4f} Accuracy: {:.4f} | F1: {:.4f}".format(epoch, mloss, acc_val, f1_val))


        ### Validation
        model.eval()
        with torch.no_grad():
            data_cm = CM_Value(np.array([]),np.array([]))
            val_losses = []
            val_tps = []
            val_fns = []
            val_fps = []
            val_tns = []
            for i, (inputs, labels) in enumerate(val_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                output = model(inputs)

                if criterion_has_cm:
                    loss, tp, fn, fp, tn = criterion(output, labels)
                    val_tps.append(tp)
                    val_fns.append(fn)
                    val_fps.append(fp)
                    val_tns.append(tn)
                else:
                    loss = criterion(output, labels)
                val_losses.append(loss)

                ## prediction
                pred = (output >= 0).float()
                batch_cm = CM_Value(pred.cpu().numpy(), labels.cpu().numpy())
                data_cm = add_cm_val(data_cm, batch_cm)

            acc_val = compute_metric_from_cm(accm, data_cm)
            f1_val_apperf = compute_metric_from_cm(f1_score, data_cm)
            writer.add_scalar('val_ap_perf/accuracy', acc_val, epoch)
            writer.add_scalar('val_ap_perf/f1', f1_val_apperf, epoch)

            logging.info("Val - Epoch ({}): Accuracy: {:.4f} | F1: {:.4f}".format(epoch, acc_val, f1_val))

            # double check val metrics
            inferencer = inference_engine(model, device, threshold=threshold)
            result_state = inferencer.run(val_loader)
            mean_val_loss = torch.stack(val_losses).mean()
            #logging.info("  val losses: {}".format(val_losses))
            writer.add_scalar('val/loss', mean_val_loss, epoch)
            if criterion_has_cm:
                writer.add_scalar('val/tps', torch.stack(val_tps).sum(), epoch)
                writer.add_scalar('val/fns', torch.stack(val_fns).sum(), epoch)
                writer.add_scalar('val/fps', torch.stack(val_fps).sum(), epoch)
                writer.add_scalar('val/tns', torch.stack(val_tns).sum(), epoch)
            writer.add_scalar('val/accuracy', result_state.metrics['accuracy'], epoch)
            writer.add_scalar('val/f1', result_state.metrics['f1'], epoch)
            writer.add_scalar('val/auroc', result_state.metrics['auroc'], epoch)
            f1_val_ignite = result_state.metrics['f1']


            # check early stopping per epoch
            patience -= 1
            if best_val_loss is None or best_val_loss > mean_val_loss:
                # save the best model
                model_file_path = '/'.join([model_path(args), '{}_best_model_{}_{}_{}={}.pth'.format(now, epoch, args.dataset, args.loss, mean_val_loss)])
                best_model_in_mem = io.BytesIO()
                torch.save(model, best_model_in_mem)
                logging.info("best model (in mem) is {}".format(model_file_path))
                best_val_loss = mean_val_loss
                patience = args.early_stopping_patience

                # compare distributions on the test set
                p_i = []
                q_i = []
                for i, (inputs, labels) in enumerate(test_loader):
                    p_i.append(labels)
                    q_i.append(model(inputs.to(device)))
                p_i = torch.hstack(p_i).to(device)
                q_i = torch.hstack(q_i).to(device)
                test_bces = bce(p_i, q_i)
                test_f1s = fbeta(p_i, q_i, thresholds).nansum()
                test_f1ls = fbeta(p_i, q_i, thresholds, approx=linear_approx()).nansum()
                test_f1ss = fbeta(p_i, q_i, thresholds, approx=sig(k=10)).nansum()
                #import pdb; pdb.set_trace();1
                # compare f1 (p) to bce (q)
                plot_histograms(device, writer, test_f1s, test_bces, epoch, 'f1_vs_bce', 'test')
                # compare f1 (p) to linear f1 approximation (q)
                plot_histograms(device, writer, test_f1s, test_f1ls, epoch, 'f1_vs_f1-linear', 'test')
                # compare f1 (p) to sigmoid f1 approximation (q)
                plot_histograms(device, writer, test_f1s, test_f1ss, epoch, 'f1_vs_f1-sigmoid', 'test')

                # check test set results
                results = inference_over_range(model, device, test_loader)
                # values correspond to the thresholds: np.arange(0.1,1,0.1), so index 4 has t=0.5
                accuracies = [r.metrics['accuracy'] for r in results]
                test_precisions = [r.metrics['precision'] for r in results]
                test_recalls = [r.metrics['recall'] for r in results]
                test_f1s = [r.metrics['f1'] for r in results]
                test_f2s = [r.metrics['f2'] for r in results]
                test_f3s = [r.metrics['f3'] for r in results]
                aurocs = [r.metrics['auroc'] for r in results]
                # record the best to print at the end
                if best_test['accuracy_05_score'] < accuracies[4]:
                    best_test['accuracy_05_score'] = accuracies[4]
                    best_test['accuracy_05_model_file'] = model_file_path
                if best_test['f1_05_score'] < test_f1s[4]:
                    best_test['f1_05_score'] = test_f1s[4]
                    best_test['f1_05_model_file'] = model_file_path
                if best_test['auroc_05_score'] < aurocs[4]:
                    best_test['auroc_05_score'] = aurocs[4]
                    best_test['auroc_05_model_file'] = model_file_path
                mean_accuracy = np.mean(accuracies)
                mean_precision = np.mean(test_precisions)
                mean_recall = np.mean(test_recalls)
                mean_f1 = np.mean(test_f1s)
                mean_f2 = np.mean(test_f2s)
                mean_f3 = np.mean(test_f3s)
                mean_auroc = np.mean(aurocs)
                if best_test['accuracy_mean_score'] < mean_accuracy:
                    best_test['accuracy_mean_score'] = mean_accuracy
                    best_test['accuracy_mean_model_file'] = model_file_path
                if best_test['f1_mean_score'] < mean_f1:
                    best_test['f1_mean_score'] = mean_f1
                    best_test['f1_mean_model_file'] = model_file_path
                if best_test['auroc_mean_score'] < mean_auroc:
                    best_test['auroc_mean_score'] = mean_auroc
                    best_test['auroc_mean_model_file'] = model_file_path
                # write to tensorboard
                writer.add_scalar('test/accuracy_05', accuracies[4], epoch)
                writer.add_scalar('test/f1_05', test_f1s[4], epoch)
                writer.add_scalar('test/auroc_05', aurocs[4], epoch)
                writer.add_scalar('test/accuracy_mean', mean_accuracy, epoch)
                writer.add_scalar('test/precision_mean', mean_precision, epoch)
                writer.add_scalar('test/recall_mean', mean_recall, epoch)
                writer.add_scalar('test/f1_mean', mean_f1, epoch)
                writer.add_scalar('test/f2_mean', mean_f2, epoch)
                writer.add_scalar('test/f3_mean', mean_f3, epoch)
                writer.add_scalar('test/auroc_mean', mean_auroc, epoch)
            logging.info(f"[{now}] {args.loss}, {args.dataset}, patience: {patience}")
            if patience <= 0:
                early_stopping = True

    logging.info(f"{args.experiment} {now}")
    # persist the best in-memory model
    with open(model_file_path, 'wb') as f:
        f.write(best_model_in_mem.getbuffer())
    logging.info("{}: {}".format(best_test, model_file_path))
    pd.DataFrame({k: [v] for k, v in best_test.items()}).to_csv('/'.join([model_path(args), f"{run_name}.csv"]))
    return now


def test(args):
    now = int(time.time())
    if args.initial_weights:
        now = args.initial_weights

    if torch.cuda.is_available():
      device = f"cuda:{args.gpu}"
    else:
      device = "cpu"

    threshold = 0.5
    if args.loss == 'ap-perf-f1':
        threshold = 0.0
    ds = dataset_from_name(args.dataset)()
    dataparams = {'batch_size': args.batch_size,
                  'shuffle': True,
                  'num_workers': 1}

    trainset = Dataset(ds['train'])
    train_loader = DataLoader(trainset, **dataparams)
    validationset = Dataset(ds['val'])
    val_loader = DataLoader(validationset, **dataparams)
    testset = Dataset(ds['test'])
    test_loader = DataLoader(testset, **dataparams)

    input_dim = ds['train']['X'][0].shape[0]

    best_model_path = None
    maxval = None
    globpath = '/'.join([args.output, 'running', args.experiment, dataset_name, args.loss, "{}_best_model_*_{}*.pth".format(now, args.loss)])
    for path in glob.iglob(globpath):
        val = float(path.split('=')[-1].split('.pth')[0])
        print(val)
        if maxval is None or val > maxval:
            maxval = val
            best_model_path = path
    print("loading: {}".format(best_model_path))
    #best_model.load_state_dict(torch.load(best_model_path))
    best_model = torch.load(best_model_path)

    inferencer = inference_engine(model, device, threshold=threshold)

    #ProgressBar(desc="Inference").attach(inferencer)

    result_state = inferencer.run(test_loader)
    print(result_state.metrics)

    return now


def train_for_datasets(args):
    if args.dataset == 'all':
        for dataset in DATASETS:
            _args = copy.deepcopy(args)
            _args.dataset = dataset
            train_for_losses(_args)
    elif args.dataset == 'real':
        for dataset in REAL_DATASETS:
            _args = copy.deepcopy(args)
            _args.dataset = dataset
            train_for_losses(_args)
    elif args.dataset == 'syn':
        for dataset in SYN_DATASETS:
            _args = copy.deepcopy(args)
            _args.dataset = dataset
            train_for_losses(_args)
    else:
        return train_for_losses(args)

def train_for_losses(args, ts=None):
    if args.loss in LOSS_COLLECTIONS.keys():
        for loss in LOSS_COLLECTIONS[args.loss]:
            _args = copy.deepcopy(args)
            _args.loss = loss
            if ts is not None:
                _args.initial_weights = ts
            # params from author
            if loss == 'ap-perf-f1':
                _args.batch_size = 20
                _args.lr = 3e-4
            ts = train(_args)
        return ts
    else:
        return train(args)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model', type=str,
            default='ff', help='feed forward or image network', choices=['ff', 'img'])
    parser.add_argument('--experiment', type=str, help='experiment name')
    parser.add_argument('--initial_weights', type=int, help='to load a prior model initial weights')
    parser.add_argument('--mode', type=str,
            required=True,
            default='train',
            choices=['train', 'test'])

    parser.add_argument('--early_stopping_patience', type=int, default=100, help="early stopping patience")
    parser.add_argument('--batch_size', type=int, default=2048, metavar='N',
                        help='input batch size for training (default: 2048)')
    parser.add_argument('--epochs', type=int, default=5000, metavar='N',
                        help='number of epochs to train (default: 5000)')
    parser.add_argument('--loss', type=str,
            required=True,
            default='bce',
            choices=LOSSES + list(LOSS_COLLECTIONS.keys()))
    parser.add_argument('--weight_loss', default=False, action='store_true')
    parser.add_argument('--oversample', default=False, action='store_true')
    parser.add_argument('--dataset', type=str,
            default='mammography',
            choices=DATASETS + ['real', 'syn', 'all'] + ['cifar-t','cifar-f'])
    parser.add_argument('--output', type=str, default="experiments",
                        help='output path for experiment data')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='N',
                        help='input batch size for training (default: 2048)')
    parser.add_argument('--gpu', type=int, default=0, metavar='N',
                        help='which gpu to use? [0, n]')

    args = parser.parse_args()

    if args.mode == 'train':
        # loop over all datasets and losses, if necessary
        train_for_datasets(args)
    elif args.mode == 'test':
        if args.loss in LOSS_COLLECTIONS.keys():
            for loss in LOSS_COLLECTIONS[args.loss]:
                args.loss = loss
                test(args)
        else:
            test(args)
    else:
        raise RuntimeError("Unknown mode")

if __name__ == '__main__':
    main()
