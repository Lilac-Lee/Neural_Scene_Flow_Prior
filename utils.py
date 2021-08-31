from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import time
from collections import defaultdict


# ANCHOR: metrics computation, follow FlowNet3D metrics....
def scene_flow_metrics(pred, labels):
    l2_norm = torch.sqrt(torch.sum((pred - labels) ** 2, 2)).cpu()  # Absolute distance error.
    labels_norm = torch.sqrt(torch.sum(labels * labels, 2)).cpu()
    relative_err = l2_norm / (labels_norm + 1e-20)

    EPE3D = torch.mean(l2_norm).item()  # Mean absolute distance error

    # NOTE: Acc_5
    error_lt_5 = torch.BoolTensor((l2_norm < 0.05))
    relative_err_lt_5 = torch.BoolTensor((relative_err < 0.05))
    acc3d_strict = torch.mean((error_lt_5 | relative_err_lt_5).float()).item()

    # NOTE: Acc_10
    error_lt_10 = torch.BoolTensor((l2_norm < 0.1))
    relative_err_lt_10 = torch.BoolTensor((relative_err < 0.1))
    acc3d_relax = torch.mean((error_lt_10 | relative_err_lt_10).float()).item()

    # NOTE: outliers
    l2_norm_gt_3 = torch.BoolTensor(l2_norm > 0.3)
    relative_err_gt_10 = torch.BoolTensor(relative_err > 0.1)
    outlier = torch.mean((l2_norm_gt_3 | relative_err_gt_10).float()).item()

    # NOTE: angle error
    unit_label = labels / labels.norm(dim=2, keepdim=True)
    unit_pred = pred / pred.norm(dim=2, keepdim=True)
    eps = 1e-7
    dot_product = (unit_label * unit_pred).sum(2).clamp(min=-1+eps, max=1-eps)
    dot_product[dot_product != dot_product] = 0  # Remove NaNs
    angle_error = torch.acos(dot_product).mean().item()

    return EPE3D, acc3d_strict, acc3d_relax, outlier, angle_error


# ANCHOR: timer!
class Timers(object):
    def __init__(self):
        self.timers = defaultdict(Timer)

    def tic(self, key):
        self.timers[key].tic()

    def toc(self, key):
        self.timers[key].toc()

    def print(self, key=None):
        if key is None:
            for k, v in self.timers.items():
                print("Average time for {:}: {:}".format(k, v.avg()))
        else:
            print("Average time for {:}: {:}".format(key, self.timers[key].avg()))

    def get_avg(self, key):
        return self.timers[key].avg()
    
    
class Timer(object):
    def __init__(self):
        self.reset()

    def tic(self):
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1

    def total(self):
        return self.total_time

    def avg(self):
        return self.total_time / float(self.calls)

    def reset(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.


# ANCHOR: generator
class GeneratorWrap:
    def __init__(self, gen):
        self.gen = gen

    def __iter__(self):
        self.value = yield from self.gen


# ANCHOR: early stopping strategy
class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if torch.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                            best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                            best * min_delta / 100)
                