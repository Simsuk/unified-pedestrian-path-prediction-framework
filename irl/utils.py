import os
import time
import torch
import numpy as np
import inspect
from contextlib import contextmanager
import subprocess
import math
import logging

# STGAT =================
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logging.info("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"



def displacement_error(pred_traj, pred_traj_gt, consider_ped=None, mode="sum"):
    """
    Input:
    - pred_traj: Tensor of shape (seq_len, batch, 2). Predicted trajectory. [12, person_num, 2]
    - pred_traj_gt: Tensor of shape (seq_len, batch, 2). Ground truth
    predictions.
    - consider_ped: Tensor of shape (batch)
    - mode: Can be one of sum, raw
    Output:
    - loss: gives the eculidian displacement error
    """

    seq_len, _, _ = pred_traj.size()
    loss = pred_traj_gt.permute(1, 0, 2) - pred_traj.permute(1, 0, 2)

    loss = loss ** 2
    if consider_ped is not None:
        loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1) * consider_ped
    else:
        loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1)
    if mode == "sum":
        return torch.sum(loss)
    elif mode == "mean":
        return torch.mean(loss)
    elif mode == "raw":
        return loss
def l2_loss(pred_traj, pred_traj_gt, loss_mask, random=0, mode="average"):
    """
    Input:
    - pred_traj: Tensor of shape (seq_len, batch, 2). Predicted trajectory.
    - pred_traj_gt: Tensor of shape (seq_len, batch, 2). Groud truth
    predictions.
    - loss_mask: Tensor of shape (batch, seq_len)
    - mode: Can be one of sum, average, raw
    Output:
    - loss: l2 loss depending on mode
    """
    seq_len, batch, _ = pred_traj.size()
    # equation below , the first part does noing, can be delete
    # print(pred_traj_gt.permute(1, 0, 2))
    loss = (pred_traj_gt.permute(1, 0, 2) - pred_traj.permute(1, 0, 2)) ** 2
    if mode == "sum":
        return torch.sum(loss)
    elif mode == "average":
        return torch.sum(loss) / torch.numel(loss_mask.data)
    elif mode == "raw":
        return loss.sum(dim=2).sum(dim=1)

def final_displacement_error(pred_pos, pred_pos_gt, consider_ped=None, mode="sum"):
    """
    Input:
    - pred_pos: Tensor of shape (batch, 2). Predicted last pos.
    - pred_pos_gt: Tensor of shape (seq_len, batch, 2). Groud truth
    last pos
    - consider_ped: Tensor of shape (batch)
    Output:
    - loss: gives the eculidian displacement error
    """

    loss = pred_pos_gt - pred_pos
    loss = loss ** 2
    if consider_ped is not None:
        loss = torch.sqrt(loss.sum(dim=1)) * consider_ped
    else:
        loss = torch.sqrt(loss.sum(dim=1))
    if mode == "raw":
        return loss
    else:
        return torch.sum(loss)
# =========================

def int_tuple(s):
    return tuple(int(i) for i in s.split(','))


def find_nan(variable, var_name):
    variable_n = variable.data.cpu().numpy()
    if np.isnan(variable_n).any():
        exit('%s has nan' % var_name)


def bool_flag(s):
    if s == '1':
        return True
    elif s == '0':
        return False
    msg = 'Invalid value "%s" for bool flag (should be 0 or 1)'
    raise ValueError(msg % s)


def lineno():
    return str(inspect.currentframe().f_back.f_lineno)


def get_total_norm(parameters, norm_type=2):
    if norm_type == float('inf'):
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            try:
                param_norm = p.grad.data.norm(norm_type)
                total_norm += param_norm**norm_type
                total_norm = total_norm**(1. / norm_type)
            except:
                continue
    return total_norm


@contextmanager
def timeit(msg, should_time=True):
    if should_time:
        torch.cuda.synchronize()
        t0 = time.time()
    yield
    if should_time:
        torch.cuda.synchronize()
        t1 = time.time()
        duration = (t1 - t0) * 1000.0
        print('%s: %.2f ms' % (msg, duration))


def get_gpu_memory():
    torch.cuda.synchronize()
    opts = [
        'nvidia-smi', '-q', '--gpu=' + str(1), '|', 'grep', '"Used GPU Memory"'
    ]
    cmd = str.join(' ', opts)
    ps = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = ps.communicate()[0].decode('utf-8')
    output = output.split("\n")[0].split(":")
    consumed_mem = int(output[1].strip().split(" ")[0])
    return consumed_mem


def get_dset_path(dset_name, dset_type):
    _dir = os.path.dirname(__file__)
    _dir = _dir.split("/")[:-1]
    _dir = "/".join(_dir)
    return os.path.join(_dir, 'datasets', dset_name, dset_type)


def relative_to_abs(rel_traj, start_pos):
    """
    Inputs:
    - rel_traj: pytorch tensor of shape (seq_len, batch, 2)
    - start_pos: pytorch tensor of shape (batch, 2)
    Outputs:
    - abs_traj: pytorch tensor of shape (seq_len, batch, 2)
    """
    # batch, seq_len, 2
    # print("rel_traj",rel_traj.shape)
    rel_traj = rel_traj.permute(1, 0, 2)
    displacement = torch.cumsum(rel_traj, dim=1)
    start_pos = torch.unsqueeze(start_pos, dim=1)
    abs_traj = displacement + start_pos
    return abs_traj.permute(1, 0, 2)


def normal_entropy(std):
    var = std.pow(2)
    entropy = 0.5 + 0.5 * torch.log(2 * var * math.pi)
    return entropy.sum(1, keepdim=True)



def normal_log_density(x, mean, log_std, std):
    # Take the probability density function (normal distribution) and take the log of that, which can be expressed
    # as the equation below (checked). The log density tells us the likelihood of action X, given the mean and variance
    # we take the sum in dimension 1 because of x and y coordinate (sum of probs because log)
    var = std.pow(2)
    log_density = -(x - mean).pow(2) / (2 * var) - 0.5 * math.log(2 * math.pi) - log_std
    # print("\n log_density: \n shape:", log_density.shape, "\n values: ", log_density)
    return log_density.sum(1, keepdim=True)


def to_device(device, *args):
    return [x.to(device) for x in args]


def print_time(start, middle, end, epoch):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    hours2, rem2 = divmod(end - middle, 3600)
    minutes2, seconds2 = divmod(rem2, 60)
    print("Epoch", epoch, "took:", "{:0>2}:{:0>2}:{:05.2f}".format(int(hours2),int(minutes2),seconds2),
          "\t Total training time:", "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))

def print_t(start, end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("time it took:", "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))