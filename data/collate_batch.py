# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch


def train_collate_fn(batch):
    imgs1, imgs2, imgs3, pids, _, _, = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    return torch.stack(imgs1, dim=0),torch.stack(imgs2, dim=0),torch.stack(imgs3, dim=0), pids


def val_collate_fn(batch):
    imgs1, imgs2, imgs3, pids, camids, _ = zip(*batch)
    return torch.stack(imgs1, dim=0),torch.stack(imgs2, dim=0),torch.stack(imgs3, dim=0), pids, camids
