import dataprocessing as D
import config as cfg
from torch.utils.data import DataLoader
import torch
from train_config.sample.model import Model, prepare_batch
from mynn import huber_loss, mean_distance, dist_to_coord
from torch import nn
import numpy as np
from ignite.contrib.handlers.param_scheduler import PiecewiseLinear, LRScheduler
from torch.optim.lr_scheduler import StepLR

workdir = 'sample_run_4'

n_epochs = 10

train_ds = D.PKUJsonDataset(
    json_annotations=cfg.TRAIN_OBJECTS_BOTH_JSON,
    images_dir=cfg.TRAIN_IMAGES,
    augment_fn=D.augment_fn_pass,
    prepare_sample_fn=D.prepare_sample_fn_v1,
    annotation_filter_fn=lambda ann: D.annotation_filter_fn_v1(ann,
        xlim=(-50, 50), ylim=(0, 50), zlim=(0, 200), dlim=(0, 100), wlim=(1, np.inf), hlim=(1, np.inf))
)

train_dl = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
iters_per_epoch = len(train_dl)

model = Model().float().cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# scheduler = PiecewiseLinear(optimizer, "lr",
#                             milestones_values=[(0, 0.000001), (iters_per_epoch, 0.0001), (6 * iters_per_epoch, 0.000001), (n_epochs * iters_per_epoch, 0.0000001)])
scheduler = LRScheduler(lr_scheduler=StepLR(optimizer=optimizer, step_size=iters_per_epoch, gamma=0.5))

def loss_fn(outputs, batch):
    cls_loss = nn.functional.cross_entropy(outputs['cls_score'], batch['label'])
    normed = outputs['rotation'] / torch.norm(outputs['rotation'], dim=-1, keepdim=True)

    rot_loss_l1 = nn.functional.l1_loss(normed, batch['rotation'])
    rot_loss_cos = 1 - (normed[:, -1] * batch['rotation'][:, -1] + (normed[:, :-1] * batch['rotation'][:, :-1]).sum(dim=1)).mean()

    trans_loss = huber_loss(dist_to_coord(outputs, batch), batch['translation'])
    true_distance = mean_distance(batch, outputs)

    loss = rot_loss_cos + 0.1 * trans_loss + cls_loss
    return dict(loss=loss, rot_loss_l1=rot_loss_l1, rot_loss_cos=rot_loss_cos, trans_loss=trans_loss, true_distance=true_distance)
