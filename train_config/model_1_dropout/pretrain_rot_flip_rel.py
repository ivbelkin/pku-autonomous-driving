import dataprocessing as D
import config as cfg
from torch.utils.data import DataLoader
import torch
from train_config.model_1.model import Model, prepare_train_batch, prepare_test_batch, decode_y_pred, decode_batch
from mynn import huber_loss, mean_distance, dist_to_coord
from torch import nn
import numpy as np
from ignite.contrib.handlers.param_scheduler import PiecewiseLinear, LRScheduler
from torch.optim.lr_scheduler import StepLR
import os
from dataprocessing import decode_test_sample_fn_v1 as decode_test_sample
import albumentations as A

workdir = os.path.join(cfg.WORKDIR, 'model_1_dropout', 'pretrain_2_rot_flip_scale100_rel')
os.makedirs(workdir, exist_ok=True)

n_epochs = 50

albu_train_transforms = A.Compose([
    A.OneOf([
        A.RandomBrightness(limit=(0.8, 1.2), always_apply=True),
        A.RGBShift(r_shift_limit=(-10, 10), g_shift_limit=(-10, 10), b_shift_limit=(-10, 10), always_apply=True),
        A.HueSaturationValue(hue_shift_limit=(-10, 10), sat_shift_limit=(-10, 10), val_shift_limit=(-10, 10), always_apply=True),
        A.RandomContrast(limit=(0.9, 1.1), always_apply=True)
    ], p=0.3),
    A.OneOf([
        A.Blur(blur_limit=3, always_apply=True),
        A.MedianBlur(blur_limit=3, always_apply=True),
        A.GaussNoise(var_limit=(10, 50), always_apply=True)
    ])
])

train_ds = D.PKUSingleObjectTrainDataset(
    json_annotations=os.path.join(cfg.CV_DIR, 'fold-1', 'train_objects_both_train.json'),
    images_dir=cfg.TRAIN_IMAGES,
    color_augment_fn=lambda dct: D.augment_fn_albu_color(dct, albu_train_transforms),
    geom_augment_fn=lambda dct: D.augment_fn_flip_rotate(D.augment_fn_bbox(dct, s=0.02), p_flip=0.5, p_rotate=0.5, rotatelim=(-5, 5)),
    prepare_sample_fn=D.prepare_train_sample_fn_v2,
    annotation_filter_fn=lambda ann: D.annotation_filter_fn_v1(ann,
        xlim=(-50, 50), ylim=(0, 50), zlim=(0, 200), dlim=(0, 100), wlim=(1, np.inf), hlim=(1, np.inf))
)
train_dl = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=8, pin_memory=False, drop_last=True)
iters_per_epoch = len(train_dl)

valid_ds = D.PKUSingleObjectTrainDataset(
    json_annotations=os.path.join(cfg.CV_DIR, 'fold-1', 'train_objects_both_valid.json'),
    images_dir=cfg.TRAIN_IMAGES,
    color_augment_fn=D.augment_fn_pass,
    prepare_sample_fn=D.prepare_train_sample_fn_v1,
    annotation_filter_fn=lambda ann: D.annotation_filter_fn_v1(ann,
        xlim=(-50, 50), ylim=(0, 50), zlim=(0, 200), dlim=(0, 100), wlim=(1, np.inf), hlim=(1, np.inf))
)
valid_dl = DataLoader(valid_ds, batch_size=64, shuffle=False, num_workers=8, pin_memory=False, drop_last=False)

test_ds = D.PKUSingleObjectTestDataset(
    json_image_info=cfg.TEST_IMAGE_INFO_JSON,
    json_detections=cfg.TEST_DETECTIONS,
    json_annotations=os.path.join(cfg.CV_DIR, 'fold-1', 'train_objects_both_train.json'),
    images_dir=cfg.TEST_IMAGES,
    masks_dir=cfg.TEST_IGNORE_MASKS,
    prepare_sample_fn=D.prepare_test_sample_fn_v1
)
test_dl = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=8, pin_memory=False, drop_last=False)

model = Model().float().cuda()
# model.load_state_dict(torch.load(os.path.join(os.path.join(cfg.WORKDIR, 'model_1', 'stage_2_bbox_aug_weight_decay_2'), 'checkpoints', '_model_12.pth')))
for p in model.backbone.parameters():
    p.requires_grad = False

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0005)
# optimizer.load_state_dict(torch.load(os.path.join(os.path.join(cfg.WORKDIR, 'model_1', 'stage_2_bbox_aug_weight_decay_2'), 'checkpoints', '_optimizer_12.pth')))

# scheduler = PiecewiseLinear(optimizer, "lr",
#                             milestones_values=[(0, 0.000001), (iters_per_epoch, 0.0001), (6 * iters_per_epoch, 0.000001), (n_epochs * iters_per_epoch, 0.0000001)])
scheduler = LRScheduler(lr_scheduler=StepLR(optimizer=optimizer, step_size=iters_per_epoch, gamma=0.95))

def loss_fn(y_pred, y):
    cls_loss = nn.functional.cross_entropy(y_pred['cls_score'], y['label'])

    rot_loss_l1 = nn.functional.l1_loss(y_pred['rotation'], y['rotation'])
    rot_loss_cos = 1 - (y_pred['rotation'] * y['rotation']).sum(dim=1).mean()

    trans_loss = huber_loss(y_pred['translation'], y['translation'], beta=1)
    true_distance = mean_distance(y_pred['translation'], y['translation'])

    loss = rot_loss_cos + 0.1 * trans_loss + cls_loss
    return dict(loss=loss, rot_loss_l1=rot_loss_l1, rot_loss_cos=rot_loss_cos, trans_loss=trans_loss, true_distance=true_distance, cls_loss=cls_loss)
