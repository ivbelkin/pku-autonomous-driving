from torch.utils.data import Dataset
from tqdm import tqdm
import json
import os
import logging
from PIL import Image
import io
import numpy as np
from utils import euler_angles_to_quaternions, rotation_to_quaternion, quaternion_to_rotation
from torchvision import transforms as T
from utils import fit_image, parse_camera_intrinsic, orient_quaternion
import config as C
import numpy as np
from tqdm import tqdm
import torch
import pickle


class PKUSingleObjectTrainDataset(Dataset):

    def __init__(self, json_annotations, images_dir, color_augment_fn=None, geom_augment_fn=None, prepare_sample_fn=None, annotation_filter_fn=None, image_keys=('image',)):
        self.json_annotations = json_annotations
        self.images_dir = images_dir
        self.color_augment_fn = color_augment_fn
        self.geom_augment_fn = geom_augment_fn
        self.prepare_sample_fn = prepare_sample_fn
        self.image_keys = image_keys

        C.logger.info("Loading annotations from %s", json_annotations)
        with open(json_annotations, 'r') as f:
            self.gt = json.load(f)

        cat_ids = set(ann['category_id'] for ann in self.gt['annotations'] if ann['iscrowd'] == 0)
        categories = [cat for cat in self.gt['categories'] if cat['id'] in cat_ids]
        self.category_id_to_label = {
            cat["id"]: label
            for label, cat in enumerate(sorted(categories, key=lambda x: x["id"]))
        }
        C.logger.debug("Number of labels: %i", len(self.category_id_to_label))
        C.logger.debug(self.category_id_to_label)

        if annotation_filter_fn is not None:
            N1 = len(self.gt['annotations'])
            self.gt['annotations'] = list(filter(annotation_filter_fn, self.gt['annotations']))
            N2 = len(self.gt['annotations'])
            C.logger.info("Removed %i objects", N1 - N2)
        C.logger.info("Number of objects: %i", len(self.gt['annotations']))

        self.images_jpeg = self.load_images()

        self.to_tensor = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.gt['annotations'])

    def __getitem__(self, idx):
        dct = self._getdct(idx)
        if self.geom_augment_fn is not None:
            dct = self.geom_augment_fn(dct)
        if self.prepare_sample_fn is not None:
            dct = self.prepare_sample_fn(dct)
        if self.color_augment_fn is not None:
            dct = self.color_augment_fn(dct)
        for k in self.image_keys:
            dct[k] = self.to_tensor(dct[k])
        return dct

    def _getdct(self, idx):
        ann = self.gt['annotations'][idx]
        image = PKUSingleObjectTrainDataset.decode_image(self.images_jpeg[ann['image_id']])
        dct = dict(
            idx=idx,
            image_id=ann['image_id'],
            image=image,
            bbox=np.array(ann['bbox']),
            translation=np.array(ann['position']),
            rotation=np.array(ann['orientation']),
            label=self.category_id_to_label[ann['category_id']],
            score=1.0,
        )
        return dct

    def load_images(self):
        C.logger.info("Loading images")
        images = {}
        for image in tqdm(self.gt['images']):
            path = os.path.join(self.images_dir, image['file_name'])
            data = open(path, 'rb').read()
            images[image['id']] = io.BytesIO(data)
        return images

    @staticmethod
    def decode_image(bytes_io):
        image = Image.open(bytes_io)
        image.load()
        return image


class PKUSingleObjectTestDataset(Dataset):

    def __init__(self, json_image_info, json_detections, json_annotations, images_dir, masks_dir, prepare_sample_fn=None, image_keys=('image',)):
        self.json_image_info = json_image_info
        self.json_detections = json_detections
        self.json_annotations = json_annotations
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.prepare_sample_fn = prepare_sample_fn
        self.image_keys = image_keys

        C.logger.info("Loading image info from %s", json_image_info)
        with open(json_image_info, 'r') as f:
            self.ii = json.load(f)

        C.logger.info("Loading detections from %s", json_detections)
        with open(json_detections, 'r') as f:
            self.dt = json.load(f)

        C.logger.info("Loading annotations from %s", json_annotations)
        with open(json_annotations, 'r') as f:
            self.gt = json.load(f)

        cat_ids = set(ann['category_id'] for ann in self.gt['annotations'] if ann['iscrowd'] == 0)
        categories = [cat for cat in self.gt['categories'] if cat['id'] in cat_ids]
        self.category_id_to_label = {
            cat["id"]: label
            for label, cat in enumerate(sorted(categories, key=lambda x: x["id"]))
        }
        C.logger.debug("Number of labels: %i", len(self.category_id_to_label))
        C.logger.debug(self.category_id_to_label)

        self.images_jpeg = self.load_images()

        self.to_tensor = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.dt)

    def __getitem__(self, idx):
        dct = self._getdct(idx)
        if self.prepare_sample_fn is not None:
            dct = self.prepare_sample_fn(dct)
        for k in self.image_keys:
            dct[k] = self.to_tensor(dct[k])
        return dct

    def _getdct(self, idx):
        ann = self.dt[idx]
        image = PKUSingleObjectTestDataset.decode_image(self.images_jpeg[ann['image_id']])
        dct = dict(
            idx=idx,
            image_id=ann['image_id'],
            image=image,
            bbox=np.array(ann['bbox']),
            score=float(ann['score'])
        )
        return dct

    def load_images(self):
        C.logger.info("Loading images")
        images = {}
        for image in tqdm(self.ii['images']):
            path = os.path.join(self.images_dir, image['file_name'])
            data = open(path, 'rb').read()
            images[image['id']] = io.BytesIO(data)
        return images

    @staticmethod
    def decode_image(bytes_io):
        image = Image.open(bytes_io)
        image.load()
        return image


def augment_fn_pass(dct):
    return dct


def augment_fn_albu_color(dct, albu):
    dct['image'] = Image.fromarray(albu(image=np.array(dct['image']))['image'])
    return dct


def augment_fn_bbox(dct):
    x, y, w, h = dct['bbox']
    cx, cy = x + w / 2, y + h / 2
    scale = np.array([w, h]) / 100
    cx, cy = np.array([cx, cy]) + scale * np.random.normal(size=2)
    w, h = np.array([w, h]) + scale * np.random.normal(size=2)
    x, y = cx - w / 2, cy - h / 2
    dct['bbox'] = [x, y, w, h]
    return dct


def prepare_train_sample_fn_v1(dct):
    k = parse_camera_intrinsic()
    x, y, w, h = dct['bbox']
    dct['bbox'] = np.array([
        (x + w / 2 - k['cx']) / k['fx'], 
        (y + h / 2 - k['cy']) / k['fy'], 
        w / k['fx'], h / k['fy']
    ])
    dct['image'] = dct['image'].crop(tuple(map(int, [x, y, x + w, y + h])))
    dct['image'] = fit_image(dct['image'], 256)
    q = rotation_to_quaternion(dct['rotation'])
    dct['rotation'] = orient_quaternion(q)
    return dct


def prepare_test_sample_fn_v1(dct):
    k = parse_camera_intrinsic()
    x, y, w, h = dct['bbox']
    dct['bbox'] = np.array([
        (x + w / 2 - k['cx']) / k['fx'], 
        (y + h / 2 - k['cy']) / k['fy'], 
        w / k['fx'], h / k['fy']
    ])
    dct['image'] = dct['image'].crop(tuple(map(int, [x, y, x + w, y + h])))
    dct['image'] = fit_image(dct['image'], 256)
    return dct


def decode_test_sample_fn_v1(x, y_pred):
    dct = dict(translation=y_pred['translation'], rotation=quaternion_to_rotation(y_pred['rotation']), score=x['score'], image_id=x['image_id'])
    return dct


def annotation_filter_fn_crowd(ann):
    return bool(ann['iscrowd'] == 0)


def annotation_filter_fn_world_x(ann, xlim):
    wx, wy, wz = ann['position']
    return xlim[0] <= wx <= xlim[1]


def annotation_filter_fn_world_y(ann, ylim):
    wx, wy, wz = ann['position']
    return ylim[0] <= wy <= ylim[1]


def annotation_filter_fn_world_z(ann, zlim):
    wx, wy, wz = ann['position']
    return zlim[0] <= wz <= zlim[1]


def annotation_filter_fn_world_box(ann, xlim, ylim, zlim):
    wx, wy, wz = ann['position']
    return (
        annotation_filter_fn_world_x(ann, xlim) 
        and annotation_filter_fn_world_y(ann, ylim) 
        and annotation_filter_fn_world_z(ann, zlim)
    )


def annotation_filter_fn_bbox_w(ann, wlim):
    x, y, w, h = ann['bbox']
    return wlim[0] <= w <= wlim[1]


def annotation_filter_fn_bbox_h(ann, hlim):
    x, y, w, h = ann['bbox']
    return hlim[0] <= h <= hlim[1]


def annotation_filter_fn_bbox(ann, wlim, hlim):
    return (
        annotation_filter_fn_bbox_w(ann, wlim)
        and annotation_filter_fn_bbox_h(ann, hlim)
    )


def annotation_filter_fn_distance(ann, dlim):
    x, y, z = ann['position']
    return dlim[0] <= np.sqrt(x**2 + y ** 2 + z ** 2) <= dlim[1]


def annotation_filter_fn_v1(ann, xlim, ylim, zlim, dlim, wlim, hlim):
    return (
        annotation_filter_fn_crowd(ann)
        and annotation_filter_fn_world_box(ann, xlim, ylim, zlim)
        and annotation_filter_fn_distance(ann, dlim)
        and annotation_filter_fn_bbox(ann, wlim, hlim)
    )
