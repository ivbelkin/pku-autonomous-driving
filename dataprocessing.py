from torch.utils.data import Dataset
from tqdm import tqdm
import json
import os
import logging
from PIL import Image
import io
import numpy as np
from utils import euler_angles_to_quaternions, rotation_to_quaternion
from torchvision import transforms as T
from utils import fit_image, parse_camera_intrinsic, orient_quaternion
import config as C
import numpy as np
from tqdm import tqdm
import torch
import pickle


class PKUSingleObjectDataset(Dataset):

    def __init__(self, json_annotations, images_dir, augment_fn=None, prepare_sample_fn=None, annotation_filter_fn=None, image_keys=('image',)):
        self.json_annotations = json_annotations
        self.images_dir = images_dir
        self.augment_fn = augment_fn
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
        if self.augment_fn is not None:
            dct = self.augment_fn(dct)
        if self.prepare_sample_fn is not None:
            dct = self.prepare_sample_fn(dct)
        for k in self.image_keys:
            dct[k] = self.to_tensor(dct[k])
        return dct

    def _getdct(self, idx):
        ann = self.gt['annotations'][idx]
        image = PKUSingleObjectDataset.decode_image(self.images_jpeg[ann['image_id']])
        dct = dict(
            idx=idx,
            image_id=ann['image_id'],
            image=image,
            bbox=np.array(ann['bbox']),
            translation=np.array(ann['position']),
            rotation=np.array(ann['orientation']),
            label=self.category_id_to_label[ann['category_id']]
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


class PKUSingleObjectWithPrecomputedImageFeaturesDataset(PKUSingleObjectDataset):

    def __init__(self, extract_featues_fn, workdir, file_name, dct_key, **kwargs):
        super().__init__(**kwargs)
        self.dct_key = dct_key

        path = os.path.join(workdir, file_name)
        if os.path.exists(path + '.data') and os.path.exists(path + '.index'):
            with open(path + '.index', 'rb') as f:
                self.image_id_to_features_index = pickle.load(f)
            print(self.image_id_to_features_index)
            with open(path + '.data', 'rb') as f:
                self.features = pickle.load(f)
        else:
            self.image_id_to_features_index, self.features = self.extract_features(extract_featues_fn)
            with open(path + '.index', 'wb') as f:
                pickle.dump(self.image_id_to_features_index, f)
            with open(path + '.data', 'wb') as f:
                pickle.dump(self.features, f)

    def __getitem__(self, idx):
        dct = super().__getitem__(idx)
        dct[self.dct_key] = self.features[self.image_id_to_features_index[dct['image_id']]]
        return dct

    def extract_features(self, extract_featues_fn):
        C.logger.info("Start feature extraction")
        image_id_to_features_index, features = {}, []
        for idx in tqdm(range(super().__len__())):
            dct = super().__getitem__(idx)
            image_id = dct['image_id']
            if image_id not in image_id_to_features_index:
                image_id_to_features_index[image_id] = len(features)
                features.append(extract_featues_fn(dct))
        return image_id_to_features_index, features


class ResnetFeatureExtractor:

    def __init__(self, model, dct_key):
        self.model = model
        self.dct_key = dct_key

    def __call__(self, dct):
        x = dct[self.dct_key].float().cuda()[None, :, :, :]

        with torch.no_grad():
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)

            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)
            x = self.model.layer4(x)

        return x.cpu().numpy()


def augment_fn_pass(dct):
    return dct


def prepare_sample_fn_v1(dct):
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
