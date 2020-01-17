import pycocotools
import os
import argparse
from PIL import Image
import json
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import pycocotools.mask as mask_utils
import cv2
from scipy.ndimage.morphology import binary_fill_holes
import pandas as pd

import utils
import config as C
import car_models


def build_parser():
    parser = argparse.ArgumentParser()
    return parser


def obj_to_bbox(vertices, triangles):
    xtl, ytl, xbr, ybr = np.inf, np.inf, 0, 0
    for t in triangles:
        coord = np.array([vertices[t[0]][:2], vertices[t[1]][:2], vertices[t[2]][:2]], dtype=np.int32)
        xtl = min(xtl, np.min(coord[:, 0]))
        xbr = max(xbr, np.max(coord[:, 0]))
        ytl = min(ytl, np.min(coord[:, 1]))
        ybr = max(ybr, np.max(coord[:, 1]))
    return xtl, ytl, xbr, ybr


def draw_segm_mask(mask, vertices, triangles, color):
    m = np.zeros_like(mask)
    for t in triangles:
        coord = np.array([vertices[t[0]][:2], vertices[t[1]][:2], vertices[t[2]][:2]], dtype=np.int32)
        cv2.fillConvexPoly(m, coord, 1)
    m = binary_fill_holes(m).astype(np.uint8)
    mask[m.astype(bool)] = color


def process_train(categories, car_id2vertices, car_id2triangles):
    cars = utils.load_train_annotations()
    image_filenames = sorted(os.listdir(C.TRAIN_IMAGES))
    name_to_cat_id = {cat["name"]: cat["id"] for cat in categories}

    annotation_id = 1
    images, annotations = [], []
    for image_id, filename in enumerate(tqdm(image_filenames)):
        car = cars[filename.split(".")[0]]
        path = os.path.join(C.TRAIN_IMAGES, filename)
        width, height = Image.open(path).size
        mask = np.zeros((height, width), dtype=np.uint8)
        
        dist = [x**2 + y**2 + z**2 for x, y, z in zip(car["xs"], car["ys"], car["zs"])]
        it = zip(dist, car["model_types"], car["yaws"], car["pitches"], car["rolls"], car["xs"], car["ys"], car["zs"])
        anns = []
        for i, (d, model_type, yaw, pitch, roll, x, y, z) in enumerate(sorted(it, reverse=True), start=1):
            model_type = int(model_type)
            vertices = car_id2vertices[model_type]
            triangles = car_id2triangles[model_type]
            img_cor_points = utils.project_vertices(vertices, yaw, pitch, roll, x, y, z)

            xtl, ytl, xbr, ybr = obj_to_bbox(img_cor_points, triangles)
            xtl = np.clip(xtl, 0, width)
            ytl = np.clip(ytl, 0, height)
            xbr = np.clip(xbr, 0, width)
            ybr = np.clip(ybr, 0, height)
            w, h = xbr - xtl, ybr - ytl

            draw_segm_mask(mask, img_cor_points, triangles, i)

            anns.append(dict(
                id=int(annotation_id),
                image_id=int(image_id),
                category_id=int(name_to_cat_id[car_models.car_id2name[model_type]]),
                area=float(w * h),
                bbox=[float(xtl), float(ytl), float(w), float(h)],
                iscrowd=0,
                position=[float(x), float(y), float(z)],
                orientation=[float(yaw), float(pitch), float(roll)]
            ))
            annotation_id += 1
        
        for i, ann in enumerate(anns, start=1):
            m = np.asfortranarray(255 * (mask == i).astype(np.uint8))
            rle = mask_utils.encode(m)
            rle["counts"] = rle["counts"].decode()
            ann["segmentation"] = rle

        annotations.extend(anns)

        images.append(dict(
            id=int(image_id),
            width=int(width),
            height=int(height),
            file_name=filename,
        ))

    coco_gt = dict(images=images, categories=categories, annotations=annotations)
    return coco_gt


def join_classes(coco_gt):
    coco_gt_new = deepcopy(coco_gt)
    if "annotations" in coco_gt_new:
        for ann in coco_gt_new["annotations"]:
            ann["category_id"] = 1
    coco_gt_new["categories"] = [{"id": 1, "name": "vehicle", "supercategory": "vehicle"}]
    return coco_gt_new


def select_segmentation(coco_gt):
    coco_gt_new = deepcopy(coco_gt)
    for ann in coco_gt_new["annotations"]:
        del ann["bbox"]
    return coco_gt_new


def select_bbox(coco_gt):
    coco_gt_new = deepcopy(coco_gt)
    for ann in coco_gt_new["annotations"]:
        del ann["segmentation"]
    return coco_gt_new


def process_test(categories):
    image_filenames = sorted(os.listdir(C.TEST_IMAGES))
    images = []
    for image_id, filename in enumerate(tqdm(image_filenames)):
        path = os.path.join(C.TEST_IMAGES, filename)
        width, height = Image.open(path).size
        images.append(dict(
            id=int(image_id),
            width=int(width),
            height=int(height),
            file_name=filename,
        ))
    coco_gt = dict(images=images, categories=categories)
    return coco_gt


def process_ignore_masks(path):
    filenames = sorted(os.listdir(path))

    filename_to_annotations = {}
    for filename in tqdm(filenames):
        filepath = os.path.join(path, filename)
        mask = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
        ret, labels = cv2.connectedComponents(mask)

        masks = []
        for label in range(1, ret):
            m = np.asfortranarray(255 * (labels == label).astype(np.uint8))
            rle = mask_utils.encode(m)
            rle["counts"] = rle["counts"].decode()
            x, y, w, h = cv2.boundingRect(m)
            masks.append(dict(
                segmentation=rle, 
                bbox=[float(x), float(y), float(w), float(h)],
                area=float(w * h)
            ))
        filename_to_annotations[filename] = masks
    
    return filename_to_annotations


def merge_with_ignore_masks(coco_gt, filename_to_annotations):
    filename_to_id = {image["file_name"]: image["id"] for image in coco_gt["images"]}
    if "annotations" in coco_gt:
        annotation_id = len(coco_gt["annotations"]) + 1
    else:
        annotation_id = 1
        coco_gt["annotations"] = []
    for filename, annotations in tqdm(filename_to_annotations.items()):
        for ann in annotations:
            coco_gt["annotations"].append(dict(
                id=int(annotation_id),
                image_id=int(filename_to_id[filename]),
                category_id=1,  # category doesn't matter and not given
                area=ann["area"],
                bbox=ann["bbox"],
                segmentation=ann["segmentation"],
                iscrowd=1  # important !!!
            ))
            annotation_id += 1


def remove_noisy_images(coco_gt):
    df = pd.read_csv(C.NOISY_IMAGES)
    print(len(coco_gt["images"]), len(coco_gt["annotations"]))
    noisy_image_ids = [image["id"] for image in coco_gt["images"] if image["file_name"] in list(df["pku_name"])]
    print(noisy_image_ids)
    coco_gt["images"] = [image for image in coco_gt["images"] if image["id"] not in noisy_image_ids]
    coco_gt["annotations"] = [ann for ann in coco_gt["annotations"] if ann["image_id"] not in noisy_image_ids]
    print(len(coco_gt["images"]), len(coco_gt["annotations"]))


def main(args):
    os.makedirs(C.JSON_DIR, exist_ok=True)

    categories = [
        {"id": int(id_) + 1, "name": name, "supercategory": "vehicle"}  # id should start from 1
        for id_, name in car_models.car_id2name.items()
    ]
    car_id2vertices, car_id2triangles = utils.load_car_models()

    train_gt = process_train(categories, car_id2vertices, car_id2triangles)
    train_masks = process_ignore_masks(C.TRAIN_IGNORE_MASKS)
    merge_with_ignore_masks(train_gt, train_masks)
    remove_noisy_images(train_gt)

    train_single_class_gt = join_classes(train_gt)

    with open(C.TRAIN_OBJECTS_BOTH_JSON, "w") as f:
        json.dump(train_gt, f)
    with open(C.TRAIN_OBJECTS_BBOX_JSON, "w") as f:
        json.dump(select_bbox(train_gt), f)
    with open(C.TRAIN_OBJECTS_SEGM_JSON, "w") as f:
        json.dump(select_segmentation(train_gt), f)

    with open(C.TRAIN_OBJECTS_BOTH_SINGLE_CLASS_JSON, "w") as f:
        json.dump(train_single_class_gt, f)
    with open(C.TRAIN_OBJECTS_BBOX_SINGLE_CLASS_JSON, "w") as f:
        json.dump(select_bbox(train_single_class_gt), f)
    with open(C.TRAIN_OBJECTS_SEGM_SINGLE_CLASS_JSON, "w") as f:
        json.dump(select_segmentation(train_single_class_gt), f)

    test_gt = process_test(categories)
    test_masks = process_ignore_masks(C.TEST_IGNORE_MASKS)
    merge_with_ignore_masks(test_gt, test_masks)
    remove_noisy_images(test_gt)

    test_single_class_gt = join_classes(test_gt)

    with open(C.TEST_OBJECTS_IMAGE_INFO_JSON, "w") as f:
        json.dump(test_gt, f)
    
    with open(C.TEST_OBJECTS_SINGLE_CLASS_IMAGE_INFO_JSON, "w") as f:
        json.dump(test_single_class_gt, f)

    # train_masks = process_ignore_masks(C.TRAIN_IGNORE_MASKS)

    # with open(C.TRAIN_OBJECTS_BOTH_JSON, "r") as f:
    #     train_gt = json.load(f)
    # merge_with_ignore_masks(train_gt, train_masks)

    # with open(C.TRAIN_OBJECTS_BOTH_JSON.replace(".json", "_with_ignore.json"), "w") as f:
    #     json.dump(train_gt, f)
    # with open(C.TRAIN_OBJECTS_BBOX_JSON.replace(".json", "_with_ignore.json"), "w") as f:
    #     json.dump(select_bbox(train_gt), f)
    # with open(C.TRAIN_OBJECTS_SEGM_JSON.replace(".json", "_with_ignore.json"), "w") as f:
    #     json.dump(select_segmentation(train_gt), f)

    # with open(C.TRAIN_OBJECTS_BOTH_SINGLE_CLASS_JSON, "r") as f:
    #     train_single_class_gt = json.load(f)
    # merge_with_ignore_masks(train_single_class_gt, train_masks)

    # with open(C.TRAIN_OBJECTS_BOTH_SINGLE_CLASS_JSON.replace(".json", "_with_ignore.json"), "w") as f:
    #     json.dump(train_single_class_gt, f)
    # with open(C.TRAIN_OBJECTS_BBOX_SINGLE_CLASS_JSON.replace(".json", "_with_ignore.json"), "w") as f:
    #     json.dump(select_bbox(train_single_class_gt), f)
    # with open(C.TRAIN_OBJECTS_SEGM_SINGLE_CLASS_JSON.replace(".json", "_with_ignore.json"), "w") as f:
    #     json.dump(select_segmentation(train_single_class_gt), f)

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
