import os
import argparse
import json
import random
from tqdm import tqdm

import config as C


def build_parser():
    parser = argparse.ArgumentParser()
    return parser


def load_image_info():
    with open(C.TRAIN_OBJECTS_BBOX_JSON, "r") as f:
        coco_gt = json.load(f)
    return coco_gt["images"]


def split_on_folds(image_info):
    folds = {fold: [] for fold in range(1, 1 + C.N_FOLDS)}
    for i, image in enumerate(image_info):
        folds[1 + i % C.N_FOLDS].append(image["id"])
    return folds


def main(args):
    os.makedirs(C.CV_DIR, exist_ok=True)

    image_info = load_image_info()

    random.seed(C.RANDOM_STATE)
    random.shuffle(image_info)
    
    folds = split_on_folds(image_info)

    for fold in tqdm(range(1, 1 + C.N_FOLDS)):
        fold_dir = os.path.join(C.CV_DIR, "fold-{}".format(fold))
        os.makedirs(fold_dir, exist_ok=True)

        for filepath in tqdm(C.ANNOTATIONS_JSON):
            filepath = filepath.replace(".json", "_with_ignore.json")
            filename = os.path.basename(filepath)
            with open(filepath, "r") as f:
                coco_gt = json.load(f)
            train_gt = dict(categories=coco_gt["categories"], images=[], annotations=[])
            valid_gt = dict(categories=coco_gt["categories"], images=[], annotations=[])
            for image in coco_gt["images"]:
                if image["id"] in folds[fold]:
                    valid_gt["images"].append(image)
                else:
                    train_gt["images"].append(image)
            for ann in coco_gt["annotations"]:
                if ann["image_id"] in folds[fold]:
                    valid_gt["annotations"].append(ann)
                else:
                    train_gt["annotations"].append(ann)
            filename_train = filename[:-5] + "_train.json"
            filename_valid = filename[:-5] + "_valid.json"
            with open(os.path.join(fold_dir, filename_train), "w") as f:
                json.dump(train_gt, f)
            with open(os.path.join(fold_dir, filename_valid), "w") as f:
                json.dump(valid_gt, f)


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
