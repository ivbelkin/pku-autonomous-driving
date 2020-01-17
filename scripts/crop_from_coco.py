import os
import cv2
import pandas as pd
import json
import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

import utils
import config as C


def main():
    os.makedirs(C.TRAIN_CROPS_DIR, exist_ok=True)
    os.makedirs(C.TRAIN_CROPS_ORIGIN, exist_ok=True)
    os.makedirs(C.TRAIN_CROPS_CALIBRATED, exist_ok=True)

    camera_matrix = utils.get_camera_matrix()
    inv_camera_matric = np.linalg.inv(camera_matrix)

    with open(C.TRAIN_OBJECTS_BBOX_JSON, "r") as f:
        train_gt = json.load(f)
    
    image_id_to_name = {image["id"]: image["file_name"] for image in train_gt["images"]}
    image_id_to_annotations = {}
    for ann in train_gt["annotations"]:
        if ann["image_id"] not in image_id_to_annotations:
            image_id_to_annotations[ann["image_id"]] = []
        image_id_to_annotations[ann["image_id"]].append(ann)
    
    images, annotations = [], []
    for image_id, anns in tqdm(image_id_to_annotations.items()):
        filename = image_id_to_name[image_id]
        name, ext = filename.split(".")
        path = os.path.join(C.TRAIN_IMAGES, filename)
        image = cv2.imread(path)

        for ann in anns:
            if ann["iscrowd"]:
                continue
            x, y, w, h = map(int, ann["bbox"])
            crop = image[y:y+h, x:x+w]

            if w * h == 0:
                print("Empty bbox at", filename, ann["id"])
                continue
            
            image_calibrated, roi_calibrated = utils.crop_calbrated(image, [x, y, w, h], camera_matrix)
            x, y, w, h = map(int, roi_calibrated)
            crop_calibrated = image_calibrated[y:y+h, x:x+w]
            
            filename_crop = name + "_" + str(ann["id"]) + "." + ext
            path_crop = os.path.join(C.TRAIN_CROPS_ORIGIN, filename_crop)
            path_crop_calibrated = os.path.join(C.TRAIN_CROPS_CALIBRATED, filename_crop)

            yaw, pitch, roll = ann["orientation"]
            yaw, pitch, roll = -pitch, -yaw, -roll

            _, Mx, My = utils.calibration_matrix([x, y, w, h], camera_matrix)
            rot1 = R.from_euler("YXZ", [yaw, pitch, roll])
            rot2 = R.from_dcm(Mx.dot(My.T).T)
            yaw_rel, pitch_rel, roll_rel = (rot2 * rot1).as_euler("YXZ")

            annotations.append(dict(
                id=ann["id"],
                image_id=len(images),
                category_id=ann["category_id"],
                bbox=ann["bbox"],
                position=ann["position"],
                orientation=[yaw, pitch, roll],
                orientation_relative=[yaw_rel, pitch_rel, roll_rel]
            ))
            images.append(dict(id=len(images), file_name=filename_crop, width=w, height=h))

            cv2.imwrite(path_crop, crop)
            cv2.imwrite(path_crop_calibrated, crop_calibrated)

    train_gt = dict(images=images, annotations=annotations, categories=train_gt["categories"])
    with open(C.TRAIN_CROPS_JSON, "w") as f:
        json.dump(train_gt, f)


if __name__ == "__main__":
    main()
