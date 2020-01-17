import os
import sys
import logging

logger = logging.getLogger('logger')
stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stream_handler)
logger.setLevel(logging.DEBUG)

DATA_BASEDIR = '/datasets/pku-autonomous-driving'

CAMERA_BASEDIR = os.path.join(DATA_BASEDIR, "camera")
CAMERA_INTRINSIC = os.path.join(CAMERA_BASEDIR, "camera_intrinsic.txt")

CAR_MODELS = os.path.join(DATA_BASEDIR, "car_models")
CAR_MODELS_JSON = os.path.join(DATA_BASEDIR, "car_models_json")

TRAIN_IMAGES = os.path.join(DATA_BASEDIR, "train_images")
TRAIN_CSV = os.path.join(DATA_BASEDIR, "train.csv")

TEST_IMAGES = os.path.join(DATA_BASEDIR, "test_images")

JSON_DIR = os.path.join(DATA_BASEDIR, "annotations", "json")

TRAIN_OBJECTS_BBOX_JSON = os.path.join(JSON_DIR, "train_objects_bbox.json")
TRAIN_OBJECTS_SEGM_JSON = os.path.join(JSON_DIR, "train_objects_segm.json")
TRAIN_OBJECTS_BOTH_JSON = os.path.join(JSON_DIR, "train_objects_both.json")

TRAIN_OBJECTS_BBOX_SINGLE_CLASS_JSON = os.path.join(JSON_DIR, "train_objects_bbox_single_class.json")
TRAIN_OBJECTS_SEGM_SINGLE_CLASS_JSON = os.path.join(JSON_DIR, "train_objects_segm_single_class.json")
TRAIN_OBJECTS_BOTH_SINGLE_CLASS_JSON = os.path.join(JSON_DIR, "train_objects_both_single_class.json")

# TEST_OBJECTS_IMAGE_INFO_JSON = os.path.join(JSON_DIR, "test_objects_cars.json")
# TEST_OBJECTS_SINGLE_CLASS_IMAGE_INFO_JSON = os.path.join(JSON_DIR, "test_objects_single_class_cars.json")

N_FOLDS = 5
CV_DIR = os.path.join(JSON_DIR, "cv")
RANDOM_STATE = 42
ANNOTATIONS_JSON = [
    TRAIN_OBJECTS_BBOX_JSON, TRAIN_OBJECTS_SEGM_JSON, TRAIN_OBJECTS_BOTH_JSON,
    TRAIN_OBJECTS_BBOX_SINGLE_CLASS_JSON, TRAIN_OBJECTS_SEGM_SINGLE_CLASS_JSON, TRAIN_OBJECTS_BOTH_SINGLE_CLASS_JSON
]

TRAIN_IGNORE_MASKS = os.path.join(DATA_BASEDIR, "train_masks")
TEST_IGNORE_MASKS = os.path.join(DATA_BASEDIR, "test_masks")

TRAIN_KEYPOINTS_JSON = os.path.join(JSON_DIR, "train_keypoints.json")
TEST_KEYPOINTS_JSON = os.path.join(JSON_DIR, "test_keypoints.json")

TRAIN_CROPS_DIR = os.path.join(DATA_BASEDIR, "train_crops")
TRAIN_CROPS_JSON = os.path.join(TRAIN_CROPS_DIR, "annotations.json")
TRAIN_CROPS_ORIGIN = os.path.join(TRAIN_CROPS_DIR, "origin")
TRAIN_CROPS_CALIBRATED = os.path.join(TRAIN_CROPS_DIR, "calibrated")

NOISY_IMAGES = "noisy_images.csv"

WORKDIR = "work_dir"