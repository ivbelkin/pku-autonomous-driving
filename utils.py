import os
import re
import numpy as np
from math import sin, cos
import json
import pandas as pd
import cv2
from PIL import Image
from scipy.spatial.transform import Rotation as R

import config as C
from car_models import car_id2name, car_name2id


def parse_camera_intrinsic():
    if not hasattr(parse_camera_intrinsic, "cache"):
        params = {}
        with open(C.CAMERA_INTRINSIC, "r") as f:
            for line in f:
                line = line.strip()
                m = re.match(r'(\w+) = ([\d\.]+);', line)
                key = m.group(1)
                value = m.group(2)
                params[key] = float(value)
        parse_camera_intrinsic.cache = params
    return parse_camera_intrinsic.cache


def get_camera_matrix():
    p = parse_camera_intrinsic()
    return np.array([
            [p["fx"],       0, p["cx"]],
            [      0, p["fy"], p["cy"]],
            [      0,       0,       1]
        ], dtype=np.float32)


def euler_to_Rot(yaw, pitch, roll):
    Y = np.array([
        [ cos(yaw), 0, sin(yaw)],
        [        0, 1,        0],
        [-sin(yaw), 0, cos(yaw)]
    ])
    P = np.array([
        [1,          0,           0],
        [0, cos(pitch), -sin(pitch)],
        [0, sin(pitch),  cos(pitch)]
    ])
    R = np.array([
        [cos(roll), -sin(roll), 0],
        [sin(roll),  cos(roll), 0],
        [        0,          0, 1]
    ])
    return np.dot(Y, np.dot(P, R))


def Rot_to_yaw(Y):
    cos = Y[0, 0]
    sin = Y[0, 2]
    return np.arctan2(sin, cos)


def Rot_to_pitch(P):
    cos = P[1, 1]
    sin = P[2, 1]
    return np.arctan2(sin, cos)


def Rot_to_roll(R):
    cos = R[0, 0]
    sin = R[1, 0]
    return np.arctan2(sin, cos)


def load_car_models():
    car_id2model = {}
    for id_, name in car_id2name.items():
        path = os.path.join(C.CAR_MODELS_JSON, name + ".json")
        with open(path) as json_file:
            car_id2model[id_] = json.load(json_file)
    
    car_id2vertices = {}
    car_id2triangles = {}
    for id_, model in car_id2model.items():
        vertices = np.array(model['vertices'])
        vertices[:, 1] = -vertices[:, 1]
        triangles = np.array(model['faces']) - 1
        
        car_id2vertices[id_] = vertices
        car_id2triangles[id_] = triangles
    
    return car_id2vertices, car_id2triangles


def obj_to_bbox(vertices, triangles):
    xtl, ytl, xbr, ybr = np.inf, np.inf, 0, 0
    for t in triangles:
        coord = np.array([vertices[t[0]][:2], vertices[t[1]][:2], vertices[t[2]][:2]], dtype=np.int32)
        xtl = min(xtl, np.min(coord[:, 0]))
        xbr = max(xbr, np.max(coord[:, 0]))
        ytl = min(ytl, np.min(coord[:, 1]))
        ybr = max(ybr, np.max(coord[:, 1]))
    return xtl, ytl, xbr, ybr


def load_train_annotations():
    train = pd.read_csv(C.TRAIN_CSV)
    
    def parse(x):
        x = x.split()
        model_types, yaws, pitches, rolls, xs, ys, zs = [x[i::7] for i in range(7)]
        model_types = list(map(int, model_types))
        yaws = list(map(float, yaws))
        pitches = list(map(float, pitches))
        rolls = list(map(float, rolls))
        xs = list(map(float, xs))
        ys = list(map(float, ys))
        zs = list(map(float, zs))
        return dict(
            model_types=model_types, 
            yaws=yaws, pitches=pitches, rolls=rolls, 
            xs=xs, ys=ys, zs=zs
        )
    
    train["parsed"] = train['PredictionString'].map(parse)
    train = train.drop("PredictionString", axis=1).set_index("ImageId")

    return train.to_dict()["parsed"]


def project_vertices(vertices, yaw, pitch, roll, x, y, z):
    k = get_camera_matrix()
    yaw, pitch, roll = -pitch, -yaw, -roll
    Rt = np.eye(4)
    t = np.array([x, y, z])
    Rt[:3, 3] = t
    Rt[:3, :3] = euler_to_Rot(yaw, pitch, roll).T
    Rt = Rt[:3, :]
    P = np.ones((vertices.shape[0],vertices.shape[1]+1))
    P[:, :-1] = vertices
    P = P.T
    img_cor_points = np.dot(k, np.dot(Rt, P))
    img_cor_points = img_cor_points.T
    img_cor_points[:, 0] /= img_cor_points[:, 2]
    img_cor_points[:, 1] /= img_cor_points[:, 2]
    return img_cor_points


def calibration_matrix(roi, camera_matrix):
    inv_camera_matrix = np.linalg.inv(camera_matrix)
    x, y, w, h = roi
    cx, cy = x + w / 2, y + h / 2
    v2 = np.array([cx, cy, 1])
    v3 = inv_camera_matrix.dot(v2)
    tg_ay, tg_ax, _ = v3
    ax = np.arctan(tg_ax)
    ay = -np.arctan(tg_ay)
    Mx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
    My = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
    M = camera_matrix.dot(Mx).dot(My).dot(inv_camera_matrix)
    return M, Mx, My


def calibrate_roi(roi, M):
    x, y, w, h = roi
    P = np.array([
        [x, y, 1],
        [x + w, y, 1],
        [x + w, y + h, 1],
        [x, y + h, 1]
    ])
    P_new = M.dot(P.T).T
    P_new /= P_new[:, 2:]
    x_new = P_new[:, 0].min()
    y_new = P_new[:, 1].min()
    w_new = P_new[:, 0].max() - x_new
    h_new = P_new[:, 1].max() - y_new
    return [x_new, y_new, w_new, h_new]


def crop_calbrated(image, roi, camera_matrix):
    H, W = image.shape[:2]
    M, _, _ = calibration_matrix(roi, camera_matrix)
    image_new = cv2.warpPerspective(image, M, (W, H))
    roi_new = calibrate_roi(roi, M)
    return image_new, roi_new


def quaternion_to_euler_angle(q):

    """Convert quaternion to euler angel.
    Input:
        q: 1 * 4 vector,
    Output:
        angle: 1 x 3 vector, each row is [roll, pitch, yaw]
    """
    w, x, y, z = q
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    X = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    Z = math.atan2(t3, t4)

    return X, Y, Z


def euler_angles_to_quaternions(angle): 
    """Convert euler angels to quaternions representation. 
    Input: 
        angle: n x 3 matrix, each row is [roll, pitch, yaw] 
    Output: 
        q: n x 4 matrix, each row is corresponding quaternion. 
    """

    in_dim = np.ndim(angle)
    if in_dim == 1:
        angle = angle[None, :]

    n = angle.shape[0]
    roll, pitch, yaw = angle[:, 0], angle[:, 1], angle[:, 2]
    q = np.zeros((n, 4))

    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)

    q[:, 0] = cy * cr * cp + sy * sr * sp
    q[:, 1] = cy * sr * cp - sy * cr * sp
    q[:, 2] = cy * cr * sp + sy * sr * cp
    q[:, 3] = sy * cr * cp - cy * sr * sp

    return q


def rotation_to_quaternion(rotation):
    rotation = R.from_euler("YXZ", rotation)
    q = rotation.as_quat()
    return q


def quaternion_to_rotation(q):
    rotation = R.from_quat(q)
    ypr = rotation.as_euler("YXZ")
    return ypr


def fit_image(image, max_size):
    w, h = image.size
    scale = max_size / max(w, h)
    w_new, h_new = int(w * scale), int(h * scale)
    image = image.resize((w_new, h_new), Image.LANCZOS)
    new_image = Image.new("RGB", (max_size, max_size))
    new_image.paste(image, ((max_size - w_new) // 2, (max_size - h_new) // 2))
    return new_image


def smooth(beta, array):
    s = []
    s.append(array[0])
    for i in range(1, len(array)):
        s.append((1 - beta) * s[-1] + beta * array[i])
    return s


def orient_quaternion(q):
    e = np.array([0, 0, 1])
    q[1:] = q[1:] * (np.dot(e, q[1:]) / (np.abs(np.dot(e, q[1:]))))
    return q
