from math import sqrt, acos, pi, sin, cos
from scipy.spatial.transform import Rotation as R
from sklearn.metrics import average_precision_score
from operator import itemgetter
from utils import rotation_to_quaternion
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
 

def TranslationDistance(pcar, gcar):
    p_position = np.array(pcar['position']).reshape((3,1))
    g_position = np.array(gcar['position']).reshape((3,1))
    #sprint(np.sqrt(np.sum((p_position-g_position)**2)))
    return np.sqrt(np.sum((p_position-g_position)**2))


def RotationDistance(p, g):
    p = p['orientation']
    g = g['orientation']
    q1 = R.from_euler("YXZ", p)
    q2 = R.from_euler("YXZ", g)
    diff = R.inv(q2) * q1
    W = np.clip(diff.as_quat()[-1], -1., 1.)

    # ask how to choose distance? This seems valid? 


    # in the official metrics code:
    # https://www.kaggle.com/c/pku-autonomous-driving/overview/evaluation
    #   return Object3D.RadianToDegree( Math.Acos(diff.W) )
    # this code treat θ and θ+2π differntly.
    # So this should be fixed as follows.


    W = (acos(W)*360)/pi
    if W > 180:
        W = 180 - W
    return W



def check_match(idx1, gt_dicts, predicted_dicts, keep_gt=False):# thre_tr_dist, thre_ro_dist, keep_gt = False):

    thres_tr_list = [0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]
    thres_ro_list = [50, 45, 40, 35, 30, 25, 20, 15, 10, 5]

    thre_tr_dist = thres_tr_list[idx1]
    thre_ro_dist = thres_ro_list[idx1]
    result_flg = [] # 1 for TP, 0 for FP
    scores = []
    MAX_VAL = 10**10
    imgs = set([p['image_id'] for p in predicted_dicts])
    for img_id in imgs:
        for pcar in sorted([p for p in predicted_dicts if p['image_id'] == img_id], key=itemgetter('score'), reverse=True):
            # find nearest GT
            min_tr_dist = MAX_VAL
            min_idx = -1
            for idx, gcar in enumerate([g for g in gt_dicts if g['image_id'] == img_id]):
                tr_dist = TranslationDistance(pcar,gcar)
                if tr_dist < min_tr_dist:
                    min_tr_dist = tr_dist
                    min_ro_dist = RotationDistance(pcar, gcar)
                    min_idx = idx
                    
            # set the result

            # ask about keeping gt

            if min_tr_dist < thre_tr_dist and min_ro_dist < thre_ro_dist:
                if not keep_gt:
                    [g for g in gt_dicts if g['image_id'] == img_id ].pop(min_idx)
                result_flg.append(1)
            else:
                result_flg.append(0)
            scores.append(pcar['score'])
    
    return result_flg, scores



def calc_map(gt_dicts, predicted_dicts, nrows=None):
    # dicts structure: [{'translation': , 'orientation': ,'score': ,'imag_id': },{},{}]

# add info from dicts

    max_workers = 10
    n_gt = len(gt_dicts)
    ap_list = []
    iterables = []
    for i in range(10):
        iterables.append((i,gt_dicts,predicted_dicts))
    p = Pool(processes=max_workers)
    for result_flg, scores in p.starmap(check_match, iterables, chunksize=1):
        if np.sum(result_flg) > 0:
            n_tp = np.sum(result_flg)
            recall = n_tp/n_gt
            ap = average_precision_score(result_flg, scores)*recall
        else:
            ap = 0
        ap_list.append(ap)
    return np.mean(ap_list)
