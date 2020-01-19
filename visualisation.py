import cv2
import os
from math import sin, cos
import numpy as np
import config as cfg
from tqdm import tqdm


def draw_line(image, points):
    color = (255, 0, 0)
    cv2.line(image, tuple(points[1][:2]), tuple(points[2][:2]), color, 2)
    cv2.line(image, tuple(points[1][:2]), tuple(points[4][:2]), color, 2)

    cv2.line(image, tuple(points[1][:2]), tuple(points[5][:2]), color, 2)
    cv2.line(image, tuple(points[2][:2]), tuple(points[3][:2]), color, 2)
    cv2.line(image, tuple(points[2][:2]), tuple(points[6][:2]), color, 2)
    cv2.line(image, tuple(points[3][:2]), tuple(points[4][:2]), color, 2)
    cv2.line(image, tuple(points[3][:2]), tuple(points[7][:2]), color, 2)

    cv2.line(image, tuple(points[4][:2]), tuple(points[8][:2]), color, 2)
    cv2.line(image, tuple(points[5][:2]), tuple(points[8][:2]), color, 2)

    cv2.line(image, tuple(points[5][:2]), tuple(points[6][:2]), color, 2)
    cv2.line(image, tuple(points[6][:2]), tuple(points[7][:2]), color, 2)
    cv2.line(image, tuple(points[7][:2]), tuple(points[8][:2]), color, 2)
    return image


def euler_to_Rot(yaw, pitch, roll):
    Y = np.array([[cos(yaw), 0, sin(yaw)],
                [0, 1, 0],
                [-sin(yaw), 0, cos(yaw)]])
    P = np.array([[1, 0, 0],
                [0, cos(pitch), -sin(pitch)],
                [0, sin(pitch), cos(pitch)]])
    R = np.array([[cos(roll), -sin(roll), 0],
                [sin(roll), cos(roll), 0],
                [0, 0, 1]])
    return np.dot(Y, np.dot(P, R))


def draw_points(image, points, confidence):
    image = np.array(image)
    for (p_x, p_y, p_z) in points:
        cv2.circle(image, (p_x, p_y), 5, (255, 0, 0), -1)
    cv2.putText(image,str(confidence), tuple(points[0][:-1]), 0, 1, (255,255,255), 1)
    return image


def visualize(submission):
    # You will also need functions from the previous cells
    x_l = 1.02
    y_l = 0.80
    z_l = 2.31
    k = np.array([[2304.5479, 0,  1686.2379],
           [0, 2305.8757, 1354.9849],
           [0, 0, 1]], dtype=np.float32)

    lines =[]
    with open(submission,'r') as s:
        for line in s:
            lines.append(line)
    lines.pop(0)

    for line in tqdm(lines):
        name = line.split(',')[0]
        all_coords = np.array(line.split(',')[1].split(' ')[:-1],dtype = np.float32)
        #print(all_coords.shape[0]//7)
        num_of_cars = all_coords.shape[0]//7
        coords = []
        #print(os.path.join(cfg.TEST_IMAGES,name+'.jpg'))

        for i in range(num_of_cars):
            coords.append({'pitch':all_coords[0+i*7], 'yaw':all_coords[1+i*7], 'roll':all_coords[2+i*7], 'x':all_coords[3+i*7], 'y':all_coords[4+i*7], 'z':all_coords[5+i*7],'score':all_coords[6+7*i]})
        
        img = cv2.imread(os.path.join(cfg.TRAIN_IMAGES,name+'.jpg'))
        img = img.copy()

        for point in coords:
            # Get values
            x, y, z = point['x'], point['y'], point['z']
            yaw, pitch, roll = -point['pitch'], -point['yaw'], -point['roll']
            # Math
            
            Rt = np.eye(4)
            t = np.array([x, y, z])
            Rt[:3, 3] = t
            Rt[:3, :3] = euler_to_Rot(yaw, pitch, roll).T
            Rt = Rt[:3, :]
            P = np.array([[0, 0, 0, 1],
                        [x_l, y_l, -z_l, 1],
                        [x_l, y_l, z_l, 1],
                        [-x_l, y_l, z_l, 1],
                        [-x_l, y_l, -z_l, 1],
                        [x_l, -y_l, -z_l, 1],
                        [x_l, -y_l, z_l, 1],
                        [-x_l, -y_l, z_l, 1],
                        [-x_l, -y_l, -z_l, 1]]).T
            img_cor_points = np.dot(k, np.dot(Rt, P))
            img_cor_points = img_cor_points.T
            img_cor_points[:, 0] /= img_cor_points[:, 2]
            img_cor_points[:, 1] /= img_cor_points[:, 2]
            # call this function before chage the dtype
            img_cor_points = img_cor_points.astype(int)
            img = draw_points(img, img_cor_points,point['score'])
            img = draw_line(img, img_cor_points)
            
    
        cv2.imwrite(os.path.join('./test_image',name+'.jpg'),img)