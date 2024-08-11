import open3d
import torch.utils.data as data
import random
import numbers
import os
import os.path
import numpy as np
import struct
import math
import torch
import torchvision
import cv2
from PIL import Image
from torchvision import transforms
import pickle
from pyquaternion import Quaternion
from multiprocessing import Process

import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from ..options import Options_Nuscenes
from data.kitti_helper import FarthestSampler, camera_matrix_cropping, camera_matrix_scaling, projection_pc_img

import pickle
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.nuscenes import NuScenes
from tqdm import tqdm
from pathlib import Path

camera_name_list = ['CAM_FRONT',
                         #     'CAM_FRONT_LEFT',
                         #     'CAM_FRONT_RIGHT',
                         #     'CAM_BACK',
                         #     'CAM_BACK_LEFT',
                         #     'CAM_BACK_RIGHT'
                        ]


def downsample_with_reflectance(pointcloud, reflectance, voxel_grid_downsample_size):
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(np.transpose(pointcloud[0:3, :]))
    reflectance_max = np.max(reflectance)

    fake_colors = np.zeros((pointcloud.shape[1], 3))
    fake_colors[:, 0] = reflectance / reflectance_max
    pcd.colors = open3d.utility.Vector3dVector(fake_colors)
    down_pcd = pcd.voxel_down_sample(voxel_size=voxel_grid_downsample_size)
    down_pcd_points = np.transpose(np.asarray(down_pcd.points))  # 3xN
    pointcloud = down_pcd_points
    reflectance = np.asarray(down_pcd.colors)[:, 0] * reflectance_max

    return pointcloud, reflectance



def load_dataset_info(filepath):
    with open(filepath, 'rb') as f:
        dataset_read = pickle.load(f)
    return dataset_read

def make_nuscenes_dataset(root_path):
    dataset = load_dataset_info(os.path.join(root_path, 'dataset_info.list'))
    return dataset


def get_sample_data_ego_pose_P(nusc, sample_data):
    sample_data_pose = nusc.get('ego_pose', sample_data['ego_pose_token'])
    sample_data_pose_R = np.asarray(Quaternion(sample_data_pose['rotation']).rotation_matrix).astype(np.float32)
    sample_data_pose_t = np.asarray(sample_data_pose['translation']).astype(np.float32)
    sample_data_pose_P = get_P_from_Rt(sample_data_pose_R, sample_data_pose_t)
    return sample_data_pose_P


def get_calibration_P(nusc, sample_data):
    calib = nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
    R = np.asarray(Quaternion(calib['rotation']).rotation_matrix).astype(np.float32)
    t = np.asarray(calib['translation']).astype(np.float32)
    P = get_P_from_Rt(R, t)
    return P


def get_P_from_Rt(R, t):
    P = np.identity(4)
    P[0:3, 0:3] = R
    P[0:3, 3] = t
    return P


def get_camera_K(nusc, camera):
    calib = nusc.get('calibrated_sensor', camera['calibrated_sensor_token'])
    return np.asarray(calib['camera_intrinsic']).astype(np.float32)


def transform_pc_np(P, pc_np):
    """

    :param pc_np: 3xN
    :param P: 4x4
    :return:
    """
    pc_homo_np = np.concatenate((pc_np,
                                 np.ones((1, pc_np.shape[1]), dtype=pc_np.dtype)),
                                axis=0)
    P_pc_homo_np = np.dot(P, pc_homo_np)
    return P_pc_homo_np[0:3, :]

def get_lidar_pc_intensity_by_token(nusc,lidar_token):
    lidar = nusc.get('sample_data', lidar_token)
    pc = LidarPointCloud.from_file(os.path.join(nusc.dataroot, lidar['filename']))
    pc_np = pc.points[0:3, :]
    intensity_np = pc.points[3:4, :]

    # remove point falls on egocar
    x_inside = np.logical_and(pc_np[0, :] < 0.8, pc_np[0, :] > -0.8)
    y_inside = np.logical_and(pc_np[1, :] < 2.7, pc_np[1, :] > -2.7)
    inside_mask = np.logical_and(x_inside, y_inside)
    outside_mask = np.logical_not(inside_mask)
    pc_np = pc_np[:, outside_mask]
    intensity_np = intensity_np[:, outside_mask]

    P_oi = get_sample_data_ego_pose_P(nusc, lidar)

    return pc_np, intensity_np, P_oi

def lidar_frame_accumulation(nusc,opt, lidar, P_io, P_lidar_vehicle, P_vehicle_lidar,
                                 direction,
                                 pc_np_list, intensity_np_list):
    counter = 1
    accumulated_counter = 0
    while accumulated_counter < opt.accumulation_frame_num:
        if lidar[direction] == '':
            break

        if counter % opt.accumulation_frame_skip != 0:
            counter += 1
            lidar = nusc.get('sample_data', lidar[direction])
            continue
        # P_vehicle_lidar = get_calibration_P(nusc, lidar[direction])
        pc_np_j, intensity_np_j, P_oj = get_lidar_pc_intensity_by_token(nusc,lidar[direction])
        P_ij = np.dot(P_io, P_oj)
        P_ij_trans = np.dot(np.dot(P_lidar_vehicle, P_ij), P_vehicle_lidar)
        pc_np_j_transformed = transform_pc_np(P_ij_trans, pc_np_j)
        pc_np_list.append(pc_np_j_transformed)
        intensity_np_list.append(intensity_np_j)

        counter += 1
        lidar = nusc.get('sample_data', lidar[direction])
        accumulated_counter += 1

    # print('accumulation %s %d' % (direction, counter))
    return pc_np_list, intensity_np_list


def accumulate_lidar_points(nusc,lidar):
    opt = Options_Nuscenes()
    pc_np_list = []
    intensity_np_list = []
    # load itself
    pc_np_i, intensity_np_i, P_oi = get_lidar_pc_intensity_by_token(nusc,lidar['token'])
    pc_np_list.append(pc_np_i)
    intensity_np_list.append(intensity_np_i)
    P_io = np.linalg.inv(P_oi)

    P_vehicle_lidar = get_calibration_P(nusc, lidar)
    P_lidar_vehicle = np.linalg.inv(P_vehicle_lidar)

    # load next
    pc_np_list, intensity_np_list = lidar_frame_accumulation(nusc,opt,lidar, P_io, P_lidar_vehicle, P_vehicle_lidar,
                                                                      'next',
                                                                      pc_np_list, intensity_np_list)

    # load prev
    pc_np_list, intensity_np_list = lidar_frame_accumulation(nusc,opt,lidar, P_io, P_lidar_vehicle, P_vehicle_lidar,
                                                                      'prev',
                                                                      pc_np_list, intensity_np_list)

    pc_np = np.concatenate(pc_np_list, axis=1)
    intensity_np = np.concatenate(intensity_np_list, axis=1)

    pc_np_curr = pc_np_list[0]
    intensity_np_curr = intensity_np_list[0]

    return pc_np, intensity_np, pc_np_curr,intensity_np_curr

def downsample_np(pc_np, intensity_np, k):
    '''if pc_np.shape[1] >= k:
        choice_idx = np.random.choice(pc_np.shape[1], k, replace=False)
    else:
        fix_idx = np.asarray(range(pc_np.shape[1]))
        while pc_np.shape[1] + fix_idx.shape[0] < k:
            fix_idx = np.concatenate((fix_idx, np.asarray(range(pc_np.shape[1]))), axis=0)
        random_idx = np.random.choice(pc_np.shape[1], k - fix_idx.shape[0], replace=False)
        choice_idx = np.concatenate((fix_idx, random_idx), axis=0)'''
    choice_idx = np.random.choice(pc_np.shape[1], k, replace=False)
    pc_np = pc_np[:, choice_idx]
    intensity_np = intensity_np[:, choice_idx]

    return pc_np, intensity_np

def make_dataset(seq,nusc,dataset,ids,save_img_dir,save_pc_dir,save_ext_dir,save_int_dir):

    for idx_seq in range(len(dataset)):
        item = dataset[idx_seq] # data
        idx_global  = ids[idx_seq] # global index
        print('seq %d : id %d/%d' % (seq,idx_seq,len(dataset)))
        # if os.path.exists(save_pc_dir / ("%06d.npy"%idx_global)):
        #     continue
        
        # dict_item = {}
        lidar_token=item[0]
        nearby_cam_token_dict=item[1]

        lidar=nusc.get('sample_data',lidar_token)
        # accumulate next/behind frame
        pc_np, intensity_np,_,_ = accumulate_lidar_points(nusc,lidar)
        if pc_np.shape[1] < 45000:
            print("seq %d: id %d/%d, num_pc=%d, < 45000!"%(seq,idx_seq,len(dataset),pc_np.shape[1]))
            continue
        # downsample
        # if pc_np.shape[1] > 2 * opt.input_pt_num:
        else:
            pc_np,intensity_np=downsample_with_reflectance(pc_np,intensity_np[0],voxel_grid_downsample_size=0.2)
            intensity_np=np.expand_dims(intensity_np,axis=0)
            pc_np=pc_np.astype(np.float32)
            intensity_np=intensity_np.astype(np.float32)
            if pc_np.shape[1] < 45000:
                print("seq %d: id %d/%d, num_pc=%d, < 45000 after downsample!"%(seq,idx_seq,len(dataset),pc_np.shape[1]))
                continue
        # random sampling
        # pc_np, intensity_np = downsample_np(pc_np, intensity_np, opt.input_pt_num)
        
        # dict_pc = {}
        # dict_pc["token"] = lidar_token
        # dict_pc["point"] = pc_np
        # dict_pc["intensity"] = intensity_np

        lidar_calib_P = get_calibration_P(nusc, lidar)
        lidar_pose_P = get_sample_data_ego_pose_P(nusc, lidar)

        # load image, random select one image in the nearby list
        # camera_name = 'CAM_FRONT'
        # dict_camera_multi = []
        # for camera_name in camera_name_list:
        nearby_cam_token_list = nearby_cam_token_dict[camera_name_list[0]]
        for nearby_camera_token in nearby_cam_token_list:
            camera = nusc.get('sample_data', nearby_camera_token)
            img = np.array(Image.open(os.path.join(nusc.dataroot, camera['filename'])))
            K = get_camera_K(nusc, camera)

            # crop top 100 rows
            img = img[opt.crop_original_top_rows:, :, :]
            K = camera_matrix_cropping(K, dx=0, dy=opt.crop_original_top_rows)
            # scale
            img = cv2.resize(img,
                         (int(round(img.shape[1] * opt.img_scale)),
                          int(round((img.shape[0] * opt.img_scale)))),
                         interpolation=cv2.INTER_LINEAR)
            K = camera_matrix_scaling(K, opt.img_scale)

            camera_calib_P = get_calibration_P(nusc, camera)
            camera_pose_P = get_sample_data_ego_pose_P(nusc, camera)

            camera_pose_P_inv = np.linalg.inv(camera_pose_P)
            camera_calib_P_inv = np.linalg.inv(camera_calib_P)
            P_cam_pc = np.dot(camera_calib_P_inv, np.dot(camera_pose_P_inv,
                                                     np.dot(lidar_pose_P, lidar_calib_P))) # from lidar coors --> camera coors

            pc_np_ = np.dot(P_cam_pc[0:3, 0:3], pc_np) + P_cam_pc[0:3, 3:]

            # pc_np_down,intensity_np_down=downsample_np(pc_np,intensity_np,opt.input_pt_num)


            H = img.shape[0]
            W = img.shape[1]
            uvz = np.dot(K, pc_np_)
            depth = uvz[2, :]
            uv = uvz[0:2, :] / uvz[2:, :]
            inside_point_num = np.sum((depth > 0) & (uv[0, :] >= 0) & (uv[0, :] <= W - 1) & (uv[1, :] >= 0) & (uv[1, :] <= H - 1))
            
            # dict_camera = {}
            # dict_camera["token"] = nearby_camera_token
            # dict_camera["K"] = K
            # dict_camera["P"] = P_cam_pc
            # dict_camera["inside"] = inside_point_num
            # dict_camera["img"] = img

            # dict_camera_multi.append(dict_camera)

            if inside_point_num > 6000:
                # save pc
                pc_intensity_np = np.concatenate([pc_np_,intensity_np],axis = 0)
                np.save(save_pc_dir / ("%06d.npy"%idx_global),pc_intensity_np)
                # save img
                np.save(save_img_dir / ("%06d.npy"%idx_global),img)
                # save ext file
                np.save(save_ext_dir / ("%06d.npy"%idx_global),P_cam_pc)
                # save int file
                np.save(save_int_dir / ("%06d.npy"%idx_global),K)
                
                break


def make_dataset_multiprocess(root,output_path,mode,opt,thread_num = 20):
    
    nuscenes_path = Path(root) /  mode
    version = 'v1.0-' + mode

    save_dir= Path(output_path) / mode
    if os.path.exists(save_dir) == False:
        os.makedirs(save_dir)

    # save_img_dir
    save_img_dir = save_dir / "img"
    save_pc_dir = save_dir / "pc"
    save_ext_dir = save_dir / "ext"
    save_int_dir = save_dir / "int"

    if os.path.exists(save_img_dir) == False:
        os.mkdir(save_img_dir)
    if os.path.exists(save_pc_dir) == False:
        os.mkdir(save_pc_dir)
    if os.path.exists(save_ext_dir) == False:
        os.mkdir(save_ext_dir)
    if os.path.exists(save_int_dir) == False:
        os.mkdir(save_int_dir)

    dataset = make_nuscenes_dataset(nuscenes_path)
    nusc = NuScenes(version=version, dataroot=nuscenes_path, verbose=True)

    # valid_count = 0
    avg_length = len(dataset) // thread_num
    remainder = len(dataset) % thread_num
    start = 0
    

    threads_list = []
    for i in range(thread_num):
        end = start + avg_length + (1 if i < remainder else 0)
        threads_list.append(Process(target = make_dataset,
                                        args = (i,
                                                nusc,
                                                dataset[start:end],
                                                [idx for idx in range(start,end)],
                                                save_img_dir,
                                                save_pc_dir,
                                                save_ext_dir,
                                                save_int_dir
                                                )
                                         ))
        start = end
    
    for thread in threads_list:
        thread.start()

    for thread in threads_list:
        thread.join()
                

        



if __name__=='__main__':
    opt = Options_Nuscenes()
    root_data = '/data/nuscenes'
    mode = "trainval" # trainval for training set, test for validation set
    make_dataset_multiprocess(root_data, '/data/nuscenes_i2pnew',mode, opt,thread_num = 20)