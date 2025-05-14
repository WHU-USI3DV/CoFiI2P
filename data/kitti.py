import os
import torch
import torch.utils.data as data
from torchvision import transforms
import numpy as np
from PIL import Image
import random
import math
import open3d as o3d
import cv2
import struct
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.sparse import coo_matrix
import time
from pathlib import Path

from model.kpconv.preprocess_data import precompute_point_cloud_stack_mode, precompute_point_cloud_cuda
from model.network import point2node
# from ...model.kpconv.kp_backbone import KPConvFPN

class KittiCalibHelper:
    def __init__(self, root_path):
        self.root_path = root_path
        self.calib_matrix_dict = self.read_calib_files()

    def read_calib_files(self):
        seq_folders = [name for name in os.listdir(
            os.path.join(self.root_path, 'calib'))]
        calib_matrix_dict = {}
        for seq in seq_folders:
            calib_file_path = os.path.join(
                self.root_path, 'calib', seq, 'calib.txt')
            with open(calib_file_path, 'r') as f:
                for line in f.readlines():
                    seq_int = int(seq)
                    if calib_matrix_dict.get(seq_int) is None:
                        calib_matrix_dict[seq_int] = {}

                    key = line[0:2]
                    mat = np.fromstring(line[4:], sep=' ').reshape(
                        (3, 4)).astype(np.float32)
                    if 'Tr' == key:
                        P = np.identity(4,dtype = np.float32)
                        P[0:3, :] = mat
                        calib_matrix_dict[seq_int][key] = P
                    else:
                        K = mat[0:3, 0:3]
                        calib_matrix_dict[seq_int][key + '_K'] = K
                        fx = K[0, 0]
                        fy = K[1, 1]
                        cx = K[0, 2]
                        cy = K[1, 2]
                        # mat[0, 3] = fx*tx + cx*tz
                        # mat[1, 3] = fy*ty + cy*tz
                        # mat[2, 3] = tz
                        tz = mat[2, 3]
                        tx = (mat[0, 3] - cx * tz) / fx
                        ty = (mat[1, 3] - cy * tz) / fy
                        P = np.identity(4,dtype = np.float32)
                        P[0:3, 3] = np.asarray([tx, ty, tz],dtype = np.float32)
                        calib_matrix_dict[seq_int][key] = P
        return calib_matrix_dict

    def get_matrix(self, seq: int, matrix_key: str):
        return self.calib_matrix_dict[seq][matrix_key]

class FarthestSampler:
    def __init__(self, dim=3):
        self.dim = dim

    def calc_distances(self, p0, points):
        return ((p0 - points) ** 2).sum(axis=0)

    def sample(self, pts, k):
        farthest_pts = np.zeros((self.dim, k))
        farthest_pts_idx = np.zeros(k, dtype=np.int)
        init_idx = np.random.randint(len(pts))
        farthest_pts[:, 0] = pts[:, init_idx]
        farthest_pts_idx[0] = init_idx
        distances = self.calc_distances(farthest_pts[:, 0:1], pts)
        for i in range(1, k):
            idx = np.argmax(distances)
            farthest_pts[:, i] = pts[:, idx]
            farthest_pts_idx[i] = idx
            distances = np.minimum(distances, self.calc_distances(farthest_pts[:, i:i+1], pts))
        return farthest_pts, farthest_pts_idx


class kitti_pc_img_dataset(data.Dataset):
    def __init__(self, opt,mode, is_front=False):
        super(kitti_pc_img_dataset, self).__init__()
        for k,v in opt.__dict__.items():
            setattr(self,k,v)
        self.mode = mode
        self.dataset = self.make_kitti_dataset(self.data_path, mode)
        self.calibhelper = KittiCalibHelper(self.data_path)
        self.farthest_sampler = FarthestSampler(dim=3)
        self.is_front=is_front
        print("%s set: %d frames"%(mode,len(self.dataset)))
        print('load %s data complete'%mode)

    def read_velodyne_bin(self, path):

        pc_list = []
        with open(path, 'rb') as f:
            content = f.read()
            pc_iter = struct.iter_unpack('ffff', content)
            for idx, point in enumerate(pc_iter):
                pc_list.append([point[0], point[1], point[2], point[3]])
        return np.asarray(pc_list, dtype=np.float32).T

    def make_kitti_dataset(self, root_path, mode):
        dataset = []

        if mode == 'train':
            seq_list = list(range(9))
        elif 'val' == mode:
            seq_list = [9, 10]
        else:
            raise Exception('Invalid mode.')

        skip_start_end = 0
        for seq in seq_list:
            img2_folder = os.path.join(
                root_path, 'sequences', '%02d' % seq, 'img_P2')
            img3_folder = os.path.join(
                root_path, 'sequences', '%02d' % seq, 'img_P3')
            pc_folder = os.path.join(
                root_path, 'sequences', '%02d' % seq, 'pc_npy_with_normal')

            K2_folder = os.path.join(
                root_path, 'sequences', '%02d' % seq, 'K_P2')
            K3_folder = os.path.join(
                root_path, 'sequences', '%02d' % seq, 'K_P3')

            sample_num = round(len(os.listdir(img2_folder)))

            for i in range(skip_start_end, sample_num - skip_start_end):
                dataset.append((img2_folder, pc_folder,
                                K2_folder, seq, i, 'P2', sample_num))
                dataset.append((img3_folder, pc_folder,
                                K3_folder, seq, i, 'P3', sample_num))
                
        return dataset


    def downsample_with_intensity_sn(self, pointcloud, intensity, sn, voxel_grid_downsample_size):
        pcd=o3d.geometry.PointCloud()
        pcd.points=o3d.utility.Vector3dVector(np.transpose(pointcloud))
        intensity_max=np.max(intensity)

        fake_colors=np.zeros((pointcloud.shape[1],3))
        fake_colors[:,0:1]=np.transpose(intensity)/intensity_max

        pcd.colors=o3d.utility.Vector3dVector(fake_colors)
        pcd.normals=o3d.utility.Vector3dVector(np.transpose(sn))

        down_pcd=pcd.voxel_down_sample(voxel_size=voxel_grid_downsample_size)
        down_pcd_points=np.transpose(np.asarray(down_pcd.points,dtype = np.float32))
        pointcloud=down_pcd_points

        intensity=np.transpose(np.asarray(down_pcd.colors,dtype = np.float32)[:,0:1])*intensity_max
        sn=np.transpose(np.asarray(down_pcd.normals,dtype = np.float32))

        return pointcloud, intensity, sn

    def downsample_np(self, pc_np, intensity_np, sn_np):
        if pc_np.shape[1] >= self.num_pc:
            choice_idx = np.random.choice(pc_np.shape[1], self.num_pc, replace=False)
        else:
            fix_idx = np.asarray(range(pc_np.shape[1]))
            while pc_np.shape[1] + fix_idx.shape[0] < self.num_pc:
                fix_idx = np.concatenate((fix_idx, np.asarray(range(pc_np.shape[1]))), axis=0)
            random_idx = np.random.choice(pc_np.shape[1], self.num_pc - fix_idx.shape[0], replace=False)
            choice_idx = np.concatenate((fix_idx, random_idx), axis=0)
        pc_np = pc_np[:, choice_idx]
        intensity_np = intensity_np[:, choice_idx]
        sn_np=sn_np[:,choice_idx]
        return pc_np, intensity_np, sn_np

    def camera_matrix_cropping(self, K: np.ndarray, dx: float, dy: float):
        K_crop = np.copy(K)
        K_crop[0, 2] -= dx
        K_crop[1, 2] -= dy
        return K_crop

    def camera_matrix_scaling(self, K: np.ndarray, s: float):
        K_scale = s * K
        K_scale[2, 2] = 1
        return K_scale

    def augment_img(self, img_np):
        brightness = (0.8, 1.2)
        contrast = (0.8, 1.2)
        saturation = (0.8, 1.2)
        hue = (-0.1, 0.1)
        color_aug = transforms.ColorJitter(
            brightness, contrast, saturation, hue)
        img_color_aug_np = np.array(color_aug(Image.fromarray(img_np)))

        return img_color_aug_np

    def angles2rotation_matrix(self, angles):
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(angles[0]), -np.sin(angles[0])],
                       [0, np.sin(angles[0]), np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                       [0, 1, 0],
                       [-np.sin(angles[1]), 0, np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                       [np.sin(angles[2]), np.cos(angles[2]), 0],
                       [0, 0, 1]])
        R = np.dot(Rz, np.dot(Ry, Rx))
        return R

    def generate_random_transform(self,mode):
        """
        :param pc_np: pc in NWU coordinate
        :return:
        """
        t = [random.uniform(-self.P_tx_amplitude, self.P_tx_amplitude),
             random.uniform(-self.P_ty_amplitude, self.P_ty_amplitude),
             random.uniform(-self.P_tz_amplitude, self.P_tz_amplitude)]
        t = [tx * 0.0 for tx in t] if mode == "train" else t # discard random translation during training

        angles = [random.uniform(-self.P_Rx_amplitude, self.P_Rx_amplitude),
                  random.uniform(-self.P_Ry_amplitude, self.P_Ry_amplitude),
                  random.uniform(-self.P_Rz_amplitude, self.P_Rz_amplitude)]
        rotation_mat = self.angles2rotation_matrix(angles)
        P_random = np.identity(4, dtype=np.float32)
        P_random[0:3, 0:3] = rotation_mat
        P_random[0:3, 3] = t

        # print('t',t)
        # print('angles',angles)

        return P_random
    
    def search_point_index(self, source_points, target_points):
        '''
        source_points: [M, 3]
        target_points: [N, 3]
        '''
        indices = []
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(source_points)
        source_kdtree = o3d.geometry.KDTreeFlann(pcd)
        for i in range(target_points.shape[0]):
            [_, index, _] = source_kdtree.search_knn_vector_3d(target_points[i], 1)
        # indices = torch.nonzero(torch.isin(source_points, target_points).all(dim=1))[:, 0]
            indices.append(index)
        # print(indices.shape)
        return np.array(indices)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # set random seed for data pre-processing
        (seed,) = np.random.SeedSequence([index]).generate_state(1)
        seed = int(seed)
        np.random.seed(seed)
        random.seed(seed)
        # obtain data from disk
        img_folder, pc_folder, K_folder, seq, seq_i, key, _ = self.dataset[index]
        img = np.load(os.path.join(img_folder, '%06d.npy' % seq_i))
        data = np.load(os.path.join(pc_folder, '%06d.npy' % seq_i))
        intensity = data[3:4, :]
        sn = data[4:, :]
        pc = data[0:3, :]


        P_Tr = np.dot(self.calibhelper.get_matrix(seq, key),
                      self.calibhelper.get_matrix(seq, 'Tr'))

        pc = np.dot(P_Tr[0:3, 0:3], pc) + P_Tr[0:3, 3:]  # transform pc to camera coordinate system
        sn = np.dot(P_Tr[0:3, 0:3], sn)
        K = np.load(os.path.join(K_folder, '%06d.npy' % seq_i))

        # print(index, type(pc), type(intensity), type(sn))
        # 1. transform pc into 40960 points
        pc, intensity, sn = self.downsample_with_intensity_sn(pc, intensity, sn, voxel_grid_downsample_size=0.1)
        pc, intensity, sn = self.downsample_np(pc, intensity,sn)

        P = self.generate_random_transform(self.mode)
        pc = np.dot(P[0:3, 0:3], pc) + P[0:3, 3:]
        sn = np.dot(P[0:3, 0:3], sn)
        
        # 2. get multi-level points and neighbor indexes for pyramid feature map
        num_stages = 5
        data_dict = precompute_point_cloud_stack_mode(pc, intensity, sn, lengths=self.num_pc, num_stages=5)
        feats = torch.from_numpy(np.concatenate([intensity, sn], axis=0).T.astype(np.float32))  

        data_dict['feats'] = feats

        coarse_points = np.array(data_dict['points'][-1], dtype=np.float32).T  # [3, 2560]
        for i in range(num_stages):
            # data_dict['points'][i] = torch.from_numpy(np.asarray(data_dict['points'][i].points, dtype=np.float32))
            data_dict['neighbors'][i] = data_dict['neighbors'][i].long()
            if i < num_stages - 1:
                data_dict['subsampling'][i] = data_dict['subsampling'][i].long()
                data_dict['upsampling'][i] = data_dict['upsampling'][i].long()
        
        # 3. scale image and camera intrinsic matrix
        img = cv2.resize(img,
                         (int(round(img.shape[1] * 0.5)),
                          int(round((img.shape[0] * 0.5)))),
                         interpolation=cv2.INTER_LINEAR)
        K = self.camera_matrix_scaling(K, 0.5)

        if 'train' == self.mode:
            img_crop_dx = random.randint(0, img.shape[1] - self.img_W)
            img_crop_dy = random.randint(0, img.shape[0] - self.img_H)
        else:
            img_crop_dx = int((img.shape[1] - self.img_W) / 2)
            img_crop_dy = int((img.shape[0] - self.img_H) / 2)
        img = img[img_crop_dy:img_crop_dy + self.img_H,
              img_crop_dx:img_crop_dx + self.img_W, :]
        K = self.camera_matrix_cropping(K, dx=img_crop_dx, dy=img_crop_dy)


        #get 1/8 scale image for correspondences
        scale_size = 0.125
        K_2 = self.camera_matrix_scaling(K,0.5)
 
        K_4=self.camera_matrix_scaling(K,scale_size)
 
        if 'train' == self.mode:
            img = self.augment_img(img)

        # # project coarse_points to image_s8 and get corrs
        # [3, 1280]
        proj_coarse_points = np.dot(K_4, np.dot(np.linalg.inv(P[0:3, 0:3]), coarse_points)-np.dot(np.linalg.inv(P[0:3, 0:3]), P[0:3, 3:]))
        coarse_points_mask = np.zeros((1, np.shape(coarse_points)[1]), dtype=np.float32)
        proj_coarse_points[0:2, :] = proj_coarse_points[0:2, :] / proj_coarse_points[2:, :]
        xy = np.floor(proj_coarse_points[0:2, :] + 0.5)
        is_in_picture = (xy[0, :] >= 1) & (xy[0, :] <= (self.img_W*scale_size - 3)) & (xy[1, :] >= 1) & (xy[1, :] <= (self.img_H*scale_size - 3)) & (proj_coarse_points[2, :] > 0)
        coarse_points_mask[:, is_in_picture] = 1.

        pc_kpt_idx=np.where(coarse_points_mask.squeeze()==1)[0]
        # assert len(pc_kpt_idx) >= 64
        sel_index=np.random.permutation(len(pc_kpt_idx))[0:self.num_kpt]
        pc_kpt_idx=pc_kpt_idx[sel_index]

        pc_outline_idx=np.where(coarse_points_mask.squeeze()==0)[0]
        sel_index=np.random.permutation(len(pc_outline_idx))[0:self.num_kpt]
        pc_outline_idx=pc_outline_idx[sel_index]

        xy2 = xy[:, is_in_picture]
        img_mask_s8 = coo_matrix((np.ones_like(xy2[0, :]), (xy2[1, :], xy2[0, :])), shape=(int(self.img_H*scale_size), int(self.img_W*scale_size))).toarray()
        img_mask_s8 = np.array(img_mask_s8)
        img_mask_s8[img_mask_s8 > 0] = 1.
        coarse_xy = xy[:, pc_kpt_idx]
        img_kpt_s8_index=xy[1,pc_kpt_idx]*self.img_W*scale_size +xy[0,pc_kpt_idx]
        img_outline_index=np.where(img_mask_s8.squeeze().reshape(-1)==0)[0]
        sel_index=np.random.permutation(len(img_outline_index))[0:self.num_kpt]
        img_outline_index=img_outline_index[sel_index]

        # project to 1/2 resolution image
        coarse_kpts = coarse_points[:, pc_kpt_idx]
        proj_points = np.dot(K_2, np.dot(np.linalg.inv(P[0:3, 0:3]), coarse_kpts)-np.dot(np.linalg.inv(P[0:3, 0:3]), P[0:3, 3:]))
        proj_points[0:2, :] = proj_points[0:2, :] / proj_points[2:, :]
        fine_xy = np.floor(proj_points[0:2, :])
        fine_is_in_picture = (fine_xy[0, :] >= 0) & (fine_xy[0, :] <= (self.img_W*0.5 - 1)) & (fine_xy[1, :] >= 0) & (fine_xy[1, :] <= (self.img_H*0.5 - 1)) & (proj_points[2, :] > 0)

        assert np.all(fine_is_in_picture==True)

        # get coarse inline points on fine feature map 
        fine_xy_kpts_index = fine_xy[1,:]*self.img_W*0.5 +fine_xy[0,:]
        fine_center_kpts_coors = coarse_xy * 4
        # indices = point2node(data_dict['points'][1], data_dict['points'][-1][pc_kpt_idx])
        indices = point2node(data_dict['points'][1], data_dict['points'][-1][torch.from_numpy(pc_kpt_idx).long()])
        return {'img': torch.from_numpy(img.astype(np.float32) / 255.).permute(2, 0, 1).contiguous(),
                'pc_data_dict': data_dict,
                'fine_pc_inline_index': indices.long(),
                'K': torch.from_numpy(K_2.astype(np.float32)),
                'K_4': torch.from_numpy(K_4.astype(np.float32)),
                'P': torch.from_numpy(np.linalg.inv(P).astype(np.float32)),
                'index': index,
                # 'pc_mask': torch.from_numpy(pc_mask).float(),       #(1,20480)
                'coarse_img_mask': torch.from_numpy(img_mask_s8).float(),     #(40,128)
                # 'img_mask': torch.from_numpy(img_mask).float(),

                'pc_kpt_idx': torch.from_numpy(pc_kpt_idx),         #128
                'pc_outline_idx':torch.from_numpy(pc_outline_idx), 
                'fine_xy_coors':torch.from_numpy(fine_xy.astype(np.int32)), 
                'coarse_img_kpt_idx':torch.from_numpy(img_kpt_s8_index).long() ,
                'fine_img_kpt_index':torch.from_numpy(fine_xy_kpts_index).long() ,      #128
                'fine_center_kpt_coors':torch.from_numpy(fine_center_kpts_coors.astype(np.int32)),
                'coarse_img_outline_index':torch.from_numpy(img_outline_index).long(),

                }
               

