import numpy as np
import math
import torch

class Options_KITTI:
    def __init__(self):
        self.epoch = 25
        self.data_path = "../corri2p_data"
        self.root_path = '.'
        self.save_path = "checkpoints"
        self.log_path = "logs"
        self.accumulation_frame_num = 3
        self.accumulation_frame_skip = 6

        self.crop_original_top_rows = 50
        self.img_scale = 0.5
        self.img_H = 160  # 320 * 0.5
        self.img_W = 512  # 1224 * 0.5 - 50
        self.img_fine_resolution_scale = 32

        self.num_pc = 20480
        self.num_kpt = 64
        self.pc_min_range = -1.0
        self.pc_max_range = 80.0
        self.node_a_num = 1280
        self.node_b_num = 1280
        self.k_ab = 16
        self.k_interp_ab = 3
        self.k_interp_point_a = 3
        self.k_interp_point_b = 3

        # CAM coordinate
        self.P_tx_amplitude = 10
        self.P_ty_amplitude = 0
        self.P_tz_amplitude = 10
        self.P_Rx_amplitude = 0.0 * math.pi / 12.0
        self.P_Ry_amplitude = 2.0 * math.pi
        self.P_Rz_amplitude = 0.0 * math.pi / 12.0
        self.dist_thres = 1.0
        self.img_thres = 0.9
        self.pc_thres = 0.9
        self.pos_margin = 0.2
        self.neg_margin = 1.8

        self.train_batch_size = 1
        self.val_batch_size = 1
        self.num_workers = 8
        self.gpu_ids = [0]
        self.device = torch.device('cuda', self.gpu_ids[0])
        # self.device = torch.device('cpu')
        self.norm = 'gn'
        self.group_norm = 32
        self.norm_momentum = 0.1
        self.activation = 'relu'
        self.lr = 1e-3
        self.min_lr = 1e-5
        self.lr_decay_step = 0.25
        self.lr_decay_scale = 0.5
        self.val_freq = 100

class Options_Nuscenes:
    def __init__(self):
        self.epoch = 10
        self.data_path = "../nuscenes_i2p"
        self.root_path = '.'
        self.save_path = "checkpoints"
        self.log_path = "logs"
        self.accumulation_frame_num = 3
        self.accumulation_frame_skip = 4
        
        self.crop_original_top_rows = 100
        self.img_scale = 0.4
        self.img_H = 160  # after scale 
        self.img_W = 320  # after scale 
        # the fine resolution is img_H/scale x img_W/scale
        self.img_fine_resolution_scale = 32

        self.num_pc = 20480
        self.num_kpt = 32
        self.pc_min_range = -1.0
        self.pc_max_range = 80.0
        self.node_a_num = 1280
        self.node_b_num = 1280
        self.k_ab = 16
        self.k_interp_ab = 3
        self.k_interp_point_a = 3
        self.k_interp_point_b = 3

        # CAM coordinate
        self.P_tx_amplitude = 0
        self.P_ty_amplitude = 0
        self.P_tz_amplitude = 0
        self.P_Rx_amplitude = 0.0 * math.pi / 12.0
        self.P_Ry_amplitude = 2.0 * math.pi
        self.P_Rz_amplitude = 0.0 * math.pi / 12.0
        self.dist_thres = 1.0
        self.img_thres = 0.9
        self.pc_thres = 0.9
        self.pos_margin = 0.2
        self.neg_margin = 1.8

        self.train_batch_size = 1
        self.val_batch_size = 1
        self.num_workers = 8
        self.gpu_ids = [0]
        self.device = torch.device('cuda', self.gpu_ids[0])
        self.norm = 'gn'
        self.group_norm = 32
        self.norm_momentum = 0.1
        self.activation = 'relu'
        self.lr = 1e-3
        self.min_lr = 1e-5
        self.lr_decay_step = 0.25
        self.lr_decay_scale = 0.5
        self.val_freq = 100



