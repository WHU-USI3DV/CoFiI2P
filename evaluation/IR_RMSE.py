import numpy as np
import os 
import open3d as o3d
import argparse
from data.options import *
from pathlib import Path

def camera_matrix_scaling(K: np.ndarray, s: float):
    K_scale = s * K
    K_scale[2, 2] = 1
    return K_scale

if __name__=='__main__':

    parser = argparse.ArgumentParser(description = "Evaluation of CoFiI2P")
    parser.add_argument("dataset", type=str, help = "dataset name")
    parser.add_argument("--eval_results_path",type=str,default = "eval_results", help = "evalution folder")
    args = parser.parse_args()

    if args.dataset == "kitti":
        opt = Options_KITTI()
    elif args.dataset == "nuscenes":
        opt = Options_Nuscenes()
    else:
        raise ValueError("only support kitti and nuscenes dataset now !")

    results_path = Path(args.eval_results_path) / args.dataset

    result_list = os.listdir(results_path)
    
    threshold = np.arange(0, 10.2, 0.2)
    ir_thre_list=[]
    rmse_thre_list=[]
    for thre in threshold:
        ir_list = []
        rmse_list = []
        for filename in result_list:
            data_dict = np.load(results_path /  filename, allow_pickle=True).item()
            gt_P = data_dict['GT_P']
            pred_P = data_dict['pred_P']
            K = data_dict['K'].cpu().numpy()
            K_4 = camera_matrix_scaling(K, 0.25)
            points = data_dict['points'].cpu().numpy()
            super_points = data_dict['superpoints'].cpu().numpy()
            superpoints_score = data_dict['superpoints_score'].cpu().numpy()
            # P=data_dict['P']
            fine_xy = data_dict['fine_xy'].cpu().numpy() # [2, n]
            object_points = data_dict['object_points'].cpu().numpy() # [n, 3]

            # project object points with GT pose
            P = np.linalg.inv(gt_P)
            proj_coarse_points = np.dot(K, np.dot(np.linalg.inv(P[0:3, 0:3]), object_points.T)-np.dot(np.linalg.inv(P[0:3, 0:3]), P[0:3, 3:]))
            gt_pixel = proj_coarse_points[0:2] / proj_coarse_points[2] # [2, n]

            residual = np.sum(np.square(fine_xy - gt_pixel),axis=0) **0.5

            pixel_threshold = thre
            rmse = np.mean(residual)
            ir = np.sum(residual <= pixel_threshold) / residual.shape[0]
            ir_list.append(ir)
            rmse_list.append(rmse)
            # if i % 100 == 0:
                # print(i, '/', len(result_list))
        ir_list = np.array(ir_list)
        rmse_list = np.array(rmse_list)
        print(f'{thre} avg ir:', np.mean(ir_list))
        print(f'{thre} avg rmse:', np.mean(rmse_list))
        ir_thre_list.append(np.mean(ir_list))
        rmse_thre_list.append(rmse_list)
    ir_thre_list = np.array(ir_thre_list)
    rmse_thre_list = np.array(rmse_thre_list)
    np.save('cofii2p_%s_ir_%d.npy'%(args.dataset,opt.num_pc), ir_thre_list)
    np.save('cofii2p_%s_rmse_%d.npy'%(args.dataset,opt.num_pc), rmse_thre_list)