import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
import argparse
import numpy as np
from scipy.spatial.transform import Rotation
from pathlib import Path
import cv2
import datetime

from model.network import CoFiI2P
from data.kitti import kitti_pc_img_dataset
from data.nuscenes import nuscenes_pc_img_dataset
from data.options import * 

def get_P_diff(P_pred_np,P_gt_np):
    P_diff=np.dot(np.linalg.inv(P_pred_np),P_gt_np)
    t_diff=np.linalg.norm(P_diff[0:3,3])
    r_diff=P_diff[0:3,0:3]
    R_diff=Rotation.from_matrix(r_diff)
    angles_diff=np.sum(np.abs(R_diff.as_euler('xzy',degrees=True)))
    return t_diff,angles_diff

if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description='Image-to-Point Cloud Registration (CoFiI2P)')
    parser.add_argument("ckpt",type = str, help = "checkpoint path")
    parser.add_argument("dataset",type = str,help = "eval dataset")
    parser.add_argument("--eval_path", type=str, default = "eval_results", help = "path for evaluation files")
    args = parser.parse_args()

    if args.dataset == "kitti":
        opt = Options_KITTI()
        dataset = kitti_pc_img_dataset(opt,"val",is_front=False)
    elif args.dataset == "nuscenes":
        opt = Options_Nuscenes()
        dataset = nuscenes_pc_img_dataset(opt,"val",is_front=False)
    else:
        raise ValueError("only support KITTI and Nuscenes now!")

    assert len(dataset) > 10
    
    testloader=torch.utils.data.DataLoader(dataset,
                                           batch_size=opt.val_batch_size,
                                           shuffle=False,
                                           drop_last=False,
                                           num_workers=opt.num_workers)
    model=CoFiI2P(opt)
    model.load_state_dict(torch.load(args.ckpt))
    model=model.cuda()

    curr_date = datetime.datetime.now()
    curr_date = curr_date.strftime("%Y%m%d_%H%M%S")
    eval_path = Path(args.eval_path) / args.dataset / curr_date     
    if os.path.exists(eval_path) == False:
        os.makedirs(eval_path)

    t_diff_set=[]
    angles_diff_set=[]
    success_num = 0
    total_time = []
    # infer_time = []
    with torch.no_grad():
        for step,data in enumerate(testloader):
            save_dict = {}
            # total_start = time.time()
            model.eval()
            mode = 'test'
            # optimizer.zero_grad()
            img=data['img'].cuda()
            pc_data_dict=data['pc_data_dict']
            for key in pc_data_dict:
                for j in range(len(pc_data_dict[key])):
                    pc_data_dict[key][j] = torch.squeeze(pc_data_dict[key][j]).cuda()
            pc_data_dict['feats'] = torch.squeeze(pc_data_dict['feats']).cuda()
            K_4=torch.squeeze(data['K_4'].cuda())
            K=torch.squeeze(data['K'].cuda())
            P=torch.squeeze(data['P']).cpu().numpy()
            coarse_img_mask=torch.squeeze(data['coarse_img_mask']).cuda()  # [20, 64]
            pc_kpt_idx=torch.squeeze(data['pc_kpt_idx']).cuda()  #(128)
            pc_outline_idx=torch.squeeze(data['pc_outline_idx']).cuda()  #(128)
            fine_img_kpt_index = torch.squeeze(data['fine_img_kpt_index']).cuda()  # [128]
            
            coarse_img_kpt_idx=torch.squeeze(data['coarse_img_kpt_idx']).cuda()  # [128]
            fine_center_kpt_coors = torch.squeeze(data['fine_center_kpt_coors']).cuda()  #[3, 128]
            fine_xy = torch.squeeze(data['fine_xy_coors']).cuda()
            fine_pc_inline_index = torch.squeeze(data['fine_pc_inline_index']).cuda()

            img_x=torch.linspace(0,coarse_img_mask.size(-1)-1,coarse_img_mask.size(-1)).view(1,-1).expand(coarse_img_mask.size(-2),coarse_img_mask.size(-1)).unsqueeze(0).cuda()
            img_y=torch.linspace(0,coarse_img_mask.size(-2)-1,coarse_img_mask.size(-2)).view(-1,1).expand(coarse_img_mask.size(-2),coarse_img_mask.size(-1)).unsqueeze(0).cuda()
            # [2, 20, 64] coarse level
            img_xy=torch.cat((img_x,img_y),dim=0)
            
            img_features,pc_features, coarse_img_score, coarse_pc_score\
                , fine_img_feature_patch, fine_pc_inline_feature, fine_center_xy\
                ,coarse_pc_points=model(pc_data_dict,img, fine_center_kpt_coors,fine_xy, fine_pc_inline_index, mode)    # [1, 128, 20, 64] ,[128, 2560]

            # fine_img_feature_flatten = rearrange(fine_img_feature_patch, 'n c h w -> n c (h w)')
            fine_pc_inline_feature = fine_pc_inline_feature.unsqueeze(-1)
            dist = torch.cosine_similarity(fine_img_feature_patch.unsqueeze(-1), fine_pc_inline_feature.unsqueeze(-2))
            dist = torch.squeeze(dist)
            predict_index = torch.argmax(dist, dim=1)
            fine_xy = fine_center_xy - 2
            fine_xy[0] = fine_xy[0] + predict_index // 4
            fine_xy[1] = fine_xy[1] + predict_index % 4

            is_success,R,t,inliers = cv2.solvePnPRansac(cameraMatrix=K.cpu().numpy(), imagePoints=fine_xy.T.cpu().numpy(), objectPoints=coarse_pc_points.cpu().numpy(), iterationsCount=10000, distCoeffs=None)
            if is_success is True:
                success_num += 1    
                R,_=cv2.Rodrigues(R)
                T_pred=np.eye(4)
                T_pred[0:3,0:3]=R
                T_pred[0:3,3:]=t
                t_diff,angles_diff=get_P_diff(T_pred,P)
                print(step, angles_diff, t_diff)
                t_diff_set.append(t_diff)
                angles_diff_set.append(angles_diff)
            # totol_end = time.time()
            # total_time.append(totol_end-total_start)

            save_dict['GT_P'] = P # [4, 4]
            save_dict['pred_P'] = T_pred # [4, 4]
            save_dict['K'] = K # [3, 3]
            save_dict['points'] = pc_data_dict['points'][1] # [10240, 3]
            save_dict['P'] = P # [1, 3, 160, 512]
            save_dict['superpoints'] = pc_data_dict['points'][-1] # [1280, 3]
            save_dict['superpoints_score'] = coarse_pc_score # [1, 1, 1280]
            save_dict['fine_xy'] = fine_xy
            save_dict['object_points'] = coarse_pc_points
            # print(eval_path / str('%06d.npy'%(step)))
            np.save(eval_path / str('%06d.npy'%(step)), save_dict)
        # total_time = np.array(total_time)
        t_diff_set = np.array(t_diff_set)
        angles_diff_set = np.array(angles_diff_set)
        print(f'success num / total num: {success_num}/{len(testloader.dataset)}')
        print(np.mean(angles_diff_set), np.mean(t_diff_set))
        # print('per frame time:', np.mean(total_time))
        np.save('%s_t_error.npy'%args.dataset, t_diff_set)
        np.save('%s_r_error.npy'%args.dataset, angles_diff_set)
            
            
            