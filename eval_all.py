import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
import argparse
from network import CoFiI2P
from kitti_pc_img_dataloader import kitti_pc_img_dataset
#from loss2 import kpt_loss, kpt_loss2, eval_recall
import datetime
import logging
import math
import numpy as np
import options
from einops import rearrange
import cv2
from scipy.spatial.transform import Rotation
import time 
def get_P_diff(P_pred_np,P_gt_np):
    P_diff=np.dot(np.linalg.inv(P_pred_np),P_gt_np)
    t_diff=np.linalg.norm(P_diff[0:3,3])
    r_diff=P_diff[0:3,0:3]
    R_diff=Rotation.from_matrix(r_diff)
    angles_diff=np.sum(np.abs(R_diff.as_euler('xzy',degrees=True)))
    return t_diff,angles_diff

if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description='Image Point Cloud Registration')
    parser.add_argument('--epoch', type=int, default=10, metavar='epoch',
                        help='number of epoch to train')
    parser.add_argument('--train_batch_size', type=int, default=1, metavar='train_batch_size',
                        help='Size of train batch')
    parser.add_argument('--val_batch_size', type=int, default=1, metavar='val_batch_size',
                        help='Size of val batch')
    parser.add_argument('--data_path', type=str, default='/home/kang/corri2p_data/', metavar='data_path',
                        help='train and test data path')
    parser.add_argument('--num_point', type=int, default=20480, metavar='num_point',
                        help='point cloud size to train')
    parser.add_argument('--num_workers', type=int, default=8, metavar='num_workers',
                        help='num of CPUs')
    parser.add_argument('--val_freq', type=int, default=300, metavar='val_freq',
                        help='')
    parser.add_argument('--lr', type=float, default=0.01, metavar='lr',
                        help='')

    parser.add_argument('--P_tx_amplitude', type=float, default=10, metavar='P_tx_amplitude',
                        help='')
    parser.add_argument('--P_ty_amplitude', type=float, default=0, metavar='P_ty_amplitude',
                        help='')
    parser.add_argument('--P_tz_amplitude', type=float, default=10, metavar='P_tz_amplitude',
                        help='')
    parser.add_argument('--P_Rx_amplitude', type=float, default=0, metavar='P_Rx_amplitude',
                        help='')
    parser.add_argument('--P_Ry_amplitude', type=float, default=2*math.pi, metavar='P_Ry_amplitude',
                        help='')
    parser.add_argument('--P_Rz_amplitude', type=float, default=0, metavar='P_Rz_amplitude',
                        help='')


    parser.add_argument('--save_path', type=str, default='./log', metavar='save_path',
                        help='path to save log and model')
    parser.add_argument('--dist_thres', type=float, default=1, metavar='dist_thres',
                        help='')    
    parser.add_argument('--pos_margin', type=float, default=0.2, metavar='pos_margin',
                        help='')  
    parser.add_argument('--neg_margin', type=float, default=1.8, metavar='neg_margin',
                        help='')                           
    args = parser.parse_args()

    test_dataset = kitti_pc_img_dataset(args.data_path, 'val', args.num_point,
                                        P_tx_amplitude=args.P_tx_amplitude,
                                        P_ty_amplitude=args.P_ty_amplitude,
                                        P_tz_amplitude=args.P_tz_amplitude,
                                        P_Rx_amplitude=args.P_Rx_amplitude,
                                        P_Ry_amplitude=args.P_Ry_amplitude,
                                        P_Rz_amplitude=args.P_Rz_amplitude,is_front=False)

    assert len(test_dataset) > 10
    
    testloader=torch.utils.data.DataLoader(test_dataset,batch_size=args.val_batch_size,shuffle=False,drop_last=True,num_workers=args.num_workers)
    
    opt=options.Options()
    
    model=CoFiI2P(opt)

    model.load_state_dict(torch.load('/home/kang/PycharmPrograms/I2P/model/mode_epoch_25_qnorm.t7'))
    # model.load_state_dict(torch.load('/home/kang/PycharmPrograms/I2P/mode_epoch_24_norm.t7'))
    model=model.cuda()
    # save_path='result_all_dist_thres_%0.2f_pos_margin_%0.2f_neg_margin_%0.2f'%(args.dist_thres,args.pos_margin,args.neg_margin)
    # try:
    #     os.mkdir(save_path)
    # except:
    #     pass
    t_diff_set=[]
    angles_diff_set=[]
    success_num = 0
    total_time = []
    # infer_time = []
    with torch.no_grad():
        for step,data in enumerate(testloader):
            save_dict = {}
            total_start = time.time()
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
            totol_end = time.time()
            total_time.append(totol_end-total_start)

            save_dict['GT_P'] = P # [4, 4]
            save_dict['pred_P'] = T_pred # [4, 4]
            save_dict['K'] = K # [3, 3]
            save_dict['points'] = pc_data_dict['points'][1] # [10240, 3]
            save_dict['P'] = P # [1, 3, 160, 512]
            save_dict['superpoints'] = pc_data_dict['points'][-1] # [1280, 3]
            save_dict['superpoints_score'] = coarse_pc_score # [1, 1, 1280]
            save_dict['fine_xy'] = fine_xy
            save_dict['object_points'] = coarse_pc_points
            np.save(os.path.join('/home/kang/PycharmPrograms/I2P/eval_results', '%06d'% (step)), save_dict)
        total_time = np.array(total_time)
        t_diff_set = np.array(t_diff_set)
        angles_diff_set = np.array(angles_diff_set)
        print(f'success num / total num: {success_num}/{step}')
        print(np.mean(angles_diff_set), np.mean(t_diff_set))
        print('per frame time:', np.mean(total_time))
        np.save('epoch_25_qnorm_t_error.npy', t_diff_set)
        np.save('epoch_25_qnorm_r_error.npy', angles_diff_set)
            
            
            