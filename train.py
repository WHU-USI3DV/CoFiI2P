import os
import torch
import argparse
from network import CoFiI2P, extract_patch
from kitti_pc_img_dataloader import kitti_pc_img_dataset
from loss import*
import numpy as np
from datetime import datetime
import logging
import math
import options
# import cv2
from scipy.spatial.transform import Rotation
from einops import rearrange
# torch.cuda.device_count()
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from torch.utils.tensorboard import SummaryWriter
import sys
sys.path.append('/home/kang/PycharmPrograms/I2P')
import time 

def get_P_diff(P_pred_np,P_gt_np):
    P_diff=np.dot(np.linalg.inv(P_pred_np),P_gt_np)
    t_diff=np.linalg.norm(P_diff[0:3,3])
    r_diff=P_diff[0:3,0:3]
    R_diff=Rotation.from_matrix(r_diff)
    angles_diff=np.sum(np.abs(R_diff.as_euler('xzy',degrees=True)))
    return t_diff,angles_diff

def test_acc(model,testdataloader,args):
    
    t_diff_set=[]
    angles_diff_set=[]
    topk_list = torch.zeros(6, 5)
    count = 0
    mode = 'val'
    for step,data in enumerate(testdataloader):
        if count >= 6:
            break
        model.eval()
        img=data['img'].cuda()
        # pc = data['pc'].cuda()  #full size
        # intensity = data['intensity'].cuda()
        # sn = data['sn'].cuda()
        pc_data_dict=data['pc_data_dict']
        for key in pc_data_dict:
            for j in range(len(pc_data_dict[key])):
                pc_data_dict[key][j] = torch.squeeze(pc_data_dict[key][j]).cuda()  
        pc_data_dict['feats'] = torch.squeeze(pc_data_dict['feats']).cuda() 
        K=torch.squeeze(data['K'].cuda())
        K_4=torch.squeeze(data['K_4'].cuda())
        P=torch.squeeze(data['P'].cuda())  
        # pc_mask=data['pc_mask'].cuda()  
        coarse_img_mask=torch.squeeze(data['coarse_img_mask']).cuda()      #1/4 size
        
        pc_kpt_idx=torch.squeeze(data['pc_kpt_idx']).cuda()  #(128)
        pc_outline_idx=torch.squeeze(data['pc_outline_idx']).cuda()  #(128)
        # img_kpt_idx=torch.squeeze(data['img_kpt_idx']).cuda()
        fine_img_kpt_index = torch.squeeze(data['fine_img_kpt_index']).cuda()  # [128]
        
        coarse_img_kpt_idx=torch.squeeze(data['coarse_img_kpt_idx']).cuda()  # [128]
        fine_center_kpt_coors = torch.squeeze(data['fine_center_kpt_coors']).cuda()  #[3, 128]
        fine_xy = torch.squeeze(data['fine_xy_coors']).cuda()
        fine_pc_inline_index = torch.squeeze(data['fine_pc_inline_index']).cuda()

        img_x=torch.linspace(0,coarse_img_mask.size(-1)-1,coarse_img_mask.size(-1)).view(1,-1).expand(coarse_img_mask.size(-2),coarse_img_mask.size(-1)).unsqueeze(0).cuda()
        img_y=torch.linspace(0,coarse_img_mask.size(-2)-1,coarse_img_mask.size(-2)).view(-1,1).expand(coarse_img_mask.size(-2),coarse_img_mask.size(-1)).unsqueeze(0).cuda()
        # [2, 20, 64]
        img_xy=torch.cat((img_x,img_y),dim=0)

        img_features,pc_features, coarse_img_score, coarse_pc_score\
                , fine_img_feature_patch, fine_pc_inline_feature\
                    , fine_center_xy, coarse_pc_points=model(pc_data_dict,img, fine_center_kpt_coors,fine_xy, fine_pc_inline_index, mode)    # [128, 20, 64] ,[128, 2560]

        pc_features_inline=torch.gather(pc_features,index=pc_kpt_idx.expand(pc_features.size(0),args.num_kpt),dim=-1)
        # pc_features_outline=torch.gather(pc_features,index=pc_outline_idx.unsqueeze(1).expand(B,pc_features.size(1),args.num_kpt),dim=-1)
        pc_xyz_inline=torch.gather(pc_data_dict['points'][-1].T,index=pc_kpt_idx.unsqueeze(0).expand(3,args.num_kpt),dim=-1)
        
        img_features_flatten=img_features.contiguous().view(img_features.size(1),-1)
        
        img_xy_flatten=img_xy.contiguous().view(2,-1)
        img_features_flatten_inline=torch.gather(img_features_flatten,index= coarse_img_kpt_idx.unsqueeze(0).expand(img_features_flatten.size(0),args.num_kpt),dim=-1)
        img_xy_flatten_inline=torch.gather(img_xy_flatten,index= coarse_img_kpt_idx.unsqueeze(0).expand(2,args.num_kpt),dim=-1)

        pc_xyz_projection=torch.mm(K_4,(torch.mm(P[0:3,0:3],pc_xyz_inline)+P[0:3,3:]))
        #pc_xy_projection=torch.floor(pc_xyz_projection[:,0:2,:]/pc_xyz_projection[:,2:,:]).float()
        pc_xy_projection=pc_xyz_projection[0:2,:]/pc_xyz_projection[2:,:]

        correspondence_mask=(torch.sqrt(torch.sum(torch.square(img_xy_flatten_inline.unsqueeze(-1)-pc_xy_projection.unsqueeze(-2)),dim=0))<=args.dist_thres).float()
    
        dist_corr = 1 - torch.sum(img_features_flatten_inline.unsqueeze(-1)*pc_features_inline.unsqueeze(-2), dim=0)
        # correspondence_mask = torch.squeeze(correspondence_mask)
        # dist_corr = torch.squeeze(dist_corr)
        dist_mask = correspondence_mask * dist_corr  # only match got non-zero value
        true_index_list = torch.nonzero(dist_mask, as_tuple=False)
        true_value_list = dist_mask[true_index_list[:, 0], true_index_list[:, 1]].tolist()
        sorted_dist, indices = torch.sort(dist_corr, dim=-1, descending=False)
        topk = [1, 2, 3, 4, 5]
 
        for k in topk:
            candidate_values = sorted_dist[:, 0:k]
            for i in range(pc_kpt_idx.shape[0]):  # 128
                candidates = candidate_values[i, :].tolist()
                for candidate in candidates:
                    if candidate in true_value_list:
                        topk_list[count, k - 1] += 1
        count += 1
    acc = torch.mean(topk_list / len(true_value_list), dim=0)
    # print(acc)
    # return np.mean(np.array(t_diff_set)),np.mean(np.array(angles_diff_set))
    return acc
if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Point Cloud Registration')
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
    parser.add_argument('--num_workers', type=int, default=2, metavar='num_workers',
                        help='num of CPUs')
    parser.add_argument('--val_freq', type=int, default=100, metavar='val_freq',
                        help='')
    parser.add_argument('--lr', type=float, default=0.001, metavar='lr',
                        help='')
    parser.add_argument('--min_lr', type=float, default=0.00001, metavar='lr',
                        help='')

    parser.add_argument('--P_tx_amplitude', type=float, default=10, metavar='P_tx_amplitude',
                        help='')
    parser.add_argument('--P_ty_amplitude', type=float, default=0, metavar='P_ty_amplitude',
                        help='')
    parser.add_argument('--P_tz_amplitude', type=float, default=10, metavar='P_tz_amplitude',
                        help='')
    parser.add_argument('--P_Rx_amplitude', type=float, default=2*math.pi*0, metavar='P_Rx_amplitude',
                        help='')
    parser.add_argument('--P_Ry_amplitude', type=float, default=2*math.pi, metavar='P_Ry_amplitude',
                        help='')
    parser.add_argument('--P_Rz_amplitude', type=float, default=2*math.pi*0, metavar='P_Rz_amplitude',
                        help='')

    parser.add_argument('--save_path', type=str, default='/home/kang/PycharmPrograms/I2P/model', metavar='save_path',
                        help='path to save log and model')
    
    parser.add_argument('--num_kpt', type=int, default=64, metavar='num_kpt',
                        help='')
    parser.add_argument('--dist_thres', type=float, default=1, metavar='num_kpt',
                        help='')

    parser.add_argument('--img_thres', type=float, default=0.9, metavar='img_thres',
                        help='')
    parser.add_argument('--pc_thres', type=float, default=0.9, metavar='pc_thres',
                        help='')

    parser.add_argument('--pos_margin', type=float, default=0.2, metavar='pos_margin',
                        help='')
    parser.add_argument('--neg_margin', type=float, default=1.8, metavar='neg_margin',
                        help='')


    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    logdir=args.save_path
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    writer = SummaryWriter(log_dir=os.path.join('/home/kang/PycharmPrograms/I2P/runs', datetime.now().strftime("%Y%m%d-%H%M%S")))
    logger=logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/log_%s.txt' % (logdir, datetime.now().strftime("%Y%m%d-%H%M%S")))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    # torch.multiprocessing.set_start_method('spawn')
                                           
    train_dataset = kitti_pc_img_dataset(args.data_path, 'train', args.num_point,
                                         P_tx_amplitude=args.P_tx_amplitude,
                                         P_ty_amplitude=args.P_ty_amplitude,
                                         P_tz_amplitude=args.P_tz_amplitude,
                                         P_Rx_amplitude=args.P_Rx_amplitude,
                                         P_Ry_amplitude=args.P_Ry_amplitude,
                                         P_Rz_amplitude=args.P_Rz_amplitude,num_kpt=args.num_kpt,is_front=False)
    test_dataset = kitti_pc_img_dataset(args.data_path, 'val', args.num_point,
                                        P_tx_amplitude=args.P_tx_amplitude,
                                        P_ty_amplitude=args.P_ty_amplitude,
                                        P_tz_amplitude=args.P_tz_amplitude,
                                        P_Rx_amplitude=args.P_Rx_amplitude,
                                        P_Ry_amplitude=args.P_Ry_amplitude,
                                        P_Rz_amplitude=args.P_Rz_amplitude,num_kpt=args.num_kpt,is_front=False)
    assert len(train_dataset) > 10
    assert len(test_dataset) > 10
    trainloader=torch.utils.data.DataLoader(train_dataset,batch_size=args.train_batch_size,shuffle=True,drop_last=True,num_workers=args.num_workers)
    testloader=torch.utils.data.DataLoader(test_dataset,batch_size=args.val_batch_size,shuffle=False,drop_last=True,num_workers=args.num_workers)
    opt=options.Options()
    
    model=CoFiI2P(opt).cuda()
    # model.load_state_dict(torch.load('/home/kang/PycharmPrograms/I2P/log_xy_40960_128/dist_thres_1.00_pos_margin_0.20_neg_margin_1.80/mode_epoch_5.t7'))
    current_lr=args.lr
    learnable_params=filter(lambda p:p.requires_grad,model.parameters())
    optimizer=torch.optim.Adam(learnable_params,lr=current_lr)
    
    logger.info(args)

    global_step=0
    

    for epoch in range(args.epoch):
        for step,data in enumerate(trainloader):
            global_step+=1
            start_time = time.time()
            model.train()
            mode = 'train'
            optimizer.zero_grad()
            img=data['img'].cuda()
            # pc = data['pc'].cuda()  #full size
            # intensity = data['intensity'].cuda()
            # sn = data['sn'].cuda()
            pc_data_dict=data['pc_data_dict']
            for key in pc_data_dict:
                for j in range(len(pc_data_dict[key])):
                    pc_data_dict[key][j] = torch.squeeze(pc_data_dict[key][j]).cuda()
            pc_data_dict['feats'] = torch.squeeze(pc_data_dict['feats']).cuda()
            K_4=torch.squeeze(data['K_4'].cuda())
            K=torch.squeeze(data['K'].cuda())
            P=torch.squeeze(data['P'].cuda())
            # pc_mask=data['pc_mask'].cuda()  
            # img_mask=torch.squeeze(data['img_mask']).cuda()     #1/4 size
            coarse_img_mask=torch.squeeze(data['coarse_img_mask']).cuda()  # [20, 64]
            # B=coarse_img_mask.size(0)
            pc_kpt_idx=torch.squeeze(data['pc_kpt_idx']).cuda()  #(128)
            pc_outline_idx=torch.squeeze(data['pc_outline_idx']).cuda()  #(128)
            # img_kpt_idx=torch.squeeze(data['img_kpt_idx']).cuda()
            fine_img_kpt_index = torch.squeeze(data['fine_img_kpt_index']).cuda()  # [128]
            
            coarse_img_kpt_idx=torch.squeeze(data['coarse_img_kpt_idx']).cuda()  # [128]
            # img_outline_idx=torch.squeeze(data['coarse_img_outline_index']).cuda()  # [128]
            fine_center_kpt_coors = torch.squeeze(data['fine_center_kpt_coors']).cuda()  #[3, 128]
            fine_xy = torch.squeeze(data['fine_xy_coors']).cuda()
            fine_pc_inline_index = torch.squeeze(data['fine_pc_inline_index']).cuda()

            img_x=torch.linspace(0,coarse_img_mask.size(-1)-1,coarse_img_mask.size(-1)).view(1,-1).expand(coarse_img_mask.size(-2),coarse_img_mask.size(-1)).unsqueeze(0).cuda()
            img_y=torch.linspace(0,coarse_img_mask.size(-2)-1,coarse_img_mask.size(-2)).view(-1,1).expand(coarse_img_mask.size(-2),coarse_img_mask.size(-1)).unsqueeze(0).cuda()
            # [2, 20, 64] coarse level
            img_xy=torch.cat((img_x,img_y),dim=0)
            # model_start = time.time()
            img_features,pc_features, coarse_img_score, coarse_pc_score\
                , fine_img_feature_patch, fine_pc_inline_feature\
                    , fine_center_xy, coarse_pc_points=model(pc_data_dict,img, fine_center_kpt_coors,fine_xy, fine_pc_inline_index, mode)    # [1, 128, 20, 64] ,[128, 2560]
            # model_end = time.time()
            # print('model calculate time:', model_end - model_start)
            '''
            coarse match and overlap detection
            '''
            # [128, 128]
            pc_features_inline=torch.gather(pc_features,index=pc_kpt_idx.expand(pc_features.size(0),args.num_kpt),dim=-1)
            # [128, 128]
            pc_features_outline=torch.gather(pc_features,index=pc_outline_idx.expand(pc_features.size(0),args.num_kpt),dim=-1)
            # [3, 128]
            pc_xyz_inline=torch.gather(pc_data_dict['points'][-1].T,index=pc_kpt_idx.unsqueeze(0).expand(3,args.num_kpt),dim=-1)
            # [128, 1280]
            img_features_flatten=img_features.contiguous().view(img_features.size(1),-1)
            # [2, 1280]
            img_xy_flatten=img_xy.contiguous().view(2,-1)
            # [128, 128]
            img_features_flatten_inline=torch.gather(img_features_flatten,index=coarse_img_kpt_idx.unsqueeze(0).expand(img_features_flatten.size(0),args.num_kpt),dim=-1)
            # [2, 128]
            img_xy_flatten_inline=torch.gather(img_xy_flatten,index=coarse_img_kpt_idx.unsqueeze(0).expand(2,args.num_kpt),dim=-1)
            # img_features_flatten_outline=torch.gather(img_features_flatten,index=img_outline_idx.unsqueeze(0).expand(2,args.num_kpt),dim=-1)
            # [3, 128]
            pc_xyz_projection=torch.mm(K_4,(torch.mm(P[0:3,0:3],pc_xyz_inline)+P[0:3,3:]))
            #pc_xy_projection=torch.floor(pc_xyz_projection[:,0:2,:]/pc_xyz_projection[:,2:,:]).float()
            pc_xy_projection=pc_xyz_projection[0:2,:]/pc_xyz_projection[2:,:]
            # [128, 128]
            correspondence_mask=(torch.sqrt(torch.sum(torch.square(img_xy_flatten_inline.unsqueeze(-1)-pc_xy_projection.unsqueeze(-2)),dim=0))<=args.dist_thres).float()

            # loss_desc = coarse_circle_loss(img_features_flatten_inline,pc_features_inline,correspondence_mask)
            loss_desc,dists=desc_loss(img_features_flatten_inline,pc_features_inline,correspondence_mask,pos_margin=args.pos_margin,neg_margin=args.neg_margin)
            
            coarse_pc_inline_score = torch.squeeze(coarse_pc_score[:, :, pc_kpt_idx])
            coarse_pc_outline_score = torch.squeeze(coarse_pc_score[:, :, pc_outline_idx])
            writer.add_scalars('pc_score', {'inline_max': coarse_pc_inline_score.max(), 'inline_min': coarse_pc_inline_score.min(), 'inline_avg': torch.mean(coarse_pc_inline_score),
                                            'outline_max': coarse_pc_outline_score.max(), 'outline_min': coarse_pc_outline_score.min(), 'outline_avg': torch.mean(coarse_pc_outline_score)}, global_step)
            loss_2 = overlap_loss(coarse_pc_inline_score, coarse_pc_outline_score)
            '''
            fine match
            '''
            
            # get point cloud indices in 1/2 resolution
            
            # extract fine feature patch
            relative_coors = fine_xy - fine_center_kpt_coors + 2
            relative_index = relative_coors[1, :]*4 + relative_coors[0,:]

            if global_step%args.val_freq==0:
                recall_num = torch.zeros(64)
                fine_pc_inline = fine_pc_inline_feature.unsqueeze(-1)
                fine_img_feature_flatten = torch.squeeze(rearrange(fine_img_feature_patch, 'b c h w -> b c (h w)'))
                fine_dist = torch.cosine_similarity(fine_img_feature_flatten.unsqueeze(-1), fine_pc_inline.unsqueeze(-2))
                fine_dist = torch.squeeze(fine_dist)
                fine_predict_index = torch.argmax(fine_dist, dim=1)
                mask = torch.where(fine_predict_index == relative_index)[0]
                recall_num[mask]=1
                fine_recall = torch.sum(recall_num) / 64
                writer.add_scalar('fine_recall', fine_recall, int(global_step / 100))
            loss_3 = fine_circle_loss(fine_img_feature_patch, fine_pc_inline_feature, relative_index, 64)
            loss=loss_desc + loss_2 + loss_3
            # back_start = time.time()
            loss.backward()
            # back_end = time.time()
            # print('back forward time:', back_end - back_start)
            optimizer.step()
            
            end_time = time.time()
            # print('total time:', end_time-start_time)
            #torch.cuda.empty_cache()
            # print(loss)
            writer.add_scalar('Desc Loss:', loss_desc, global_step)
            writer.add_scalar('Loss_2:', loss_2.detach().cpu().numpy(), global_step)
            writer.add_scalar('Loss_3:', loss_3.detach().cpu().numpy(), global_step)
            if global_step%10==0:
                logger.info('%s-%d-%d, loss: %f, loss_desc: %f, loss_2: %f, loss_3: %f'%('train',epoch,global_step,loss.data.cpu().numpy(), loss_desc.data.cpu().numpy(), loss_2.data.cpu().numpy(),loss_3.data.cpu().numpy()))
            
            if global_step%args.val_freq==0:
                acc=test_acc(model,testloader,args)
                logger.info('acc: top5 %s %s %s %s %s',acc[0].cpu().numpy(),acc[1].cpu().numpy(), acc[2].cpu().numpy(), acc[3].cpu().numpy(), acc[4].cpu().numpy())
                writer.add_scalars('test_acc/topk', {'k=1': acc[0], 'k=2': acc[1], 'k=3': acc[2], 'k=4': acc[3], 'k=5': acc[4]}, int(global_step / 100))
    
        
        if epoch%5==0 and epoch>0:
            current_lr=current_lr*0.25
            if current_lr<args.min_lr:
                current_lr=args.min_lr
            for param_group in optimizer.param_groups:
                param_group['lr']=current_lr
            logger.info('%s-%d-%d, updata lr, current lr is %f'%('train',epoch,global_step,current_lr))
        torch.save(model.state_dict(),os.path.join(logdir,'mode_epoch_%d.t7'%epoch))
