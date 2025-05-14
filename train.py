import os
import torch
import argparse
import numpy as np
from datetime import datetime
import logging
from scipy.spatial.transform import Rotation
from einops import rearrange
from torch.utils.tensorboard import SummaryWriter
import time 
from pathlib import Path

from model.network import CoFiI2P
from data.kitti import kitti_pc_img_dataset
from data.nuscenes import nuscenes_pc_img_dataset
from data.options import Options_KITTI,Options_Nuscenes
from model.loss import*

def get_P_diff(P_pred_np,P_gt_np):
    P_diff=np.dot(np.linalg.inv(P_pred_np),P_gt_np)
    t_diff=np.linalg.norm(P_diff[0:3,3])
    r_diff=P_diff[0:3,0:3]
    R_diff=Rotation.from_matrix(r_diff)
    angles_diff=np.sum(np.abs(R_diff.as_euler('xzy',degrees=True)))
    return t_diff,angles_diff

def test_acc(device, model,testdataloader,opt,topk_range = 5):
    
    t_diff_set=[]
    angles_diff_set=[]
    topk_list = torch.zeros(6, topk_range)
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
                    , fine_center_xy, coarse_pc_points=model(device, pc_data_dict,img, fine_center_kpt_coors,fine_xy, fine_pc_inline_index, mode)    # [128, 20, 64] ,[128, 2560]

        pc_features_inline=torch.gather(pc_features,index=pc_kpt_idx.expand(pc_features.size(0),opt.num_kpt),dim=-1)
        pc_xyz_inline=torch.gather(pc_data_dict['points'][-1].T,index=pc_kpt_idx.unsqueeze(0).expand(3,opt.num_kpt),dim=-1)
        
        img_features_flatten=img_features.contiguous().view(img_features.size(1),-1)
        img_xy_flatten=img_xy.contiguous().view(2,-1)
        img_features_flatten_inline=torch.gather(img_features_flatten,index= coarse_img_kpt_idx.unsqueeze(0).expand(img_features_flatten.size(0),opt.num_kpt),dim=-1)
        img_xy_flatten_inline=torch.gather(img_xy_flatten,index= coarse_img_kpt_idx.unsqueeze(0).expand(2,opt.num_kpt),dim=-1)

        pc_xyz_projection=torch.mm(K_4,(torch.mm(P[0:3,0:3],pc_xyz_inline)+P[0:3,3:]))
        #pc_xy_projection=torch.floor(pc_xyz_projection[:,0:2,:]/pc_xyz_projection[:,2:,:]).float()
        pc_xy_projection=pc_xyz_projection[0:2,:]/pc_xyz_projection[2:,:]

        correspondence_mask=(torch.sqrt(torch.sum(torch.square(img_xy_flatten_inline.unsqueeze(-1)-pc_xy_projection.unsqueeze(-2)),dim=0)) <= opt.dist_thres).float()
    
        dist_corr = 1 - torch.sum(img_features_flatten_inline.unsqueeze(-1)*pc_features_inline.unsqueeze(-2), dim=0)
        # correspondence_mask = torch.squeeze(correspondence_mask)
        # dist_corr = torch.squeeze(dist_corr)
        dist_mask = correspondence_mask * dist_corr  # only match got non-zero value
        true_index_list = torch.nonzero(dist_mask, as_tuple=False)
        true_value_list = dist_mask[true_index_list[:, 0], true_index_list[:, 1]].tolist()
        sorted_dist, indices = torch.sort(dist_corr, dim=-1, descending=False)
        topk = range(1,topk_range + 1)
 
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

    parser = argparse.ArgumentParser(description='Image-to-Point Cloud Registration (CoFiI2P)')
    parser.add_argument("dataset",type = str,help = "training dataset")
    parser.add_argument("--ft_from",type = str,required=False,help = "fine-tume from exist checkpoint")


    args = parser.parse_args()

    if args.dataset == "kitti":
        options = Options_KITTI
        dataset = kitti_pc_img_dataset
    elif args.dataset == "nuscenes":
        options = Options_Nuscenes
        dataset = nuscenes_pc_img_dataset
    else:
        raise ValueError("dataset name invalid, only support KITTI Odometry and Nuscenes now!")

    opt = options()
    train_dataset = dataset(opt, 
                            'train', 
                            is_front=False)
    
    test_dataset = dataset(opt,
                            'val', 
                            is_front=False)

    assert len(train_dataset) > 10
    assert len(test_dataset) > 10

    trainloader=torch.utils.data.DataLoader(train_dataset,
                                            batch_size = opt.train_batch_size,
                                            shuffle = True,
                                            drop_last = True,
                                            num_workers = opt.num_workers)
    
    testloader=torch.utils.data.DataLoader(test_dataset,
                                           batch_size = opt.val_batch_size,
                                           shuffle = False,
                                           drop_last = False,
                                           num_workers = opt.num_workers)
    device = opt.device
    model=CoFiI2P(opt).to(device)
    if args.ft_from:
        model.load_state_dict(torch.load(args.ft_from),strict = True)
        
    current_lr=opt.lr
    learnable_params=filter(lambda p:p.requires_grad,model.parameters())
    optimizer=torch.optim.Adam(learnable_params,lr=current_lr)

    curr_time = datetime.now().strftime("%Y%m%d-%H%M%S")

    save_path = Path(opt.root_path) / opt.save_path / args.dataset / curr_time
    log_path = Path(opt.root_path) / opt.log_path  /  args.dataset / curr_time

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(Path(log_path)):
        os.makedirs(log_path)

    writer = SummaryWriter(log_dir = log_path)
    logger=logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/log.txt' % log_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(opt)
    global_step=0
    
    for epoch in range(opt.epoch):
        for step,data in enumerate(trainloader):
            global_step+=1
            index = data['index']
            start_time = time.time()
            model.train()
            mode = 'train'
            optimizer.zero_grad()
            img=data['img'].to(device)
            # pc = data['pc'].cuda()  #full size
            # intensity = data['intensity'].cuda()
            # sn = data['sn'].cuda()
            pc_data_dict=data['pc_data_dict']
            for key in pc_data_dict:
                for j in range(len(pc_data_dict[key])):
                    pc_data_dict[key][j] = torch.squeeze(pc_data_dict[key][j]).to(device)
            pc_data_dict['feats'] = torch.squeeze(pc_data_dict['feats']).to(device)
            K_4=torch.squeeze(data['K_4'].to(device))
            K=torch.squeeze(data['K'].to(device))
            P=torch.squeeze(data['P'].to(device))
            # pc_mask=data['pc_mask'].cuda()  
            # img_mask=torch.squeeze(data['img_mask']).cuda()     #1/4 size
            coarse_img_mask=torch.squeeze(data['coarse_img_mask']).to(device)  # [20, 64]
            # B=coarse_img_mask.size(0)
            pc_kpt_idx=torch.squeeze(data['pc_kpt_idx']).to(device) #(128)
            pc_outline_idx=torch.squeeze(data['pc_outline_idx']).to(device)  #(128)
            # img_kpt_idx=torch.squeeze(data['img_kpt_idx']).cuda()
            fine_img_kpt_index = torch.squeeze(data['fine_img_kpt_index']).to(device)  # [128]
            
            coarse_img_kpt_idx=torch.squeeze(data['coarse_img_kpt_idx']).to(device)  # [128]
            # img_outline_idx=torch.squeeze(data['coarse_img_outline_index']).cuda()  # [128]
            fine_center_kpt_coors = torch.squeeze(data['fine_center_kpt_coors']).to(device)  #[3, 128]
            fine_xy = torch.squeeze(data['fine_xy_coors']).to(device)
            fine_pc_inline_index = torch.squeeze(data['fine_pc_inline_index']).to(device)

            img_x=torch.linspace(0,coarse_img_mask.size(-1)-1,coarse_img_mask.size(-1)).view(1,-1).expand(coarse_img_mask.size(-2),coarse_img_mask.size(-1)).unsqueeze(0).to(device)
            img_y=torch.linspace(0,coarse_img_mask.size(-2)-1,coarse_img_mask.size(-2)).view(-1,1).expand(coarse_img_mask.size(-2),coarse_img_mask.size(-1)).unsqueeze(0).to(device)
            # [2, 20, 64] coarse level
            img_xy=torch.cat((img_x,img_y),dim=0).to(device)
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
            pc_features_inline=torch.gather(pc_features,index=pc_kpt_idx.expand(pc_features.size(0),opt.num_kpt),dim=-1)
            # [128, 128]
            pc_features_outline=torch.gather(pc_features,index=pc_outline_idx.expand(pc_features.size(0),opt.num_kpt),dim=-1)
            # [3, 128]
            pc_xyz_inline=torch.gather(pc_data_dict['points'][-1].T,index=pc_kpt_idx.unsqueeze(0).expand(3,opt.num_kpt),dim=-1)
            # [128, 1280]
            img_features_flatten=img_features.contiguous().view(img_features.size(1),-1)
            # [2, 1280]
            img_xy_flatten=img_xy.contiguous().view(2,-1)
            # [128, 128]
            img_features_flatten_inline=torch.gather(img_features_flatten,index=coarse_img_kpt_idx.unsqueeze(0).expand(img_features_flatten.size(0),opt.num_kpt),dim=-1)
            # [2, 128]
            img_xy_flatten_inline=torch.gather(img_xy_flatten,index=coarse_img_kpt_idx.unsqueeze(0).expand(2,opt.num_kpt),dim=-1)
            # [3, 128]
            pc_xyz_projection=torch.mm(K_4,(torch.mm(P[0:3,0:3],pc_xyz_inline)+P[0:3,3:]))
            #pc_xy_projection=torch.floor(pc_xyz_projection[:,0:2,:]/pc_xyz_projection[:,2:,:]).float()
            pc_xy_projection=pc_xyz_projection[0:2,:]/pc_xyz_projection[2:,:]
            # [128, 128]
            correspondence_mask=(torch.sqrt(torch.sum(torch.square(img_xy_flatten_inline.unsqueeze(-1)-pc_xy_projection.unsqueeze(-2)),dim=0)) <= opt.dist_thres).float()

            # loss_desc = coarse_circle_loss(img_features_flatten_inline,pc_features_inline,correspondence_mask)
            loss_desc,dists=desc_loss(device, img_features_flatten_inline,pc_features_inline,correspondence_mask,pos_margin = opt.pos_margin,neg_margin = opt.neg_margin)
            
            coarse_pc_inline_score = torch.squeeze(coarse_pc_score[:, :, pc_kpt_idx])
            coarse_pc_outline_score = torch.squeeze(coarse_pc_score[:, :, pc_outline_idx])
            writer.add_scalars('pc_score', {'inline_max': coarse_pc_inline_score.max(), 'inline_min': coarse_pc_inline_score.min(), 'inline_avg': torch.mean(coarse_pc_inline_score),
                                            'outline_max': coarse_pc_outline_score.max(), 'outline_min': coarse_pc_outline_score.min(), 'outline_avg': torch.mean(coarse_pc_outline_score)}, global_step)
            loss_coarse = overlap_loss(device, coarse_pc_inline_score, coarse_pc_outline_score)
            '''
            fine match
            '''
            
            # get point cloud indices in 1/2 resolution
            
            # extract fine feature patch
            relative_coors = fine_xy - fine_center_kpt_coors + 2
            relative_index = relative_coors[1, :] * 4 + relative_coors[0,:]

            if global_step % opt.val_freq==0:
                recall_num = torch.zeros(opt.num_kpt).to(device)
                fine_pc_inline = fine_pc_inline_feature.unsqueeze(-1)
                fine_img_feature_flatten = torch.squeeze(rearrange(fine_img_feature_patch, 'b c h w -> b c (h w)'))
                fine_dist = torch.cosine_similarity(fine_img_feature_flatten.unsqueeze(-1), fine_pc_inline.unsqueeze(-2))
                fine_dist = torch.squeeze(fine_dist)
                fine_predict_index = torch.argmax(fine_dist, dim=1)
                mask = torch.where(fine_predict_index == relative_index)[0]
                recall_num[mask]=1
                fine_recall = torch.sum(recall_num).cpu() / opt.num_kpt
                writer.add_scalar('fine_recall', fine_recall, int(global_step / 100.0))
            loss_fine = fine_circle_loss(device, fine_img_feature_patch, fine_pc_inline_feature, relative_index, opt.num_kpt)
            loss = loss_desc + loss_coarse + loss_fine
            # back_start = time.time()
            loss.backward()
            # back_end = time.time()
            # print('back forward time:', back_end - back_start)
            optimizer.step()
            
            end_time = time.time()
            # print('total time:', end_time-start_time)
            #torch.cuda.empty_cache()
            # print(loss)
            writer.add_scalar("loss:", loss.detach().cpu().numpy(), global_step)
            writer.add_scalar('loss_desc:', loss_desc.detach().cpu().numpy(), global_step)
            writer.add_scalar('loss_coarse:', loss_coarse.detach().cpu().numpy(), global_step)
            writer.add_scalar('loss_fine:', loss_fine.detach().cpu().numpy(), global_step)
            
            if global_step % 10 == 0:
                logger.info('%s-%d-%d, loss: %f, loss_desc: %f, loss_coarse: %f, loss_fine: %f'
                            %('train',epoch,global_step,
                            loss.data.cpu().numpy(), 
                            loss_desc.data.cpu().numpy(), 
                            loss_coarse.data.cpu().numpy(),
                            loss_fine.data.cpu().numpy()))
            
            if global_step % opt.val_freq == 0:
                acc = test_acc(device, model,testloader,opt)
                logger.info('acc: top5 %s %s %s %s %s',
                            acc[0].cpu().numpy(),
                            acc[1].cpu().numpy(), 
                            acc[2].cpu().numpy(), 
                            acc[3].cpu().numpy(), 
                            acc[4].cpu().numpy())
                
                writer.add_scalars('test_acc/topk', 
                                   {'k=1': acc[0], 
                                    'k=2': acc[1], 
                                    'k=3': acc[2], 
                                    'k=4': acc[3], 
                                    'k=5': acc[4]}, 
                                    int(global_step / 100))
    
        
        if epoch % 5 == 0 and epoch > 0:
            current_lr=current_lr * opt.lr_decay_step
            if current_lr < opt.min_lr:
                current_lr = opt.min_lr
            for param_group in optimizer.param_groups:
                param_group['lr']=current_lr
            logger.info('%s-%d-%d, updata lr, current lr is %f'
                        %('train',epoch,global_step,current_lr))
            
        torch.save(model.state_dict(),os.path.join(save_path,'mode_epoch_%d.t7'%epoch))
