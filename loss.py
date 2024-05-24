# from numpy import positive
import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from einops import rearrange
from scipy import ndimage

# def coarse_circle_loss(img_features_flatten_inline,pc_features_inline,correspondence_mask): # [128, 64], [128, 64], [64, 64]
#     m = 0.2
#     gamma = 5

#     # print(fine_img_feature.shape, fine_pc_feature.shape) [128, 64, 16] [128, 64, 1]
#     dist = torch.cosine_similarity(img_features_flatten_inline.unsqueeze(-1), pc_features_inline.unsqueeze(-2), dim=0)
#     # print(dist.max(), dist.min())

#     pos = correspondence_mask
#     neg = 1 - correspondence_mask
#     sp = dist * pos
#     sn = dist * neg
#     ap = torch.relu(-sp.detach() + pos + pos * m)
#     an = torch.relu(sn.detach() + neg * m)

#     delta_p = 1 - m
#     delta_n = m

#     logit_p = - ap * (sp - pos * delta_p) * gamma
#     logit_n = an * (sn - neg * delta_n) * gamma
#     # print(logit_n.max(), logit_p.max())
#     loss_p = torch.sum(torch.exp(logit_p) * pos, dim=-1)
#     loss_n = torch.sum(torch.exp(logit_n) * neg, dim=-1)
#     loss = torch.mean(torch.log(1 + loss_n * loss_p))
#     return loss

def fine_circle_loss(fine_img_feature,fine_pc_feature, relative_index, num_kpt = 64):
    m = 0.2
    gamma = 5
    

    fine_img_feature_flatten = rearrange(fine_img_feature, 'n c h w -> n c (h w)')
    fine_pc_feature = fine_pc_feature.unsqueeze(-1)
    # print(fine_img_feature.shape, fine_pc_feature.shape) [128, 64, 16] [128, 64, 1]
    dist = torch.cosine_similarity(fine_img_feature_flatten.unsqueeze(-1), fine_pc_feature.unsqueeze(-2))
    # print(dist.max(), dist.min())
    label = torch.zeros(num_kpt, 16).cuda()
    index = torch.arange(0, num_kpt, 1).cuda()
    true_index = torch.cat((index.unsqueeze(1), relative_index.unsqueeze(1)), 1)
    label[true_index[:, 0], true_index[:, 1]] = 1
    
    # inv_label_np = 1 - np.array(label.cpu().numpy()).reshape([num_kpt, 4, 4])
    # On_list = []
    # for i in range(inv_label_np.shape[0]):
    #     dist_pixel = ndimage.distance_transform_edt(inv_label_np[i])
    #     inv_norm_dist_pixel = 1 - normalize_distance(dist_pixel)
    #     inv_norm_dist_pixel[inv_norm_dist_pixel == 1] = 0
    #     On_list.append(inv_norm_dist_pixel)
    # On_list = torch.from_numpy(np.array(On_list).reshape([num_kpt, 16])).cuda()  

    # label = label.cuda()
    dist = torch.squeeze(dist)
    pos = label
    neg = 1 - label
    sp = dist * pos
    sn = dist * neg
    ap = torch.relu(-sp.detach() + pos + pos * m)
    # an = torch.relu(sn.detach() + neg * m - On_list)
    an = torch.relu(sn.detach() + neg * m)
    delta_p = 1 - m
    delta_n = m

    logit_p = - ap * (sp - pos * delta_p) * gamma
    logit_n = an * (sn - neg * delta_n) * gamma
    # print(logit_n.max(), logit_p.max())
    loss_p = torch.sum(torch.exp(logit_p) * pos, dim=-1)
    loss_n = torch.sum(torch.exp(logit_n) * neg, dim=-1)
    loss = torch.mean(torch.log(1 + loss_n * loss_p))
    return loss
        
def overlap_loss(inline_pc_score, outline_pc_score):
    bce_loss = nn.BCELoss()
    pos_label = torch.ones(inline_pc_score.shape[0]).cuda()
    neg_label = torch.zeros(outline_pc_score.shape[0]).cuda()
    label = torch.cat((pos_label, neg_label), 0)
    score = torch.cat((inline_pc_score, outline_pc_score), 0)
    loss = bce_loss(score, label)
    return loss

def normalize_distance(distance_matrix):
    max_arr = np.max(distance_matrix)
    min_arr = np.min(distance_matrix)

    normalized_arr = (distance_matrix - min_arr) / (max_arr - min_arr) 
    return normalized_arr

def desc_loss(img_features,pc_features,mask,pos_margin=0.1,neg_margin=1.4,log_scale=10,num_kpt=512):
    pos_mask=mask
    neg_mask=1-mask
    #dists=torch.sqrt(torch.sum((img_features.unsqueeze(-1)-pc_features.unsqueeze(-2))**2,dim=1))
    dists=1-torch.sum(img_features.unsqueeze(-1)*pc_features.unsqueeze(-2),dim=0)
    # print(dists.shape)
    pos=dists-1e5*neg_mask
    pos_weight=(pos-pos_margin).detach()
    pos_weight=torch.max(torch.zeros_like(pos_weight),pos_weight)
    
    lse_positive_row=torch.logsumexp(log_scale*(pos-pos_margin)*pos_weight,dim=-1)
    lse_positive_col=torch.logsumexp(log_scale*(pos-pos_margin)*pos_weight,dim=-2)

    neg=dists+1e5*pos_mask
    neg_weight=(neg_margin-neg).detach()
    neg_weight=torch.max(torch.zeros_like(neg_weight),neg_weight)
    
    lse_negative_row=torch.logsumexp(log_scale*(neg_margin-neg)*neg_weight,dim=-1)
    lse_negative_col=torch.logsumexp(log_scale*(neg_margin-neg)*neg_weight,dim=-2)

    loss_col=F.softplus(lse_positive_row+lse_negative_row)/log_scale
    loss_row=F.softplus(lse_positive_col+lse_negative_col)/log_scale
    loss=loss_col+loss_row
    
    return torch.mean(loss),dists


# def det_loss(img_score_inline,img_score_outline,pc_score_inline,pc_score_outline,dists,mask):
#     #score (B,N)
#     pids=torch.FloatTensor(np.arange(mask.size(-1))).to(mask.device)
#     diag_mask=torch.eq(torch.unsqueeze(pids,dim=1),torch.unsqueeze(pids,dim=0)).unsqueeze(0).expand(mask.size()).float()
#     furthest_positive,_=torch.max(dists*diag_mask,dim=1)     #(B,N)
#     closest_negative,_=torch.min(dists+1e5*mask,dim=1)  #(B,N)
#     loss_inline=torch.mean((furthest_positive-closest_negative)*(img_score_inline.squeeze()+pc_score_inline.squeeze()))
#     loss_outline=torch.mean(img_score_outline)+torch.mean(pc_score_outline)
#     return loss_inline+loss_outline

# def det_loss2(img_score_inline,img_score_outline,pc_score_inline,pc_score_outline,dists,mask):
#     #score (B,N)
#     pids=torch.FloatTensor(np.arange(mask.size(-1))).to(mask.device)
#     diag_mask=torch.eq(torch.unsqueeze(pids,dim=1),torch.unsqueeze(pids,dim=0)).unsqueeze(0).expand(mask.size()).float()
#     furthest_positive,_=torch.max(dists*diag_mask,dim=1)     #(B,N)
#     closest_negative,_=torch.min(dists+1e5*mask,dim=1)  #(B,N)
#     #loss_inline=torch.mean((furthest_positive-closest_negative)*(img_score_inline.squeeze()+pc_score_inline.squeeze())) +torch.mean(1-img_score_inline)+torch.mean(1-pc_score_inline)
#     loss_inline=torch.mean(1-img_score_inline)+torch.mean(1-pc_score_inline)
#     loss_outline=torch.mean(img_score_outline)+torch.mean(pc_score_outline)
#     return loss_inline+loss_outline


def cal_acc(img_features,pc_features,mask):
    dist=torch.sum((img_features.unsqueeze(-1)-pc_features.unsqueeze(-2))**2,dim=1) #(B,N,N)
    furthest_positive,_=torch.max(dist*mask,dim=1)
    closest_negative,_=torch.min(dist+1e5*mask,dim=1)
    '''print(furthest_positive)
    print(closest_negative)
    print(torch.max(torch.sum(mask,dim=1)))
    assert False'''
    diff=furthest_positive-closest_negative
    accuracy=(diff<0).sum(dim=1)/dist.size(1)
    return accuracy

# class CircleLoss(nn.Module):
#     def __init__(self, m:float, gamma:float):
#         super(CircleLoss, self).__init__()
#         self.m = m
#         self.gamma = gamma

#     def forward(self, img_feature, pc_feature, mask):
#         pos = mask
#         neg = 1 - mask
#         # dist = similarity_calculation(img_feature, pc_feature)
#         dist = torch.cosine_similarity(img_feature.unsqueeze(-1), pc_feature.unsqueeze(-2), dim=1)
#         # print(dist.device, dist.shape)
#         # print(pos.device, pos.shape)
#         # sp = dist[pos == 1]
#         # sn = dist[neg == 1]
#         sp = dist * pos
#         sn = dist * neg
#         # print(sp.max(), sn.max())
#         # ap = torch.relu(- sp.detach() + 1 + self.m)
#         # an = torch.relu(sn.detach() + self.m)
#         ap = torch.relu(-sp.detach() + pos + pos * self.m)
#         an = torch.relu(sn.detach() + neg * self.m)

#         delta_p = 1 - self.m
#         delta_n = self.m

#         logit_p = - ap * (sp - pos * delta_p) * self.gamma
#         logit_n = an * (sn - neg * delta_n) * self.gamma
#         # print(logit_n.max(), logit_p.max())
#         loss_p = torch.sum(torch.exp(logit_p) * pos, dim=-1)
#         loss_n = torch.sum(torch.exp(logit_n) * neg, dim=-1)
#         loss = torch.mean(torch.log(1 + loss_n * loss_p))
#         return loss