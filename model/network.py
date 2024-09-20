import torch
import torch.nn as nn
import torch.nn.functional as F

# import imagenet as imagenet
from .imagenet import ResidualConv,ImageUpSample,ImageEncoder
from .transformer.attention import*
from .transformer.transformer import* 
from .transformer.position_encoding import*
from .kpconv.kp_backbone import KPConvFPN
from .kpconv.modules import*
import open3d as o3d

class CoFiI2P(nn.Module):
    def __init__(self,opt):
        super(CoFiI2P, self).__init__()
        self.opt=opt
        self.pe_H = int(self.opt.img_H / 8)
        self.pe_W = int(self.opt.img_W / 8)
        
        # encoder
        # self.pc_encoder=pointnet2.PCEncoder(opt,Ca=64,Cb=128,Cg=512)
        self.img_encoder = ImageEncoder()
        self.pc_encoder = KPConvFPN(input_dim=4, output_dim=64, init_dim=64, kernel_size=15, init_radius=4.25*0.1, init_sigma=2*0.1, norm = opt.norm, group_norm=32)
        
        self.H_fine_res = int(round(self.opt.img_H / self.opt.img_fine_resolution_scale))
        self.W_fine_res = int(round(self.opt.img_W / self.opt.img_fine_resolution_scale))

        self.pc_feature_layer=nn.Sequential(nn.Linear(2048,1024,bias=False),nn.LayerNorm(1024),nn.ReLU(),nn.Linear(1024,512,bias=False),nn.LayerNorm(512),nn.ReLU(),nn.Linear(512,128,bias=False))

        self.img_feature_layer=nn.Sequential(nn.Conv2d(128,128,1,bias=False),nn.InstanceNorm2d(128),nn.ReLU(),nn.Conv2d(128,128,1,bias=False),nn.InstanceNorm2d(128),nn.ReLU(),nn.Conv2d(128,128,1,bias=False))
        # transformer
        self.img_pos_encoding = PositionEmbeddingCoordsSine(2, 128)
        self.pc_pos_encoding = PositionEmbeddingCoordsSine(3, 128)
        self.transformer = LocalFeatureTransformer(D_MODEL=128, NHEAD=4, LAYER_NAMES=['self', 'cross'] * 4, ATTENTION = 'full')
        self.fine_img_pos_encoding = PositionEmbeddingLearned(2, 64)
        self.fine_pc_pos_encoding = PositionEmbeddingLearned(3, 64)
        # upsample
        self.img_upsample_1 = ImageUpSample(128+64, 128)
        self.img_upsample_2 = ImageUpSample(128+64, 64)
        # self.img_upsample_3 = ImageUpSample()
        self.pc_score_layer=nn.Sequential(nn.Conv1d(128,128,1,bias=False),nn.InstanceNorm1d(128),nn.ReLU(),nn.Conv1d(128,64,1,bias=False),nn.InstanceNorm1d(64),nn.ReLU(),nn.Conv1d(64,1,1,bias=False),nn.Sigmoid())
        self.img_score_layer=nn.Sequential(nn.Conv2d(128,128,1,bias=False),nn.InstanceNorm2d(128),nn.ReLU(),nn.Conv2d(128,64,1,bias=False),nn.InstanceNorm2d(64),nn.ReLU(),nn.Conv2d(64,1,1,bias=False),nn.Sigmoid())

        # self.fine_fusion = nn.Sequential(nn.Conv2d(128,128,1), nn.BatchNorm2d(128), nn.ReLU(), nn.Conv2d(128, 64, 1), nn.BatchNorm2d(64), nn.ReLU(), nn.Conv2d(64, 1, 1))
    def gather_topk_features(self, min_k_idx, features):
        """

        :param min_k_idx: BxNxk
        :param features: BxCxM
        :return:
        """
        B, N, k = min_k_idx.size(0), min_k_idx.size(1), min_k_idx.size(2)
        C, M = features.size(1), features.size(2)

        return torch.gather(features.unsqueeze(3).expand(B, C, M, k),
                            index=min_k_idx.unsqueeze(1).expand(B, C, N, k),
                            dim=2)  # BxCxNxk

    def upsample_by_interpolation(self,
                                  interp_ab_topk_idx,
                                  node_a,
                                  node_b,
                                  up_node_b_features):
        interp_ab_topk_node_b = self.gather_topk_features(interp_ab_topk_idx, node_b)  # Bx3xMaxk
        # Bx3xMa -> Bx3xMaxk -> BxMaxk
        interp_ab_node_diff = torch.norm(node_a.unsqueeze(3) - interp_ab_topk_node_b, dim=1, p=2, keepdim=False)
        interp_ab_weight = 1 - interp_ab_node_diff / torch.sum(interp_ab_node_diff, dim=2, keepdim=True)  # BxMaxk
        interp_ab_topk_node_b_features = self.gather_topk_features(interp_ab_topk_idx, up_node_b_features)  # BxCxMaxk
        # BxCxMaxk -> BxCxMa
        interp_ab_weighted_node_b_features = torch.sum(interp_ab_weight.unsqueeze(1) * interp_ab_topk_node_b_features,
                                                       dim=3)
        return interp_ab_weighted_node_b_features
    def forward(self,pc_data_dict,img, fine_center_kpt_coors,fine_xy, fine_pc_inline_index, mode):

        pc_feature_set = self.pc_encoder(pc_data_dict)
        img_feature_set=self.img_encoder(img)
        
        # pc encode and decode
        pc_encode_feature_map = pc_feature_set[-1]  # [2560, 2048]
        pc_decode_1 = pc_feature_set[-2]  # [5120, 1024]
        pc_decode_2 = pc_feature_set[-3]  # [10240, 512]
        pc_decode_3 = F.normalize( pc_feature_set[-4], dim=1, p=2)  # [20480, 64] fine matching features
        pc_feature_middle = F.normalize(self.pc_feature_layer(pc_encode_feature_map), dim=1, p=2) # coarse matching features
        
        # img encode 
        img_global_feature=img_feature_set[-1]  #[1, 512, 1, 1]
        img_s32_feature_map=img_feature_set[-2] #[1, 512, 5, 16]
        img_s16_feature_map=img_feature_set[-3] #[1, 256, 10, 32]
        img_s8_feature_map=F.normalize(img_feature_set[-4], dim=1, p=2)  #[1, 128, 20, 64]
        img_s4_feature_map=img_feature_set[-5]  #[1, 64, 40, 128]
        img_s2_feature_map=img_feature_set[-6]  #[1, 64, 80, 256]
        
        '''
        Transformer
        '''
        #########
        # Position Embedding
        #########
        # print(node_a.shape) # [12, 3, 128]
        # img_x, img_y = torch.meshgrid(torch.linspace(0,19,20), torch.linspace(0, 63, 64), indexing='ij')
        # img_xy=rearrange(torch.cat((img_x.unsqueeze(-1),img_y.unsqueeze(-1)),dim=2).expand(1, 20, 64, 2), 'b h w d -> b (h w) d').cuda()
        # kitti
        img_x, img_y = torch.meshgrid(torch.arange(0,self.pe_H), torch.arange(0, self.pe_W), indexing='ij')
        img_xy=rearrange(torch.cat((img_x.unsqueeze(-1),img_y.unsqueeze(-1)),dim=2).expand(1, self.pe_H, self.pe_W, 2), 'b h w d -> b (h w) d').cuda()
        img_pos = self.img_pos_encoding(img_xy)  # [12, 1280, 64]
        pc_pos = self.pc_pos_encoding(pc_data_dict['points'][-1].unsqueeze(0))  # [2560, 128]
        # print(img_pos.shape, pc_pos.shape)
        #########
        # Transformer
        #########
        
        image_feature_8 = rearrange(img_s8_feature_map, 'b c h w -> b (h w) c') + img_pos
        pc_feature_8 = pc_feature_middle.unsqueeze(0) + pc_pos # [b, n, c]
        image_feature_mid, pc_features_fusion = self.transformer(image_feature_8, pc_feature_8)

        image_feature_mid = rearrange(image_feature_mid, 'b (h w) c -> b c h w', h=img_s8_feature_map.shape[2])
        pc_features_fusion = rearrange(pc_features_fusion, 'b n c -> b c n')
        
        # img_feature=self.img_feature_layer(img_s8_feature_map)
        
        # coarse level get score and feature
        coarse_pc_score = self.pc_score_layer(pc_features_fusion)
        coarse_img_score = self.img_score_layer(image_feature_mid)
        pc_feature_norm = F.normalize(torch.squeeze(pc_features_fusion), dim=0, p=2)
        img_feature_norm=F.normalize(image_feature_mid, dim=1,p=2)

        # upsample
        img_upsample_s4 = self.img_upsample_1(img_s8_feature_map, img_s4_feature_map)
        img_upsample_s2 = F.normalize(self.img_upsample_2(img_upsample_s4, img_s2_feature_map),dim=1,p=2)  # [1, 64, 80, 256]
        
        # fine transformer
        # fine_img_x, fine_img_y = torch.meshgrid(torch.linspace(0,3,4), torch.linspace(0,3,4), indexing='ij')
        # fine_img_xy=rearrange(torch.cat((fine_img_x.unsqueeze(-1),fine_img_y.unsqueeze(-1)),dim=2).expand(128, 4, 4, 2), 'b h w d -> b (h w) d').cuda()
        # fine_img_pos = self.fine_img_pos_encoding(fine_img_xy)
        
        if mode == 'train'  or mode == 'val':
            fine_pc_inline_feature = pc_decode_3[fine_pc_inline_index] #[128, 64]
            
            # [128, 64, 4, 4]
            fine_img_feature_patch = torch.squeeze(extract_patch(img_upsample_s2, fine_center_kpt_coors))
            fine_center_xy = None
            coarse_pc_points = None

        elif mode == 'test':
            coarse_xy, pc_inline_index = fine_process(coarse_pc_score, pc_feature_norm, img_feature_norm)
            coarse_pc_points = pc_data_dict['points'][-1][pc_inline_index]
            coarse_pc_index = point2node(pc_data_dict['points'][1], coarse_pc_points)
            # coarse_pc_index = search_point_index(pc_data_dict['points'][1].cpu().numpy(), coarse_pc_points.cpu().numpy())
            
            fine_center_xy = coarse_xy.cuda() * 4
            fine_img_feature_patch = torch.squeeze(extract_patch(img_upsample_s2, fine_center_xy))
            fine_img_feature_patch = rearrange(fine_img_feature_patch, 'n c h w -> n c (h w)')
            
            
            fine_pc_inline_feature = pc_decode_3[torch.squeeze(torch.Tensor(coarse_pc_index).long())]

        
        return img_feature_norm, pc_feature_norm, coarse_img_score, coarse_pc_score, fine_img_feature_patch, fine_pc_inline_feature, fine_center_xy, coarse_pc_points


def fine_process(coarse_pc_score, coarse_pc_feature, coarse_img_feature):
    coarse_pc_score = torch.squeeze(coarse_pc_score)
    pc_inline_index = torch.where(coarse_pc_score >= 0.9)[0]
    # print(len(pc_inline_index))
    coarse_pc_inline_feature = coarse_pc_feature[:, pc_inline_index.long()] # [C, N]
    coarse_img_feature_flatten = torch.squeeze(rearrange(coarse_img_feature, 'b c h w -> b c (h w)'))
    # [1280, 391]
    coarse_feature_distance = 1 - torch.sum(coarse_img_feature_flatten.unsqueeze(-1) * coarse_pc_inline_feature.unsqueeze(-2), dim=0)
    # mask = coarse_feature_distance <= 0.2
    # dist = coarse_feature_distance * mask
    
    # index in range 1280
    corr_img_index = torch.argmin(coarse_feature_distance, dim=0)
    img_x=torch.linspace(0,coarse_img_feature.size(-1)-1,coarse_img_feature.size(-1)).view(1,-1).expand(coarse_img_feature.size(-2),coarse_img_feature.size(-1)).unsqueeze(0).cuda()
    img_y=torch.linspace(0,coarse_img_feature.size(-2)-1,coarse_img_feature.size(-2)).view(-1,1).expand(coarse_img_feature.size(-2),coarse_img_feature.size(-1)).unsqueeze(0).cuda()
    img_xy=rearrange(torch.cat((img_x,img_y),dim=0), 'c h w -> c (h w)')
    coarse_xy = img_xy[:,corr_img_index]
    coarse_img_mask = (coarse_xy[0] >= 2) & (coarse_xy[0]<=62) & (coarse_xy[1]<=18) & (coarse_xy[1] >= 2)
    coarse_xy_inline = coarse_xy[:, coarse_img_mask]
    pc_inline_index = pc_inline_index[coarse_img_mask]
    return coarse_xy_inline, pc_inline_index

def search_point_index(source_points, target_points):
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


def extract_patch(feature_map, center_points ,size=4):
    '''
    Extract patch on feature_map through center points
    :param center_points: coarse points project to fine feature map in shape [N, 2]
    :param feature_map: fine feature map in shape [B, C, H, W]
    :param size: size of patch in type int
    '''
    left_top = torch.floor(center_points - size/2)
    right_bottom = torch.floor(center_points + size/2)
    patch_list = []
    for i in range(center_points.shape[1]):
        left = int(left_top[0, i].item())
        top = int(left_top[1, i].item())
        right = int(right_bottom[0, i].item())
        bottom = int(right_bottom[1, i].item())
        patch = feature_map[:, :, top:bottom, left:right]
        assert patch.shape==(feature_map.size(0), feature_map.size(1),4, 4)
        patch_list.append(patch)
    # 将 patch 列表转换为张量
    patch_tensor = torch.stack(patch_list)
    return patch_tensor

def square_distance(src, tgt, normalize=False):
    '''
    Calculate Euclide distance between every two points
    :param src: source point cloud in shape [B, N, C]
    :param tgt: target point cloud in shape [B, M, C]
    :param normalize: whether to normalize calculated distances
    :return:
    '''

    B, N, _ = src.shape
    _, M, _ = tgt.shape
    dist = -2. * torch.matmul(src, tgt.permute(0, 2, 1).contiguous())
    if normalize:
        dist += 2
    else:
        dist += torch.sum(src ** 2, dim=-1).unsqueeze(-1)
        dist += torch.sum(tgt ** 2, dim=-1).unsqueeze(-2)

    dist = torch.clamp(dist, min=1e-12, max=None)
    return dist


def point2node(nodes, points):
    '''
    Assign each point to a certain node according to nearest neighbor search
    :param nodes: [M, 3]
    :param points: [N, 3]
    :return: idx [N], indicating the id of node that each point belongs to
    '''
    # M, _ = nodes.size()
    # N, _ = points.size()
    dist = square_distance(points.unsqueeze(0), nodes.unsqueeze(0))[0]

    idx = dist.topk(k=1, dim=-1, largest=False)[1] #[B, N, 1], ignore the smallest element as it's the query itself

    idx = idx.squeeze(-1)
    return idx


class CoFiI2P_wrapper(nn.Module):
    """model wrapper for efficiency analysis"""
    def __init__(self, opt):
        super().__init__()
        self.cofii2p = CoFiI2P(opt)
    
    def forward(self,inputs):
        return self.cofii2p.forward(*inputs)


    