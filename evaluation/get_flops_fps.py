import argparse
import cv2
import torch
import itertools
import numpy as np 
import warnings

from ..data.kitti import kitti_pc_img_dataset
from ..data.options import Options_KITTI
from ..model.network import CoFiI2P_wrapper
from ..evaluation.eval_all import get_P_diff
warnings.filterwarnings('ignore')

def fps_params_flops(model, input):
    import time
    device = torch.device('cuda')
    model.eval()
    model.to(device)
    iterations = None
    
    # input = torch.randn(size).to(device)
    with torch.no_grad():
        for _ in range(10):
            model(input)

        if iterations is None:
            elapsed_time = 0
            iterations = 100
            while elapsed_time < 1:
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                t_start = time.time()
                for _ in range(iterations):
                    model(input)
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                elapsed_time = time.time() - t_start
                iterations *= 2
            FPS = iterations / elapsed_time
            iterations = int(FPS * 6)

        print('=========Speed Testing (network inference)=========')
        torch.cuda.synchronize()
        t_start = time.time()
        for _ in range(iterations):
            model(input)
        torch.cuda.synchronize()
        elapsed_time = time.time() - t_start
        latency = elapsed_time / iterations * 1000
    torch.cuda.empty_cache()
    FPS = 1000 / latency
    print(latency, ">>>latency. ")
    print(FPS, ">>>fps. ")

    from fvcore.nn import FlopCountAnalysis, ActivationCountAnalysis

    flops = FlopCountAnalysis(model, input)
    param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    acts = ActivationCountAnalysis(model, input)

    print(f"total flops : {flops.total()}")
    print(f"total activations: {acts.total()}")
    print(f"number of parameter: {param}")

def fps_pnpransac(model, input,K):
    import time
    device = torch.device('cuda')
    model.eval()
    model.to(device)
    iterations = None
    K = K[0].cpu().numpy()
    
    
    # input = torch.randn(size).to(device)
    with torch.no_grad():
        img_features,pc_features, coarse_img_score, coarse_pc_score\
                    , fine_img_feature_patch, fine_pc_inline_feature, fine_center_xy\
                    ,coarse_pc_points = model(input)
        fine_pc_inline_feature = fine_pc_inline_feature.unsqueeze(-1)
        dist = torch.cosine_similarity(fine_img_feature_patch.unsqueeze(-1), fine_pc_inline_feature.unsqueeze(-2))
        dist = torch.squeeze(dist)
        predict_index = torch.argmax(dist, dim=1)
        fine_xy = fine_center_xy - 2
        fine_xy[0] = fine_xy[0] + predict_index // 4
        fine_xy[1] = fine_xy[1] + predict_index % 4
        
        
        fine_xy = fine_xy.T.cpu().numpy()
        coarse_pc_points = coarse_pc_points.cpu().numpy()

    if iterations is None:
        elapsed_time = 0
        iterations = 100
        while elapsed_time < 1:
            t_start = time.time()
            for _ in range(iterations):
                cv2.solvePnPRansac(cameraMatrix=K, imagePoints=fine_xy, objectPoints=coarse_pc_points, iterationsCount=10000, distCoeffs=None)
            elapsed_time = time.time() - t_start
            iterations *= 2
        FPS = iterations / elapsed_time
        iterations = int(FPS * 6)

        print('=========Speed Testing (pose estimation)=========')
        torch.cuda.synchronize()
        t_start = time.time()
        for _ in range(iterations):
            cv2.solvePnPRansac(cameraMatrix=K, imagePoints=fine_xy, objectPoints=coarse_pc_points, iterationsCount=10000, distCoeffs=None)
            # torch.cuda.synchronize()
        elapsed_time = time.time() - t_start
        latency = elapsed_time / iterations * 1000
    FPS = 1000 / latency
    print(latency, ">>>latency.")
    print(FPS, ">>>FPS. ")

def fps_pipeline(model,input,P,K):
    import time
    device = torch.device('cuda')
    model.eval()
    model.to(device)
    iterations = None
    P = P[0].cpu().numpy()
    K = K[0].cpu().numpy()

    if iterations is None:
        elapsed_time = 0
        iterations = 100
        while elapsed_time < 1:
            t_start = time.time()
            for _ in range(iterations):
                # input = torch.randn(size).to(device)
                with torch.no_grad():
                    img_features,pc_features, coarse_img_score, coarse_pc_score\
                                , fine_img_feature_patch, fine_pc_inline_feature, fine_center_xy\
                                ,coarse_pc_points = model(input)
                    fine_pc_inline_feature = fine_pc_inline_feature.unsqueeze(-1)
                    dist = torch.cosine_similarity(fine_img_feature_patch.unsqueeze(-1), fine_pc_inline_feature.unsqueeze(-2))
                    dist = torch.squeeze(dist)
                    predict_index = torch.argmax(dist, dim=1)
                    fine_xy = fine_center_xy - 2
                    fine_xy[0] = fine_xy[0] + predict_index // 4
                    fine_xy[1] = fine_xy[1] + predict_index % 4
                    
                    fine_xy = fine_xy.T.cpu().numpy()
                    coarse_pc_points = coarse_pc_points.cpu().numpy()
                is_success,R,t,inliers = cv2.solvePnPRansac(cameraMatrix=K, imagePoints=fine_xy, objectPoints=coarse_pc_points, iterationsCount=10000, distCoeffs=None)
                if is_success is True:
                    # success_num += 1    
                    R,_=cv2.Rodrigues(R)
                    T_pred=np.eye(4)
                    T_pred[0:3,0:3]=R
                    T_pred[0:3,3:]=t
                    t_diff,angles_diff=get_P_diff(T_pred,P)
            elapsed_time = time.time() - t_start
            iterations *= 2
        FPS = iterations / elapsed_time
        iterations = int(FPS * 6)

        print('=========Speed Testing (pipeline)=========')
        torch.cuda.synchronize()
        t_start = time.time()
        for _ in range(iterations):
            with torch.no_grad():
                img_features,pc_features, coarse_img_score, coarse_pc_score\
                                , fine_img_feature_patch, fine_pc_inline_feature, fine_center_xy\
                                ,coarse_pc_points = model(input)
                fine_pc_inline_feature = fine_pc_inline_feature.unsqueeze(-1)
                dist = torch.cosine_similarity(fine_img_feature_patch.unsqueeze(-1), fine_pc_inline_feature.unsqueeze(-2))
                dist = torch.squeeze(dist)
                predict_index = torch.argmax(dist, dim=1)
                fine_xy = fine_center_xy - 2
                fine_xy[0] = fine_xy[0] + predict_index // 4
                fine_xy[1] = fine_xy[1] + predict_index % 4
                
                # K = K.cpu().numpy()
                fine_xy = fine_xy.T.cpu().numpy()
                coarse_pc_points = coarse_pc_points.cpu().numpy()
                is_success,R,t,inliers = cv2.solvePnPRansac(cameraMatrix=K, imagePoints=fine_xy, objectPoints=coarse_pc_points, iterationsCount=10000, distCoeffs=None)
                if is_success is True:
                    # success_num += 1    
                    R,_=cv2.Rodrigues(R)
                    T_pred=np.eye(4)
                    T_pred[0:3,0:3]=R
                    T_pred[0:3,3:]=t
                    t_diff,angles_diff=get_P_diff(T_pred,P)
        elapsed_time = time.time() - t_start
        latency = elapsed_time / iterations * 1000
    FPS = 1000 / latency
    print(latency, ">>>latency.")
    print(FPS, ">>>FPS. ")


def main():

    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('ckpt_path',type=str,help="checkpoint path",required = True)
    args = parser.parse_args()

    opt = Options_KITTI()
    model = CoFiI2P_wrapper(opt)
    model.cofii2p.load_state_dict(torch.load(args.ckpt_path),strict = True)
    model.cuda()
    model.eval()

    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')

    test_dataset = kitti_pc_img_dataset(opt)
    testloader=torch.utils.data.DataLoader(test_dataset,
                                           batch_size=opt.val_batch_size,
                                           shuffle=False,
                                           num_workers=4)

    data  = next(itertools.islice(enumerate(testloader),50,51),None)[1]

    img=data['img'].cuda()            
    pc_data_dict=data['pc_data_dict']
    for key in pc_data_dict:
        for j in range(len(pc_data_dict[key])):
            pc_data_dict[key][j] = torch.squeeze(pc_data_dict[key][j]).cuda()  
    pc_data_dict['feats'] = torch.squeeze(pc_data_dict['feats']).cuda()
    K=data['K'].cuda()
    P=data['P'].cuda()
    
    fine_center_kpt_coors = torch.squeeze(data['fine_center_kpt_coors']).cuda()  #[3, 128]
    fine_xy = torch.squeeze(data['fine_xy_coors']).cuda()
    fine_pc_inline_index = torch.squeeze(data['fine_pc_inline_index']).cuda()
    
    input = [pc_data_dict,img, fine_center_kpt_coors,fine_xy, fine_pc_inline_index, "test"]
    
    fps_params_flops(model, input)

    fps_pnpransac(model,input,K)

    fps_pipeline(model,input,P,K)

if __name__ == '__main__':
    main()