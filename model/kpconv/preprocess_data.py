
# from functools import partial
import open3d as o3d
from open3d.ml.torch.layers import FixedRadiusSearch, KNNSearch
import numpy as np
import torch
# from kpconv.preprocess_data_kpconv import*
import time
# Stack mode utilities


def radius_search(pcd_tree, lengths, query_pcd, radius, neighbor_limits, mode=None):
    '''
        pcd_tree: KDTree of database points
        query_pcd: query points
    '''
    neighbor_list = []
    # print(len(query_pcd))
    for i in range(lengths):
        [k, idx, coors] = pcd_tree.search_hybrid_vector_3d(query_pcd.points[i], radius=radius, max_nn=neighbor_limits)
        neighbor_list.append(np.asarray(idx))
    
    shapes = [x.shape[0] for x in neighbor_list]
    # 找到所有向量中的最大维度
    max_len = max(shapes)
    # 对每个向量进行填充
    if mode == 'upsample':
        padded = [np.pad(x, (0, max_len - x.shape[0]), mode='constant', constant_values=lengths//2 - 1) for x in neighbor_list]
    else:
        padded = [np.pad(x, (0, max_len - x.shape[0]), mode='constant', constant_values=lengths - 1) for x in neighbor_list]
    # 将所有向量拼接成一个矩阵
    neighbor = np.vstack(padded)
    return neighbor


def precompute_point_cloud_stack_mode(points, intensity, normals, lengths, num_stages):
    # assert num_stages == len(neighbor_limits)
    radius_num = 128
    points_list = []
    lengths_list = []
    neighbors_list = []
    subsampling_list = []
    upsampling_list = []

    pcd = o3d.geometry.PointCloud()
    pcd.points=o3d.utility.Vector3dVector(np.transpose(points))
    # intensity_max=np.max(intensity)

    # fake_colors=np.zeros((points.shape[1],3))
    # fake_colors[:,0:1]=np.transpose(intensity)/intensity_max

    # pcd.colors=o3d.utility.Vector3dVector(fake_colors)
    # pcd.normals=o3d.utility.Vector3dVector(np.transpose(normals))
    # grid subsampling
    for i in range(num_stages):
        if i > 0:
            # random sample half points of last stage(except stage1)
            pcd = pcd.random_down_sample(0.5)
            # print(pcd.shape)

        # down_pcd_points=np.transpose(np.asarray(points.points))
        # intensity=np.transpose(np.asarray(points.colors)[:,0:1])*intensity_max
        # sn=np.transpose(np.asarray(points.normals))
        points_list.append(torch.Tensor(np.asarray(pcd.points)))
        lengths_list.append(int(lengths))
        lengths = lengths // 2
        # intensity_list.append(intensity)
        # normals_list.append(sn)

    # radius search
    for i in range(num_stages):
        nsearch = KNNSearch(return_distances=True)
        # fixed_radius_search = FixedRadiusSearch(return_distances=True)

        cur_points = points_list[i]
        # start = time.time()
        # pcd_tree = o3d.geometry.KDTreeFlann(cur_points)
        # neighbors = radius_search(pcd_tree, lengths_list[i], cur_points, radius=radius, neighbor_limits=100)
        ml_neighbors = nsearch(cur_points, cur_points,radius_num)
        # fixed_neighbors = fixed_radius_search(cur_points, cur_points, radius=0.5)
        # end = time.time()
        # print(end-start)

        neighbors = ml_neighbors.neighbors_index.reshape(lengths_list[i], radius_num)
        neighbors_list.append(neighbors)

        if i < num_stages - 1:
            sub_points = points_list[i + 1]
            # sub_lengths = lengths_list[i + 1]

            subsampling = nsearch(cur_points, sub_points, radius_num).neighbors_index.reshape(lengths_list[i+1], radius_num)
            subsampling_list.append(subsampling)

            upsampling = nsearch(sub_points, cur_points, radius_num).neighbors_index.reshape(lengths_list[i], radius_num)
            # upsampling = radius_search(sub_pcdtree, lengths_list[i], cur_points, radius=radius * 2, neighbor_limits=100, mode='upsample')
            upsampling_list.append(upsampling)

    return {
        'points': points_list,
        'lengths': lengths_list,
        'neighbors': neighbors_list,
        'subsampling': subsampling_list,
        'upsampling': upsampling_list,
            }

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


def knn(nodes, points, radius_num):
    '''
    Assign each point to a certain node according to nearest neighbor search
    :param nodes: [M, 3]
    :param points: [N, 3]
    :return: idx [N], indicating the id of node that each point belongs to
    '''
    # M, _ = nodes.size()
    # N, _ = points.size()
    dist = square_distance(points.unsqueeze(0), nodes.unsqueeze(0))[0]

    idx = dist.topk(k=radius_num, dim=-1, largest=False)[1] #[B, N, 1], ignore the smallest element as it's the query itself
    return idx

def precompute_point_cloud_cuda(points, intensity, normals, lengths, num_stages):
    # assert num_stages == len(neighbor_limits)
    radius_num = 128
    points_list = []
    lengths_list = []
    neighbors_list = []
    subsampling_list = []
    upsampling_list = []

    pcd = o3d.geometry.PointCloud()
    pcd.points=o3d.utility.Vector3dVector(np.transpose(points))

    # subsampling
    for i in range(num_stages):
        if i > 0:
            # random sample half points of last stage(except stage1)
            pcd = pcd.random_down_sample(0.5)
            # print(pcd.shape)

        # down_pcd_points=np.transpose(np.asarray(points.points))
        # intensity=np.transpose(np.asarray(points.colors)[:,0:1])*intensity_max
        # sn=np.transpose(np.asarray(points.normals))
        points_list.append(torch.Tensor(np.asarray(pcd.points)))
        lengths_list.append(int(lengths))
        lengths = lengths // 2
        # intensity_list.append(intensity)
        # normals_list.append(sn)

    # radius search
    for i in range(num_stages):

        cur_points = points_list[i]
        # start = time.time()
        # pcd_tree = o3d.geometry.KDTreeFlann(cur_points)
        # neighbors = radius_search(pcd_tree, lengths_list[i], cur_points, radius=radius, neighbor_limits=100)
        ml_neighbors = knn(cur_points, cur_points,radius_num)

        # end = time.time()
        # print(end-start)

        # neighbors = ml_neighbors.neighbors_index.reshape(lengths_list[i], radius_num)
        neighbors_list.append(ml_neighbors)

        if i < num_stages - 1:
            sub_points = points_list[i + 1]
            # sub_lengths = lengths_list[i + 1]

            subsampling = knn(cur_points, sub_points, radius_num)
            subsampling_list.append(subsampling)
            upsampling = knn(sub_points, cur_points, radius_num)
            upsampling_list.append(upsampling)

    return {
        'points': points_list,
        'lengths': lengths_list,
        'neighbors': neighbors_list,
        'subsampling': subsampling_list,
        'upsampling': upsampling_list,
            }
