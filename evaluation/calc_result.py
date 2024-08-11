import numpy as np

def dist_with_thrs(r_error,t_error,r_thrs,t_thrs):
    succ_mask = np.logical_and(r_error < r_thrs, t_error < t_thrs)
    # good_index = np.where((r_error < r_thrs) & (t_error < t_thrs))[0]
    # good_rate=np.sum(r_error<r_thrs&(t_error<t_thrs)) / len(r_error)
    print("--------------error calculation---------------------")
    print("r_thrs: %.2f, t_thrs: %.2f"%(r_thrs,t_thrs))
    print('rot thrs: %.4f, trans thrs: %.4f, successful rate %0.2f %%'%(r_thrs,t_thrs,succ_mask.sum() / len(succ_mask) * 100.0))
    succ_r = r_error[succ_mask]
    succ_t = t_error[succ_mask]
    succ_r_mean,succ_r_std = succ_r.mean(),succ_r.std()
    succ_t_mean,succ_t_std = succ_t.mean(),succ_t.std()
    print('succ_r_mean: %.2f, succ_r_std: %.2f'%(succ_r_mean,succ_r_std))
    print('succ_t_mean: %.2f, succ_t_std: %.2f'%(succ_t_mean,succ_t_std))
    print("----------Done!----------")


r_error = np.load('nuscenes_r_error.npy')
t_error = np.load('nuscenes_t_error.npy')

r_thrs_0 = 1e5
t_thrs_0 = 1e5

r_thrs_1 = 45
t_thrs_1 = 10

r_thrs_2 = 10
t_thrs_2 = 5

dist_with_thrs(r_error,t_error,r_thrs_0,t_thrs_0)
dist_with_thrs(r_error,t_error,r_thrs_1,t_thrs_1)
dist_with_thrs(r_error,t_error,r_thrs_2,t_thrs_2)