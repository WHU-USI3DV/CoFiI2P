import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

r_error = np.load('epoch_25_norm_r_error.npy')
t_error = np.load('epoch_25_norm_t_error.npy')

# plt.figure(1)
# plt.hist(t_error,bins=np.arange(0,15,0.2),weights=np.ones(t_error.shape[0]) / t_error.shape[0])
# plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
# # plt.xlabel('RTE (m)')
# # plt.ylabel('Percentage')
# plt.title('CoFiI2P RTE Histogram')
# plt.savefig('t_error_distribution.png', dpi=600)
# plt.figure(2)
# plt.hist(r_error,bins=np.arange(0,30,0.4),weights=np.ones(t_error.shape[0]) / t_error.shape[0])
# plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
# # plt.xlabel('RRE (°)')
# # plt.ylabel('Percentage')
# plt.title('CoFiI2P RRE Histogram')
# plt.savefig('r_error_distribution.png', dpi=600)

# bad_index=np.where((r_error<20)&(r_error>10)&(t_error>5)& (t_error<10))[0]
# bad_t_error_set= t_error[bad_index]
# bad_r_error_set= r_error[bad_index]
# for i in range(bad_t_error_set.shape[0]):
#     print(bad_index[i],bad_t_error_set[i],bad_r_error_set[i])
# idx=3104
# print('selected error',r_error[idx], t_error[idx])

print(np.mean(r_error), np.std(r_error), np.mean(t_error), np.std(t_error))
r_threshold = 10
t_threshold = 5
good_index = np.where((r_error < r_threshold) & (t_error < t_threshold))[0]
good_rate=np.sum((r_error<r_threshold)&(t_error<t_threshold))/np.shape(r_error)[0]
print('successful rate %0.4f'%good_rate)
good_r = r_error[good_index]
good_t = t_error[good_index]

bad_index = np.where((r_error > 45) | (t_error > 10))[0]
print(bad_index)
print(np.mean(good_r), np.std(good_r), np.mean(good_t), np.std(good_t))
# print(good_r.shape, good_t.shape)