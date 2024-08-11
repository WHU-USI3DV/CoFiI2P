import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib.ticker import PercentFormatter


plt.rcParams.update({'font.size': 18})
plt.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


corr_ir = np.load('corri2p_ir_data.npy')
cofi_ir = np.load('cofii2p_ir_data.npy')
cofi_rmse = np.load('cofii2p_nuscenes_rmse_data.npy')
corr_rmse = np.load('corri2p_rmse_data_nuscenes.npy')
cofi_rmse = cofi_rmse[0]
corr_rmse = corr_rmse[0]

print(corr_ir.shape, corr_rmse.shape)
print(corr_rmse.max())
print(cofi_rmse.max())
x = np.arange(1, 10.2, 0.2)

plt.figure()  
plt.xlim((1, 10))
plt.ylim((0, 100))
plt.xticks(range(1, 11))
plt.xlabel("Threshold (pixel)")     
plt.ylabel("Inlier Ratio (%)")        

plt.plot(x,cofi_ir[5:]*100, label='CoFiI2P')
plt.plot(x, corr_ir[5:]*100, label='CorrI2P')
plt.legend()
plt.tight_layout() 
plt.savefig('ir_kitti.png', dpi=600)
plt.close()
