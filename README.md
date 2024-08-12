<h2> 
<a href="https://whu-usi3dv.github.io/CoFiI2P/" target="_blank">CoFiI2P: Coarse-to-Fine Correspondences-Based Image-to-Point Cloud Registration</a>
</h2>

This is the official PyTorch implementation of the following publication:

> **CoFiI2P: Coarse-to-Fine Correspondences-Based Image-to-Point Cloud Registration**<br/>
> [Shuhao Kang*](https://kang-1-2-3.github.io/), [Youqi Liao*](https://martin-liao.github.io/), , [Jianping Li](https://kafeiyin00.github.io/), [FuxunLiang](https://scholar.google.com/citations?user=0Ds4eg8AAAAJ&hl=zh-CN&oi=ao), [Yuhao Li](https://whu-lyh.github.io/), [Xianghong Zou](https://scholar.google.com/citations?hl=zh-CN&user=vTQOkJwAAAAJ), [Fangning Li](http://cki.com.cn/en/), [Xieyuanli Chen](https://xieyuanli-chen.com/), [Zhen Dong](https://dongzhenwhu.github.io/index.html), [Bisheng Yang](https://3s.whu.edu.cn/info/1025/1415.htm)<br/>
> *Arxiv 2023*<br/>
> [Paper] | [**Arxiv**](https://arxiv.org/abs/2309.14660v2) | [**Project-page**](https://whu-usi3dv.github.io/CoFiI2P/) | [**Video**](https://www.youtube.com/ovbedasXuZE)


## 🔭 Introduction
<p align="center">
<strong>TL;DR: CoFiI2P is a coarse-to-fine framework for image-to-point cloud registration task.</strong>
</p>
<img src="./motivation.png" alt="Motivation" style="zoom:25%; display: block; margin-left: auto; margin-right: auto; max-width: 100%;">

<p align="justify">
<strong>Abstract:</strong> —Image-to-point cloud (I2P) registration is a fundamental task for robots and autonomous vehicles to achieve crossmodality data fusion and localization. Current I2P registration methods primarily focus on estimating correspondences at the
point or pixel level, often neglecting global alignment. As a result, I2P matching can easily converge to a local optimum if it lacks high-level guidance from global constraints. To improve the success rate and general robustness, this paper introduces CoFiI2P, a novel I2P registration network that extracts correspondences in a coarse-to-fine manner. First, the image and point cloud data are processed through a two-stream encoder-decoder network for hierarchical feature extraction. Second, a coarseto-fine matching module is designed to leverage these features and establish robust feature correspondences. Specifically, In the coarse matching phase, a novel I2P transformer module is employed to capture both homogeneous and heterogeneous global information from the image and point cloud data. This enables the estimation of coarse super-point/super-pixel matching pairs with discriminative descriptors. In the fine matching module, point/pixel pairs are established with the guidance of superpoint/super-pixel correspondences. Finally, based on matching pairs, the transform matrix is estimated with the EPnP-RANSAC algorithm. Experiments conducted on the KITTI Odometry dataset demonstrate that CoFiI2P achieves impressive results, with a relative rotation error (RRE) of 1.14 degrees and a relative translation error (RTE) of 0.29 meters, while maintaining realtime speed. These results represent a significant improvement of 84% in RRE and 89% in RTE compared to the current state-ofthe-art (SOTA) method. Additional experiments on the Nuscenes datasets confirm our method’s generalizability.
</p>

## 🆕 News
- 2024-05-08: [Project page](https://whu-usi3dv.github.io/CoFiI2P/) (with introduction video) is available!🎉  
- 2024-05-24: [code](https://github.com/WHU-USI3DV/CoFiI2P) is available!🎉
- 2024-08-11: support [Nuscenes](https://www.nuscenes.org/) dataset now!🎉 

## 💻 Installation
An example for ```CUDA=11.6``` and ```pytorch=1.13.1```:
```
pip3 install fvcore
pip3 install open3d==0.17.0
pip3 install opencv-python
pip3 install torchvision=0.14.1
```
We will provide a Docker image for quick start.

## 🚅 Usage

### KITTI data preprocessing
You could download the processed data [here](https://portland-my.sharepoint.com/:u:/g/personal/siyuren2-c_my_cityu_edu_hk/EY_3Cwr3PhZHiNj_ijDPIp0BZx23H9T1J-wrmd6gqbgykw?e=4quHFS) or process it from source. For more details, please refer to [CorrI2P](https://github.com/rsy6318/CorrI2P).

### Nuscenes data preprocessing 
Due to the extremely large scale of processed data (200G approximately), we only provide the data pre-processing code now. Please download the source data [here](https://www.nuscenes.org/nuscenes) and refer to following steps for building image-to-point cloud registration data:
- build datainfo:
```
python -m data.build_nuscenes.build_datainfo
```
- build dataset:
```
python -m data.build_nuscenes.build_dataset
```
**Please reserve enough space for processed Nuscenes dataset (200G+) !!!**

### Evaluation
For the KITTI Odometry dataset and Nuscenes dataset, we provide pre-trained models on [OneDrive](https://1drv.ms/f/s!AhENMf-PTXKL0Bq11ewNUGBbA9m3?e=xGthS3) and [Baidu Disk](https://pan.baidu.com/s/1Vo4WiyJ6J4sKgveFXrycVQ?pwd=51p0).
Please download the weights of CoFiI2P from webdrive and put them in a folder like ```ckpt/```.
Example: evaluate ```CoFiI2P``` on the KITTI Odometry dataset

```
python -m evaluation.eval_all ./checkpoints/cofii2p_kitti.t7 kitti
```
Above operation calculates the per-frame registration error and save intermediate results.

Then:
```
python -m evaluation.calc_result
```

### Training
Example: train ```CoFiI2P``` on the KITTI Odometry dataset
```
python -m train kitti
```

## 💡 Citation
If you find this repo helpful, please give us a star~.Please consider citing Mobile-Seed if this program benefits your project.
```
@article{kang2023cofii2p,
  title={CoFiI2P: Coarse-to-Fine Correspondences-Based Image-to-Point Cloud Registration},
  author={Shuhao Kang and Youqi Liao and Jianping Li and Fuxun Liang and Yuhao Li and Xianghong Zou and Fangning Li and Xieyuanli Chen and Zhen Dong and Bisheng Yang3},
  year={2023},
    eprint={2309.14660},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

## 🔗 Related Projects
We sincerely thank the excellent projects:
- [CorrI2P](https://github.com/rsy6318/CorrI2P) for correspondence-based I2P method;
- [DeepI2P](https://github.com/lijx10/DeepI2P) for pionerring I2P registration method;
- [Freereg](https://github.com/WHU-USI3DV/FreeReg) for excellent template; 