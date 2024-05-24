<h2> 
<a href="https://whu-usi3dv.github.io/Mobile-Seed/" target="_blank">CoFiI2P: Coarse-to-Fine Correspondences-Based Image-to-Point Cloud Registration</a>
</h2>

This is the official PyTorch implementation of the following publication:

> **CoFiI2P: Coarse-to-Fine Correspondences-Based Image-to-Point Cloud Registration**<br/>
> [Shuhao Kang](https://kang-1-2-3.github.io/), [Youqi Liao](https://martin-liao.github.io/), , [Jianping Li](https://kafeiyin00.github.io/), [FuxunLiang](https://scholar.google.com/citations?user=0Ds4eg8AAAAJ&hl=zh-CN&oi=ao), [Yuhao Li](https://whu-lyh.github.io/), [Xianghong Zou](https://scholar.google.com/citations?hl=zh-CN&user=vTQOkJwAAAAJ), [Fangning Li](http://cki.com.cn/en/), [Xieyuanli Chen](https://xieyuanli-chen.com/), [Zhen Dong](https://dongzhenwhu.github.io/index.html), [Bisheng Yang](https://3s.whu.edu.cn/info/1025/1415.htm)<br/>
> *Arxiv 2023*<br/>
> [Paper] | [**Arxiv**](https://arxiv.org/abs/2309.14660v2) | [**Project-page**](https://whu-usi3dv.github.io/CoFiI2P/) | [**Video**](https://www.youtube.com/ovbedasXuZE)


## 🔭 Introduction
<p align="center">
<strong>TL;DR: CoFiI2P is a coarse-to-fine framework for image-to-point cloud registration task.</strong>
</p>
<img src="./workflow.png" alt="Motivation" style="zoom:25%;">

<p align="justify">
<strong>Abstract:</strong> Image-to-point cloud (I2P) registration is a fundamental task for robots and autonomous vehicles to achieve cross-modality data fusion and localization. Existing I2P registration methods estimate correspondences at the point/pixel level, often overlooking global alignment. However, I2P matching can easily converge to a local optimum when performed without high-level guidance from global constraints. To address this issue, this paper introduces CoFiI2P, a novel I2P registration network that extracts correspondences in a coarse-to-fine manner to achieve the globally optimal solution. First, the image and point cloud data are processed through a Siamese encoder-decoder network for hierarchical feature extraction. Second, a coarse-to-fine matching module is designed to leverage these features and establish robust feature correspondences. Specifically, In the coarse matching phase, a novel I2P transformer module is employed to capture both homogeneous and heterogeneous global information from the image and point cloud data. This enables the estimation of coarse super-point/super-pixel matching pairs with discriminative descriptors. In the fine matching module, point/pixel pairs are established with the guidance of super-point/super-pixel correspondences. Finally, based on matching pairs, the transform matrix is estimated with the EPnP-RANSAC algorithm. Extensive experiments conducted on the KITTI dataset demonstrate that CoFiI2P achieves impressive results, with a relative rotation error (RRE) of 1.14 degrees and a relative translation error (RTE) of 0.29 meters. These results represent a significant improvement of 84% in RRE and 89% in RTE compared to the current state-of-the-art (SOTA) method.
</p>

## 🆕 News
- 2024-05-08: [Project page](https://whu-usi3dv.github.io/CoFiI2P/) (with introduction video) is available!🎉  
- 2024-05-24: [code](https://github.com/WHU-USI3DV/CoFiI2P) is available!🎉

## 💻 Installation

## 🚅 Usage
### Model
We upload our model on [Google Drive](https://drive.google.com/file/d/1xM0a_HYzcfMEod9Ttdy9Ob6eurld7sc3/view?usp=sharing) or [Baidu Disk](https://pan.baidu.com/s/162wEXwbOhqE4r5oaoo2TUA?pwd=d9qk) , please download it to the folder model.
### Evaluation
```
python eval_all.py
```

Then run 

```
python calc_result.py
```
### Training
```
python train.py
```
### Data preprocessing
We employ the same method as [[CorrI2P]](https://github.com/rsy6318/CorrI2P), you can download the dataset or process it from KITTI.
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
- 