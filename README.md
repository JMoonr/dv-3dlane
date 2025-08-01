<br />
<p align="center">
  
  <h3 align="center"><strong>DV-3DLane: End-to-end Multi-modal 3D Lane Detection with Dual-view Representation</strong></h3>

<p align="center">
  <a href="https://openreview.net/forum?id=l1U6sEgYkb" target='_blank'>
    <img src="https://img.shields.io/badge/ICLR2024-lightblue.svg">
  </a>
  <a href="" target='_blank'>
    <img src="https://visitor-badge.laobi.icu/badge?page_id=JMoonr.dv-3dlane&left_color=gray&right_color=lightpink">
  </a>
    <a href="https://github.com/JMoonr/dv-3dlane" target='_blank'>
     <img src="https://img.shields.io/github/stars/JMoonr/dv-3dlane?style=social">
  </a>
  
</p>


![fig2](/assets/main.png)  

## News
  - Now:  Releasing the code. 
    
    We apologize for the delayed release of the code and instructions for this work. We should have made the repository available much earlier. If you encounter any issues with the implementation or have questions about the code, please feel free to contact me at [222010057@link.cuhk.edu.cn](mailto:222010057@link.cuhk.edu.cn).

  - **2024-01-15** :confetti_ball: Our new work [DV-3DLane: End-to-end Multi-modal 3D Lane Detection with Dual-view Representation](https://openreview.net/pdf?id=l1U6sEgYkb) is accepted by ICLR2024.



## Environments
To set up the required packages, please refer to the [installation guide](./docs/install.md).

## Data
Please follow [data preparation](./docs/prepare_data.md) to prepare the dataset.

## Train & evaluation
Please follow the steps in [train and evaluation](./docs/train_eval.md#train).

## Acknowledgment

This library is inspired by [LATR](https://github.com/JMoonr/LATR), [OpenLane](https://github.com/OpenDriveLab/PersFormer_3DLane), [GenLaneNet](https://github.com/yuliangguo/Pytorch_Generalized_3D_Lane_Detection), [mmdetection3d](https://github.com/open-mmlab/mmdetection3d), [SparseInst](https://github.com/hustvl/SparseInst), [PillarNet](https://github.com/VISION-SJTU/PillarNet-LTS), and many other related works, we thank them for sharing the code and datasets.

## Citation
If you find DV-3DLane is useful for your research, please consider citing our paper:

```tex
@inproceedings{
luo2024dvdlane,
title={{DV}-3{DL}ane: End-to-end Multi-modal 3D Lane Detection with Dual-view Representation},
author={Yueru Luo and Shuguang Cui and Zhen Li},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=l1U6sEgYkb}
}
```