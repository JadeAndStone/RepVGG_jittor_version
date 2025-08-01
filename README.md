# RepVGG_jittor_version
## A simple reproduction of RepVGG based on jittor
## 复现论文简介
RepVGG: Making VGG-style ConvNets Great Again (CVPR-2021) (PyTorch)

原项目地址
https://github.com/DingXiaoH/RepVGG

RepVGG——一种基于经典 VGG 架构的改良架构：

1. 训练部分：类似 ResNet 引入多分支，提升训练的稳定性，进而能搭建足够深的网络进行学习，优化精度
2. 部署部分：将训练时的多分支结构重参数化合并为3x3的卷积层(包括 BN 层)，利用 VGG 结构简单且 plain 的优势，极大提升推理速度
## 实验环境
操作系统： (WSL) Ubuntu 22.04

显卡： NVIDIA GeForce RTX 4060 Laptop GPU

jittor版本： 1.3.9.14 (python 3.7.3)

pytorch版本： 2.7.1+cu128 (python 3.10.18)

## 数据集
原论文中采用 Imagenet 完整数据集

鉴于本地算力资源有限，故采用 Mini-Imagenet (54000_train && 6000_val)
（经均匀采样脚本整理）
### 原数据集结构

├── mini-imagenet: 数据集根目录

     ├── images: 所有图片（60000）

     ├── train.csv: 原训练集标签（38400）

     ├── val.csv: 原验证集标签（9600）

     └── test.csv: 原测试集标签（12000）

原数据集三个部分的样本类别是不重合的，需经过均匀随机采样整理才适用本项目
### 整理后数据集结构

├── mini-imagenet: 数据集根目录

     ├── train_img: 训练集图片（54000）

     └── val_img: 验证集图片（6000）

数据集下载链接

百度网盘下载：

链接: https://pan.baidu.com/s/1Uro6RuEbRGGCQ8iXvF2SAQ 密码: hl31

https://blog.csdn.net/qq_37541097/article/details/113027489
## 实验记录
| |jittor|pytorch|
|------|------|------|
|num_epoch (with Earlystopper)| 20 | 50 |
|origin_deploy_time| 5.839050531387329 s| 6.785406827926636 s|
|rep_deploy_time| 1.8714940547943115 s| 4.278181314468384 s|
|origin_top1_acc| 0.48216666666666674 | 0.4884999841451645 |
|rep_top1_acc| 0.4821666666666667 | 0.4886666541298231 |
|origin_top5_acc| 0.764 | 0.7664999773104986 | 0.7664999773104986 |
|rep_top5_acc| 0.7640000000000001 | 0.7663333127895992 |
## 实验分析
由上不难看出，rep 前后的模型精度基本一致（经验证，在归一化条件下，输出 logit 误差在1e-7量级），而 rep 后的推理速度有明显提升，这与论文结论一致。

而且，对比 jittor 和 pytorch 的结果可以看出，jittor 的速度提升幅度更大，由于即时编译运算符的特性，jittor 能够很好地适应 rep 操作后运算类型较少的特性，故而优势更为明显。

此外，jittor 的整体推理速度也显著优于 pytorch 。jittor 的动态编译过程相比较传统的静态编译，使其可以在运行时获得更多的额外信息，如计算图上下文，形状信息等等，这些信息都可以进一步用于提升算子性能。