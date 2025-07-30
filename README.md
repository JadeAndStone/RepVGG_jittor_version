# RepVGG_jittor_version
## A simple reproduction of RepVGG based on jittor
## Environment
操作系统： (WSL) Ubuntu 22.04

显卡： NVIDIA GeForce RTX 4060 Laptop GPU

jittor版本： 1.3.9.14 (python 3.7.3)

pytorch版本： 2.7.1+cu128 (python 3.10.18)

## Dataset
原论文中采用Imagenet完整数据集

鉴于本地算力资源有限，故采用Mini-Imagenet(54000_train && 6000_val)

数据集下载链接

百度网盘下载：

链接: https://pan.baidu.com/s/1Uro6RuEbRGGCQ8iXvF2SAQ 密码: hl31

https://blog.csdn.net/qq_37541097/article/details/113027489
## Experiment Log
| |jittor|pytorch|
|------|------|------|
|origin_deploy_time| 5.839050531387329 s| 6.785406827926636 s|
|rep_deploy_time| 1.8714940547943115 s| 4.278181314468384 s|
|origin_top1_acc| 0.48216666666666674 | 0.4884999841451645 |
|rep_top1_acc| 0.4821666666666667 | 0.4886666541298231 |
|origin_top5_acc| 0.764 | 0.7664999773104986 | 0.7664999773104986 |
|rep_top5_acc| 0.7640000000000001 | 0.7663333127895992 |
## Analysis
