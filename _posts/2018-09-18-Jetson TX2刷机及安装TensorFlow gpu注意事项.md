---
layout: post
title: "Jetson TX2刷机及安装TensorFlow gpu注意事项"
date: 2018-09-18
categories:
- 移动平台
tag:
- 移动平台
- TX2
- TensorFlow
---

这是在2017年12月份时候用TX2做目标跟踪时，搭建的深度学习系统，奈何TX2的运算能力还是不太够，
就没在继续再使用这个平台了

### 正文：

- 新买的TX2建议直接刷机，不要用自带系统，刷机时安装jetpack中所有包。

- 刷机：当安装完系统镜像时，会提示你重启，先重启，再安装其他的包（cuda、cudnn等）(注意：将系统镜像那一项以上的都选为no action，只安装target board)。

- 等待安装完成后，开始安装tensorflow，采用编译安装，详见[链接](https://syed-ahmed.gitbooks.io/nvidia-jetson-tx2-recipes/content/first-question.html ),该链接教程已实验成功，不过tensorflow版本为1.0。

- 推荐安装jupyter notebook。

### 注意事项：

- 编译前增加swap空间为8G。

- 测试程序记得配置session：

```python
sess =tf.Session(config=tf.ConfigProto(log_device_placement=True,allow_soft_placement=True))
```

之前一直没设置 allow_soft_placement参数，导致没有成功使用gpu，根据官方doc说明，这个是当指定设备不可用时寻找替代设备的标志位，至于为什么这样就可以了，还不太清楚。
