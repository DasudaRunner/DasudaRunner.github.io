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

<br>
- 本文为Haibo原创文章，转载请注明：[Haibo的主页](https://dasuda.top)

- 如果对本站的文章有疑问或者有合作需求的，可以联系QQ: 827091497，或者发送邮件到：[haibo.david@qq.com](mailto:haibo.david@qq.com) 。

<br>
### 正文：

&emsp;&emsp;1、新买的TX2建议直接刷机，不要用自带系统，刷机时安装jetpack中所有包。

&emsp;&emsp;2、刷机：当安装完系统镜像时，会提示你重启，先重启，再安装其他的包（cuda、cudnn等）(注意：将系统镜像那一项以上的都选为no action，只安装target board)。

&emsp;&emsp;3、等待安装完成后，开始安装tensorflow，采用编译安装，详见[链接](https://syed-ahmed.gitbooks.io/nvidia-jetson-tx2-recipes/content/first-question.html ),该链接教程已实验成功，不过tensorflow版本为1.0。

&emsp;&emsp;4、推荐安装jupyter notebook。

#### 注意事项：
&emsp;&emsp; 1、编译前增加swap空间为8G。

&emsp;&emsp; 2、测试程序记得配置session：
sess =tf.Session(config=tf.ConfigProto(log_device_placement=True,allow_soft_placement=True))

&emsp;&emsp;之前一直没设置 allow_soft_placement参数，导致没有成功使用gpu，根据官方doc说明，这个是当指定设备不可用时寻找替代设备的标志位，至于为什么这样就可以了，还不太清楚。
