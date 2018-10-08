---
layout: post
title: "卸载cuda8和cudnn6以及安装cuda9和cudnn7"
date: 2018-09-26
categories:
- TensorFlow
tag:
- TensorFlow
- CUDA
---

我本来的环境是`cuda8.0`+`cudnn6.0`+`driver384.130`+`tensorflow-gpu==1.4`。
真是逼着你升级，我已经坚持1.4很久了，但是很多新的特性不能用，无奈只能升级，不然的话谁愿意折腾，其中显卡驱动不需要变。

### 卸载cuda8.0及cudnn6.0

```bash
sudo apt-get remove cuda 

sudo apt-get autoclean

sudo apt-get remove cuda*

cd /usr/local/

sudo rm -r cuda-8.0
```
其实上面已经卸载完cuda和cudnn，用库方式配置过cudnn的都知道，cudnn只是一个头文件和一堆链接库，在cuda文件夹里，删除即可。

### 卸载nvidia-cuda-toolkit

```bash
sudo apt-get autoremove nvidia-cuda-toolkit
```

### 下载cuda9.0和cudnn7.1.4

- 下载需要nvidia开发者账号

- 下载链接：[cuda9.0](https://developer.nvidia.com/cuda-90-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1604&target_type=deblocal)及[cudnn7.1.4](https://developer.nvidia.com/rdp/cudnn-archive)

- 其中cudnn选择`cuDNN v7.1.4 Library for Linux`下载，下来为压缩文件。

### 安装cuda9.0及cudnn7.1.4

- cuda安装

```bash
sudo dpkg -i cuda-repo-ubuntu1604-9-0-local_9.0.176-1_amd64.deb 

sudo apt-key add /var/cuda-repo-9-0-local/7fa2af80.pub

sudo apt-get update

sudo apt-get install cuda
```

- cuda环境变量设置

```bash
export PATH=/usr/local/cuda-9.0/bin${PATH:+:${PATH}}

#64位系统
export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

#32位系统
export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

- cudnn安装

在`~/Download`目录下，解压下载cudnn文件夹

```bash
sudo cp cuda/include/cudnn.h /usr/local/cuda/include

sudo cp -a cuda/lib64/libcudnn* /usr/local/cuda/lib64
```

### 安装nvidia-cuda-toolkit *(10月5号修正：此步骤不需要，详见[安装、编译踩坑](https://dasuda.top/%E7%BC%96%E8%AF%91%E5%AE%89%E8%A3%85/2018/10/01/%E5%AE%89%E8%A3%85-%E7%BC%96%E8%AF%91%E8%B8%A9%E5%9D%91/)中的"nvcc -V显示版本和实际安装版本不一致"问题)*

```bash
sudo apt-get install nvidia-cuda-toolkit
```

### 验证

```bash
nvcc -V #最后一行可显示cuda版本，可验证是否安装成功

cat /usr/local/cuda/version.txt #输出的是cuda版本，但只是验证版本，不代表安装一定成功

cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2 #从输出的宏定义可以查看cudnn的版本，只能验证版本，不代表安装成功
```

### 结语

`/usr/local/cuda`文件夹是安装cuda时安装的，我最开始一次的顺序是：卸载`cuda8.0`和`cudnn6.0`，安装`cuda9.0`和`cudnn7.1`，卸载`nvidia-cuda-toolkit`，在安装`nvidia-cuda-toolkit`，发现`cuda/`文件夹没了，尴尬，应该是卸载`nvidia-cuda-toolkit`时删掉了，所以要安装上文的顺序去卸载和安装，一般不会出错。
