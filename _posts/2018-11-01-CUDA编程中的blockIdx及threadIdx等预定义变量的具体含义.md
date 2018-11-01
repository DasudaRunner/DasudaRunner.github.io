---
layout: post
title: "CUDA编程中的blockIdx及threadIdx等预定义变量的具体含义"
date: 2018-11-01
categories:
- CUDA
tag:
- CUDA
- 编程语言
---

最近在用CUDA复现图像处理算法时，发现对threadIdx等这类基本概念还是模棱两可，故在此详细总结一下，加深印象。

### 看图说来
<img src="/assets/images/posts/cuda/cuda.jpg"> 

图中，红色的小格子表示一个thread，黄色的2*2格子表示一个block，thread中的黑色数字x表示threadIdx.x，y表示threadIdx.y。block中的红色加粗x表示blockIdx.x，y表示blockIdx.y。

- threadIdx：它是相对于它所在的block而言的，x表示当前thread所在的列数，y表示所在的行数，你会发现第一个block的第一个thread为（x:0 , y:0），第二个block的第一个thread也为（x:0 , y:0）。
- blockIdx：它是相对grid而言的，x表示当前block所在的列数，y表示所在的行数。
- blockDim：blockDim.x表示一个block中一共有几列thread，blockDim.y表示一个block中一共有几行thread。
- gridDim：gridDim.x表示一个grid中一共有几列block，blockDim.y表示一个grid中一共有几行block。

### 实际应用

在编程中，当处理图像这类高维矩阵时，我们也会使用三维的block和grid，所以thread的全局ID计算会常用到。

```cpp
const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
const unsigned int idy = threadIdx.y + blockDim.y * blockIdx.y;

//tid即thread的全局ID
const unsigned int tid = idx + blockDim.x * gridDim.x * idy;
```