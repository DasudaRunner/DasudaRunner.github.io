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
<img src="/assets/images/posts/idx/idx.jpg"> 

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

参考：CUDA官方论坛回答
```text
The best way to understand these values is to look at some of the schematics in the Introduction to CUDA Programming document, but I'll an explanation a shot.

Basically threadIdx.x and threadIdx.y are the numbers associated with each thread within a block. Let's say you declare your block size to be one dimensional with a size of 8 threads (normally you would want something in multiples of 32 like 192 or 256 depending on your specific code). The variable threadIdx.x would be simultaneously 0,1,2,3,4,5,6 and 7 inside each block. If you declared a two dimensional block size (say (3,3) ) then threadIdx.x would be 0,1,2 and you would now have a threadIdx.y value corresponding to 0,1,2. There are actually nine threads associated with the (3,3) block size. For instance, the thread indices (0,0) (0,1) (1,2) etc refer to independent threads. This convention is very useful for two dimensional applications like working with matrices. Remember, threadIdx.x starts at 0 for each block. Your block can be up to three dimensions which allows for a threadIdx.z index as well.

The blockIdx.x and blockIdx.y refers to the label associated with a block in a grid. You are allowed up to a 2-dimensional grid (allowing for blockIdx.x and blockIdx.y). Basically, the blockIdx.x variable is similar to the thread index except it refers to the number associated with the block. 

Let's say you want 2 blocks in a 1D grid with 5 threads in each block. Your threadIdx.x would be 0, 1,.....,4 for each block and your blockIdx.x would be 0 and 1 depending on the specific block.

Now, let's say you want to load an array of 10 values into a kernel using these two blocks of 5 threads. How would you do this since your thread index only goes 0 - 4 for each block? You would use a third parameter given in CUDA -- blockDim.x. This holds the size of the block (in this case blockDim.x = 5). You can refer to the specific element in the array by saying something like...

int idx = blockDim.x*blockIdx.x + threadIdx.x

This makes idx = 0,1,2,3,4 for the first block because blockIdx.x for the first block is 0. The second block picks up where the first left off because blockIdx.x = 1 and blockDim.x = 5. This makes idx = 5,6,7,8,9 for the second block.

Once again, refer to the beginner manual for more on this subject. Hope this helps.
```
