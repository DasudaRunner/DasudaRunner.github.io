---
layout: post
title: "CUDA中常用的host端及device端函数理解"
date: 2018-10-16
categories:
- CUDA
tag:
- CUDA
- GPU
---

总结一些常用的host及device端函数，解释其内部具体的运行流程，包括一些重要结构体里面的变量，学习CUDA编程之后才发现，GTX 1070性能是真的差啊，被老黄阉割的不忍直视，不过这系列的显卡定位也不是数据计算的。

### cudaMalloc
**函数原型：**`extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMalloc(void **devPtr, size_t size);`

**用法：**
```cpp
int* gpudata；
cudaMalloc((void**)&gpudata, sizeof(int)*1024);
```
**用途：**向GPU申请空间，类型为int型一维数组，大小为1024，GPU上的申请到的空间的地址保存在指针gpudata中，可能有同学会对`(void**)&gpudata`产生疑问，为什么是这种形式？下面解释一下，指针也是变量，只不过它里面的值是另外普通变量的地址，既然是变量，那它也就有地址，也就是说指针也是有地址的，我们也可以对指针进行取址操作。
- 第一个`&`，获得了gpudata的地址
- 靠近`&`的`*`，或得了gpudata里面储存的地址
- 第二个`*`，获得了gpudata里面地址对应的数据

### cudaDeviceProp结构体
- *name[256]：*保存GPU的名字，例如：GeForce GTX 1070

- *totalGlobalMem：*我们通常说的显存，例如GTX 1070 8G为 8504868864 bytes

- *sharedMemPerBlock：*block中share memory的大小，速度比全局存储空间快，例如GTX 1070 8G为 49152 bytes

- *regsPerBlock：*block中register memory的大小，里面存储的寄存器

- *warpSize：*线程束大小

- *maxThreadsPerBlock：*每个block最多的thread数量

- *maxThreadsDim[3]：*block每个维度的最大值

- *maxGridSize[3]：*grid每个维度的最大值

- *clockRate：*GPU的时钟频率

- *multiProcessorCount：*GPU上SM的数量，说实话老黄真是一个好刀客，本来性能就不行，还不停地砍性能


