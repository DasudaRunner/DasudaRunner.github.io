---
layout: post
title: "OpenCV中CUDA代码分析之hist篇"
date: 2018-11-20
categories:
- CUDA
tag:
- CUDA
- OpenCV
- DeltaCV
---

这篇文章是分析OpenCV中直方图部分的CUDA代码，代码位于`~/opencv-3.4.1/modules/cudaimgproc/src/cuda/hist.cu`，代码量：**少**。

### histogram256Kernel
```cpp
__global__ void histogram256Kernel(const uchar* src, int cols, int rows, size_t step, int* hist)
{
    __shared__ int shist[256]; //声明共享内存，用于提高计算效率，每个block分别计算完后，在进行汇总

    const int y = blockIdx.x * blockDim.y + threadIdx.y; //求出thread的全局y值
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;//thread在block中的全局ID

    shist[tid] = 0; //清空共享内存
    __syncthreads(); //同步

    if (y < rows)
    {
        const unsigned int* rowPtr = (const unsigned int*) (src + y * step); //获得当前thread需要处理的数据地址，step应该是Mat中的step相同定义，但是这里的运算运应该是被重载了，不清楚具体含义
        const int cols_4 = cols / 4; //应为每次要处理一个字节，所以看看列数能被分成几组
        for (int x = threadIdx.x; x < cols_4; x += blockDim.x)
        {
            unsigned int data = rowPtr[x];
			//每个thread每次读取4个字节（也就是说一个thread每次处理4个像素），也是就是一个warp读取128个字节，充分利用存储带宽
            Emulation::smem::atomicAdd(&shist[(data >>  0) & 0xFFU], 1);
            Emulation::smem::atomicAdd(&shist[(data >>  8) & 0xFFU], 1);
            Emulation::smem::atomicAdd(&shist[(data >> 16) & 0xFFU], 1);
            Emulation::smem::atomicAdd(&shist[(data >> 24) & 0xFFU], 1);
        }
		//前面处理的是满足被4整除的部分，因为图像的尺寸可能不是4的整数倍，所以这里单独处理一下剩余的部分
        if (cols % 4 != 0 && threadIdx.x == 0)
        {
            for (int x = cols_4 * 4; x < cols; ++x)//也就是遍历了cols_4 * 4到cols部分，循环次数<=3
            {
                unsigned int data = ((const uchar*)rowPtr)[x];
                Emulation::smem::atomicAdd(&shist[data], 1);
            }
        }
    }
    __syncthreads();
	//当所有现成完成计算后，将所有block计算的结果进行汇总，当某一bin的个数为0时，不必参与运算，减少了内存从冲突的情况zongjie
    const int histVal = shist[tid];
    if (histVal > 0)
        ::atomicAdd(hist + tid, histVal);
}

//主机函数
void histogram256(PtrStepSzb src, int* hist, cudaStream_t stream)
{
    const dim3 block(32, 8);
    const dim3 grid(divUp(src.rows, block.y));//向上取整

    histogram256Kernel<<<grid, block, 0, stream>>>(src.data, src.cols, src.rows, src.step, hist);
    cudaSafeCall( cudaGetLastError() );

    if (stream == 0)
        cudaSafeCall( cudaDeviceSynchronize() );
}
```
其中`divUp`函数的定义为，其实就是个向上取整的作用，现在我们的grid尺寸依然沿用这种方式：
```cpp
__host__ __device__ __forceinline__ int divUp(int total, int grain)
{
    return (total + grain - 1) / grain;
}
```

### 分析

首先从主机函数下手，看到block和grid的尺寸设计，不禁浮想联翩，为什么要设计成这样（其实blcok的大小刚好为256），为什么不每个线程分配一个像素，也就是thread数目 = 图像尺寸？他这样做意味着一个thread有可能需要处理多个像素，但是毕竟编写的人员肯定是这块的大佬，所以我们慢慢来分析这样做的优势，分析分为两块：存储和逻辑。

我们简要分析一下读取，程序里最大化读取效率，每次读取4个字节，也就是warp的合并访问达到了最大效率—128个字节。存储上，首先每个block计算结果储存到自己的shares memory上，原子加法存在延迟，另外，为了能够每次读取4个字节，程序中使用了类型强制转换，这在GPU上是低效的。

另外在《CUDA并行程序设计》一书中也分析了在这种情况下，原子操作的延迟远大于存储延迟。那么我们怎样才能减少这种延迟呢，那就是使用共享内存，将一部分全局内存存储的工作替换为共享内存的储存。

在计算能力为6.1的设备上发现一次读取4字节的方法比一次单字节读取还慢，而且使用共享内存也比不用慢,应该是图像本来就小(480*640)，所以太复杂的逻辑可能导致计算效率下降。

### 直方图均衡化

OpenCV中的实现各种跳转，均衡化过程非常简单，不知为何长达五六层的调用，而且找到最终的实现，发现一堆模板，重载运算符的，果断放弃，直接给出我在DelataCV中的实现[代码](https://github.com/DasudaRunner/DeltaCV/blob/master/cu/src/getHist.cu)。

