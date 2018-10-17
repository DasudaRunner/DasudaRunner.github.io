---
layout: post
title: "CUDA中共享内存中的bank conflict问题"
date: 2018-10-16
categories:
- CUDA
tag:
- CUDA
- bank conflict
---

在进行block内的thread同步时，会使用shared memory，那就不得不考虑它的bank conflict问题，当程序里面出现bank conflict时，程序运行效率会受到较大影响

### Bank in Shared Memory
在软件中，每个block都会有自己的共享内存（shared memory），同一个blcok中的所有thread可以使用共享内存进行通信及同步。在硬件中，每个SM具有96KB的共享内存（GTX 1070）来分给下面的block，而共享内存又被分为多个bank，而bank的个数则根据每个bank占用的字节数来决定的，在计算能力3.0之后，每个bank占用的字节数用户是可以配置的
```cpp
/**
 * CUDA shared memory configuration
 */
enum __device_builtin__ cudaSharedMemConfig
{
    cudaSharedMemBankSizeDefault   = 0,
    cudaSharedMemBankSizeFourByte  = 1,
    cudaSharedMemBankSizeEightByte = 2
};
```
例如：我们在调用内核函数时动态申请了128字节的共享内存，并且我们每个bank占用的字节数使用默认配置，即4字节，那么一共会有96/4=24个bank，则0-3字节分配在bank[0]，4-7分配在bank[1]，...，92-95分配在bank[23]，而96-99会分配在bank[0]，以此类推，也就是说GPU是采用平衡负载的思路去将共享内存分配给bank。

### Bank Conflict Problem

bank conflict问题发生在不同的thread对同一bank进行操作的时候，此时调度器会强制让这些冲突的thread顺序访问bank，这样就会使我们的程序效率降低，但我们**一般**可以在设计程序时避免这种情况
