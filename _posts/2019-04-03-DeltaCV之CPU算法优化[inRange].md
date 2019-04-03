---
layout: post
title: "DeltaCV之CPU算法优化[inRange]"
date: 2019-04-02
categories:
- DeltaCV
tag:
- DeltaCV
- SSE
- SIMD
excerpt: 这篇文章我们先来认识一下DeltaCV中CPU上做的相关优化工作，使用SSE及AVX指令集对图像处理中的阈值分割过程进行重构，来提升计算效率。
---
* 目录
{:toc}
这篇文章我们先来认识一下DeltaCV中CPU上做的相关优化工作，使用SSE及AVX指令集对图像处理中的阈值分割操纵进行重构，来提升计算效率。

源文件: [https://github.com/DasudaRunner/DeltaCV/blob/master/cpu/src/SIMD/inRange.cpp](https://github.com/DasudaRunner/DeltaCV/blob/master/cpu/src/SIMD/inRange.cpp)

## OpenCV中的inRange()
OpenCV中的inRange()函数常常用来对图像的各个通道进行阈值分割，但是当图像尺寸过大时，该步骤较为耗时，故我们使用SIMD指令集的并行处理能力来提升像素的处理效率。

###　函数原型
```cpp
void inRange(InputArray src, InputArray lowerb, InputArray upperb, OutputArray dst)
```
他的工作原理就是在src上对每个通道做一次判断，在lowerb和upperb之间的为1，最后输出到dst上，所以dst是一副二值图。

## DeltaCV中的优化
针对OpenCV中的inRange()，我们主要做如下改进：当图像是单通道时,使用AVX中的256位数据格式,当图像为三通道时,使用SSE的128位数据格式(由于`_mm_shuffle_epi8`函数缘故).

### 函数原型
```cpp
    /*
     * src: input
     * dst: output
     * width:
     * height:
     * lower: lower boundary
     * upper: upper boundary
     */
    int inRange(unsigned char *src, unsigned char *dst, int width, int height,deltaCV::scalar lower, deltaCV::scalar upper)
```
其中`deltaCV::scalar`与`cv::Scalar`类似,定义见[DeltaCV](https://github.com/DasudaRunner/DeltaCV/blob/master/cpu/include/deltaCV/SIMD/DataTypes.hpp).
```cpp
assert(lower.channels()==upper.channels());
int channel = lower.channels();
if(channel==3)
{
	...
}else
{
    ...
}
```
我们这里需要根据输入图像的通道数分别进行处理.当通道数为3时,此时的优化也最明显(毕竟数据规模越大,并行的优势也就越大).

### 思路
由于SSE中的128位数据结构特点,我们可以每次处理16组像素点,为什么呢?因为在数据存储时,三通道像素点是依次排列的,像B G R B G R这样,而SIMD指令只能对128位数据做同样的处理,所以我们有必要使用三个`__m128i`分别储存三个通道的值,而且128/8=16,所以我们每次处理16组像素.

### 载入变量
```cpp
int blockSize = 16; //16*3
int block = height * width  / blockSize;

__m128i ch0_min_sse = _mm_setr_epi8(lower[0],lower[0],lower[0],lower[0],lower[0],lower[0],lower[0],lower[0],lower[0],lower[0],lower[0],lower[0],lower[0],lower[0],lower[0],lower[0]);
__m128i ch1_min_sse = _mm_setr_epi8(lower[1],lower[1],lower[1],lower[1],lower[1],lower[1],lower[1],lower[1],lower[1],lower[1],lower[1],lower[1],lower[1],lower[1],lower[1],lower[1]);
__m128i ch2_min_sse = _mm_setr_epi8(lower[2],lower[2],lower[2],lower[2],lower[2],lower[2],lower[2],lower[2],lower[2],lower[2],lower[2],lower[2],lower[2],lower[2],lower[2],lower[2]);
 __m128i ch0_max_sse = _mm_setr_epi8(upper[0],upper[0],upper[0],upper[0],upper[0],upper[0],upper[0],upper[0],upper[0],upper[0],upper[0],upper[0],upper[0],upper[0],upper[0],upper[0]);
__m128i ch1_max_sse = _mm_setr_epi8(upper[1],upper[1],upper[1],upper[1],upper[1],upper[1],upper[1],upper[1],upper[1],upper[1],upper[1],upper[1],upper[1],upper[1],upper[1],upper[1]);
__m128i ch2_max_sse = _mm_setr_epi8(upper[2],upper[2],upper[2],upper[2],upper[2],upper[2],upper[2],upper[2],upper[2],upper[2],upper[2],upper[2],upper[2],upper[2],upper[2],upper[2]);
```
### 循环处理
```cpp
for(int i=0; i<block; ++i,src+=blockSize*channel,dst+=blockSize)
{
	// 读取数据
	__m128i src1 = _mm_loadu_si128((__m128i *) (src + 0));
	__m128i src2 = _mm_loadu_si128((__m128i *) (src + blockSize*1));
	__m128i src3 = _mm_loadu_si128((__m128i *) (src + blockSize*2));

	//三通道分离
    __m128i Ch0 = _mm_shuffle_epi8(src1, _mm_setr_epi8(0, 3, 6, 9, 12, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1));
    Ch0 = _mm_or_si128(Ch0, _mm_shuffle_epi8(src2,_mm_setr_epi8(-1, -1, -1, -1, -1, -1, 2, 5, 8, 11, 14, -1, -1,-1, -1, -1)));
    Ch0 = _mm_or_si128(Ch0, _mm_shuffle_epi8(src3,_mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 4,7, 10, 13)));
    __m128i Ch1 = _mm_shuffle_epi8(src1,_mm_setr_epi8(1, 4, 7, 10, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1));
    Ch1 = _mm_or_si128(Ch1, _mm_shuffle_epi8(src2,_mm_setr_epi8(-1, -1, -1, -1, -1, 0, 3, 6, 9, 12, 15, -1, -1,-1, -1, -1)));
    Ch1 = _mm_or_si128(Ch1, _mm_shuffle_epi8(src3,_mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 5,8, 11, 14)));

	__m128i Ch2 = _mm_shuffle_epi8(src1,_mm_setr_epi8(2, 5, 8, 11, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1));
    Ch2 = _mm_or_si128(Ch2, _mm_shuffle_epi8(src2,_mm_setr_epi8(-1, -1, -1, -1, -1, 1, 4, 7, 10, 13, -1, -1, -1,-1, -1, -1)));
    Ch2 = _mm_or_si128(Ch2, _mm_shuffle_epi8(src3,_mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 3, 6,9, 12, 15)));
    // 阈值判断
    __m128i Result;
    //ch0
    Result = _mm_cmpge_up_epu8(Ch0, ch0_min_sse);
    Result = _mm_and_si128(Result, _mm_cmpge_down_epu8(Ch0, ch0_max_sse));
    //ch1
    Result = _mm_and_si128(Result, _mm_cmpge_up_epu8(Ch1, ch1_min_sse));
    Result = _mm_and_si128(Result, _mm_cmpge_down_epu8(Ch1, ch1_max_sse));
    //ch2
    Result = _mm_and_si128(Result, _mm_cmpge_up_epu8(Ch2, ch2_min_sse));
    Result = _mm_and_si128(Result, _mm_cmpge_down_epu8(Ch2, ch2_max_sse));
	//输出
    _mm_storeu_si128((__m128i *) (dst + 0), Result);
}
```
从最开始的读取数据可以看出,我们每次加载了48个字节的数据,也就是16组像素点,然后又使用`_mm_shuffle_epi8`将依次排列的48个像素点重组在3个`__m128i`变量Ch0,Ch1,Ch2中.
```shell
函数原型:__m128i _mm_shuffle_epi8 (__m128i a, __m128i b)
FOR j := 0 to 15
	i := j*8
	IF b[i+7] == 1
		dst[i+7:i] := 0
	ELSE
		index[3:0] := b[i+3:i]
		dst[i+7:i] := a[index*8+7:index*8]
	FI
ENDFOR
```
其实说白了就是将a中的每一个字节进行重排**(但是需要注意的是,真正其索引作用的是每个字节的低4位,也就是索引值范围为0-15,这也是为什么`__mm_shuffle_epi8`不能配合`__m256i`类型数据使用,因为`__m256i`为32个字节,后面的16个字节无法选中,谨记!!!)**.
比如`__m128i Ch0 = _mm_shuffle_epi8(src1, _mm_setr_epi8(0, 3, 6, 9, 12, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1))`就是将src1的第1个元素放在第一位,第4个元素放在第二位,第7个元素放在第三位,以此类推.组成一个新的变量,-1表示该字节置0,用数组来表示的话,Ch0中的元素为{src1[0],src1[3],src1[6],src1[12],src1[15],0,0,0,0,0,0,0,0,0,0},这里你会发现,我们这样取正好是将src1中像素的第一个通道值取出来了.`Ch0 = _mm_or_si128(Ch0, _mm_shuffle_epi8(src2,_mm_setr_epi8(-1, -1, -1, -1, -1, -1, 2, 5, 8, 11, 14, -1, -1,-1, -1, -1)));`又把src2中的像素的第一通道值取出来,然后与前面求得的在进行与操作,所以现在Ch0中储存的就是src1和src2中的像素的第一个通道值,最后我们在把src3中的也取出来进行与操作,此时Ch0中就包含了本次循环所有处理像素的第一通道的值.
接下来的Ch1,Ch2也是分别储存了第二,三通道的像素值.
再下面跟的是阈值判断部分,这里主要使用了`_mm_cmpge_up_epu8(a,b)`和`_mm_cmpge_down_epu8(a,b)`两个函数,前者是将a中大于等于b的字节置为1,其它置为0,后者是小于等于的置为1,其它置0.这两个函数不在官方库中,官方只提供了有符号的比较函数,所以我们根据网上提供的代码:
```cpp
#define _mm_cmpge_up_epu8(a, b) _mm_cmpeq_epi8(_mm_max_epu8(a, b), a) //大于等于的留下
#define _mm_cmpge_down_epu8(a, b) _mm_cmpeq_epi8(_mm_min_epu8(a, b), a) //小于等于的留下
```
通过讲`_mm_cmpeq_epi8`与`_mm_max_epu8`巧妙地结合起来,实现无符号比较函数.
```cpp
 _mm_storeu_si128((__m128i *) (dst + 0), Result);
```
最后的这句话是将SSE变量中的值载出到外部变量中.

### 收尾
```cpp
//剩余不足一个block的单独处理
for (int j = blockSize * block * channel; j < height * width; ++j, src += channel, dst++) 
{
	uint8_t _ch0 = src[0], _ch1 = src[1], _ch2 = src[2];
	if (_ch0 >= lower[0] && _ch0 <= upper[0] && _ch1 >= lower[1] && _ch1 <= upper[1] && _ch2 >= lower[2] &&_ch2 <= upper[2]) 
	{
		dst[0] = 255;
	} else {
		dst[0] = 0;
	}
}
```
因为数据的总数不一定总是blockSize的整数倍,所以最后再加一个循环将不足一个block的像素单独处理.

### 单通道图像
当输入的图像为单通道时:
```cpp
int blockSize = 32;
int block = height * width  / blockSize;

// 这里需要将阈值载入到256位的变量中
__m256i ch0_min_sse = _mm256_setr_epi8(lower[0],lower[0],lower[0],lower[0],lower[0],lower[0],lower[0],lower[0],lower[0],lower[0],lower[0],lower[0],lower[0],lower[0],lower[0],lower[0],lower[0],lower[0],lower[0],lower[0],lower[0],lower[0],lower[0],lower[0],lower[0],lower[0],lbubishiyongower[0],lower[0],lower[0],lower[0],lower[0],lower[0]);

__m256i ch0_max_sse = _mm256_setr_epi8(upper[0],upper[0],upper[0],upper[0],upper[0],upper[0],upper[0],upper[0],upper[0],upper[0],upper[0],upper[0],upper[0],upper[0],upper[0],upper[0],upper[0],upper[0],upper[0],upper[0],upper[0],upper[0],upper[0],upper[0],upper[0],upper[0],upper[0],upper[0],upper[0],upper[0],upper[0],upper[0]);

for(int i=0; i<block; ++i,src+=blockSize,dst+=blockSize)
{
	// 一次加载32个像素
	__m256i src1 = _mm256_loadu_si256((__m256i *) (src + 0));
	__m256i Result;
	Result = _mm256_cmpge_up_epu8(src1, ch0_min_sse);
	Result = _mm256_and_si256(Result, _mm256_cmpge_down_epu8(src1,ch0_max_sse));
	_mm256_storeu_si256((__m256i *) (dst + 0), Result);
}

for (int j = blockSize * block * channel; j < height * width; ++j, src += channel, dst++) 
{
	uint8_t _ch0 = src[0], _ch1 = src[1], _ch2 = src[2];
	if (_ch0 >= lower[0] && _ch0 <= upper[0] && _ch1 >= lower[1] && _ch1 <= upper[1] && _ch2 >= lower[2] && _ch2 <= upper[2]) 
	{
		dst[0] = 255;dao
	} else {
		dst[0] = 0;
	}
}
```





