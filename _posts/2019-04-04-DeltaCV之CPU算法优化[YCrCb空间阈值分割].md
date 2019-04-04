---
layout: post
title: "DeltaCV之CPU算法优化[YCrCb空间阈值分割]"
date: 2019-04-02
categories:
- DeltaCV
tag:
- DeltaCV
- SSE
- SIMD
excerpt: 这篇文章我们先来认识一下DeltaCV中CPU上做的相关优化工作，使用SSE指令集优化RGB转YCrCb颜色空间过程,并进行阈值分割。

---
* 目录
{:toc}
这篇文章我们先来认识一下DeltaCV中CPU上做的相关优化工作，使用SSE指令集优化RGB转YCrCb颜色空间过程,并进行阈值分割。

## OpenCV中的函数

在OpenCV中完成RGB图像在YCrCb空间的阈值分割,需要用到两个函数,cv::cvtColor(颜色空间转换)及cv::inRange(阈值处理),.

### 函数原型
```cpp

void cvtColor( InputArray src, OutputArray dst, int code, int dstCn = 0 );

void inRange(InputArray src, InputArray lowerb,InputArray upperb, OutputArray dst);
```

## DeltaCV中的优化

### 函数原型
```cpp
    /**  @brief: more details - >　
     *
     * src: input(BGR image)
     * dst: output
     * width:
     * height:
     * lower: lower boundary
     * upper: upper boundary
     */
    bool ycrcbWithSeg(unsigned char *src, unsigned char *dst,const int width,const int height,scalar lower,scalar upper);
```

### 加载变量
```cpp
assert(lower.channels()==3 && upper.channels()==3);
int blockSize = 48; //16*3
int block = (height * width * 3) / blockSize;

// 加载阈值
__m128i ch0_min_sse = _mm_setr_epi8(lower[0],lower[0],lower[0],lower[0],lower[0],lower[0],lower[0],lower[0],lower[0],lower[0],lower[0],lower[0],lower[0],lower[0],lower[0],lower[0]);
__m128i ch1_min_sse = _mm_setr_epi8(lower[1],lower[1],lower[1],lower[1],lower[1],lower[1],lower[1],lower[1],lower[1],lower[1],lower[1],lower[1],lower[1],lower[1],lower[1],lower[1]);
__m128i ch2_min_sse = _mm_setr_epi8(lower[2],lower[2],lower[2],lower[2],lower[2],lower[2],lower[2],lower[2],lower[2],lower[2],lower[2],lower[2],lower[2],lower[2],lower[2],lower[2]);

__m128i ch0_max_sse = _mm_setr_epi8(upper[0],upper[0],upper[0],upper[0],upper[0],upper[0],upper[0],upper[0],upper[0],upper[0],upper[0],upper[0],upper[0],upper[0],upper[0],upper[0]);
__m128i ch1_max_sse = _mm_setr_epi8(upper[1],upper[1],upper[1],upper[1],upper[1],upper[1],upper[1],upper[1],upper[1],upper[1],upper[1],upper[1],upper[1],upper[1],upper[1],upper[1]);
__m128i ch2_max_sse = _mm_setr_epi8(upper[2],upper[2],upper[2],upper[2],upper[2],upper[2],upper[2],upper[2],upper[2],upper[2],upper[2],upper[2],upper[2],upper[2],upper[2],upper[2]);

const int Shift = 15;
const int HalfV = 1<<(Shift-1);

//这里不能用网上给的转换公式，因为bgr转ycrcb有很多版本，我是直接看opencv源码算出的系数,
// static const float coeffs_crb[] = { R2YF, G2YF, B2YF, YCRF, YCBF }; in /opencv-4.0.1/modules/imgproc/src/color_yuv.cpp
//为什么要和opencv一样的系数?从结果来看，这套系数得到的cr cb通道对比度更大，也就是说红色和蓝色与其他背景区分度更高.
const int Y_B_WT = 0.114f * (1 << Shift), Y_G_WT = 0.587f * (1 << Shift), Y_R_WT = 0.299f * (1 << Shift), Y_C_WT = 1;
const int Cr_B_WT = 0.499f * (1 << Shift), Cr_G_WT = -0.419f * (1 << Shift), Cr_R_WT = -0.081f * (1 << Shift), Cr_C_WT = 257;
const int Cb_B_WT = 0.395f * (1 << Shift), Cb_G_WT = -0.331f * (1 << Shift), Cb_R_WT = -0.064f * (1 << Shift), Cb_C_WT = 257;

// load 上面的权值
__m128i Weight_YBG = _mm_setr_epi16(Y_B_WT, Y_G_WT, Y_B_WT, Y_G_WT, Y_B_WT, Y_G_WT, Y_B_WT, Y_G_WT);
__m128i Weight_YRC = _mm_setr_epi16(Y_R_WT, Y_C_WT, Y_R_WT, Y_C_WT, Y_R_WT, Y_C_WT, Y_R_WT, Y_C_WT);
__m128i Weight_UBG = _mm_setr_epi16(Cr_B_WT, Cr_G_WT, Cr_B_WT, Cr_G_WT, Cr_B_WT, Cr_G_WT, Cr_B_WT, Cr_G_WT);
__m128i Weight_URC = _mm_setr_epi16(Cr_R_WT, Cr_C_WT, Cr_R_WT, Cr_C_WT, Cr_R_WT, Cr_C_WT, Cr_R_WT, Cr_C_WT);
__m128i Weight_VBG = _mm_setr_epi16(Cb_B_WT, Cb_G_WT, Cb_B_WT, Cb_G_WT, Cb_B_WT, Cb_G_WT, Cb_B_WT, Cb_G_WT);
__m128i Weight_VRC = _mm_setr_epi16(Cb_R_WT, Cb_C_WT, Cb_R_WT, Cb_C_WT, Cb_R_WT, Cb_C_WT, Cb_R_WT, Cb_C_WT);
__m128i Half = _mm_setr_epi16(0, HalfV, 0, HalfV, 0, HalfV, 0, HalfV);
```
- 刚开始我们进行了阈值的通道检查(因为这个函数输入的数据必须是RGB图像).
- 紧接着,再将高低两个阈值加载进SSE变量中.
- 下面是重点,RGB转ycrcb的公式里面是每个系数都是0-1之间的浮点型,而我们前面取到的像素是以8位无符号整形保存在SSE变量中,所以我们需要将系数左移15位,既保证了精度,又满足了计算要求(没有超过16位,正好两个8位).

### 计算主体

```cpp
for (int i = 0; i < block; ++i, src += blockSize, dst += 16)
{
	__m128i src1, src2, src3;

    src1 = _mm_loadu_si128((__m128i *) (src + 0)); //一次性读取个字节
    src2 = _mm_loadu_si128((__m128i *) (src + 16));
    src3 = _mm_loadu_si128((__m128i *) (src + 32));

// 在_mm_shuffle_epi8中构造出16位的数据
	__m128i BGLL = _mm_shuffle_epi8(src1,_mm_setr_epi8(0,-1,1,-1,3,-1,4,-1,6,-1,7,-1,9,-1,10,-1));
	__m128i BGLH = _mm_shuffle_epi8(src1,_mm_setr_epi8(12,-1,13,-1,15,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1));
	BGLH = _mm_or_si128(BGLH,_mm_shuffle_epi8(src2,_mm_setr_epi8(-1,-1,-1,-1,-1,-1,0,-1,2,-1,3,-1,5,-1,6,-1)));

	__m128i BGHL = _mm_shuffle_epi8(src2, _mm_setr_epi8(8, -1, 9, -1, 11, -1, 12, -1, 14, -1, 15, -1, -1, -1, -1, -1));
	BGHL = _mm_or_si128(BGHL,_mm_shuffle_epi8(src3,_mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, 2, -1)));
	__m128i BGHH = _mm_shuffle_epi8(src3, _mm_setr_epi8(4, -1, 5, -1, 7, -1, 8, -1, 10, -1, 11, -1, 13, -1, 14, -1));

	__m128i RCLL = _mm_shuffle_epi8(src1,_mm_setr_epi8(2,-1,-1,-1,5,-1,-1,-1,8,-1,-1,-1,11,-1,-1,-1));
	RCLL = _mm_or_si128(RCLL,Half);
	__m128i RCLH = _mm_shuffle_epi8(src1,_mm_setr_epi8(14,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1));
	RCLH = _mm_or_si128(_mm_or_si128(RCLH,_mm_shuffle_epi8(src2,_mm_setr_epi8(-1,-1,-1,-1,1,-1,-1,-1,4,-1,-1,-1,7,-1,-1,-1))),Half);

	__m128i RCHL = _mm_shuffle_epi8(src2,_mm_setr_epi8(10,-1,-1,-1,13,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1));
	RCHL = _mm_or_si128(_mm_or_si128(RCHL,_mm_shuffle_epi8(src3,_mm_setr_epi8(-1,-1,-1,-1,-1,-1,-1,-1,0,-1,-1,-1,3,-1,-1,-1))),Half);
	__m128i RCHH = _mm_shuffle_epi8(src3,_mm_setr_epi8(6,-1,-1,-1,9,-1,-1,-1,12,-1,-1,-1,15,-1,-1,-1));
	RCHH = _mm_or_si128(RCHH,Half);

// 下面是进行计算
	__m128i Result;

	__m128i Y_LL = _mm_srai_epi32(_mm_add_epi32(_mm_madd_epi16(BGLL, Weight_YBG), _mm_madd_epi16(RCLL, Weight_YRC)), Shift);
	__m128i Y_LH = _mm_srai_epi32(_mm_add_epi32(_mm_madd_epi16(BGLH, Weight_YBG), _mm_madd_epi16(RCLH, Weight_YRC)), Shift);
	__m128i Y_HL = _mm_srai_epi32(_mm_add_epi32(_mm_madd_epi16(BGHL, Weight_YBG), _mm_madd_epi16(RCHL, Weight_YRC)), Shift);
	__m128i Y_HH = _mm_srai_epi32(_mm_add_epi32(_mm_madd_epi16(BGHH, Weight_YBG), _mm_madd_epi16(RCHH, Weight_YRC)), Shift);

	__m128i Ch0 = _mm_packus_epi16(_mm_packus_epi32(Y_LL, Y_LH), _mm_packus_epi32(Y_HL, Y_HH));
	Result = _mm_cmpge_up_epu8(Ch0, ch0_min_sse);
	Result = _mm_and_si128(Result, _mm_cmpge_down_epu8(Ch0, ch0_max_sse));

	__m128i U_LL = _mm_srai_epi32(_mm_add_epi32(_mm_madd_epi16(BGLL, Weight_UBG), _mm_madd_epi16(RCLL, Weight_URC)), Shift);
	__m128i U_LH = _mm_srai_epi32(_mm_add_epi32(_mm_madd_epi16(BGLH, Weight_UBG), _mm_madd_epi16(RCLH, Weight_URC)), Shift);
	__m128i U_HL = _mm_srai_epi32(_mm_add_epi32(_mm_madd_epi16(BGHL, Weight_UBG), _mm_madd_epi16(RCHL, Weight_URC)), Shift);
	__m128i U_HH = _mm_srai_epi32(_mm_add_epi32(_mm_madd_epi16(BGHH, Weight_UBG), _mm_madd_epi16(RCHH, Weight_URC)), Shift);

	__m128i Ch1 = _mm_packus_epi16(_mm_packus_epi32(U_LL, U_LH), _mm_packus_epi32(U_HL, U_HH));
	Result = _mm_and_si128(Result, _mm_cmpge_up_epu8(Ch1, ch1_min_sse));
	Result = _mm_and_si128(Result, _mm_cmpge_down_epu8(Ch1, ch1_max_sse));

	__m128i V_LL = _mm_srai_epi32(_mm_add_epi32(_mm_madd_epi16(BGLL, Weight_VBG), _mm_madd_epi16(RCLL, Weight_VRC)), Shift);
	__m128i V_LH = _mm_srai_epi32(_mm_add_epi32(_mm_madd_epi16(BGLH, Weight_VBG), _mm_madd_epi16(RCLH, Weight_VRC)), Shift);
	__m128i V_HL = _mm_srai_epi32(_mm_add_epi32(_mm_madd_epi16(BGHL, Weight_VBG), _mm_madd_epi16(RCHL, Weight_VRC)), Shift);
	__m128i V_HH = _mm_srai_epi32(_mm_add_epi32(_mm_madd_epi16(BGHH, Weight_VBG), _mm_madd_epi16(RCHH, Weight_VRC)), Shift);

	__m128i Ch2 = _mm_packus_epi16(_mm_packus_epi32(V_LL, V_LH), _mm_packus_epi32(V_HL, V_HH));
	Result = _mm_and_si128(Result, _mm_cmpge_up_epu8(Ch2, ch2_min_sse));
	Result = _mm_and_si128(Result, _mm_cmpge_down_epu8(Ch2, ch2_max_sse));

	_mm_storeu_si128((__m128i *)dst, Result);
}

//剩余不足一个block的单独处理
for (int j = blockSize * block; j < height * width; ++j, src += 3, dst++) {
	uint8_t _ch0 = src[0], _ch1 = src[1], _ch2 = src[2];

	uint8_t YUV_Y = (Y_B_WT * _ch0 + Y_G_WT * _ch1 + Y_R_WT * _ch2 + Y_C_WT * HalfV) >> Shift;
	uint8_t YUV_U = (Cr_B_WT * _ch0 + Cr_G_WT * _ch1 + Cr_R_WT * _ch2 + Cr_C_WT * HalfV) >> Shift;
	uint8_t YUV_V = (Cb_B_WT * _ch0 + Cb_G_WT * _ch1 + Cb_R_WT * _ch2 + Cb_C_WT * HalfV) >> Shift;
	if ((YUV_Y >= lower[0] && YUV_Y <= upper[0] && YUV_U >= lower[1] && YUV_U <= upper[1] && YUV_V >= lower[2] &&
YUV_V <= upper[2])) {
	dst[0] = 255;
	} else {
		dst[0] = 0;
	}
}
```

- 这里我们同样先从输入图像中加载待处理的数据.
- 接着我们将B G R B G R B G R这样的数据拆分成B 0 G 0 B 0 G 0 和R 0 half 0 R 0 half 0这样的形式,为了后面使用`_mm_madd_epi16`将像素与各自的权重相乘再相加.
- Y_C_WT, Cr_C_WT, Cb_C_WT这三个变量是干什么? 
我们来看一下rgb转ycrcb的公式:[参考Imageshop])(https://www.cnblogs.com/Imageshop/p/8405517.html)
```cpp
Y = (Y_B_WT * Blue + Y_G_WT * Green + Y_R_WT * Red + HalfV) >> Shift;
Cr = ((Cr_B_WT * Blue + Cr_G_WT * Green + Cr_R_WT * Red + HalfV) >> Shift) + 128;
Cb = ((Cb_B_WT * Blue + Cb_G_WT * Green + Cb_R_WT * Red + HalfV) >> Shift) + 128;
```
经过变形成为:
```cpp
Y = (Y_B_WT * Blue + Y_G_WT * Green + Y_R_WT * Red + 1 * HalfV) >> Shift;
Cr = (U_B_WT * Blue + U_G_WT * Green + U_R_WT * Red + 257 * HalfV) >> Shift;
Cb = (V_B_WT * Blue + V_G_WT * Green + V_R_WT * Red + 257 * HalfV) >> Shift;
这就是Y_C_WT=1,Cr_C_WT=257,Cb_C_WT=257的由来
```
- 虽然上面代码量看着较多,但是都是基本运算,而且比OpenCV的函数快了7-8倍.

### 性能对比
处理1000次取平均值

Image Size: 1024 x 1280（H x W）
| Function | OpenCV/ms | DeltaCV/ms | Speed-up |
|:-:|:-:|:-:|:-:|
|ycrcbWithSeg|6.68 - 6.75 |0.88 - 0.90|7.4 - 7.6|






