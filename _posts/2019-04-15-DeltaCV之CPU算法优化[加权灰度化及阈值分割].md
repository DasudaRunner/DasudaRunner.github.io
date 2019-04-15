---
layout: post
title: "DeltaCV之CPU算法优化[加权灰度化及阈值分割]"
date: 2019-04-15
categories:
- DeltaCV
tag:
- DeltaCV
- SSE
- SIMD
excerpt: 这篇文章我们先来认识一下DeltaCV中CPU上做的相关优化工作，使用SSE及AVX指令集优化加权灰度化过程,并进行阈值分割。

---
* 目录
{:toc}
这篇文章我们先来认识一下DeltaCV中CPU上做的相关优化工作，使用SSE及AVX指令集优化加权灰度化过程,并进行阈值分割。

## OpenCV中的过程
```cpp
    cv::Mat s_colors[3],weighted_gray,opencvOUT;
    cv::split(img,s_colors);
    weighted_gray = s_colors[0]*0.9+s_colors[1]*0.05+s_colors[2]*0.05;
    cv::threshold(weighted_gray,opencvOUT,125,255,cv::THRESH_BINARY);
```
可以看到，上述过程涉及三个步骤：分离通道，加权计算，阈值分割。

可以优化的地方还是两点：
- 并行优化
- 一次遍历

## DeltaCV中的优化

### 函数原型
```cpp
    /*
     * src: input(BGR image)
     * dst: output
     * width:
     * height:
     * lower: lower boundary: 1 channels
     * upper: upper boundary: 1 channels
     */
    void weightedGrayWithSeg(unsigned char *src, unsigned char *dst,const int width,const int height,scalar_d weights,
                             scalar lower,scalar upper);
```
### 加载变量
```cpp
        // 加载阈值
__m256i ch0_min_sse = _mm256_setr_epi8(lower[0],lower[0],lower[0],lower[0],lower[0],lower[0],lower[0],lower[0],lower[0],lower[0],lower[0],lower[0],lower[0],lower[0],lower[0],lower[0],lower[0],lower[0],lower[0],lower[0],lower[0],lower[0],lower[0],lower[0],lower[0],lower[0],lower[0],lower[0],lower[0],lower[0],lower[0],lower[0]);

__m256i ch0_max_sse = _mm256_setr_epi8(upper[0],upper[0],upper[0],upper[0],upper[0],upper[0],upper[0],upper[0],upper[0],upper[0],upper[0],upper[0],upper[0],upper[0],upper[0],upper[0],upper[0],upper[0],upper[0],upper[0],upper[0],upper[0],upper[0],upper[0],upper[0],upper[0],upper[0],upper[0],upper[0],upper[0],upper[0],upper[0]);

const int Shift = 15;

const int W_B = weights[0] * (1 << Shift), W_G = weights[1] * (1 << Shift), W_R = weights[2] * (1 << Shift);

__m256i SSE_WBG = _mm256_setr_epi16(W_B, W_G, W_B, W_G, W_B, W_G, W_B, W_G, W_B, W_G, W_B, W_G, W_B, W_G, W_B, W_G);

__m256i SSE_WRC = _mm256_setr_epi16(W_R, 0, W_R, 0, W_R, 0, W_R, 0,W_R, 0, W_R, 0, W_R, 0, W_R, 0);
```
主要是将两个阈值及加权系数加载进变量中，注意这里是使用的`__m256i`类型，因为我们后面进行的计算和判断都将在AVX指令集上进行。

### 循环主体
```cpp
__m128i src1, src2, src3,src4,src5,src6;

src1 = _mm_loadu_si128((__m128i *) (src + 0));
src2 = _mm_loadu_si128((__m128i *) (src + 16));
src3 = _mm_loadu_si128((__m128i *) (src + 32));
src4 = _mm_loadu_si128((__m128i *) (src + 48));
src5 = _mm_loadu_si128((__m128i *) (src + 64));
src6 = _mm_loadu_si128((__m128i *) (src + 80));

__m128i BG0_00 = _mm_shuffle_epi8(src1,_mm_setr_epi8(0,-1,1,-1,3,-1,4,-1,6,-1,7,-1,9,-1,10,-1));
 __m128i BG0_01 = _mm_shuffle_epi8(src1,_mm_setr_epi8(12,-1,13,-1,15,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1));
BG0_01 = _mm_or_si128(BG0_01,_mm_shuffle_epi8(src2,_mm_setr_epi8(-1,-1,-1,-1,-1,-1,0,-1,2,-1,3,-1,5,-1,6,-1)));

__m128i BG0_10 = _mm_shuffle_epi8(src2, _mm_setr_epi8(8, -1, 9, -1, 11, -1, 12, -1, 14, -1, 15, -1, -1, -1, -1, -1));
BG0_10 = _mm_or_si128(BG0_10,_mm_shuffle_epi8(src3,_mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, 2, -1)));
__m128i BG0_11 = _mm_shuffle_epi8(src3, _mm_setr_epi8(4, -1, 5, -1, 7, -1, 8, -1, 10, -1, 11, -1, 13, -1, 14, -1));

__m128i BG1_00 = _mm_shuffle_epi8(src4,_mm_setr_epi8(0,-1,1,-1,3,-1,4,-1,6,-1,7,-1,9,-1,10,-1));
__m128i BG1_01 = _mm_shuffle_epi8(src4,_mm_setr_epi8(12,-1,13,-1,15,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1));
BG1_01 = _mm_or_si128(BG1_01,_mm_shuffle_epi8(src5,_mm_setr_epi8(-1,-1,-1,-1,-1,-1,0,-1,2,-1,3,-1,5,-1,6,-1)));

__m128i BG1_10 = _mm_shuffle_epi8(src5, _mm_setr_epi8(8, -1, 9, -1, 11, -1, 12, -1, 14, -1, 15, -1, -1, -1, -1, -1));
BG1_10 = _mm_or_si128(BG1_10,_mm_shuffle_epi8(src6,_mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, 2, -1)));
__m128i BG1_11 = _mm_shuffle_epi8(src6, _mm_setr_epi8(4, -1, 5, -1, 7, -1, 8, -1, 10, -1, 11, -1, 13, -1, 14, -1));

__m256i BG0 = _mm256_combine_si128(BG0_00,BG1_00);
__m256i BG1 = _mm256_combine_si128(BG0_01,BG1_01);
__m256i BG2 = _mm256_combine_si128(BG0_10,BG1_10);
__m256i BG3 = _mm256_combine_si128(BG0_11,BG1_11);

__m128i RC0_00 = _mm_shuffle_epi8(src1,_mm_setr_epi8(2,-1,-1,-1,5,-1,-1,-1,8,-1,-1,-1,11,-1,-1,-1));
__m128i RC0_01 = _mm_shuffle_epi8(src1,_mm_setr_epi8(14,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1));
RC0_01 = _mm_or_si128(RC0_01,_mm_shuffle_epi8(src2,_mm_setr_epi8(-1,-1,-1,-1,1,-1,-1,-1,4,-1,-1,-1,7,-1,-1,-1)));

__m128i RC0_10 = _mm_shuffle_epi8(src2,_mm_setr_epi8(10,-1,-1,-1,13,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1));
RC0_10 = _mm_or_si128(RC0_10,_mm_shuffle_epi8(src3,_mm_setr_epi8(-1,-1,-1,-1,-1,-1,-1,-1,0,-1,-1,-1,3,-1,-1,-1)));
__m128i RC0_11 = _mm_shuffle_epi8(src3,_mm_setr_epi8(6,-1,-1,-1,9,-1,-1,-1,12,-1,-1,-1,15,-1,-1,-1));

__m128i RC1_00 = _mm_shuffle_epi8(src4,_mm_setr_epi8(2,-1,-1,-1,5,-1,-1,-1,8,-1,-1,-1,11,-1,-1,-1));

__m128i RC1_01 = _mm_shuffle_epi8(src4,_mm_setr_epi8(14,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1));
RC1_01 = _mm_or_si128(RC1_01,_mm_shuffle_epi8(src5,_mm_setr_epi8(-1,-1,-1,-1,1,-1,-1,-1,4,-1,-1,-1,7,-1,-1,-1)));

__m128i RC1_10 = _mm_shuffle_epi8(src5,_mm_setr_epi8(10,-1,-1,-1,13,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1));
RC1_10 = _mm_or_si128(RC1_10,_mm_shuffle_epi8(src6,_mm_setr_epi8(-1,-1,-1,-1,-1,-1,-1,-1,0,-1,-1,-1,3,-1,-1,-1)));
__m128i RC1_11 = _mm_shuffle_epi8(src6,_mm_setr_epi8(6,-1,-1,-1,9,-1,-1,-1,12,-1,-1,-1,15,-1,-1,-1));

__m256i RC0 = _mm256_combine_si128(RC0_00,RC1_00);
__m256i RC1 = _mm256_combine_si128(RC0_01,RC1_01);
__m256i RC2 = _mm256_combine_si128(RC0_10,RC1_10);
__m256i RC3 = _mm256_combine_si128(RC0_11,RC1_11);
```
上述代码主要是加载图像中的像素，一次加载了96个（与之前的48不同），然后先使用SSE指令构造出BG(B 0 G 0 B 0 G 0...)和RC(R 0 0 0 R 0 0 0...)组合，然后再将它们拼接为`__m256i`，为什么不直接在AVX上进行呢？因为AVX中的shuffle指令在这里用起来不太合适，当然也能构造出想要的数据，但是会多出很多额外的运算，大家可以试一试，如果更搞笑的方法，请在github的issues提出。

### 加权计算及阈值分割

```cpp
__m256i Result;

__m256i gray0 = _mm256_srai_epi32(_mm256_add_epi32(_mm256_madd_epi16(BG0, SSE_WBG), _mm256_madd_epi16(RC0, SSE_WRC)), Shift);
__m256i gray1 = _mm256_srai_epi32(_mm256_add_epi32(_mm256_madd_epi16(BG1, SSE_WBG), _mm256_madd_epi16(RC1, SSE_WRC)), Shift);
__m256i gray2 = _mm256_srai_epi32(_mm256_add_epi32(_mm256_madd_epi16(BG2, SSE_WBG), _mm256_madd_epi16(RC2, SSE_WRC)), Shift);
__m256i gray3 = _mm256_srai_epi32(_mm256_add_epi32(_mm256_madd_epi16(BG3, SSE_WBG), _mm256_madd_epi16(RC3, SSE_WRC)), Shift);

__m256i packed_gray = _mm256_packus_epi16(_mm256_packus_epi32(gray0, gray1), _mm256_packus_epi32(gray2, gray3));
Result = _mm256_cmpge_up_epu8(packed_gray, ch0_min_sse);
Result = _mm256_and_si256(Result, _mm256_cmpge_down_epu8(packed_gray, ch0_max_sse));

_mm256_storeu_si256((__m256i *)dst, Result);
```
首先将4组数据进行加权运算得到4个`__m256i`变量，这个就是灰度空间的像素值，然后最后再接上阈值判断，处于范围中的置255，其它置0.

### 收尾
```cpp
for (int j = blockSize * block; j < height * width; ++j, src += 3, dst++) {
    uint8_t _ch0 = src[0], _ch1 = src[1], _ch2 = src[2];
    uint8_t gray_ = (W_B * _ch0 + W_G * _ch1 + W_R * _ch2) >> Shift;
    if (gray_ >= lower[0] && gray_ <= upper[0]) 
    {
        dst[0] = 255;
    } else {
        dst[0] = 0;
    }
}
```
老规矩，不足一个block的单独处理。

### 性能对比
处理1000次取平均值

Image Size: 1024 x 1280（H x W）

opencv函数耗时: 1.56 - 1.69 ms

DeltaCV函数耗时: 0.39 - 0.46 ms

加速比: 3.39 - 4.33