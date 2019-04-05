---
layout: post
title: "Principles of training multi-layer neural network using backpropagation[翻译]"
date: 2019-04-05
categories:
- 翻译
tag:
- 翻译
- BP
- DL
excerpt: 这篇文章是介绍神经网络中的训练方法-BP算法,算是一篇比较经典的文章,最近又自己用SIMD指令集重写一个小型的神经网络,借此机会再好好巩固一下BP算法.

---
* 目录
{:toc}
这篇文章是介绍神经网络中的训练方法-BP算法,算是一篇比较经典的文章,最近又自己用SIMD指令集重写一个小型的神经网络,借此机会再好好巩固一下BP算法.

[原文链接](http://galaxy.agh.edu.pl/~vlsi/AI/backp_t_en/backprop.html)

## 使用反向传播算法训练多层神经网络的原理
本项目使用BP算法来介绍多层神经网络的训练过程,下图展示了一个包含三层神经元的网络,具有两个输入,一个输出.

<img src="/assets/images/posts/BP/img01.gif" div align=center/>

每个神经元有两个操作组成,一个是将系数(包括权重和偏置)和输入信号进行叠加,第二个是引入非线性,我们一般称之为神经元激活函数.其中$e$就是叠加信号,$$

<img src="/assets/images/posts/BP/img02.gif"/>













