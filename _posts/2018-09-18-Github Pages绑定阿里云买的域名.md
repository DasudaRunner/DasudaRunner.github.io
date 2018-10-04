---
layout: post
title: "Github Pages绑定阿里云买的域名"
date: 2018-09-18
categories:
- 博客建设
tag:
- 博客建设
- 阿里云
---

使用Github Pages搭建完博客之后，总觉得用原始的域名太长了，又听说在阿里云注册新域名也挺便宜的，
索性就买了一个域名，三年才五十多块，也不贵，本文记录了将新买的域名解析到Github Pages所在的IP。

### 1、购买域名

我是从阿里云购买的域名dasuda.top（其实haibo.lnk也挺便宜的），买了三年，花费55元，想着刚好硕士三年，等毕业后找到工作了，是不续费还是续费，到时候看这个域名的价值吧。

<img src="/assets/images/posts/GithubPages/1.jpg"/>

### **2、查看GithubPages的IP**
&emsp;&emsp;有趣的是在这，按照网上99%的教程都是说在域名解析里面新建两条记录，记录类型为A，记录值为ping username.github.io查到的IP，但是当你连续ping的时候会发现，IP是不固定的，最后在[Setting up an apex domain](https://help.github.com/articles/setting-up-an-apex-domain/)知道，GithubPages服务器现在的IP为4个：
```python
185.199.108.153
185.199.109.153
185.199.110.153
185.199.111.153
```
所以以前的ping方法不适用了，至少没有直接查看[Setting up an apex domain](https://help.github.com/articles/setting-up-an-apex-domain/)网页更方便、更准确了，那么域名解析里面的条目就需要添加8个。另外记录类型为CHAME，记录值为username.github.io，经测试不成功。

<img src="/images/posts/GithubPages/2.jpg"/>
<br>
#### **3、网站根目录CHAME文件设置**

&emsp;&emsp;在网站根目录的CHAME文件（没有则新建一个）里面添加dasuda.top，整个文件就只需写进购买的域名即可。
