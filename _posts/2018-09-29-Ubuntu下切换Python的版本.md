---
layout: post
title: "Ubuntu下切换Python的版本"
date: 2018-09-29
categories:
- Ubuntu
tag:
- Ubuntu
- Python
---

这篇文章也是一直躺在我的有道云笔记里面，最近又有人问我这个问题，就索性写在这里吧，但是随着官方对
python2版本不再维护，大家还是主要使用python3，不能把这个坑越挖越大，现在注意一点，利人利己。

### Python2和Python3共存情况下的默认版本切换

#### 第1种方法

- 切换默认版本为Python2

```bash
sudo update-alternatives --install /usr/bin/python python /usr/bin/python2 100

sudo update-alternatives --config python
```

- 切换默认版本为Python3

```bash
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3 150

sudo update-alternatives --config python
```
#### 第二种方法

- 切换默认版本为Python2

```bash
#查看python版本
python --version
sudo rm /usr/local/bin/python
#将python2.x换成你的版本
sudo ln -s /usr/bin/python2.x /usr/local/bin/python
```

- 切换默认版本为Python3

```bash
#查看python版本
python3 --version
sudo rm /usr/local/bin/python
#将python3.x换成你的版本
sudo ln -s /usr/bin/python3.x /usr/local/bin/python
```

#### 验证

最后在命令行输入`python`回车后会进入python编程环境，即可查看当前系统默认的Python版本。