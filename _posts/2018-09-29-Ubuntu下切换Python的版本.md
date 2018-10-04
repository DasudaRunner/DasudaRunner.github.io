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

<br>
- 本文为Haibo原创文章，转载请注明：[Haibo的主页](https://dasuda.top)

- 如果对本站的文章有疑问或者有合作需求的，可以联系QQ: 827091497，或者发送邮件到：[haibo.david@qq.com](mailto:haibo.david@qq.com) 。

### Python2和Python3共存情况下的默认版本切换

#### **第1种方法**

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

#### **第二种方法**

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

#### **验证**

最后在命令行输入`python`回车后会进入python编程环境，即可查看当前系统默认的Python版本。