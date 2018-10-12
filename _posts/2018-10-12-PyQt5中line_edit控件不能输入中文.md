---
layout: post
title: "PyQt5中line_edit控件不能输入中文"
date: 2018-10-12
categories:
- PyQt5
tag:
- PyQt5
- line_edit
---
在用PyQt5写界面的时候，测试的时候发现选中line\_edit时，输入法不能切换到中文，即不能在line\_edit中输入中文，之前在windows平台上开发的时候并没有出现此问题，所以怀疑是qt对ubuntu下的输入法不支持。

### 解决方法

- **原因：** PyQt5的包缺少一个动态链接库
- **找到链接库：** `/usr/lib/x86_64-linux-gnu/qt5/plugins/platforminputcontexts/libfcitxplatforminputcontextplugin.so`
- **将上述链接库复制到PyQt5包所在目录：** 一般包的位置在`/usr/local/lib/python3.5/dist-packages`下，但我的不是在这里，而是在home目录下的一个叫做`.local`的隐藏文件夹里，而且我后面安装的包都在这里。通过`pip show package-name`命令可以查看包的详细安装路径。则你需要将链接库`libfcitxplatforminputcontextplugin.so`复制到`your_PyQt5_path/Qt/plugins/platforminputcontexts/`目录下即可
- **重新运行软件即可在line_edit中输入中文**