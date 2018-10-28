---
layout: post
title: "CMakeLists快速入门"
date: 2018-10-25
categories:
- 程序员的基本素养
tag:
- CMakeLists
- CMake
---
编写CMakeLists文件是一个程序员的基本素养，如何优雅地编写CMakeLists，是快乐编程的第一步。之前一直是用的最简单形式的CMakeLists文件，但随着开发的项目越来越庞大，不禁发觉CMakeLists的学问还是蛮大的。

### 最简单的CMakeLists
本例只包含一个目录，所有的工程文件都在根目录，想当年，我把整个工程的源文件、头文件都放在根目录，就是为了CMakeLists.txt好写一点。
```text
project/
|— CMakeLists.txt
|— main_1.cpp
|— main_2.hpp
```
```cmake
#指定了cmake的最低版本
cmake_minimum_required(VERSION 3.2)
#工程名字，自己起的
project(testProject)
#生成可执行文件main，我们在后面将工程中用到的源文件和头文件列出来
add_executable(main main_1.cpp main_2.hpp)
```
我猜，这可能是最简单的CMakeLists文件了吧。

### 增加工程文件
```bash
project/
|— CMakeLists.txt
|— main.cpp
|— main_1.cpp
...
|— main_100.cpp
```
假如像我之前一样，把工程的所有文件都放在一个目录下，如果还按照上面的写法把所有的源文件都列在add_executable中，不仅效率低，而且很不优雅。
```cmake
#指定了cmake的最低版本
cmake_minimum_required(VERSION 3.2)
#工程名字，自己起的
project(testProject)

#新增加的语句
aux_source_directory(. ALL_SRCS)

#生成可执行文件main，我们在后面将工程中用到的源文件和头文件列出来
add_executable(main ${ALL_SRCS})
```
只需稍微修改一下，即可优雅地完成我们的要求，那就是`aux_source_directory`，它的作用的搜索指定目录`dir`下的所有**源文件**（注意是源文件），将文件名当作列表保存在`variable`中，用法为`aux_source_directory(< dir > < variable >)`。上述例子中是将101个源文件储存在`ALL_SRCS`变量中，所以`add_executable`中只需引用变量`ALL_SRCS`即可（这里写`${ALL_SRCS}`等同于手动列出那101个源文件）。

### 调整工程目录

但是稍微有开发经验的程序员都不会像我们那样组织工程目录，我们会按照函数的功能对整个工程进行划分，以及头文件和源文件分离等等易于组织的结构。

```bash
project/
| — CMakeLists.txt
| — main.cpp
| — function/
    | — func1.cpp
    | — func1.hpp
```
这里我们将功能函数单独放在`function`目录下，使整个工程变得有条理一点，此时你的`CMakeLists.txt`可以这样写，仿照上面的例子。

```cmake
#指定了cmake的最低版本
cmake_minimum_required(VERSION 3.2)
#工程名字，自己起的
project(testProject)
#生成可执行文件main，我们在后面将工程中用到的源文件和头文件列出来
add_executable(main main.cpp function/func1.cpp function/func1.hpp)
```
这样完全OK的，但是不够优雅，你可以这样来写

```cmake
#指定了cmake的最低版本
cmake_minimum_required(VERSION 3.2)
#工程名字，自己起的
project(testProject)

#新增语句1
add_subdirectory(function)
#新增语句2
include_directories(function)

#生成可执行文件main，我们在后面将工程中用到的源文件和头文件列出来
add_executable(main main.cpp)
#新添加语句3
target_link_libraries(main funcLib)
```
然后我们再在`function`目录下新建一个`CMakeLists.txt`文件

```cmake
#将该目录下的所有的源文件名字保存在DIR_FUNC_SRCS中
aux_source_directory(. DIR_FUNC_SRCS)
#生成链接库，名字为funcLib 
add_library(funcLib ${DIR_FUNC_SRCS})
```

这里采用的思路是将`function`目录下的源文件作为链接库供工程使用，在根目录下的`CMakeLists.txt`中：
- 新增语句1是告诉编译器工程还包含一个子目录`function`
- 新增语句2添加了头文件目录，因为在`function`目录下含有工程使用的头文件，也就是说你在程序里`#include`本地头文件时，编译器搜索的目录，更直观一点说，没有这句，你在main里面应该这样写`#include "function/func1.hpp"`，如果你写了这句，则可以这样写`#include "func1.hpp"`，也就是编译器知道去哪里去找这个头文件。
- 新增语句3是将生成的链接库`funcLib`链接到可执行文件上。

