# 构建 TensorFlow Lite Python Wheel 软件包

本页介绍如何为 x86_64 和各种 ARM 设备构建 TensorFlow Lite `tflite_runtime` Python 库。

以下说明已在 Ubuntu 16.04.3 64 位 PC (AMD64)、macOS Catalina (x86_64) 和 TensorFlow devel Docker 镜像 [tensorflow/tensorflow:devel](https://hub.docker.com/r/tensorflow/tensorflow/tags/) 上进行测试。

**注**：此功能从 2.4 版本开始提供。

#### Prerequisites

您需要安装 CMake 和 TensorFlow 源代码的副本。有关详细信息，请参阅[使用 CMake 构建 TensorFlow Lite](https://www.tensorflow.org/lite/guide/build_cmake) 页面。

要为您的工作站构建 pip 包，可以运行以下命令。

```sh
PYTHON=python3 tensorflow/lite/tools/pip_package/build_pip_package_with_cmake.sh native
```

**注**：如果您有多个可用的 Python 解释器，请使用 `PYTHON` 变量。（目前支持 Python 3.7 或更高版本）

## ARM 交叉编译

对于 ARM 交叉编译，建议使用 Docker，因为它可以更容易地设置跨构建环境。此外，您还需要 `target` 选项来确定目标架构。

在 Makefile `tensorflow/lite/tools/pip_package/Makefile` 中有一个辅助工具可用于使用预定义的 Docker 容器调用构建命令。您可以在 Docker 主机上运行如下所示的构建命令。

```sh
make -C tensorflow/lite/tools/pip_package docker-build \
  TENSORFLOW_TARGET=<target> PYTHON_VERSION=<python3 version>
```

**注**：支持 Python 3.7 或更高版本。

### 可用目标名称

`tensorflow/lite/tools/pip_package/build_pip_package_with_cmake.sh` 脚本需要目标名称来确定目标架构。以下是支持的目标的列表。

目标 | Target architecture | 注释
--- | --- | ---
armhf | ARMv7 VFP 带 Neon | 兼容 Raspberry Pi 3 和 4
rpi0 | ARMv6 | 兼容 Raspberry Pi Zero
aarch64 | aarch64（ARM 64 位） | [Coral Mendel Linux 4.0](https://coral.ai/) <br> Raspberry Pi with [Ubuntu Server 20.04.01 LTS 64 位](https://ubuntu.com/download/raspberry-pi)
native | 您的工作站 | 通过 "-mnative" 优化构建
<default></default> | 您的工作站 | 默认目标

### 构建示例

以下是您可以使用的一些命令示例。

#### 用于 Python 3.7 的 armhf 目标

```sh
make -C tensorflow/lite/tools/pip_package docker-build \
  TENSORFLOW_TARGET=armhf PYTHON_VERSION=3.7
```

#### 用于 Python 3.8 的 aarch64 目标

```sh
make -C tensorflow/lite/tools/pip_package docker-build \
  TENSORFLOW_TARGET=aarch64 PYTHON_VERSION=3.8
```

#### 如何使用自定义工具链？

如果生成的二进制文件与您的目标不兼容，则需要使用您自己的工具链或提供自定义的构建标志。（请查看[此页面](https://www.tensorflow.org/lite/guide/build_cmake_arm#check_your_target_environment)以了解您的目标环境）。在这种情况下，您需要修改 `tensorflow/lite/tools/cmake/download_toolchains.sh` 以使用您自己的工具链。工具链脚本为 `build_pip_package_with_cmake.sh` 脚本定义了以下两个变量。

变量 | Purpose | 示例
--- | --- | ---
ARMCC_PREFIX | 定义工具链前缀 | arm-linux-gnueabihf-
ARMCC_FLAGS | 编译标志 | -march=armv7-a -mfpu=neon-vfpv4

**注**：ARMCC_FLAGS 可能需要包含 Python 库（包括路径）。请参阅 `download_toolchains.sh` 作为参考。
