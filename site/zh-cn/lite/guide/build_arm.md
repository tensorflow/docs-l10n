# 为 ARM 开发板构建 TensorFlow Lite

本页介绍了如何为基于 ARM 的计算机构建 TensorFlow Lite 库。

TensorFlow Lite 支持两种构建系统，而每种构建系统支持的功能不完全相同。请参考下表选择合适的构建系统。

功能 | Bazel | CMake
--- | --- | ---
预定义工具链 | armhf、aarch64 | armel、armhf、aarch64
自定义工具链 | 难用 | 易用
[Select TF ops](https://www.tensorflow.org/lite/guide/ops_select) | 支持 | 不支持
[GPU delegate](https://www.tensorflow.org/lite/performance/gpu) | 仅适用于 Android | 任何支持 OpenCL 的平台
XNNPack | 支持 | 支持
[Python Wheel](https://www.tensorflow.org/lite/guide/build_cmake_pip) | 支持 | 支持
[C API](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/c/README.md) | 支持 | [supported](https://www.tensorflow.org/lite/guide/build_cmake#build_tensorflow_lite_c_library)
[C++ API](https://www.tensorflow.org/lite/guide/inference#load_and_run_a_model_in_c) | 支持 Bazel 项目 | 支持 CMake 项目

## 使用 CMake 对 ARM 进行交叉编译

如果您有一个 CMake 项目，或者您想使用自定义工具链，那么您最好使用 CMake 进行交叉编译。有一个单独的[使用 CMake 交叉编译 TensorFlow Lite](https://www.tensorflow.org/lite/guide/build_cmake_arm) 页面可供参考。

## 使用 Bazel 对 ARM 进行交叉编译

如果您有一个 Bazel 项目，或者您想使用 TF 运算，那么您最好使用 Bazel 构建系统。您将使用集成的 [ARM GCC 8.3 工具链](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/toolchains/embedded/arm-linux)配合 Bazel 构建 ARM32/64 共享库。

目标架构 | Bazel 配置 | 兼容设备
--- | --- | ---
armhf (ARM32) | --config=elinux_armhf | RPI3、RPI4 运行 32 位
:                     :                         : Raspberry Pi OS            : |  |
AArch64 (ARM64) | --config=elinux_aarch64 | Coral、RPI4 运行 Ubuntu 64
:                     :                         : 位                        : |  |

注：生成的共享库需要 glibc 2.28 或更高版本才能运行。

以下指令已在 Ubuntu 16.04.3 64 位 PC (AMD64) 和 TensorFlow devel docker 镜像 [tensorflow/tensorflow:devel](https://hub.docker.com/r/tensorflow/tensorflow/tags/) 上进行了测试。

要使用 Bazel 交叉编译 TensorFlow Lite，请按照以下步骤进行操作：

#### 第 1 步：安装 Bazel

Bazel 是 TensorFlow 的主要构建系统。安装最新版本的 [Bazel 构建系统](https://bazel.build/versions/master/docs/install.html)。

**注**：如果您使用的是 TensorFlow Docker 镜像，则可直接使用 Bazel。

#### 步骤 2. 克隆 TensorFlow 仓库

```sh
git clone https://github.com/tensorflow/tensorflow.git tensorflow_src
```

**注**：如果您使用的是 TensorFlow Docker 镜像，则 `/tensorflow_src/` 中已经提供了该仓库。

#### 第 3 步：构建 ARM 二进制文件

##### C 库

```bash
bazel build --config=elinux_aarch64 -c opt //tensorflow/lite/c:libtensorflowlite_c.so
```

您可以在以下位置找到共享库：`bazel-bin/tensorflow/lite/c/libtensorflowlite_c.so`。

**注**：请使用 `elinux_armhf` 进行 [32 位 ARM 硬浮点](https://wiki.debian.org/ArmHardFloatPort)构建。

请查看 [TensorFlow Lite C API](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/c/README.md) 页面了解详细信息。

##### C++ 库

```bash
bazel build --config=elinux_aarch64 -c opt //tensorflow/lite:libtensorflowlite.so
```

您可以在以下位置找到共享库：`bazel-bin/tensorflow/lite/libtensorflowlite.so`。

目前，没有一种直接方式可以提取需要的所有头文件，因此您必须将来自 TensorFlow 仓库的所有头文件都包含在 tensorflow/lite/ 中。此外，您还将需要来自 FlatBuffers 和 Abseil 的头文件。

##### 其他信息

您也可以用工具链构建其他 Bazel 目标。下面是一些有用的目标。

- //tensorflow/lite/tools/benchmark:benchmark_model
- //tensorflow/lite/examples/label_image:label_image
