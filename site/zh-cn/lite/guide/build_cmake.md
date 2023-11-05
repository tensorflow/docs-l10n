# 使用 CMake 选择性构建 TensorFlow Lite

本页面介绍如何使用 [CMake](https://cmake.org/) 工具构建并使用 TensorFlow Lite 库。

以下说明已在 Ubuntu 16.04.3 64 位 PC (AMD64)、macOS Catalina (x86_64)、Windows 10 和 TensorFlow devel Docker 镜像 [tensorflow/tensorflow:devel](https://hub.docker.com/r/tensorflow/tensorflow/tags/) 上进行测试。

**注**：此功能从 2.4 版本开始提供。

### 第 1 步：安装 CMake 工具

需要 CMake 3.16 或更高版本。在 Ubuntu 上，您只需运行以下命令即可。

```sh
sudo apt-get install cmake
```

或者您也可以按照 [CMake 官方安装指南](https://cmake.org/install/)进行操作。

### 步骤 2. 克隆 TensorFlow 仓库

```sh
git clone https://github.com/tensorflow/tensorflow.git tensorflow_src
```

**注**：如果您使用的是 TensorFlow Docker 镜像，则 `/tensorflow_src/` 中已经提供了该仓库。

### 第 3 步：创建 CMake 构建目录

```sh
mkdir tflite_build
cd tflite_build
```

### 第 4 步：使用配置运行 CMake 工具

#### 发布构建

默认情况下，它会生成优化后的发布二进制文件。如果您想要为您的工作站进行构建，只需运行以下命令。

```sh
cmake ../tensorflow_src/tensorflow/lite
```

#### 调试构建

如果您需要生成具有符号信息的调试版本，则需要提供 `-DCMAKE_BUILD_TYPE=Debug` 选项。

```sh
cmake ../tensorflow_src/tensorflow/lite -DCMAKE_BUILD_TYPE=Debug
```

#### 使用内核单元测试进行构建

为了能够运行内核测试，您需要提供 {code 0}-DTFLITE_KERNEL_TEST=on{/code 0} 标志。单元测试交叉编译细节可以在下一小节中找到。

```sh
cmake ../tensorflow_src/tensorflow/lite -DTFLITE_KERNEL_TEST=on
```

#### 构建可安装软件包

要使用 `find_package(tensorflow-lite CONFIG)` 构建可被另一个 CMake 项目用作依赖项的可安装软件包，请使用 `-DTFLITE_ENABLE_INSTALL=ON` 选项。

理想情况下，您还应提供自有版本的库依赖项。依赖 TF Lite 的项目也需要使用这些依赖项。您可以使用 `-DCMAKE_FIND_PACKAGE_PREFER_CONFIG=ON` 并将 `<PackageName>_DIR` 变量设置为指向您的库安装。

```sh
cmake ../tensorflow_src/tensorflow/lite -DTFLITE_ENABLE_INSTALL=ON \
  -DCMAKE_FIND_PACKAGE_PREFER_CONFIG=ON \
  -DSYSTEM_FARMHASH=ON \
  -DSYSTEM_PTHREADPOOL=ON \
  -Dabsl_DIR=<install path>/lib/cmake/absl \
  -DEigen3_DIR=<install path>/share/eigen3/cmake \
  -DFlatBuffers_DIR=<install path>/lib/cmake/flatbuffers \
  -Dgemmlowp_DIR=<install path>/lib/cmake/gemmlowp \
  -DNEON_2_SSE_DIR=<install path>/lib/cmake/NEON_2_SSE \
  -Dcpuinfo_DIR=<install path>/share/cpuinfo \
  -Druy_DIR=<install path>/lib/cmake/ruy
```

**注**：请参阅 [`find_package`](https://cmake.org/cmake/help/latest/command/find_package.html) 的 CMake 文档以详细了解如何处理和定位软件包。

#### 交叉编译

您可以使用 CMake 为 ARM64 或 Android 目标架构构建二进制文件。

为了交叉编译 TF Lite，您需要提供 SDK 的路径（例如 Android 的 ARM64 SDK 或 NDK），并带 `-DCMAKE_TOOLCHAIN_FILE` 标志。

```sh
cmake -DCMAKE_TOOLCHAIN_FILE=<CMakeToolchainFileLoc> ../tensorflow/lite/
```

##### Android 交叉编译的详细信息

Android 交叉编译需要安装 [Android NDK](https://developer.android.com/ndk) 并为 NDK 路径提供上述 `-DCMAKE_TOOLCHAIN_FILE` 标志。您还需要使用 `-DANDROID_ABI` 标志来设置目标 ABI。

```sh
cmake -DCMAKE_TOOLCHAIN_FILE=<NDK path>/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a ../tensorflow_src/tensorflow/lite
```

##### 内核（单元）测试交叉编译的详细信息

单元测试的交叉编译需要用于主机架构的 flatc 编译器。为此，在`tensorflow/lite/tools/cmake/native_tools/flatbuffers` 中有一个 CMakeLists，用于使用主机工具链在单独的构建目录中使用 CMake 提前构建 flatc 编译器。

```sh
mkdir flatc-native-build && cd flatc-native-build
cmake ../tensorflow_src/tensorflow/lite/tools/cmake/native_tools/flatbuffers
cmake --build .
```

也可以将 *flatc* **安装**到自定义安装位置（例如，安装到包含其他本地构建工具的目录中，而不是 CMake 构建目录中）：

```sh
cmake -DCMAKE_INSTALL_PREFIX=<native_tools_dir> ../tensorflow_src/tensorflow/lite/tools/cmake/native_tools/flatbuffers
cmake --build .
```

对于 TF Lite 交叉编译本身，需要提供指向包含本机 *flatc* 二进制文件目录的附加参数 `-DTFLITE_HOST_TOOLS_DIR=<flatc_dir_path>`，以及上述 `-DTFLITE_KERNEL_TEST=on` 标志。

```sh
cmake -DCMAKE_TOOLCHAIN_FILE=${OE_CMAKE_TOOLCHAIN_FILE} -DTFLITE_KERNEL_TEST=on -DTFLITE_HOST_TOOLS_DIR=<flatc_dir_path> ../tensorflow/lite/
```

##### 在目标上启动交叉编译的内核（单元）测试

单元测试可以作为单独的可执行文件运行，也可以使用 CTest 实用工具运行。就 CTest 而言，如果为 TF Lite 版本启用了参数 `TFLITE_ENABLE_NNAPI, TFLITE_ENABLE_XNNPACK` 或 `TFLITE_EXTERNAL_DELEGATE` 中的至少一个，则生成的测试会带有两个不同的**标签**（使用相同的测试可执行文件）：- *普通* - 表示在 CPU 后端运行的测试 - *委托* - 表示测试期待用于所使用的委托规范的附加启动参数

`CTestTestfile.cmake` 和 `run-tests.cmake`（如下所述）都在 `<build_dir>/kernels` 中。

启动带有 CPU 后端的单元测试（前提是当前目录中的目标上存在 `CTestTestfile.cmake`）：

```sh
ctest -L plain
```

使用委托启动单元测试的示例（前提是当前目录中的目标上存在 `CTestTestfile.cmake` 和 `run-tests.cmake` 文件）：

```sh
cmake -E env TESTS_ARGUMENTS=--use_nnapi=true\;--nnapi_accelerator_name=vsi-npu ctest -L delegate
cmake -E env TESTS_ARGUMENTS=--use_xnnpack=true ctest -L delegate
cmake -E env TESTS_ARGUMENTS=--external_delegate_path=<PATH> ctest -L delegate
```

这种向单元测试提供与委托相关的其他启动参数的方式的一个**已知限制**是，它仅能有效支持**预期返回值为 0** 的参数。不同的返回值将被报告为测试失败。

#### OpenCL GPU 委托

如果您的目标机器支持 OpenCL，您可以使用可以利用您的 GPU 能力的 [GPU 委托](https://www.tensorflow.org/lite/performance/gpu)。

要配置 OpenCL GPU 委托支持，请运行以下代码：

```sh
cmake ../tensorflow_src/tensorflow/lite -DTFLITE_ENABLE_GPU=ON
```

**注**：此功能处于实验阶段，从 TensorFlow 2.5 开始提供。可能会出现兼容性问题。仅在 Android 设备和 NVidia CUDA OpenCL 1.2 上进行了验证。

### 第 5 步：构建 TensorFlow Lite

在 tflite_build 目录中运行以下代码，

```sh
cmake --build . -j
```

**注**：这将在当前目录中生成静态库 `libtensorflow-lite.a`，但该库无法独立使用，因为其中不包括所有可传递依赖项。要正确使用该库，您需要创建一个 CMake 项目。请参阅[“创建使用 TensorFlow Lite 的 CMake 项目”](#create_a_cmake_project_which_uses_tensorflow_lite)部分。

### 第 6 步：构建 TensorFlow Lite 基准测试工具和标签图像示例（可选）

在 tflite_build 目录中运行以下代码，

```sh
cmake --build . -j -t benchmark_model
```

```sh
cmake --build . -j -t label_image
```

## 可用于构建 TensorFlow Lite 的选项

以下是可用选项列表。您可以使用 `-D<option_name>=[ON|OFF]` 对其进行重写。例如，通过 `-DTFLITE_ENABLE_XNNPACK=OFF` 禁用默认情况下处于启用状态的 XNNPACK。

选项名称 | 功能 | Android | Linux | macOS | Windows
--- | --- | --- | --- | --- | ---
`TFLITE_ENABLE_RUY` | 启用 RUY | 开 | 关 | 关 | 关
:                       : matrix         :         :       :       :         : |  |  |  |  |
:                       : multiplication :         :       :       :         : |  |  |  |  |
:                       : library        :         :       :       :         : |  |  |  |  |
`TFLITE_ENABLE_NNAPI` | 启用 NNAPI | 开 | 关 | 不适用 | 不适用
:                       : 委托       :         :       :       :         : |  |  |  |  |
`TFLITE_ENABLE_GPU` | 启用 GPU | 关 | 关 | 不适用 | 不适用
:                       : 委托       :         :       :       :         : |  |  |  |  |
`TFLITE_ENABLE_XNNPACK` | 启用 XNNPACK | 开 | 开 | 开 | 开
:                       : 委托       :         :       :       :         : |  |  |  |  |
`TFLITE_ENABLE_MMAP` | 启用 MMAP | 开 | 开 | 开 | 不适用

## 创建使用 TensorFlow Lite 的 CMake 项目

以下是 [TFLite 最小示例](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/examples/minimal)的 CMakeLists.txt。

您需要为 TensorFlow Lite 目录添加 add_subdirectory() 并使用 target_link_libraries() 链接 `tensorflow-lite`。

```
cmake_minimum_required(VERSION 3.16)
project(minimal C CXX)

set(TENSORFLOW_SOURCE_DIR "" CACHE PATH
  "Directory that contains the TensorFlow project" )
if(NOT TENSORFLOW_SOURCE_DIR)
  get_filename_component(TENSORFLOW_SOURCE_DIR
    "${CMAKE_CURRENT_LIST_DIR}/../../../../" ABSOLUTE)
endif()

add_subdirectory(
  "${TENSORFLOW_SOURCE_DIR}/tensorflow/lite"
  "${CMAKE_CURRENT_BINARY_DIR}/tensorflow-lite" EXCLUDE_FROM_ALL)

add_executable(minimal minimal.cc)
target_link_libraries(minimal tensorflow-lite)
```

## 构建 TensorFlow Lite C 库

如果您想为 [C API](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/c/README.md) 构建 TensorFlow Lite 共享库，请先完成[第 1 步](#step-1-install-cmake-tool)到[第 3 步](#step-3-create-cmake-build-directory)。之后，运行以下命令。

```sh
cmake ../tensorflow_src/tensorflow/lite/c
cmake --build . -j
```

此命令在当前目录中生成以下共享库。

**注**：在 Windows 系统上，您可以在 `debug` 目录下找到 `tensorflowlite_c.dll`。

平台 | 库名称
--- | ---
Linux | `libtensorflowlite_c.so`
macOS | `libtensorflowlite_c.dylib`
Windows | `tensorflowlite_c.dll`

**注**：您需要公共头（`tensorflow/lite/c_api.h`、`tensorflow/lite/c_api_experimental.h`、`tensorflow/ lite/c_api_types.h` 和 `tensorflow/lite/common.h`），以及这些公共头包含的私有头（`tensorflow/lite/core/builtin_ops.h{ /code5}、<code data-md-type="codespan">tensorflow/lite/core/c/*.h` 和 `tensorflow/lite/core/async/c/*.h`）来使用生成的共享库。
