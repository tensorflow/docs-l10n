# 使用 CMake 对 TensorFlow Lite 进行交叉编译

本页介绍如何为各种 ARM 设备构建 TensorFlow Lite 库。

以下说明已在 Ubuntu 16.04.3 64 位 PC (AMD64) 和 TensorFlow devel docker 镜像 [tensorflow/tensorflow:devel](https://hub.docker.com/r/tensorflow/tensorflow/tags/) 上进行测试。

**注**：此功能从 2.4 版本开始提供。

### <a>先决条件</a>

您需要安装 CMake 并下载 TensorFlow 源代码。有关详细信息，请参阅[使用 CMake 构建 TensorFlow Lite](https://www.tensorflow.org/lite/guide/build_cmake) 页面。

### 检查您的目标环境

以下示例已在 Raspberry Pi OS、Ubuntu Server 20.04 LTS 和 Mendel Linux 4.0 上进行测试。根据您的目标 glibc 版本和 CPU 能力的不同，您可能需要使用不同版本的工具链和构建参数。

#### 检查 glibc 版本

```sh
ldd --version
```

<pre class="tfo-notebook-code-cell-output">ldd (Debian GLIBC 2.28-10) 2.28
Copyright (C) 2018 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
Written by Roland McGrath and Ulrich Drepper.
</pre>

#### 检查 ABI 兼容性

如果您的目标是 ARM 32 位，则根据 VFP 可用性，有两个 ABI 可用：[armhf](https://wiki.debian.org/ArmHardFloatPort) 和 [armel](https://wiki.debian.org/ArmEabiPort)。本文档展示了一个 armhf 示例，您需要为 armel 目标使用不同的工具链。

#### 检查 CPU 能力

对于 ARMv7，您应该知道目标支持的 VFP 版本和 NEON 可用性。

```sh
cat /proc/cpuinfo
```

<pre class="tfo-notebook-code-cell-output">processor   : 0
model name  : ARMv7 Processor rev 3 (v7l)
BogoMIPS    : 108.00
Features    : half thumb fastmult vfp edsp neon vfpv3 tls vfpv4 idiva idivt vfpd32 lpae evtstrm crc32
CPU implementer : 0x41
CPU architecture: 7
CPU variant : 0x0
CPU part    : 0xd08
CPU revision    : 3
</pre>

## 为 AArch64 (ARM64) 构建

本说明介绍如何构建与 [Coral Mendel Linux 4.0](https://coral.ai/) 和 Raspberry Pi（已安装 [Ubuntu Server 20.04.01 LTS 64-bit](https://ubuntu.com/download/raspberry-pi)）兼容的 AArch64 二进制文件。

#### 下载工具链

以下命令会将 gcc-arm-8.3-2019.03-x86_64-aarch64-linux-gnu 工具链安装到 ${HOME}/toolchains 下。

```sh
curl -LO https://storage.googleapis.com/mirror.tensorflow.org/developer.arm.com/media/Files/downloads/gnu-a/8.3-2019.03/binrel/gcc-arm-8.3-2019.03-x86_64-aarch64-linux-gnu.tar.xz
mkdir -p ${HOME}/toolchains
tar xvf gcc-arm-8.3-2019.03-x86_64-aarch64-linux-gnu.tar.xz -C ${HOME}/toolchains
```

**注**：使用 GCC 8.3 构建的二进制文件需要 glibc 2.28 或更高版本。如果您的目标是较低版本的 glibc，则需要使用旧版 GCC 工具链。

#### 运行 CMake

```sh
ARMCC_PREFIX=${HOME}/toolchains/gcc-arm-8.3-2019.03-x86_64-aarch64-linux-gnu/bin/aarch64-linux-gnu-
ARMCC_FLAGS="-funsafe-math-optimizations"
cmake -DCMAKE_C_COMPILER=${ARMCC_PREFIX}gcc \
  -DCMAKE_CXX_COMPILER=${ARMCC_PREFIX}g++ \
  -DCMAKE_C_FLAGS="${ARMCC_FLAGS}" \
  -DCMAKE_CXX_FLAGS="${ARMCC_FLAGS}" \
  -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
  -DCMAKE_SYSTEM_NAME=Linux \
  -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
  ../tensorflow/lite/
```

**注**：如果您的目标设备支持 OpenCL 1.2 或更高版本，您可以使用 "-DTFLITE_ENABLE_GPU=ON" 启用 GPU 委托。

## 为已启用 NEON 的 ARMv7 构建

本说明介绍如何使用 VFPv4 和与 Raspberry Pi 3 和 4 兼容的已启用 NEON 的二进制文件构建 ARMv7。

#### 下载工具链

以下命令会将 gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf 工具链安装到 ${HOME}/toolchains 下。

```sh
curl -LO https://storage.googleapis.com/mirror.tensorflow.org/developer.arm.com/media/Files/downloads/gnu-a/8.3-2019.03/binrel/gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf.tar.xz
mkdir -p ${HOME}/toolchains
tar xvf gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf.tar.xz -C ${HOME}/toolchains
```

**注**：使用 GCC 8.3 构建的二进制文件需要 glibc 2.28 或更高版本。如果您的目标是较低版本的 glibc，则需要使用旧版 GCC 工具链。

#### 运行 CMake

```sh
ARMCC_FLAGS="-march=armv7-a -mfpu=neon-vfpv4 -funsafe-math-optimizations -mfp16-format=ieee"
ARMCC_PREFIX=${HOME}/toolchains/gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf/bin/arm-linux-gnueabihf-
cmake -DCMAKE_C_COMPILER=${ARMCC_PREFIX}gcc \
  -DCMAKE_CXX_COMPILER=${ARMCC_PREFIX}g++ \
  -DCMAKE_C_FLAGS="${ARMCC_FLAGS}" \
  -DCMAKE_CXX_FLAGS="${ARMCC_FLAGS}" \
  -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
  -DCMAKE_SYSTEM_NAME=Linux \
  -DCMAKE_SYSTEM_PROCESSOR=armv7 \
  ../tensorflow/lite/
```

**注**：由于 ARMv7 架构的多样性，您可能需要为您的目标设备配置文件更新 ARMCC_FLAGS。例如，在 Tensorflow Lite 2.8 中启用 XNNPACK（即，`XNNPACK=ON`）进行编译时，请将 `-mfp16-format=ieee` 添加到 ARMCC_FLAGS。

## 为 Raspberry Pi Zero (ARMv6) 构建

本说明介绍如何构建与 Raspberry Pi Zero 兼容的 ARMv6 二进制文件。

#### 下载工具链

以下命令会将 arm-rpi-linux-gnueabihf 工具链安装到 ${HOME}/toolchains 下。

```sh
curl -L https://github.com/rvagg/rpi-newer-crosstools/archive/eb68350c5c8ec1663b7fe52c742ac4271e3217c5.tar.gz -o rpi-toolchain.tar.gz
tar xzf rpi-toolchain.tar.gz -C ${HOME}/toolchains
mv ${HOME}/toolchains/rpi-newer-crosstools-eb68350c5c8ec1663b7fe52c742ac4271e3217c5 ${HOME}/toolchains/arm-rpi-linux-gnueabihf
```

#### 运行 CMake

```sh
ARMCC_PREFIX=${HOME}/toolchains/arm-rpi-linux-gnueabihf/x64-gcc-6.5.0/arm-rpi-linux-gnueabihf/bin/arm-rpi-linux-gnueabihf-
ARMCC_FLAGS="-march=armv6 -mfpu=vfp -funsafe-math-optimizations"
cmake -DCMAKE_C_COMPILER=${ARMCC_PREFIX}gcc \
  -DCMAKE_CXX_COMPILER=${ARMCC_PREFIX}g++ \
  -DCMAKE_C_FLAGS="${ARMCC_FLAGS}" \
  -DCMAKE_CXX_FLAGS="${ARMCC_FLAGS}" \
  -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
  -DCMAKE_SYSTEM_NAME=Linux \
  -DCMAKE_SYSTEM_PROCESSOR=armv6 \
  -DTFLITE_ENABLE_XNNPACK=OFF \
  ../tensorflow/lite/
```

**注**：由于没有 NEON 支持，禁用了 XNNPACK。
