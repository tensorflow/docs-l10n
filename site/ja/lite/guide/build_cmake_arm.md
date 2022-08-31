# CMake を使用した TensorFlow Lite のクロスコンパイル

このページでは、さまざまな ARM デバイスの TensorFlow Lite ライブラリを構築する方法について説明します。

次の手順は、Ubuntu 16.04.3 64 ビット PC（AMD64）および TensorFlow devel Docker イメージ [tensorflow/tensorflow:devel](https://hub.docker.com/r/tensorflow/tensorflow/tags/) でテストされています。

**注意:** この機能はバージョン 2.4 以降で利用できます。

### Prerequisites

CMake をインストールし、TensorFlow ソースコードをダウンロードする必要があります。詳細については、[CMake を使用した TensorFlow Lite の構築](https://www.tensorflow.org/lite/guide/build_cmake)ページを参照してください。

### ターゲット環境の確認

次の例は、Raspberry Pi OS、Ubuntu Server 20.04 LTS、および Mendel Linux 4.0 でテストされています。ターゲット glibc バージョンと CPU 能力によっては、別のバージョンのツールチェーンとビルドパラメータを使用しなければならない場合があります。

#### glibc バージョンの確認

```sh
ldd --version
```

<pre class="tfo-notebook-code-cell-output">ldd (Debian GLIBC 2.28-10) 2.28
Copyright (C) 2018 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
Written by Roland McGrath and Ulrich Drepper.
</pre>

#### ABI 互換性の確認

ターゲットが ARM 32 ビットの場合、VFP の利用可能状況に応じて、2 つの ABI ([armhf](https://wiki.debian.org/ArmHardFloatPort) と [armel](https://wiki.debian.org/ArmEabiPort)) を利用できます。このドキュメントでは、armh の例を示します。armel ターゲットでは別のツールチェーンを使用する必要があります。

#### CPU 能力の確認

ARMv7 では、ターゲットのサポートされている VFP バージョンと NEON の利用可能状況を確認してください。

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

## AArch64 (ARM64) 向けの構築

この手順では、[Coral Mendel Linux 4.0](https://coral.ai/)、Raspberry Pi ([Ubuntu Server 20.04.01 LTS 64-bit](https://ubuntu.com/download/raspberry-pi) がインストールされていること) と互換性がある AArch64 バイナリを構築する方法について説明します。

#### ツールチェーンのダウンロード

これらのコマンドを実行すると、gcc-arm-8.3-2019.03-x86_64-aarch64-linux-gnu ツールチェーンが ${HOME}/toolchains の下にインストールされます。

```sh
curl -LO https://storage.googleapis.com/mirror.tensorflow.org/developer.arm.com/media/Files/downloads/gnu-a/8.3-2019.03/binrel/gcc-arm-8.3-2019.03-x86_64-aarch64-linux-gnu.tar.xz
mkdir -p ${HOME}/toolchains
tar xvf gcc-arm-8.3-2019.03-x86_64-aarch64-linux-gnu.tar.xz -C ${HOME}/toolchains
```

**注意:** GCC 8.3 で構築されたバイナリでは、glibc 2.28 以上が必要です。glibc バージョンがこれよりも低い場合は、古い GCC ツールチェーンを使用する必要があります。

#### CMake の実行

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

**注意:** ターゲットデバイスで OpenCL 1.2 以上がサポートされている場合は、"-DTFLITE_ENABLE_GPU=ON" を使用して GPU デリゲートを有効化できます。

## NEON に対応した ARMv7 向けの構築

この手順では、VFPv4 および NEON に対応したバイナリを使用して、Raspberry Pi 3 および 4 と互換性がある ARMv7 を構築する方法を示します。

#### ツールチェーンのダウンロード

これらのコマンドを実行すると、gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf ツールチェーンが ${HOME}/toolchains の下にインストールされます。

```sh
curl -LO https://storage.googleapis.com/mirror.tensorflow.org/developer.arm.com/media/Files/downloads/gnu-a/8.3-2019.03/binrel/gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf.tar.xz
mkdir -p ${HOME}/toolchains
tar xvf gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf.tar.xz -C ${HOME}/toolchains
```

**注意:** GCC 8.3 で構築されたバイナリでは、glibc 2.28 以上が必要です。glibc バージョンがこれよりも低い場合は、古い GCC ツールチェーンを使用する必要があります。

#### CMake の実行

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

**注意:** ARMv7 アーキテクチャは多様であるため、ターゲットデバイスプロファイルの ARMCC_FLAGS を更新しなければならない場合があります。たとえば、Tensorflow Lite 2.8 で XNNPACK を有効 (`XNNPACK=ON`) にしてコンパイルするときには、`-mfp16-format=ieee` を ARMCC_FLAGS に追加してください。

## Raspberry Pi Zero (ARMv6) 向けの構築

この手順では、Raspberry Pi Zero と互換性がある ARMv6 バイナリを構築する方法を示します。

#### ツールチェーンのダウンロード

これらのコマンドを実行すると、arm-rpi-linux-gnueabihf ツールチェーンが ${HOME}/toolchains の下にインストールされます。

```sh
curl -L https://github.com/rvagg/rpi-newer-crosstools/archive/eb68350c5c8ec1663b7fe52c742ac4271e3217c5.tar.gz -o rpi-toolchain.tar.gz
tar xzf rpi-toolchain.tar.gz -C ${HOME}/toolchains
mv ${HOME}/toolchains/rpi-newer-crosstools-eb68350c5c8ec1663b7fe52c742ac4271e3217c5 ${HOME}/toolchains/arm-rpi-linux-gnueabihf
```

#### CMake の実行

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

**注意:** NEON サポートがないため、XNNPACK は無効です。
