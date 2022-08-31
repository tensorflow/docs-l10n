# CMake를 사용한 교차 컴파일 TensorFlow Lite

이 페이지에서는 다양한 ARM 장치용 TensorFlow Lite 라이브러리를 빌드하는 방법을 설명합니다.

다음 지침은 Ubuntu 16.04.3 64비트 PC(AMD64), TensorFlow devel docker image [tensorflow/tensorflow:devel](https://hub.docker.com/r/tensorflow/tensorflow/tags/)에서 테스트되었습니다.

**참고:** 이 기능은 버전 2.4부터 사용할 수 있습니다.

### Prerequisites

CMake를 설치하고 TensorFlow 소스 코드를 다운로드해야 합니다. 자세한 내용 [CMake로 TensorFlow Lite 빌드하기](https://www.tensorflow.org/lite/guide/build_cmake) 페이지를 확인하세요.

### 타겟 환경 확인하기

다음 예제는 Raspberry Pi OS, Ubuntu Server 20.04 LTS 및 Mendel Linux 4.0에서 테스트되었습니다. 타겟 glibc 버전 및 CPU 성능에 따라 다른 버전의 툴체인 및 빌드 매개변수를 사용해야 할 수도 있습니다.

#### glibc 버전 확인하기

```sh
ldd --version
```

<pre class="tfo-notebook-code-cell-output">ldd (Debian GLIBC 2.28-10) 2.28
Copyright (C) 2018 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
Written by Roland McGrath and Ulrich Drepper.
</pre>

#### ABI 호환성 확인하기

타겟이 ARM 32비트인 경우, VFP 가용성에 따라 [armhf](https://wiki.debian.org/ArmHardFloatPort) 및 [armel](https://wiki.debian.org/ArmEabiPort)의 두 가지 ABI를 사용할 수 있습니다. 이 문서에서는 armhf 예제를 보여주며 armel 타겟에는 다른 툴체인을 사용해야 합니다.

#### CPU 성능 확인하기

ARMv7의 경우, 타겟의 지원되는 VFP 버전과 NEON 가용성을 알아야 합니다.

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

## AArch64(ARM64)용 빌드

이 지침은 [Coral Mendel Linux 4.0](https://coral.ai/), Raspberry Pi([Ubuntu Server 20.04.01 LTS 64비트](https://ubuntu.com/download/raspberry-pi) 설치)와 호환되는 AArch64 바이너리를 빌드하는 방법을 보여줍니다.

#### 툴체인 다운로드

다음 명령으로 ${HOME}/toolchains 아래에 gcc-arm-8.3-2019.03-x86_64-aarch64-linux-gnu 툴체인이 설치됩니다.

```sh
curl -LO https://storage.googleapis.com/mirror.tensorflow.org/developer.arm.com/media/Files/downloads/gnu-a/8.3-2019.03/binrel/gcc-arm-8.3-2019.03-x86_64-aarch64-linux-gnu.tar.xz
mkdir -p ${HOME}/toolchains
tar xvf gcc-arm-8.3-2019.03-x86_64-aarch64-linux-gnu.tar.xz -C ${HOME}/toolchains
```

**참고:** GCC 8.3으로 빌드된 바이너리에는 glibc 2.28 이상이 필요합니다. 타겟의 glibc 버전이 더 낮으면 이전 GCC 툴체인을 사용해야 합니다.

#### CMake 실행하기

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

**참고:** 타겟 장치가 OpenCL 1.2 이상을 지원하는 경우 "-DTFLITE_ENABLE_GPU=ON"으로 GPU 대리자를 활성화할 수 있습니다.

## ARMv7 NEON용 빌드 활성화

이 지침은 Raspberry Pi 3 및 4와 호환되는 VFPv4 및 NEON 지원 바이너리로 ARMv7을 빌드하는 방법을 보여줍니다.

#### 툴체인 다운로드

다음 명령으로 ${HOME}/toolchains 아래에 gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf 툴체인이 설치됩니다.

```sh
curl -LO https://storage.googleapis.com/mirror.tensorflow.org/developer.arm.com/media/Files/downloads/gnu-a/8.3-2019.03/binrel/gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf.tar.xz
mkdir -p ${HOME}/toolchains
tar xvf gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf.tar.xz -C ${HOME}/toolchains
```

**참고:** GCC 8.3으로 빌드된 바이너리에는 glibc 2.28 이상이 필요합니다. 타겟의 glibc 버전이 더 낮으면 이전 GCC 툴체인을 사용해야 합니다.

#### CMake 실행하기

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

**참고:** ARMv7 아키텍처는 다양하기 때문에 타겟 장치 프로파일에 대해 ARMCC_FLAGS를 업데이트해야 할 수 있습니다. 예를 들어, Tensorflow Lite 2.8에서 XNNPACK이 활성화된 상태(즉, `XNNPACK=ON`)로 컴파일하는 경우, ARMCC_FLAGS에 `-mfp16-format=ieee`를 추가하세요.

## Raspberry Pi Zero(ARMv6)용 빌드

이 지침은 Raspberry Pi Zero와 호환되는 ARMv6 바이너리를 빌드하는 방법을 보여줍니다.

#### 툴체인 다운로드

다음 명령으로 ${HOME}/toolchains 아래에 arm-rpi-linux-gnueabihf 툴체인을 설치합니다.

```sh
curl -L https://github.com/rvagg/rpi-newer-crosstools/archive/eb68350c5c8ec1663b7fe52c742ac4271e3217c5.tar.gz -o rpi-toolchain.tar.gz
tar xzf rpi-toolchain.tar.gz -C ${HOME}/toolchains
mv ${HOME}/toolchains/rpi-newer-crosstools-eb68350c5c8ec1663b7fe52c742ac4271e3217c5 ${HOME}/toolchains/arm-rpi-linux-gnueabihf
```

#### CMake 실행하기

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

**참고:** NEON 지원이 없기 때문에 XNNPACK이 비활성화됩니다.
