# Compilação cruzada do TensorFlow com o CMake

Esta página descreve como compilar a biblioteca do TensorFlow Lite para diversos dispositivos ARM.

As instruções abaixo foram testadas em um PC (AMD64) com Ubuntu 16.04.3 de 64 bits, na imagem devel docker do TensorFlow [tensorflow/tensorflow:devel](https://hub.docker.com/r/tensorflow/tensorflow/tags/).

**Observação:** esse recurso está disponível a partir a versão 2.4.

### Pré-requisitos

Você precisa instalar o CMake e baixar o código fonte do TensorFlow. Confira mais detalhes na página [Compile o TensorFlow Lite com o CMake](https://www.tensorflow.org/lite/guide/build_cmake).

### Confira o ambiente desejado

Os exemplos abaixo foram testados no Raspberry Pi OS, Ubuntu Server 20.04 LTS e Mendel Linux 4.0. Dependendo a versão do glibc desejada e dos recursos de CPU, talvez você precise usar uma versão diferente da toolchain e parâmetros de compilação diferentes.

#### Verifique a versão do glibc

```sh
ldd --version
```

<pre class="tfo-notebook-code-cell-output">
ldd (Debian GLIBC 2.28-10) 2.28
Copyright (C) 2018 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
Written by Roland McGrath and Ulrich Drepper.
</pre>

#### Verifique a compatibilidade do ABI

Se você deseja compilar para ARM de 32 bits, existem duas ABIs, dependendo da disponibilidade de VFP: [armhf](https://wiki.debian.org/ArmHardFloatPort) e [armel](https://wiki.debian.org/ArmEabiPort). Este documento mostra um exemplo para armhf. Você precisa usar uma toolchain diferente para compilar para armel.

#### Verifique a capacidade de CPU

Para o ARMv7, você precisa saber a versão com suporte a VFP e a disponibilidade de NEON da plataforma desejada.

```sh
cat /proc/cpuinfo
```

<pre class="tfo-notebook-code-cell-output">
processor   : 0
model name  : ARMv7 Processor rev 3 (v7l)
BogoMIPS    : 108.00
Features    : half thumb fastmult vfp edsp neon vfpv3 tls vfpv4 idiva idivt vfpd32 lpae evtstrm crc32
CPU implementer : 0x41
CPU architecture: 7
CPU variant : 0x0
CPU part    : 0xd08
CPU revision    : 3
</pre>

## Compile para AArch64 (ARM64)

Estas instruções mostram como compilar o binário AArch64 que é compatível com [Coral Mendel Linux 4.0](https://coral.ai/), Raspberry Pi (com o [Ubuntu Server 20.04.01 LTS de 64 bits](https://ubuntu.com/download/raspberry-pi) instalado).

#### Baixe a toolchain

Estes comandos instalam a toolchain `gcc-arm-8.3-2019.03-x86_64-aarch64-linux-gnu` em ${HOME}/toolchains.

```sh
curl -LO https://storage.googleapis.com/mirror.tensorflow.org/developer.arm.com/media/Files/downloads/gnu-a/8.3-2019.03/binrel/gcc-arm-8.3-2019.03-x86_64-aarch64-linux-gnu.tar.xz
mkdir -p ${HOME}/toolchains
tar xvf gcc-arm-8.3-2019.03-x86_64-aarch64-linux-gnu.tar.xz -C ${HOME}/toolchains
```

**Observação:** os binários compilados com GCC 8.3 requerem o glibc 2.28 ou superior. Se a plataforma desejada tiver uma versão inferior do glibc, você precisa usar a toolchain GCC mais antiga.

#### Execute o CMake

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

**Observação:** você pode ativar o delegado de GPU com `-DTFLITE_ENABLE_GPU=ON` se o dispositivo desejado tiver suporte ao OpenCL 1.2 ou superior.

## Compile para ARMv7 com NEON

Estas instruções mostram como compilar o binário ARMv7 com VFPv4 e NEON que é compatível com o Raspberry Pi 3 e 4.

#### Baixe a toolchain

Estes comandos instalam a toolchain `gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf` em ${HOME}/toolchains.

```sh
curl -LO https://storage.googleapis.com/mirror.tensorflow.org/developer.arm.com/media/Files/downloads/gnu-a/8.3-2019.03/binrel/gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf.tar.xz
mkdir -p ${HOME}/toolchains
tar xvf gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf.tar.xz -C ${HOME}/toolchains
```

**Observação:** os binários compilados com GCC 8.3 requerem o glibc 2.28 ou superior. Se a plataforma desejada tiver uma versão inferior do glibc, você precisa usar a toolchain GCC mais antiga.

#### Execute o CMake

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

**Observação:** como a arquitetura ARMv7 é muito diversa, talvez você precise atualizar `ARMCC_FLAGS` para os perfis de dispositivos desejados. Por exemplo, ao compilar com XNNPACK ativado (ou seja, `XNNPACK=ON`) no Tensorflow Lite 2.8, adicione `-mfp16-format=ieee` a `ARMCC_FLAGS`.

## Compile para o Raspberry Pi Zero (ARMv6)

Estas instruções mostram como compilar o binário ARMv6 que é compatível com o Raspberry Pi Zero.

#### Baixe a toolchain

Estes comandos instalam a toolchain `gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf` em ${HOME}/toolchains.

```sh
curl -LO https://storage.googleapis.com/mirror.tensorflow.org/developer.arm.com/media/Files/downloads/gnu-a/8.3-2019.03/binrel/gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf.tar.xz
mkdir -p ${HOME}/toolchains
tar xvf gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf.tar.xz -C ${HOME}/toolchains
```

**Observação:** os binários compilados com GCC 8.3 requerem o glibc 2.28 ou superior. Se a plataforma desejada tiver uma versão inferior do glibc, você precisa usar a toolchain GCC mais antiga.

#### Execute o CMake

```sh
ARMCC_FLAGS="-march=armv6 -mfpu=vfp -mfloat-abi=hard -funsafe-math-optimizations"
ARMCC_PREFIX=${HOME}/toolchains/gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf/bin/arm-linux-gnueabihf-
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

**Observação:** o XNNPACK está desativado, já que não há suporte ao NEON.
