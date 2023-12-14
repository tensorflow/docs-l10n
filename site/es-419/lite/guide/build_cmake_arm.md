# Compilación cruzada TensorFlow Lite con CMake

Esta página describe cómo generar la librería TensorFlow Lite para varios dispositivos ARM.

Las siguientes instrucciones han sido analizadas en Ubuntu 16.04.3 64-bit PC (AMD64) y la imagen docker devel TensorFlow devel [tensorflow/tensorflow:devel](https://hub.docker.com/r/tensorflow/tensorflow/tags/).

**Nota:** Esta función está disponible desde la versión 2.4.

### Requisitos previos

Necesita tener instalado CMake y descargado el código fuente de TensorFlow. Visite la página [Generar TensorFlow Lite con CMake](https://www.tensorflow.org/lite/guide/build_cmake) para más detalles.

### Verifique su entorno destino

Los siguientes ejemplos han sido analizados bajo Raspberry Pi OS, Ubuntu Server 20.04 LTS y Mendel Linux 4.0. Dependiendo de la versión glibc de su destino y de las capacidades de su CPU, es posible que tenga que usar una versión diferente de cadena de herramientas y de los parámetros de compilación.

#### Comprobación de la versión glibc

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

#### Comprobación de la compatibilidad de ABI

Si su destino es ARM 32-bit, hay dos ABI disponibles dependiendo de la disponibilidad de VFP. [armhf](https://wiki.debian.org/ArmHardFloatPort) y [armel](https://wiki.debian.org/ArmEabiPort). Este documento muestra un ejemplo armhf, necesita usar una cadena de herramientas diferente para destinos armel.

#### Comprobar la compatibilidad de la CPU

Para ARMv7, debe conocer la versión de VFP soportada por el destino y la disponibilidad de NEON.

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

## Generación para AArch64 (ARM64)

Estas instrucciones muestran cómo generar un binario AArch64 compatible con [Coral Mendel Linux 4.0](https://coral.ai/), Raspberry Pi (con [Ubuntu Server 20.04.01 LTS 64-bit](https://ubuntu.com/download/raspberry-pi) instalado).

#### Descargar la cadena de herramientas

Estos comandos instalan la cadena de herramientas `gcc-arm-8.3-2019.03-x86_64-aarch64-linux-gnu` bajo ${HOME}/toolchains.

```sh
curl -LO https://storage.googleapis.com/mirror.tensorflow.org/developer.arm.com/media/Files/downloads/gnu-a/8.3-2019.03/binrel/gcc-arm-8.3-2019.03-x86_64-aarch64-linux-gnu.tar.xz
mkdir -p ${HOME}/toolchains
tar xvf gcc-arm-8.3-2019.03-x86_64-aarch64-linux-gnu.tar.xz -C ${HOME}/toolchains
```

**Nota:** Los binarios generados con GCC 8.3 requieren glibc 2.28 o superior. Si su destino tiene una versión de glibc inferior, deberá usar una cadena de herramientas GCC más antigua.

#### Ejecutar CMake

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

**Nota:** Puede habilitar el delegado de GPU con `-DTFLITE_ENABLE_GPU=ON` si su dispositivo de destino admite OpenCL 1.2 o superior.

## Generar para ARMv7 NEON habilitado

Esta instrucción muestra cómo generar ARMv7 con VFPv4 y NEON habilitado binario que es compatible con Raspberry Pi 3 y 4.

#### Descargar la cadena de herramientas

Estos comandos instalan la cadena de herramientas `gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf` en ${HOME}/toolchains.

```sh
curl -LO https://storage.googleapis.com/mirror.tensorflow.org/developer.arm.com/media/Files/downloads/gnu-a/8.3-2019.03/binrel/gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf.tar.xz
mkdir -p ${HOME}/toolchains
tar xvf gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf.tar.xz -C ${HOME}/toolchains
```

**Nota:** Los binarios generados con GCC 8.3 requieren glibc 2.28 o superior. Si su destino tiene una versión de glibc inferior, deberá usar una cadena de herramientas GCC más antigua.

#### Ejecutar CMake

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

**Nota:** Dado que la arquitectura ARMv7 es diversa, puede que necesite actualizar `ARMCC_FLAGS` para los perfiles de su dispositivo de destino. Por ejemplo, al compilar con XNNPACK activado (es decir, `XNNPACK=ON`) en Tensorflow Lite 2.8, añada `-mfp16-format=ieee` a `ARMCC_FLAGS`.

## Generar para Raspberry Pi Zero (ARMv6)

Esta instrucción muestra cómo generar un binario ARMv6 compatible con la Raspberry Pi Zero.

#### Descargar la cadena de herramientas

Estos comandos instalan la cadena de herramientas `gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf` en ${HOME}/toolchains.

```sh
curl -LO https://storage.googleapis.com/mirror.tensorflow.org/developer.arm.com/media/Files/downloads/gnu-a/8.3-2019.03/binrel/gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf.tar.xz
mkdir -p ${HOME}/toolchains
tar xvf gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf.tar.xz -C ${HOME}/toolchains
```

**Nota:** Los binarios generados con GCC 8.3 requieren glibc 2.28 o superior. Si su destino tiene una versión de glibc inferior, deberá usar una cadena de herramientas GCC más antigua.

#### Ejecutar CMake

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

**Nota:** XNNPACK está desactivado ya que no hay soporte para NEON.
