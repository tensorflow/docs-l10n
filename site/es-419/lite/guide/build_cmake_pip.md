# Construir el paquete de wheels de TensorFlow Lite Python

Esta página describe cómo construir la librería Python de TensorFlow Lite `tflite_runtime` para x86_64 y varios dispositivos ARM.

Las siguientes instrucciones han sido analizadas en Ubuntu 16.04.3 64-bit PC (AMD64), macOS Catalina (x86_64) y la imagen docker devel de TensorFlow [tensorflow/tensorflow:devel](https://hub.docker.com/r/tensorflow/tensorflow/tags/).

**Nota:** Esta función está disponible desde la versión 2.4.

#### Requisitos previos

Necesita tener instalado CMake y una copia del código fuente de TensorFlow. Visite la página [Generar TensorFlow Lite con CMake](https://www.tensorflow.org/lite/guide/build_cmake) para más detalles.

Para construir el paquete PIP para su estación de trabajo, puede ejecutar los siguientes comandos.

```sh
PYTHON=python3 tensorflow/lite/tools/pip_package/build_pip_package_with_cmake.sh native
```

**Nota:** Si dispone de varios intérpretes de Python, especifique la versión exacta de Python con la variable `PYTHON`. (Por el momento, admite Python 3.7 o superior)

## Compilación cruzada ARM

Para la compilación cruzada ARM, se recomienda usar Docker ya que hace más fácil configurar el entorno de compilación cruzada. También necesita una opción `target` para averiguar la arquitectura de destino.

Hay una herramienta de ayuda en Makefile `tensorflow/lite/tools/pip_package/Makefile` disponible para invocar un comando de compilación utilizando un contenedor Docker predefinido. En una máquina host Docker, puede ejecutar un comando de compilación de la siguiente manera.

```sh
make -C tensorflow/lite/tools/pip_package docker-build \
  TENSORFLOW_TARGET=<target> PYTHON_VERSION=<python3 version>
```

**Nota:** Admite la versión 3.7 o superior de Python.

### Nombres de destino disponibles

`tensorflow/lite/tools/pip_package/build_pip_package_with_cmake.sh` el script necesita un nombre de destino para averiguar la arquitectura de destino. Esta es la lista de destinos admitidos.

Destino | Arquitectura destino | Comentarios
--- | --- | ---
armhf | ARMv7 VFP con Neon | Compatible con Raspberry Pi 3 y 4
rpi0 | ARMv6 | Compatible con Raspberry Pi Zero
aarch64 | aarch64 (ARM de 64 bits) | [Coral Mendel Linux 4.0](https://coral.ai/) <br> Raspberry Pi con [Ubuntu Server 20.04.01 LTS de 64 bits](https://ubuntu.com/download/raspberry-pi)
nativo | Su estación de trabajo | Se genera con la optimización "-mnative"
<default></default> | Su estación de trabajo | Destino por default

### Ejemplos de generación

Aquí tiene algunos comandos de ejemplo que puede usar.

#### destino armhf para Python 3.7

```sh
make -C tensorflow/lite/tools/pip_package docker-build \
  TENSORFLOW_TARGET=armhf PYTHON_VERSION=3.7
```

#### destino aarch64 para Python 3.8

```sh
make -C tensorflow/lite/tools/pip_package docker-build \
  TENSORFLOW_TARGET=aarch64 PYTHON_VERSION=3.8
```

#### ¿Cómo usar una cadena de herramientas personalizada?

Si los binarios generados no son compatibles con su destino, deberá usar su propia cadena de herramientas o utilizar indicadores de compilación personalizados. (Revise [esto](https://www.tensorflow.org/lite/guide/build_cmake_arm#check_your_target_environment) para entender el entorno de su destino) En ese caso, necesita modificar `tensorflow/lite/tools/cmake/download_toolchains.sh` para usar su propia cadena de herramientas. El script de la cadena de herramientas define las dos variables siguientes para el script `build_pip_package_with_cmake.sh`.

Variable | Propósito | ejemplo
--- | --- | ---
`ARMCC_PREFIX` | define el prefijo de la cadena de herramientas | arm-linux-gnueabihf-
`ARMCC_FLAGS` | Indicadores de compilación | -march=armv7-a -mfpu=neon-vfpv4

**Nota:** `ARMCC_FLAGS` puede necesitar contener la ruta de inclusión de la librería Python. Vea la `download_toolchains.sh` para obtener la referencia.
