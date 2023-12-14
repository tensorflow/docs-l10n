# Generar TensorFlow Lite para placas ARM

Esta página describe cómo construir las librerías TensorFlow Lite para computadoras basadas en ARM.

TensorFlow Lite admite dos sistemas de compilación y las características soportadas de cada sistema de compilación no son idénticas. Revise la siguiente tabla para elegir un sistema de compilación adecuado.

Característica | Bazel | CMake
--- | --- | ---
Cadenas de herramientas predefinidas | armhf, aarch64 | armel, armhf, aarch64
Cadenas de herramientas personalizadas | más difícil de usar | fácil de usar
[Select TF ops](https://www.tensorflow.org/lite/guide/ops_select) | compatible | no compatible
[Delegado de GPU](https://www.tensorflow.org/lite/performance/gpu) | sólo disponible para Android | cualquier plataforma compatible con OpenCL
XNNPack | compatible | compatible
[Python Wheel](https://www.tensorflow.org/lite/guide/build_cmake_pip) | compatible | compatible
[API C](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/c/README.md) | compatible | [compatible](https://www.tensorflow.org/lite/guide/build_cmake#build_tensorflow_lite_c_library)
[API C++](https://www.tensorflow.org/lite/guide/inference#load_and_run_a_model_in_c) | compatible para projectos Bazel | compatible para proyectos CMake

## Compilación cruzada para ARM con CMake

Si tiene un proyecto CMake o si desea usar una cadena de herramientas personalizada, es mejor que use CMake para la compilación cruzada. Hay una página aparte [Compilación cruzada de TensorFlow Lite con CMake](https://www.tensorflow.org/lite/guide/build_cmake_arm) disponible para este tema.

## Compilación cruzada para ARM con Bazel

Si tiene un proyecto Bazel o si quiere usar TF ops, será mejor que use el sistema de generación Bazel. Usará las cadenas de herramientas [ARM GCC 8.3 integradas](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/toolchains/embedded/arm-linux) con Bazel para generar una librería compartida ARM32/64.

Arquitectura objetivo | Configuración de Bazel | Dispositivos compatibles
--- | --- | ---
armhf (ARM32) | --config=elinux_armhf | RPI3, RPI4 con 32 bit
:                     :                         : Raspberry Pi OS            : |  |
AArch64 (ARM64) | --config=elinux_aarch64 | Coral, RPI4 con Ubuntu 64
:                     :                         : bit                        : |  |

Nota: La librería compartida generada requiere glibc 2.28 o superior para funcionar.

Las siguientes instrucciones han sido analizadas en Ubuntu 16.04.3 64-bit PC (AMD64) y la imagen docker devel TensorFlow devel[tensorflow/tensorflow:devel](https://hub.docker.com/r/tensorflow/tensorflow/tags/).

Para realizar una compilación cruzada de TensorFlow Lite con Bazel, siga los siguientes pasos:

#### Paso 1. Instale Bazel

Bazel es el principal sistema de compilación para TensorFlow. Instale la última versión del sistema de compilación [Bazel](https://bazel.build/versions/master/docs/install.html).

**Nota:** Si está usando la imagen Docker de TensorFlow, Bazel ya está disponible.

#### Paso 2. Clone el repositorio de TensorFlow

```sh
git clone https://github.com/tensorflow/tensorflow.git tensorflow_src
```

**Nota:** Si está usando la imagen Docker de TensorFlow, el repositorio ya está disponible en `/tensorflow_src/`.

#### Paso 3. Genere el binario ARM

##### Librería en C

```bash
bazel build --config=elinux_aarch64 -c opt //tensorflow/lite/c:libtensorflowlite_c.so
```

Puede encontrar una librería compartida en: `bazel-bin/tensorflow/lite/c/libtensorflowlite_c.so`.

**Nota:** Use `elinux_armhf` para la compilación [32bit ARM hard float](https://wiki.debian.org/ArmHardFloatPort).

Consulte la página [TensorFlow Lite C API](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/c/README.md) para obtener más detalles.

##### Librería en C++

```bash
bazel build --config=elinux_aarch64 -c opt //tensorflow/lite:libtensorflowlite.so
```

Puede encontrar una librería compartida en: `bazel-bin/tensorflow/lite/libtensorflowlite.so`.

Actualmente, no existe una forma directa de extraer todos los archivos de cabecera necesarios, por lo que deberá incluir todos los archivos de cabecera en tensorflow/lite/ del repositorio de TensorFlow. Además, necesitará los archivos de cabecera de FlatBuffers y Abseil.

##### Etc

También puede generar otros destinos Bazel con la cadena de herramientas. Aquí tiene algunos destinos útiles.

- //tensorflow/lite/tools/benchmark:benchmark_model
- //tensorflow/lite/examples/label_image:label_image
