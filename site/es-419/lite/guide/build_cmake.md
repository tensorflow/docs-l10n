# Generar TensorFlow Lite con CMake

Esta página describe cómo generar y usar la librería TensorFlow Lite con la herramienta [CMake](https://cmake.org/).

Las siguientes instrucciones han sido analizadas en Ubuntu 16.04.3 64-bit PC (AMD64) macOS Catalina (x86_64), Windows 10, y la imagen docker devel TensorFlow devel[tensorflow/tensorflow:devel](https://hub.docker.com/r/tensorflow/tensorflow/tags/).

**Nota:** Esta función está disponible desde la versión 2.4.

### Paso 1. Instale la herramienta CMake

Se requiere CMake 3.16 o superior. En Ubuntu, puede simplemente ejecutar el siguiente comando.

```sh
sudo apt-get install cmake
```

O puede seguir [la guía oficial de instalación de cmake](https://cmake.org/install/)

### Paso 2. Clone el repositorio de TensorFlow

```sh
git clone https://github.com/tensorflow/tensorflow.git tensorflow_src
```

**Nota:** Si está usando la imagen Docker de TensorFlow, el repositorio ya está disponible en `/tensorflow_src/`.

### Paso 3. Cree el directorio de generación CMake

```sh
mkdir tflite_build
cd tflite_build
```

### Paso 4. Ejecute la herramienta CMake con las configuraciones

#### Liberar compilación

Genera un binario de liberación optimizado de forma predeterminada. Si desea generar para su estación de trabajo, simplemente ejecute el siguiente comando.

```sh
cmake ../tensorflow_src/tensorflow/lite
```

#### Depurar la compilación

Si necesita producir una compilación de depuración que tenga información de símbolos, deberá proporcionar la opción `-DCMAKE_BUILD_TYPE=Debug`.

```sh
cmake ../tensorflow_src/tensorflow/lite -DCMAKE_BUILD_TYPE=Debug
```

#### Generar con las pruebas de unidad del kernel

Para poder ejecutar las pruebas del kernel, debe facilitar el indicador `-DTFLITE_KERNEL_TEST=on`. Los detalles de la compilación cruzada de las pruebas de unidad se pueden encontrar en la siguiente subsección.

```sh
cmake ../tensorflow_src/tensorflow/lite -DTFLITE_KERNEL_TEST=on
```

#### Generar paquete instalable

Para generar un paquete instalable que pueda ser usado como dependencia por otro proyecto CMake con `find_package(tensorflow-lite CONFIG)`, use la opción `-DTFLITE_ENABLE_INSTALL=ON`.

Lo ideal es que también aporte sus propias versiones de las dependencias de librerías. Éstas también deberán ser usadas por el proyecto que dependa de TF Lite. Puede usar la opción `-DCMAKE_FIND_PACKAGE_PREFER_CONFIG=ON` y fijar las variables `<PackageName>_DIR` para que apunten a sus instalaciones de bibliotecas.

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

**Nota:** Consulte la documentación de CMake para [`find_package`](https://cmake.org/cmake/help/latest/command/find_package.html) para saber más sobre el manejo y localización de paquetes.

#### Compilación cruzada

Puede usar CMake para generar binarios para arquitecturas de destino ARM64 o Android.

Para realizar la compilación cruzada de TF Lite, es necesario indicar la ruta al SDK (por ejemplo, ARM64 SDK o NDK en el caso de Android) con el indicador `-DCMAKE_TOOLCHAIN_FILE`.

```sh
cmake -DCMAKE_TOOLCHAIN_FILE=<CMakeToolchainFileLoc> ../tensorflow/lite/
```

##### Particularidades de la compilación cruzada para Android

Para la compilación cruzada de Android, necesita instalar [Android NDK](https://developer.android.com/ndk) y dar la ruta del NDK con el indicador `-DCMAKE_TOOLCHAIN_FILE` mencionado anteriormente. También es necesario ajustar el ABI de destino con el indicador `-DANDROID_ABI`.

```sh
cmake -DCMAKE_TOOLCHAIN_FILE=<NDK path>/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a ../tensorflow_src/tensorflow/lite
```

##### Particularidades de la compilación cruzada de las pruebas de unidad (kernel)

La compilación cruzada de las pruebas de unidad requiere el compilador flatc para la arquitectura anfitriona. Para ello, hay un CMakeLists ubicado en `tensorflow/lite/tools/cmake/native_tools/flatbuffers` para generar el compilador flatc con CMake por adelantado en un directorio de compilación separado usando la cadena de herramientas del host.

```sh
mkdir flatc-native-build && cd flatc-native-build
cmake ../tensorflow_src/tensorflow/lite/tools/cmake/native_tools/flatbuffers
cmake --build .
```

También es posible **instalar** el *flatc* en una ubicación de instalación personalizada (por ejemplo, en un directorio que contenga otras herramientas creadas de forma nativa en lugar del directorio de creación de CMake):

```sh
cmake -DCMAKE_INSTALL_PREFIX=<native_tools_dir> ../tensorflow_src/tensorflow/lite/tools/cmake/native_tools/flatbuffers
cmake --build .
```

Para la propia compilación cruzada de TF Lite, es necesario indicar el parámetro adicional `-DTFLITE_HOST_TOOLS_DIR=<flatc_dir_path>` que apunte al directorio que contenga el binario nativo *flatc* junto con el Indicador `-DTFLITE_KERNEL_TEST=on` mencionado anteriormente.

```sh
cmake -DCMAKE_TOOLCHAIN_FILE=${OE_CMAKE_TOOLCHAIN_FILE} -DTFLITE_KERNEL_TEST=on -DTFLITE_HOST_TOOLS_DIR=<flatc_dir_path> ../tensorflow/lite/
```

##### Lanzamiento de pruebas de kernel (unidad) de compilación cruzada en destino

Las pruebas de unidad pueden ejecutarse como ejecutables independientes o usando la utilidad CTest. En lo que respecta a CTest, si al menos uno de los parámetros `TFLITE_ENABLE_NNAPI, TFLITE_ENABLE_XNNPACK` o `TFLITE_EXTERNAL_DELEGATE` está habilitado para la compilación de TF Lite, las pruebas resultantes se generan con dos **etiquetas** diferentes (utilizando el mismo ejecutable de prueba): - *llano* - que denota las pruebas que se ejecutan en el backend de la CPU - *delegado* - que denota las pruebas que esperan argumentos de lanzamiento adicionales usados para la especificación del delegado usado

Tanto `CTestTestfile.cmake` como `run-tests.cmake` (como se refiere a continuación) están disponibles en `<build_dir>/kernels`.

Lanzamiento de pruebas de unidad con backend de CPU (siempre que el archivo `CTestTestfile.cmake` esté presente en el destino en el directorio actual):

```sh
ctest -L plain
```

Inicie ejemplos de pruebas de unidad usando delegados (siempre que el archivo `CTestTestfile.cmake` así como `run-tests.cmake` estén presentes en el destino en el directorio actual):

```sh
cmake -E env TESTS_ARGUMENTS=--use_nnapi=true\;--nnapi_accelerator_name=vsi-npu ctest -L delegate
cmake -E env TESTS_ARGUMENTS=--use_xnnpack=true ctest -L delegate
cmake -E env TESTS_ARGUMENTS=--external_delegate_path=<PATH> ctest -L delegate
```

**Una limitación conocida** de esta forma de aportar argumentos de lanzamiento adicionales relacionados con los delegados a las pruebas de unidad es que, efectivamente, sólo admite aquellos con un **valor de retorno esperado de 0**. Los valores de retorno diferentes se notificarán como un fallo de la prueba.

#### Delegado de GPU OpenCL

Si su máquina de destino tiene soporte OpenCL, puede usar un [delegado GPU](https://www.tensorflow.org/lite/performance/gpu) que puede aprovechar la potencia de su GPU.

Para configurar el soporte de delegados GPU OpenCL:

```sh
cmake ../tensorflow_src/tensorflow/lite -DTFLITE_ENABLE_GPU=ON
```

**Nota:** Es experimental y está disponible a partir de TensorFlow 2.5. Podría haber problemas de compatibilidad. Sólo se ha verificado con dispositivos Android y NVidia CUDA OpenCL 1.2.

### Paso 5. Genere TensorFlow Lite

En el directorio `tflite_build`,

```sh
cmake --build . -j
```

**Nota:** Esto genera una librería estática `libtensorflow-lite.a` en el directorio actual pero la librería no es autocontenida ya que no se incluyen todas las dependencias transitivas. Para usar la librería correctamente, necesita crear un proyecto CMake. Consulte la sección ["Crear un proyecto CMake que use TensorFlow Lite"](#create_a_cmake_project_which_uses_tensorflow_lite).

### Paso 6. Genere la herramienta TensorFlow Lite Benchmark y etiquete el ejemplo de imagen (opcional)

En el directorio `tflite_build`,

```sh
cmake --build . -j -t benchmark_model
```

```sh
cmake --build . -j -t label_image
```

## Opciones disponibles para generar TensorFlow Lite

Esta es la lista de opciones disponibles. Puede anularla con `-D<option_name>=[ON|OFF]`. Por ejemplo, `-DTFLITE_ENABLE_XNNPACK=OFF` para desactivar XNNPACK que está activado de forma predeterminada.

Nombre de opción | Característica | Android | Linux | macOS | Windows
--- | --- | --- | --- | --- | ---
`TFLITE_ENABLE_RUY` | Habilitar RUY | ON | OFF | OFF | OFF
:                         : matriz         :         :       :       :         : |  |  |  |  |
:                         : multiplicación :         :       :       :         : |  |  |  |  |
:                         : librería        :         :       :       :         : |  |  |  |  |
`TFLITE_ENABLE_NNAPI` | Habilitar NNAPI | ON | OFF | N/A | N/A
:                         : delegado       :         :       :       :         : |  |  |  |  |
`TFLITE_ENABLE_GPU` | Habilitar GPU | OFF | OFF | N/A | N/A
:                         : delegado       :         :       :       :         : |  |  |  |  |
`TFLITE_ENABLE_XNNPACK` | Habilitar XNNPACK | ON | ON | ON | ON
:                         : delegado       :         :       :       :         : |  |  |  |  |
`TFLITE_ENABLE_MMAP` | Habilitar MMAP | ON | ON | ON | N/A

## Crear un proyecto CMake que use TensorFlow Lite

Aquí está el CMakeLists.txt de ejemplo mínimo de TFLite.

Es necesario tener add_subdirectory() para el directorio de TensorFlow Lite y enlazar `tensorflow-lite` con target_link_libraries().

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

## Generar librería TensorFlow Lite en C

Si desea construir la librería compartida de TensorFlow Lite para la [API en C](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/c/README.md), siga primero [el paso 1](#step-1-install-cmake-tool) al [el paso 3](#step-3-create-cmake-build-directory). Después, ejecute los siguientes comandos.

```sh
cmake ../tensorflow_src/tensorflow/lite/c
cmake --build . -j
```

Este comando genera la siguiente librería compartida en el directorio actual.

**Nota:** En el sistema Windows, puede encontrar el `tensorflowlite_c.dll` en el directorio `debug`.

Plataforma | Nombre de la librería
--- | ---
Linux | `libtensorflowlite_c.so`
macOS | `libtensorflowlite_c.dylib`
Windows | `tensorflowlite_c.dll`

**Nota:** Necesita las cabeceras públicas (`tensorflow/lite/c_api.h`, `tensorflow/lite/c_api_experimental.h`, `tensorflow/lite/c_api_types.h`, y `tensorflow/lite/common. h`), y las cabeceras privadas que dichas cabeceras públicas incluyen (`tensorflow/lite/core/builtin_ops.h`, `tensorflow/lite/core/c/*.h`, y `tensorflow/lite/core/async/c/*.h`), para usar la librería compartida generada.
