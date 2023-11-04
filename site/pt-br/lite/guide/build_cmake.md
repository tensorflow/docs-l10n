# Compile o TensorFlow Lite com o CMake

Esta página descreve como compilar e usar a biblioteca do TensorFlow Lite com a ferramenta [CMake](https://cmake.org/).

As instruções abaixo foram testadas em um PC (AMD64) com Ubuntu 16.04.3 de 64 bits, macOS Catalina (x86_64), Windows 10 e imagem devel docker do TensorFlow [tensorflow/tensorflow:devel](https://hub.docker.com/r/tensorflow/tensorflow/tags/).

**Observação:** esse recurso está disponível a partir a versão 2.4.

### Etapa 1. Instale a ferramenta CMake

É necessário o CMake 3.16 ou superior. No Ubuntu, basta executar o seguinte comando:

```sh
sudo apt-get install cmake
```

Ou confira [o guia de instalação oficial do CMake](https://cmake.org/install/).

### Etapa 2. Clone o repositório do TensorFlow

```sh
git clone https://github.com/tensorflow/tensorflow.git tensorflow_src
```

**Observação:** se você estiver usando a imagem docker do TensorFlow, o repositório já está disponível em `/tensorflow_src/`.

### Etapa 3. Crie o diretório de builds do CMake

```sh
mkdir tflite_build
cd tflite_build
```

### Etapa 4. Execute a ferramenta CMake com configurações

#### Build de lançamento

É gerado um binário de lançamento otimizado por padrão. Se você deseja compilar para sua estação de trabalho, basta executar o seguinte comando:

```sh
cmake ../tensorflow_src/tensorflow/lite
```

#### Build de depuração

Se você precisar gerar uma build de depuração que tenha informações de símbolo, precisa fornecer a opção `-DCMAKE_BUILD_TYPE=Debug`.

```sh
cmake ../tensorflow_src/tensorflow/lite -DCMAKE_BUILD_TYPE=Debug
```

#### Compilação dos testes de unidade do kernel

Para poder executar os testes do kernel, você precisa fornecer o sinalizador `-DTFLITE_KERNEL_TEST=on`. As especificidades da compilação cruzada dos testes de unidade estão disponíveis na próxima subseção.

```sh
cmake ../tensorflow_src/tensorflow/lite -DTFLITE_KERNEL_TEST=on
```

#### Compile pacotes instaláveis

Para compilar um pacote instalável que poderá ser usado como dependência por outro projeto do CMake com `find_package(tensorflow-lite CONFIG)`, use a opção `-DTFLITE_ENABLE_INSTALL=ON`.

Idealmente, você deve fornecer suas próprias versões das dependências de bibliotecas, que também precisarão ser usadas pelo projeto que depende do TF Lite. Você pode usar a variável `-DCMAKE_FIND_PACKAGE_PREFER_CONFIG=ON` e definir que a variável `<PackageName>_DIR` aponte para as instalações das suas bibliotecas.

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

**Observação:** Consulte a documentação do CMake sobre [`find_package`](https://cmake.org/cmake/help/latest/command/find_package.html) para saber mais sobre como tratar e localizar pacotes.

#### Compilação cruzada

Você pode usar o CMake para compilar binários para arquiteturas ARM64 ou Android.

Para fazer a compilação cruzada do TF Lite, você precisa fornecer o caminho do SDK (por exemplo, ARM64 SDK ou NDK no caso do Android) com o sinalizador `-DCMAKE_TOOLCHAIN_FILE`.

```sh
cmake -DCMAKE_TOOLCHAIN_FILE=<CMakeToolchainFileLoc> ../tensorflow/lite/
```

##### Especificidades da compilação cruzada para Android

No caso de compilação cruzada para Android, você precisa instalar o [Android NDK](https://developer.android.com/ndk) e fornecer o caminho do NDK com o sinalizador `-DCMAKE_TOOLCHAIN_FILE` mencionado acima. Além disso, você precisa definir a ABI desejada com o sinalizador `-DANDROID_ABI`.

```sh
cmake -DCMAKE_TOOLCHAIN_FILE=<NDK path>/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a ../tensorflow_src/tensorflow/lite
```

##### Especificidades da compilação cruzada dos testes (de unidade) do kernel

A compilação cruzada dos testes de unidade requer o compilador flatc para a arquitetura do host. Para essa finalidade, existe uma CMakeLists localizada em `tensorflow/lite/tools/cmake/native_tools/flatbuffers` para compilar o compilador flatc com o CMake antecipadamente em um diretório de builds separado usando a toolchain do host.

```sh
mkdir flatc-native-build && cd flatc-native-build
cmake ../tensorflow_src/tensorflow/lite/tools/cmake/native_tools/flatbuffers
cmake --build .
```

Também é possível **instalar** o *flatc* em um local de instalação personalizado (por exemplo, um diretório contendo outras ferramentas compiladas nativamente em vez do diretório de builds do CMake):

```sh
cmake -DCMAKE_INSTALL_PREFIX=<native_tools_dir> ../tensorflow_src/tensorflow/lite/tools/cmake/native_tools/flatbuffers
cmake --build .
```

Para a compilação cruzada do TF Lite em si, o parâmetro adicional `-DTFLITE_HOST_TOOLS_DIR=<flatc_dir_path>` apontando para o diretório contendo o binário *flatc* nativo precisa ser fornecido, juntamente com o sinalizador `-DTFLITE_KERNEL_TEST=on` mencionado acima.

```sh
cmake -DCMAKE_TOOLCHAIN_FILE=${OE_CMAKE_TOOLCHAIN_FILE} -DTFLITE_KERNEL_TEST=on -DTFLITE_HOST_TOOLS_DIR=<flatc_dir_path> ../tensorflow/lite/
```

##### Execução dos testes (de unidade) do kernel com compilação cruzada no alvo

Os testes de unidade podem ser executados como executáveis separados ou usando o utilitário CTest. Quanto ao CTest, se pelo menos um dos parâmetros `TFLITE_ENABLE_NNAPI, TFLITE_ENABLE_XNNPACK` ou `TFLITE_EXTERNAL_DELEGATE` estiver ativado na build do TF Lite, os testes resultantes serão gerados com dois **labels** diferentes (utilizando o mesmo executável de teste); *plain* – que denota os testes executados no back-end de CPU; e *delegate* – que denota os testes que esperam argumentos de execução adicionais usados para a especificação do delegado.

Tanto `CTestTestfile.cmake` quanto `run-tests.cmake` (conforme indicado acima) estão disponíveis em `<build_dir>/kernels`.

Execução de testes de unidade com back-end de CPU (desde que `CTestTestfile.cmake` esteja presente no diretório atual do alvo):

```sh
ctest -L plain
```

Exemplos de execução de testes de unidade usando delegados (desde que `CTestTestfile.cmake` bem como `run-tests.cmake` estejam presentes no diretório atual do alvo):

```sh
cmake -E env TESTS_ARGUMENTS=--use_nnapi=true\;--nnapi_accelerator_name=vsi-npu ctest -L delegate
cmake -E env TESTS_ARGUMENTS=--use_xnnpack=true ctest -L delegate
cmake -E env TESTS_ARGUMENTS=--external_delegate_path=<PATH> ctest -L delegate
```

**Uma limitação conhecida** desta forma de fornecer argumentos de execução relacionados a delegados para testes de unidade é que só há suporte para testes em que **o valor de retorno esperado é 0**. Valores de retorno diferentes serão indicados como falha do teste.

#### Delegado de GPU do OpenCL

Se a sua máquina escolhida tiver suporte ao OpenCL, você pode usar a [delegado de GPU](https://www.tensorflow.org/lite/performance/gpu), que podem usar o poder das suas GPUs.

Para configurar o suporte a delegado de CPU do OpenCL:

```sh
cmake ../tensorflow_src/tensorflow/lite -DTFLITE_ENABLE_GPU=ON
```

**Observação:** isso é experimental e está disponível a partir do TensorFlow 2.5. Pode haver problemas de compatibilidade. Isso só é verificado em dispositivos Android e NVidia CUDA OpenCL 1.2.

### Etapa 5. Compile o TensorFlow Lite

No diretório `tflite_build`:

```sh
cmake --build . -j
```

**Observação:** é gerada uma biblioteca estática `libtensorflow-lite.a` no diretório atual, mas a biblioteca não é autocontida, já que todas as dependências transitivas não estão incluídas. Para usar a biblioteca corretamente, você precisa criar um projeto do CMake. Confira a seção [Crie um projeto do CMake que use o TensorFlow Lite](#create_a_cmake_project_which_uses_tensorflow_lite).

### Etapa 6. Compile a ferramenta de benchmark do TensorFlow Lite e o exemplo de imagem de rótulos (opcional)

No diretório `tflite_build`:

```sh
cmake --build . -j -t benchmark_model
```

```sh
cmake --build . -j -t label_image
```

## Opções disponíveis para compilar o TensorFlow Lite

Veja a lista de opções disponíveis. Você pode sobrescrever com `-D<option_name>=[ON|OFF]`. Por exemplo, use `-DTFLITE_ENABLE_XNNPACK=OFF` para desativar o XNNPACK, que é ativado por padrão.

Nome da opção | Recurso | Android | Linux | macOS | Windows
--- | --- | --- | --- | --- | ---
`TFLITE_ENABLE_RUY` | Ativa o RUY | ON | OFF | OFF | OFF
:                         : matriz         :         :       :       :         : |  |  |  |  |
:                         : multiplicação :         :       :       :         : |  |  |  |  |
:                         : biblioteca        :         :       :       :         : |  |  |  |  |
`TFLITE_ENABLE_NNAPI` | Ativa o NNAPI | ON | OFF | N.D. | N.D.
:                         : delegado       :         :       :       :         : |  |  |  |  |
`TFLITE_ENABLE_GPU` | Ativa a GPU | OFF | OFF | N.D. | N.D.
:                         : delegado       :         :       :       :         : |  |  |  |  |
`TFLITE_ENABLE_XNNPACK` | Ativa o XNNPACK | ON | ON | ON | ON
:                         : delegado       :         :       :       :         : |  |  |  |  |
`TFLITE_ENABLE_MMAP` | Ativa o MMAP | ON | ON | ON | N.D.

## Crie um projeto do CMake que use o TensorFlow Lite

Aqui está o arquivo CMakeLists.txt do [exemplo mínimo do TF Lite](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/examples/minimal).

Você precisa adicionar add_subdirectory() para o diretório do TensorFlow Lite e vincular `tensorflow-lite` com target_link_libraries().

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

## Compile a biblioteca C do TensorFlow Lite

Se você quiser compilar a biblioteca compartilhada do TensorFlow Lite para a [API do C](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/c/README.md), primeiro siga a [etapa 1](#step-1-install-cmake-tool) até a [etapa 3](#step-3-create-cmake-build-directory). Em seguida, execute os comandos abaixo:

```sh
cmake ../tensorflow_src/tensorflow/lite/c
cmake --build . -j
```

Esses comandos geram a seguinte biblioteca compartilhada no diretório atual.

**Observaçao:** em sistemas Windows, você encontra `tensorflowlite_c.dll` no diretório `debug`.

Plataforma | Nome da biblioteca
--- | ---
Linux | `libtensorflowlite_c.so`
macOS | `libtensorflowlite_c.dylib`
Windows | `tensorflowlite_c.dll`

**Observação:** você precisa dos cabeçalhos públicos (`tensorflow/lite/c_api.h`, `tensorflow/lite/c_api_experimental.h`, `tensorflow/lite/c_api_types.h` e `tensorflow/lite/common.h`) e dos cabeçalhos privados que esses cabeçalhos públicos incluem (`tensorflow/lite/core/builtin_ops.h`, `tensorflow/lite/core/c/*.h` e `tensorflow/lite/core/async/c/*.h`) para usar a biblioteca compartilhada gerada.
