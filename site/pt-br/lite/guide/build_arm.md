# Compile o TensorFlow Lite para placas ARM

Esta página descreve como compilar as bibliotecas do TensorFlow Lite para computadores ARM.

O TensorFlow Lite tem suporte a dois sistemas de build, mas os recursos de cada sistema de compilação não são idênticos. Confira a tabela abaixo para escolher o sistema de compilação adequado.

Recurso | Bazel | CMake
--- | --- | ---
Toolchains predefinidas | armhf, aarch64 | armel, armhf, aarch64
Toolchains personalizadas | Mais difícil de usar | Fácil de usar
[Select TF ops](https://www.tensorflow.org/lite/guide/ops_select) | Com suporte | Sem suporte
[Delegado de GPU](https://www.tensorflow.org/lite/performance/gpu) | Disponível somente para Android | Qualquer plataforma com suporte ao OpenCL
XNNPack | Com suporte | Com suporte
[Wheel do Python](https://www.tensorflow.org/lite/guide/build_cmake_pip) | Com suporte | Com suporte
[API do C](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/c/README.md) | Com suporte | [Com suporte](https://www.tensorflow.org/lite/guide/build_cmake#build_tensorflow_lite_c_library)
[API do C++](https://www.tensorflow.org/lite/guide/inference#load_and_run_a_model_in_c) | Com suporte para projetos Bazel | Com suporte para projetos CMake

## Compilação cruzada para ARM com CMake

Se você tiver um projeto CMake ou se quiser usar uma toolchain personalizada, é melhor usar o CMake para fazer a compilação cruzada. Existe uma página [Compilação cruzada do TensorFlow Lite com o CMake](https://www.tensorflow.org/lite/guide/build_cmake_arm) específica para isso.

## Compilação cruzada para ARM com Bazel

Se você tiver um projeto Bazel ou se quiser usar operações do TF, é melhor usar o sistema de build Bazel. Você usará as [toolchains ARM GCC 8.3](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/toolchains/embedded/arm-linux) integradas com o Bazel para compilar uma biblioteca ARM32/64 compartilhada.

Arquitetura desejada | Configuração do Bazel | Dispositivos compatíveis
--- | --- | ---
armhf (ARM32) | --config=elinux_armhf | RPI3, RPI4 com 32 bits
:                     :                         : Raspberry Pi OS            : |  |
AArch64 (ARM64) | --config=elinux_aarch64 | Coral, RPI4 com Ubuntu 64
:                     :                         : bits                        : |  |

Observação: a biblioteca compartilhada gerada requer o glibc 2.28 ou superior para ser executada.

As instruções abaixo foram testadas em um PC (AMD64) com Ubuntu 16.04.3 de 64 bits e na imagem devel docker do TensorFlow [tensorflow/tensorflow:devel](https://hub.docker.com/r/tensorflow/tensorflow/tags/).

Para fazer a compilação cruzada do TensorFlow Lite com o Bazel, siga estas etapas:

#### Etapa 1. Instale o Bazel

O Bazel é o principal sistema de compilação para o TensorFlow. Instale a versão mais recente do [sistema de compilação Bazel](https://bazel.build/versions/master/docs/install.html).

**Observação:** se você estiver usando a imagem docker do TensorFlow, o Bazel já está disponível.

#### Etapa 2. Clone o repositório do TensorFlow

```sh
git clone https://github.com/tensorflow/tensorflow.git tensorflow_src
```

**Observação:** se você estiver usando a imagem docker do TensorFlow, o repositório já está disponível em `/tensorflow_src/`.

#### Etapa 3. Compile o binário ARM

##### Biblioteca do C

```bash
bazel build --config=elinux_aarch64 -c opt //tensorflow/lite/c:libtensorflowlite_c.so
```

A biblioteca compartilhada está disponível em: `bazel-bin/tensorflow/lite/c/libtensorflowlite_c.so`.

**Observação:** use `elinux_armhf` para uma compilação [ARM hard-float de 32 bits](https://wiki.debian.org/ArmHardFloatPort).

Confira mais detalhes na página da [API do C para TensorFlow Lite](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/c/README.md).

##### Biblioteca do C++

```bash
bazel build --config=elinux_aarch64 -c opt //tensorflow/lite:libtensorflowlite.so
```

A biblioteca compartilhada está disponível em: `bazel-bin/tensorflow/lite/libtensorflowlite.so`.

No momento, não há uma maneira simples de extrair todos os arquivos de cabeçalho necessários, então você precisa incluir todos os arquivos em tensorflow/lite/ a partir do repositório do TensorFlow. Além disso, você precisará dos arquivos de cabeçalho de FlatBuffers e Abseil.

##### Etc.

Você também pode compilar outros alvos do Bazel com a toolchain. Veja alguns alvos úteis:

- //tensorflow/lite/tools/benchmark:benchmark_model
- //tensorflow/lite/examples/label_image:label_image
