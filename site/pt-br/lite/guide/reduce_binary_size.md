# Reduza o tamanho do binário do TensorFlow Lite

## Visão geral

Ao implantar modelos para aplicativos de aprendizado de máquina no dispositivo (ODML, na sigla em inglês), é importante saber que a memória disponível em dispositivos móveis é limitada. O tamanho do binário dos modelos está intimamente relacionado ao número de operações usadas no modelo. O TensorFlow Lite permite reduzir o tamanho do binário do modelo por meio do uso de builds seletivas, que ignoram operações não usadas em seu conjunto de modelos e geram uma biblioteca compacta, apenas com o runtime e os kernels das operações necessárias para que o modelo seja executado em seu dispositivo móvel.

As builds seletivas se aplicam às três bibliotecas de operações abaixo:

1. [Operações integradas do TensorFlow Lite](https://www.tensorflow.org/lite/guide/ops_compatibility)
2. [Operações personalizadas do TensorFlow Lite](https://www.tensorflow.org/lite/guide/ops_custom)
3. [Operações específicas do TensorFlow](https://www.tensorflow.org/lite/guide/ops_select)

A tabela abaixo indica o impacto de builds seletivas para alguns casos de uso comuns:

<table>
  <thead>
    <tr>
      <th>Nome do modelo</th>
      <th>Domínio</th>
      <th>Arquitetura alvo</th>
      <th>Tamanho do arquivo AAR</th>
    </tr>
  </thead>
  <tr>
    <td rowspan="2">       <a href="https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224.tgz">Mobilenet_1.0_224(float)</a> </td>
    <td rowspan="2">Classificação de imagens</td>
    <td>armeabi-v7a</td>
    <td>tensorflow-lite.aar (296.635 bytes)</td>
  </tr>
   <tr>
    <td>arm64-v8a</td>
    <td>tensorflow-lite.aar (382.892 bytes)</td>
  </tr>
  <tr>
    <td rowspan="2">       <a href="https://tfhub.dev/google/lite-model/spice/">SPICE</a> </td>
    <td rowspan="2">Extração de tons sonoros</td>
    <td>armeabi-v7a</td>
    <td>tensorflow-lite.aar (375.813 bytes)<br>tensorflow-lite-select-tf-ops.aar (1.676.380 bytes)</td>
  </tr>
   <tr>
    <td>arm64-v8a</td>
    <td>tensorflow-lite.aar (421.826 bytes)<br>tensorflow-lite-select-tf-ops.aar (2.298.630 bytes)</td>
  </tr>
  <tr>
    <td rowspan="2">       <a href="https://tfhub.dev/deepmind/i3d-kinetics-400/1">i3d-kinetics-400</a> </td>
    <td rowspan="2">Classificação de vídeos</td>
    <td>armeabi-v7a</td>
    <td>tensorflow-lite.aar (240.085 bytes)<br>tensorflow-lite-select-tf-ops.aar (1.708.597 bytes)</td>
  </tr>
   <tr>
    <td>arm64-v8a</td>
    <td>tensorflow-lite.aar (273.713 bytes)<br>tensorflow-lite-select-tf-ops.aar (2.339.697 bytes)</td>
  </tr>
 </table>

Observação: este recurso está em fase experimental, disponível desde a versão 2.4, e sujeito a alterações.

## Compile o TensorFlow Lite seletivamente com o Bazel

Esta seção pressupõe que você tenha baixado os códigos-fonte do TensorFlow e [configurado o ambiente de desenvolvimento local](https://www.tensorflow.org/lite/android/lite_build#set_up_build_environment_without_docker) no Bazel.

### Compile arquivos AAR para projetos do Android

Para compilar os AARs personalizados do TensorFlow Lite, basta fornecer os caminhos de arquivo abaixo ao seu modelo.

```sh
sh tensorflow/lite/tools/build_aar.sh \
  --input_models=/a/b/model_one.tflite,/c/d/model_two.tflite \
  --target_archs=x86,x86_64,arm64-v8a,armeabi-v7a
```

O comando acima vai gerar o arquivo AAR `bazel-bin/tmp/tensorflow-lite.aar` para as operações personalizadas e integradas do TensorFlow. Opcionalmente, vai gerar o arquivo AAR `bazel-bin/tmp/tensorflow-lite-select-tf-ops.aar` se os seus modelos tiverem operações específicas do TensorFlow. Observe que é gerado um AAR "fat", com diversas arquiteturas diferentes. Se você não precisar de todas elas, use um subconjunto apropriado para seu ambiente de desenvolvimento.

### Compile com operações personalizadas

Se você tiver implantado modelos do Tensorflow Lite com operações personalizadas, pode compilá-los adicionando os sinalizadores abaixo ao comando build:

```sh
sh tensorflow/lite/tools/build_aar.sh \
  --input_models=/a/b/model_one.tflite,/c/d/model_two.tflite \
  --target_archs=x86,x86_64,arm64-v8a,armeabi-v7a \
  --tflite_custom_ops_srcs=/e/f/file1.cc,/g/h/file2.h \
  --tflite_custom_ops_deps=dep1,dep2
```

O sinalizador `tflite_custom_ops_srcs` contém os arquivos fonte das operações personalizadas, e o sinalizador `tflite_custom_ops_deps` contém as dependências para compilar esses arquivos fonte. Atenção: essas dependências precisam existir no repositório do TensorFlow.

### Usos avançados – Regras personalizadas do Bazel

Se o seu projeto estiver usando o Bazel e você quiser definir dependências personalizadas do TF Lite para um determinado conjunto de modelos, pode definir as regras abaixo no repositório do projeto:

Para modelos com apenas operações integradas:

```bazel
load(
    "@org_tensorflow//tensorflow/lite:build_def.bzl",
    "tflite_custom_android_library",
    "tflite_custom_c_library",
    "tflite_custom_cc_library",
)

# A selectively built TFLite Android library.
tflite_custom_android_library(
    name = "selectively_built_android_lib",
    models = [
        ":model_one.tflite",
        ":model_two.tflite",
    ],
)

# A selectively built TFLite C library.
tflite_custom_c_library(
    name = "selectively_built_c_lib",
    models = [
        ":model_one.tflite",
        ":model_two.tflite",
    ],
)

# A selectively built TFLite C++ library.
tflite_custom_cc_library(
    name = "selectively_built_cc_lib",
    models = [
        ":model_one.tflite",
        ":model_two.tflite",
    ],
)
```

Para modelos com [operações específicas do TF](../guide/ops_select.md):

```bazel
load(
    "@org_tensorflow//tensorflow/lite/delegates/flex:build_def.bzl",
    "tflite_flex_android_library",
    "tflite_flex_cc_library",
)

# A Select TF ops enabled selectively built TFLite Android library.
tflite_flex_android_library(
    name = "selective_built_tflite_flex_android_lib",
    models = [
        ":model_one.tflite",
        ":model_two.tflite",
    ],
)

# A Select TF ops enabled selectively built TFLite C++ library.
tflite_flex_cc_library(
    name = "selective_built_tflite_flex_cc_lib",
    models = [
        ":model_one.tflite",
        ":model_two.tflite",
    ],
)
```

### Usos avançados – Compile bibliotecas compartilhadas do C/C++ personalizadas

Se você quiser compilar seus próprios objetos compartilhados do C/C++ para TF Lite personalizados referentes aos modelos determinados, pode seguir as etapas abaixo:

Crie um arquivo BUILD temporário executando o comando abaixo no diretório raiz do código-fonte do TensorFlow:

```sh
mkdir -p tmp && touch tmp/BUILD
```

#### Compilando objetos compartilhados do C personalizados

Se você quiser compilar um objeto compartilhado do C para TF Lite personalizado, adicione o seguinte ao arquivo `tmp/BUILD`:

```bazel
load(
    "//tensorflow/lite:build_def.bzl",
    "tflite_custom_c_library",
    "tflite_cc_shared_object",
)

tflite_custom_c_library(
    name = "selectively_built_c_lib",
    models = [
        ":model_one.tflite",
        ":model_two.tflite",
    ],
)

# Generates a platform-specific shared library containing the TensorFlow Lite C
# API implementation as define in `c_api.h`. The exact output library name
# is platform dependent:
#   - Linux/Android: `libtensorflowlite_c.so`
#   - Mac: `libtensorflowlite_c.dylib`
#   - Windows: `tensorflowlite_c.dll`
tflite_cc_shared_object(
    name = "tensorflowlite_c",
    linkopts = select({
        "//tensorflow:ios": [
            "-Wl,-exported_symbols_list,$(location //tensorflow/lite/c:exported_symbols.lds)",
        ],
        "//tensorflow:macos": [
            "-Wl,-exported_symbols_list,$(location //tensorflow/lite/c:exported_symbols.lds)",
        ],
        "//tensorflow:windows": [],
        "//conditions:default": [
            "-z defs",
            "-Wl,--version-script,$(location //tensorflow/lite/c:version_script.lds)",
        ],
    }),
    per_os_targets = True,
    deps = [
        ":selectively_built_c_lib",
        "//tensorflow/lite/c:exported_symbols.lds",
        "//tensorflow/lite/c:version_script.lds",
    ],
)
```

O alvo recém-adicionado pode ser compilado da seguinte forma:

```sh
bazel build -c opt --cxxopt=--std=c++17 \
  //tmp:tensorflowlite_c
```

Para Android, substitua `android_arm` por `android_arm64` para 64 bits:

```sh
bazel build -c opt --cxxopt=--std=c++17 --config=android_arm \
  //tmp:tensorflowlite_c
```

#### Compilando objetos compartilhados do C++ personalizados

Se você quiser compilar um objeto compartilhado do C++ para TF Lite personalizado, adicione o seguinte ao arquivo `tmp/BUILD`:

```bazel
load(
    "//tensorflow/lite:build_def.bzl",
    "tflite_custom_cc_library",
    "tflite_cc_shared_object",
)

tflite_custom_cc_library(
    name = "selectively_built_cc_lib",
    models = [
        ":model_one.tflite",
        ":model_two.tflite",
    ],
)

# Shared lib target for convenience, pulls in the core runtime and builtin ops.
# Note: This target is not yet finalized, and the exact set of exported (C/C++)
# APIs is subject to change. The output library name is platform dependent:
#   - Linux/Android: `libtensorflowlite.so`
#   - Mac: `libtensorflowlite.dylib`
#   - Windows: `tensorflowlite.dll`
tflite_cc_shared_object(
    name = "tensorflowlite",
    # Until we have more granular symbol export for the C++ API on Windows,
    # export all symbols.
    features = ["windows_export_all_symbols"],
    linkopts = select({
        "//tensorflow:macos": [
            "-Wl,-exported_symbols_list,$(location //tensorflow/lite:tflite_exported_symbols.lds)",
        ],
        "//tensorflow:windows": [],
        "//conditions:default": [
            "-Wl,-z,defs",
            "-Wl,--version-script,$(location //tensorflow/lite:tflite_version_script.lds)",
        ],
    }),
    per_os_targets = True,
    deps = [
        ":selectively_built_cc_lib",
        "//tensorflow/lite:tflite_exported_symbols.lds",
        "//tensorflow/lite:tflite_version_script.lds",
    ],
)
```

O alvo recém-adicionado pode ser compilado da seguinte forma:

```sh
bazel build -c opt  --cxxopt=--std=c++17 \
  //tmp:tensorflowlite
```

Para Android, substitua `android_arm` por `android_arm64` para 64 bits:

```sh
bazel build -c opt --cxxopt=--std=c++17 --config=android_arm \
  //tmp:tensorflowlite
```

Para modelos com operações específicas do TF, você também precisa compartilhar a seguinte biblioteca compartilhada:

```bazel
load(
    "@org_tensorflow//tensorflow/lite/delegates/flex:build_def.bzl",
    "tflite_flex_shared_library"
)

# Shared lib target for convenience, pulls in the standard set of TensorFlow
# ops and kernels. The output library name is platform dependent:
#   - Linux/Android: `libtensorflowlite_flex.so`
#   - Mac: `libtensorflowlite_flex.dylib`
#   - Windows: `libtensorflowlite_flex.dll`
tflite_flex_shared_library(
  name = "tensorflowlite_flex",
  models = [
      ":model_one.tflite",
      ":model_two.tflite",
  ],
)

```

O alvo recém-adicionado pode ser compilado da seguinte forma:

```sh
bazel build -c opt --cxxopt='--std=c++17' \
      --config=monolithic \
      --host_crosstool_top=@bazel_tools//tools/cpp:toolchain \
      //tmp:tensorflowlite_flex
```

Para Android, substitua `android_arm` por `android_arm64` para 64 bits:

```sh
bazel build -c opt --cxxopt='--std=c++17' \
      --config=android_arm \
      --config=monolithic \
      --host_crosstool_top=@bazel_tools//tools/cpp:toolchain \
      //tmp:tensorflowlite_flex
```

## Compile o TensorFlow Lite seletivamente com o Docker

Esta seção pressupõe que você tenha instalado o [Docker](https://docs.docker.com/get-docker/) em sua máquina local e baixado o Dockerfile do TensorFlow Lite [aqui](https://www.tensorflow.org/lite/android/lite_build#set_up_build_environment_using_docker).

Após baixar o Dockerfile acima, você pode compilar a imagem docker executando:

```shell
docker build . -t tflite-builder -f tflite-android.Dockerfile
```

### Compile arquivos AAR para projetos do Android

Baixe o script para compilar com o Docker executando:

```sh
curl -o build_aar_with_docker.sh \
  https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/lite/tools/build_aar_with_docker.sh &&
chmod +x build_aar_with_docker.sh
```

Em seguida, para compilar os AARs personalizados do TensorFlow Lite, basta fornecer os caminhos de arquivo abaixo ao seu modelo.

```sh
sh build_aar_with_docker.sh \
  --input_models=/a/b/model_one.tflite,/c/d/model_two.tflite \
  --target_archs=x86,x86_64,arm64-v8a,armeabi-v7a \
  --checkpoint=master \
  [--cache_dir=<path to cache directory>]
```

O sinalizador `checkpoint` é um commit, um branch ou uma tag do repositório do TensorFlow do qual você deseja fazer o checkout antes de compilar as bibliotecas. Por padrão, é o branch da versão mais recente. O comando acima vai gerar o arquivo AAR `tensorflow-lite.aar` para as operações personalizadas e integradas do TensorFlow Lite e, opcionalmente, vai gerar o arquivo AAR `tensorflow-lite-select-tf-ops.aar` para operações específicas do TensorFlow no diretório atual.

O argumento --cache_dir especifica o diretório de cache. Caso não seja fornecido, o script vai criar um diretório chamado `bazel-build-cache` no diretório de trabalho atual para fazer o cache.

## Adicione arquivos AAR ao projeto

Adicione arquivos AAR [importando diretamente o AAR ao projeto](https://www.tensorflow.org/lite/android/lite_build#add_aar_directly_to_project) ou [publicando o AAR personalizado em seu repositório Maven atual](https://www.tensorflow.org/lite/android/lite_build#install_aar_to_local_maven_repository). Atenção: você também precisa adicionar os arquivos AAR para `tensorflow-lite-select-tf-ops.aar`, se for gerá-lo.

## Compile seletivamente para iOS

Confira a [seção Compilando localmente](../guide/build_ios.md#building_locally) para configurar o ambiente de compilação e o workspace do TensorFlow. Depois, siga o [guia](../guide/build_ios.md#selectively_build_tflite_frameworks) para usar o script de builds seletivas para iOS.
