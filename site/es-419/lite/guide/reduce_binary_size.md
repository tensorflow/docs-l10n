# Reducir el tamaño del binario de TensorFlow Lite

## Visión general

Al implementar modelos para aplicaciones de aprendizaje automático en el dispositivo (ODML), es importante ser consciente de la limitada memoria disponible en los dispositivos móviles. Los tamaños del binario del modelo están estrechamente correlacionados con el número de ops usadas en el modelo. TensorFlow Lite le permite reducir los tamaños binarios del modelo usando construcciones selectivas. Las construcciones selectivas omiten las operaciones no utilizadas en su conjunto de modelos y producen una librería compacta con sólo el runtime y los kernels op necesarios para que el modelo funcione en su dispositivo móvil.

La compilación selectiva se aplica en las tres librerías de operaciones siguientes.

1. [Librería de ops integrada en TensorFlow Lite](https://www.tensorflow.org/lite/guide/ops_compatibility)
2. [Ops personalizadas de TensorFlow Lite](https://www.tensorflow.org/lite/guide/ops_custom)
3. [Librería de ops seleccionadas de TensorFlow](https://www.tensorflow.org/lite/guide/ops_select)

La tabla siguiente demuestra el impacto de las compilaciones selectivas para algunos casos de uso comunes:

<table>
  <thead>
    <tr>
      <th>Nombre del modelo</th>
      <th>Dominio</th>
      <th>Arquitectura destino</th>
      <th>Tamaño(s) del archivo AAR</th>
    </tr>
  </thead>
  <tr>
    <td rowspan="2">       <a href="https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224.tgz">Mobilenet_1.0_224(float)</a>
</td>
    <td rowspan="2">Clasificación de imágenes</td>
    <td>armeabi-v7a</td>
    <td>tensorflow-lite.aar (296,635 bytes)</td>
  </tr>
   <tr>
    <td>arm64-v8a</td>
    <td>tensorflow-lite.aar (382,892 bytes)</td>
  </tr>
  <tr>
    <td rowspan="2">       <a href="https://tfhub.dev/google/lite-model/spice/">SPICE</a>
</td>
    <td rowspan="2">Extracción del tono del sonido</td>
    <td>armeabi-v7a</td>
    <td>tensorflow-lite.aar (375,813 bytes)<br>tensorflow-lite-select-tf-ops.aar (1,676,380 bytes)</td>
  </tr>
   <tr>
    <td>arm64-v8a</td>
    <td>tensorflow-lite.aar (421,826 bytes)<br>tensorflow-lite-select-tf-ops.aar (2,298,630 bytes)</td>
  </tr>
  <tr>
    <td rowspan="2">       <a href="https://tfhub.dev/deepmind/i3d-kinetics-400/1">i3d-kinetics-400</a>
</td>
    <td rowspan="2">Clasificación de vídeos</td>
    <td>armeabi-v7a</td>
    <td>tensorflow-lite.aar (240,085 bytes)<br>tensorflow-lite-select-tf-ops.aar (1,708,597 bytes)</td>
  </tr>
   <tr>
    <td>arm64-v8a</td>
    <td>tensorflow-lite.aar (273,713 bytes)<br>tensorflow-lite-select-tf-ops.aar (2,339,697 bytes)</td>
  </tr>
 </table>

Nota: Esta característica es actualmente experimental y está disponible desde la versión 2.4 y puede cambiar.

## Generar selectivamente TensorFlow Lite con Bazel

Esta sección asume que ha descargado los códigos fuente de TensorFlow y [ha configurado el entorno de desarrollo local](https://www.tensorflow.org/lite/android/lite_build#set_up_build_environment_without_docker) a Bazel.

### Generar archivos AAR para el proyecto Android

Puede generar los AAR personalizados de TensorFlow Lite indicando las rutas de los archivos de su modelo de la siguiente forma.

```sh
sh tensorflow/lite/tools/build_aar.sh \
  --input_models=/a/b/model_one.tflite,/c/d/model_two.tflite \
  --target_archs=x86,x86_64,arm64-v8a,armeabi-v7a
```

El comando anterior generará el archivo AAR `bazel-bin/tmp/tensorflow-lite.aar` para las ops incorporadas y personalizadas de TensorFlow Lite; y opcionalmente, genera el archivo aar `bazel-bin/tmp/tensorflow-lite-select-tf-ops.aar` si sus modelos contienen ops de TensorFlow seleccionadas. Tenga en cuenta que esto genera un AAR "gordo" con varias arquitecturas diferentes; si no las necesita todas, use el subconjunto apropiado para su entorno de implementación.

### Generar con ops personalizadas

Si ha desarrollado modelos Tensorflow Lite con ops personalizadas, puede generarlas añadiendo los siguientes indicadores al comando de compilación:

```sh
sh tensorflow/lite/tools/build_aar.sh \
  --input_models=/a/b/model_one.tflite,/c/d/model_two.tflite \
  --target_archs=x86,x86_64,arm64-v8a,armeabi-v7a \
  --tflite_custom_ops_srcs=/e/f/file1.cc,/g/h/file2.h \
  --tflite_custom_ops_deps=dep1,dep2
```

El indicador `tflite_custom_ops_srcs` contiene los archivos fuente de sus ops personalizadas y el indicador `tflite_custom_ops_deps` contiene las dependencias para generar esos archivos fuente. Tenga en cuenta que estas dependencias deben existir en el repositorio de TensorFlow.

### Usos avanzados: Reglas de Bazel personalizadas

Si su proyecto utiliza Bazel y desea configurar dependencias TFLite personalizadas para un determinado conjunto de modelos, puede configurar la(s) siguiente(s) regla(s) en el repositorio de su proyecto:

Sólo para los modelos con las ops integradas:

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

Para los modelos con las [ops de TF seleccionadas](../guide/ops_select.md):

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

### Usos avanzados: Generar librerías compartidas C/C++ personalizadas

Si desea generar sus propios objetos compartidos TFLite C/C++ personalizados a partir de los modelos dados, puede seguir los pasos que se indican a continuación:

Cree un archivo BUILD temporal ejecutando el siguiente comando en el directorio raíz del código fuente de TensorFlow:

```sh
mkdir -p tmp && touch tmp/BUILD
```

#### Generación de objetos compartidos en C personalizados

Si desea generar un objeto compartido TFLite C personalizado, añada lo siguiente al archivo `tmp/BUILD`:

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

El nuevo destino añadido puede generarse de la siguiente manera:

```sh
bazel build -c opt --cxxopt=--std=c++17 \
  //tmp:tensorflowlite_c
```

y para Android (sustituya `android_arm` por `android_arm64` para 64 bits):

```sh
bazel build -c opt --cxxopt=--std=c++17 --config=android_arm \
  //tmp:tensorflowlite_c
```

#### Generación de objetos compartidos en C++ personalizados

Si desea generar un objeto compartido TFLite C++ personalizado, añada lo siguiente al archivo `tmp/BUILD`:

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

El nuevo destino añadido puede generarse de la siguiente manera:

```sh
bazel build -c opt  --cxxopt=--std=c++17 \
  //tmp:tensorflowlite
```

y para Android (sustituya `android_arm` por `android_arm64` para 64 bits):

```sh
bazel build -c opt --cxxopt=--std=c++17 --config=android_arm \
  //tmp:tensorflowlite
```

Para los modelos con las ops de TF seleccionadas, también necesita generar la siguiente librería compartida:

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

El nuevo destino añadido puede generarse de la siguiente manera:

```sh
bazel build -c opt --cxxopt='--std=c++17' \
      --config=monolithic \
      --host_crosstool_top=@bazel_tools//tools/cpp:toolchain \
      //tmp:tensorflowlite_flex
```

y para Android (sustituya `android_arm` por `android_arm64` para 64 bits):

```sh
bazel build -c opt --cxxopt='--std=c++17' \
      --config=android_arm \
      --config=monolithic \
      --host_crosstool_top=@bazel_tools//tools/cpp:toolchain \
      //tmp:tensorflowlite_flex
```

## Generar selectivamente TensorFlow Lite con Docker

Esta sección asume que usted ha instalado [Docker](https://docs.docker.com/get-docker/) en su máquina local y descargado el Dockerfile de TensorFlow Lite [aquí](https://www.tensorflow.org/lite/android/lite_build#set_up_build_environment_using_docker).

Después de descargar el Dockerfile anterior, puede generar la imagen Docker ejecutando:

```shell
docker build . -t tflite-builder -f tflite-android.Dockerfile
```

### Generar archivos AAR para el proyecto Android

Descargue el script para compilar con Docker ejecutando:

```sh
curl -o build_aar_with_docker.sh \
  https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/lite/tools/build_aar_with_docker.sh &&
chmod +x build_aar_with_docker.sh
```

Luego, puede generar los AAR personalizados de TensorFlow Lite indicando las rutas de los archivos de su modelo de la siguiente forma.

```sh
sh build_aar_with_docker.sh \
  --input_models=/a/b/model_one.tflite,/c/d/model_two.tflite \
  --target_archs=x86,x86_64,arm64-v8a,armeabi-v7a \
  --checkpoint=master \
  [--cache_dir=<path to cache directory>]
```

El Indicador `checkpoint` es un commit, una derivación o una etiqueta del repositorio de TensorFlow al que desea hacer checkout antes de generar las librerías; por defecto es la última variante de lanzamiento. El comando anterior generará el archivo AAR `tensorflow-lite.aar` para las ops incorporadas y personalizadas de TensorFlow Lite y, opcionalmente, el archivo AAR `tensorflow-lite-select-tf-ops.aar` para las ops de TensorFlow seleccionadas en su directorio actual.

El --cache_dir especifica el directorio de caché. Si no se indica, el script creará un directorio llamado `bazel-build-cache` bajo el directorio de trabajo actual para la caché.

## Añadir archivos AAR al proyecto

Añada los archivos AAR directamente [importando el AAR a su proyecto](https://www.tensorflow.org/lite/android/lite_build#add_aar_directly_to_project), o [publicando el AAR personalizado en su repositorio local de Maven](https://www.tensorflow.org/lite/android/lite_build#install_aar_to_local_maven_repository). Tenga en cuenta que tiene que añadir los archivos AAR para `tensorflow-lite-select-tf-ops.aar` también si lo genera.

## Compilación selectiva para iOS

Consulte la [sección Compilar localmente](../guide/build_ios.md#building_locally) para configurar el entorno de compilación y configurar el espacio de trabajo de TensorFlow y, a continuación, siga la [guía](../guide/build_ios.md#selectively_build_tflite_frameworks) para usar la secuencia de comandos de compilación selectiva para iOS.
