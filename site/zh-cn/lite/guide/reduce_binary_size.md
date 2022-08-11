# 减小 TensorFlow Lite 二进制文件大小

## 概述

为设备端机器学习 (ODML) 应用部署模型时，必须注意移动设备上的内存有限。模型二进制文件的大小与模型中使用的算子数量密切相关。通过选择性构建，TensorFlow Lite 让您可以缩减模型二进制文件的大小。选择性构建会跳过在模型集中不使用的算子，从而产生一个只包含让模型在移动设备上运行所必需的运行时和算子内核的紧凑库。

选择性构建适用于以下三个运算库。

1. [TensorFlow Lite 内置运算库](https://www.tensorflow.org/lite/guide/ops_compatibility)
2. [TensorFlow Lite 自定义运算](https://www.tensorflow.org/lite/guide/ops_custom)
3. [选择 TensorFlow 运算库](https://www.tensorflow.org/lite/guide/ops_select)

下表说明了某些常见用例的选择性构建：

<table>
  <thead>
    <tr>
      <th>模型名称</th>
      <th>域</th>
      <th>目标架构</th>
      <th>AAR 文件大小</th>
    </tr>
  </thead>
  <tr>
    <td rowspan="2">
      <a href="https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224.tgz">Mobilenet_1.0_224(float)</a>
    </td>
    <td rowspan="2">图像分类</td>
    <td>armeabi-v7a</td>
    <td>tensorflow-lite.aar（296,635 字节）</td>
  </tr>
   <tr>
    <td>arm64-v8a</td>
    <td>tensorflow-lite.aar（382,892 字节）</td>
  </tr>
  <tr>
    <td rowspan="2">
      <a href="https://tfhub.dev/google/lite-model/spice/">SPICE</a>
    </td>
    <td rowspan="2">声音基音提取</td>
    <td>armeabi-v7a</td>
    <td>tensorflow-lite.aar（375,813 字节）<br>tensorflow-lite-select-tf-ops.aar（1,676,380 字节）</td>
  </tr>
   <tr>
    <td>arm64-v8a</td>
    <td>tensorflow-lite.aar（421,826 字节）<br>tensorflow-lite-select-tf-ops.aar（2,298,630 字节）</td>
  </tr>
  <tr>
    <td rowspan="2">
      <a href="https://tfhub.dev/deepmind/i3d-kinetics-400/1">i3d-kinetics-400</a>
    </td>
    <td rowspan="2">视频分类</td>
    <td>armeabi-v7a</td>
    <td>tensorflow-lite.aar（240,085 字节）<br>tensorflow-lite-select-tf-ops.aar（1,708,597 字节）</td>
  </tr>
   <tr>
    <td>arm64-v8a</td>
    <td>tensorflow-lite.aar（273,713 字节）<br>tensorflow-lite-select-tf-ops.aar（2,339,697 字节）</td>
  </tr>
 </table>

注：此功能目前处于实验阶段，从 2.4 版本开始提供，并且可能发生更改。

## 使用 Bazel 选择性构建 TensorFlow Lite

This section assumes that you have downloaded TensorFlow source codes and [set up the local development environment](https://www.tensorflow.org/lite/android/lite_build#set_up_build_environment_without_docker) to Bazel.

### 为 Android 项目构建 AAR 文件

通过按如下方式提供模型文件路径，您可以构建自定义 TensorFlow Lite AAR。

```sh
sh tensorflow/lite/tools/build_aar.sh \
  --input_models=/a/b/model_one.tflite,/c/d/model_two.tflite \
  --target_archs=x86,x86_64,arm64-v8a,armeabi-v7a
```

上面的命令将为 TensorFlow Lite 的内置和自定义算子生成 AAR 文件 `bazel-bin/tmp/tensorflow-lite.aar`；如果您的模型包含 Select TensorFlow 算子，您还可以选择生成 AAR 文件 `bazel-bin/tmp/tensorflow-lite-select-tf-ops.aar`。请注意，这会构建具有多个不同架构的“胖”AAR 文件；如果您不需要所有架构，请使用适用于您的部署环境的子集。

### Build with custom ops

如果您已经使用自定义算子开发了 Tensorflow Lite 模型，则可以通过将以下标记添加到构建命令来构建这些模型：

```sh
sh tensorflow/lite/tools/build_aar.sh \
  --input_models=/a/b/model_one.tflite,/c/d/model_two.tflite \
  --target_archs=x86,x86_64,arm64-v8a,armeabi-v7a \
  --tflite_custom_ops_srcs=/e/f/file1.cc,/g/h/file2.h \
  --tflite_custom_ops_deps=dep1,dep2
```

`tflite_custom_ops_srcs` 标记包含您的自定义算子的源文件，`tflite_custom_ops_deps` 标记则包含构建这些源文件的依赖项。请注意，TensorFlow 仓库中必须存在这些依赖项。

### Advanced Usages: Custom Bazel rules

If your project is using Bazel and you would like to define custom TFLite dependencies for a given set of models, you can define following rule(s) in your project repository:

For the models with the builtin ops only:

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

For the models with the [Select TF ops](../guide/ops_select.md):

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

### Advanced Usages: Build custom C/C++ shared libraries

If you would like to build your own custom TFLite C/C++ shared objects towards the given models, you can follow the below steps:

Create a temporary BUILD file by running the following command at the root directory of the TensorFlow source code:

```sh
mkdir -p tmp && touch tmp/BUILD
```

#### Building custom C shared objects

If you would like to build a custom TFLite C shared object, add the following to `tmp/BUILD` file:

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

The newly added target can be built as follows:

```sh
bazel build -c opt --cxxopt=--std=c++17 \
  //tmp:tensorflowlite_c
```

and for Android (replace `android_arm` with `android_arm64` for 64-bit):

```sh
bazel build -c opt --cxxopt=--std=c++17 --config=android_arm \
  //tmp:tensorflowlite_c
```

#### Building custom C++ shared objects

If you would like to build a custom TFLite C++ shared object, add the following to `tmp/BUILD` file:

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

The newly added target can be built as follows:

```sh
bazel build -c opt  --cxxopt=--std=c++17 \
  //tmp:tensorflowlite
```

and for Android (replace `android_arm` with `android_arm64` for 64-bit):

```sh
bazel build -c opt --cxxopt=--std=c++17 --config=android_arm \
  //tmp:tensorflowlite
```

For the models with the Select TF ops, you also need to build the following shared library as well:

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

The newly added target can be built as follows:

```sh
bazel build -c opt --cxxopt='--std=c++17' \
      --config=monolithic \
      --host_crosstool_top=@bazel_tools//tools/cpp:toolchain \
      //tmp:tensorflowlite_flex
```

and for Android (replace `android_arm` with `android_arm64` for 64-bit):

```sh
bazel build -c opt --cxxopt='--std=c++17' \
      --config=android_arm \
      --config=monolithic \
      --host_crosstool_top=@bazel_tools//tools/cpp:toolchain \
      //tmp:tensorflowlite_flex
```

## 使用 Docker 选择性构建 TensorFlow Lite

This section assumes that you have installed [Docker](https://docs.docker.com/get-docker/) on your local machine and downloaded the TensorFlow Lite Dockerfile [here](https://www.tensorflow.org/lite/android/lite_build#set_up_build_environment_using_docker).

下载上述 Dockerfile 之后，您可以通过运行以下命令构建 Docker 镜像：

```shell
docker build . -t tflite-builder -f tflite-android.Dockerfile
```

### 为 Android 项目构建 AAR 文件

运行以下命令，下载使用 Docker 构建模型的脚本：

```sh
curl -o build_aar_with_docker.sh \
  https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/lite/tools/build_aar_with_docker.sh &&
chmod +x build_aar_with_docker.sh
```

随后，通过按如下方式提供模型文件路径，您可以构建自定义 TensorFlow Lite AAR。

```sh
sh build_aar_with_docker.sh \
  --input_models=/a/b/model_one.tflite,/c/d/model_two.tflite \
  --target_archs=x86,x86_64,arm64-v8a,armeabi-v7a \
  --checkpoint=master \
  [--cache_dir=<path to cache directory>]
```

`checkpoint` 标记是您希望在构建库之前签出的 TensorFlow 仓库的提交、分支或标签。上面的命令将为 TensorFlow Lite 的内置和自定义算子生成 AAR 文件 `tensorflow-lite.aar`，对于您的当前目录下的 Select TensorFlow 算子，还可以选择生成 AAR 文件 `tensorflow-lite-select-tf-ops.aar`。

--cache_dir 指定缓存目录。如果未提供，脚本将在当前工作目录下创建一个名为 `bazel-build-cache` 的目录用于缓存。

## 将 AAR 文件添加到项目

Add AAR files by directly [importing the AAR into your project](https://www.tensorflow.org/lite/android/lite_build#add_aar_directly_to_project), or by [publishing the custom AAR to your local Maven repository](https://www.tensorflow.org/lite/android/lite_build#install_aar_to_local_maven_repository). Note that you have to add the AAR files for `tensorflow-lite-select-tf-ops.aar` as well if you generate it.

## Selective Build for iOS

Please see the [Building locally section](../guide/build_ios.md#building_locally) to set up the build environment and configure TensorFlow workspace and then follow the [guide](../guide/build_ios.md#selectively_build_tflite_frameworks) to use the selective build script for iOS.
