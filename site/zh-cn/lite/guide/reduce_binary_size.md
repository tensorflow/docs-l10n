# 减小 TensorFlow Lite 二进制文件大小

## 概述

为设备端机器学习 (ODML) 应用部署模型时，必须注意移动设备上的内存有限。模型二进制文件的大小与模型中使用的算子数量密切相关。TensorFlow Lite 使您可以通过选择性构建来减小模型二进制文件的大小。选择性构建会跳过在您的模型集中用不到的算子，从而生成只包含供模型在移动设备上运行所必需的运行时和算子内核的紧凑库。

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
    <td rowspan="2">       Mobilenet_1.0_224(float)</td>
    <td rowspan="2">图像分类</td>
    <td>armeabi-v7a</td>
    <td>tensorflow-lite.aar（296,635 字节）</td>
  </tr>
   <tr>
    <td>arm64-v8a</td>
    <td>tensorflow-lite.aar（382,892 字节）</td>
  </tr>
  <tr>
    <td rowspan="2">       SPICE</td>
    <td rowspan="2">声音基音提取</td>
    <td>armeabi-v7a</td>
    <td>tensorflow-lite.aar（375,813 字节）<br>tensorflow-lite-select-tf-ops.aar（1,676,380 字节）</td>
  </tr>
   <tr>
    <td>arm64-v8a</td>
    <td>tensorflow-lite.aar（421,826 字节）<br>tensorflow-lite-select-tf-ops.aar（2,298,630 字节）</td>
  </tr>
  <tr>
    <td rowspan="2">       i3d-kinetics-400</td>
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

本部分假设您已下载 TensorFlow 源代码，并且已[设置 Bazel 的本地开发环境](https://www.tensorflow.org/lite/android/lite_build#set_up_build_environment_without_docker)。

### 为 Android 项目构建 AAR 文件

随后，您可以按如下方式提供模型文件路径，构建自定义 TensorFlow Lite AAR。

```sh
sh tensorflow/lite/tools/build_aar.sh \
  --input_models=/a/b/model_one.tflite,/c/d/model_two.tflite \
  --target_archs=x86,x86_64,arm64-v8a,armeabi-v7a
```

上面的命令将为 TensorFlow Lite 的内置算子和自定义算子生成 AAR 文件 `bazel-bin/tmp/tensorflow-lite.aar`；如果您的模型包含 Select TensorFlow 算子，您还可以选择生成 AAR 文件 `bazel-bin/tmp/tensorflow-lite-select-tf-ops.aar`。请注意，这会构建具有多个不同架构的“肥胖”AAR 文件；如果您不需要所有架构，请使用适用于您的部署环境的子集。

### 使用自定义算子构建

如果您已经使用自定义算子开发了 TensorFlow Lite 模型，则可以通过将以下标记添加到构建命令来构建这些模型：

```sh
sh tensorflow/lite/tools/build_aar.sh \
  --input_models=/a/b/model_one.tflite,/c/d/model_two.tflite \
  --target_archs=x86,x86_64,arm64-v8a,armeabi-v7a \
  --tflite_custom_ops_srcs=/e/f/file1.cc,/g/h/file2.h \
  --tflite_custom_ops_deps=dep1,dep2
```

`tflite_custom_ops_srcs` 标记包含您的自定义算子的源文件，`tflite_custom_ops_deps` 标记则包含构建这些源文件的依赖项。请注意，TensorFlow 仓库中必须存在这些依赖项。

### 高级用法：自定义 Bazel 规则

如果您的项目使用 Bazel，并且您希望为给定的一组模型定义自定义 TFLite 依存项，则可以在项目存储库中定义以下规则：

仅适用于具有内置算子的模型：

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

适用于具有[精选 TF 算子](../guide/ops_select.md)的模型：

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

### 高级用法：构建自定义 C/C++ 共享库

如果您想要针对给定的模型构建您自己的自定义 TFLite C/C++ 共享对象，可以执行以下步骤：

通过在 TensorFlow 源代码的根目录中运行以下命令来创建临时 BUILD 文件：

```sh
mkdir -p tmp && touch tmp/BUILD
```

#### 构建自定义 C 共享对象

如果要构建自定义 TFLite C 共享对象，请将以下内容添加到 `tmp/BUILD` 文件中：

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

新添加的目标可以构建如下：

```sh
bazel build -c opt --cxxopt=--std=c++17 \
  //tmp:tensorflowlite_c
```

对于 Android（对于 64 位，将 `android_arm` 替换为 `android_arm64`）：

```sh
bazel build -c opt --cxxopt=--std=c++17 --config=android_arm \
  //tmp:tensorflowlite_c
```

#### 构建自定义 C++ 共享对象

如果要构建自定义 TFLite C++ 共享对象，请将以下内容添加到 `tmp/BUILD` 文件中：

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

新添加的目标可以构建如下：

```sh
bazel build -c opt  --cxxopt=--std=c++17 \
  //tmp:tensorflowlite
```

对于 Android（对于 64 位，将 `android_arm` 替换为 `android_arm64`）：

```sh
bazel build -c opt --cxxopt=--std=c++17 --config=android_arm \
  //tmp:tensorflowlite
```

对于带有精选 TF 算子的模型，您还需要构建以下共享库：

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

新添加的目标可以构建如下：

```sh
bazel build -c opt --cxxopt='--std=c++17' \
      --config=monolithic \
      --host_crosstool_top=@bazel_tools//tools/cpp:toolchain \
      //tmp:tensorflowlite_flex
```

对于 Android（对于 64 位，将 `android_arm` 替换为 `android_arm64`）：

```sh
bazel build -c opt --cxxopt='--std=c++17' \
      --config=android_arm \
      --config=monolithic \
      --host_crosstool_top=@bazel_tools//tools/cpp:toolchain \
      //tmp:tensorflowlite_flex
```

## 使用 Docker 选择性构建 TensorFlow Lite

本部分假设您已在本地计算机上安装了 [Docker](https://docs.docker.com/get-docker/)，并且已从[此处](https://www.tensorflow.org/lite/android/lite_build#set_up_build_environment_using_docker)下载了 TensorFlow Lite Dockerfile。

下载上述 Dockerfile 之后，您可以通过运行以下命令来构建 Docker 镜像：

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

您可以通过直接[将 AAR 导入项目](https://www.tensorflow.org/lite/android/lite_build#add_aar_directly_to_project)，或者[将自定义 AAR 发布到本地 Maven 存储库](https://www.tensorflow.org/lite/android/lite_build#install_aar_to_local_maven_repository)来添加 AAR 文件。请注意，如果生成了 `tensorflow-lite-select-tf-ops.aar`，您也必须为它添加 AAR 文件。

## 针对 iOS 的选择性构建

请参阅[本地构建部分](../guide/build_ios.md#building_locally)以设置构建环境并配置 TensorFlow 工作区，然后按照[指南](../guide/build_ios.md#selectively_build_tflite_frameworks)使用 iOS 的选择性构建脚本。
