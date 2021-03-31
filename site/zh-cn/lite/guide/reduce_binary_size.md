# 缩减 TensorFlow Lite 二进制文件大小

## 概述

为设备端机器学习 (ODML) 应用部署模型时，必须注意移动设备上的内存有限。模型二进制文件的大小与模型中使用的算子数量密切相关。通过选择性构建，TensorFlow Lite 让您可以缩减模型二进制文件的大小。选择性构建会跳过在模型集中不使用的算子，从而产生一个只包含让模型在移动设备上运行所必需的运行时和算子内核的紧凑库。

选择性构建适用于以下三个算子库。

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
    <td rowspan="2">       <a href="https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224.tgz">Mobilenet_1.0_224(float)</a>
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
    <td rowspan="2">       <a href="https://tfhub.dev/google/lite-model/spice/">SPICE</a>
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
    <td rowspan="2">       <a href="https://tfhub.dev/deepmind/i3d-kinetics-400/1">i3d-kinetics-400</a>
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

注：此功能从 2.4 版本开始提供，目前处于实验阶段，随时可能发生更改。

## 已知问题/限制

1. 对于 C API 和 iOS 版本，目前不支持选择性构建。

## 使用 Bazel 选择性构建 TensorFlow Lite

本部分假设您已下载 TensorFlow 源代码，并且已[设置 Bazel 的本地开发环境](https://www.tensorflow.org/lite/guide/android#build_tensorflow_lite_locally)。

### 为 Android 项目构建 AAR 文件

通过按如下方式提供模型文件路径，您可以构建自定义 TensorFlow Lite AAR。

```sh
sh tensorflow/lite/tools/build_aar.sh \
  --input_models=/a/b/model_one.tflite,/c/d/model_two.tflite \
  --target_archs=x86,x86_64,arm64-v8a,armeabi-v7a
```

上面的命令将为 TensorFlow Lite 的内置和自定义算子生成 AAR 文件 `bazel-bin/tmp/tensorflow-lite.aar`；如果您的模型包含 Select TensorFlow 算子，您还可以选择生成 AAR 文件 `bazel-bin/tmp/tensorflow-lite-select-tf-ops.aar`。请注意，这会构建具有多个不同架构的“胖”AAR 文件；如果您不需要所有架构，请使用适用于您的部署环境的子集。

### 高级用法：使用自定义算子构建

如果您已经使用自定义算子开发了 Tensorflow Lite 模型，则可以通过将以下标记添加到构建命令来构建这些模型：

```sh
sh tensorflow/lite/tools/build_aar.sh \
  --input_models=/a/b/model_one.tflite,/c/d/model_two.tflite \
  --target_archs=x86,x86_64,arm64-v8a,armeabi-v7a \
  --tflite_custom_ops_srcs=/e/f/file1.cc,/g/h/file2.h \
  --tflite_custom_ops_deps=dep1,dep2
```

`tflite_custom_ops_srcs` 标记包含您的自定义算子的源文件，`tflite_custom_ops_deps` 标记则包含构建这些源文件的依赖项。请注意，TensorFlow 仓库中必须存在这些依赖项。

## 使用 Docker 选择性构建 TensorFlow Lite

本部分假设您已在本地计算机上安装 [Docker](https://docs.docker.com/get-docker/)，并且已从[此处](https://www.tensorflow.org/lite/guide/build_android#set_up_build_environment_using_docker)下载了  TensorFlow Lite Dockerfile。

下载上述 Dockerfile 之后，您可以通过运行以下命令构建 Docker 镜像：

```shell
docker build . -t tflite-builder -f tflite-android.Dockerfile
```

### 为 Android 项目构建 AAR 文件

运行以下命令，下载使用 Docker 进行构建的脚本：

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

通过直接[将 AAR 导入项目](https://www.tensorflow.org/lite/guide/android#add_aar_directly_to_project)，或者[将自定义 AAR 发布到本地 Maven 存储库](https://www.tensorflow.org/lite/guide/android#install_aar_to_local_maven_repository)，您可以添加 AAR 文件。请注意，您还必须为 `tensorflow-lite-select-tf-ops.aar` 添加 AAR 文件。
