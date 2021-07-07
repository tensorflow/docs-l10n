# Android 快速入门

要开始在 Android 上使用 TensorFlow Lite，我们建议浏览以下示例。

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android">Android 图像分类示例</a>

有关源代码的说明，请阅读 [TensorFlow Lite Android 图像分类](https://tensorflow.google.cn/lite/models/image_classification/android)。

此示例应用使用[图像分类](https://tensorflow.google.cn/lite/models/image_classification/overview)对从设备的后置摄像头看到的图像进行连续分类。此应用可以在设备或模拟器上运行。

使用 TensorFlow Lite Java API 和 [TensorFlow Lite Android Support Library](https://tensorflow.google.cn/lite/performance/post_training_quantization) 执行推断。演示应用实时对帧进行分类，同时显示最可能的分类。它允许用户在浮点模型或[量化](https://developer.android.com/ndk/guides/neuralnetworks)模型之间进行选择，选择线程数，以及决定是在 CPU 上、GPU 上还是通过 [NNAPI](https://developer.android.com/ndk/guides/neuralnetworks) 运行。

注：在各种用例中演示 TensorFlow Lite 的其他 Android 应用可在[示例](https://tensorflow.google.cn/lite/examples)中获得。

## 在 Android Studio 中构建

要在 Android Studio 构建示例，请按照 [README.md](https://github.com/tensorflow/examples/blob/master/lite/examples/image_classification/android/README.md) 中的说明操作。

## 创建您自己的 Android 应用

要快速开始编写自己的 Android 代码，我们建议使用 [Android 图像分类示例](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android)作为起点。

以下部分包含一些有关如何在 Android 上使用 TensorFlow Lite 的有用信息。

### 使用 Android Studio 机器学习模型绑定

注：需要 [Android Studio 4.1](https://developer.android.com/studio) 或更高版本

要导入 TensorFlow Lite (TFLite) 模型，请执行以下操作：

1. 右键点击您要使用 TFLite 模型的模块，或者点击 `File`，然后依次点击 `New`&gt;`Other`&gt;`TensorFlow Lite Model` ![右键点击菜单可访问 TensorFlow Lite 导入功能](https://github.com/tensorflow/docs-l10n/blob/master/site/zh-cn/lite/images/android/right_click_menu.png?raw=true)

2. 选择 TFLite 文件的位置。请注意，该工具将使用机器学习模型绑定代您配置模块的依赖关系，所有依赖项将自动插入 Android 模块的 `build.gradle` 文件。

    可选：如果要使用 [GPU 加速](../performance/gpu)，请选择用于导入 TensorFlow GPU 的第二个复选框。![TFLite 模型的导入对话框](../images/android/import_dialog.png)

3. 点击 `Finish`。

4. 导入成功后，会出现以下界面。要开始使用该模型，请选择 Kotlin 或 Java，复制并粘贴 `Sample Code` 部分的代码。在 Android Studio 中双击 `ml` 目录下的 TFLite 模型，可以返回此界面。![Android Studio 中的模型详细信息页面](https://github.com/tensorflow/docs-l10n/blob/master/site/zh-cn/lite/images/android/model_details.png?raw=true)

### 使用 TensorFlow Lite Task Library

TensorFlow Lite Task Library 包含一组功能强大且易于使用的任务特定库，可供应用开发者利用 TFLite 创造机器学习体验。它为流行的机器学习任务（例如图像分类、问答等）提供了经过优化的开箱即用模型接口。这些模型接口专为每项任务而设计，可实现最佳性能和易用性。Task Library 可以跨平台运行，并在 Java、C++ 和 Swift（即将推出）上受支持。

要在您的 Android 应用中使用 Support Library，我们建议将 MavenCentral 中托管的 AAR 分别用于 [Task Vision 库](https://search.maven.org/artifact/org.tensorflow/tensorflow-lite-task-vision)和 [Task Text 库](https://search.maven.org/artifact/org.tensorflow/tensorflow-lite-task-text)。

您可以在 `build.gradle` 依赖项中加以指定，如下所示：

```build
dependencies {
    implementation 'org.tensorflow:tensorflow-lite-task-vision:0.1.0'
    implementation 'org.tensorflow:tensorflow-lite-task-text:0.1.0'
}
```

要使用 Nightly 快照，请确保您已添加 [Sonatype 快照存储库](./build_android#use_nightly_snapshots)。

请参阅 [TensorFlow Lite Task Library 概述](https://developer.android.com/ndk/guides/abis)中的简介，了解更多详细信息。

### 使用 TensorFlow Lite Android Support Library

借助 TensorFlow Lite Android Support Library，可以更轻松地将模型集成到应用中。它提供了高级 API，可帮助用户将原始输入数据转换为模型所需的形式，并解释模型的输出，从而减少所需的样板代码量。

它支持输入和输出的常用数据格式，包括图像和数组。此外，它还提供可执行各种任务（例如图像大小调整和裁剪）的预处理和后处理单元。

要在您的 Android 应用中使用 Support Library，我们建议使用 [MavenCentral 中托管的 TensorFlow Lite Support Library AAR](https://search.maven.org/artifact/org.tensorflow/tensorflow-lite-support)。

您可以在 `build.gradle` 依赖项中加以指定，如下所示：

```build
dependencies {
    implementation 'org.tensorflow:tensorflow-lite-support:0.1.0'
}
```

要使用 Nightly 快照，请确保您已添加 [Sonatype 快照存储库](./build_android#use_nightly_snapshots)。

首先，请遵循 [TensorFlow Lite Android Support Library](../inference_with_metadata/lite_support.md) 中的说明。

### 使用 MavenCentral 中的 TensorFlow Lite AAR

要在您的 Android 应用中使用 TensorFlow Lite，我们建议使用 [MavenCentral 中托管的 TensorFlow Lite AAR](https://search.maven.org/artifact/org.tensorflow/tensorflow-lite)。

您可以在 `build.gradle` 依赖项中加以指定，如下所示：

```build
dependencies {
    implementation 'org.tensorflow:tensorflow-lite:0.0.0-nightly-SNAPSHOT'
}
```

要使用 Nightly 快照，请确保您已添加 [Sonatype 快照存储库](./build_android#use_nightly_snapshots)。

此 AAR 包含所有 [Android ABI](https://developer.android.com/ndk/guides/abis) 的二进制文件。您可以通过仅包含需要支持的 ABI 来缩减应用二进制文件的大小。

我们建议大多数开发者忽略 `x86`、`x86_64` 和 `arm32` ABI。这可以通过以下 Gradle 配置来实现，此配置仅包含 `armeabi-v7a` 和 `arm64-v8a`，应当可以涵盖大多数新型 Android 设备。

```build
android {
    defaultConfig {
        ndk {
            abiFilters 'armeabi-v7a', 'arm64-v8a'
        }
    }
}
```

要详细了解 `abiFilters`，请参阅 Android Gradle 文档中的 [`NdkOptions`](https://google.github.io/android-gradle-dsl/current/com.android.build.gradle.internal.dsl.NdkOptions.html)。

## 使用 C++ 构建 Android 应用

如果使用 NDK 构建应用，则可以利用以下两种方式通过 C++ 使用 TFLite：

### 使用 TFLite C API

这是*推荐*方式。下载 [MavenCentral 中托管的 TensorFlow Lite AAR](https://search.maven.org/artifact/org.tensorflow/tensorflow/tensorflow-lite)，将其重命名为 `tensorflow-lite-*.zip`，然后解压缩该文件。您必须在 NDK 项目中包含 `headers/tensorflow/lite/` 和 `headers/tensorflow/lite/c/` 文件夹中的四个头文件，以及 `jni/` 文件夹中的相关 `libtensorflowlite_jni.so` 动态库。

`c_api.h` 头文件包含有关使用 TFLite C API 的基本文档。

### 使用 TFLite C++ API

如果要通过 C++ API 使用 TFLite，您可以构建 C++ 共享库：

32 位 armeabi-v7a：

```sh
bazel build -c opt --config=android_arm //tensorflow/lite:libtensorflowlite.so
```

64 位 arm64-v8a：

```sh
bazel build -c opt --config=android_arm64 //tensorflow/lite:libtensorflowlite.so
```

目前，没有一种直接方式可以提取需要的所有头文件，因此您必须将 TensorFlow 仓库中的所有头文件都包含在 `tensorflow/lite/` 中。此外，您还将需要 [FlatBuffers](https://github.com/google/flatbuffers) 和 [Abseil](https://github.com/abseil/abseil-cpp) 中的头文件。
