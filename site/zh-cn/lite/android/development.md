# 适用于 Android 的开发工具

TensorFlow Lite 提供了许多工具，用于将模型集成到 Android 应用中。本页面介绍了用于使用 Kotlin、Java 和 C++ 构建应用的开发工具，以及对 Android Studio 中的 TensorFlow Lite 开发的支持。

要点：一般来说，您应该使用 [TensorFlow Lite Task Library](#task_library) 将 TensorFlow Lite 集成到您的 Android 应用中，除非该库不支持您的用例。如果 Task Library 不支持，请使用 [TensorFlow Lite Library](#lite_lib) 和 [Support Library](#support_lib)。

要快速开始编写 Android 代码，请参阅 [Android 快速入门](../android/quickstart)

## 使用 Kotlin 和 Java 构建时用到的工具

以下各部分介绍适用于 TensorFlow Lite 并使用 Kotlin 和 Java 语言的开发工具。

### TensorFlow Lite Task Library {:#task_library}

TensorFlow Lite Task Library 包含了一套功能强大且易于使用的任务专用库，供应用开发者使用 TensorFlow Lite 进行构建。它为热门的机器学习任务（如图像分类、问答等）提供了经过优化的开箱即用的模型接口。模型接口专为每个任务而设计，以实现最佳性能和可用性。Task Library 可跨平台工作，并且支持 Java 和 C++。

要在您的 Android 应用中使用 Task Library，请分别为 [Task Vision 库](https://search.maven.org/artifact/org.tensorflow/tensorflow-lite-task-vision)、[Task Text 库](https://search.maven.org/artifact/org.tensorflow/tensorflow-lite-task-text)和 [Task Audio 库](https://search.maven.org/artifact/org.tensorflow/tensorflow-lite-task-audio)使用 MavenCentral 的 AAR。

您可以在 `build.gradle` 依赖项中加以指定，如下所示：

```build
dependencies {
    implementation 'org.tensorflow:tensorflow-lite-task-vision:+'
    implementation 'org.tensorflow:tensorflow-lite-task-text:+'
    implementation 'org.tensorflow:tensorflow-lite-task-audio:+'
}
```

如果您使用 Nightly 快照，请确保将 [Sonatype 快照仓库](./lite_build#use_nightly_snapshots)添加到项目中。

请参阅 [TensorFlow Lite Task Library 概述](../inference_with_metadata/task_library/overview.md)中的简介，了解更多详细信息。

### TensorFlow Lite Library {:#lite_lib}

可以通过将[托管在 MavenCentral 的 AAR](https://search.maven.org/artifact/org.tensorflow/tensorflow-lite) 添加到您的开发项目来使用 TensorFlow Lite Livrary。

您可以在 `build.gradle` 依赖项中加以指定，如下所示：

```build
dependencies {
    implementation 'org.tensorflow:tensorflow-lite:+'
}
```

如果您使用 Nightly 快照，请确保将 [Sonatype 快照仓库](./lite_build#use_nightly_snapshots)添加到项目中。

此 AAR 包含所有 [Android ABI](https://developer.android.com/ndk/guides/abis) 的二进制文件。您可以通过仅包含需要支持的 ABI 来缩减应用二进制文件的大小。

除非您以特定硬件为目标，否则在大多数情况下，您应该忽略 `x86`、`x86_64` 和 `arm32` ABI。您可以使用以下 Gradle 配置对其进行配置。此配置仅包含 `armeabi-v7a` 和 `arm64-v8a`，应该可以涵盖大多数新型 Android 设备。

```build
android {
    defaultConfig {
        ndk {
            abiFilters 'armeabi-v7a', 'arm64-v8a'
        }
    }
}
```

要详细了解 `abiFilters`，请参阅 Android NDK 文档中的 [Android ABI](https://developer.android.com/ndk/guides/abis)。

### TensorFlow Lite Support Library {:#support_lib}

借助 TensorFlow Lite Android Support Library，可以更轻松地将模型集成到应用中。它提供了高级 API，可帮助用户将原始输入数据转换为模型所需的形式，并解释模型的输出，从而减少所需的样板代码量。

它支持输入和输出的常用数据格式，包括图像和数组。此外，它还提供可执行各种任务（例如图像大小调整和裁剪）的预处理和后处理单元。

您可以通过包含 [MavenCentral 中托管的 TensorFlow Lite Support Library AAR](https://search.maven.org/artifact/org.tensorflow/tensorflow-lite-support) ，在您的 Android 应用中使用 Support Library。

您可以在 `build.gradle` 依赖项中加以指定，如下所示：

```build
dependencies {
    implementation 'org.tensorflow:tensorflow-lite-support:+'
}
```

如果您使用 Nightly 快照，请确保将 [Sonatype 快照仓库](./lite_build#use_nightly_snapshots)添加到项目中。

有关如何入门的说明，请参阅 [TensorFlow Lite Android Support Library](../inference_with_metadata/lite_support.md)。

### 用于库的最低 Android SDK 版本

库 | `minSdkVersion` | 设备要求
--- | --- | ---
tensorflow-lite | 19 | NNAPI 使用要求
:                             :                 : API 27+                : |  |
tensorflow-lite-gpu | 19 | GLES 3.1 或 OpenCL
:                             :                 : （通常仅）        : |  |
:                             :                 : 适用于 API 21+ : |  |
tensorflow-lite-hexagon | 19 | -
tensorflow-lite-support | 19 | -
tensorflow-lite-task-vision | 21 | android.graphics.Color
:                             :                 : 相关 API 要求   : |  |
:                             :                 : API 26+                : |  |
tensorflow-lite-task-text | 21 | -
tensorflow-lite-task-audio | 23 | -
tensorflow-lite-metadata | 19 | -

### 使用 Android Studio

除了上面描述的开发库，Android Studio 还支持集成 TensorFlow Lite 模型，如下所述。

#### Android Studio 机器学习模型绑定

Android Studio 4.1 及更高版本的机器学习模型绑定功能允许您将 `.tflite` 模型文件导入到现有的 Android 应用中，并生成接口类，以便更轻松地将代码与模型集成。

要导入 TensorFlow Lite (TFLite) 模型，请执行以下操作：

1. 右键点击您要使用 TFLite 模型的模块，或者点击 **File &gt; New &gt; Other &gt; TensorFlow Lite Model**

2. 选择 TensorFlow Lite 文件的位置。请注意，该工具使用机器学习模型绑定配置模块的依赖项，并会自动将所有必需的依赖项添加到您的 Android 模块的 `build.gradle` 文件中。

    注：如果要使用 [GPU 加速](../performance/gpu)，请选择用于导入 TensorFlow GPU 的第二个复选框。

3. 点击 `Finish` 开始导入过程。导入完成后，该工具将显示一个描述模型的界面，包括其输入和输出张量。

4. 要开始使用该模型，请选择 Kotlin 或 Java，然后将代码复制并粘贴到 **Sample Code** 部分中。

在 Android Studio 中，双击 `ml` 目录下的 TensorFlow Lite 模型，即可返回模型信息界面。有关使用 Android Studio 的模型绑定功能的更多信息，请参阅 Android Studio [发布说明](https://developer.android.com/studio/releases#4.1-tensor-flow-lite-models)。有关在 Android Studio 中使用模型绑定的概述，请参阅代码示例[说明](https://github.com/tensorflow/examples/blob/master/lite/examples/image_classification/android/README.md)。

## 使用 C 和 C++ 进行构建的工具

TensorFlow Lite 的 C 和 C++ 库主要面向使用 Android Native Development Kit (NDK) 构建应用的开发者。如果您使用 NDK 构建应用，有两种通过 C++ 使用 TFLite 的方式：

### TFLite C API

对于使用 NDK 的开发者，使用此 API 是*推荐*方式。下载 [MavenCentral 中托管的 TensorFlow Lite AAR](https://search.maven.org/artifact/org.tensorflow/tensorflow/tensorflow-lite) 文件，将其重命名为 `tensorflow-lite-*.zip`，然后解压缩该文件。您必须在 NDK 项目的 `headers/tensorflow/lite/` 和 `headers/tensorflow/lite/c/` 文件夹中包含四个头文件，并在 `jni/` 文件夹中包含相关的 `libtensorflowlite_jni.so` 动态库。

`c_api.h` 头文件包含有关使用 TFLite C API 的基本文档。

### TFLite C++ API

如果要通过 C++ API 使用 TFLite，您可以构建 C++ 共享库：

32 位 armeabi-v7a：

```sh
bazel build -c opt --config=android_arm //tensorflow/lite:libtensorflowlite.so
```

64 位 arm64-v8a：

```sh
bazel build -c opt --config=android_arm64 //tensorflow/lite:libtensorflowlite.so
```

目前没有直接的方式来提取所需的所有头文件，因此您必须从 TensorFlow 仓库中将所有头文件包括在 tensorflow/lite/ 目录下。此外，您还需要 FlatBuffers 和 Abseil 的头文件。
