# 精选 TensorFlow 算子

由于 TensorFlow Lite 内置算子库仅支持有限数量的 TensorFlow 算子，所以并非所有模型都可以转换。有关详细信息，请参阅[算子兼容性](ops_compatibility.md)。

为了允许进行转换，用户可以在 TensorFlow Lite 模型中启用[特定 TensorFlow 算子](op_select_allowlist.md)的使用。但是，运行带 TensorFlow 算子的 TensorFlow Lite 模型需要引入核心 TensorFlow 运行时，这会增加 TensorFlow Lite 解释器的二进制文件大小。对于 Android，您可以通过有选择地仅构建所需 Tensorflow 算子来避免这种情况。有关详情，请参阅[缩减二进制文件大小](../guide/reduce_binary_size.md)。

本文档概述了如何在您选择的平台上[转换](#convert_a_model)和[运行](#run_inference)包含 TensorFlow 算子的 TensorFlow Lite 模型。此外，它还探讨了[性能和大小指标](#metrics)以及[已知的限制](#known_limitations)。

## 转换模型

以下示例显示了如何使用精选 TensorFlow 算子生成 TensorFlow Lite 模型。

```python
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
  tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
]
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)
```

## 运行推断

使用已转换并支持精选 TensorFlow 算子的 TensorFlow Lite 模型时，客户端还必须使用包含必要的 TensorFlow 算子库的 TensorFlow Lite 运行时。

### Android AAR

要缩减二进制文件的大小，请按照[下一部分](#building-the-android-aar)中的指导构建您自己的自定义 AAR 文件。如果二进制文件大小不是特别大的问题，我们建议使用预构建的 [AAR，其中 TensorFlow 算子托管在 MavenCentral 中](https://search.maven.org/artifact/org.tensorflow/tensorflow-lite-select-tf-ops)。

您可以在 `build.gradle` 依赖项中进行指定，方法是将其与标准 TensorFlow Lite AAR 一起添加，如下所示：

```build
dependencies {
    implementation 'org.tensorflow:tensorflow-lite:0.0.0-nightly-SNAPSHOT'
    // This dependency adds the necessary TF op support.
    implementation 'org.tensorflow:tensorflow-lite-select-tf-ops:0.0.0-nightly-SNAPSHOT'
}
```

要使用 Nightly 快照，请确保您已添加 [Sonatype 快照存储库](../android/lite_build.md#use_nightly_snapshots)。

在添加依赖项后，用于处理计算图的 TensorFlow 算子的所需委托应对需要它们的计算图自动安装。

*注*：TensorFlow 算子依赖项相对较大，因此，您可能需要通过设置 `abiFilters` 在 `.gradle` 文件中滤除不必要的 x86 ABI。

```build
android {
    defaultConfig {
        ndk {
            abiFilters 'armeabi-v7a', 'arm64-v8a'
        }
    }
}
```

#### 构建 Android AAR

为了缩减二进制文件大小或其他高级情况，您也可以手动构建库。假设 [TensorFlow Lite 构建环境正常运行](../android/quickstart.md)，请使用以下 TensorFlow 算子构建 Android AAR：

```sh
sh tensorflow/lite/tools/build_aar.sh \
  --input_models=/a/b/model_one.tflite,/c/d/model_two.tflite \
  --target_archs=x86,x86_64,arm64-v8a,armeabi-v7a
```

这将为 TensorFlow Lite 内置算子和自定义算子生成 AAR 文件 `bazel-bin/tmp/tensorflow-lite.aar`；并为 TensorFlow 算子生成 AAR 文件 `bazel-bin/tmp/tensorflow-lite-select-tf-ops.aar`。如果没有正在运行的构建环境，您也可以[使用 Docker 构建上述文件](../guide/reduce_binary_size.md#selectively_build_tensorflow_lite_with_docker)。

在 Docker 中，您可以将 AAR 文件直接导入项目，或者将自定义 AAR 文件发布到您的本地 Maven 仓库：

```sh
mvn install:install-file \
  -Dfile=bazel-bin/tmp/tensorflow-lite.aar \
  -DgroupId=org.tensorflow \
  -DartifactId=tensorflow-lite -Dversion=0.1.100 -Dpackaging=aar
mvn install:install-file \
  -Dfile=bazel-bin/tmp/tensorflow-lite-select-tf-ops.aar \
  -DgroupId=org.tensorflow \
  -DartifactId=tensorflow-lite-select-tf-ops -Dversion=0.1.100 -Dpackaging=aar
```

最后，请确保在应用的 `build.gradle` 中添加了 `mavenLocal()` 依赖项，并将标准 TensorFlow Lite 依赖项替换为支持精选 TensorFlow 算子的依赖项：

```build
allprojects {
    repositories {
        mavenCentral()
        maven {  // Only for snapshot artifacts
            name 'ossrh-snapshot'
            url 'https://oss.sonatype.org/content/repositories/snapshots'
        }
        mavenLocal()
    }
}

dependencies {
    implementation 'org.tensorflow:tensorflow-lite:0.1.100'
    implementation 'org.tensorflow:tensorflow-lite-select-tf-ops:0.1.100'
}
```

### iOS

#### 使用 CocoaPods

TensorFlow Lite 提供适用于 `arm64` 的 Nightly 预构建精选 TF 算子 CocoaPods，您可以将其与 `TensorFlowLiteSwift` 或 `TensorFlowLiteObjC` CocoaPods 一起使用。

*注*：如果需要在 `x86_64` 模拟器中使用精选 TF 算子，则可以自己构建精选算子框架。请参阅[使用 Bazel + Xcode](#using_bazel_xcode) 部分了解详细信息。

```ruby
# In your Podfile target:
  pod 'TensorFlowLiteSwift'   # or 'TensorFlowLiteObjC'
  pod 'TensorFlowLiteSelectTfOps', '~> 0.0.1-nightly'
```

运行 `pod install` 后，您需要提供一个附加的链接器标志，以将精选 TF 算子框架强制加载到您的项目中。在您的 Xcode 项目中，转到 `Build Settings` -&gt; `Other Linker Flags`，然后添加：

对于 &gt;= 2.9.0 的版本：

```text
-force_load $(SRCROOT)/Pods/TensorFlowLiteSelectTfOps/Frameworks/TensorFlowLiteSelectTfOps.xcframework/ios-arm64/TensorFlowLiteSelectTfOps.framework/TensorFlowLiteSelectTfOps
```

对于 &lt; 2.9.0 的版本：

```text
-force_load $(SRCROOT)/Pods/TensorFlowLiteSelectTfOps/Frameworks/TensorFlowLiteSelectTfOps.framework/TensorFlowLiteSelectTfOps
```

然后，您应该能够在您的 iOS 应用中运行任何使用 `SELECT_TF_OPS` 转换的模型。例如，您可以修改[图像分类 iOS 应用](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/ios)来测试精选 TF 算子功能。

- 将模型文件替换为已启用 `SELECT_TF_OPS` 的模型文件。
- 按照说明将 `TensorFlowLiteSelectTfOps` 依赖项添加到 `Podfile` 中。
- 按照上述方法添加附加的链接器标志。
- 运行示例应用，然后查看模型是否正常工作。

#### 使用 Bazel + Xcode

可以使用 Bazel 构建带有适用于 iOS 的精选 TensorFlow 算子的 TensorFlow Lite。首先，按照 [iOS 构建说明](build_ios.md)正确配置您的 Bazel 工作区和 `.bazelrc` 文件。

在启用 iOS 支持的情况下配置工作区后，您可以使用以下命令构建精选 TF 算子加载项框架，该框架可以添加到常规的 `TensorFlowLiteC.framework` 之上。请注意，不能为 `i386` 架构构建精选 TF 算子框架，因此您需要显式提供除 `i386` 之外的目标架构列表。

```sh
bazel build -c opt --config=ios --ios_multi_cpus=arm64,x86_64 \
  //tensorflow/lite/ios:TensorFlowLiteSelectTfOps_framework
```

这将在 `bazel-bin/tensorflow/lite/ios/` 目录下生成框架。您可以按照 iOS 构建指南的 [Xcode 项目设置](./build_ios.md#modify_xcode_project_settings_directly)部分中所述的类似步骤，将此新框架添加到您的 Xcode 项目中。

将框架添加到您的应用项目后，应在您的应用项目中指定一个附加的链接器标志，以强制加载精选 TF 算子框架。在您的 Xcode 项目中，转到 `Build Settings` -&gt; `Other Linker Flags`，然后添加：

```text
-force_load <path/to/your/TensorFlowLiteSelectTfOps.framework/TensorFlowLiteSelectTfOps>
```

### C/C++

如果您使用 Bazel 或 [CMake](https://www.tensorflow.org/lite/guide/build_cmake) 构建 TensorFlow Lite 解释器，则可以通过链接 TensorFlow Lite Flex 委托共享库来启用 Flex 委托。您可以使用 Bazel 构建它，命令如下。

```
bazel build -c opt --config=monolithic tensorflow/lite/delegates/flex:tensorflowlite_flex
```

此命令会在 `bazel-bin/tensorflow/lite/delegates/flex` 中生成以下共享库。

平台 | 库名称
--- | ---
Linux | libtensorflowlite_flex.so
macOS | libtensorflowlite_flex.dylib
Windows | tensorflowlite_flex.dll

请注意，只要链接了共享库，在运行时创建解释器时，就会自动安装必要的 `TfLiteDelegate`。不需要像其他委托类型通常要求的那样显式安装委托实例。

**注**：此功能从 2.7 版本开始提供。

### Python

包含精选 TensorFlow 算子的 TensorFlow Lite 将自动与 [TensorFlow pip 软件包](https://www.tensorflow.org/install/pip)一起安装。此外，您也可以选择仅安装 [TensorFlow Lite Interpreter pip 软件包](https://www.tensorflow.org/lite/guide/python#install_just_the_tensorflow_lite_interpreter)。

注：在自 2.3（适用于 Linux）和 2.4（适用于其他环境）起的 TensorFlow pip 软件包版本中，可以使用包含精选 TensorFlow 算子的 TensorFlow Lite。

## 各种指标

### 性能

混合使用内置和精选 TensorFlow 算子时，所有相同的 TensorFlow Lite 优化和优化后的内置算子都将可用并可与转换后的模型一起使用。

下表给出了在 Pixel 2 的 MobileNet 上运行推断所需的平均时间。列出的时间是运行 100 次的平均值。这些目标是使用以下标志为 Android 构建的：`--config=android_arm64 -c opt`。

构建 | 时间（毫秒）
--- | ---
仅内置算子 (`TFLITE_BUILTIN`) | 260.7
仅使用 TF 算子 (`SELECT_TF_OPS`) | 264.5

### 二进制文件大小

下表给出了每个构建的 TensorFlow Lite 的二进制文件大小。这些目标是使用 `--config=android_arm -c opt` 为 Android 构建的。

构建 | C++ 二进制文件大小 | Android APK 大小
--- | --- | ---
仅内置算子 | 796 KB | 561 KB
内置算子 + TF 算子 | 23.0 MB | 8.0 MB
内置算子 + TF 算子 (1) | 4.1 MB | 1.8 MB

(1) 这些库是为包含 8 个 TFLite 内置算子和 3 个 TensorFlow 算子的 [i3d-kinetics-400 模型](https://tfhub.dev/deepmind/i3d-kinetics-400/1)选择性构建的。有关详情，请参阅[缩减 TensorFlow Lite 二进制文件大小](../guide/reduce_binary_size.md)部分。

## 已知限制

- 不支持的类型：某些 TensorFlow 算子可能不支持 TensorFlow 中通常可用的全套输入​​/输出类型。

## 更新

- 版本 2.6
    - 改进了对基于 GraphDef 特性的算子和 HashTable 资源初始化的支持。
- 版本 2.5
    - 您可以应用称为[后训练量化](../performance/post_training_quantization.md)的优化
- 版本 2.4
    - 改进了与硬件加速委托的兼容性
