# 集成图像分类器

图像分类是机器学习的一种常见用途，用于识别图像所代表的内容。例如，我们可能想知道一张给定的图片中出现了哪种类型的动物。预测图像所代表的内容的任务称为*图像分类*。图像分类器经过训练，可以识别各种类别的图像。例如，可以训练一个模型来识别代表三种不同类型动物的照片：兔子、仓鼠和狗。有关图像分类器的详细信息，请参阅[图像分类简介](../../models/image_classification/overview.md)。

使用 Task Library `ImageClassifier` API 可将自定义图像分类器或预训练图像分类器部署到您的模型应用中。

## ImageClassifier API 的主要功能

- 输入图像处理，包括旋转、调整大小和色彩空间转换。

- 输入图像的感兴趣区域。

- 标注映射区域。

- 筛选结果的得分阈值。

- Top-k 分类结果。

- 标注允许列表和拒绝列表。

## 支持的图像分类器模型

以下模型保证可与 `ImageClassifier` API 兼容。

- 由[适用于图像分类的 TensorFlow Lite Model Maker](https://www.tensorflow.org/lite/tutorials/model_maker_image_classification) 创建的模型。

- [TensorFlow Lite 托管模型中的预训练图像分类模型](https://www.tensorflow.org/lite/guide/hosted_models#image_classification)。

- [TensorFlow Hub 上的预训练图像分类模型](https://tfhub.dev/tensorflow/collections/lite/task-library/image-classifier/1)。

- 由 [AutoML Vision Edge 图像分类](https://cloud.google.com/vision/automl/docs/edge-quickstart)创建的模型。

- 符合[模型兼容性要求](#model-compatibility-requirements)的自定义模型。

## 用 Java 运行推断

请参阅[图像分类参考应用](https://github.com/tensorflow/examples/blob/master/lite/examples/image_classification/android/EXPLORE_THE_CODE.md)，获得如何在 Android 应用中使用 `ImageClassifier` 的示例。

### 步骤 1：导入 Gradle 依赖项和其他设置

将 `.tflite` 模型文件复制到将要运行模型的 Android 模块的资源目录下。指定不压缩该文件，并将 TensorFlow Lite 库添加到模块的 `build.gradle` 文件中。

```java
android {
    // Other settings

    // Specify tflite file should not be compressed for the app apk
    aaptOptions {
        noCompress "tflite"
    }

}

dependencies {
    // Other dependencies

    // Import the Task Vision Library dependency
    implementation 'org.tensorflow:tensorflow-lite-task-vision:0.1.0'
}
```

### 步骤 2：使用模型

```java
// Initialization
ImageClassifierOptions options = ImageClassifierOptions.builder().setMaxResults(1).build();
ImageClassifier imageClassifier = ImageClassifier.createFromFileAndOptions(context, modelFile, options);

// Run inference
List<Classifications> results = imageClassifier.classify(image);
```

有关配置 `ImageClassifier` 的更多选项，请参阅[源代码和 Javadoc](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/java/src/java/org/tensorflow/lite/task/vision/classifier/ImageClassifier.java)。

## 用 C++ 运行推断

注：我们正在改善 C++ Task Library 的可用性，如提供预先构建的二进制文件，并创建用户友好的工作流以从源代码进行构建。C++ API 可能会发生变化。

```c++
// Initialization
ImageClassifierOptions options;
options.mutable_model_file_with_metadata()->set_file_name(model_file);
std::unique_ptr<ImageClassifier> image_classifier = ImageClassifier::CreateFromOptions(options).value();

// Run inference
const ClassificationResult result = image_classifier->Classify(*frame_buffer).value();
```

有关配置 `ImageClassifier` 的更多选项，请参阅<a>源代码</a>。

## 结果示例

下面是一个[鸟类分类器](https://tfhub.dev/google/lite-model/aiy/vision/classifier/birds_V1/3)的分类结果的示例。

<img src="images/sparrow.jpg" alt="sparrow" width="50%">

```
Results:
  Rank #0:
   index       : 671
   score       : 0.91406
   class name  : /m/01bwb9
   display name: Passer domesticus
  Rank #1:
   index       : 670
   score       : 0.00391
   class name  : /m/01bwbt
   display name: Passer montanus
  Rank #2:
   index       : 495
   score       : 0.00391
   class name  : /m/0bwm6m
   display name: Passer italiae
```

用您自己的模型和测试数据试用简单的 [ImageClassifier CLI 演示工具](https://github.com/tensorflow/tflite-support/tree/master/tensorflow_lite_support/examples/task/vision/desktop#image-classifier)。

## 模型兼容性要求

`ImageClassifier` API 需要具有强制性 [TFLite 模型元数据](../../convert/metadata.md)的 TFLite 模型。

兼容的图像分类器模型应满足以下要求：

- 输入图像张量 (kTfLiteUInt8/kTfLiteFloat32)

    - 图像输入大小为 `[batch x height x width x channels]`。
    - 不支持批量推断（`batch` 必须为 1）。
    - 仅支持 RGB 输入（`channels` 必须为 3）。
    - 如果类型为 kTfLiteFloat32，则必须将 NormalizationOptions 附加到元数据以进行输入归一化。

- 输出分数张量 (kTfLiteUInt8/kTfLiteFloat32)

    - 具有 `N` 个类和 2 或 4 个维度，即 `[1 x N]` 或 `[1 x 1 x 1 x N]`
    - 可选（但推荐）标签映射可作为 AssociatedFile-s，类型为 TENSOR_AXIS_LABELS，每行包含一个标签。第一个此类 AssociatedFile（如果有）用于填充结果的 `label` 字段（在 C++ 中命名为 `class_name`）。`display_name` 字段由其区域与创建时所用的 `ImageClassifierOptions` 的 `display_names_locale` 字段（默认为“en”，即英语）相匹配的 AssociatedFile（如果有）填充。如果上述选项均不可用，将仅填充结果中的 `index` 字段。
