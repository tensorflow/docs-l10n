# 集成物体检测器

物体检测器可以识别已知的一组物体中可能存在的物体，并提供它们在给定图像或视频流中的位置信息。物体检测器经过训练，可以检测多类物体的存在和位置。例如，可以用包含各种水果的图像，以及指定它们所代表水果类别（如苹果、香蕉或草莓）的*标签*和指定每个物体在图像中出现位置的数据来训练模型。有关物体检测器的详细信息，请参阅[物体检测简介](../../models/object_detection/overview.md)。

使用 Task Library `ObjectDetector` API 将自定义物体检测器或预训练的物体检测器部署到您的模型应用中。

## ObjectDetector API 的主要功能

- 输入图像处理，包括旋转、调整大小和色彩空间转换。

- 标注映射区域。

- 筛选结果的分数阈值。

- Top-k 检测结果。

- 标注允许列表和拒绝列表。

## 支持的物体检测器模型

以下模型保证可与 `ObjectDetector` API 兼容。

- TensorFlow Hub 上的[预训练物体检测模型](https://tfhub.dev/tensorflow/collections/lite/task-library/object-detector/1)。

- 由 [AutoML Vision Edge Object Detection](https://cloud.google.com/vision/automl/object-detection/docs) 创建的模型。

- 符合[模型兼容性要求](#model-compatibility-requirements)的自定义模型。

## 用 Java 运行推断

请参阅[物体检测参考应用](https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection/android/)，获得如何在 Android 应用中使用 `ObjectDetector` 的示例。

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
ObjectDetectorOptions options = ObjectDetectorOptions.builder().setMaxResults(1).build();
ObjectDetector objectDetector = ObjectDetector.createFromFileAndOptions(context, modelFile, options);

// Run inference
List<Detection> results = objectDetector.detect(image);
```

有关配置 `ObjectDetector` 的更多选项，请参阅[源代码和 Javadoc](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/java/src/java/org/tensorflow/lite/task/vision/detector/ObjectDetector.java)。

## 用 C++ 运行推断

注：我们正在改善 C++ Task Library 的可用性，如提供预先构建的二进制文件，并创建用户友好的工作流以从源代码进行构建。C++ API 可能会发生变化。

```c++
// Initialization
ObjectDetectorOptions options;
options.mutable_model_file_with_metadata()->set_file_name(model_file);
std::unique_ptr<ObjectDetector> object_detector = ObjectDetector::CreateFromOptions(options).value();

// Run inference
const DetectionResult result = object_detector->Detect(*frame_buffer).value();
```

有关配置 `ObjectDetector` 的更多选项，请参阅[源代码](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/vision/object_detector.h)。

## 结果示例

以下是来自 TensorFlow Hub 的 [ssd mobilenet v1](https://tfhub.dev/tensorflow/lite-model/ssd_mobilenet_v1/1/metadata/1) 的检测结果示例。

<img src="images/dogs.jpg" alt="dogs" width="50%">

```
Results:
 Detection #0 (red):
  Box: (x: 355, y: 133, w: 190, h: 206)
  Top-1 class:
   index       : 17
   score       : 0.73828
   class name  : dog
 Detection #1 (green):
  Box: (x: 103, y: 15, w: 138, h: 369)
  Top-1 class:
   index       : 17
   score       : 0.73047
   class name  : dog
```

将边界框呈现在输入图像上：

<img src="images/detection-output.png" alt="detection output" width="50%">

用您自己的模型和测试数据试用简单的 [ObjectDetector CLI 演示工具](https://github.com/tensorflow/tflite-support/tree/master/tensorflow_lite_support/examples/task/vision/desktop#object-detector)。

## 模型兼容性要求

`ObjectDetector` API 需要具有强制性 [TFLite 模型元数据](../../convert/metadata.md)的 TFLite 模型。

兼容的物体检测模型应满足以下要求：

- 输入图像张量：(kTfLiteUInt8/kTfLiteFloat32)

    - 图像输入大小为 `[batch x height x width x channels]`。
    - 不支持批量推断（`batch` 必须为 1）。
    - 仅支持 RGB 输入（`channels` 必须为 3）。
    - 如果类型为 kTfLiteFloat32，则必须将 NormalizationOptions 附加到元数据以进行输入归一化。

- 输出张量必须是 `DetectionPostProcess` 算子的 4 个输出，即：

    - 区域张量 (kTfLiteFloat32)

        - 大小为 `[1 x num_results x 4]` 的张量，内部数组表示边界框，形式为 [top，left，right，bottom]。
        - 必须将 BoundingBoxProperties 附加到元数据，且必须指定 `type=BOUNDARIES` 和 `coordinate_type = RATIO。

    - 类张量 (kTfLiteFloat32)

        - 大小为 `[1 x num_results]` 的张量，每个值代表一个类的整数索引。
        - 可选的（但推荐）标签映射可作为 AssociatedFile-s 进行附加，类型为 TENSOR_AXIS_LABELS，每行包含一个标签。第一个此类 AssociatedFile （如果有）用于填充结果的 `class_name` 字段。`display_name` 字段由其区域与创建时所用的 `ObjectDetectorOptions` 的 `display_names_locale` 字段（默认为“en”，即英语）相匹配的 AssociatedFile（如果有）填充。如果上述选项均不可用，将仅填充结果中的 `index` 字段。

    - 分数张量 (kTfLiteFloat32)

        - 大小为 `[1 x num_results]` 的张量，每个值代表检测到的物体的分数。

    - 检测张量的数量 (kTfLiteFloat32)

        - 整数 num_results 作为大小为 `[1]` 的张量。
