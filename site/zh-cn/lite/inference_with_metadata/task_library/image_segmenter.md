# 集成图像分割器

图像分割器会预测图像中的每个像素是否与某个类相关联。这与<a href="../../models/object_detection/overview.md">物体检测</a>（检测矩形区域中的物体）和<a href="../../models/image_classification/overview.md">图像分类</a>（对整体图像进行分类）不同。有关图像分割器的详细信息，请参阅[图像分割简介](../../models/segmentation/overview.md)。

使用 Task Library `ImageSegmenter` API 可将自定义图像分割器或预训练图像分割器部署到您的模型应用中。

## ImageSegmenter API 的主要功能

- 输入图像处理，包括旋转、调整大小和色彩空间转换。

- 标注映射区域。

- 两种输出类型，类别掩码和置信掩码。

- 用于显示目的的彩色标签。

## 支持的图像分割器模型

以下模型保证可与 `ImageSegmenter` API 兼容。

- [TensorFlow Hub 上的预训练图像分割模型](https://tfhub.dev/tensorflow/collections/lite/task-library/image-segmenter/1)。

- 符合[模型兼容性要求](#model-compatibility-requirements)的自定义模型。

## 用 Java 运行推断

请参阅[图像分割参考应用](https://github.com/tensorflow/examples/tree/master/lite/examples/image_segmentation/android/)，获得如何在 Android 应用中使用 `ImageSegmenter` 的示例。

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
ImageSegmenterOptions options = ImageSegmenterOptions.builder().setOutputType(OutputType.CONFIDENCE_MASK).build();
ImageSegmenter imageSegmenter = ImageSegmenter.createFromFileAndOptions(context, modelFile, options);

// Run inference
List<Segmentation> results = imageSegmenter.segment(image);
```

有关配置 `ImageSegmenter` 的更多选项，请参阅[源代码和 Javadoc](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/java/src/java/org/tensorflow/lite/task/vision/segmenter/ImageSegmenter.java)。

## 用 C++ 运行推断

注：我们正在改善 C++ Task Library 的可用性，如提供预先构建的二进制文件，并创建用户友好的工作流以从源代码进行构建。C++ API 可能会发生变化。

```c++
// Initialization
ImageSegmenterOptions options;
options.mutable_model_file_with_metadata()->set_file_name(model_file);
std::unique_ptr<ImageSegmenter> image_segmenter = ImageSegmenter::CreateFromOptions(options).value();

// Run inference
const SegmentationResult result = image_segmenter->Segment(*frame_buffer).value();
```

有关配置 `ImageSegmenter` 的更多选项，请参阅[源代码](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/vision/image_segmenter.h)。

## 结果示例

下面是 TensorFlow Hub 上的通用分割模型 [deeplab_v3](https://tfhub.dev/tensorflow/lite-model/deeplabv3/1/metadata/1) 的分割结果的示例。

<img src="images/plane.jpg" alt="plane" width="50%">

```
Color Legend:
 (r: 000, g: 000, b: 000):
  index       : 0
  class name  : background
 (r: 128, g: 000, b: 000):
  index       : 1
  class name  : aeroplane

# (omitting multiple lines for conciseness) ...

 (r: 128, g: 192, b: 000):
  index       : 19
  class name  : train
 (r: 000, g: 064, b: 128):
  index       : 20
  class name  : tv
Tip: use a color picker on the output PNG file to inspect the output mask with
this legend.
```

分割类别掩码应如下所示：

<img src="images/segmentation-output.png" alt="segmentation-output" width="30%">

用您自己的模型和测试数据试用简单的 [ImageSegmenter CLI 演示工具](https://github.com/tensorflow/tflite-support/tree/master/tensorflow_lite_support/examples/task/vision/desktop#image-segmenter)。

## 模型兼容性要求

`ImageSegmenter` API 需要具有强制性 [TFLite 模型元数据](../../convert/metadata.md)的 TFLite 模型。

- 输入图像张量 (kTfLiteUInt8/kTfLiteFloat32)

    - 图像输入大小为 `[batch x height x width x channels]`。
    - 不支持批量推断（`batch` 必须为 1）。
    - 仅支持 RGB 输入（`channels` 必须为 3）。
    - 如果类型为 kTfLiteFloat32，则必须将 NormalizationOptions 附加到元数据以进行输入归一化。

- 输出掩码张量 (kTfLiteUInt8/kTfLiteFloat32)

    - 张量大小为 `[batch x mask_height x mask_width x num_classes]`，其中 `batch` 必须为 1，`mask_width` 和 `mask_height` 为模型产生的分割掩码的维度，`num_classes` 为模型支持的类数。
    - 可选的（但推荐）标签映射可作为 AssociatedFile-s 进行附加，类型为 TENSOR_AXIS_LABELS，每行包含一个标签。第一个此类 AssociatedFile （如果有）用于填充结果的 `label` 字段（在 C++ 中，名称为 `class_name`）。`display_name` 字段由其区域与创建时所用的 `ImageSegmenterOptions` 的 `display_names_locale` 字段（默认为“en”，即英语）相匹配的 AssociatedFile（如果有）填充。如果上述选项均不可用，则仅填充结果中的 `index` 字段。
