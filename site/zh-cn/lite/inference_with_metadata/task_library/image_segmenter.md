# 集成图像分割器

图像分割器会预测图像中的每个像素是否与某个类相关联。这与<a href="../../examples/object_detection/overview">目标检测</a>（检测矩形区域中的目标）和<a href="../../examples/image_classification/overview">图像分类</a>（对整体图像进行分类）不同。请参阅[图像分割概述](../../examples/segmentation/overview)，了解有关图像分割器的详细信息。

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

    // Import the Task Vision Library dependency (NNAPI is included)
    implementation 'org.tensorflow:tensorflow-lite-task-vision'
    // Import the GPU delegate plugin Library for GPU inference
    implementation 'org.tensorflow:tensorflow-lite-gpu-delegate-plugin'
}
```

注：从 Android Gradle 插件的 4.1 版开始，默认情况下，.tflite 将被添加到 noCompress 列表中，不再需要上面的 aaptOptions。

### 步骤 2：使用模型

```java
// Initialization
ImageSegmenterOptions options =
    ImageSegmenterOptions.builder()
        .setBaseOptions(BaseOptions.builder().useGpu().build())
        .setOutputType(OutputType.CONFIDENCE_MASK)
        .build();
ImageSegmenter imageSegmenter =
    ImageSegmenter.createFromFileAndOptions(context, modelFile, options);

// Run inference
List<Segmentation> results = imageSegmenter.segment(image);
```

有关配置 `ImageSegmenter` 的更多选项，请参阅[源代码和 Javadoc](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/java/src/java/org/tensorflow/lite/task/vision/segmenter/ImageSegmenter.java)。

## 在 iOS 中运行推断

### 第 1 步：安装依赖项

Task Library 支持使用 CocoaPods 进行安装。请确保您的系统上已安装 CocoaPods。有关说明，请参阅 [CocoaPods 安装指南](https://guides.cocoapods.org/using/getting-started.html#getting-started)。

有关向 Xcode 项目添加 Pod 的详细信息，请参阅 [CocoaPods 指南](https://guides.cocoapods.org/using/using-cocoapods.html)。

在 Podfile 中添加 `TensorFlowLiteTaskVision`。

```
target 'MyAppWithTaskAPI' do
  use_frameworks!
  pod 'TensorFlowLiteTaskVision'
end
```

请确保您的应用捆绑包中存在用于推断的 `.tflite` 模型。

### 第 2 步：使用模型

#### Swift

```swift
// Imports
import TensorFlowLiteTaskVision

// Initialization
guard let modelPath = Bundle.main.path(forResource: "deeplabv3",
                                            ofType: "tflite") else { return }

let options = ImageSegmenterOptions(modelPath: modelPath)

// Configure any additional options:
// options.outputType = OutputType.confidenceMasks

let segmenter = try ImageSegmenter.segmenter(options: options)

// Convert the input image to MLImage.
// There are other sources for MLImage. For more details, please see:
// https://developers.google.com/ml-kit/reference/ios/mlimage/api/reference/Classes/GMLImage
guard let image = UIImage (named: "plane.jpg"), let mlImage = MLImage(image: image) else { return }

// Run inference
let segmentationResult = try segmenter.segment(mlImage: mlImage)
```

#### Objective C

```objc
// Imports
#import <TensorFlowLiteTaskVision/TFLTaskVision.h>

// Initialization
NSString *modelPath = [[NSBundle mainBundle] pathForResource:@"deeplabv3" ofType:@"tflite"];

TFLImageSegmenterOptions *options =
    [[TFLImageSegmenterOptions alloc] initWithModelPath:modelPath];

// Configure any additional options:
// options.outputType = TFLOutputTypeConfidenceMasks;

TFLImageSegmenter *segmenter = [TFLImageSegmenter imageSegmenterWithOptions:options
                                                                      error:nil];

// Convert the input image to MLImage.
UIImage *image = [UIImage imageNamed:@"plane.jpg"];

// There are other sources for GMLImage. For more details, please see:
// https://developers.google.com/ml-kit/reference/ios/mlimage/api/reference/Classes/GMLImage
GMLImage *gmlImage = [[GMLImage alloc] initWithImage:image];

// Run inference
TFLSegmentationResult *segmentationResult =
    [segmenter segmentWithGMLImage:gmlImage error:nil];
```

请参阅[源代码](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/ios/task/vision/sources/TFLImageSegmenter.h)，了解有关配置 `TFLImageSegmenter` 的更多选项。

## 用 Python 运行推断

### 第 1 步：安装 pip 软件包

```
pip install tflite-support
```

### 第 2 步：使用模型

```python
# Imports
from tflite_support.task import vision
from tflite_support.task import core
from tflite_support.task import processor

# Initialization
base_options = core.BaseOptions(file_name=model_path)
segmentation_options = processor.SegmentationOptions(
    output_type=processor.SegmentationOptions.OutputType.CATEGORY_MASK)
options = vision.ImageSegmenterOptions(base_options=base_options, segmentation_options=segmentation_options)
segmenter = vision.ImageSegmenter.create_from_options(options)

# Alternatively, you can create an image segmenter in the following manner:
# segmenter = vision.ImageSegmenter.create_from_file(model_path)

# Run inference
image_file = vision.TensorImage.create_from_file(image_path)
segmentation_result = segmenter.segment(image_file)
```

请参阅[源代码](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/python/task/vision/image_segmenter.py)，了解有关配置 `ImageSegmenter` 的更多选项。

## 用 C++ 运行推断

```c++
// Initialization
ImageSegmenterOptions options;
options.mutable_base_options()->mutable_model_file()->set_file_name(model_path);
std::unique_ptr<ImageSegmenter> image_segmenter = ImageSegmenter::CreateFromOptions(options).value();

// Create input frame_buffer from your inputs, `image_data` and `image_dimension`.
// See more information here: tensorflow_lite_support/cc/task/vision/utils/frame_buffer_common_utils.h
std::unique_ptr<FrameBuffer> frame_buffer = CreateFromRgbRawBuffer(
      image_data, image_dimension);

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

`ImageSegmenter` API 需要具有强制性 [TFLite Model Metadata](../../models/convert/metadata) 的 TFLite 模型。请参阅使用 [TensorFlow Lite Metadata Writer API](../../models/convert/metadata_writer_tutorial.ipynb#image_segmenters) 为图像分割器创建元数据的示例。

- 输入图像张量 (kTfLiteUInt8/kTfLiteFloat32)

    - 图像输入大小为 `[batch x height x width x channels]`。
    - 不支持批量推断（`batch` 必须为 1）。
    - 仅支持 RGB 输入（`channels` 必须为 3）。
    - 如果类型为 kTfLiteFloat32，则必须将 NormalizationOptions 附加到元数据以进行输入归一化。

- 输出掩码张量 (kTfLiteUInt8/kTfLiteFloat32)

    - 张量大小为 `[batch x mask_height x mask_width x num_classes]`，其中 `batch` 必须为 1，`mask_width` 和 `mask_height` 为模型产生的分割掩码的维度，`num_classes` 为模型支持的类数。
    - 可选的（但推荐）标签映射可作为 AssociatedFile-s 进行附加，类型为 TENSOR_AXIS_LABELS，每行包含一个标签。第一个此类 AssociatedFile （如果有）用于填充结果的 `label` 字段（在 C++ 中，名称为 `class_name`）。`display_name` 字段由其区域与创建时所用的 `ImageSegmenterOptions` 的 `display_names_locale` 字段（默认为“en”，即英语）相匹配的 AssociatedFile（如果有）填充。如果上述选项均不可用，则仅填充结果中的 `index` 字段。
