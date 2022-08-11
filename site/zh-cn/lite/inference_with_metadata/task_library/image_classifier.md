# 集成图像分类器

图像分类是机器学习的一种常见用途，用于识别图像所代表的内容。例如，我们可能想知道一张给定的图片中出现了哪种类型的动物。预测图像所代表内容的任务称为*图像分类*。图像分类器经过训练，可以识别各种类别的图像。例如，可以训练一个模型来识别代表三种不同类型动物的照片：兔子、仓鼠和狗。请参阅[图像分类概述](../../examples/image_classification/overview)，了解有关图像分类器的详细信息。

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

- 由[适用于图像分类的 TensorFlow Lite Model Maker](https://www.tensorflow.org/lite/models/modify/model_maker/image_classification) 创建的模型。

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

    // Import the Task Vision Library dependency (NNAPI is included)
    implementation 'org.tensorflow:tensorflow-lite-task-vision'
    // Import the GPU delegate plugin Library for GPU inference
    implementation 'org.tensorflow:tensorflow-lite-gpu-delegate-plugin'
}
```

### 步骤 2：使用模型

```java
// Initialization
ImageClassifierOptions options =
    ImageClassifierOptions.builder()
        .setBaseOptions(BaseOptions.builder().useGpu().build())
        .setMaxResults(1)
        .build();
ImageClassifier imageClassifier =
    ImageClassifier.createFromFileAndOptions(
        context, modelFile, options);

// Run inference
List<Classifications> results = imageClassifier.classify(image);
```

有关配置 `ImageClassifier` 的更多选项，请参阅[源代码和 Javadoc](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/java/src/java/org/tensorflow/lite/task/vision/classifier/ImageClassifier.java)。

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
guard let modelPath = Bundle.main.path(forResource: "birds_V1",
                                            ofType: "tflite") else { return }

let options = ImageClassifierOptions(modelPath: modelPath)

// Configure any additional options:
// options.classificationOptions.maxResults = 3

let classifier = try ImageClassifier.classifier(options: options)

// Convert the input image to MLImage.
// There are other sources for MLImage. For more details, please see:
// https://developers.google.com/ml-kit/reference/ios/mlimage/api/reference/Classes/GMLImage
guard let image = UIImage (named: "sparrow.jpg"), let mlImage = MLImage(image: image) else { return }

// Run inference
let classificationResults = try classifier.classify(mlImage: mlImage)
```

#### Objective C

```objc
// Imports
#import <TensorFlowLiteTaskVision/TFLTaskVision.h>

// Initialization
NSString *modelPath = [[NSBundle mainBundle] pathForResource:@"birds_V1" ofType:@"tflite"];

TFLImageClassifierOptions *options =
    [[TFLImageClassifierOptions alloc] initWithModelPath:modelPath];

// Configure any additional options:
// options.classificationOptions.maxResults = 3;

TFLImageClassifier *classifier = [TFLImageClassifier imageClassifierWithOptions:options
                                                                          error:nil];

// Convert the input image to MLImage.
UIImage *image = [UIImage imageNamed:@"sparrow.jpg"];

// There are other sources for GMLImage. For more details, please see:
// https://developers.google.com/ml-kit/reference/ios/mlimage/api/reference/Classes/GMLImage
GMLImage *gmlImage = [[GMLImage alloc] initWithImage:image];

// Run inference
TFLClassificationResult *classificationResult =
    [classifier classifyWithGMLImage:gmlImage error:nil];
```

请参阅[源代码](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/ios/task/vision/sources/TFLImageClassifier.h)，了解有关配置 `TFLImageClassifier` 的更多选项。

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
classification_options = processor.ClassificationOptions(max_results=2)
options = vision.ImageClassifierOptions(base_options=base_options, classification_options=classification_options)
classifier = vision.ImageClassifier.create_from_options(options)

# Alternatively, you can create an image classifier in the following manner:
# classifier = vision.ImageClassifier.create_from_file(model_path)

# Run inference
image = vision.TensorImage.create_from_file(image_path)
classification_result = classifier.classify(image)
```

请参阅[源代码](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/python/task/vision/image_classifier.py)，了解有关配置 `ImageClassifier` 的更多选项。

## 用 C++ 运行推断

```c++
// Initialization
ImageClassifierOptions options;
options.mutable_base_options()->mutable_model_file()->set_file_name(model_path);
std::unique_ptr<ImageClassifier> image_classifier = ImageClassifier::CreateFromOptions(options).value();

// Create input frame_buffer from your inputs, `image_data` and `image_dimension`.
// See more information here: tensorflow_lite_support/cc/task/vision/utils/frame_buffer_common_utils.h

std::unique_ptr<FrameBuffer> frame_buffer = CreateFromRgbRawBuffer(
      image_data, image_dimension);

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

`ImageClassifier` API 需要具有强制性 [TFLite Model Metadata](../../models/convert/metadata) 的 TFLite 模型。请参阅使用 [TensorFlow Lite Metadata Writer API](../../models/convert/metadata_writer_tutorial.ipynb#image_classifiers) 为图像分类器创建元数据的示例。

兼容的图像分类器模型应满足以下要求：

- 输入图像张量 (kTfLiteUInt8/kTfLiteFloat32)

    - 图像输入大小为 `[batch x height x width x channels]`。
    - 不支持批量推断（`batch` 必须为 1）。
    - 仅支持 RGB 输入（`channels` 必须为 3）。
    - 如果类型为 kTfLiteFloat32，则必须将 NormalizationOptions 附加到元数据以进行输入归一化。

- 输出分数张量 (kTfLiteUInt8/kTfLiteFloat32)

    - 具有 `N` 个类和 2 或 4 个维度，即 `[1 x N]` 或 `[1 x 1 x 1 x N]`
    - 可选（但推荐）标签映射可作为 AssociatedFile-s，类型为 TENSOR_AXIS_LABELS，每行包含一个标签。请参阅[示例标签文件](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/metadata/python/tests/testdata/image_classifier/labels.txt)。第一个此类 AssociatedFile（如果有）用于填充结果的 <code>label</code> 字段（在 C++ 中命名为 `class_name`）。`display_name` 字段由其区域与创建时所用的 `ImageClassifierOptions` 的 `display_names_locale` 字段（默认为“en”，即英语）相匹配的 AssociatedFile（如果有）填充。如果上述选项均不可用，将仅填充结果中的 `index` 字段。
