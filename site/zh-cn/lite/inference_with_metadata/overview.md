# 使用元数据进行 TensorFlow Lite 推断

[用元数据来推断模型](../models/convert/metadata.md)可以简单到只需几行代码。TensorFlow Lite 元数据包含了有关模型功能以及使用方法的丰富描述。它可以授权代码生成器自动生成推断代码，例如使用 [Android Studio 机器学习绑定功能](codegen.md#mlbinding)或 [TensorFlow Lite Android 代码生成器](codegen.md#codegen)。它还可以用来配置自定义推断流水线。

## 工具和库

TensorFlow Lite 提供了多种工具和库来满足不同层次的部署要求，如下所示：

### 使用 Android 代码生成器生成模型接口

有两种方式可以为带有元数据的 TensorFlow Lite 模型自动生成必要的 Android 封装容器代码：

1. Android Studio 中的 [Android Studio 机器学习模型绑定](codegen.md#mlbinding)工具可通过图形界面导入 TensorFlow Lite 模型。Android Studio 将自动为项目配置设置，并根据模型元数据生成封装容器类。

2. [TensorFlow Lite Code Generator](codegen.md#codegen) 是一个根据元数据自动生成模型接口的可执行文件。目前它支持 Android 与 Java。封装容器代码消除了直接与 `ByteBuffer` 交互的需要。相反，开发人员可以使用 `Bitmap` 和 `Rect` 等类型化对象与 TensorFlow Lite 模型进行交互。Android Studio 用户也可以通过 [Android Studio 机器学习绑定](codegen.md#mlbinding)来访问 codegen 功能。

### 利用 TensorFlow Lite Task Library 中的开箱即用的 API

[TensorFlow Lite Task Library](task_library/overview.md) 为热门的机器学习任务（如图像分类、问答等）提供了经过优化的现成的模型接口。模型接口专为每个任务而设计，以实现最佳性能和可用性。Task Library 可跨平台工作，支持 Java、C++ 和 Swift。

### 使用 TensorFlow Lite Support Library 构建自定义推断流水线

[TensorFlow Lite Support Library](lite_support.md) 是一个跨平台的库，可帮助自定义模型接口和构建推断流水线。它包含各种实用工具方法和数据结构，以执行前/后处理和数据转换。它还设计为与 TF.Image 和 TF.Text 等 TensorFlow 模块的行为相匹配，确保了从训练到推断的一致性。

## 探索带有元数据的预训练模型

浏览 [TensorFlow Lite 托管模型](https://www.tensorflow.org/lite/guide/hosted_models)和 [TensorFlow Hub](https://tfhub.dev/s?deployment-format=lite)，下载带有元数据的预训练模型，用于视觉和文本任务。另请参阅[可视化元数据](../models/convert/metadata.md#visualize-the-metadata)的不同选项。

## TensorFlow Lite Support GitHub 仓库

请访问 [TensorFlow Lite Support GitHub 仓库](https://github.com/tensorflow/tflite-support)获取更多示例和源代码，并通过创建[新的 GitHub 议题](https://github.com/tensorflow/tflite-support/issues/new)让我们了解您的反馈。
