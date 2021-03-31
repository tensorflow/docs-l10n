# 集成自然语言分类器

Task Library 的 `NLClassifier` API 可以将输入的文本分为不同类别，它是一个通用且可配置的 API，能够处理大多数文本分类模型。

## NLClassifier API 的主要功能

- 将单个字符串作为输入，对该字符串进行分类，并输出 &lt;Label, Score&gt; 对作为分类结果。

- 可选的输入文本的正则表达式标记化。

- 可配置以适应不同的分类模型。

## 支持的 NLClassifier 模型

以下模型保证可与 `NLClassifier` API 兼容。

- <a href="../../models/text_classification/overview.md">电影评论情感分类</a>模型。

- 由[适用于文本分类的 TensorFlow Lite Model Maker](https://www.tensorflow.org/lite/tutorials/model_maker_text_classification) 创建的具有 `average_word_vec` 规范的模型。

- 符合[模型兼容性要求](#model-compatibility-requirements)的自定义模型。

## 用 Java 运行推断

请参阅[文本分类参考应用](https://github.com/tensorflow/examples/blob/master/lite/examples/text_classification/android/lib_task_api/src/main/java/org/tensorflow/lite/examples/textclassification/client/TextClassificationClient.java)，获得如何在 Android 应用中使用 `NLClassifier` 的示例。

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

    // Import the Task Text Library dependency
    implementation 'org.tensorflow:tensorflow-lite-task-text:0.1.0'
}
```

### 步骤 2：使用 API 运行推断

```java
// Initialization, use NLClassifierOptions to configure input and output tensors
NLClassifierOptions options = NLClassifierOptions.builder().setInputTensorName(INPUT_TENSOR_NAME).setOutputScoreTensorName(OUTPUT_SCORE_TENSOR_NAME).build();
NLClassifier classifier = NLClassifier.createFromFileAndOptions(context, modelFile, options);

// Run inference
List<Category> results = classifier.classify(input);
```

有关配置 `NLClassifier` 的更多选项，请参阅[源代码](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/java/src/java/org/tensorflow/lite/task/text/nlclassifier/NLClassifier.java)。

## 用 Swift 运行推断

### 步骤 1：导入 CocoaPods

在 Podfile 中添加 TensorFlowLiteTaskText Pod。

```
target 'MySwiftAppWithTaskAPI' do
  use_frameworks!
  pod 'TensorFlowLiteTaskText', '~> 0.0.1-nightly'
end
```

### 步骤 2：使用 API 运行推断

```swift
// Initialization
var modelOptions:TFLNLClassifierOptions = TFLNLClassifierOptions()
modelOptions.inputTensorName = inputTensorName
modelOptions.outputScoreTensorName = outputScoreTensorName
let nlClassifier = TFLNLClassifier.nlClassifier(
      modelPath: modelPath,
      options: modelOptions)

// Run inference
let categories = nlClassifier.classify(text: input)
```

有关详情，请参阅[源代码](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/ios/task/text/nlclassifier/Sources/TFLNLClassifier.h)。

## 用 C++ 运行推断

注：我们正在改善 C++ Task Library 的可用性，如提供预先构建的二进制文件，并创建用户友好的工作流以从源代码进行构建。C++ API 可能会发生变化。

```c++
// Initialization
std::unique_ptr<NLClassifier> classifier = NLClassifier::CreateFromFileAndOptions(
    model_path,
    {
      .input_tensor_name=kInputTensorName,
      .output_score_tensor_name=kOutputScoreTensorName,
    }).value();

// Run inference
std::vector<core::Category> categories = classifier->Classify(kInput);
```

有关详情，请参阅[源代码](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/text/nlclassifier/nl_classifier.h)。

## 结果示例

下面是[电影评论模型](https://www.tensorflow.org/lite/models/text_classification/overview)的分类结果示例。

输入：“What a waste of my time.”

输出：

```
category[0]: 'Negative' : '0.81313'
category[1]: 'Positive' : '0.18687'
```

用您自己的模型和测试数据试用简单的 [NLClassifier CLI 演示工具](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/examples/task/text/desktop/README.md#nlclassifier)。

## 模型兼容性要求

根据用例，`NLClassifier` API 可以加载带或不带 [TFLite 模型元数据](../../convert/metadata.md)的 TFLite 模型。

兼容的模型应满足以下要求：

- 输入张量 (kTfLiteString/kTfLiteInt32)

    - 模型的输入应为 kTfLiteString 张量原始输入字符串或用于原始输入字符串的正则表达式标记化索引的 kTfLiteInt32 张量。
    - 如果输入类型为 kTfLiteString，则模型不需要[元数据](../../convert/metadata.md)。
    - 如果输入类型为 kTfLiteInt32，则需要在输入张量的[元数据](../../convert/metadata.md)中设置 `RegexTokenizer`。

- 输入分数张量：(kTfLiteUInt8/kTfLiteInt8/kTfLiteInt16/kTfLiteFloat32/kTfLiteFloat64)

    - 每个分类类别分数的强制性输出张量。

    - 如果类型是 Int 类型中的一种，将其去量化为 double/float 到相应的平台

    - 可以在输出张量的对应类别标签的[元数据](../../convert/metadata.md)中包含一个可选的关联文件，该文件应为纯文本文件，每行一个标签，并且标签数量应与模型输出的类别数量相匹配。

- 输出标签张量：(kTfLiteString/kTfLiteInt32)

    - 每个类别的标签的可选输出张量应与输出分数张量的长度相同。如果不存在此张量，则 API 使用分数索引作为类名。

    - 如果输出分数张量的元数据中存在关联的标签文件，则会被忽略。
