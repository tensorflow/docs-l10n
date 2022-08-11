# 集成 BERT 自然语言分类器

Task Library `BertNLClassifier` API 与将输入文本分类为不同类别的 `NLClassifier` 非常相似，不同之处在于，该 API 专门为 BERT 相关的模型定制，这些模型需要在 TFLite 模型之外使用 WordPiece 和 SentencePiece 标记化。

## BertNLClassifier API 的主要功能

- 将单个字符串作为输入，对该字符串进行分类，并输出 &lt;Label, Score&gt; 对作为分类结果。

- 对输入文本执行计算图外的 [WordPiece](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/text/tokenizers/bert_tokenizer.h) 或 [SentencePiece](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/text/tokenizers/sentencepiece_tokenizer.h) 标记化。

## 支持的 BertNLClassifier 模型

以下模型与 `BertNLClassifier` API 兼容。

- 由[适用于文本分类的 TensorFlow Lite Model Maker](https://www.tensorflow.org/lite/models/modify/model_maker/text_classification) 创建的 BERT 模型。

- 符合[模型兼容性要求](#model-compatibility-requirements)的自定义模型。

## 用 Java 运行推断

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

    // Import the Task Text Library dependency (NNAPI is included)
    implementation 'org.tensorflow:tensorflow-lite-task-text:0.3.0'
}
```

注：从 Android Gradle 插件的 4.1 版开始，默认情况下，.tflite 将被添加到 noCompress 列表中，不再需要上面的 aaptOptions。

### 步骤 2：使用 API 运行推断

```java
// Initialization
BertNLClassifierOptions options =
    BertNLClassifierOptions.builder()
        .setBaseOptions(BaseOptions.builder().setNumThreads(4).build())
        .build();
BertNLClassifier classifier =
    BertNLClassifier.createFromFileAndOptions(context, modelFile, options);

// Run inference
List<Category> results = classifier.classify(input);
```

有关详情，请参阅[源代码](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/java/src/java/org/tensorflow/lite/task/text/nlclassifier/BertNLClassifier.java)。

## 用 Swift 运行推断

### 步骤 1：导入 CocoaPods

在 Podfile 中添加 TensorFlowLiteTaskText Pod。

```
target 'MySwiftAppWithTaskAPI' do
  use_frameworks!
  pod 'TensorFlowLiteTaskText', '~> 0.2.0'
end
```

### 步骤 2：使用 API 运行推断

```swift
// Initialization
let bertNLClassifier = TFLBertNLClassifier.bertNLClassifier(
      modelPath: bertModelPath)

// Run inference
let categories = bertNLClassifier.classify(text: input)
```

有关详情，请参阅[源代码](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/ios/task/text/nlclassifier/Sources/TFLBertNLClassifier.h)。

## 用 C++ 运行推断

```c++
// Initialization
BertNLClassifierOptions options;
options.mutable_base_options()->mutable_model_file()->set_file_name(model_path);
std::unique_ptr<BertNLClassifier> classifier = BertNLClassifier::CreateFromOptions(options).value();

// Run inference with your input, `input_text`.
std::vector<core::Category> categories = classifier->Classify(input_text);
```

请参阅[源代码](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/text/bert_nl_classifier.h)，了解详细信息。

## 结果示例

下面是使用 Model Maker 中的 [MobileBert](https://www.tensorflow.org/lite/models/modify/model_maker/text_classification) 模型对电影评论进行分类的结果示例。

输入：“it's a charming and often affecting journey”

输出：

```
category[0]: 'negative' : '0.00006'
category[1]: 'positive' : '0.99994'
```

用您自己的模型和测试数据试用简单的 [BertNLClassifier CLI 演示工具](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/examples/task/text/desktop/README.md#bertnlclassifier)。

## 模型兼容性要求

`BetNLClassifier` API 需要具有强制性 [TFLite 模型元数据](../../models/convert/metadata.md)的 TFLite 模型。

元数据应满足以下要求：

- 用于 WordPiece/SentencePiece 标记器的 input_process_units

- 3 个名称为 "ids"、"mask" 和 "segment_ids" 的输入张量，用于标记器的输出

- 1 个类型为 float32 的输出张量，并可选择附加标签文件。如果附加了标签文件，则该文件应该为纯文本文件，每行一个标签，标签的数量应与模型输出的类别数量相匹配。
