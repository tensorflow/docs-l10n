# 集成 BERT 问答器

Task Library `BertQuestionAnswerer` API 能够加载 BERT 模型并根据给定段落的内容回答问题。有关详情，请参阅<a href="../../examples/bert_qa/overview">问答模型文档</a>。

## BertQuestionAnswerer API 的主要功能

- 将两个文本输入作为问题和上下文，并输出可能答案的列表。

- 对输入文本执行计算图外的 <a>WordPiece</a> 或 <a>SentencePiece</a> 标记。

## 支持的 BertQuestionAnswerer 模型

以下模型与 `BertNLClassifier` API 兼容。

- 由[适用于 BERT 问答的 TensorFlow Lite Model Maker](https://www.tensorflow.org/lite/models/modify/model_maker/question_answer) 创建的模型。

- [TensorFlow Hub 上预训练 BERT 模型](https://tfhub.dev/tensorflow/collections/lite/task-library/bert-question-answerer/1)。

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
BertQuestionAnswererOptions options =
    BertQuestionAnswererOptions.builder()
        .setBaseOptions(BaseOptions.builder().setNumThreads(4).build())
        .build();
BertQuestionAnswerer answerer =
    BertQuestionAnswerer.createFromFileAndOptions(
        androidContext, modelFile, options);

// Run inference
List<QaAnswer> answers = answerer.answer(contextOfTheQuestion, questionToAsk);
```

有关详情，请参阅[源代码](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/java/src/java/org/tensorflow/lite/task/text/qa/BertQuestionAnswerer.java)。

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
let mobileBertAnswerer = TFLBertQuestionAnswerer.questionAnswerer(
      modelPath: mobileBertModelPath)

// Run inference
let answers = mobileBertAnswerer.answer(
      context: context, question: question)
```

有关详情，请参阅[源代码](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/ios/task/text/qa/Sources/TFLBertQuestionAnswerer.h)。

## 用 C++ 运行推断

```c++
// Initialization
BertQuestionAnswererOptions options;
options.mutable_base_options()->mutable_model_file()->set_file_name(model_path);
std::unique_ptr<BertQuestionAnswerer> answerer = BertQuestionAnswerer::CreateFromOptions(options).value();

// Run inference with your inputs, `context_of_question` and `question_to_ask`.
std::vector<QaAnswer> positive_results = answerer->Answer(context_of_question, question_to_ask);
```

请参阅[源代码](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/text/bert_question_answerer.h)，了解详细信息。

## 结果示例

下面是 [ALBERT 模型](https://tfhub.dev/tensorflow/lite-model/albert_lite_base/squadv1/1)回答结果的示例。

上下文：“The Amazon rainforest, alternatively, the Amazon Jungle, also known in English as Amazonia, is a moist broadleaf tropical rainforest in the Amazon biome that covers most of the Amazon basin of South America. This basin encompasses 7,000,000 km2 (2,700,000 sq mi), of which 5,500,000 km2 (2,100,000 sq mi) are covered by the rainforest. This region includes territory belonging to nine nations.”

问题：“Where is Amazon rainforest?”

回答：

```
answer[0]:  'South America.'
logit: 1.84847, start_index: 39, end_index: 40
answer[1]:  'most of the Amazon basin of South America.'
logit: 1.2921, start_index: 34, end_index: 40
answer[2]:  'the Amazon basin of South America.'
logit: -0.0959535, start_index: 36, end_index: 40
answer[3]:  'the Amazon biome that covers most of the Amazon basin of South America.'
logit: -0.498558, start_index: 28, end_index: 40
answer[4]:  'Amazon basin of South America.'
logit: -0.774266, start_index: 37, end_index: 40

```

用您自己的模型和测试数据试用简单的 [BertQuestionAnswerer CLI 演示工具](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/examples/task/text/desktop/README.md#bert-question-answerer)。

## 模型兼容性要求

`BertQuestionAnswerer` API 需要具有强制性 [TFLite 模型元数据](../../models/convert/metadata)的 TFLite 模型。

元数据应满足以下要求：

- 用于 WordPiece/SentencePiece 标记器的 input_process_units

- 3 个名称为 "ids"、"mask" 和 "segment_ids" 的输入张量，用于标记器的输出

- 2 个名称为 "end_logits" 和 "start_logits" 的输出张量，以表示答案在上下文中的相对位置
