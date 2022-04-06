# BERT 質問応答機能を統合する

Task Library `BertQuestionAnswerer` API は Bert モデルを読み込み、特定のパッセージの内容に基づいて質問に答えます。詳細については、質問応答モデルのドキュメントを<a href="../../models/bert_qa/overview.md">こちら</a>からご覧ください。

## BertQuestionAnswerer API の主な機能

- 質問と文脈の 2 つのテキストを入力として受け取り、可能性の高い回答のリストを出力します。

- 入力するテキストに対してグラフ外の Wordpiece または Sentencepiece トークン化を実行します。

## サポートされている BertQuestionAnswerer モデル

以下のモデルは、`BertNLClassifier` API と互換性があります。

- [TensorFlow Lite Model Maker for BERT Question Answer ](https://www.tensorflow.org/lite/tutorials/model_maker_question_answer)により作成されたモデル。

- [TensorFlow Hub の事前トレーニング済み BERT モデル](https://tfhub.dev/tensorflow/collections/lite/task-library/bert-question-answerer/1)。

- [モデルの互換性要件](#model-compatibility-requirements)を満たすカスタムモデル。

## Java で推論を実行する

### ステップ 1: Gradle の依存関係とその他の設定をインポートする

`.tflite`モデルファイルを、モデルが実行される Android モジュールのアセットディレクトリにコピーします。ファイルを圧縮しないように指定し、TensorFlow Lite ライブラリをモジュールの`build.gradle`ファイルに追加します。

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

注意：Android Gradle プラグインのバージョン 4.1 以降、.tflite はデフォルトで noCompress リストに追加され、上記の aaptOptions は不要になりました。

### ステップ 2: API を使用して推論を実行する

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

詳細については[ソースコード](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/java/src/java/org/tensorflow/lite/task/text/qa/BertQuestionAnswerer.java)をご覧ください。

## Swift で推論を実行する

### ステップ 1: CocoaPods をインポートする

Podfile に TensorFlowLiteTaskText ポッドを追加します

```
target 'MySwiftAppWithTaskAPI' do
  use_frameworks!
  pod 'TensorFlowLiteTaskText', '~> 0.0.1-nightly'
end
```

### ステップ 2: API を使用して推論を実行する

```swift
// Initialization
let mobileBertAnswerer = TFLBertQuestionAnswerer.questionAnswerer(
      modelPath: mobileBertModelPath)

// Run inference
let answers = mobileBertAnswerer.answer(
      context: context, question: question)
```

詳細については[ソースコード](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/text/qa/bert_question_answerer.h)をご覧ください。

## C++ で推論を実行する

```c++
// Initialization
BertQuestionAnswererOptions options;
options.mutable_base_options()->mutable_model_file()->set_file_name(model_file);
std::unique_ptr<BertQuestionAnswerer> answerer = BertQuestionAnswerer::CreateFromOptions(options).value();

// Run inference
std::vector<QaAnswer> positive_results = answerer->Answer(context_of_question, question_to_ask);
```

詳細については[ソースコード](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/text/qa/bert_question_answerer.h)をご覧ください。

## 結果の例

[ALBERT モデル](https://tfhub.dev/tensorflow/lite-model/albert_lite_base/squadv1/1)の回答結果の例を次に示します。

文脈: "The Amazon rainforest, alternatively, the Amazon Jungle, also known in English as Amazonia, is a moist broadleaf tropical rainforest in the Amazon biome that covers most of the Amazon basin of South America. This basin encompasses 7,000,000 km2 (2,700,000 sq mi), of which 5,500,000 km2 (2,100,000 sq mi) are covered by the rainforest. This region includes territory belonging to nine nations."

質問: "Where is Amazon rainforest?"

回答:

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

独自のモデルとテストデータを使用して、シンプルな[BertQuestionAnswerer用 CLI デモツール](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/examples/task/text/desktop/README.md#bert-question-answerer)をお試しください。

## モデルの互換性要件

`BertQuestionAnswerer` API では、[ TFLite モデルメタデータ](../../convert/metadata.md)を持つ TFLite モデル が必要です。

メタデータは次の要件を満たす必要があります。

- Wordpiece/Sentencepiece Tokenizer の input_process_units

- Tokenizer の出力用の「ids」、「mask」、「segment_ids」という名前の 3 つの入力テンソル

- 「end_logits」および「start_logits」という名前の 2 つの出力テンソルは、コンテキストにおける回答の相対位置を示します。
