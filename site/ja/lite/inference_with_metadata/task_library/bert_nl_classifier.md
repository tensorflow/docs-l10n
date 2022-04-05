# BERT 自然言語分類器の統合

Task Library の`BertNLClassifier`API は、入力テキストをさまざまなカテゴリに分類する`NLClassifier`とよく似ていますが、この API は、TFLite モデルの外で Wordpiece および Sentencepiece トークン化を必要とする Bert 関連モデル専用に設計されています。

## BertNLClassifier API の主な機能

- 単一の文字列を入力として受け取り、その文字列で分類を実行し、分類の結果として &lt;Label、Score&gt; のペアを出力します。

- 入力するテキストに対してグラフ外で [Wordpiece ](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/text/tokenizers/bert_tokenizer.h) または [Sentencepiece](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/text/tokenizers/sentencepiece_tokenizer.h) トークン化を実行します。

## サポートされている BertNLClassifier モデル

以下のモデルは、`BertNLClassifier` API と互換性があります。

- [TensorFlow Lite Model Maker for text Classfication](https://www.tensorflow.org/lite/tutorials/model_maker_text_classification) が作成した Bert モデル。

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

注：Android Gradle プラグインのバージョン 4.1 以降、.tflite はデフォルトで noCompress リストに追加され、上記の aaptOptions は不要になりました。

### ステップ 2: API を使用して推論を実行する

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

詳細については[ソースコード](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/java/src/java/org/tensorflow/lite/task/text/nlclassifier/BertNLClassifier.java)をご覧ください。

## Swift で推論を実行する

### ステップ 1: CocoaPods をインポートする

Podfile に TensorFlowLiteTaskText ポッドを追加します

```
target 'MySwiftAppWithTaskAPI' do
  use_frameworks!
  pod 'TensorFlowLiteTaskText', '~> 0.2.0'
end
```

### ステップ 2: API を使用して推論を実行する

```swift
// Initialization
let bertNLClassifier = TFLBertNLClassifier.bertNLClassifier(
      modelPath: bertModelPath)

// Run inference
let categories = bertNLClassifier.classify(text: input)
```

詳細については[ソースコード](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/text/nlclassifier/bert_nl_classifier.h)をご覧ください。

## C++ で推論を実行する

```c++
// Initialization
BertNLClassifierOptions options;
options.mutable_base_options()->mutable_model_file()->set_file_name(model_file);
std::unique_ptr<BertNLClassifier> classifier = BertNLClassifier::CreateFromOptions(options).value();

// Run inference
std::vector<core::Category> categories = classifier->Classify(kInput);
```

詳細については[ソースコード](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/text/nlclassifier/bert_nl_classifier.h)をご覧ください。

## 結果の例

これは、モデルメーカーの [MobileBert](https://www.tensorflow.org/lite/tutorials/model_maker_text_classification) モデルを使用した映画レビューの分類結果の例です。

入力: "it's a charming and often affecting journey"

出力:

```
category[0]: 'negative' : '0.00006'
category[1]: 'positive' : '0.99994'
```

独自のモデルとテストデータを使用して、シンプルな [BertNLClassifier 用 CLI デモツール](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/examples/task/text/desktop/README.md#bertnlclassifier)を試してみてください。

## モデルの互換性要件

`BetNLClassifier` API では、[ TFLite モデルメタデータ](../../convert/metadata.md)を持つ TFLite モデルが必要です。

メタデータは次の要件を満たす必要があります。

- Wordpiece/Sentencepiece Tokenizer の input_process_units

- Tokenizer の出力用の「ids」、「mask」、「segment_ids」という名前の 3 つの入力テンソル

- float32 型の 1 つの出力テンソル。オプションでラベルファイルが添付されている場合があります。ラベルファイルが添付されている場合、ファイルは 1 行に 1 つのラベルが付いたプレーンテキストファイルである必要があり、ラベルの数はモデルの出力としてカテゴリの数と一致する必要があります。
