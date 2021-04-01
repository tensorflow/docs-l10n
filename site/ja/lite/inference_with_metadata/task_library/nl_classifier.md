# 自然言語分類器の統合

Task Library の`NLClassifier` API は、入力テキストをさまざまなカテゴリに分類し、ほとんどのテキスト分類モデルを処理できる多用途で構成可能な API です。

## NLClassifier API の主な機能

- 単一の文字列を入力として受け取り、その文字列で分類を実行し、分類の結果として &lt;Label、Score&gt; のペアを出力します。

- 入力テキストの正規表現トークン化（オプション）。

- さまざまな分類モデルに適応するように構成可能。

## サポートされている NLClassifier モデル

次のモデルは、`NLClassifier` API との互換性が保証されています。

- <a href="../../models/text_classification/overview.md">映画レビューの感情分類</a>モデル。

- [テキスト分類用 TensorFlow Lite Model Maker ](https://www.tensorflow.org/lite/tutorials/model_maker_text_classification)によって作成された`average_word_vec` 仕様のモデル。

- [モデルの互換性要件](#model-compatibility-requirements)を満たすカスタムモデル。

## Java で推論を実行する

Android アプリケーションで`NLClassifier`を使用する方法の例については、[テキスト分類リファレンスアプリ](https://github.com/tensorflow/examples/blob/master/lite/examples/text_classification/android/lib_task_api/src/main/java/org/tensorflow/lite/examples/textclassification/client/TextClassificationClient.java)を参照してください。

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

    // Import the Task Text Library dependency
    implementation 'org.tensorflow:tensorflow-lite-task-text:0.1.0'
}
```

### ステップ 2: API を使用して推論を実行する

```java
// Initialization, use NLClassifierOptions to configure input and output tensors
NLClassifierOptions options = NLClassifierOptions.builder().setInputTensorName(INPUT_TENSOR_NAME).setOutputScoreTensorName(OUTPUT_SCORE_TENSOR_NAME).build();
NLClassifier classifier = NLClassifier.createFromFileAndOptions(context, modelFile, options);

// Run inference
List<Category> results = classifier.classify(input);
```

`NLClassifier`を構成するその他のオプションについては、<a>ソースコード</a> をご覧ください。

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
var modelOptions:TFLNLClassifierOptions = TFLNLClassifierOptions()
modelOptions.inputTensorName = inputTensorName
modelOptions.outputScoreTensorName = outputScoreTensorName
let nlClassifier = TFLNLClassifier.nlClassifier(
      modelPath: modelPath,
      options: modelOptions)

// Run inference
let categories = nlClassifier.classify(text: input)
```

詳細については[ソースコード](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/text/nlclassifier/nl_classifier.h)をご覧ください。

## C++ で推論を実行する

注意: C++ タスクライブラリでは、使いやすさを向上するために構築済みのバイナリを提供したり、ユーザーフレンドリーなワークフローを作成してソースコードから構築できるようしています。C++ API は変更される可能性があります。

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

詳細については[ソースコード](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/text/nlclassifier/nl_classifier.h)をご覧ください。

## 結果の例

これは、[映画レビューモデル](https://www.tensorflow.org/lite/models/text_classification/overview)の分類結果の例です。

入力：時間の無駄。

出力：

```
category[0]: 'Negative' : '0.81313'
category[1]: 'Positive' : '0.18687'
```

独自のモデルとテストデータを使用して、シンプルな [NLClassifier 用 CLI デモツール](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/examples/task/text/desktop/README.md#nlclassifier)をお試しください。

## モデルの互換性要件

ユースケースに応じて、`NLClassifier` API は、[TFLite モデルメタデータ](../../convert/metadata.md)の有無にかかわらず TFLite モデルを読み込めます。

互換性のあるモデルは、次の要件を満たす必要があります。

- 入力テンソル：(kTfLiteString/kTfLiteInt32)

    - モデルの入力は、kTfLiteString テンソル生入力文字列または生入力文字列の正規表現トークン化インデックス用の kTfLiteInt32 テンソルのいずれかである必要があります。
    - 入力型が kTfLiteString の場合、モデルに[メタデータ](../../convert/metadata.md)は必要ありません。
    - 入力型が kTfLiteInt32 の場合、`RegexTokenizer`を入力テンソルの[メタデータ](../../convert/metadata.md)に設定する必要があります。

- 出力スコアテンソル：(kTfLiteUInt8/kTfLiteInt8/kTfLiteInt16/kTfLiteFloat32/kTfLiteFloat64)

    - 分類された各カテゴリのスコアの必須出力テンソル。

    - 型が Int 型のいずれかである場合は、対応するプラットフォームに倍精度/浮動小数点数で非量子化します。

    - カテゴリラベルの出力テンソルの対応する[メタデータ](../../convert/metadata.md)にオプションの関連ファイルを含めることができます。ファイルは 1 行に 1 つのラベルを持つプレーンテキストファイルである必要があり、ラベルの数はモデル出力としてカテゴリの数と一致する必要があります。

- 出力ラベルテンソル：(kTfLiteString/kTfLiteInt32)

    - 各カテゴリのラベルのオプションの出力テンソルは、出力スコアテンソルと同じ長さである必要があります。このテンソルが存在しない場合、API はクラス名としてスコアインデックスを使用します。

    - 関連するラベルファイルが出力スコアテンソルのメタデータに存在する場合は無視されます。
