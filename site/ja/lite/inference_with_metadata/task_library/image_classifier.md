# 画像分類器の統合

画像分類は、一般的な機械学習の使用例で、機械学習を使用して画像が表すものを識別します。たとえば、特定の写真にどのような種類の動物が写っているのかを知りたい場合に画像が何を表すかを予測するタスクは、*画像分類*と呼ばれます。画像分類器は、画像のさまざまなクラスを認識するようにトレーニングされています。たとえば、ウサギ、ハムスター、犬の3種類の動物を表す写真を認識するようにモデルをトレーニングすることができます。画像分類器の詳細については、[画像分類の概要](../../models/image_classification/overview.md)をご覧ください。

Task Library `ImageClassifier` APIを使用して、カスタム画像分類器または事前トレーニング済みのものをモデルアプリにデプロイします。

## ImageClassifier API の主な機能

- 回転、サイズ変更、色空間変換などの入力画像処理。

- 入力画像の関心領域。

- マップロケールのラベル付け。

- 結果をフィルタリングするスコアしきい値。

- Top-k 分類結果。

- 許可リストと拒否リストのラベルを付け。

## サポートされている画像分類モデル

次のモデルは、`ImageClassifier` API との互換性が保証されています。

- [TensorFlow Lite Model Maker による画像分類](https://www.tensorflow.org/lite/tutorials/model_maker_image_classification)によって作成されたモデル。

- [TensorFlow Lite ホステッドモデルの事前トレーニング済み画像分類モデル](https://www.tensorflow.org/lite/guide/hosted_models#image_classification)。

- [TensorFlow Hub の事前トレーニング済み画像分類モデル](https://tfhub.dev/tensorflow/collections/lite/task-library/image-classifier/1)。

- [AutoML Vision Edge 画像分類](https://cloud.google.com/vision/automl/docs/edge-quickstart)によって作成されたモデル。

- [モデルの互換性要件](#model-compatibility-requirements)を満たすカスタムモデル。

## Java で推論を実行する

Android アプリケーションで<code>ImageClassifier</code>を使用する方法の例については、<a>画像分類リファレンスアプリ</a>を参照してください。

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

    // Import the Task Vision Library dependency
    implementation 'org.tensorflow:tensorflow-lite-task-vision:0.1.0'
}
```

### ステップ 2: モデルを使用する

```java
// Initialization
ImageClassifierOptions options = ImageClassifierOptions.builder().setMaxResults(1).build();
ImageClassifier imageClassifier = ImageClassifier.createFromFileAndOptions(context, modelFile, options);

// Run inference
List<Classifications> results = imageClassifier.classify(image);
```

`ImageClassifier`を構成するその他のオプションについては、[ソースコードと javadoc](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/java/src/java/org/tensorflow/lite/task/vision/classifier/ImageClassifier.java) をご覧ください。

## C++ で推論を実行する

注: C++  Task Library では、使いやすさを向上するために構築済みのバイナリを提供したり、ユーザーフレンドリーなワークフローを作成してソースコードから構築できるようしています。C++ API は変更される可能性があります。

```c++
// Initialization
ImageClassifierOptions options;
options.mutable_model_file_with_metadata()->set_file_name(model_file);
std::unique_ptr<ImageClassifier> image_classifier = ImageClassifier::CreateFromOptions(options).value();

// Run inference
const ClassificationResult result = image_classifier->Classify(*frame_buffer).value();
```

<code>ImageClassifier</code>を構成するその他のオプションについては、<a>ソースコード</a>をご覧ください。

## 結果の例

以下は、[鳥分類器](https://tfhub.dev/google/lite-model/aiy/vision/classifier/birds_V1/3)の分類結果の例です。

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

独自のモデルとテストデータを使用して、シンプルな[ImageClassifier用 CLI デモツール](https://github.com/tensorflow/tflite-support/tree/master/tensorflow_lite_support/examples/task/vision/desktop#image-classifier)をお試しください。

## モデルの互換性要件

`ImageClassifier` API では、[ TFLite モデルメタデータ](../../convert/metadata.md)を持つ TFLite モデル が必要です。

互換性のある画像分類モデルは、次の要件を満たす必要があります。

- 入力画像テンソル (kTfLiteUInt8/kTfLiteFloat32)

    - サイズ`[batch x height x width x channels]`の画像入力。
    - バッチ推論はサポートされていません (`batch`は 1 である必要があります)。
    - RGB 入力のみがサポートされています (` channels `は 3 である必要があります)。
    - 型が kTfLiteFloat32 の場合、入力の正規化のためにメタデータに NormalizationOptions をアタッチする必要があります。

- 出力画像テンソル (kTfLiteUInt8/kTfLiteFloat32)

    - `N` クラスと 2 次元または 4 次元のいずれか (`[1 x N]`または`[1 x 1 x 1 x N]`)
    - TENSOR_AXIS_LABELS 型の AssociatedFile ラベルマップ (オプションですが推薦されます)。1 行に 1 つのラベルが含まれます。最初の AssociatedFile (存在する場合) は、結果の`label`フィールド (C ++では`class_name`と名付けられています) を入力ために使用されます。`display_name`フィールドは、AssociatedFile (存在する場合) から入力されます。そのロケールは、作成時に使用される`ImageClassifierOptions`の`display_names_locale`フィールドと一致します（デフォルトでは「en (英語)」）。これらのいずれも使用できない場合、結果の`index`フィールドのみが入力されます。
