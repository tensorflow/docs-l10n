# 物体検出器の統合

物体検出器は、既知の物体セットのどれが存在するかを識別し、特定の画像またはビデオストリーム内のそれらの位置に関する情報を提供できます。物体検出器は、物体の複数のクラスの存在と位置を検出するようにトレーニングされています。たとえば、さまざまな果物を含む画像でモデルをトレーニングし、それらが表す果物のクラスを指定する*ラベル* (リンゴ、バナナ、イチゴなど) と各物体が画像のどこに現れるかを特定するデータを提供できます。物体検出器の詳細については、[物体検出の概要](../../models/object_detection/overview.md)をご覧ください。

Task Library `ObjectDetector` API を使用して、カスタム物体検出器または事前トレーニング済みの検出器をモデルアプリにデプロイします。

## ObjectDetector API の主な機能

- 回転、サイズ変更、色空間変換などの入力画像処理。

- マップロケールのラベル付け。

- 結果をフィルタリングするスコアしきい値。

- Top-k 分類結果。

- 許可リストと拒否リストのラベルを付け。

## サポートされている物体検出モデル

次のモデルは、`ObjectDetector` APIとの互換性が保証されています。

- [TensorFlow Hub の事前トレーニング済み物体検出モデル](https://tfhub.dev/tensorflow/collections/lite/task-library/object-detector/1)。

- [AutoML Vision Edge 物体検出](https://cloud.google.com/vision/automl/object-detection/docs)によって作成されたモデル。

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

    // Import the Task Vision Library dependency
    implementation 'org.tensorflow:tensorflow-lite-task-vision:0.0.0-nightly'
}
```

### ステップ 2: モデルを使用する

```java
// Initialization
ObjectDetectorOptions options = ObjectDetectorOptions.builder().setMaxResults(1).build();
ObjectDetector objectDetector = ObjectDetector.createFromFileAndOptions(context, modelFile, options);

// Run inference
List<Detection> results = objectDetector.detect(image);
```

<code>ObjectDetector</code>を構成するその他のオプションについては、<a>ソースコードと javadoc</a> をご覧ください。

## C++ で推論を実行する

注: C++ Task Library では、使いやすさを向上するために構築済みのバイナリを提供したり、ユーザーフレンドリーなワークフローを作成してソースコードから構築できるようしています。C++ API は変更される可能性があります。

```c++
// Initialization
ObjectDetectorOptions options;
options.mutable_model_file_with_metadata()->set_file_name(model_file);
std::unique_ptr<ObjectDetector> object_detector = ObjectDetector::CreateFromOptions(options).value();

// Run inference
const DetectionResult result = object_detector->Detect(*frame_buffer).value();
```

<code>ObjectDetector</code>を構成するその他のオプションについては、<a>ソースコード</a>をご覧ください。

## 結果の例

TensorFlow Hub からの [ssd mobilenet v1](https://tfhub.dev/tensorflow/lite-model/ssd_mobilenet_v1/1/metadata/1) の検出結果の例を次に示します。

 <img src="https://github.com/tensorflow/docs-l10n/blob/master/site/ja/lite/inference_with_metadata/task_library/images/dogs.jpg?raw=true" alt="dogs" class="">

```
Results:
 Detection #0 (red):
  Box: (x: 355, y: 133, w: 190, h: 206)
  Top-1 class:
   index       : 17
   score       : 0.73828
   class name  : dog
 Detection #1 (green):
  Box: (x: 103, y: 15, w: 138, h: 369)
  Top-1 class:
   index       : 17
   score       : 0.73047
   class name  : dog
```

境界矩形を入力画像にレンダリングします。

 <img src="https://github.com/tensorflow/docs-l10n/blob/master/site/ja/lite/inference_with_metadata/task_library/images/detection-output.png?raw=true" alt="detection output" class="">

独自のモデルとテストデータを使用して、シンプルな [ObjectDetector 向け CLI デモツール](https://github.com/tensorflow/tflite-support/tree/master/tensorflow_lite_support/examples/task/vision/desktop#object-detector)をお試しください。

## モデルの互換性要件

`ObjectDetector` APIでは、[ TFLite モデルメタデータ](../../convert/metadata.md)を持つ TFLite モデル が必要です。

互換性のある物体検出モデルは、次の要件を満たす必要があります。

- 入力画像テンソル: (kTfLiteUInt8/kTfLiteFloat32)

    - サイズ`[batch x height x width x channels]`の画像入力。
    - バッチ推論はサポートされていません (`batch`は 1 である必要があります)。
    - RGB 入力のみがサポートされています (` channels `は 3 である必要があります)。
    - 型が kTfLiteFloat32 の場合、入力の正規化のためにメタデータに NormalizationOptions をアタッチする必要があります。

- 出力テンソルは、以下のように`DetectionPostProcess`演算の 4 つの出力でなければなりません。

    - 位置テンソル (kTfLiteFloat32)

        - サイズ`[1 x num_results x 4]`のテンソル。[上、左、右、下]の形式で境界矩形を表す内部配列。
        - BoundingBoxProperties はメタデータに添付する必要があり、`type=BOUNDARIES`および `coordinate_type = RATIO を指定する必要があります。

    - クラステンソル (kTfLiteFloat32)

        - サイズ`[1 x num_results]`のテンソル。各値はクラスの整数インデックスを表します。
        - TENSOR_AXIS_LABELS 型の AssociatedFile ラベルマップ (オプションですが推薦されます)。1 行に 1 つのラベルが含まれます。最初の AssociatedFile (存在する場合) は、結果の`class_name`フィールドの入力のために使用されます。`display_name`フィールドは、AssociatedFile (存在する場合) から入力されます。そのロケールは、作成時に使用される`ObjectDetectorOptions`の`display_names_locale`フィールドと一致します（デフォルトでは「en (英語)」）。これらのいずれも使用できない場合、結果の<code>index</code>フィールドのみが入力されます。

    - スコアテンソル (kTfLiteFloat32)

        - サイズ`[1 x num_results]`のテンソル。各値は検出された物体のスコアを表します。

    - 検出テンソル数 (kTfLiteFloat32)

        - テンソルサイズ `[1]`の整数の num_results。
