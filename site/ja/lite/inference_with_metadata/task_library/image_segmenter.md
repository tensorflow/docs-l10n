# 画像セグメンタの統合

画像セグメンタは、画像の各ピクセルが特定のクラスに関連付けられているかどうかを予測します。これは、矩形の領域でオブジェクトを検出する<a href="../../models/object_detection/overview.md">オブジェクト検出</a>、および画像全体を分類する<a href="../../models/image_classification/overview.md">画像分類</a>とは対照的です。画像セグメンタの詳細については、[画像セグメンテーションの概要](../../models/segmentation/overview.md)をご覧ください。

Task Library `ImageSegmenter`API を使用して、カスタム画像セグメンタまたは事前トレーニングされたものをモデルアプリにデプロイします。

## ImageSegmenter API の主な機能

- 回転、サイズ変更、色空間変換などの入力画像処理。

- マップロケールのラベル付け。

- カテゴリマスクと信頼マスクの 2 つの出力型。

- 表示用の色付きラベル。

## サポートされている画像セグメンタモデル

次のモデルは、`ImageSegmenter` API との互換性が保証されています。

- [TensorFlow Hub の事前トレーニング済みセグメンテーションモデル](https://tfhub.dev/tensorflow/collections/lite/task-library/image-segmenter/1)。

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
ImageSegmenterOptions options = ImageSegmenterOptions.builder().setOutputType(OutputType.CONFIDENCE_MASK).build();
ImageSegmenter imageSegmenter = ImageSegmenter.createFromFileAndOptions(context, modelFile, options);

// Run inference
List<Segmentation> results = imageSegmenter.segment(image);
```

<code>ImageSegmenter</code>を構成するその他のオプションについては、<a>ソースコードと javadoc </a>をご覧ください。

## C++ で推論を実行する

注: C++ Task Library では、使いやすさを向上するために構築済みのバイナリを提供したり、ユーザーフレンドリーなワークフローを作成してソースコードから構築できるようしています。C++ API は変更される可能性があります。

```c++
// Initialization
ImageSegmenterOptions options;
options.mutable_model_file_with_metadata()->set_file_name(model_file);
std::unique_ptr<ImageSegmenter> image_segmenter = ImageSegmenter::CreateFromOptions(options).value();

// Run inference
const SegmentationResult result = image_segmenter->Segment(*frame_buffer).value();
```

<code>ImageSegmenter</code>を構成するその他のオプションについては、<a>ソースコード</a>をご覧ください。

## 結果の例

これは、TensorFlow Hub で利用可能な一般的なセグメンテーションモデルである [deeplab_v3](https://tfhub.dev/tensorflow/lite-model/deeplabv3/1/metadata/1) のセグメンテーション結果の例です。

<img src="images/plane.jpg" alt="plane" width="50%">

```
Color Legend:
 (r: 000, g: 000, b: 000):
  index       : 0
  class name  : background
 (r: 128, g: 000, b: 000):
  index       : 1
  class name  : aeroplane

# (omitting multiple lines for conciseness) ...

 (r: 128, g: 192, b: 000):
  index       : 19
  class name  : train
 (r: 000, g: 064, b: 128):
  index       : 20
  class name  : tv
Tip: use a color picker on the output PNG file to inspect the output mask with
this legend.
```

セグメンテーションカテゴリマスクは次のようになります。


<img src="images/segmentation-output.png" alt="segmentation-output" width="30%">

独自のモデルとテストデータを使用して、シンプルな [ImageSegmenter 向け CLI デモツール](https://github.com/tensorflow/tflite-support/tree/master/tensorflow_lite_support/examples/task/vision/desktop#image-segmenter)をお試しください。

## モデルの互換性要件

`ImageSegmenter` API では、[ TFLite モデルメタデータ](../../convert/metadata.md)を持つ TFLite モデル が必要です。

- 入力画像テンソル (kTfLiteUInt8/kTfLiteFloat32)

    - サイズ`[batch x height x width x channels]`の画像入力。
    - バッチ推論はサポートされていません (`batch`は 1 である必要があります)。
    - RGB 入力のみがサポートされています (` channels `は 3 である必要があります)。
    - 型が kTfLiteFloat32 の場合、入力の正規化のためにメタデータに NormalizationOptions をアタッチする必要があります。

- 出力マスクテンソル: (kTfLiteUInt8/kTfLiteFloat32)

    - テンソルサイズは`[batch x mask_height x mask_width x num_classes]`で`batch`は 1 である必要があります。`mask_width`と`mask_height`はモデルによって生成されたセグメンテーションマスクのサイズで、`num_classes`は、モデルでサポートされているクラスの数です。
    - TENSOR_AXIS_LABELS 型の AssociatedFile ラベルマップ (オプションですが推薦されます)。1 行に 1 つのラベルが含まれます。最初の AssociatedFile (存在する場合) は、結果の`label`フィールド (C++では`class_name`と名付けられています) を入力ために使用されます。`display_name`フィールドは、AssociatedFile (存在する場合) から入力されます。そのロケールは、作成時に使用される`ImageSegmenterOptions`の`display_names_locale`フィールドと一致します (デフォルトでは「en (英語)」)。これらのいずれも使用できない場合、結果の`index`フィールドのみが入力されます。
