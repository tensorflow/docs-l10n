# 画像セグメンタの統合

画像セグメンタは、画像の各ピクセルが特定のクラスに関連付けられているかどうかを予測します。これは、矩形の領域内の物体を検出する<a href="../../examples/object_detection/overview">物体検出</a>、および画像全体を分類する<a href="../../examples/image_classification/overview">画像分類</a>とは対照的です。画像セグメンタの詳細については、[画像セグメンテーションの概要](../../examples/segmentation/overview)をご覧ください。

Task Library `ImageSegmenter`API を使用して、カスタム画像セグメンタまたは事前トレーニングされたものをモバイルアプリにデプロイします。

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

Android アプリケーションで<code>ImageSegmenter</code>を使用する方法の例については、<a>画像セグメンテーションリファレンスアプリ</a>を参照してください。

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

    // Import the Task Vision Library dependency (NNAPI is included)
    implementation 'org.tensorflow:tensorflow-lite-task-vision'
    // Import the GPU delegate plugin Library for GPU inference
    implementation 'org.tensorflow:tensorflow-lite-gpu-delegate-plugin'
}
```

注：Android Gradle プラグインのバージョン 4.1 以降、.tflite はデフォルトで noCompress リストに追加され、上記の aaptOptions は不要になりました。

### ステップ 2: モデルを使用する

```java
// Initialization
ImageSegmenterOptions options =
    ImageSegmenterOptions.builder()
        .setBaseOptions(BaseOptions.builder().useGpu().build())
        .setOutputType(OutputType.CONFIDENCE_MASK)
        .build();
ImageSegmenter imageSegmenter =
    ImageSegmenter.createFromFileAndOptions(context, modelFile, options);

// Run inference
List<Segmentation> results = imageSegmenter.segment(image);
```

<code>ImageSegmenter</code>を構成するその他のオプションについては、<a>ソースコードと javadoc </a>をご覧ください。

## iOS で推論を実行する

### 手順 1: 依存関係をインストールする

タスクライブラリは、CocoaPods を使用したインストールをサポートしています。CocoaPods がシステムにインストールされていることを確認してください。手順については、[CocoaPods インストールガイド](https://guides.cocoapods.org/using/getting-started.html#getting-started)を参照してください。

ポッドを Xcode に追加する詳細な方法については、[CocoaPods ガイド](https://guides.cocoapods.org/using/using-cocoapods.html)を参照してください。

Podfile に `TensorFlowLiteTaskVision` ポッドを追加します。

```
target 'MyAppWithTaskAPI' do
  use_frameworks!
  pod 'TensorFlowLiteTaskVision'
end
```

推論で使用する `.tflite` モデルがアプリバンドルに存在することを確認します。

### 手順 2: モデルを使用する

#### Swift

```swift
// Imports
import TensorFlowLiteTaskVision

// Initialization
guard let modelPath = Bundle.main.path(forResource: "deeplabv3",
                                            ofType: "tflite") else { return }

let options = ImageSegmenterOptions(modelPath: modelPath)

// Configure any additional options:
// options.outputType = OutputType.confidenceMasks

let segmenter = try ImageSegmenter.segmenter(options: options)

// Convert the input image to MLImage.
// There are other sources for MLImage. For more details, please see:
// https://developers.google.com/ml-kit/reference/ios/mlimage/api/reference/Classes/GMLImage
guard let image = UIImage (named: "plane.jpg"), let mlImage = MLImage(image: image) else { return }

// Run inference
let segmentationResult = try segmenter.segment(mlImage: mlImage)
```

#### Objective C

```objc
// Imports
#import <TensorFlowLiteTaskVision/TensorFlowLiteTaskVision.h>

// Initialization
NSString *modelPath = [[NSBundle mainBundle] pathForResource:@"deeplabv3" ofType:@"tflite"];

TFLImageSegmenterOptions *options =
    [[TFLImageSegmenterOptions alloc] initWithModelPath:modelPath];

// Configure any additional options:
// options.outputType = TFLOutputTypeConfidenceMasks;

TFLImageSegmenter *segmenter = [TFLImageSegmenter imageSegmenterWithOptions:options
                                                                      error:nil];

// Convert the input image to MLImage.
UIImage *image = [UIImage imageNamed:@"plane.jpg"];

// There are other sources for GMLImage. For more details, please see:
// https://developers.google.com/ml-kit/reference/ios/mlimage/api/reference/Classes/GMLImage
GMLImage *gmlImage = [[GMLImage alloc] initWithImage:image];

// Run inference
TFLSegmentationResult *segmentationResult =
    [segmenter segmentWithGMLImage:gmlImage error:nil];
```

<code>TFLImageSegmenter</code>を構成するその他のオプションについては、<a>ソースコード</a>をご覧ください。

## Python で推論を実行する

### 手順 1: pip パッケージをインストールする

```
pip install tflite-support
```

### 手順 2: モデルを使用する

```python
# Imports
from tflite_support.task import vision
from tflite_support.task import core
from tflite_support.task import processor

# Initialization
base_options = core.BaseOptions(file_name=model_path)
segmentation_options = processor.SegmentationOptions(
    output_type=processor.SegmentationOptions.OutputType.CATEGORY_MASK)
options = vision.ImageSegmenterOptions(base_options=base_options, segmentation_options=segmentation_options)
segmenter = vision.ImageSegmenter.create_from_options(options)

# Alternatively, you can create an image segmenter in the following manner:
# segmenter = vision.ImageSegmenter.create_from_file(model_path)

# Run inference
image_file = vision.TensorImage.create_from_file(image_path)
segmentation_result = segmenter.segment(image_file)
```

<code>ImageSegmenter</code>を構成するその他のオプションについては、<a>ソースコード</a>をご覧ください。

## C++ で推論を実行する

```c++
// Initialization
ImageSegmenterOptions options;
options.mutable_base_options()->mutable_model_file()->set_file_name(model_path);
std::unique_ptr<ImageSegmenter> image_segmenter = ImageSegmenter::CreateFromOptions(options).value();

// Create input frame_buffer from your inputs, `image_data` and `image_dimension`.
// See more information here: tensorflow_lite_support/cc/task/vision/utils/frame_buffer_common_utils.h
std::unique_ptr<FrameBuffer> frame_buffer = CreateFromRgbRawBuffer(
      image_data, image_dimension);

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

 <img src="https://github.com/tensorflow/docs-l10n/blob/master/site/ja/lite/inference_with_metadata/task_library/images/segmentation-output.png?raw=true" alt="segmentation-output" class="">

独自のモデルとテストデータを使用して、シンプルな [ImageSegmenter 向け CLI デモツール](https://github.com/tensorflow/tflite-support/tree/master/tensorflow_lite_support/examples/task/vision/desktop#image-segmenter)をお試しください。

## モデルの互換性要件

`ImageSegmenter` API は、必須の [TFLite モデル メタデータ](../../models/convert/metadata)を持つ TFLite モデルを想定しています。[TensorFlow Lite Metadata Writer API](../../models/convert/metadata_writer_tutorial.ipynb#image_segmenters) を使用して画像セグメンタのメタデータを作成する例をご覧ください。

- 入力画像テンソル (kTfLiteUInt8/kTfLiteFloat32)

    - サイズ`[batch x height x width x channels]`の画像入力。
    - バッチ推論はサポートされていません (`batch`は 1 である必要があります)。
    - RGB 入力のみがサポートされています (`channels`は 3 である必要があります)。
    - 型が kTfLiteFloat32 の場合、入力の正規化のためにメタデータに NormalizationOptions をアタッチする必要があります。

- 出力マスクテンソル: (kTfLiteUInt8/kTfLiteFloat32)

    - テンソルサイズは`[batch x mask_height x mask_width x num_classes]`で`batch`は 1 である必要があります。`mask_width`と`mask_height`はモデルによって生成されたセグメンテーションマスクのサイズで、`num_classes`は、モデルでサポートされているクラスの数です。
    - TENSOR_AXIS_LABELS 型の AssociatedFile ラベルマップ (オプションですが推薦されます)。1 行に 1 つのラベルが含まれます。最初の AssociatedFile (存在する場合) は、結果の`label`フィールド (C++では`class_name`と名付けられています) を入力ために使用されます。`display_name`フィールドは、AssociatedFile (存在する場合) から入力されます。そのロケールは、作成時に使用される`ImageSegmenterOptions`の`display_names_locale`フィールドと一致します (デフォルトでは「en (英語)」)。これらのいずれも使用できない場合、結果の`index`フィールドのみが入力されます。
