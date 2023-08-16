# 物体検出器の統合

物体検出器は、既知の物体セットのどれが存在するかを識別し、特定の画像またはビデオストリーム内のそれらの位置に関する情報を提供します。物体検出器は、物体の複数のクラスの存在と位置を検出するようにトレーニングされています。たとえば、さまざまな果物を含む画像でモデルをトレーニングし、それらが表す果物のクラスを指定する*ラベル* (リンゴ、バナナ、イチゴなど) と各物体が画像のどこに現れるかを特定するデータを提供できます。物体検出器の詳細については、[物体検出の概要](../../examples/object_detection/overview)をご覧ください。

Task Library `ObjectDetector` API を使用して、カスタム物体検出器または事前トレーニング済みの検出器をモバイルアプリにデプロイします。

## ObjectDetector API の主な機能

- 回転、サイズ変更、色空間変換などの入力画像処理。

- マップロケールのラベル付け。

- 結果をフィルタリングするスコアしきい値。

- Top-k 検出結果。

- 許可リストと拒否リストのラベルを付け。

## サポートされている物体検出モデル

次のモデルは、`ObjectDetector` APIとの互換性が保証されています。

- [TensorFlow Hub の事前トレーニング済み物体検出モデル](https://tfhub.dev/tensorflow/collections/lite/task-library/object-detector/1)。

- [AutoML Vision Edge 物体検出](https://cloud.google.com/vision/automl/object-detection/docs)によって作成されたモデル。

- [物体検出器向け TensorFlow Lite Model Maker](https://www.tensorflow.org/lite/guide/model_maker) により作成されたモデル。

- [モデルの互換性要件](#model-compatibility-requirements)を満たすカスタムモデル。

## Java で推論を実行する

Android アプリケーションで<code>ObjectDetector</code>を使用する方法の例については、<a>物体検出リファレンスアプリ</a>を参照してください。

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

注意：Android Gradle プラグインのバージョン 4.1 以降、.tflite はデフォルトで noCompress リストに追加され、上記の aaptOptions は不要になりました。

### ステップ 2: モデルを使用する

```java
// Initialization
ObjectDetectorOptions options =
    ObjectDetectorOptions.builder()
        .setBaseOptions(BaseOptions.builder().useGpu().build())
        .setMaxResults(1)
        .build();
ObjectDetector objectDetector =
    ObjectDetector.createFromFileAndOptions(
        context, modelFile, options);

// Run inference
List<Detection> results = objectDetector.detect(image);
```

<code>ObjectDetector</code>を構成するその他のオプションについては、<a>ソースコードと javadoc</a> をご覧ください。

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
guard let modelPath = Bundle.main.path(forResource: "ssd_mobilenet_v1",
                                            ofType: "tflite") else { return }

let options = ObjectDetectorOptions(modelPath: modelPath)

// Configure any additional options:
// options.classificationOptions.maxResults = 3

let detector = try ObjectDetector.detector(options: options)

// Convert the input image to MLImage.
// There are other sources for MLImage. For more details, please see:
// https://developers.google.com/ml-kit/reference/ios/mlimage/api/reference/Classes/GMLImage
guard let image = UIImage (named: "cats_and_dogs.jpg"), let mlImage = MLImage(image: image) else { return }

// Run inference
let detectionResult = try detector.detect(mlImage: mlImage)
```

#### Objective C

```objc
// Imports
#import <TensorFlowLiteTaskVision/TensorFlowLiteTaskVision.h>

// Initialization
NSString *modelPath = [[NSBundle mainBundle] pathForResource:@"ssd_mobilenet_v1" ofType:@"tflite"];

TFLObjectDetectorOptions *options = [[TFLObjectDetectorOptions alloc] initWithModelPath:modelPath];

// Configure any additional options:
// options.classificationOptions.maxResults = 3;

TFLObjectDetector *detector = [TFLObjectDetector objectDetectorWithOptions:options
                                                                     error:nil];

// Convert the input image to MLImage.
UIImage *image = [UIImage imageNamed:@"dogs.jpg"];

// There are other sources for GMLImage. For more details, please see:
// https://developers.google.com/ml-kit/reference/ios/mlimage/api/reference/Classes/GMLImage
GMLImage *gmlImage = [[GMLImage alloc] initWithImage:image];

// Run inference
TFLDetectionResult *detectionResult = [detector detectWithGMLImage:gmlImage error:nil];
```

`TFLObjectDetector` を構成するその他のオプションについては、<a>ソースコード</a>を参照してください。

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
detection_options = processor.DetectionOptions(max_results=2)
options = vision.ObjectDetectorOptions(base_options=base_options, detection_options=detection_options)
detector = vision.ObjectDetector.create_from_options(options)

# Alternatively, you can create an object detector in the following manner:
# detector = vision.ObjectDetector.create_from_file(model_path)

# Run inference
image = vision.TensorImage.create_from_file(image_path)
detection_result = detector.detect(image)
```

<code>ObjectDetector</code>を構成するその他のオプションについては、<a>ソースコード</a>をご覧ください。

## C++ で推論を実行する

```c++
// Initialization
ObjectDetectorOptions options;
options.mutable_base_options()->mutable_model_file()->set_file_name(model_path);
std::unique_ptr<ObjectDetector> object_detector = ObjectDetector::CreateFromOptions(options).value();

// Create input frame_buffer from your inputs, `image_data` and `image_dimension`.
// See more information here: tensorflow_lite_support/cc/task/vision/utils/frame_buffer_common_utils.h
std::unique_ptr<FrameBuffer> frame_buffer = CreateFromRgbRawBuffer(
      image_data, image_dimension);

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

`ObjectDetector` API は、必須の [TFLite モデル メタデータ](../../models/convert/metadata)を持つ TFLite モデルを想定しています。[TensorFlow Lite Metadata Writer API](../../models/convert/metadata_writer_tutorial.ipynb#object_detectors) を使用して物体検出器のメタデータを作成する例をご覧ください。

互換性のある物体検出モデルは、次の要件を満たす必要があります。

- 入力画像テンソル: (kTfLiteUInt8/kTfLiteFloat32)

    - サイズ`[batch x height x width x channels]`の画像入力。
    - バッチ推論はサポートされていません (`batch`は 1 である必要があります)。
    - RGB 入力のみがサポートされています (`channels`は 3 である必要があります)。
    - 型が kTfLiteFloat32 の場合、入力の正規化のためにメタデータに NormalizationOptions をアタッチする必要があります。

- 出力テンソルは、以下のように`DetectionPostProcess`演算の 4 つの出力でなければなりません。

    - 位置テンソル (kTfLiteFloat32)

        - サイズ`[1 x num_results x 4]`のテンソル。[上、左、右、下]の形式で境界矩形を表す内部配列。
        - BoundingBoxProperties はメタデータに添付する必要があり、`type=BOUNDARIES`および `coordinate_type = RATIO を指定する必要があります。

    - クラステンソル (kTfLiteFloat32)

        - サイズ`[1 x num_results]`のテンソル。各値はクラスの整数インデックスを表します。
        - オプション（ただし推奨）のラベルマップを TENSOR_VALUE_LABELS 型の AssociatedFile-s として添付できます。1 行に 1 つのラベルが含まれます。最初の AssociatedFile（存在する場合）は、結果の <code>class_name</code> フィールドを入力するために使用されます。`display_name` フィールドは、AssociatedFile（存在する場合）から入力されます。そのロケールは、作成時に使用される `ObjectDetectorOptions` の `display_names_locale`フィールドと一致します（デフォルトでは「en (英語)」）。これらのいずれも使用できない場合、結果の `index` フィールドのみを使用できます。

    - スコアテンソル (kTfLiteFloat32)

        - サイズ`[1 x num_results]`のテンソル。各値は検出された物体のスコアを表します。

    - 検出テンソル数 (kTfLiteFloat32)

        - テンソルサイズ `[1]`の整数の num_results。
