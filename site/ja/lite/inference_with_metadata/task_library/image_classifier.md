# 画像分類器の統合

画像分類は、一般的な機械学習の使用例で、機械学習を使用して画像が表すものを識別します。たとえば、特定の写真にどのような種類の動物が写っているのかを知りたい場合に画像が何を表すかを予測するタスクは、*画像分類*と呼ばれます。画像分類器は、画像のさまざまなクラスを認識するようにトレーニングされています。たとえば、ウサギ、ハムスター、犬の 3 種類の動物を表す写真を認識するようにモデルをトレーニングすることができます。画像分類器の詳細については、[画像分類の概要](https://www.tensorflow.org/lite/examples/image_classification/overview)をご覧ください。

Task Library `ImageClassifier` APIを使用して、カスタム画像分類器または事前トレーニング済みのものを モバイルアプリ.

## ImageClassifier API の主な機能

- 回転、サイズ変更、色空間変換などの入力画像処理。

- 入力画像の関心領域。

- マップロケールのラベル付け。

- 結果をフィルタリングするスコアしきい値。

- Top-k 分類結果。

- 許可リストと拒否リストのラベルを付け。

## サポートされている画像分類モデル

次のモデルは、`ImageClassifier` API との互換性が保証されています。

- [TensorFlow Lite Model Maker による画像分類](https://www.tensorflow.org/lite/models/modify/model_maker/image_classification)によって作成されたモデル。

- [TensorFlow Hub の事前トレーニング済み画像分類モデル](https://tfhub.dev/tensorflow/collections/lite/task-library/image-classifier/1)。

- [AutoML Vision Edge 画像分類](https://cloud.google.com/vision/automl/docs/edge-quickstart)によって作成されたモデル。

- [モデルの互換性要件](#model-compatibility-requirements)を満たすカスタムモデル。

## Java で推論を実行する

[Image Classification reference app](https://github.com/tensorflow/examples/blob/master/lite/examples/image_classification/android/README.md) 画像分類リファレンスアプリを参照し、Android アプリで `ImageClassifier` を使用する方法の例をご覧ください。

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

### ステップ 2: モデルを使用する

```java
// Initialization
ImageClassifierOptions options =
    ImageClassifierOptions.builder()
        .setBaseOptions(BaseOptions.builder().useGpu().build())
        .setMaxResults(1)
        .build();
ImageClassifier imageClassifier =
    ImageClassifier.createFromFileAndOptions(
        context, modelFile, options);

// Run inference
List<Classifications> results = imageClassifier.classify(image);
```

[ソースコードと javadoc](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/java/src/java/org/tensorflow/lite/task/vision/classifier/ImageClassifier.java) を参照し、`ImageClassifier` を構成するその他のオプションについてご覧ください。

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
guard let modelPath = Bundle.main.path(forResource: "birds_V1",
                                            ofType: "tflite") else { return }

let options = ImageClassifierOptions(modelPath: modelPath)

// Configure any additional options:
// options.classificationOptions.maxResults = 3

let classifier = try ImageClassifier.classifier(options: options)

// Convert the input image to MLImage.
// There are other sources for MLImage. For more details, please see:
// https://developers.google.com/ml-kit/reference/ios/mlimage/api/reference/Classes/GMLImage
guard let image = UIImage (named: "sparrow.jpg"), let mlImage = MLImage(image: image) else { return }

// Run inference
let classificationResults = try classifier.classify(mlImage: mlImage)
```

#### Objective C

```objc
// Imports
#import <TensorFlowLiteTaskVision/TensorFlowLiteTaskVision.h>

// Initialization
NSString *modelPath = [[NSBundle mainBundle] pathForResource:@"birds_V1" ofType:@"tflite"];

TFLImageClassifierOptions *options =
    [[TFLImageClassifierOptions alloc] initWithModelPath:modelPath];

// Configure any additional options:
// options.classificationOptions.maxResults = 3;

TFLImageClassifier *classifier = [TFLImageClassifier imageClassifierWithOptions:options
                                                                          error:nil];

// Convert the input image to MLImage.
UIImage *image = [UIImage imageNamed:@"sparrow.jpg"];

// There are other sources for GMLImage. For more details, please see:
// https://developers.google.com/ml-kit/reference/ios/mlimage/api/reference/Classes/GMLImage
GMLImage *gmlImage = [[GMLImage alloc] initWithImage:image];

// Run inference
TFLClassificationResult *classificationResult =
    [classifier classifyWithGMLImage:gmlImage error:nil];
```

<code>TFLImageClassifier</code>を構成するその他のオプションについては、<a>ソースコード</a>をご覧ください。

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
classification_options = processor.ClassificationOptions(max_results=2)
options = vision.ImageClassifierOptions(base_options=base_options, classification_options=classification_options)
classifier = vision.ImageClassifier.create_from_options(options)

# Alternatively, you can create an image classifier in the following manner:
# classifier = vision.ImageClassifier.create_from_file(model_path)

# Run inference
image = vision.TensorImage.create_from_file(image_path)
classification_result = classifier.classify(image)
```

<code>ImageClassifier</code>を構成するその他のオプションについては、<a>ソースコード</a>をご覧ください。

## C++ で推論を実行する

```c++
// Initialization
ImageClassifierOptions options;
options.mutable_base_options()->mutable_model_file()->set_file_name(model_path);
std::unique_ptr<ImageClassifier> image_classifier = ImageClassifier::CreateFromOptions(options).value();

// Create input frame_buffer from your inputs, `image_data` and `image_dimension`.
// See more information here: tensorflow_lite_support/cc/task/vision/utils/frame_buffer_common_utils.h

std::unique_ptr<FrameBuffer> frame_buffer = CreateFromRgbRawBuffer(
      image_data, image_dimension);

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

`ImageClassifier` API は、必須の [TFLite モデル メタデータ](https://www.tensorflow.org/lite/models/convert/metadata)を持つ TFLite モデルを想定しています。[TensorFlow Lite Metadata Writer API](../../models/convert/metadata_writer_tutorial.ipynb#image_classifiers) を使用して画像セグメンタのメタデータを作成する例をご覧ください。

互換性のある画像分類モデルは、次の要件を満たす必要があります。

- 入力画像テンソル (kTfLiteUInt8/kTfLiteFloat32)

    - サイズ`[batch x height x width x channels]`の画像入力。
    - バッチ推論はサポートされていません (`batch`は 1 である必要があります)。
    - RGB 入力のみがサポートされています (`channels`は 3 である必要があります)。
    - 型が kTfLiteFloat32 の場合、入力の正規化のためにメタデータに NormalizationOptions をアタッチする必要があります。

- 出力画像テンソル (kTfLiteUInt8/kTfLiteFloat32)

    - `N` クラスと 2 次元または 4 次元のいずれか (`[1 x N]`または`[1 x 1 x 1 x N]`)
    - TENSOR_AXIS_LABELS 型の AssociatedFile-s ラベルマップ (オプションですが推薦されます)。1 行に 1 つのラベルが含まれます。[サンプルラベルファイル](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/metadata/python/tests/testdata/image_classifier/labels.txt)をご覧ください。最初の AssociatedFile（存在する場合）は、結果の <code>label</code> フィールド（C++ では `class_name` と名付けられています）を入力ために使用されます。`display_name` フィールドは、AssociatedFile（存在する場合）から入力されます。そのロケールは、作成時に使用される `ImageClassifierOptions` の `display_names_locale` フィールドと一致します（デフォルトでは「en（英語）」）。これらのいずれも使用できない場合、結果の `index` フィールドのみが入力されます。
