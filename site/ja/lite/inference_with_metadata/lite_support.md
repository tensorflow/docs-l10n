# TensorFlow Lite サポートライブラリを使用して入力データと出力データを処理する

注意: TensorFlow Lite サポートライブラリは現在、Android のみをサポートしています。

Mobile application developers typically interact with typed objects such as bitmaps or primitives such as integers. However, the TensorFlow Lite interpreter API that runs the on-device machine learning model uses tensors in the form of ByteBuffer, which can be difficult to debug and manipulate. The [TensorFlow Lite Android Support Library](https://github.com/tensorflow/tflite-support/tree/master/tensorflow_lite_support/java) is designed to help process the input and output of TensorFlow Lite models, and make the TensorFlow Lite interpreter easier to use.

## はじめに

### Gradle の依存関係とその他の設定をインポートする

`.tflite` モデルファイルを、モデルが実行される Android モジュールのアセットディレクトリにコピーします。ファイルを圧縮しないように指定し、TensorFlow Lite ライブラリをモジュールの `build.gradle` ファイルに追加します。

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

    // Import tflite dependencies
    implementation 'org.tensorflow:tensorflow-lite:0.0.0-nightly-SNAPSHOT'
    // The GPU delegate library is optional. Depend on it as needed.
    implementation 'org.tensorflow:tensorflow-lite-gpu:0.0.0-nightly-SNAPSHOT'
    implementation 'org.tensorflow:tensorflow-lite-support:0.0.0-nightly-SNAPSHOT'
}
```

注：Android Gradle プラグインのバージョン 4.1 以降、.tflite はデフォルトで noCompress リストに追加され、上記の aaptOptions は不要になりました。

Explore the [TensorFlow Lite Support Library AAR hosted at MavenCentral](https://search.maven.org/artifact/org.tensorflow/tensorflow-lite-support) for different versions of the Support Library.

### 基本的な画像の操作と変換

TensorFlow Lite サポートライブラリには、トリミングやサイズ変更などの基本的な画像操作メソッド一式が含まれています。このようなメソッドを使用するには、`ImagePreprocessor` を作成し、必要な操作を追加します。画像を TensorFlow Lite インタープリターに必要なテンソル形式に変換するには、次のように入力として使用する `TensorImage` を作成します。

```java
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;

// Initialization code
// Create an ImageProcessor with all ops required. For more ops, please
// refer to the ImageProcessor Architecture section in this README.
ImageProcessor imageProcessor =
    new ImageProcessor.Builder()
        .add(new ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR))
        .build();

// Create a TensorImage object. This creates the tensor of the corresponding
// tensor type (uint8 in this case) that the TensorFlow Lite interpreter needs.
TensorImage tensorImage = new TensorImage(DataType.UINT8);

// Analysis code for every frame
// Preprocess the image
tensorImage.load(bitmap);
tensorImage = imageProcessor.process(tensorImage);
```

`DataType` of a tensor can be read through the [metadata extractor library](../models/convert/metadata.md#read-the-metadata-from-models) as well as other model information.

### Basic audio data processing

The TensorFlow Lite Support Library also defines a `TensorAudio` class wrapping some basic audio data processing methods. It's mostly used together with [AudioRecord](https://developer.android.com/reference/android/media/AudioRecord) and captures audio samples in a ring buffer.

```java
import android.media.AudioRecord;
import org.tensorflow.lite.support.audio.TensorAudio;

// Create an `AudioRecord` instance.
AudioRecord record = AudioRecord(...)

// Create a `TensorAudio` object from Android AudioFormat.
TensorAudio tensorAudio = new TensorAudio(record.getFormat(), size)

// Load all audio samples available in the AudioRecord without blocking.
tensorAudio.load(record)

// Get the `TensorBuffer` for inference.
TensorBuffer buffer = tensorAudio.getTensorBuffer()
```

### 出力オブジェクトを作成してモデルを実行する

モデルを実行する前に、次のように結果を格納するコンテナオブジェクトを作成する必要があります。

```java
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

// Create a container for the result and specify that this is a quantized model.
// Hence, the 'DataType' is defined as UINT8 (8-bit unsigned integer)
TensorBuffer probabilityBuffer =
    TensorBuffer.createFixedSize(new int[]{1, 1001}, DataType.UINT8);
```

次のようにモデルをロードして推論を実行します。

```java
import java.nio.MappedByteBuffer;
import org.tensorflow.lite.InterpreterFactory;
import org.tensorflow.lite.InterpreterApi;

// Initialise the model
try{
    MappedByteBuffer tfliteModel
        = FileUtil.loadMappedFile(activity,
            "mobilenet_v1_1.0_224_quant.tflite");
    InterpreterApi tflite = new InterpreterFactory().create(
        tfliteModel, new InterpreterApi.Options());
} catch (IOException e){
    Log.e("tfliteSupport", "Error reading model", e);
}

// Running inference
if(null != tflite) {
    tflite.run(tImage.getBuffer(), probabilityBuffer.getBuffer());
}
```

### 結果にアクセスする

開発者は `probabilityBuffer.getFloatArray()` を介して出力に直接アクセスできます。モデルが量子化された出力を生成する場合は、必ず結果を変換するようにしてください。MobileNet 量子化モデルの場合、開発者は各出力値を 255 で割り、各カテゴリに対する 0（最も可能性が低い）から 1（最も可能性が高い）の範囲の確率を取得する必要があります。

### 任意: 結果をラベルにマッピングする

開発者は必要に応じて結果をラベルにマップすることもできます。まず、ラベルを含むテキストファイルをモジュールのアセットディレクトリにコピーします。次に、次のコードを使用してラベルファイルをロードします。

```java
import org.tensorflow.lite.support.common.FileUtil;

final String ASSOCIATED_AXIS_LABELS = "labels.txt";
List<String> associatedAxisLabels = null;

try {
    associatedAxisLabels = FileUtil.loadLabels(this, ASSOCIATED_AXIS_LABELS);
} catch (IOException e) {
    Log.e("tfliteSupport", "Error reading label file", e);
}
```

次のスニペットは、確率をカテゴリラベルに関連付ける方法を示しています。

```java
import java.util.Map;
import org.tensorflow.lite.support.common.TensorProcessor;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.label.TensorLabel;

// Post-processor which dequantize the result
TensorProcessor probabilityProcessor =
    new TensorProcessor.Builder().add(new NormalizeOp(0, 255)).build();

if (null != associatedAxisLabels) {
    // Map of labels and their corresponding probability
    TensorLabel labels = new TensorLabel(associatedAxisLabels,
        probabilityProcessor.process(probabilityBuffer));

    // Create a map to access the result based on label
    Map<String, Float> floatMap = labels.getMapWithFloatValue();
}
```

## 現在のユースケース対応状況

TensorFlow Lite サポートライブラリの現在のバージョンは以下に対応しています。

- common data types (float, uint8, images, audio and array of these objects) as inputs and outputs of tflite models.
- 基本的な画像操作（画像のトリミング、サイズ変更、回転）。
- 正規化と量子化
- ファイルユーティリティ

今後のバージョンでは、テキスト関連のアプリケーションのサポートが改善される予定です。

## ImageProcessor のアーキテクチャ

The design of the `ImageProcessor` allowed the image manipulation operations to be defined up front and optimised during the build process. The `ImageProcessor` currently supports three basic preprocessing operations, as described in the three comments in the code snippet below:

```java
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.common.ops.QuantizeOp;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp;
import org.tensorflow.lite.support.image.ops.Rot90Op;

int width = bitmap.getWidth();
int height = bitmap.getHeight();

int size = height > width ? width : height;

ImageProcessor imageProcessor =
    new ImageProcessor.Builder()
        // Center crop the image to the largest square possible
        .add(new ResizeWithCropOrPadOp(size, size))
        // Resize using Bilinear or Nearest neighbour
        .add(new ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR));
        // Rotation counter-clockwise in 90 degree increments
        .add(new Rot90Op(rotateDegrees / 90))
        .add(new NormalizeOp(127.5, 127.5))
        .add(new QuantizeOp(128.0, 1/128.0))
        .build();
```

See more details [here](../models/convert/metadata.md#normalization-and-quantization-parameters) about normalization and quantization.

このサポートライブラリの最終的な目標は、すべての [`tf.image`](https://www.tensorflow.org/api_docs/python/tf/image) 変換をサポートすることです。つまり、変換が TensorFlow と同じものになり、その実装がオペレーティングシステムに依存しないものになることを目指しています。

開発者は独自のプロセッサを作成することもできます。その場合はトレーニングプロセスとの整合性を図ることが重要です。つまり、トレーニングと推論の両方に同じ前処理を適用して再現性を高める必要があります。

## 量子化

`TensorImage` または `TensorBuffer` のような入力オブジェクトや出力オブジェクトを初期化する場合、その型を `DataType.UINT8` または `DataType.FLOAT32` に指定する必要があります。

```java
TensorImage tensorImage = new TensorImage(DataType.UINT8);
TensorBuffer probabilityBuffer =
    TensorBuffer.createFixedSize(new int[]{1, 1001}, DataType.UINT8);
```

`TensorProcessor` は、入力テンソルの量子化や出力テンソルの非量子化を目的に使用できます。たとえば量子化された出力 `TensorBuffer` を処理する場合、開発者は結果を 0 から 1 の間の浮動小数点確率に逆量子化する目的で `DequantizeOp` を使用できます。

```java
import org.tensorflow.lite.support.common.TensorProcessor;

// Post-processor which dequantize the result
TensorProcessor probabilityProcessor =
    new TensorProcessor.Builder().add(new DequantizeOp(0, 1/255.0)).build();
TensorBuffer dequantizedBuffer = probabilityProcessor.process(probabilityBuffer);
```

The quantization parameters of a tensor can be read through the [metadata extractor library](../models/convert/metadata.md#read-the-metadata-from-models).
