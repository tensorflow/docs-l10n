# TensorFlow Lite サポートライブラリを使用して入力データと出力データを処理する

注意: TensorFlow Lite サポートライブラリは現在、Android のみをサポートしています。

モバイルアプリケーション開発者は通常、ビットマップなどの型指定されたオブジェクトや整数などのプリミティブを操作します。ただし、デバイス上の機械学習モデルを実行する TensorFlow Lite インタープリターは ByteBuffer の形式でテンソルを使用するため、デバッグや操作が難しい場合があります。[TensorFlow Lite Android サポートライブラリ](https://github.com/tensorflow/tflite-support/tree/master/tensorflow_lite_support/java) は、TensorFlow Lite モデルの入力と出力の処理を支援し、TensorFlow Lite インタープリターを使いやすくするように設計されています。

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

各種バージョンのサポートライブラリについては、[JCenter で提供されている TensorFlow Lite サポートライブラリ AAR](https://bintray.com/google/tensorflow/tensorflow-lite-support) を参照してください。

### 基本的な画像の操作と変換

TensorFlow Lite サポートライブラリには、トリミングやサイズ変更などの基本的な画像操作メソッド一式が含まれています。このようなメソッドを使用するには、`ImagePreprocessor` を作成し、必要な操作を追加します。画像を TensorFlow Lite インタープリターに必要なテンソル形式に変換するには、次のように入力として使用する `TensorImage` を作成します。

```java
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
TensorImage tImage = new TensorImage(DataType.UINT8);

// Analysis code for every frame
// Preprocess the image
tImage.load(bitmap);
tImage = imageProcessor.process(tImage);
```

テンソルの `DataType` は[メタデータ抽出ライブラリ](../convert/metadata.md#read-the-metadata-from-models)やその他のモデル情報を介して読み取ることができます。

### 出力オブジェクトを作成してモデルを実行する

モデルを実行する前に、次のように結果を格納するコンテナオブジェクトを作成する必要があります。

```java
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

// Create a container for the result and specify that this is a quantized model.
// Hence, the 'DataType' is defined as UINT8 (8-bit unsigned integer)
TensorBuffer probabilityBuffer =
    TensorBuffer.createFixedSize(new int[]{1, 1001}, DataType.UINT8);
```

次のようにモデルをロードして推論を実行します。

```java
import org.tensorflow.lite.support.model.Model;

// Initialise the model
try{
    MappedByteBuffer tfliteModel
        = FileUtil.loadMappedFile(activity,
            "mobilenet_v1_1.0_224_quant.tflite");
    Interpreter tflite = new Interpreter(tfliteModel)
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
List associatedAxisLabels = null;

try {
    associatedAxisLabels = FileUtil.loadLabels(this, ASSOCIATED_AXIS_LABELS);
} catch (IOException e) {
    Log.e("tfliteSupport", "Error reading label file", e);
}
```

次のスニペットは、確率をカテゴリラベルに関連付ける方法を示しています。

```java
import org.tensorflow.lite.support.common.TensorProcessor;
import org.tensorflow.lite.support.label.TensorLabel;

// Post-processor which dequantize the result
TensorProcessor probabilityProcessor =
    new TensorProcessor.Builder().add(new NormalizeOp(0, 255)).build();

if (null != associatedAxisLabels) {
    // Map of labels and their corresponding probability
    TensorLabel labels = new TensorLabel(associatedAxisLabels,
        probabilityProcessor.process(probabilityBuffer));

    // Create a map to access the result based on label
    Map floatMap = labels.getMapWithFloatValue();
}
```

## 現在のユースケース対応状況

TensorFlow Lite サポートライブラリの現在のバージョンは以下に対応しています。

- tflite モデルの入力および出力としての一般的なデータ型（float、uint8、画像、およびこれらのオブジェクトの配列）。
- 基本的な画像操作（画像のトリミング、サイズ変更、回転）。
- 正規化と量子化
- ファイルユーティリティ

今後のバージョンでは、テキスト関連のアプリケーションのサポートが改善される予定です。

## ImageProcessor のアーキテクチャ

`ImageProcessor` の設計では、事前に画像操作演算を定義し、ビルドプロセス中に最適化できていました。`ImageProcessor` は現在、次のような 3 つの基本的な前処理演算をサポートしています。

```java
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

正規化と量子化の詳細については、[こちら](../convert/metadata.md#normalization-and-quantization-parameters)を参照してください。

このサポートライブラリの最終的な目標は、すべての [`tf.image`](https://www.tensorflow.org/api_docs/python/tf/image) 変換をサポートすることです。つまり、変換が TensorFlow と同じものになり、その実装がオペレーティングシステムに依存しないものになることを目指しています。

開発者は独自のプロセッサを作成することもできます。その場合はトレーニングプロセスとの整合性を図ることが重要です。つまり、トレーニングと推論の両方に同じ前処理を適用して再現性を高める必要があります。

## 量子化

`TensorImage` または `TensorBuffer` のような入力オブジェクトや出力オブジェクトを初期化する場合、その型を `DataType.UINT8` または `DataType.FLOAT32` に指定する必要があります。

```java
TensorImage tImage = new TensorImage(DataType.UINT8);
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

テンソルの量子化パラメーターは、[メタデータ実行ライブラリを](../convert/metadata.md#read-the-metadata-from-models)を介して読み取ることができます。
