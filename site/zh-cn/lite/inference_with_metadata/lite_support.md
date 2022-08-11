# 使用 TensorFlow Lite Support Library 处理输入和输出数据

注：TensorFlow Lite Support Library 目前只支持 Android。

移动应用开发者通常会与类型化的对象（如位图）或基元（如整数）进行交互。然而，在设备端运行机器学习模型的 TensorFlow Lite Interpreter API 使用的是 ByteBuffer 形式的张量，可能难以调试和操作。[TensorFlow Lite Android Support Library](https://github.com/tensorflow/tflite-support/tree/master/tensorflow_lite_support/java) 旨在帮助处理 TensorFlow Lite 模型的输入和输出，并使 TensorFlow Lite 解释器更易于使用。

## 开始

### 导入 Gradle 依赖项和其他设置

将 `.tflite` 模型文件复制到将要运行模型的 Android 模块的资源目录下。指定不压缩该文件，并将 TensorFlow Lite 库添加到模块的 `build.gradle` 文件中。

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

注：从 Android Gradle 插件的 4.1 版开始，默认情况下，.tflite 将被添加到 noCompress 列表中，不再需要上面的 aaptOptions。

探索[托管在 MavenCentral 上的 TensorFlow Lite Support Library AAR](https://search.maven.org/artifact/org.tensorflow/tensorflow-lite-support)，以获取不同版本的 Support Library。

### 基本的图像处理和转换

TensorFlow Lite Support Library 有一套基本的图像处理方法，如裁剪和调整大小。要使用它，请创建 `ImagePreprocessor`，并添加所需的运算。要将图像转换为 TensorFlow Lite 解释器所需的张量格式，请创建 `TensorImage` 用作输入：

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

张量的 `DataType` 可通过 [Metadata Exractor 库](../models/convert/metadata.md#read-the-metadata-from-models)和其他模型信息进行读取。

### 基本音频数据处理

TensorFlow Lite Support Library 还定义了一个 `TensorAudio` 类，该类封装了一些基本的音频数据处理方法。它主要和 [AudioRecord](https://developer.android.com/reference/android/media/AudioRecord) 配合使用，在环形缓冲区中捕获音频样本。

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

### 创建输出对象并运行模型

在运行模型之前，我们需要创建用于存储结果的容器对象：

```java
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

// Create a container for the result and specify that this is a quantized model.
// Hence, the 'DataType' is defined as UINT8 (8-bit unsigned integer)
TensorBuffer probabilityBuffer =
    TensorBuffer.createFixedSize(new int[]{1, 1001}, DataType.UINT8);
```

加载模型并运行推断：

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

### 访问结果

开发者可以直接通过 `probabilityBuffer.getFloatArray()` 访问输出。如果模型产生了量化输出，记得要将结果进行转换。对于 MobileNet 量化模型，开发者需要将每个输出值除以 255，以获得每个类别从 0（最不可能）到 1（最有可能）的概率。

### 可选：将结果映射到标签

开发者还可以选择将结果映射到标签。首先，将包含标签的文本文件复制到模块的资源目录中。接下来，使用以下代码加载标签文件：

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

以下代码段演示了如何将概率与类别标签关联起来：

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

## 当前用例覆盖范围

当前版本的 TensorFlow Lite Support Library 涵盖了以下内容：

- 常见的数据类型（浮点、uint8、图像、音频，以及这些对象的数组）作为 tflite 模型的输入和输出。
- 基本的图像运算（裁剪图像，调整大小和旋转）。
- 归一化和量化
- 文件实用工具

未来的版本将改进对文本相关应用的支持。

## ImageProcessor 架构

`ImageProcessor` 的设计允许预先定义图像处理运算，并在构建过程中进行优化。code1}ImageProcessor 目前支持三种基本的预处理运算，如下面代码段中的三条注释所述：

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

请在[此处](../models/convert/metadata.md#normalization-and-quantization-parameters)参阅有关归一化和量化的详细信息。

支持库的最终目标是支持所有 [`tf.image`](https://www.tensorflow.org/api_docs/python/tf/image) 转换。这意味着转换将与 TensorFlow 相同，且实现将独立于操作系统。

我们还欢迎开发者创建自定义处理程序。在这些情况下，与训练过程保持一致很重要，即相同的预处理应同时适用于训练和推断，以提高可重现性。

## 量化

初始化类似 `TensorImage` 或 `TensorBuffer` 的输入或输出对象时，需要将它们的类型指定为 `DataType.UINT8` 或`DataType.FLOAT32`。

```java
TensorImage tensorImage = new TensorImage(DataType.UINT8);
TensorBuffer probabilityBuffer =
    TensorBuffer.createFixedSize(new int[]{1, 1001}, DataType.UINT8);
```

`TensorProcessor` 可以用来量化输入张量或去量化输出张量。例如，当处理量化的输出 `TensorBuffer` 时，开发者可以使用 `DequantizeOp` 将结果去量化为 0 和 1 之间的浮点概率：

```java
import org.tensorflow.lite.support.common.TensorProcessor;

// Post-processor which dequantize the result
TensorProcessor probabilityProcessor =
    new TensorProcessor.Builder().add(new DequantizeOp(0, 1/255.0)).build();
TensorBuffer dequantizedBuffer = probabilityProcessor.process(probabilityBuffer);
```

张量的量化参数可以通过 [Metadata Exractor 库](../models/convert/metadata.md#read-the-metadata-from-models)来读取。
