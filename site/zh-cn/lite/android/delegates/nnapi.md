# TensorFlow Lite NNAPI 委托

[Android Neural Networks API (NNAPI)](https://developer.android.com/ndk/guides/neuralnetworks) 在所有运行 Android 8.1（API 级别 27）或更高版本的 Android 设备上可用。在具有支持的硬件加速器的 Android 设备上，它可以为 TensorFlow Lite 模型提供加速。支持的硬件加速器包括：

- 图形处理单元 (GPU)
- 数字信号处理器 (DSP)
- 神经处理单元 (NPU)

根据设备上可用的特定硬件，性能可能有所不同。

本页介绍在 Java 和 Kotlin 中如何将 NNAPI 委托与 TensorFlow Lite 解释器结合使用。对于 Android C API，请参阅 [Android Native Development Kit 文档](https://developer.android.com/ndk/guides/neuralnetworks)。

## 在自己的模型上尝试 NNAPI 委托

### Gradle 导入

The NNAPI delegate is part of the TensorFlow Lite Android interpreter, release 1.14.0 or higher. You can import it to your project by adding the following to your module gradle file:

```groovy
dependencies {
   implementation 'org.tensorflow:tensorflow-lite:2.0.0'
}
```

### 初始化 NNAPI 委托

Add the code to initialize the NNAPI delegate before you initialize the TensorFlow Lite interpreter.

注：虽然从 API 级别 27 (Android Oreo MR1) 开始就支持 NNAPI，但是，在 API 级别 28 (Android Pie) 及以后的版本上，对运算的支持大有改善。因此，对于大多数情形，我们建议开发者为 Android Pie 或更高的版本使用 NNAPI 委托。

```java
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.nnapi.NnApiDelegate;

Interpreter.Options options = (new Interpreter.Options());
NnApiDelegate nnApiDelegate = null;
// Initialize interpreter with NNAPI delegate for Android Pie or above
if(Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
    nnApiDelegate = new NnApiDelegate();
    options.addDelegate(nnApiDelegate);
}

// Initialize TFLite interpreter
try {
    tfLite = new Interpreter(loadModelFile(assetManager, modelFilename), options);
} catch (Exception e) {
    throw new RuntimeException(e);
}

// Run inference
// ...

// Unload delegate
tfLite.close();
if(null != nnApiDelegate) {
    nnApiDelegate.close();
}
```

## 最佳做法

### 在部署前测试性能

Runtime performance can vary significantly due to model architecture, size, operations, hardware availability, and runtime hardware utilization. For example, if an app heavily utilizes the GPU for rendering, NNAPI acceleration may not improve performance due to resource contention. We recommend running a simple performance test using the debug logger to measure inference time. Run the test on several phones with different chipsets (manufacturer or models from the same manufacturer) that are representative of your user base before enabling NNAPI in production.

For advanced developers, TensorFlow Lite also offers [a model benchmark tool for Android](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark).

### 创建设备排除列表

In production, there may be cases where NNAPI does not perform as expected. We recommend developers maintain a list of devices that should not use NNAPI acceleration in combination with particular models. You can create this list based on the value of `"ro.board.platform"`, which you can retrieve using the following code snippet:

```java
String boardPlatform = "";

try {
    Process sysProcess =
        new ProcessBuilder("/system/bin/getprop", "ro.board.platform").
        redirectErrorStream(true).start();

    BufferedReader reader = new BufferedReader
        (new InputStreamReader(sysProcess.getInputStream()));
    String currentLine = null;

    while ((currentLine=reader.readLine()) != null){
        boardPlatform = line;
    }
    sysProcess.destroy();
} catch (IOException e) {}

Log.d("Board Platform", boardPlatform);
```

For advanced developers, consider maintaining this list via a remote configuration system. The TensorFlow team is actively working on ways to simplify and automate discovering and applying the optimal NNAPI configuration.

### 量化

Quantization reduces model size by using 8-bit integers or 16-bit floats instead of 32-bit floats for computation. 8-bit integer model sizes are a quarter of the 32-bit float versions; 16-bit floats are half of the size. Quantization can improve performance significantly though the process could trade off some model accuracy.

目前有多种训练后量化技术可用，但是为了在最新的硬件上获得最佳支持和加速，我们建议使用[全整数量化](post_training_quantization#full_integer_quantization_of_weights_and_activations)。这种方式会将权重和运算都转换成整数。此量化过程需要一个代表数据集才能运行。

### 使用支持的模型和运算

If the NNAPI delegate does not support some of the ops or parameter combinations in a model, the framework only runs the supported parts of the graph on the accelerator. The remainder runs on the CPU, which results in split execution. Due to the high cost of CPU/accelerator synchronization, this may result in slower performance than executing the whole network on the CPU alone.

NNAPI performs best when models only use [supported ops](https://developer.android.com/ndk/guides/neuralnetworks#model). The following models are known to be compatible with NNAPI:

- [MobileNet v1 (224x224) image classification (float model download)](https://ai.googleblog.com/2017/06/mobilenets-open-source-models-for.html)[(quantized model download)](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224_quant.tgz)
    *(image classification model designed for mobile and embedded based vision
    applications)*
- [MobileNet v2 SSD object detection](https://ai.googleblog.com/2018/07/accelerated-training-and-inference-with.html)[(download)](https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/mobile_ssd_v2_float_coco.tflite)
    *(image classification model that detects multiple objects with bounding
    boxes)*
- [MobileNet v1(300x300) Single Shot Detector (SSD) object detection](https://ai.googleblog.com/2018/07/accelerated-training-and-inference-with.html) [(download)] (https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip)
- [PoseNet for pose estimation](https://github.com/tensorflow/tfjs-models/tree/master/posenet)[(download)](https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/multi_person_mobilenet_v1_075_float.tflite)
    *(vision model that estimates the poses of a person(s) in image or video)*

NNAPI acceleration is also not supported when the model contains dynamically-sized outputs. In this case, you will get a warning like:

```none
ERROR: Attempting to use a delegate that only supports static-sized tensors \
with a graph that has dynamic-sized tensors.
```

### Enable NNAPI CPU implementation

A graph that can't be processed completely by an accelerator can fall back to the NNAPI CPU implementation. However, since this is typically less performant than the TensorFlow interpreter, this option is disabled by default in the NNAPI delegate for Android 10 (API Level 29) or above. To override this behavior, set `setUseNnapiCpu` to `true` in the `NnApiDelegate.Options` object.
