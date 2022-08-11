# TensorFlow Lite NNAPI 委托

[Android Neural Networks API (NNAPI)](https://developer.android.com/ndk/guides/neuralnetworks) 在所有运行 Android 8.1（API 级别 27）或更高版本的 Android 设备上可用。在具有支持的硬件加速器的 Android 设备上，它可以为 TensorFlow Lite 模型提供加速。支持的硬件加速器包括：

- 图形处理单元 (GPU)
- 数字信号处理器 (DSP)
- 神经处理单元 (NPU)

根据设备上可用的特定硬件，性能可能有所不同。

本页介绍在 Java 和 Kotlin 中如何将 NNAPI 委托与 TensorFlow Lite 解释器结合使用。对于 Android C API，请参阅 [Android Native Development Kit 文档](https://developer.android.com/ndk/guides/neuralnetworks)。

## 在自己的模型上尝试 NNAPI 委托

### Gradle 导入

NNAPI 委托是 TensorFlow Lite Android 解释器（1.14.0 或更高版本）的一部分。您可以通过将以下代码添加到模块的 Gradle 文件，将其导入项目：

```groovy
dependencies {
   implementation 'org.tensorflow:tensorflow-lite:2.0.0'
}
```

### 初始化 NNAPI 委托

添加以下代码，先初始化 NNAPI 委托，然后再初始化 TensorFlow Lite 解释器。

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

由于模型架构、大小、运算、硬件可用性和运行时硬件利用率不同，运行时性能可能会有显著差异。例如，如果某个应用使用大量 GPU 资源进行渲染，则 NNAPI 加速可能因资源竞争而无法改善性能。我们建议使用调试记录器运行简单的性能测试，以便衡量推断时间。在正式环境中启用 NNAPI 之前，先在代表您的用户群体的几部手机上运行测试。这些手机具有不同的芯片组（来自不同制造商或同一制造商的不同型号）。

对于高级开发者，TensorFlow Lite 还提供了一个[适用于 Android 的模型基准测试工具](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark)。

### 创建设备排除列表

在正式环境中，NNAPI 可能出现无法按预期执行的情况。我们建议开发者维护一个不应将 NNAPI 加速与特定型号结合使用的设备列表。您可以根据 `"ro.board.platform"` 的值创建此列表。使用以下代码段可以检索该值：

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

对于高级开发者，可以考虑通过远程配置系统维护此列表。TensorFlow 团队正在积极研究简化和自动化发现并应用最佳 NNAPI 配置的方式。

### 量化

通过为计算使用 8 位整数或 16 位浮点数（而不是 32 位浮点数），量化可以缩减模型大小。8 位整数模型的大小是 32 位浮点版本的四分之一；16 位浮点版本的大小则为其一半。量化可以显著提高性能，不过这可能对模型的准确率有一定影响。

目前有多种训练后量化技术可用，但是为了在最新的硬件上获得最佳支持和加速，我们建议使用[全整数量化](post_training_quantization#full_integer_quantization_of_weights_and_activations)。这种方式会将权重和运算都转换成整数。此量化过程需要一个代表数据集才能运行。

### 使用支持的模型和运算

如果 NNAPI 委托不支持模型中的某些运算或参数组合，则框架只会在加速器上运行受支持的计算图部分。剩下的计算图将在 CPU 上运行，这会产生执行拆分。由于 CPU/加速器同步的开销很大，因此，这会导致性能比完全在 CPU 上执行整个网络时更低。

当模型仅使用[支持的运算](https://developer.android.com/ndk/guides/neuralnetworks#model)时，NNAPI 表现最佳。已知下面的模型与 NNAPI 兼容：

- [MobileNet v1 (224x224) 图像分类（浮点模型下载）](https://ai.googleblog.com/2017/06/mobilenets-open-source-models-for.html) [（量化模型下载）](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224_quant.tgz) <br> *（专为基于移动设备和嵌入式设备视觉应用设计的图像分类模型）*
- [MobileNet v2 SSD 目标检测](https://ai.googleblog.com/2018/07/accelerated-training-and-inference-with.html) [（下载）](https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/mobile_ssd_v2_float_coco.tflite) <br> *（使用边界框检测多个目标的图像分类模型）*
- [MobileNet v1(300x300) Single Shot Detector (SSD) object detection](https://ai.googleblog.com/2018/07/accelerated-training-and-inference-with.html) [(download)] (https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip)
- [用于姿态预测的 PoseNet](https://github.com/tensorflow/tfjs-models/tree/master/posenet) [（下载）](https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/multi_person_mobilenet_v1_075_float.tflite) <br> *（预测图像或视频中人物姿态的视觉模型）*

NNAPI acceleration is also not supported when the model contains dynamically-sized outputs. In this case, you will get a warning like:

```none
ERROR: Attempting to use a delegate that only supports static-sized tensors \
with a graph that has dynamic-sized tensors.
```

### Enable NNAPI CPU implementation

A graph that can't be processed completely by an accelerator can fall back to the NNAPI CPU implementation. However, since this is typically less performant than the TensorFlow interpreter, this option is disabled by default in the NNAPI delegate for Android 10 (API Level 29) or above. To override this behavior, set `setUseNnapiCpu` to `true` in the `NnApiDelegate.Options` object.
