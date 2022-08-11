# TensorFlow Lite GPU 代理

[TensorFlow Lite](https://www.tensorflow.org/lite) 支持多种硬件加速器。本文档介绍了如何通过 TensorFlow Lite 委托 API 在 Android 和 iOS 上使用 GPU 后端。

GPU 采用高吞吐量式设计，可处理大规模可并行化的工作负载。因此，它们非常适合包含大量算子的深度神经网络，每个算子都会处理一个或多个输入张量，可以轻松地划分为较小的工作负载且并行执行，这通常可以降低延迟。在最佳情况下，GPU 上的推断速度现已足够快，适用于以前无法实现的实时应用。

与 CPU 不同，GPU 支持 16 位或 32 位浮点数运算，并且无需量化即可获得最佳性能。委托确实接受 8 位量化模型，但是将以浮点数进行计算。请参阅[高级文档](gpu_advanced.md)以了解详情。

GPU 推断的另一个优势是其功效。GPU 以非常高效且经优化的方式执行计算，因此与在 CPU 上执行相同任务时相比，GPU 的功耗和产生的热量更低。

## 演示应用教程

尝试 GPU 委托的最简单方式就是跟随以下教程操作，教程将贯穿我们整个使用 GPU 构建的分类演示应用。GPU 代码现在只有二进制形式，但是很快就会开源。一旦您了解如何使演示正常运行后，就可以在您的自定义模型上尝试此操作。

### Android（使用 Android Studio）

如果需要分步教程，请观看[适用于 Android 的 GPU 委托](https://youtu.be/Xkhgre8r5G0)视频。

注：要求 OpenCL 或者 OpenGL ESS（3.1 或者更高版本）。

#### 第 1 步. 克隆 TensorFlow 源代码并在 Android Studio 中打开

```sh
git clone https://github.com/tensorflow/tensorflow
```

#### 第 2 步. 编辑 `app/build.gradle` 以使用 Nightly 版本的 GPU AAR

注：您现在可以在清单中使用 **Android S+** with `targetSdkVersion="S"`，或在您的 Gradle `defaultConfig`（API 级别待定）中使用 `targetSdkVersion "S"`。在这种情况下，您应将 [`AndroidManifestGpu.xml`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/java/AndroidManifestGpu.xml) 的内容合并到 Android 应用的清单中。如果没有此更改，GPU 代理将无法访问 OpenCL 库进行加速。*需要 AGP 4.2.0 或更高版本才能运行。*

在现有 `dependencies` 块中现有 `tensorflow-lite` 软件包的位置下添加 `tensorflow-lite-gpu` 软件包。

```
dependencies {
    ...
    implementation 'org.tensorflow:tensorflow-lite:2.3.0'
    implementation 'org.tensorflow:tensorflow-lite-gpu:2.3.0'
}
```

#### 第 3 步. 编译和运行

运行 → 运行“应用”。运行应用时，您会看到一个用于启用 GPU 的按钮。从量化模型转换为浮点模型，然后点击 GPU 以在 GPU 上运行。

![运行 Android GPU 演示并切换到 GPU](https://github.com/tensorflow/docs-l10n/blob/master/site/zh-cn/lite/performance/images/android_gpu_demo.gif?raw=true)

### iOS（使用 XCode）

如果需要分步教程，请观看[适用于 iOS 的 GPU 委托](https://youtu.be/a5H4Zwjp49c)视频。

注：要求 XCode 10.1 或者更高版本。

#### 第 1 步. 获取演示应用的源码并确保它可以编译。

按照我们的 iOS 演示应用[教程](https://www.tensorflow.org/lite/guide/ios)操作。这会向您展示未经修改的 iOS 相机演示应用如何在您的手机上的运行。

#### 第 2 步. 修改 Podfile 文件以使用 TensorFlow Lite GPU CocoaPod

从 2.3.0 版本开始，默认情况下会从 Pod 中排除 GPU 委托，以缩减二进制文件的大小。您可以通过指定子规范来包含 GPU 委托。对于 `TensorFlowLiteSwift` Pod：

```ruby
pod 'TensorFlowLiteSwift/Metal', '~> 0.0.1-nightly',
```

或者

```ruby
pod 'TensorFlowLiteSwift', '~> 0.0.1-nightly', :subspecs => ['Metal']
```

如果您要使用 Objective-C（从 2.4.0 版本开始）或 C API，则可以对 `TensorFlowLiteObjC` 或 `TensorFlowLitC` 进行类似的操作。

<div>
  <devsite-expandable>
    <h4 class="showalways">2.3.0 之前版本</h4>
    <h4>TensorFlow Lite 2.0.0 之前版本</h4>
    <p>我们构建了包含 GPU 委托的二进制 CocoaPod。要切换项目以使用它，请修改 `tensorflow/tensorflow/lite/examples/ios/camera/Podfile` 文件以使用 `TensorFlowLiteGpuExperimental` Pod 而非 `TensorFlowLite`。</p>
    <pre class="prettyprint lang-ruby notranslate" translate="no"><code>
    target 'YourProjectName'
      # pod 'TensorFlowLite', '1.12.0'
      pod 'TensorFlowLiteGpuExperimental'
    </code></pre>
    <h4>TensorFlow Lite 2.2.0 之前版本</h4>
    <p>从 TensorFlow Lite 2.1.0 到 2.2.0 版本，`TensorFlowLiteC` Pod 中已包含 GPU 委托。您可以根据所用语言在 `TensorFlowLiteC` 和 `TensorFlowLiteSwift` 之间进行选择。</p>
  </devsite-expandable>
</div>

#### 第 3 步. 启用 GPU 委托

要启用将使用 GPU 委托的代码，您需要将 `CameraExampleViewController.h` 中的 `TFLITE_USE_GPU_DELEGATE` 从 0 更改为 1。

```c
#define TFLITE_USE_GPU_DELEGATE 1
```

#### 第 4 步. 构建并运行演示应用

按照上一步操作后，您应当能够运行应用。

#### 第 5 步. 发布模式

在第 4 步中以调试模式运行时，为了获得更高性能，您应当更改为具有适当优化 Metal 设置的发布构建。特别是，要编辑这些设置，请转至 `Product > Scheme > Edit Scheme…`。选择 `Run`。在 `Info` 标签页中，将 `Build Configuration` 从 `Debug` 更改为 `Release`，取消选中 `Debug executable`。

![设置 Metal 选项](https://gitlocalize.com/repo/4592/zh-cn/site/en-snapshot/lite/performance/images/iosmetal.png)

然后，点击 `Options` 标签页并将 `GPU Frame Capture` 更改为 `Disabled`，将 `Metal API Validation` 更改为 `Disabled`。

![设置发布](https://gitlocalize.com/repo/4592/zh-cn/site/en-snapshot/lite/performance/images/iosdebug.png)

最后，确保在 64 位架构上选择仅发布构建。在 `Project navigator -> tflite_camera_example -> PROJECT -> tflite_camera_example -> Build Settings` 下，将 `Build Active Architecture Only > Release` 设置为 Yes。

![设置发布选项](https://gitlocalize.com/repo/4592/zh-cn/site/en-snapshot/lite/performance/images/iosrelease.png)

## 在您自己的模型上尝试 GPU 委托

### Android

注：必须在与运行相同的线程上创建 TensorFlow Lite 解释器。否则，可能会发生 `TfLiteGpuDelegate Invoke: GpuDelegate must run on the same thread where it was initialized.`。

您可以通过两种方式调用模型加速，具体取决于您使用的是 [Android Studio 机器学习模型绑定](../inference_with_metadata/codegen#acceleration)还是 TensorFlow Lite 解释器。

#### TensorFlow Lite 解释器

请查看演示以了解如何添加委托。在您的应用中，以上述方式添加 AAR，导入 `org.tensorflow.lite.gpu.GpuDelegate` 模块，然后使用 `addDelegate` 函数将 GPU 委托注册到解释器：

<div>
  <devsite-selector>
    <section>
      <h3>Kotlin</h3>
      <p></p>
<pre class="prettyprint lang-kotlin">    import org.tensorflow.lite.Interpreter
    import org.tensorflow.lite.gpu.CompatibilityList
    import org.tensorflow.lite.gpu.GpuDelegate

    val compatList = CompatibilityList()

    val options = Interpreter.Options().apply{
        if(compatList.isDelegateSupportedOnThisDevice){
            // if the device has a supported GPU, add the GPU delegate
            val delegateOptions = compatList.bestOptionsForThisDevice
            this.addDelegate(GpuDelegate(delegateOptions))
        } else {
            // if the GPU is not supported, run on 4 threads
            this.setNumThreads(4)
        }
    }

    val interpreter = Interpreter(model, options)

    // Run inference
    writeToInput(input)
    interpreter.run(input, output)
    readFromOutput(output)
      </pre>
    </section>
    <section>
      <h3>Java</h3>
      <p></p>
<pre class="prettyprint lang-java">    import org.tensorflow.lite.Interpreter;
    import org.tensorflow.lite.gpu.CompatibilityList;
    import org.tensorflow.lite.gpu.GpuDelegate;

    // Initialize interpreter with GPU delegate
    Interpreter.Options options = new Interpreter.Options();
    CompatibilityList compatList = CompatibilityList();

    if(compatList.isDelegateSupportedOnThisDevice()){
        // if the device has a supported GPU, add the GPU delegate
        GpuDelegate.Options delegateOptions = compatList.getBestOptionsForThisDevice();
        GpuDelegate gpuDelegate = new GpuDelegate(delegateOptions);
        options.addDelegate(gpuDelegate);
    } else {
        // if the GPU is not supported, run on 4 threads
        options.setNumThreads(4);
    }

    Interpreter interpreter = new Interpreter(model, options);

    // Run inference
    writeToInput(input);
    interpreter.run(input, output);
    readFromOutput(output);
      </pre>
    </section>
  </devsite-selector>
</div>

### iOS

注：GPU 委托也可以将 C API 用于 Objective-C 代码。在 TensorFlow Lite 2.4.0 版本之前，这是唯一的选择。

<div>
  <devsite-selector>
    <section>
      <h3>Swift</h3>
      <p></p>
<pre class="prettyprint lang-swift">    import TensorFlowLite

    // Load model ...

    // Initialize TensorFlow Lite interpreter with the GPU delegate.
    let delegate = MetalDelegate()
    if let interpreter = try Interpreter(modelPath: modelPath,
                                         delegates: [delegate]) {
      // Run inference ...
    }
      </pre>
    </section>
    <section>
      <h3>Objective-C</h3>
      <p></p>
<pre class="prettyprint lang-objc">    // Import module when using CocoaPods with module support
    @import TFLTensorFlowLite;

    // Or import following headers manually
    #import "tensorflow/lite/objc/apis/TFLMetalDelegate.h"
    #import "tensorflow/lite/objc/apis/TFLTensorFlowLite.h"

    // Initialize GPU delegate
    TFLMetalDelegate* metalDelegate = [[TFLMetalDelegate alloc] init];

    // Initialize interpreter with model path and GPU delegate
    TFLInterpreterOptions* options = [[TFLInterpreterOptions alloc] init];
    NSError* error = nil;
    TFLInterpreter* interpreter = [[TFLInterpreter alloc]
                                    initWithModelPath:modelPath
                                              options:options
                                            delegates:@[ metalDelegate ]
                                                error:&amp;error];
    if (error != nil) { /* Error handling... */ }

    if (![interpreter allocateTensorsWithError:&amp;error]) { /* Error handling... */ }
    if (error != nil) { /* Error handling... */ }

    // Run inference ...
    ```
      </pre>
    </section>
    <section>
      <h3>C（支持到 2.3.0 版）</h3>
      <p></p>
<pre class="prettyprint lang-c">    #include "tensorflow/lite/c/c_api.h"
    #include "tensorflow/lite/delegates/gpu/metal_delegate.h"

    // Initialize model
    TfLiteModel* model = TfLiteModelCreateFromFile(model_path);

    // Initialize interpreter with GPU delegate
    TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
    TfLiteDelegate* delegate = TFLGPUDelegateCreate(nil);  // default config
    TfLiteInterpreterOptionsAddDelegate(options, metal_delegate);
    TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, options);
    TfLiteInterpreterOptionsDelete(options);

    TfLiteInterpreterAllocateTensors(interpreter);

    NSMutableData *input_data = [NSMutableData dataWithLength:input_size * sizeof(float)];
    NSMutableData *output_data = [NSMutableData dataWithLength:output_size * sizeof(float)];
    TfLiteTensor* input = TfLiteInterpreterGetInputTensor(interpreter, 0);
    const TfLiteTensor* output = TfLiteInterpreterGetOutputTensor(interpreter, 0);

    // Run inference
    TfLiteTensorCopyFromBuffer(input, inputData.bytes, inputData.length);
    TfLiteInterpreterInvoke(interpreter);
    TfLiteTensorCopyToBuffer(output, outputData.mutableBytes, outputData.length);

    // Clean up
    TfLiteInterpreterDelete(interpreter);
    TFLGpuDelegateDelete(metal_delegate);
    TfLiteModelDelete(model);
      </pre>
    </section>
  </devsite-selector>
</div>

## 支持的模型和运算

随着 GPU 委托的发布，我们提供了一些可以在后端运行的模型：

- [MobileNet v1 (224x224) 图像分类](https://ai.googleblog.com/2017/06/mobilenets-open-source-models-for.html) [[下载]](https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/mobilenet_v1_1.0_224.tflite)<br><i>（专为基于移动设备和嵌入式设备视觉应用设计的图像分类模型）</i>
- [DeepLab 分割 (257x257)](https://ai.googleblog.com/2018/03/semantic-image-segmentation-with.html) [[下载]](https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/deeplabv3_257_mv_gpu.tflite)<br><i>（为输入图像中的每个像素分配语义标签（例如狗、猫、汽车）的图像分割模型）</i>
- [MobileNet SSD 目标检测](https://ai.googleblog.com/2018/07/accelerated-training-and-inference-with.html) [[下载]](https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/mobile_ssd_v2_float_coco.tflite)<br><i>（使用边界框检测多个目标的图像分类模型）</i>
- [用于姿势预测的 PoseNet](https://github.com/tensorflow/tfjs-models/tree/master/posenet) [[下载]](https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/multi_person_mobilenet_v1_075_float.tflite)<br><i>（预测图像或视频中人物姿势的视觉模型）</i>

要查看支持的运算的完整列表，请参阅[高级文档](gpu_advanced.md)。

## 不支持的模型和运算

如果某些运算不受 GPU 委托支持，则该框架将仅在 GPU 上运行计算图中的一部分，而在 CPU 上运行其余部分。由于 CPU/GPU 同步的高昂成本，与单独在 CPU 上运行整个网络相比，此类拆分执行模式通常会导致性能下降。在这种情况下，用户将收到类似以下警告：

```none
WARNING: op code #42 cannot be handled by this delegate.
```

我们没有为此错误提供回调，因为这并非真正的运行时错误，而是开发者在尝试在委托上运行网络时可以发现的问题。

## 优化建议

### 针对移动设备进行优化

一些在 CPU 上微不足道的运算可能会为移动设备上的 GPU 带来很高的开销。改造运算的运行成本特别高，包括 `BATCH_TO_SPACE`、`SPACE_TO_BATCH`、`SPACE_TO_DEPTH` 等。您应该仔细检查改造运算的使用情况，并考虑到这些运算可能仅应用于浏览数据或模型的早期迭代。移除它们可以显著提高性能。

在 GPU 上，张量数据被拆分为 4 通道。因此，在形状为 `[B,H,W,5]` 的张量和形状为 `[B,H,W,8]` 的张量上的计算结果大致相同，但明显比 `[B,H,W,4]` 差。从这个意义上说，如果摄像头硬件支持 RGBA 中的图像帧，则馈送 4 通道输入的速度要快得多，因为可以避免内存复制（从 3 通道 RGB 到 4 通道 RGBX）。

为了获得最佳性能，您应该考虑使用移动优化的网络架构重新训练分类器。通过充分利用移动硬件特性，对设备端推断进行优化可以显著降低延迟和功耗。

### 通过序列化缩短初始化时间

GPU委托功能允许您从预编译的内核代码和模型数据中加载序列化的数据，并将其保存在磁盘上，这些数据来自以前的运行。这种方法避免了重新编译，并且最高可将启动时间缩短 90%。有关如何将序列化应用于项目的说明，请参阅 [GPU 委托序列化](gpu_advanced.md#gpu_delegate_serialization)。
