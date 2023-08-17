# TensorFlow Lite 的 GPU 委托

使用图形处理单元 (GPU) 运行机器学习 (ML) 模型可以显著改善模型的性能和支持 ML 的应用的用户体验。TensorFlow Lite 支持通过称为[*委托*](./delegates)的硬件驱动程序来使用 GPU 和其他专用处理器。在您的 TensorFlow Lite ML 应用中启用 GPU 可以提供以下好处：

- **速度** - GPU 专为应对大规模并行处理工作负载的高吞吐量而构建。这种设计使它们非常适合由大量算子构成的深度神经网络，其中每个算子都会处理可并行处理的输入张量，这通常会降低延迟。在最佳情况下，在 GPU 上运行模型的速度将足以满足曾无法实现的实时应用对于处理速度的要求。
- **电源效率** - GPU 以非常高效且经优化的方式执行 ML 计算，与在 CPU 上运行相同的任务相比，这种方式通常可以降低功耗和产生的热量。

本文档概述了 TensorFlow Lite 中的 GPU 支持，以及 GPU 处理器的一些高级用途。有关在特定平台上实现 GPU 支持的更多具体信息，请参阅以下指南：

- [Android 平台 GPU 支持](../android/delegates/gpu)
- [iOS 平台 GPU 支持](../ios/delegates/gpu)

## GPU ML 运算支持 {:#supported_ops}

TensorFlow Lite GPU 委托可以加速哪些 TensorFlow ML 运算（或 *op*）存在一些限制。委托支持以下 16 位和 32 位浮点精度的运算：

- `ADD`
- `AVERAGE_POOL_2D`
- `CONCATENATION`
- `CONV_2D`
- `DEPTHWISE_CONV_2D v1-2`
- `EXP`
- `FULLY_CONNECTED`
- `LOGICAL_AND`
- `LOGISTIC`
- `LSTM v2 (Basic LSTM only)`
- `MAX_POOL_2D`
- `MAXIMUM`
- `MINIMUM`
- `MUL`
- `PAD`
- `PRELU`
- `RELU`
- `RELU6`
- `RESHAPE`
- `RESIZE_BILINEAR v1-3`
- `SOFTMAX`
- `STRIDED_SLICE`
- `SUB`
- `TRANSPOSE_CONV`

默认情况下，只有版本 1 支持所有运算。启用[量化支持](#quantized-models)可以允许相应的版本，例如 ADD v2。

### GPU 支持故障排查

如果某些运算不受 GPU 委托支持，则该框架将仅在 GPU 上运行计算图中的一部分，而在 CPU 上运行其余部分。由于 CPU/GPU 同步的高昂成本，与单独在 CPU 上运行整个网络相比，此类拆分执行模式通常会导致性能下降。在这种情况下，应用将生成类似下面所示的警告：

```none
WARNING: op code #42 cannot be handled by this delegate.
```

没有针对此类故障的回调，因为这并非实际运行时故障。在使用 GPU 委托测试模型的执行时，您应该警惕这些警告。出现大量此类警告可能表明您的模型不适合用于 GPU 加速，并且可能需要重构模型。

## 示例模型

我们构建了以下示例模型，它们利用了 TensorFlow Lite 的 GPU 加速，可供参考和测试：

- [MobileNet v1 (224x224) 图像分类](https://ai.googleblog.com/2017/06/mobilenets-open-source-models-for.html) - 专为基于移动设备和嵌入式设备视觉应用设计的图像分类模型。（[模型](https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/classification/5)）
- [DeepLab 分割 (257x257)](https://ai.googleblog.com/2018/03/semantic-image-segmentation-with.html) - 为输入图像中的每个像素分配语义标签（例如狗、猫、汽车）的图像分割模型。（[模型](https://tfhub.dev/tensorflow/lite-model/deeplabv3/1/default/1)）
- [MobileNet SSD 目标检测](https://ai.googleblog.com/2018/07/accelerated-training-and-inference-with.html) - 使用边界框检测多个目标的图像分类模型。（[模型](https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/mobile_ssd_v2_float_coco.tflite)）
- [用于姿态预测的 PoseNet](https://github.com/tensorflow/tfjs-models/tree/master/pose-detection) - 预测图像或视频中人物姿态的视觉模型。（[模型](https://tfhub.dev/tensorflow/lite-model/posenet/mobilenet/float/075/1/default/1)）

## 针对 GPU 进行优化

在使用 TensorFlow Lite GPU 委托在 GPU 硬件上运行模型时，以下技术可以帮助您获得更佳性能：

- **改造运算** - 一些在 CPU 上处理速度较快的运算可能会为移动设备上的 GPU 带来很高的开销。改造运算的运行成本特别高，包括 `BATCH_TO_SPACE、<code data-md-type="codespan">SPACE_TO_BATCH`、`SPACE_TO_DEPTH` 等。您应该仔细检查改造运算的使用情况，并考虑到这些运算可能仅应用于浏览数据或模型的早期迭代。移除它们可以显著提高性能。

- **图像数据通道** - 在 GPU 上，张量数据被拆分为 4 通道。因此，在形状为 `[B,H,W,5]` 的张量与形状为 `[B,H,W,8]` 的张量上的计算执行效果大致相同，但明显比 `[B,H,W,4]` 差。如果您使用的摄像头硬件支持 RGBA 格式的图像帧，则馈送 4 通道输入的速度会快得多，因为它避免了从 3 通道 RGB 到 4 通道 RGBX 的内存复制。

- **移动端优化模型** - 为了获得最佳性能，您应该考虑使用经移动端优化的网络架构重新训练分类器。通过充分利用移动硬件特性，对设备端推断进行优化可以显著降低延迟和功耗。

## 高级 GPU 支持

采用 GPU 处理时，您可以使用包括量化和序列化在内的其他高级技术来为您的模型提供更好的性能。以下部分将更详细地介绍这些技术。

### 使用量化模型 {:#quantized-models}

本部分将介绍 GPU 委托如何加速 8 位量化模型，包括以下内容：

- 使用[量化感知训练](https://www.tensorflow.org/model_optimization/guide/quantization/training)训练的模型
- 训练后[动态范围量化](https://www.tensorflow.org/lite/performance/post_training_quant)
- 训练后[全整数量化](https://www.tensorflow.org/lite/performance/post_training_integer_quant)

为了优化性能，请使用具有浮点输入和输出张量的模型。

#### 运作方式

由于 GPU 后端仅支持浮点执行，我们通过为其提供原始模型的“浮点视图”来运行量化模型。概括来讲，这需要执行以下步骤：

- *常量张量*（例如权重/偏置）进入 GPU 内存后会立即去量化。在为 TensorFlow Lite 启用委托时就会进行这项运算。

- 如果为 8 位量化，则 GPU 程序的*输入和输出*将分别针对每个推断进行去量化和量化。此运算在 CPU 上使用 TensorFlow Lite 的优化内核完成。

- 在运算之间插入*量化模拟器*以模拟量化行为。对于运算期望激活会遵循量化期间学习的边界的模型而言，这是一种必要方式。

有关对 GPU 委托启用此功能的信息，请参阅以下内容：

- [在 Android 平台的 GPU 上使用量化模型](../android/delegates/gpu#quantized-models)
- [在 iOS 平台的 GPU 上使用量化模型](../ios/delegates/gpu#quantized-models)

### 通过序列化缩短初始化时间 {:#delegate_serialization}

借助 GPU 委托功能，您可以从预编译的内核代码以及已在先前运行的磁盘上序列化和保存的模型数据中进行加载。这种方式避免了重新编译，并且可以缩短启动时间高达 90%。这项改进的实现原理是用磁盘空间换取时间。您在启用此功能时可以使用一些配置选项，示例代码如下：

<div>
  <devsite-selector>
    <section>
      <h3>C++</h3>
      <p></p>
<pre class="prettyprint lang-cpp">    TfLiteGpuDelegateOptionsV2 options = TfLiteGpuDelegateOptionsV2Default();
    options.experimental_flags |= TFLITE_GPU_EXPERIMENTAL_FLAGS_ENABLE_SERIALIZATION;
    options.serialization_dir = kTmpDir;
    options.model_token = kModelToken;

    auto* delegate = TfLiteGpuDelegateV2Create(options);
    if (interpreter-&gt;ModifyGraphWithDelegate(delegate) != kTfLiteOk) return false;
      </pre>
    </section>
    <section>
      <h3>Java</h3>
      <p></p>
<pre class="prettyprint lang-java">    GpuDelegate delegate = new GpuDelegate(
      new GpuDelegate.Options().setSerializationParams(
        /* serializationDir= */ serializationDir,
        /* modelToken= */ modelToken));

    Interpreter.Options options = (new Interpreter.Options()).addDelegate(delegate);
      </pre>
    </section>
  </devsite-selector>
</div>

使用序列化功能时，请确保您的代码符合以下实现规则：

- 将序列化数据存储在其他应用无法访问的目录中。在 Android 设备上请使用 [`getCodeCacheDir()`](https://developer.android.com/reference/android/content/Context#getCacheDir())，它会指向当前应用的私有位置。
- 对于特定型号的设备，型号令牌必须是唯一的。您可以通过使用诸如 [`farmhash::Fingerprint64`](https://github.com/google/farmhash) 的库从型号数据生成指纹来计算型号令牌。

注：使用此序列化功能需要 [OpenCL SDK](https://github.com/KhronosGroup/OpenCL-SDK)。
