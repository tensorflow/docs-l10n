# GPU 上的 TensorFlow Lite

[TensorFlow Lite ](https://www.tensorflow.org/mobile/tflite/) 支持多种硬件加速器。本文档介绍如何通过 TensorFlow Lite 委托 API 在 Android（要求 OpenCL 或者 OpenGL ES 3.1 及更高版本）和 iOS（要求 iOS 8 或更高版本）上使用 GPU 后端。

## GPU 加速的好处

### 速度

GPU 采用高吞吐量式设计，可处理大规模可并行化的工作负载。因此，它们非常适合包含大量算子的深度神经网络，每个算子都会处理一个或多个输入张量，可以轻松地划分为较小的工作负载且并行执行，这通常可以降低延迟。在最佳情况下，GPU 上的推断速度已足够快，适用于以前无法实现的实时应用。

### 准确性

GPU 使用 16 位或 32 位浮点数进行计算，并且（与 CPU 不同）不需要量化即可获得最佳性能。如果准确率降低使模型的量化无法达到要求，那么在 GPU 上运行神经网络可以消除这种担忧。

### 能效

GPU 推断的另一个优势是其功效。GPU 以非常高效且经优化的方式执行计算，因此与在 CPU 上执行相同任务时相比，功耗和产生的热量更低。

## 支持的运算

GPU 上的 TensorFlow Lite 支持 16 位和 32 位浮点精度的以下运算：

- `ADD`
- `AVERAGE_POOL_2D`
- `CONCATENATION`
- `CONV_2D`
- `DEPTHWISE_CONV_2D v1-2`
- `EXP`
- `FULLY_CONNECTED`
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

默认情况下，只有版本 1 支持所有运算。启用[实验性量化支持](gpu_advanced.md#running-quantized-models-experimental-android-only)可以允许相应的版本，例如 ADD v2。

## 基本用法

您可以通过两种方式调用模型加速，具体取决于您使用的是 [Android Studio 机器学习模型绑定](../inference_with_metadata/codegen#acceleration)还是 TensorFlow Lite 解释器。

### Android 通过 TensorFlow Lite 解释器

在现有 `dependencies` 块中现有 `tensorflow-lite` 软件包的位置下添加 `tensorflow-lite-gpu` 软件包。

```
dependencies {
    ...
    implementation 'org.tensorflow:tensorflow-lite:2.3.0'
    implementation 'org.tensorflow:tensorflow-lite-gpu:2.3.0'
}
```

然后，使用 `TfLiteDelegate` 在 GPU 上运行 TensorFlow Lite。在 Java 中，您可以通过 `Interpreter.Options` 指定 `GpuDelegate`。

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

### Android (C/C++)

对于 Android 上 TensorFlow Lite GPU 的 C/C++ 用法，可以使用 `TfLiteGpuDelegateV2Create()` 创建 GPU 委托，使用 `TfLiteGpuDelegateV2Delete()` 销毁 GPU 委托。

```c++
// Set up interpreter.
auto model = FlatBufferModel::BuildFromFile(model_path);
if (!model) return false;
ops::builtin::BuiltinOpResolver op_resolver;
std::unique_ptr<Interpreter> interpreter;
InterpreterBuilder(*model, op_resolver)(&interpreter);

// NEW: Prepare GPU delegate.
auto* delegate = TfLiteGpuDelegateV2Create(/*default options=*/nullptr);
if (interpreter->ModifyGraphWithDelegate(delegate) != kTfLiteOk) return false;

// Run inference.
WriteToInputTensor(interpreter->typed_input_tensor<float>(0));
if (interpreter->Invoke() != kTfLiteOk) return false;
ReadFromOutputTensor(interpreter->typed_output_tensor<float>(0));

// NEW: Clean up.
TfLiteGpuDelegateV2Delete(delegate);
```

要在GPU上运行TensorFlow Lite，需要通过`NewGpuDelegate()`对GPU委托（delegate），然后将其传递给`Interpreter::ModifyGraphWithDelegate()`（而不是调用`Interpreter::AllocateTensors()`）

TFLite GPU for Android C/C++ 使用 [Bazel](https://bazel.io) 构建系统。例如，可以使用以下命令构建委托：

```sh
bazel build -c opt --config android_arm64 tensorflow/lite/delegates/gpu:delegate                           # for static library
bazel build -c opt --config android_arm64 tensorflow/lite/delegates/gpu:libtensorflowlite_gpu_delegate.so  # for dynamic library
```

注：调用 `Interpreter::ModifyGraphWithDelegate()` 或 `Interpreter::Invoke()` 时，调用者在当前线程中必须具有 `EGLContext`，并且 `Interpreter::Invoke()` 必须从相同的 `EGLContext` 调用。如果 `EGLContext` 不存在，委托将在内部创建一个，但开发者随后必须确保该 `Interpreter::Invoke()` 始终从调用 `Interpreter::ModifyGraphWithDelegate()` 的同一个线程调用。

### iOS (C++)

注：有关 Swift/Objective-C/C 用例，请参阅 [GPU 委托指南](gpu#ios)

注：仅当您使用 Bazel 或自行构建 TensorFlow Lite 时，此功能才可用。C++ API 不能与 CocoaPods 一起使用。

要在 GPU 上使用 TensorFlow Lite，请通过 `TFLGpuDelegateCreate()` 获取 GPU 委托，然后将其传递给 `Interpreter::ModifyGraphWithDelegate()`（而不是调用 `Interpreter::AllocateTensors()`）。

```c++
// Set up interpreter.
auto model = FlatBufferModel::BuildFromFile(model_path);
if (!model) return false;
tflite::ops::builtin::BuiltinOpResolver op_resolver;
std::unique_ptr<Interpreter> interpreter;
InterpreterBuilder(*model, op_resolver)(&interpreter);

// NEW: Prepare GPU delegate.

auto* delegate = TFLGpuDelegateCreate(/*default options=*/nullptr);
if (interpreter->ModifyGraphWithDelegate(delegate) != kTfLiteOk) return false;

// Run inference.
WriteToInputTensor(interpreter->typed_input_tensor<float>(0));
if (interpreter->Invoke() != kTfLiteOk) return false;
ReadFromOutputTensor(interpreter->typed_output_tensor<float>(0));

// Clean up.
TFLGpuDelegateDelete(delegate);
```

## 高级用法

### 委托（Delegate）iOS 选项

GPU 委托的构造函数接受选项的 `struct`。（[Swift API](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/swift/Sources/MetalDelegate.swift)、[Objective-C API](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/objc/apis/TFLMetalDelegate.h)、[C API](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/gpu/metal_delegate.h)）

向初始值设定项传递 `nullptr` (C API) 或不传递任何内容（Objective-C 和 Swift API）即可设置默认选项（上方“基本用法”示例已作说明）。

<div>
  <devsite-selector>
    <section>
      <h3>Swift</h3>
      <p></p>
<pre class="prettyprint lang-swift">    // THIS:
    var options = MetalDelegate.Options()
    options.isPrecisionLossAllowed = false
    options.waitType = .passive
    options.isQuantizationEnabled = true
    let delegate = MetalDelegate(options: options)

    // IS THE SAME AS THIS:
    let delegate = MetalDelegate()
      </pre>
    </section>
    <section>
      <h3>Objective-C</h3>
      <p></p>
<pre class="prettyprint lang-objc">    // THIS:
    TFLMetalDelegateOptions* options = [[TFLMetalDelegateOptions alloc] init];
    options.precisionLossAllowed = false;
    options.waitType = TFLMetalDelegateThreadWaitTypePassive;
    options.quantizationEnabled = true;

    TFLMetalDelegate* delegate = [[TFLMetalDelegate alloc] initWithOptions:options];

    // IS THE SAME AS THIS:
    TFLMetalDelegate* delegate = [[TFLMetalDelegate alloc] init];
      </pre>
    </section>
    <section>
      <h3>C</h3>
      <p></p>
<pre class="prettyprint lang-c">    // THIS:
    const TFLGpuDelegateOptions options = {
      .allow_precision_loss = false,
      .wait_type = TFLGpuDelegateWaitType::TFLGpuDelegateWaitTypePassive,
      .enable_quantization = true,
    };

    TfLiteDelegate* delegate = TFLGpuDelegateCreate(options);

    // IS THE SAME AS THIS:
    TfLiteDelegate* delegate = TFLGpuDelegateCreate(nullptr);
      </pre>
    </section>
  </devsite-selector>
</div>

尽管使用 `nullptr` 或默认构造函数十分方便，但建议您显式设置选项，以避免将来因更改默认值而发生任何意外行为。

### 在 GPU 上运行量化模型

本部分将说明 GPU 委托如何加速 8 位量化模型。这包括所有量化方式，包括：

- 使用[量化感知训练](https://www.tensorflow.org/lite/models/convert/quantization)训练的模型
- [训练后动态范围量化](https://www.tensorflow.org/lite/performance/post_training_quant)
- [训练后全整数量化](https://www.tensorflow.org/lite/performance/post_training_integer_quant)

为了优化性能，请使用具有浮点输入和输出张量的模型。

#### 运作方式

由于 GPU 后端仅支持浮点执行，因此我们通过为其提供原始模型的“浮点视图”来运行量化模型。在较高层面上讲，这需要执行以下步骤：

- *常量张量*（例如权重/偏置）进入 GPU 内存后会立即去量化。将委托应用于 TFLite 解释器时，就会发生这种情况。

- 如果为 8 位量化，则 GPU 程序的*输入和输出* 将分别针对每个推断进行去量化和量化。此操作在 CPU 上使用 TFLite 的优化内核完成。

- 通过在运算之间插入*量化模拟器*来修改 GPU 程序以模仿量化行为。如果运算期望激活函数遵循在量化过程中学习的边界，则对于这种模型而言，这是必需步骤。

可以使用委托选项来启用此功能，如下所示：

#### Android

Android API 默认支持量化模型。要停用，请执行以下操作：

**C++ API**

```c++
struct GpuDelegateOptions {
  // Allows to quantify tensors, downcast values, process in float16 etc.
  bool allow_precision_loss;

  enum class WaitType {
    // waitUntilCompleted
    kPassive,
    // Minimize latency. It uses active spinning instead of mutex and consumes
    // additional CPU resources.
    kActive,
    // Useful when the output is used with GPU pipeline then or if external
    // command encoder is set
    kDoNotWait,
  };
  WaitType wait_type;
};
```

**Java API**

```java
// THIS:
const GpuDelegateOptions options = {
  .allow_precision_loss = false,
  .wait_type = kGpuDelegateOptions::WaitType::Passive,
};

auto* delegate = NewGpuDelegate(options);

// IS THE SAME AS THIS:
auto* delegate = NewGpuDelegate(nullptr);
```

#### iOS

iOS API 默认支持量化模型。要停用，请执行以下操作：

<div>
  <devsite-selector>
    <section>
      <h3>Swift</h3>
      <p></p>
<pre class="prettyprint lang-swift">    var options = MetalDelegate.Options()
    options.isQuantizationEnabled = false
    let delegate = MetalDelegate(options: options)
      </pre>
    </section>
    <section>
      <h3>Objective-C</h3>
      <p></p>
<pre class="prettyprint lang-objc">    TFLMetalDelegateOptions* options = [[TFLMetalDelegateOptions alloc] init];
    options.quantizationEnabled = false;
      </pre>
    </section>
    <section>
      <h3>C</h3>
      <p></p>
<pre class="prettyprint lang-c">    TFLGpuDelegateOptions options = TFLGpuDelegateOptionsDefault();
    options.enable_quantization = false;

    TfLiteDelegate* delegate = TFLGpuDelegateCreate(options);
      </pre>
    </section>
  </devsite-selector>
</div>

### 输入/输出缓冲区（仅适用于 iOS，C++ API）

注：仅当您使用 Bazel 或自行构建 TensorFlow Lite 时，此功能才可用。C++ API 不能与 CocoaPods 一起使用。

要在 GPU 上执行计算，则必须使数据可用于 GPU。这通常需要执行内存复制。如果可能，最好不要越过 CPU/GPU 内存边界，因为这会占用大量时间。通常，这种越界是不可避免的，但在某些特殊情况下却可以忽略其中一种内存。

如果网络的输入为 GPU 内存中已加载的图像（例如，包含摄像头feed 的 GPU 纹理），则它可以驻留在 GPU 内存中而无需进入 CPU 内存。同样，如果网络的输出为可渲染图像形式（例如，[图像风格转换](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)），则可以直接在屏幕上显示。

为了获得最佳性能，TensorFlow Lite 使用户可以直接从 TensorFlow 硬件缓冲区读取和写入数据，并绕过可避免的内存复制过程。

假设图像输入位于 GPU 内存中，则必须首先将其转换为 Metal 的 `MTLBuffer` 对象。您可以使用 `TFLGpuDelegateBindMetalBufferToTensor()` 将 TfLiteTensor 关联至用户准备的 `MTLBuffer`。请注意，必须在 `Interpreter::ModifyGraphWithDelegate()` 之后调用 `TFLGpuDelegateBindMetalBufferToTensor()`。此外，在默认情况下，推断输出会从 GPU 内存复制到 CPU 内存。可以通过在初始化期间调用 `Interpreter::SetAllowBufferHandleOutput(true)` 来关闭此行为。

```c++
// Ensure a valid EGL rendering context.
EGLContext eglContext = eglGetCurrentContext();
if (eglContext.equals(EGL_NO_CONTEXT)) return false;

// Create an SSBO.
int[] id = new int[1];
glGenBuffers(id.length, id, 0);
glBindBuffer(GL_SHADER_STORAGE_BUFFER, id[0]);
glBufferData(GL_SHADER_STORAGE_BUFFER, inputSize, null, GL_STREAM_COPY);
glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);  // unbind
int inputSsboId = id[0];

// Create interpreter.
Interpreter interpreter = new Interpreter(tfliteModel);
Tensor inputTensor = interpreter.getInputTensor(0);
GpuDelegate gpuDelegate = new GpuDelegate();
// The buffer must be bound before the delegate is installed.
gpuDelegate.bindGlBufferToTensor(inputTensor, inputSsboId);
interpreter.modifyGraphWithDelegate(gpuDelegate);

// Run inference; the null input argument indicates use of the bound buffer for input.
fillSsboWithCameraImageTexture(inputSsboId);
float[] outputArray = new float[outputSize];
interpreter.runInference(null, outputArray);
```

注：关闭默认行为后，要将推断输出从 GPU 内存复制到 CPU 内存，则需要对每个输出张量显式调用 `Interpreter::EnsureTensorDataIsReadable()`。

注：这也适用于量化模型，但是您仍然需要**带有 float32 数据的 float32 大小的缓冲区**，因为该缓冲区将绑定到内部去量化缓冲区。

### GPU 委托序列化

使用来自先前初始化的 GPU 内核代码和模型数据的序列化可以将 GPU 委托初始化的延迟降低高达 90%。这项改进是通过交换磁盘空间以节省时间的方式实现的。您可以使用一些配置选项来启用此功能，示例代码如下：

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

- 将序列化数据存储在其他应用无法访问的目录中。在 Android 设备上，使用指向当前应用程序私有位置的 [`getCodeCacheDir()`](https://developer.android.com/reference/android/content/Context#getCacheDir())。
- 对于特定型号的设备，型号令牌必须是唯一的。您可以通过从型号数据生成指纹来计算型号令牌（例如，使用 [`farmhash::Fingerprint64`](https://github.com/google/farmhash)）。

注：此功能需要 [OpenCL SDK](https://github.com/KhronosGroup/OpenCL-SDK) 来支持序列化。

## 提示和技巧

- CPU 上一些琐碎的运算对于 GPU 而言可能会造成高昂的成本。各种形式的整形运算就是此类运算中的一类（包括 `BATCH_TO_SPACE`、`SPACE_TO_BATCH`、`SPACE_TO_DEPTH` 以及类似运算）。如果并不需要这些运算（例如，插入它们只为帮助网络架构师分析系统，但不会影响输出），则有必要移除它们以提高性能。

- 在 GPU 上，张量数据会被切分成 4 个通道。因此，对形状为 `[B, H, W, 5]` 的张量执行计算将与对形状为 `[B, H, W, 8]` 的张量执行计算大致相同，但与 `[B, H, W, 4]` 会有显著差距。

    - 例如，如果相机硬件支持 RGBA 图像帧，那么馈送这种 4 通道输入的速度会显著提升，因为这避免了内存复制过程（从 3 通道 RGB 到 4 通道 RGBX）。

- 为了获得最佳性能，请立即使用针对移动设备进行优化的网络架构来重新训练您的分类器。这是优化设备端推断的重要部分。
