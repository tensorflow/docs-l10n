# Android 的 GPU 加速委托

使用图形处理单元 (GPU) 运行机器学习 (ML) 模型可以显著改善模型的性能和支持 ML 的应用的用户体验。在 Android 设备上，您可以使用[*委托*](../../performance/delegates)启用对模型的 GPU 加速执行。委托充当 TensorFlow Lite 的硬件驱动程序，允许您在 GPU 处理器上运行模型的代码。

本页介绍了如何在 Android 应用中为 TensorFlow Lite 模型启用 GPU 加速。有关将 GPU 委托用于 TensorFlow Lite 的更多信息，包括最佳做法和高级技术，请参阅 [GPU 委托](../../performance/gpu)页面。

## 将 GPU 与 Task Library API 结合使用

TensorFlow Lite [Task Library](../../inference_with_metadata/task_library/overview) 提供了一组用于构建机器学习应用的任务特定 API。本部分介绍如何将 GPU 加速器委托与这些 API 结合使用。

### 添加项目依赖项

通过添加以下依赖项，启用 TensorFlow Lite Task Library 对 GPU 委托 API 的访问，更新您的开发项目 `build.gradle` 文件以包含 `tensorflow-lite-gpu-delegate-plugin` 软件包，如以下代码示例所示：

```
dependencies {
  ...
  implementation 'org.tensorflow:tensorflow-lite-gpu-delegate-plugin'
}
```

### 启用 GPU 加速

使用 [`BaseOptions`](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/task/core/BaseOptions.Builder) 类为您的 Task API 模型类启用 GPU 委托选项。例如，您可以在 `ObjectDetector` 中设置 GPU，如以下代码示例所示：

<div>
  <devsite-selector>
    <section>
      <h3>Kotlin</h3>
      <p></p>
<pre class="prettyprint lang-kotlin">    import org.tensorflow.lite.task.core.BaseOptions
    import org.tensorflow.lite.task.gms.vision.detector.ObjectDetector

    val baseOptions = BaseOptions.builder().useGpu().build()

    val options =
        ObjectDetector.ObjectDetectorOptions.builder()
            .setBaseOptions(baseOptions)
            .setMaxResults(1)
            .build()

    val objectDetector = ObjectDetector.createFromFileAndOptions(
      context, model, options)

      </pre>
    </section>
    <section>
      <h3>Java</h3>
      <p></p>
<pre class="prettyprint lang-java">    import org.tensorflow.lite.task.core.BaseOptions
    import org.tensorflow.lite.task.gms.vision.detector.ObjectDetector

    BaseOptions baseOptions = BaseOptions.builder().useGpu().build();

    ObjectDetectorOptions options =
        ObjectDetectorOptions.builder()
            .setBaseOptions(baseOptions)
            .setMaxResults(1)
            .build();

    val objectDetector = ObjectDetector.createFromFileAndOptions(
      context, model, options);
      </pre>
    </section>
  </devsite-selector>
</div>

## 将 GPU 与 Interpreter API 结合使用

TensorFlow Lite [Interpreter API](../../api_docs/java/org/tensorflow/lite/InterpreterApi) 提供了一组用于构建机器学习应用的通用 API。本部分介绍如何将 GPU 加速器委托与这些 API 结合使用。

### 添加项目依赖项

通过添加以下依赖项，启用对 GPU 委托 API 的访问，更新您的开发项目 `build.gradle` 文件以包含 `org.tensorflow:tensorflow-lite-gpu` 软件包，如以下代码示例所示：

```
dependencies {
    ...
    implementation 'org.tensorflow:tensorflow-lite'
    implementation 'org.tensorflow:tensorflow-lite-gpu'
}
```

### 启用 GPU 加速

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

注：必须在运行它的同一线程上创建 GPU 委托。否则，可能会看到以下错误：`TfLiteGpuDelegate Invoke: GpuDelegate must run on the same thread where it was initialized.`

GPU 委托也可以与 Android Studio 中的 ML 模型绑定一起使用。有关更多信息，请参阅[使用元数据生成模型接口](../../inference_with_metadata/codegen#acceleration)。

## 高级 GPU 支持

本部分介绍 Android 的 GPU 委托的高级用法，包括 C API、C++ API 和量化模型的使用。

### Android 的 C/C++ API

通过使用 `TfLiteGpuDelegateV2Create()` 创建委托并使用 `TfLiteGpuDelegateV2Delete()` 销毁委托，在 C 或 C++ 中使用 Android 的 TensorFlow Lite GPU 委托，如以下示例代码所示：

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

查看 `TfLiteGpuDelegateOptionsV2` 对象代码以使用自定义选项构建一个委托实例。您可以使用 `TfLiteGpuDelegateOptionsV2Default()` 初始化默认选项，然后根据需要对其进行修改。

C 或 C++ 中 Android 的 TensorFlow Lite GPU 委托使用 [Bazel](https://bazel.io) 构建系统。可以使用以下命令构建委托：

```sh
bazel build -c opt --config android_arm64 tensorflow/lite/delegates/gpu:delegate                           # for static library
bazel build -c opt --config android_arm64 tensorflow/lite/delegates/gpu:libtensorflowlite_gpu_delegate.so  # for dynamic library
```

调用 `Interpreter::ModifyGraphWithDelegate()` 或 `Interpreter::Invoke()` 时，调用者在当前线程中必须具有 `EGLContext`，并且 `Interpreter::Invoke()` 必须从相同的 `EGLContext` 调用。如果 `EGLContext` 不存在，则委托将在内部创建一个，但您随后必须确保该 `Interpreter::Invoke()` 始终从调用 `Interpreter::ModifyGraphWithDelegate()` 的同一个线程调用。

### 量化模型 {:#quantized-models}

Android GPU 委托库默认支持量化模型。您无需更改任何代码即可将量化模型与 GPU 委托一起使用。以下部分说明了如何停用量化支持以用于测试或实验目的。

#### 停用量化模型支持

以下代码显示了如何***停用***对量化模型的支持。

<div>
  <devsite-selector>
    <section>
      <h3>Java</h3>
      <p></p>
<pre class="prettyprint lang-java">GpuDelegate delegate = new GpuDelegate(new GpuDelegate.Options().setQuantizedModelsAllowed(false));

Interpreter.Options options = (new Interpreter.Options()).addDelegate(delegate);
      </pre>
    </section>
    <section>
      <h3>C++</h3>
      <p></p>
<pre class="prettyprint lang-c++">TfLiteGpuDelegateOptionsV2 options = TfLiteGpuDelegateOptionsV2Default();
options.experimental_flags = TFLITE_GPU_EXPERIMENTAL_FLAGS_NONE;

auto* delegate = TfLiteGpuDelegateV2Create(options);
if (interpreter-&gt;ModifyGraphWithDelegate(delegate) != kTfLiteOk) return false;
      </pre>
    </section>
  </devsite-selector>
</div>

有关使用 GPU 加速运行量化模型的更多信息，请参阅 [GPU 委托](../../performance/gpu#quantized-models)概述。
