# Google Play 服务中的 TensorFlow Lite

TensorFlow Lite is available in Google Play services runtime for all Android devices running the current version of Play services. This runtime allows you to run machine learning (ML) models without statically bundling TensorFlow Lite libraries into your app.

With the Google Play services API, you can reduce the size of your apps and gain improved performance from the latest stable version of the libraries. TensorFlow Lite in Google Play services is the recommended way to use TensorFlow Lite on Android.

You can get started with the Play services runtime with the [Quickstart](../android/quickstart), which provides a step-by-step guide to implement a sample application. If you are already using stand-alone TensorFlow Lite in your app, refer to the [Migrating from stand-alone TensorFlow Lite](#migrating) section to update an existing app to use the Play services runtime. For more information about Google Play services, see the [Google Play services](https://developers.google.com/android/guides/overview) website.

<aside class="note"> <b>Terms:</b> By accessing or using TensorFlow Lite in Google Play services APIs, you agree to the <a href="#tos">Terms of Service</a>. Please read and understand all applicable terms and policies before accessing the APIs. </aside>

## Using the Play services runtime

Google Play 服务中的 TensorFlow Lite 可以通过 [TensorFlow Lite Task API](../api_docs/java/org/tensorflow/lite/task/core/package-summary) 和 [TensorFlow Lite Interpreter API](../api_docs/java/org/tensorflow/lite/InterpreterApi) 获得。Task Library 为使用视觉、音频和文本数据的常见机器学习任务提供了优化的开箱即用模型接口。TensorFlow Lite Interpreter API 由 TensorFlow 运行时和支持库提供，为构建和运行机器学习模型提供了更通用的接口。

The following sections provide instructions on how to implement the Interpreter and Task Library APIs in Google Play services. While it is possible for an app to use both the Interpreter APIs and Task Library APIs, most apps should only use one set of APIs.

### 使用 Task Library API

TensorFlow Lite Task API 封装了 Interpreter API，并为使用视觉、音频和文本数据的常见机器学习任务提供了高级编程接口。如果您的应用需要[支持的任务](../inference_with_metadata/task_library/overview#supported_tasks)之一，您应该使用 Task API。

#### 1. 添加项目依赖项

您的项目依赖项取决于您的机器学习用例。Task API 包含以下库：

- 视觉库：`org.tensorflow:tensorflow-lite-task-vision-play-services`
- 音频库：`org.tensorflow:tensorflow-lite-task-audio-play-services`
- 文本库：`org.tensorflow:tensorflow-lite-task-text-play-services`

Add one of the dependencies to your app project code to access the Play services API for TensorFlow Lite. For example, use the following to implement a vision task:

```
dependencies {
...
    implementation 'org.tensorflow:tensorflow-lite-task-vision-play-services:0.4.2'
...
}
```

Caution: The TensorFlow Lite Tasks Audio library version 0.4.2 maven repository is incomplete. Use version 0.4.2.1 for this library instead: `org.tensorflow:tensorflow-lite-task-audio-play-services:0.4.2.1`.

#### 2. 添加 TensorFlow Lite 的初始化

在使用 TensorFlow Lite API *之前*，先初始化 Google Play 服务 API 的 TensorFlow Lite 组件。以下示例可以初始化视觉库：

<div>
  <devsite-selector>
    <section>
      <h3>Kotlin</h3>
<pre class="prettyprint">init {
  TfLiteVision.initialize(context)
    }
  }
</pre>
    </section>
  </devsite-selector>
</div>

重要提示：确保在执行访问 TensorFlow Lite API 的代码之前完成 `TfLite.initialize` 任务。

提示：TensorFlow Lite 模块在从 Play 商店安装或更新应用时安装。您可以使用 Google Play 服务 API 中的 `ModuleInstallClient` 检查模块的可用性。有关检查模块可用性的更多信息，请参阅[使用 ModuleInstallClient 确保 API 可用性](https://developers.google.com/android/guides/module-install-apis)。

#### 3. 运行推断

初始化 TensorFlow Lite 组件后，调用 `detect()` 方法生成推断。`detect()` 方法中的确切代码因库和用例而异。以下是使用 `TfLiteVision` 库的简单目标检测用例：

<div>
  <devsite-selector>
    <section>
      <h3>Kotlin</h3>
<pre class="prettyprint">fun detect(...) {
  if (!TfLiteVision.isInitialized()) {
    Log.e(TAG, "detect: TfLiteVision is not initialized yet")
    return
  }

  if (objectDetector == null) {
    setupObjectDetector()
  }

  ...

}
</pre>
    </section>
  </devsite-selector>
</div>

根据数据格式，您可能还需要在 `detect()` 方法中预处理和转换数据，然后再生成推断。例如，目标检测器的图像数据需要以下代码：

```kotlin
val imageProcessor = ImageProcessor.Builder().add(Rot90Op(-imageRotation / 90)).build()
val tensorImage = imageProcessor.process(TensorImage.fromBitmap(image))
val results = objectDetector?.detect(tensorImage)
```

### 使用 Interpreter API

Interpreter API 比 Task Library API 提供更多的控制和灵活性。如果 Task Library 不支持您的机器学习任务，或者如果您需要更通用的接口来构建和运行机器学习模型，应该使用 Interpreter API。

#### 1. 添加项目依赖项

Add the following dependencies to your app project code to access the Play services API for TensorFlow Lite:

```
dependencies {
...
    // Tensorflow Lite dependencies for Google Play services
    implementation 'com.google.android.gms:play-services-tflite-java:16.0.0'
    // Optional: include Tensorflow Lite Support Library
    implementation 'com.google.android.gms:play-services-tflite-support:16.0.0'
...
}
```

#### 2. 添加 TensorFlow Lite 的初始化

在使用 TensorFlow Lite API *之前*，先初始化 Google Play 服务 API 的 TensorFlow Lite 组件：

<div>
  <devsite-selector>
    <section>
      <h3>Kotlin</h3>
<pre class="prettyprint">val initializeTask: Task&lt;Void&gt; by lazy { TfLite.initialize(this) }
</pre>
    </section>
    <section>
      <h3>Java</h3>
<pre class="prettyprint">Task&lt;Void&gt; initializeTask = TfLite.initialize(context);
</pre>
    </section>
  </devsite-selector>
</div>

注：请确保在执行访问 TensorFlow Lite API 的代码之前完成 `TfLite.initialize` 任务。使用 `addOnSuccessListener()` 方法，如下一部分所示。

#### 3. 创建解释器并设置运行时选项 {:#step_3_interpreter}

使用 `InterpreterApi.create()` 创建解释器，并通过调用 `InterpreterApi.Options.setRuntime()` 将其配置为使用 Google Play 服务运行时，如以下示例代码所示：

<div>
  <devsite-selector>
    <section>
      <h3>Kotlin</h3>
<pre class="prettyprint">import org.tensorflow.lite.InterpreterApi
import org.tensorflow.lite.InterpreterApi.Options.TfLiteRuntime
...
private lateinit var interpreter: InterpreterApi
...
initializeTask.addOnSuccessListener {
  val interpreterOption =
    InterpreterApi.Options().setRuntime(TfLiteRuntime.FROM_SYSTEM_ONLY)
  interpreter = InterpreterApi.create(
    modelBuffer,
    interpreterOption
  )}
  .addOnFailureListener { e -&gt;
    Log.e("Interpreter", "Cannot initialize interpreter", e)
  }
</pre>
    </section>
    <section>
      <h3>Java</h3>
<pre class="prettyprint">import org.tensorflow.lite.InterpreterApi
import org.tensorflow.lite.InterpreterApi.Options.TfLiteRuntime
...
private InterpreterApi interpreter;
...
initializeTask.addOnSuccessListener(a -&gt; {
    interpreter = InterpreterApi.create(modelBuffer,
      new InterpreterApi.Options().setRuntime(TfLiteRuntime.FROM_SYSTEM_ONLY));
  })
  .addOnFailureListener(e -&gt; {
    Log.e("Interpreter", String.format("Cannot initialize interpreter: %s",
          e.getMessage()));
  });
</pre>
    </section>
  </devsite-selector>
</div>

您应该使用上面的实现，因为它能够避免阻塞 Android 界面线程。如果您需要更紧密地管理线程执行，可以添加一个 `Tasks.await()` 调用来创建解释器：

<div>
  <devsite-selector>
    <section>
      <h3>Kotlin</h3>
<pre class="prettyprint">import androidx.lifecycle.lifecycleScope
...
lifecycleScope.launchWhenStarted { // uses coroutine
  initializeTask.await()
}
</pre>
    </section>
    <section>
      <h3>Java</h3>
<pre class="prettyprint">@BackgroundThread
InterpreterApi initializeInterpreter() {
    Tasks.await(initializeTask);
    return InterpreterApi.create(...);
}
</pre>
    </section>
  </devsite-selector>
</div>

警告：请勿在前台界面线程上调用 `.await()`，因为这会中断界面元素的显示并产生不好的用户体验。

#### 4. 运行推断

使用您创建的 `interpreter` 对象，调用 `run()` 方法来生成推断。

<div>
  <devsite-selector>
    <section>
      <h3>Kotlin</h3>
<pre class="prettyprint">interpreter.run(inputBuffer, outputBuffer)
</pre>
    </section>
    <section>
      <h3>Java</h3>
<pre class="prettyprint">interpreter.run(inputBuffer, outputBuffer);
</pre>
    </section>
  </devsite-selector>
</div>

## 硬件加速 {:#hardware-acceleration}

TensorFlow Lite allows you to accelerate the performance of your model using specialized hardware processors, such as graphics processing units (GPUs). You can take advantage of these specialized processors using hardware drivers called [*delegates*](https://www.tensorflow.org/lite/performance/delegates). You can use the following hardware acceleration delegates with TensorFlow Lite in Google Play services:

- *[GPU 委托](https://www.tensorflow.org/lite/performance/gpu)（推荐）* – 此委托通过 Google Play 服务提供并动态加载，就像 Play 服务版本的 Task API 和 Interpreter API 一样。

- [*NNAPI 委托*](https://www.tensorflow.org/lite/android/delegates/nnapi) – 此委托可作为您的 Android 开发项目中包含的库依赖项使用，并绑定到您的应用中。

有关使用 TensorFlow Lite 进行硬件加速的更多信息，请参阅 [TensorFlow Lite 委托](https://www.tensorflow.org/lite/performance/delegates)页面。

### Checking device compatibility

Not all devices support GPU hardware acceleration with TFLite. In order to mitigate errors and potential crashes, use the `TfLiteGpu.isGpuDelegateAvailable` method to check whether a device is compatible with the GPU delegate.

Use this method to confirm whether a device is compatible with GPU, and use CPU or the NNAPI delegate as a fallback for when GPU is not supported.

```
useGpuTask = TfLiteGpu.isGpuDelegateAvailable(context)
```

Once you have a variable like `useGpuTask`, you can use it to determine whether devices use the GPU delegate. The following examples show how this can be done with both the Task Library and Interpreter APIs.

**使用 Task API**

<div>
<devsite-selector>
<section>
  <h3>Kotlin</h3>
    <pre class="prettyprint">lateinit val optionsTask = useGpuTask.continueWith { task -&gt;
  val baseOptionsBuilder = BaseOptions.builder()
  if (task.result) {
    baseOptionsBuilder.useGpu()
  }
 ObjectDetectorOptions.builder()
          .setBaseOptions(baseOptionsBuilder.build())
          .setMaxResults(1)
          .build()
}
    </pre>
</section>
<section>
  <h3>Java</h3>
    <pre class="prettyprint">Task&lt;ObjectDetectorOptions&gt; optionsTask = useGpuTask.continueWith({ task -&gt;
  BaseOptions baseOptionsBuilder = BaseOptions.builder();
  if (task.getResult()) {
    baseOptionsBuilder.useGpu();
  }
  return ObjectDetectorOptions.builder()
          .setBaseOptions(baseOptionsBuilder.build())
          .setMaxResults(1)
          .build()
});
    </pre>
</section>
</devsite-selector>
</div>

**使用 Interpreter API**

<div>
<devsite-selector>
<section>
  <h3>Kotlin</h3>
    <pre class="prettyprint">val interpreterTask = useGpuTask.continueWith { task -&gt;
  val interpreterOptions = InterpreterApi.Options()
      .setRuntime(TfLiteRuntime.FROM_SYSTEM_ONLY)
  if (task.result) {
      interpreterOptions.addDelegateFactory(GpuDelegateFactory())
  }
  InterpreterApi.create(FileUtil.loadMappedFile(context, MODEL_PATH), interpreterOptions)
}
    </pre>
</section>
<section>
  <h3>Java</h3>
    <pre class="prettyprint">Task&lt;InterpreterApi.Options&gt; interpreterOptionsTask = useGpuTask.continueWith({ task -&gt;
  InterpreterApi.Options options =
      new InterpreterApi.Options().setRuntime(TfLiteRuntime.FROM_SYSTEM_ONLY);
  if (task.getResult()) {
     options.addDelegateFactory(new GpuDelegateFactory());
  }
  return options;
});
    </pre>
</section>
</devsite-selector>
</div>

### GPU 与 Task Library API

要将 GPU 委托与 Task API 一起使用，请执行以下操作：

1. Update the project dependencies to use the GPU delegate from Play services:

    ```
    implementation 'com.google.android.gms:play-services-tflite-gpu:16.0.0'
    ```

2. 使用 `setEnableGpuDelegateSupport` 初始化 GPU委托。例如，您可以使用以下代码初始化 `TfLiteVision` 的 GPU 委托：

    <div>
    <devsite-selector>
    <section>
      <h3>Kotlin</h3>
        <pre class="prettyprint">          TfLiteVision.initialize(context, TfLiteInitializationOptions.builder().setEnableGpuDelegateSupport(true).build())
                    </pre>
    </section>
    <section>
      <h3>Java</h3>
        <pre class="prettyprint">          TfLiteVision.initialize(context, TfLiteInitializationOptions.builder().setEnableGpuDelegateSupport(true).build());
                    </pre>
    </section>
    </devsite-selector>
    </div>

3. 使用 [`BaseOptions`](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/task/core/BaseOptions.Builder) 启用 GPU 委托选项：

    <div>
    <devsite-selector>
    <section>
      <h3>Kotlin</h3>
        <pre class="prettyprint">          val baseOptions = BaseOptions.builder().useGpu().build()
                    </pre>
    </section>
    <section>
      <h3>Java</h3>
        <pre class="prettyprint">          BaseOptions baseOptions = BaseOptions.builder().useGpu().build();
                    </pre>
    </section>
    </devsite-selector>
    </div>

4. 使用 `.setBaseOptions` 配置选项。例如，您可以使用以下代码在 `ObjectDetector` 中设置 GPU：

    <div>
    <devsite-selector>
    <section>
      <h3>Kotlin</h3>
        <pre class="prettyprint">        val options =
                        ObjectDetectorOptions.builder()
                            .setBaseOptions(baseOptions)
                            .setMaxResults(1)
                            .build()
                    </pre>
    </section>
    <section>
      <h3>Java</h3>
        <pre class="prettyprint">        ObjectDetectorOptions options =
                        ObjectDetectorOptions.builder()
                            .setBaseOptions(baseOptions)
                            .setMaxResults(1)
                            .build();
                    </pre>
    </section>
    </devsite-selector>
    </div>

### GPU 与 Interpreter API

要将 GPU 委托与 Interpreter API 一起使用，请执行以下操作：

1. Update the project dependencies to use the GPU delegate from Play services:

    ```
    implementation 'com.google.android.gms:play-services-tflite-gpu:16.0.0'
    ```

2. 在 TFlite 初始化中启用 GPU 委托选项：

    <div>
    <devsite-selector>
    <section>
      <h3>Kotlin</h3>
        <pre class="prettyprint">        TfLite.initialize(context,
              TfLiteInitializationOptions.builder()
               .setEnableGpuDelegateSupport(true)
               .build())
            </pre>
    </section>
    <section>
      <h3>Java</h3>
        <pre class="prettyprint">        TfLite.initialize(context,
              TfLiteInitializationOptions.builder()
               .setEnableGpuDelegateSupport(true)
               .build());
            </pre>
    </section>
    </devsite-selector>
    </div>

3. 通过调用 `InterpreterApi.Options()` 中的 `addDelegateFactory()` 在解释器选项中设置 GPU 委托以使用 `DelegateFactory`：

    <div>
    <devsite-selector>
    <section>
      <h3>Kotlin</h3>
        <pre class="prettyprint">
                      val interpreterOption = InterpreterApi.Options()
                       .setRuntime(TfLiteRuntime.FROM_SYSTEM_ONLY)
                       .addDelegateFactory(GpuDelegateFactory())
                    </pre>
    </section>
    <section>
      <h3>Java</h3>
        <pre class="prettyprint">
                      Options interpreterOption = InterpreterApi.Options()
                        .setRuntime(TfLiteRuntime.FROM_SYSTEM_ONLY)
                        .addDelegateFactory(new GpuDelegateFactory());
                    </pre>
    </section>
    </devsite-selector>
    </div>

## 从独立的 TensorFlow Lite 迁移 {:#migrating}

如果您计划将应用从独立 TensorFlow Lite 迁移到 Play 服务 API，请查看以下有关更新应用项目代码的附加指南：

1. 请查看此页面的[限制](#limitations)部分，以确保您的用例受支持。
2. 在更新代码之前，请对模型进行性能和准确率检查，特别是如果您使用的是 2.1 之前版本的 TensorFlow Lite，这样您就有一个可以用来与新实现进行比较的基准。
3. 如果您已迁移所有代码以使用适用于 TensorFlow Lite 的 Play 服务 API，则应从您的 build.gradle 文件中移除现有的 TensorFlow Lite *runtime library* 依赖项（带有 <code>org.tensorflow:**tensorflow-lite**:*</code> 的条目），以便缩减应用大小。
4. 识别代码中创建 `new Interpreter` 对象的所有匹配项，并进行修改，使其使用 InterpreterApi.create() 调用。这个新的 API 为异步 API，这意味着在大多数情况下，它不是临时替代，当调用完成时，您必须注册一个侦听器。请参见[第 3 步](#step_3_interpreter)代码中的代码段。
5. 将 `import org.tensorflow.lite.InterpreterApi;` 和 `import org.tensorflow.lite.InterpreterApi.Options.TfLiteRuntime;` 添加到任何使用 `org.tensorflow.lite.Interpreter` 或 `org.tensorflow.lite.InterpreterApi` 类的源文件中。
6. 如果对 `InterpreterApi.create()` 的任何结果调用只有一个参数，请将 `new InterpreterApi.Options()` 追加到参数列表。
7. 将 `.setRuntime(TfLiteRuntime.FROM_SYSTEM_ONLY)` 追加到对 `InterpreterApi.create()` 的任何调用的最后一个参数。
8. 将 `org.tensorflow.lite.Interpreter` 类的其他匹配项替换为 `org.tensorflow.lite.InterpreterApi`。

如果想同时使用独立的 TensorFlow Lite 和 Play 服务 API，您必须使用 TensorFlow Lite 2.9 或更高版本。TensorFlow Lite 2.8 及之前的版本与 Play 服务 API 版本不兼容。

## 限制

Google Play 服务中的 TensorFlow Lite 存在以下限制：

- 对硬件加速委托的支持仅限于[硬件加速](#hardware-acceleration)部分中列出的委托。不支持其他加速委托。
- 不支持通过[原生 API](https://www.tensorflow.org/lite/guide/inference#load_and_run_a_model_in_c) 访问 TensorFlow Lite。只有 TensorFlow Lite Java API 可以通过 Google Play 服务访问。
- 不支持实验性或过时的 TensorFlow Lite API，包括自定义运算。

## 支持和反馈 {:#support}

You can provide feedback and get support through the TensorFlow Issue Tracker. Please report issues and support requests using the [Issue template](https://github.com/tensorflow/tensorflow/issues/new?title=TensorFlow+Lite+in+Play+Services+issue&template=tflite-in-play-services.md) for TensorFlow Lite in Google Play services.

## 服务条款 {:#tos}

Use of TensorFlow Lite in Google Play services APIs is subject to the [Google APIs Terms of Service](https://developers.google.com/terms/).

### 隐私和数据收集

When you use TensorFlow Lite in Google Play services APIs, processing of the input data, such as images, video, text, fully happens on-device, and TensorFlow Lite in Google Play services APIs does not send that data to Google servers. As a result, you can use our APIs for processing data that should not leave the device.

Google Play 服务 API 中的 TensorFlow Lite 可能会不时联系 Google 服务器，以便接收错误修复、更新的模型和硬件加速器兼容性信息等内容。Google Play 服务 API 中的 TensorFlow Lite 还将有关应用中 API 的性能和利用率的指标发送给 Google。Google 使用这些指标数据来衡量性能，调试、维护和改进 API，并检测误用或滥用，有关详细信息，请参阅我们的[隐私权政策](https://policies.google.com/privacy)。

**You are responsible for informing users of your app about Google's processing of TensorFlow Lite in Google Play services APIs metrics data as required by applicable law.**

我们收集的数据包括：

- 设备信息（例如制造商、型号、操作系统版本和内部版本号）和可用的机器学习硬件加速器（GPU 和 DSP）。用于诊断和使用情况分析。
- 用于诊断和使用情况分析的设备标识符。
- 应用信息（软件包名称、应用版本)。用于诊断和使用情况分析。
- API 配置（例如正在使用哪些委托）。用于诊断和使用情况分析。
- 事件类型（例如解释器创建、推断）。用于诊断和使用情况分析。
- 错误代码。用于诊断。
- 性能指标。用于诊断。

## 后续步骤

有关使用 TensorFlow Lite 在移动应用中实现机器学习的更多信息，请参阅 [TensorFlow Lite 开发者指南](https://www.tensorflow.org/lite/guide)。您可以在 [TensorFlow Hub](https://tfhub.dev/s?deployment-format=lite) 上找到用于图像分类、目标检测和其他应用的其他 TensorFlow Lite 模型。
