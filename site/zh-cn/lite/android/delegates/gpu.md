# 使用 Interpreter API 进行 GPU 加速委托

使用图形处理单元 (GPU) 运行机器学习 (ML) 模型可以显著提高支持机器学习的应用的性能和用户体验。在 Android 设备上，您可以使用

[*委托*](../../performance/delegates)和以下 API 之一启用加速执行：

- Interpreter API - 本指南
- Task Library API - [指南](./gpu_task)
- 原生 (C/C++) API - [指南](./gpu_native)

本页介绍了如何在 Android 应用中使用 Interpreter API 为 TensorFlow Lite 模型启用 GPU 加速。有关将 GPU 委托用于 TensorFlow Lite 的更多信息，包括最佳做法和高级技术，请参阅 [GPU 委托](../../performance/gpu)页面。

## 将 GPU 与 Google Play 服务中的 TensorFlow Lite 一起使用

TensorFlow Lite [Interpreter API](https://tensorflow.org/lite/api_docs/java/org/tensorflow/lite/InterpreterApi) 提供了一组用于构建机器学习应用的通用 API。本部分介绍如何通过 Google Play 服务中的 TensorFlow Lite 将 GPU 加速器委托与这些 API 结合使用。

[Google Play 服务中的 TensorFlow Lite](../play_services) 是在 Android 上使用 TensorFlow Lite 的推荐途径。如果您的应用面向未运行 Google Play 的设备，请参阅[通过 Interpreter API 将 GPU 与独立版 TensorFlow Lite 一起使用](#standalone)部分。

### 添加项目依赖项

要启用对 GPU 委托的访问，请将 `com.google.android.gms:play-services-tflite-gpu` 添加到应用的 `build.gradle` 文件中：

```
dependencies {
    ...
    implementation 'com.google.android.gms:play-services-tflite-java:16.0.1'
    implementation 'com.google.android.gms:play-services-tflite-gpu:16.1.0'
}
```

### 启用 GPU 加速

然后，初始化支持 GPU 的 Google Play 服务中的 TensorFlow Lite：

<div>
  <devsite-selector>
    <section>
      <h3>Kotlin</h3>
      <p></p>
<pre class="prettyprint lang-kotlin">    val useGpuTask = TfLiteGpu.isGpuDelegateAvailable(context)

    val interpreterTask = useGpuTask.continueWith { useGpuTask -&gt;
      TfLite.initialize(context,
          TfLiteInitializationOptions.builder()
          .setEnableGpuDelegateSupport(useGpuTask.result)
          .build())
      }
      </pre>
    </section>
    <section>
      <h3>Java</h3>
      <p></p>
<pre class="prettyprint lang-java">    Task&lt;boolean&gt; useGpuTask = TfLiteGpu.isGpuDelegateAvailable(context);

    Task&lt;Options&gt; interpreterOptionsTask = useGpuTask.continueWith({ task -&gt;
      TfLite.initialize(context,
      TfLiteInitializationOptions.builder()
        .setEnableGpuDelegateSupport(true)
        .build());
    });
      </pre>
    </section>
  </devsite-selector>
</div>

您最终可以通过 `Interpreter Api.Options{/code 1} 传递 {code 0}GpuDelegateFactory` 来初始化解释器：

<div>
  <devsite-selector>
    <section>
      <h3>Kotlin</h3>
      <p></p>
<pre class="prettyprint lang-kotlin">
    val options = InterpreterApi.Options()
      .setRuntime(TfLiteRuntime.FROM_SYSTEM_ONLY)
      .addDelegateFactory(GpuDelegateFactory())

    val interpreter = InterpreterApi(model, options)

    // Run inference
    writeToInput(input)
    interpreter.run(input, output)
    readFromOutput(output)
      </pre>
    </section>
    <section>
      <h3>Java</h3>
      <p></p>
<pre class="prettyprint lang-java">
    Options options = InterpreterApi.Options()
      .setRuntime(TfLiteRuntime.FROM_SYSTEM_ONLY)
      .addDelegateFactory(new GpuDelegateFactory());

    Interpreter interpreter = new InterpreterApi(model, options);

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

## 将 GPU 与独立版 TensorFlow Lite 一起使用{:#standalone}

如果您的应用面向未运行 Google Play 的设备，则可以将 GPU 委托捆绑到您的应用并将其与独立版 TensorFlow Lite 一起使用。

### 添加项目依赖项

要启用对 GPU 委托的访问，请将 `org.tensorflow:tensorflow-lite-gpu-delegate-plugin` 添加到应用的 `build.gradle` 文件中：

```
dependencies {
    ...
    implementation 'org.tensorflow:tensorflow-lite'
    implementation 'org.tensorflow:tensorflow-lite-gpu-delegate-plugin'
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
  </devsite-selector>
</div>

有关使用 GPU 加速运行量化模型的更多信息，请参阅 [GPU 委托](../../performance/gpu#quantized-models)概述。
