# Google Play サービスの TensorFlow Lite

TensorFlow Lite is available in Google Play services runtime for all Android devices running the current version of Play services. This runtime allows you to run machine learning (ML) models without statically bundling TensorFlow Lite libraries into your app.

With the Google Play services API, you can reduce the size of your apps and gain improved performance from the latest stable version of the libraries. TensorFlow Lite in Google Play services is the recommended way to use TensorFlow Lite on Android.

You can get started with the Play services runtime with the [Quickstart](../android/quickstart), which provides a step-by-step guide to implement a sample application. If you are already using stand-alone TensorFlow Lite in your app, refer to the [Migrating from stand-alone TensorFlow Lite](#migrating) section to update an existing app to use the Play services runtime. For more information about Google Play services, see the [Google Play services](https://developers.google.com/android/guides/overview) website.

<aside class="note"> <b>Terms:</b> By accessing or using TensorFlow Lite in Google Play services APIs, you agree to the <a href="#tos">Terms of Service</a>. Please read and understand all applicable terms and policies before accessing the APIs. </aside>

## Using the Play services runtime

TensorFlow Lite in Google Play services is available through the [TensorFlow Lite Task API](../api_docs/java/org/tensorflow/lite/task/core/package-summary) and [TensorFlow Lite Interpreter API](../api_docs/java/org/tensorflow/lite/InterpreterApi). The Task Library provides optimized out-of-box model interfaces for common machine learning tasks using visual, audio, and text data. The TensorFlow Lite Interpreter API, provided by the TensorFlow runtime and support libraries, provides a more general-purpose interface for building and running ML models.

The following sections provide instructions on how to implement the Interpreter and Task Library APIs in Google Play services. While it is possible for an app to use both the Interpreter APIs and Task Library APIs, most apps should only use one set of APIs.

### Using the Task Library APIs

The TensorFlow Lite Task API wraps the Interpreter API and provides a high-level programming interface for common machine learning tasks that use visual, audio, and text data. You should use the Task API if your application requires one of the [supported tasks](../inference_with_metadata/task_library/overview#supported_tasks).

#### 1. プロジェクト依存関係の追加

Your project dependency depends on your machine learning use case. The Task APIs contain the following libraries:

- Vision library: `org.tensorflow:tensorflow-lite-task-vision-play-services`
- Audio library: `org.tensorflow:tensorflow-lite-task-audio-play-services`
- Text library: `org.tensorflow:tensorflow-lite-task-text-play-services`

Add one of the dependencies to your app project code to access the Play services API for TensorFlow Lite. For example, use the following to implement a vision task:

```
dependencies {
...
    implementation 'org.tensorflow:tensorflow-lite-task-vision-play-services:0.4.2'
...
}
```

Caution: The TensorFlow Lite Tasks Audio library version 0.4.2 maven repository is incomplete. Use version 0.4.2.1 for this library instead: `org.tensorflow:tensorflow-lite-task-audio-play-services:0.4.2.1`.

#### 2. TensorFlow Lite の初期化

Initialize the TensorFlow Lite component of the Google Play services API *before* using the TensorFlow Lite APIs. The following example initializes the vision library:

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

Important: Make sure the `TfLite.initialize` task completes before executing code that accesses TensorFlow Lite APIs.

Tip: The TensorFlow Lite modules are installed at the same time your application is installed or updated from the Play Store. You can check the availability of the modules by using `ModuleInstallClient` from the Google Play services APIs. For more information on checking module availability, see [Ensuring API availability with ModuleInstallClient](https://developers.google.com/android/guides/module-install-apis).

#### 3. Run inferences

After initializing the TensorFlow Lite component, call the `detect()` method to generate inferences. The exact code within the `detect()` method varies depending on the library and use case. The following is for a simple object detection use case with the `TfLiteVision` library:

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

Depending on the data format, you may also need to preprocess and convert your data within the `detect()` method before generating inferences. For example, image data for an object detector requires the following:

```kotlin
val imageProcessor = ImageProcessor.Builder().add(Rot90Op(-imageRotation / 90)).build()
val tensorImage = imageProcessor.process(TensorImage.fromBitmap(image))
val results = objectDetector?.detect(tensorImage)
```

### Using the Interpreter APIs

The Interpreter APIs offer more control and flexibility than the Task Library APIs. You should use the Interpreter APIs if your machine learning task is not supported by the Task library, or if you require a more general-purpose interface for building and running ML models.

#### 1. Add project dependencies

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

#### 2. Add initialization of TensorFlow Lite

TensorFlow Lite API を使用する*前*に、Google Play services API の TensorFlow Lite コンポーネントを初期化します。

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

注意: TensorFlow Lite API にアクセスするコードを実行する前に、`TfLite.initialize` タスクが完了していることを確認してください。次のセクションで示すように、`addOnSuccessListener()` メソッドを使用してください。

#### 3. Create an Interpreter and set runtime option {:#step_3_interpreter}

次のサンプルコードで示すように、`InterpreterApi.create()` を使用してインタープリタを作成し、`InterpreterApi.Options.setRuntime()` を呼び出して Google Play サービスランタイムを使用するように構成します。

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

You should use the implementation above because it avoids blocking the Android user interface thread. If you need to manage thread execution more closely, you can add a `Tasks.await()` call to interpreter creation:

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

Warning: Do not call `.await()` on the foreground user interface thread because it interrupts display of user interface elements and creates a poor user experience.

#### 4. 推論の実行

作成した `interpreter` オブジェクトを使用して、`run()` メソッドを呼び出し、推論を生成します。

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

## Hardware acceleration {:#hardware-acceleration}

TensorFlow Lite allows you to accelerate the performance of your model using specialized hardware processors, such as graphics processing units (GPUs). You can take advantage of these specialized processors using hardware drivers called [*delegates*](https://www.tensorflow.org/lite/performance/delegates). You can use the following hardware acceleration delegates with TensorFlow Lite in Google Play services:

- *[GPU delegate](https://www.tensorflow.org/lite/performance/gpu) (recommended)* - This delegate is provided through Google Play services and is dynamically loaded, just like the Play services versions of the Task API and Interpreter API.

- [*NNAPI delegate*](https://www.tensorflow.org/lite/android/delegates/nnapi) - This delegate is available as an included library dependency in your Android development project, and is bundled into your app.

For more information about hardware acceleration with TensorFlow Lite, see the [TensorFlow Lite Delegates](https://www.tensorflow.org/lite/performance/delegates) page.

### Checking device compatibility

Not all devices support GPU hardware acceleration with TFLite. In order to mitigate errors and potential crashes, use the `TfLiteGpu.isGpuDelegateAvailable` method to check whether a device is compatible with the GPU delegate.

Use this method to confirm whether a device is compatible with GPU, and use CPU or the NNAPI delegate as a fallback for when GPU is not supported.

```
useGpuTask = TfLiteGpu.isGpuDelegateAvailable(context)
```

Once you have a variable like `useGpuTask`, you can use it to determine whether devices use the GPU delegate. The following examples show how this can be done with both the Task Library and Interpreter APIs.

**With the Task Api**

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

**With the Interpreter Api**

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

### GPU with Task Library APIs

To use the GPU delegate with the Task APIs:

1. Update the project dependencies to use the GPU delegate from Play services:

    ```
    implementation 'com.google.android.gms:play-services-tflite-gpu:16.0.0'
    ```

2. Initialize the GPU delegate with `setEnableGpuDelegateSupport`. For example, you can initialize the GPU delegate for `TfLiteVision` with the following:

    <div>
    <devsite-selector>
    <section>
      <h3>Kotlin</h3>
        <pre class="prettyprint">        TfLiteVision.initialize(context, TfLiteInitializationOptions.builder().setEnableGpuDelegateSupport(true).build())
            </pre>
    </section>
    <section>
      <h3>Java</h3>
        <pre class="prettyprint">        TfLiteVision.initialize(context, TfLiteInitializationOptions.builder().setEnableGpuDelegateSupport(true).build());
            </pre>
    </section>
    </devsite-selector>
    </div>

3. Enable the GPU delegate option with [`BaseOptions`](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/task/core/BaseOptions.Builder):

    <div>
    <devsite-selector>
    <section>
      <h3>Kotlin</h3>
        <pre class="prettyprint">        val baseOptions = BaseOptions.builder().useGpu().build()
            </pre>
    </section>
    <section>
      <h3>Java</h3>
        <pre class="prettyprint">        BaseOptions baseOptions = BaseOptions.builder().useGpu().build();
            </pre>
    </section>
    </devsite-selector>
    </div>

4. Configure the options using `.setBaseOptions`. For example, you can set up GPU in `ObjectDetector` with the following:

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

### GPU with Interpreter APIs

To use the GPU delegate with the Interpreter APIs:

1. Update the project dependencies to use the GPU delegate from Play services:

    ```
    implementation 'com.google.android.gms:play-services-tflite-gpu:16.0.0'
    ```

2. Enable the GPU delegate option in the TFlite initialization:

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

3. Set GPU delegate in interpreter options to use `DelegateFactory` by calling `addDelegateFactory()` within `InterpreterApi.Options()`:

    <div>
    <devsite-selector>
    <section>
      <h3>Kotlin</h3>
        <pre class="prettyprint">        val interpreterOption = InterpreterApi.Options()
             .setRuntime(TfLiteRuntime.FROM_SYSTEM_ONLY)
             .addDelegateFactory(GpuDelegateFactory())
            </pre>
    </section>
    <section>
      <h3>Java</h3>
        <pre class="prettyprint">        Options interpreterOption = InterpreterApi.Options()
              .setRuntime(TfLiteRuntime.FROM_SYSTEM_ONLY)
              .addDelegateFactory(new GpuDelegateFactory());
            </pre>
    </section>
    </devsite-selector>
    </div>

## Migrating from stand-alone TensorFlow Lite {:#migrating}

If you are planning to migrate your app from stand-alone TensorFlow Lite to the Play services API, review the following additional guidance for updating your app project code:

1. このページの[制限事項](#limitations)セクションを確認し、ユースケースがサポートされていることを確かめます。
2. コードを更新する前に、特にバージョン 2.1 より前の TensorFlow Lite を使用している場合は、モデルのパフォーマンスチェックと精度チェックを実行して、新しい実装と比較するベースラインを策定します。
3. すべてのコードを移行し、TensorFlow Lite で Play services API を使用する場合は、既存の TensorFlow Lite *ランタイムライブラリ*の依存関係 (<code>org.tensorflow:**tensorflow-lite**:*</code> のエントリ) を build.gradle ファイルから削除し、アプリのサイズを小さくしてください。
4. Identify all occurrences of `new Interpreter` object creation in your code, and modify it so that it uses the InterpreterApi.create() call. This new API is asynchronous, which means in most cases it's not a drop-in replacement, and you must register a listener for when the call completes. Refer to the code snippet in [Step 3](#step_3_interpreter) code.
5. `org.tensorflow.lite.Interpreter` または `org.tensorflow.lite.InterpreterApi` クラスを使用して、`import org.tensorflow.lite.InterpreterApi;` と `import org.tensorflow.lite.InterpreterApi.Options.TfLiteRuntime;` をすべてのソースファイルに追加します。
6. 結果として `InterpreterApi.create()` の呼び出しのいずれかで引数が 1 つしかない場合は、`new InterpreterApi.Options()` を引数リストの最後に追加します。
7. `.setRuntime(TfLiteRuntime.FROM_SYSTEM_ONLY)` を、`InterpreterApi.create()` の呼び出しの最後の引数に追加します。
8. `org.tensorflow.lite.Interpreter` クラスの部分をすべて `org.tensorflow.lite.InterpreterApi` で置換します。

If you want to use stand-alone TensorFlow Lite and the Play services API side-by-side, you must use TensorFlow Lite 2.9 (or later). TensorFlow Lite 2.8 and earlier versions are not compatible with the Play services API version.

## Limitations

TensorFlow Lite in Google Play services has the following limitations:

- Support for hardware acceleration delegates is limited to the delegates listed in the [Hardware acceleration](#hardware-acceleration) section. No other acceleration delegates are supported.
- [ネイティブ API](https://www.tensorflow.org/lite/guide/inference#load_and_run_a_model_in_c) 経由での TensorFlow Lite へのアクセスはサポートされていません。TensorFlow Lite Java API のみが Google Play サービスで提供されています。
- カスタム ops を含む実験用または廃止予定の TensorFlow Lite API はサポートされていません。

## サポートとフィードバック {:#support}

You can provide feedback and get support through the TensorFlow Issue Tracker. Please report issues and support requests using the [Issue template](https://github.com/tensorflow/tensorflow/issues/new?title=TensorFlow+Lite+in+Play+Services+issue&template=tflite-in-play-services.md) for TensorFlow Lite in Google Play services.

## Terms of service {:#tos}

Use of TensorFlow Lite in Google Play services APIs is subject to the [Google APIs Terms of Service](https://developers.google.com/terms/).

### Privacy and data collection

When you use TensorFlow Lite in Google Play services APIs, processing of the input data, such as images, video, text, fully happens on-device, and TensorFlow Lite in Google Play services APIs does not send that data to Google servers. As a result, you can use our APIs for processing data that should not leave the device.

Google Play services API の TensorFlow Lite は、バグ修正、更新されたモデル、ハードウェアアクセラレータの互換性情報といったデータを受信するために、時々 Google サーバーに接続することがあります。Google Play services API の TensorFlow Lite では、アプリの API のパフォーマンスと使用状況に関するメトリクスも Google に送信されます。Google は、このメトリクスデータを使用して、パフォーマンス測定、デバッグ、API の保守と改良、誤用または悪用を検出します。詳細については、[プライバシーポリシー](https://policies.google.com/privacy)を参照してください。

**You are responsible for informing users of your app about Google's processing of TensorFlow Lite in Google Play services APIs metrics data as required by applicable law.**

Google が収集するデータは次のとおりです。

- デバイス情報 (メーカー、モデル、OS バージョン、ビルドなど) と使用可能な ML ハードウェアアクセラレータ (GPU および DSP)。診断および使用状況分析で使用されます。
- デバイス識別子。診断および使用状況分析で使用されます。
- アプリ情報 (パッケージ名、アプリバージョン)。診断および使用状況分析で使用されます。
- API 構成 (使用中のデリゲートなど)。診断および使用状況分析で使用されます。
- イベントタイプ (インタープリタ作成、推論など)。診断および使用状況分析で使用されます。
- エラーコード。診断で使用されます。
- パフォーマンスメトリクス。診断で使用されます。

## Next steps

For more information about implementing machine learning in your mobile application with TensorFlow Lite, see the [TensorFlow Lite Developer Guide](https://www.tensorflow.org/lite/guide). You can find additional TensorFlow Lite models for image classification, object detection, and other applications on the [TensorFlow Hub](https://tfhub.dev/s?deployment-format=lite).
