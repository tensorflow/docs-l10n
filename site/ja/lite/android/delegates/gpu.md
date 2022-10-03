# Android の GPU アクセラレーションデリゲート

グラフィックス処理装置（GPU）を使用して機械学習（ML）モデルを実行すると、モデルのパフォーマンスと ML 対応アプリケーションのユーザーエクスペリエンスが大幅に向上します。Android デバイスでは、[*デリゲート*](../../performance/delegates)を使用して GPU で高速化されたモデルの実行を有効にできます。デリゲートは TensorFlow Lite のハードウェアドライバーとして機能し、モデルのコードを GPU プロセッサで実行できるようにします。

このページでは、Android アプリで TensorFlow Lite モデルの GPU アクセラレーションを有効にする方法について説明します。ベストプラクティスや高度な手法など、TensorFlow Lite の GPU デリゲートの使用に関する詳細については、[GPU デリゲート](../../performance/gpu)のページを参照してください。

## Task Library API で GPU を使用する

TensorFlow Lite [Task Library](../../inference_with_metadata/task_library/overview) は、機械学習アプリケーションをビルドするための一連のタスク固有の API を提供します。このセクションでは、これらの API で GPU アクセラレータデリゲートを使用する方法について説明します。

### プロジェクト依存関係の追加

次の依存関係を追加して、TensorFlow Lite Task Library で GPU デリゲート API へのアクセスを有効にします。以下のコード例のように、開発プロジェクトの `build.gradle` ファイルを更新して、 `tensorflow-lite-gpu-delegate-plugin` パッケージを含めます。

```
dependencies {
  ...
  implementation 'org.tensorflow:tensorflow-lite-gpu-delegate-plugin'
}
```

### GPU アクセラレーションの有効化

[`BaseOptions`](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/task/core/BaseOptions.Builder) クラスを使用して、Task API モデルクラスの GPU デリゲートオプションを有効にします。たとえば、次のコード例のように、`ObjectDetector` で GPU を設定できます。

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

## Interpreter API による GPU の使用

TensorFlow Lite [Interpreter API](../../api_docs/java/org/tensorflow/lite/InterpreterApi) は、機械学習アプリケーションをビルドするための一連の汎用 API を提供します。このセクションでは、これらの API で GPU アクセラレータデリゲートを使用する方法について説明します。

### プロジェクト依存関係の追加

次の依存関係を追加して GPU デリゲート API へのアクセスを有効にします。次のコード例のように、開発プロジェクトの `build.gradle` ファイルを更新して `org.tensorflow:tensorflow-lite-gpu` パッケージを含めます。

```
dependencies {
    ...
    implementation 'org.tensorflow:tensorflow-lite'
    implementation 'org.tensorflow:tensorflow-lite-gpu'
}
```

### GPU アクセラレーションの有効化

次に、`TfLiteDelegate` を使用して GPU で TensorFlow Lite を実行します。Java では、`Interpreter.Options` から `GpuDelegate` を指定できます。

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

注意: GPU デリゲートは、実行するスレッドと同じスレッドで作成する必要があります。そうでないと、「`TfLiteGpuDelegate Invoke: GpuDelegate must run on the same thread where it was initialized.`」が表示される可能性があります。

GPU デリゲートは、Android Studio の ML モデルバインディングでも使用できます。詳細については、[メタデータを使用してモデルインターフェイスを生成する](../../inference_with_metadata/codegen#acceleration)を参照してください。

## 高度な GPU サポート

このセクションでは、C API、C++ API、量子化モデルの使用など、Android の GPU デリゲートの高度な使用法について説明します。

### Android 用 C/C++ API

次のコード例に示すように、`TfLiteGpuDelegateV2Delete()` でデリゲートを作成し、`TfLiteGpuDelegateV2Create()` で破棄することにより、C または C++ で Android 用の TensorFlow Lite GPU デリゲートを使用します。

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

`TfLiteGpuDelegateOptionsV2` オブジェクトコードを確認して、カスタムオプションを使用してデリゲートインスタンスをビルドします。`TfLiteGpuDelegateOptionsV2Default()` でデフォルトオプションを初期化し、必要に応じて変更できます。

C または C++ の Android 用の TensorFlow Lite GPU デリゲートは、[Bazel](https://bazel.io) ビルドシステムを使用します。次のコマンドを使用してデリゲートをビルドできます。

```sh
bazel build -c opt --config android_arm64 tensorflow/lite/delegates/gpu:delegate                           # for static library
bazel build -c opt --config android_arm64 tensorflow/lite/delegates/gpu:libtensorflowlite_gpu_delegate.so  # for dynamic library
```

`Interpreter::ModifyGraphWithDelegate()` または `Interpreter::Invoke()` を呼び出す場合、呼び出し元はその時点のスレッドに `EGLContext` を持ち、`Interpreter::Invoke()` は同じ `EGLContext` から呼び出す必要があります。`EGLContext` が存在しない場合、デリゲートは内部で EGLContext を作成しますが、`Interpreter::ModifyGraphWithDelegate()` が呼び出されたスレッドと同じスレッドから `Interpreter::Invoke()` が常に呼び出されるようにする必要があります。

### 量子化モデル {:#quantized-models}

Android GPU デリゲートライブラリは、デフォルトで量子化モデルをサポートします。 GPU デリゲートで量子化モデルを使用するためにコードを変更する必要はありません。次のセクションでは、テストまたは実験目的で量子化サポートを無効にする方法について説明します。

#### 量子化モデルのサポートの無効化

次のコードは、量子化されたモデルのサポートを***無効***にする方法を示しています。

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

GPU アクセラレーションを使用した量子化モデルの実行の詳細については、[GPU デリゲート](../../performance/gpu#quantized-models)の概要を参照してください。
