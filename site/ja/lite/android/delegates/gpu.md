# Interpreter API による GPU アクセラレーションのデリゲート

GPU を使って機械学習（ML）モデルを実行すると、ML 駆動型アプリケーションのパフォーマンスとユーザーエクスペリエンスを改善できます。Android デバイスでは、以下が可能になります。

[*デリゲート*](../../performance/delegates)と、以下のいずれかの API:

- Interpreter API - このガイド
- Task library API - [ガイド](./gpu_task)
- Native（C/C++）API - [ガイド](./gpu_native)

このページでは、Interpreter API を使って、Android アプリで TensorFlow Lite モデルの GPU アクセラレーションを有効にする方法を説明します。ベストプラクティスや高度な手法など、TensorFlow Lite で GPU アクセラレーションを使用する方法についての詳細は、[GPU デリゲート](../../performance/gpu)のページをご覧ください。

## Google Play サービスによる TensorFlow Lite で GPU を使用する

TensorFlow Lite [Interpreter API](https://tensorflow.org/lite/api_docs/java/org/tensorflow/lite/InterpreterApi) は、機械学習アプリケーションを構築するための汎用 API をまとめた API です。このセクションでは、これらの API を TensorFlow Lite と Google Play サービスを使用して GPU アクセラレータデリゲートを使用する方法について説明します。

[Google Play サービスによる TensorFlow Lite](../play_services) は、Android で TensorFlow Lite を使用するために推奨されている手段です。アプリケーションが Google Play を実行していないデバイスをターゲットとしている場合は、[Interpreter API とスタンドアロン型 TensorFlow Lite による GPU](#standalone) のセクションをご覧ください。

### プロジェクト依存関係の追加

GPU デリゲートへのアクセスを有効にするには、`com.google.android.gms:play-services-tflite-gpu` をアプリの `build.gradle` ファイルに追加します。

```
dependencies {
    ...
    implementation 'com.google.android.gms:play-services-tflite-java:16.0.1'
    implementation 'com.google.android.gms:play-services-tflite-gpu:16.1.0'
}
```

### GPU アクセラレーションの有効化

次に、GPU サポート付きで Google Play サービスによる TensorFlow Lite を初期化します。

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

これでようやく `InterpreterApi.Options` で `GpuDelegateFactory` を渡してインタープリタを初期化できます。

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

注意: GPU デリゲートは、実行するスレッドと同じスレッドで作成する必要があります。そうでないと、「`TfLiteGpuDelegate Invoke: GpuDelegate must run on the same thread where it was initialized.`」が表示される可能性があります。

GPU デリゲートは、Android Studio の ML モデルバインディングでも使用できます。詳細については、[メタデータを使用してモデルインターフェイスを生成する](../../inference_with_metadata/codegen#acceleration)を参照してください。

## スタンドアロン型 TensorFlow Lite で GPU を使用する {:#standalone}

アプリケーションが Google Play を実行していないデバイスをターゲットとしている場合は、GPU デリゲートをアプリケーションにバンドルして、TensorFlow Lite のスタンドアロンバージョンでそれを使用することができます。

### プロジェクト依存関係の追加

GPU デリゲートへのアクセスを有効にするには、`org.tensorflow:tensorflow-lite-gpu-delegate-plugin` をアプリの `build.gradle` ファイルに追加します。

```
dependencies {
    ...
    implementation 'org.tensorflow:tensorflow-lite'
    implementation 'org.tensorflow:tensorflow-lite-gpu-delegate-plugin'
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
  </devsite-selector>
</div>

GPU アクセラレーションを使用した量子化モデルの実行の詳細については、[GPU デリゲート](../../performance/gpu#quantized-models)の概要を参照してください。
