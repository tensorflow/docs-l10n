# Task ライブラリによる GPU アクセラレーションのデリゲート

GPU を使って機械学習（ML）モデルを実行すると、ML 駆動型アプリケーションのパフォーマンスとユーザーエクスペリエンスを大幅に改善できます。Android デバイスでは、[*デリゲート*](../../performance/delegates)と以下のいずれかの API を使用して、モデルの GPU 高速実行が可能になります。

- Interpreter API - [ガイド](./gpu)
- Task library API - このガイド
- Native（C/C++）API - [ガイド](./gpu_native)

このページでは、Task ライブラリを使って、Android アプリで TensorFlow Lite モデルの GPU アクセラレーションを有効にする方法を説明します。ベストプラクティスや高度な手法など、TensorFlow Lite で GPU アクセラレーションを使用する方法についての詳細は、[GPU デリゲート](../../performance/gpu)のページをご覧ください。

## Google Play サービスによる TensorFlow Lite で GPU を使用する

TensorFlow Lite [Task ライブラリ](../../inference_with_metadata/task_library/overview)には、機械学習アプリケーションをビルドするための一連のタスク固有の API が含まれています。このセクションでは、Google Play サービスによる TensorFlow Lite を使用して、これらの API で GPU アクセラレータデリゲートを使用する方法について説明します。

[Google Play サービスによる TensorFlow Lite](../play_services) は、Android で TensorFlow Lite を使用するために推奨されている手段です。アプリケーションが Google Play を実行していないデバイスをターゲットとしている場合は、[Task ライブラリとスタンドアロン型 TensorFlow Lite による GPU](#standalone) のセクションをご覧ください。

### プロジェクト依存関係の追加

Google Play サービスを使用して、TensorFlow Lite Task ライブラリで GPU デリゲートへのアクセスを有効にするには、`com.google.android.gms:play-services-tflite-gpu` をアプリの `build.gradle` ファイルの依存関係に追加します。

```
dependencies {
  ...
  implementation 'com.google.android.gms:play-services-tflite-gpu:16.0.0'
}
```

### GPU アクセラレーションの有効化

次に、[`TfLiteGpu`](https://developers.google.com/android/reference/com/google/android/gms/tflite/gpu/support/TfLiteGpu) クラスを使用して、GPU デリゲートがデバイスに有効であるかを非同期的に確認し、[`BaseOptions`](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/task/core/BaseOptions.Builder) クラスで Task API モデルクラスの GPU デリゲートオプションを有効にします。たとえば、以下のコード例のように、`ObjectDetector` で GPU をセットアップできます。

<div>
  <devsite-selector>
    <section>
      <h3>Kotlin</h3>
      <p></p>
<pre class="prettyprint lang-kotlin">        val useGpuTask = TfLiteGpu.isGpuDelegateAvailable(context)

        lateinit val optionsTask = useGpuTask.continueWith { task -&gt;
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
      <p></p>
<pre class="prettyprint lang-java">      Task&lt;Boolean&gt; useGpuTask = TfLiteGpu.isGpuDelegateAvailable(context);

      Task&lt;ObjectDetectorOptions&gt; optionsTask = useGpuTask.continueWith({ task -&gt;
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

## スタンドアロン型 TensorFlow Lite で GPU を使用する {:#standalone}

アプリケーションが Google Play を実行していないデバイスをターゲットとしている場合は、GPU デリゲートをアプリケーションにバンドルして、TensorFlow Lite のスタンドアロンバージョンでそれを使用することができます。

### プロジェクト依存関係の追加

スタンドアロンバージョンの TensorFlow Lite を使用して TensorFlow Lite Task ライブラリで GPU デリゲートへのアクセスを有効にするには、`org.tensorflow:tensorflow-lite-gpu-delegate-plugin` をアプリの `build.gradle` ファイルの依存関係に追加します。

```
dependencies {
  ...
  implementation 'org.tensorflow:tensorflow-lite'
  implementation 'org.tensorflow:tensorflow-lite-gpu-delegate-plugin'
}
```

### GPU アクセラレーションの有効化

次に、[`BaseOptions`](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/task/core/BaseOptions.Builder) クラスを使用して、Task API モデルクラスの GPU デリゲートオプションを有効にします。たとえば、次のコード例のように、`ObjectDetector` で GPU を設定できます。

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
