# Google Play サービスの TensorFlow Lite

TensorFlow Lite は、最新バージョンの Play サービスを実行しているすべての Android デバイスを対象に、Google Play サービスランタイムで提供されています。このランタイムでは、TensorFlow Lite ライブラリを静的にアプリにバンドルせずに、機械学習モデルを実行することができます。

Google Play サービス API を使用すると、アプリのサイズを縮小し、最新の安定したバージョンのライブラリによりパフォーマンスを向上させることができます。 Android で TensorFlow Lite を使用するには、Google Play サービスの TensorFlow Lite を使用することをお勧めします。

サンプルアプリケーションを実装するためのステップバイステップガイドを提供する[クイックスタート](../android/quickstart)を使用して、Play サービスランタイムを開始できます。アプリでスタンドアロンの TensorFlow Lite を既に使用している場合は、[スタンドアロンの TensorFlow Lite からの移行](#migrating)セクションを参照して、Play サービスランタイムを使用するように既存のアプリを更新してください。Google Play サービスの詳細については、 [Google Play サービス](https://developers.google.com/android/guides/overview)のウェブサイトを参照してください。

<aside class="note"><b>利用規約:</b> Google Play サービス API で TensorFlow Lite にアクセスまたは使用することにより、<a href="#tos">利用規約</a>に同意したことになります。API にアクセスする前に、該当するすべての条件とポリシーを読み、理解してください。</aside>

## Play サービス ランタイムの使用

Google Play サービスの TensorFlow Lite は、[TensorFlow Lite Task API](../api_docs/java/org/tensorflow/lite/task/core/package-summary) および [TensorFlow Lite Interpreter API](../api_docs/java/org/tensorflow/lite/InterpreterApi) を通じて利用できます。Task Library は、ビジュアル、オーディオ、およびテキストデータを使用して、一般的な機械学習タスク用に最適化されたすぐに使えるモデルインターフェースを提供します。TensorFlow ランタイムおよびサポートライブラリによって提供される TensorFlow Lite Interpreter API は、ML モデルを構築および実行するためのより汎用的なインターフェースを提供します。

以下のセクションでは、Google Play サービスで Interpreter API と Task Library API を実装する方法について説明します。アプリで Interpreter API と Task Library API の両方を使用することは可能ですが、ほとんどのアプリでは 1 つの API セットのみを使用する必要があります。

### Task Library API の使用

TensorFlow Lite Task API は Interpreter API をラップし、ビジュアル、オーディオ、およびテキストデータを使用する一般的な機械学習タスク用の高レベルのプログラミングインターフェースを提供します。[サポートされているタスク](../inference_with_metadata/task_library/overview#supported_tasks)のいずれかがアプリケーションに必要な場合は、Task API を使用する必要があります。

#### 1. プロジェクト依存関係の追加

プロジェクトの依存関係は、機械学習のユースケースによって異なります。Task API には、次のライブラリが含まれています。

- ビジョンライブラリ: `org.tensorflow:tensorflow-lite-task-vision-play-services`
- オーディオライブラリ: `org.tensorflow:tensorflow-lite-task-audio-play-services`
- テキストライブラリ: `org.tensorflow:tensorflow-lite-task-text-play-services`

依存関係の 1 つをアプリプロジェクトコードに追加して、TensorFlow Lite の Play サービス API にアクセスします。たとえば、次を使用してビジョンタスクを実装します。

```
dependencies {
...
    implementation 'org.tensorflow:tensorflow-lite-task-vision-play-services:0.4.2'
...
}
```

注意: TensorFlow Lite Tasks Audio ライブラリバージョン 0.4.2 の maven リポジトリは不完全です。代わりに、このライブラリにはバージョン 0.4.2.1 を使用してください: `org.tensorflow:tensorflow-lite-task-audio-play-services:0.4.2.1` 。

#### 2. TensorFlow Lite の初期化

TensorFlow Lite API を使用する*前*に、Google Play サービス API の TensorFlow Lite コンポーネントを初期化します。

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

重要: TensorFlow Lite API にアクセスするコードを実行する前に、 `TfLite.initialize` タスクが完了していることを確認してください。

ヒント: TensorFlow Lite モジュールは、アプリケーションが Play Store からインストールまたは更新されると同時にインストールされます。Google Play サービス API から `ModuleInstallClient` を使用して、モジュールの可用性を確認できます。モジュールの可用性の確認の詳細については、[ModuleInstallClient による API の可用性の確保](https://developers.google.com/android/guides/module-install-apis)を参照してください。

#### 3. 推論の実行

TensorFlow Lite コンポーネントを初期化した後、`detect()` メソッドを呼び出して推論を生成します。`detect()` メソッド内の正確なコードは、ライブラリとユースケースによって異なります。以下は、 `TfLiteVision` ライブラリを使用した単純なオブジェクト検出のユースケースです。

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

データ形式によっては、推論を生成する前に、`detect()` メソッド内でデータを前処理して変換する必要がある場合もあります。たとえば、オブジェクト検出器の画像データには次が必要です。

```kotlin
val imageProcessor = ImageProcessor.Builder().add(Rot90Op(-imageRotation / 90)).build()
val tensorImage = imageProcessor.process(TensorImage.fromBitmap(image))
val results = objectDetector?.detect(tensorImage)
```

### Interpreter API の使用

Interpreter API は、Task Library API よりも優れた制御と柔軟性を提供します。機械学習タスクが Task ライブラリでサポートされていない場合、または ML モデルを構築および実行するためにより汎用的なインターフェースが必要な場合は、Interpreter API を使用する必要があります。

#### 1. プロジェクト依存関係の追加

次の依存関係をアプリプロジェクトコードに追加し、TensorFlow Lite 用の Play サービス API にアクセスします。

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

#### 2. TensorFlow Lite の初期化

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

#### 3. Interpreter の作成とランタイムオプションの設定{:#step_3_interpreter}

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

上記の方法で実装してください。この方法では、Android ユーザーインターフェーススレッドのブロックを回避できます。スレッド実行をさらに細かく管理する必要がある場合は、次のように、`Tasks.await()` 呼び出しをインタープリタ作成に追加できます。

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

警告: `.await()` は、フォアグラウンドユーザーインターフェーススレッドで呼び出さないでください。ユーザーインターフェース要素の表示が中断され、ユーザーエクスペリエンスの質が低下します。

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

## ハードウェアアクセラレーション {:#hardware-acceleration}

TensorFlow Lite を使用すると、グラフィックス処理装置 (GPU) などの特殊なハードウェアプロセッサを使用して、モデルのパフォーマンスを高速化できます。[*デリゲート*](https://www.tensorflow.org/lite/performance/delegates)と呼ばれるハードウェアドライバーを使用して、これらの特殊なプロセッサを利用できます。Google Play サービスの TensorFlow Lite では、次のハードウェアアクセラレーションデリゲートを使用できます。

- *[GPU デリゲート](https://www.tensorflow.org/lite/performance/gpu)（おすすめ）* - このデリゲートは Google Play サービスを通じて提供され、Task API および Interpreter API の Play サービスバージョンと同様に動的に読み込まれます。

- [*NNAPI デリゲート*](https://www.tensorflow.org/lite/android/delegates/nnapi) - このデリゲートは Android 開発プロジェクトに含まれるライブラリの依存関係として利用でき、アプリにバンドルされています。

TensorFlow Lite でのハードウェアアクセラレーションの詳細については、[TensorFlow Lite デリゲート](https://www.tensorflow.org/lite/performance/delegates)ページを参照してください。

### デバイスの互換性の確認

すべてのデバイスが TFLite による GPU ハードウェアアクセラレーションをサポートしているわけではありません。エラーと潜在的なクラッシュを軽減するために、`TfLiteGpu.isGpuDelegateAvailable` メソッドを使用して、デバイスに GPU デリゲートとの互換性があるかどうかを確認します。

このメソッドを使用して、デバイスに GPU との互換性があるかどうかを確認し、GPU がサポートされていない場合のフォールバックとして CPU または NNAPI デリゲートを使用します。

```
useGpuTask = TfLiteGpu.isGpuDelegateAvailable(context)
```

`useGpuTask` のような変数を取得したら、それを使用して、デバイスが GPU デリゲートを使用するかどうかを判断できます。次の例は、Task Library API と Interpreter API の両方を使用してこれを行う方法を示しています。

**Task API の使用**

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

**Interpreter API の使用**

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

### Task Library API を使用した GPU

Task API で GPU デリゲートを使用するには:

1. プロジェクトの依存関係を更新して、Play サービスから GPU デリゲートを使用します。

    ```
    implementation 'com.google.android.gms:play-services-tflite-gpu:16.0.0'
    ```

2. GPU デリゲートを `setEnableGpuDelegateSupport` で初期化します。たとえば、`TfLiteVision` の GPU デリゲートを次のように初期化できます。

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

3. [`BaseOptions`](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/task/core/BaseOptions.Builder) で GPU デリゲートオプションを有効にします。

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

4. `.setBaseOptions` を使用してオプションを構成します。たとえば、次のように `ObjectDetector` で GPU を設定できます。

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

### Interpreter API を使用した GPU

Interpreter API で GPU デリゲートを使用するには:

1. プロジェクトの依存関係を更新して、Play サービスから GPU デリゲートを使用します。

    ```
    implementation 'com.google.android.gms:play-services-tflite-gpu:16.0.0'
    ```

2. TFlite の初期化で GPU デリゲートオプションを有効にします。

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

3. `InterpreterApi.Options()` 内で `addDelegateFactory()` を呼び出して、 `DelegateFactory` を使用するようにインタープリタオプションで GPU デリゲートを設定します。

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

## スタンドアロン TensorFlow Lite からの移行 {:#migrating}

スタンドアロン TensorFlow Lite から Play services API にアプリを移行する計画の場合は、アプリのプロジェクトコードの更新について、次の追加の指針を確認してください。

1. このページの[制限事項](#limitations)セクションを確認し、ユースケースがサポートされていることを確かめます。
2. コードを更新する前に、特にバージョン 2.1 より前の TensorFlow Lite を使用している場合は、モデルのパフォーマンスチェックと精度チェックを実行して、新しい実装と比較するベースラインを策定します。
3. すべてのコードを移行し、TensorFlow Lite で Play services API を使用する場合は、既存の TensorFlow Lite *ランタイムライブラリ*の依存関係 (<code>org.tensorflow:**tensorflow-lite**:*</code> のエントリ) を build.gradle ファイルから削除し、アプリのサイズを小さくしてください。
4. コードで `new Interpreter` オブジェクト作成処理をすべて特定し、InterpreterApi.create() 呼び出しを使用するように修正します。この新しい API は非同期です。つまり、ほとんどの場合、ドロップイン置換ではありません。このため、呼び出しの完了時にリスナーを登録する必要があります。[手順 3](#step_3_interpreter) のコードのコードスニペットを参照してください。
5. `org.tensorflow.lite.Interpreter` または `org.tensorflow.lite.InterpreterApi` クラスを使用して、`import org.tensorflow.lite.InterpreterApi;` と `import org.tensorflow.lite.InterpreterApi.Options.TfLiteRuntime;` をすべてのソースファイルに追加します。
6. 結果として `InterpreterApi.create()` の呼び出しのいずれかで引数が 1 つしかない場合は、`new InterpreterApi.Options()` を引数リストの最後に追加します。
7. `.setRuntime(TfLiteRuntime.FROM_SYSTEM_ONLY)` を、`InterpreterApi.create()` の呼び出しの最後の引数に追加します。
8. `org.tensorflow.lite.Interpreter` クラスの部分をすべて `org.tensorflow.lite.InterpreterApi` で置換します。

スタンドアロン TensorFlow Lite と Play services API をサイドバイサイドで使用する場合は、TensorFlow Lite 2.9 (以降) を使用する必要があります。TensorFlow Lite 2.8 以前のバージョンは、Play services API バージョンと互換性がありません。

## 制限事項

Google Play サービスの TensorFlow Lite には次の制限があります。

- ハードウェアアクセラレーションデリゲートのサポートは、[ハードウェアアクセラレーション](#hardware-acceleration)セクションにリストされているデリゲートに限定されます。その他のアクセラレーションデリゲートはサポートされていません。
- [ネイティブ API](https://www.tensorflow.org/lite/guide/inference#load_and_run_a_model_in_c) 経由での TensorFlow Lite へのアクセスはサポートされていません。TensorFlow Lite Java API のみが Google Play サービスで提供されています。
- カスタム ops を含む実験用または廃止予定の TensorFlow Lite API はサポートされていません。

## サポートとフィードバック {:#support}

TensorFlow Issue Tracker を使用すると、フィードバックを提出し、サポートを受けることができます。Google Play サービスの TensorFlow Lite 用の[問題テンプレート](https://github.com/tensorflow/tensorflow/issues/new?title=TensorFlow+Lite+in+Play+Services+issue&template=tflite-in-play-services.md)を使用して、問題およびサポート要求を報告してください。

## 利用規約 {:#tos}

Google Play サービス API での TensorFlow Lite の使用には、[Google API サービス利用規約](https://developers.google.com/terms/)が適用されます。

### プライバシーとデータ収集

Google Play サービス API で TensorFlow Lite を使用する場合、画像、動画、テキストなどの入力データの処理はすべてデバイス上で行われ、Google Play サービス API の TensorFlow Lite はデータを Google サーバーに送信しません。その結果、デバイスから出てはならないデータを処理するために API を使用できます。

Google Play services API の TensorFlow Lite は、バグ修正、更新されたモデル、ハードウェアアクセラレータの互換性情報といったデータを受信するために、時々 Google サーバーに接続することがあります。Google Play services API の TensorFlow Lite では、アプリの API のパフォーマンスと使用状況に関するメトリクスも Google に送信されます。Google は、このメトリクスデータを使用して、パフォーマンス測定、デバッグ、API の保守と改良、誤用または悪用を検出します。詳細については、[プライバシーポリシー](https://policies.google.com/privacy)を参照してください。

**適用される法律の義務に従い、Google が Google Play サービスメトリクスデータで行う TensorFlow Lite の処理について、アプリのユーザーに通知するのは、あなたの責任です。**

Google が収集するデータは次のとおりです。

- デバイス情報 (メーカー、モデル、OS バージョン、ビルドなど) と使用可能な ML ハードウェアアクセラレータ (GPU および DSP)。診断および使用状況分析で使用されます。
- デバイス識別子。診断および使用状況分析で使用されます。
- アプリ情報 (パッケージ名、アプリバージョン)。診断および使用状況分析で使用されます。
- API 構成 (使用中のデリゲートなど)。診断および使用状況分析で使用されます。
- イベントタイプ (インタープリタ作成、推論など)。診断および使用状況分析で使用されます。
- エラーコード。診断で使用されます。
- パフォーマンスメトリクス。診断で使用されます。

## Next steps

TensorFlow Lite を使用したモバイルアプリケーションでの機械学習の実装の詳細については、[TensorFlow Lite 開発者ガイド](https://www.tensorflow.org/lite/guide)を参照してください。画像分類、オブジェクト検出、他の用途で使用されるその他の TensorFlow Lite モデルについては、[TensorFlow Hub](https://tfhub.dev/s?deployment-format=lite) を参照してください。
