# Android の音声および単語認識

このチュートリアルでは、TensorFlow Lite と構築済みの機械学習モデルを使用し、Android アプリで音声と発話を認識する方法について説明します。このチュートリアルで示すような音声分類モデルを使用すると、活動を検出したり、行動を特定したり、音声コマンドを認識したりできます。

![音声認識アニメーションデモ](https://storage.googleapis.com/download.tensorflow.org/tflite/examples/audio_classification.gif){: .attempt-right} このチュートリアルでは、サンプルコードをダウンロードして、プロジェクトを [Android Studio](https://developer.android.com/studio/) に読み込み、コードサンプルの重要な部分を説明して、この機能を自分のアプリに追加できるようにします。サンプルアプリコードは TensorFlow [Task Library for Audio](https://www.tensorflow.org/lite/inference_with_metadata/task_library/audio_classifier) を使用します。これはほとんどの音声データの録音と前処理を実行します。音声を前処理して、機械学習モデルで使用する詳細な方法については、[Audio Data Preparation and Augmentation](https://www.tensorflow.org/io/tutorials/audio) を参照してください。

## 機械学習による音声分類

このチュートリアルの機械学習モデルでは、Android デバイスでマイクを通して録音された音声サンプルの音声と単語を認識します。このチュートリアルのサンプルアプリでは、音声を認識するモデルである [YAMNet/classifier](https://tfhub.dev/google/lite-model/yamnet/classification/tflite/1) と、TensorFlow Lite [Model Maker](https://www.tensorflow.org/lite/models/modify/model_maker) ツールを使用して[トレーニング](https://www.tensorflow.org/lite/models/modify/model_maker/speech_recognition)された、特定の発話された単語を認識するモデルを切り替えることができます。このモデルでは、それぞれに 15600 サンプルが含まれた約 1 秒間の音声クリップに対して、予測を実行します。

## サンプルの設定と実行

このチュートリアルの最初の部分では、GitHub からサンプルをダウンロードし、Android Studio を使用して実行します。このチュートリアルの次のセクションでは、コードサンプルの関連するセクションを考察し、独自の Android アプリに応用できるようにします。

### システム要件

- [Android Studio](https://developer.android.com/studio/index.html) バージョン 2021.1.1 (Bumblebee) 以上
- Android SDK バージョン 31 以上
- OS バージョン SDK 24 (Android 7.0 - Nougat) 以上が搭載された Android デバイス (開発者モードが有効であること)

### サンプルコードの取得

サンプルコードのローカルコピーを作成します。このコードを使用して、Android Studio でプロジェクトを作成し、サンプルアプリケーションを実行します。

サンプルコードを複製してセットアップするには、次の手順を実行します。

1. git リポジトリを複製します。
    <pre class="devsite-click-to-copy">    git clone https://github.com/tensorflow/examples.git
        </pre>
2. sparse checkout を使用するように git インスタンスを構成します。これで、サンプルアプリのファイルのみを取得できます。
    ```
    cd examples
    git sparse-checkout init --cone
    git sparse-checkout set lite/examples/audio_classification/android
    ```

### プロジェクトのインポートと実行

ダウンロードしたサンプルコードからプロジェクトを作成し、プロジェクトをビルドして、実行します。

サンプルコードプロジェクトをインポートしてビルドするには、次の手順を実行します。

1. [Android Studio](https://developer.android.com/studio) を起動します。
2. Android Studio で **[File] &gt; [New] &gt; [Import Project]** を選択します。
3. `build.gradle` ファイルがあるサンプルコードディレクトリ (`.../examples/lite/examples/audio_classification/android/build.gradle`) に移動し、ディレクトリを選択します。

正しいディレクトリを選択すると、Android Studio で新しいプロジェクトが作成、ビルドされます。Android Studio を他のプロジェクトでも使用している場合、コンピューターの速度によっては、この処理に数分かかる場合があります。ビルドが完了すると、Android Studio の <strong>[Build Output]</strong> ステータスパネルに <code>BUILD SUCCESSFUL</code> メッセージが表示されます。

プロジェクトを実行するには、次の手順を実行します。

1. Android Studio で **[Run] &gt; [Run 'app']** を選択して、プロジェクトを実行します。
2. 接続されたマイクを搭載した Android デバイスを選択し、アプリをテストします。

注意: エミュレータを使用してアプリを実行する場合は、必ずホストコンピュータから[音声入力を有効化](https://developer.android.com/studio/releases/emulator#29.0.6-host-audio)してください。

次のセクションでは、このサンプルアプリを参考にして、この機能を独自のアプリに追加するために、既存のプロジェクトに行う必要がある修正について説明します。

## プロジェクト依存関係の追加

独自のアプリケーションで、TensorFlow Lite 機械学習モデルを実行するための特定のプロジェクト依存関係を追加する必要があります。また、音声などのデータを、使用中のモデルで処理できるテンソルデータに変換するユーティリティ関数にアクセスする必要があります。

サンプルアプリは次の TensorFlow Lite ライブラリを使用します。

- <em>TensorFlow Lite Task library Audio API</em> - 必要な音声データ入力クラス、機械学習モデルの実行、モデル処理の出力結果を提供します。

次の手順では、必要なプロジェクト依存関係を Android アプリプロジェクトに追加する方法について説明します。

モジュール依存関係を追加するには、次の手順を実行します。

1. TensorFlow Lite を使用するモジュールで、モジュールの `build.gradle` ファイルを更新し、次の依存関係を追加します。サンプルコードでは、このファイルは次の場所にあります。`.../examples/lite/examples/audio_classification/android/build.gradle`
    ```
    dependencies {
    ...
        implementation 'org.tensorflow:tensorflow-lite-task-audio'
    }
    ```
2. Android Studio で、**[File] &gt; [Sync Project with Gradle Files]** を選択して、プロジェクト依存関係を同期します。

## ML モデルの初期化

Android アプリで、モデルで予測を実行する前に、パラメータを使用して TensorFlow Lite 機械学習モデルを初期化する必要があります。これらの初期化パラメータはモデルに依存し、モデルが認識できる単語または音声の予測とラベルに関する既定の最低精度しきい値といった設定を含めることができます。

TensorFlow Lite モデルには、モデルを含む `*.tflite` ファイルがあります。このモデルファイルには、予測ロジックが含まれます。また、一般的に、予測クラス名などの予測結果を解釈する方法に関する[メタデータ](../../models/convert/metadata)が含まれます。モデルファイルは、コードサンプルのように、開発プロジェクトの `src/main/assets` ディレクトリに置いてください。

- `<project>/src/main/assets/yamnet.tflite`

便宜上の観点と、コードを読みやすくするため、この例では、モデルの設定を定義する比較オブジェクトが宣言されています。

アプリでモデルを初期化するには、次の手順を実行します。

1. モデルの設定を定義する比較オブジェクトを作成します。
    ```
    companion object {
      const val DISPLAY_THRESHOLD = 0.3f
      const val DEFAULT_NUM_OF_RESULTS = 2
      const val DEFAULT_OVERLAP_VALUE = 0.5f
      const val YAMNET_MODEL = "yamnet.tflite"
      const val SPEECH_COMMAND_MODEL = "speech.tflite"
    }
    ```
2. `AudioClassifier.AudioClassifierOptions` オブジェクトをビルドして、モデルの設定を作成します。
    ```
    val options = AudioClassifier.AudioClassifierOptions.builder()
      .setScoreThreshold(classificationThreshold)
      .setMaxResults(numOfResults)
      .setBaseOptions(baseOptionsBuilder.build())
      .build()
    ```
3. この設定オブジェクトを使用して、モデルを含む TensorFlow Lite [`AudioClassifier`](https://www.tensorflow.org/lite/inference_with_metadata/task_library/audio_classifier) オブジェクトを作成します。
    ```
    classifier = AudioClassifier.createFromFileAndOptions(context, "yamnet.tflite", options)
    ```

### ハードウェアアクセラレーションの有効化

アプリで TensorFlow Lite モデルを初期化するときには、ハードウェアアクセラレーション機能を使用して、モデルの予測計算を高速化することを検討してください。TensorFlow Lite [デリゲート](https://www.tensorflow.org/lite/performance/delegates)は、グラフィックス処理装置 (GPU) またはテンソル処理装置 (TPU) といった、モバイルデバイスの専用処理ハードウェアを使用して、機械学習の実行を高速化するソフトウェアモジュールです。コードサンプルでは、NNAPI Delegate を使用して、モデル実行のハードウェアアクセラレーションを処理しています。

```
val baseOptionsBuilder = BaseOptions.builder()
   .setNumThreads(numThreads)
...
when (currentDelegate) {
   DELEGATE_CPU -> {
       // Default
   }
   DELEGATE_NNAPI -> {
       baseOptionsBuilder.useNnapi()
   }
}
```

デリゲートを使用して TensorFlow Lite モデルを実行することをお勧めしますが、必須ではありません。TensorFlow Lite でのデリゲートの使用の詳細については、[TensorFlow Lite Delegates](https://www.tensorflow.org/lite/performance/delegates) を参照してください。

## モデルのデータの準備

Android アプリでは、コードによって、音声クリップなどの既存のデータが、モデルで処理できる[テンソル](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/Tensor)データ形式に変換されて、モデルに入力され、解釈されます。 テンソルのデータには、モデルをトレーニングするために使用されたデータ形式と一致する固有の次元または形状が必要です。

このコードサンプルで使用されている [YAMNet/classifier model](https://tfhub.dev/google/lite-model/yamnet/classification/tflite/1) とカスタマイズされた[読み上げコマンド](https://www.tensorflow.org/lite/models/modify/model_maker/speech_recognition) モデルでは、単一チャネルを表すテンソルデータオブジェクト、または 0.975 秒のクリップ (15600 サンプル) に 16kHz で録音されたモノ音声クリップを表すテンソルデータオブジェクトが許可されます。新しい音声データに対して予測を実行すると、音声データをそのサイズと形状のテンソルデータオブジェクトに変換する必要があります。データ変換は、TensorFlow Lite Task Library [Audio API](https://www.tensorflow.org/lite/inference_with_metadata/task_library/audio_classifier) によって処理されます。

サンプルコードの `AudioClassificationHelper` クラスでは、Android [AudioRecord](https://developer.android.com/reference/android/media/AudioRecord) オブジェクトを使用して、デバイスのマイクから入力されたライブ音声をアプリで録音します。このコードは、[AudioClassifier](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/task/audio/classifier/AudioClassifier) を使用して、そのオブジェクトをビルドおよび構成し、モデルに適したサンプリングレートで音声を録音します。また、コードは、AudioClassifier を使用して、[TensorAudio](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/support/audio/TensorAudio) オブジェクトをビルドし、変換された音声データを格納します。その後、TensorAudio オブジェクトがモデルに渡されて分析されます。

音声データを ML モデルに入力するには、次の手順を実行します。

- `AudioClassifier` オブジェクトを使用して、`TensorAudio` オブジェクトと `AudioRecord` オブジェクトを作成します。
    ```
    fun initClassifier() {
    ...
      try {
        classifier = AudioClassifier.createFromFileAndOptions(context, currentModel, options)
        // create audio input objects
        tensorAudio = classifier.createInputTensorAudio()
        recorder = classifier.createAudioRecord()
      }
    ```

注意: アプリは、Android デバイスのマイクを使用して音声を録音する権限を要求する必要があります。例については、プロジェクトの `fragments/PermissionsFragment` クラスを参照してください。権限の要求の詳細については、[Permissions on Android](https://developer.android.com/guide/topics/permissions/overview) を参照してください。

## 予測の実行

Android アプリで、[AudioRecord](https://developer.android.com/reference/android/media/AudioRecord) オブジェクトと [TensorAudio](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/support/audio/TensorAudio) オブジェクトを AudioClassifier オブジェクトに関連付けると、そのデータに対してモデルを実行し、予測または*推論*を生成できます。このチュートリアルのサンプルコードでは、特定のレートでライブ録音された音声入力ストリームのクリップに対して予測を実行します。

モデル実行ではリソースの消費量が非常に多くなるため、別のバックグラウンドスレッドで ML モデル予測を実行することが重要です。サンプルアプリでは、`[ScheduledThreadPoolExecutor](https://developer.android.com/reference/java/util/concurrent/ScheduledThreadPoolExecutor)` オブジェクトを使用して、アプリの他の機能とモデル処理を分離します。

単語のように先頭と末尾がはっきりした音声を認識する音声分類モデルでは、オーバーラップする音声クリップを分析することで、入力音声ストリームに対してより正確な予測を生成できます。このアプローチでは、クリップの最後に切り取られる単語に対する予測が欠落するのを回避できます。サンプルアプリでは、予測を実行するたびに、コードでは直近の 0.975 秒のクリップが音声録音バッファから取得され、分析されます。オーバーラップする音声クリップをモデルで分析するには、モデル分析スレッド実行プールの `interval` 値を、分析するクリップの長さよりも短い値に設定します。たとえば、モデルで 1 秒のクリップが分析され、間隔を 500 ミリ秒に設定する場合、毎回、前のクリップの後半と新しい音声データの 500 ミリ秒分が分析され、クリップ分析のオーバーラップが 50% になります。

音声データで予測の実行を開始するには、次の手順を実行します。

1. `AudioClassificationHelper.startAudioClassification()` メソッドを使用して、モデルの音声録音を開始します。
    ```
    fun startAudioClassification() {
      if (recorder.recordingState == AudioRecord.RECORDSTATE_RECORDING) {
        return
      }
      recorder.startRecording()
    }
    ```
2. `ScheduledThreadPoolExecutor` オブジェクトで固定レート `interval` を設定して、モデルで音声クリップから推論が生成される頻度を設定します。
    ```
    executor = ScheduledThreadPoolExecutor(1)
    executor.scheduleAtFixedRate(
      classifyRunnable,
      0,
      interval,
      TimeUnit.MILLISECONDS)
    ```
3. 上記のコードの `classifyRunnable` オブジェクトでは`AudioClassificationHelper.classifyAudio()` メソッドが実行されます。このメソッドでは、最新の使用可能な音声データがレコーダから読み込まれ、予測が実行されます。
    ```
    private fun classifyAudio() {
      tensorAudio.load(recorder)
      val output = classifier.classify(tensorAudio)
      ...
    }
    ```

注意: アプリケーションのメインの実行スレッドに対して ML モデル予測を実行しないでください。そのようにすると、アプリユーザーインターフェイスが低速になるか、応答しなくなる可能性があります。

### 予測処理の停止

アプリの音声処理フラグメントまたはアクティビティのフォーカスが失われるときに、アプリコードで音声分類の実行が停止することを確認します。機械学習モデルを実行すると、継続的に Android デバイスのバッテリ寿命に重大な影響を及ぼします。音声分類に関連付けられた Android アクティビティまたはフラグメントの `onPause()` メソッドを使用して、音声録音と予測処理を停止してください。

音声の録音と分類を停止するには、次の手順を実行します。

- 次のように、`AudioFragment` クラスで、`AudioClassificationHelper.stopAudioClassification()` メソッドを使用して、録音およびモデルの実行を停止します。
    ```
    override fun onPause() {
      super.onPause()
      if (::audioHelper.isInitialized ) {
        audioHelper.stopAudioClassification()
      }
    }
    ```

## モデル出力の処理

Android アプリで、音声クリップを処理した後に、追加のビジネスロジックを実行するか、結果をユーザーに表示するか、他のアクションを実行することで、アプリコードで処理する必要がある予測のリストがモデルで生成されます。特定の TensorFlow Lite モデルの出力では、生成される予測の数 (1 つ以上) と各予測に対する説明情報が異なります。サンプルアプリのモデルの場合、予測は認識された音声または単語のリストです。コードサンプルで使用される AudioClassifier オプションオブジェクトでは、`setMaxResults()` メソッドによる予測の最大数を設定できます。[Initialize the ML model](#Initialize_the_ML_model) セクションを参照してください。

モデルから予測結果を取得するには、次の手順を実行します。

1. [AudioClassifier](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/task/audio/classifier/AudioClassifier) オブジェクトの `classify()` メソッドの結果を取得し、それらをリスナーオブジェクトに渡します (コードリファレンス)。
    ```
    private fun classifyAudio() {
      ...
      val output = classifier.classify(tensorAudio)
      listener.onResult(output[0].categories, inferenceTime)
    }
    ```
2. リスナーの onResult() 関数を使用して、ビジネスロジックを実行するか、結果をユーザーに対して表示させ、出力を処理します。
    ```
    private val audioClassificationListener = object : AudioClassificationListener {
      override fun onResult(results: List<Category>, inferenceTime: Long) {
        requireActivity().runOnUiThread {
          adapter.categoryList = results
          adapter.notifyDataSetChanged()
          fragmentAudioBinding.bottomSheetLayout.inferenceTimeVal.text =
            String.format("%d ms", inferenceTime)
        }
      }
    ```

この例で使用されるモデルの場合、分類された音声または単語のラベル、および予測の信頼度を表す 0 ～ 1 の浮動小数点数の予測スコア (1 が最高の信頼度評価) が含まれます。一般的に、予測のスコアが 50% (0.5) 未満の場合、決定的ではないと見なされます。ただし、低い値の予測結果を処理する方法は、開発者やアプリケーションのニーズによって異なります。

予測結果がモデルによって返されると、結果がユーザーに表示されるか、追加のロジックが実行され、アプリケーションでその予測に対して処理が実行されます。サンプルコードの場合、アプリユーザーインターフェイスに特定された音声または単語の一覧が表示されます。

## 次のステップ

音声処理用の追加の TensorFlow Lite モデルについては、[TensorFlow Hub](https://tfhub.dev/s?deployment-format=lite&module-type=audio-embedding,audio-pitch-extraction,audio-event-classification,audio-stt) の [Pre-trained models guide](https://www.tensorflow.org/lite/models/trained) ページを使用を参照してください。TensorFlow Lite を使用したモバイルアプリケーションでの機械学習の実装の詳細については、[TensorFlow Lite 開発者ガイド](https://www.tensorflow.org/lite/guide)を参照してください。
