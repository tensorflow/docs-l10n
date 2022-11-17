# Android で質問に答える

![Android での質問応答サンプルアプリ](../../examples/bert_qa/images/screenshot.gif){: .attempt-right width="250px"}

このチュートリアルでは、TensorFlow Lite を使用して Android アプリケーションを構築し、自然言語テキストで構造化された質問への回答を提供する方法を示します。[サンプルアプリケーション](https://github.com/tensorflow/examples/tree/master/lite/examples/bert_qa/android)では、[自然言語（NL）のタスクライブラリ](../../inference_with_metadata/task_library/overview#supported_tasks)内の *BERT 質問応答*（[`BertQuestionAnswerer`](../../inference_with_metadata/task_library/bert_question_answerer)）API を使用して、質問応答機械学習モデルを有効にします。アプリケーションは、物理的な Android デバイス用に設計されていますが、デバイスエミュレーターでも実行できます。

既存のプロジェクトを更新する場合は、サンプルアプリケーションをリファレンスまたはテンプレートとして使用できます。既存のアプリケーションに質問応答機能を追加する方法については、[アプリケーションの更新と変更](#modify_applications)を参照してください。

## 質問応答の概要

*質問応答*は、自然言語で提示された質問に答える機械学習タスクです。トレーニング済みの質問応答モデルは、入力としてテキストのパッセージと質問を受け取り、パッセージ内の情報の解釈に基づいて質問に回答しようとします。

質問応答モデルは、テキストのさまざまなセグメントに基づく質問と回答のペアと共に読解力データセットで構成される質問応答データセットでトレーニングされます。

このチュートリアルのモデルの生成方法の詳細については、[TensorFlow Lite モデルメーカーを使用した BERT 質問応答](https://www.tensorflow.org/lite/models/modify/model_maker/question_answer)チュートリアルをご覧ください。

## モデルとデータセット

サンプルアプリでは、モバイル BERT Q&amp;A（[`mobilebert`](https://tfhub.dev/tensorflow/lite-model/mobilebert/1/metadata/1)）モデルを使用します。これは、[BERT](https://arxiv.org/abs/1810.04805)（トランスフォーマーからの双方向エンコーダ表現）の軽量で高速なバージョンです。`mobilebert` の詳細については、研究論文の [MobileBERT: a Compact Task-Agnostic BERT for Resource-Limited Devices](https://arxiv.org/abs/2004.02984) をご覧ください。

`mobilebert` モデルは、Stanford Question Answering Dataset（[SQuAD](https://rajpurkar.github.io/SQuAD-explorer/)）データセット、ウィキペディアの記事と各記事の一連の質問と回答のペアで構成される読解力データセットを使用してトレーニングされました。

## サンプルアプリのセットアップと実行

質問応答アプリケーションをセットアップするには、サンプルアプリを [GitHub](https://github.com/tensorflow/examples/tree/master/lite/examples/bert_qa/android) からダウンロードし、[Android Studio](https://developer.android.com/studio/) を使用して実行します。

### システム要件

- **[Android Studio](https://developer.android.com/studio/index.html)** バージョン 2021.1.1（Bumblebee）以降。
- Android SDK バージョン 31 以上
- OS バージョン SDK 21（Android 7.0 - Nougat）以上が搭載された Android デバイス（[開発者モード](https://developer.android.com/studio/debug/dev-options)が有効であること、または Android Emulator を使用）

### サンプルコードの取得

サンプルコードのローカルコピーを作成します。このコードを使用して、Android Studio でプロジェクトを作成し、サンプルアプリケーションを実行します。

サンプルコードを複製してセットアップするには、次の手順を実行します。

1. git リポジトリを複製します
    <pre class="devsite-click-to-copy">    git clone https://github.com/tensorflow/examples.git
        </pre>
2. 必要に応じて、sparse checkout を使用するように git インスタンスを構成します。これで、サンプルアプリのファイルのみを取得できます。
    <pre class="devsite-click-to-copy">    cd examples
        git sparse-checkout init --cone
        git sparse-checkout set lite/examples/bert_qa/android
        </pre>

### プロジェクトのインポートと実行

ダウンロードしたサンプルコードからプロジェクトを作成し、プロジェクトをビルドして、実行します。

サンプルコードのプロジェクトをインポートしてビルドするには、次の手順を実行します。

1. [Android Studio](https://developer.android.com/studio) を起動します。
2. Android Studio で、**File &gt; New &gt; Import Project** を選択します。
3. build.gradle ファイル（`.../examples/lite/examples/bert_qa/android/build.gradle`）があるサンプルコードのディレクトリに移動し、そのディレクトリを選択します。
4. Android Studio で Gradle Sync が要求される場合は、OK をクリックします。
5. Android デバイスがコンピュータに接続され、開発者モードが有効であることを確認します。緑色の `Run` 矢印をクリックします。

正しいディレクトリを選択すると、Android Studio で新しいプロジェクトが作成、ビルドされます。Android Studio を他のプロジェクトでも使用している場合、コンピュータの速度によっては、この処理に数分かかる場合があります。ビルドが完了すると、Android Studio の <strong>Build Output</strong> ステータスパネルに <code>BUILD SUCCESSFUL</code> メッセージが表示されます。

プロジェクトを実行するには、次の手順を実行します。

1. Android Studio で **Run &gt; Run…** を選択して、プロジェクトを実行します。
2. 接続されている Android デバイス（またはエミュレーター）を選択して、アプリをテストします。

### アプリケーションの使用

Android Studio でプロジェクトを実行すると、接続されたデバイスまたはデバイスエミュレーターでアプリケーションが自動的に開きます。

質問応答のサンプルアプリを使用するには、次を実行します。

1. テーマのリストからトピックを選択します。
2. 提案された質問を選択するか、テキストボックスに独自の質問を入力します。
3. オレンジ色の矢印を切り替えて、モデルを実行します。

アプリケーションは、パッセージテキストから質問への回答を特定しようとします。モデルがパッセージ内で回答を検出すると、アプリケーションはユーザーのために関連するテキストの範囲を強調表示します。

これで、機能する質問応答アプリケーションができました。以下のセクションを使用して、サンプルアプリケーションがどのように機能するか、および質問応答機能を本番アプリケーションに実装する方法をよりよく理解してください。

- [アプリケーションの仕組み](#how_it_works) - サンプルアプリケーションの構造と主要なファイルのチュートリアル。

- [アプリケーションの変更](#modify_applications) - 質問応答を既存のアプリケーションに追加する手順。

## サンプルアプリの仕組み {:#how_it_works}

このアプリケーションは、<a>自然言語（NL）パッケージのタスクライブラリ</a>内で <code>BertQuestionAnswerer</code> API を使用します。MobileBERT モデルは、TensorFlow Lite [Model Maker](https://www.tensorflow.org/lite/models/modify/model_maker/question_answer) を使用してトレーニングされました。アプリケーションはデフォルトで CPU で実行され、GPU または NNAPI デリゲートを使用したハードウェアアクセラレーションのオプションがあります。

次のファイルとディレクトリには、このアプリケーションの重要なコードが含まれています。

- [BertQaHelper.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/bert_qa/android/app/src/main/java/org/tensorflow/lite/examples/bertqa/BertQaHelper.kt) - 質問応答を初期化し、モデルとデリゲートの選択を処理します。
- [ResultsAdapter.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/bert_qa/android/app/src/main/java/org/tensorflow/lite/examples/bertqa/fragments/QaFragment.kt) - 結果を処理してフォーマットします。
- [MainActivity.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/bert_qa/android/app/src/main/java/org/tensorflow/lite/examples/bertqa/MainActivity.kt) - アプリの組織化のロジックを提供します。

## アプリケーションの変更{:#modify_applications}

次のセクションでは、独自の Android アプリを変更して、サンプルアプリに示されているモデルを実行するための主要な手順について説明します。これらの手順では、サンプルアプリを参照ポイントとして使用します。独自のアプリに必要な特定の変更は、サンプルアプリとは異なる場合があります。

### Android プロジェクトを開く、または作成する

これらの手順の残りの部分に従うには、Android Studio の Android 開発プロジェクトが必要です。以下の手順に沿って、既存のプロジェクトを開くか、新しいプロジェクトを作成します。

既存の Android 開発プロジェクトを開くには、次を実行します。

- Android Studio で、*File &gt; Open* を選択し、既存のプロジェクトを選択します。

基本的な Android 開発プロジェクトを作成するには、次を実行します。

- Android Studio の手順に沿って、[基本的なプロジェクトを作成](https://developer.android.com/studio/projects/create-project)します。

Android Studio の使用の詳細については、[Android Studio のドキュメント](https://developer.android.com/studio/intro)を参照してください。

### プロジェクト依存関係の追加

独自のアプリケーションで、特定のプロジェクトの依存関係を追加して、TensorFlow Lite 機械学習モデルを実行し、ユーティリティ関数にアクセスします。これらの関数は、文字列などのデータをモデルで処理できるテンソルデータ形式に変換します。次の手順では、必要なプロジェクトとモジュールの依存関係を独自の Android アプリ プロジェクトに追加する方法について説明します。

モジュール依存関係を追加するには、次の手順を実行します。

1. TensorFlow Lite を使用するモジュールで、モジュールの `build.gradle` ファイルを更新して、次の依存関係を追加します。

    サンプルアプリケーションでは、依存関係は [app/build.gradle](https://github.com/tensorflow/examples/blob/master/lite/examples/bert_qa/android/app/build.gradle) にあります。

    ```
    dependencies {
      ...
      // Import tensorflow library
      implementation 'org.tensorflow:tensorflow-lite-task-text:0.3.0'

      // Import the GPU delegate plugin Library for GPU inference
      implementation 'org.tensorflow:tensorflow-lite-gpu-delegate-plugin:0.4.0'
      implementation 'org.tensorflow:tensorflow-lite-gpu:2.9.0'
    }
    ```

    プロジェクトには Text タスクライブラリ（`tensorflow-lite-task-text`）が含まれている必要があります。

    グラフィックス処理装置（GPU）で実行するためにこのアプリを変更する場合、GPU ライブラリ（`tensorflow-lite-gpu-delegate-plugin`）は GPU でアプリを実行するためのインフラストラクチャを提供し、デリゲート（`tensorflow-lite-gpu`）は、互換性リストを提供します。

2. Android Studio で、**File &gt; Sync Project with Gradle Files** を選択して、プロジェクト依存関係を同期します。

### ML モデルの初期化 {:#initialize_models}

Android アプリでは、モデルで予測を実行する前に、TensorFlow Lite 機械学習モデルをパラメータで初期化する必要があります。

TensorFlow Lite モデルは `*.tflite` ファイルとして保存されます。モデルファイルには予測ロジックが含まれており、通常は予測結果の解釈方法に関する[メタデータ](../../models/convert/metadata)が含まれています。通常、モデルファイルは、コード例のように、開発プロジェクトの `src/main/assets` ディレクトリに保存されます。

- `<project>/src/main/assets/mobilebert_qa.tflite`

注意: サンプルアプリでは、[`download_model.gradle`](https://github.com/tensorflow/examples/blob/master/lite/examples/bert_qa/android/app/download_models.gradle) ファイルを使用して、ビルド時に [mobilebert_qa](https://www.tensorflow.org/lite/inference_with_metadata/task_library/bert_question_answerer) モデルと[パッセージのテキスト](https://storage.googleapis.com/download.tensorflow.org/models/tflite/bert_qa/contents_from_squad.json)をダウンロードします。このアプローチは、本番アプリには必要ありません。

便宜上の観点と、コードを読みやすくするため、この例では、モデルの設定を定義する比較オブジェクトが宣言されています。

アプリでモデルを初期化するには、次の手順を実行します。

1. コンパニオンオブジェクトを作成して、モデルの設定を定義します。サンプルアプリケーションでは、このオブジェクトは [BertQaHelper.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/bert_qa/android/app/src/main/java/org/tensorflow/lite/examples/bertqa/BertQaHelper.kt#L100-L106) にあります。

    ```
    companion object {
        private const val BERT_QA_MODEL = "mobilebert.tflite"
        private const val TAG = "BertQaHelper"
        const val DELEGATE_CPU = 0
        const val DELEGATE_GPU = 1
        const val DELEGATE_NNAPI = 2
    }
    ```

2. Create the settings for the model by building a `BertQaHelper` オブジェクトをビルドしてモデルの設定を作成し、`bertQuestionAnswerer` で TensorFlow Lite オブジェクトを作成します。

    サンプルアプリケーションでは、これは [BertQaHelper.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/bert_qa/android/app/src/main/java/org/tensorflow/lite/examples/bertqa/BertQaHelper.kt#L41-L76) 内の `setupBertQuestionAnswerer()` 関数にあります。

    ```
    class BertQaHelper(
        ...
    ) {
        ...
        init {
            setupBertQuestionAnswerer()
        }

        fun clearBertQuestionAnswerer() {
            bertQuestionAnswerer = null
        }

        private fun setupBertQuestionAnswerer() {
            val baseOptionsBuilder = BaseOptions.builder().setNumThreads(numThreads)
            ...
            val options = BertQuestionAnswererOptions.builder()
                .setBaseOptions(baseOptionsBuilder.build())
                .build()

            try {
                bertQuestionAnswerer =
                    BertQuestionAnswerer.createFromFileAndOptions(context, BERT_QA_MODEL, options)
            } catch (e: IllegalStateException) {
                answererListener
                    ?.onError("Bert Question Answerer failed to initialize. See error logs for details")
                Log.e(TAG, "TFLite failed to load model with error: " + e.message)
            }
        }
        ...
        }
    ```

### ハードウェアアクセラレーションの有効化（オプション）{:#hardware_acceleration}

アプリで TensorFlow Lite モデルを初期化するときには、ハードウェアアクセラレーション機能を使用して、モデルの予測計算を高速化することを検討してください。TensorFlow Lite [デリゲート](https://www.tensorflow.org/lite/performance/delegates)は、グラフィックス処理装置（GPU）またはテンソル処理装置（TPU）といった、モバイルデバイスの専用処理ハードウェアを使用して、機械学習の実行を高速化するソフトウェアモジュールです。

アプリでハードウェアアクセラレーションを有効にするには、次の手順を実行します。

1. アプリケーションが使用するデリゲートを定義する変数を作成します。サンプルアプリケーションでは、この変数は [BertQaHelper.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/bert_qa/android/app/src/main/java/org/tensorflow/lite/examples/bertqa/BertQaHelper.kt#L31) の早い段階にあります。

    ```
    var currentDelegate: Int = 0
    ```

2. デリゲートセレクタを作成します。サンプルアプリケーションでは、デリゲートセレクタは [BertQaHelper.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/bert_qa/android/app/src/main/java/org/tensorflow/lite/examples/bertqa/BertQaHelper.kt#L48-L62) 内の `setupBertQuestionAnswerer` 関数にあります。

    ```
    when (currentDelegate) {
        DELEGATE_CPU -> {
            // Default
        }
        DELEGATE_GPU -> {
            if (CompatibilityList().isDelegateSupportedOnThisDevice) {
                baseOptionsBuilder.useGpu()
            } else {
                answererListener?.onError("GPU is not supported on this device")
            }
        }
        DELEGATE_NNAPI -> {
            baseOptionsBuilder.useNnapi()
        }
    }
    ```

デリゲートを使用して TensorFlow Lite モデルを実行することをお勧めしますが、必須ではありません。TensorFlow Lite でのデリゲートの使用の詳細については、[TensorFlow Lite Delegates](https://www.tensorflow.org/lite/performance/delegates) をご覧ください。

### モデルのデータの準備

Android アプリでは、未加工のテキストなどの既存のデータをモデルで処理できる[テンソル](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/Tensor)データ形式に変換することで、コードは解釈用のモデルにデータを提供します。モデルに渡すテンソルには、モデルのトレーニングに使用されるデータの形式と一致する特定の次元または形状が必要です。この質問応答アプリは、テキストのパッセージと質問の両方の入力として[文字列](https://developer.android.com/reference/java/lang/String.html)を受け入れます。このモデルは、特殊文字と英語以外の単語を認識しません。

モデルにパッセージのテキストデータを提供するには、次の手順を実行します。

1. `LoadDataSetClient` オブジェクトを使用して、パッセージのテキストデータをアプリに読み込みます。サンプルアプリケーションでは、これは [LoadDataSetClient.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/bert_qa/android/app/src/main/java/org/tensorflow/lite/examples/bertqa/dataset/LoadDataSetClient.kt#L25-L45) にあります。

    ```
    fun loadJson(): DataSet? {
        var dataSet: DataSet? = null
        try {
            val inputStream: InputStream = context.assets.open(JSON_DIR)
            val bufferReader = inputStream.bufferedReader()
            val stringJson: String = bufferReader.use { it.readText() }
            val datasetType = object : TypeToken<DataSet>() {}.type
            dataSet = Gson().fromJson(stringJson, datasetType)
        } catch (e: IOException) {
            Log.e(TAG, e.message.toString())
        }
        return dataSet
    }
    ```

2. `DatasetFragment` オブジェクトを使用して、テキストの各パッセージのタイトルを一覧表示し、**TFL の質問と回答**の画面を開始します。サンプルアプリケーションでは、これは [DatasetFragment.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/bert_qa/android/app/src/main/java/org/tensorflow/lite/examples/bertqa/fragments/DatasetFragment.kt) にあります。

    ```
    class DatasetFragment : Fragment() {
        ...
        override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
            super.onViewCreated(view, savedInstanceState)
            val client = LoadDataSetClient(requireActivity())
            client.loadJson()?.let {
                titles = it.getTitles()
            }
            ...
        }
       ...
    }
    ```

3. `DatasetAdapter` オブジェクト内で `onCreateViewHolder` 関数を使用して、テキストの各パッセージのタイトルを表示します。サンプルアプリケーションでは、これは [DatasetAdapter.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/bert_qa/android/app/src/main/java/org/tensorflow/lite/examples/bertqa/fragments/DatasetAdapter.kt) にあります。

    ```
    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
        val binding = ItemDatasetBinding.inflate(
            LayoutInflater.from(parent.context),
            parent,
            false
        )
        return ViewHolder(binding)
    }
    ```

モデルにユーザーの質問を提供するには、次の手順を実行します。

1. `QaAdapter` オブジェクトを使用して、モデルに質問を提供します。サンプルアプリケーションでは、これは [QaAdapter.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/bert_qa/android/app/src/main/java/org/tensorflow/lite/examples/bertqa/fragments/QaAdapter.kt) にあります。

    ```
    class QaAdapter(private val question: List<String>, private val select: (Int) -> Unit) :
      RecyclerView.Adapter<QaAdapter.ViewHolder>() {

      inner class ViewHolder(private val binding: ItemQuestionBinding) :
          RecyclerView.ViewHolder(binding.root) {
          init {
              binding.tvQuestionSuggestion.setOnClickListener {
                  select.invoke(adapterPosition)
              }
          }

          fun bind(question: String) {
              binding.tvQuestionSuggestion.text = question
          }
      }
      ...
    }
    ```

### 予測の実行

Android アプリで [BertQuestionAnswerer](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/task/text/qa/BertQuestionAnswerer) オブジェクトを初期化すると、自然言語テキストの形式で質問をモデルに入力できるようになります。モデルは、テキストのパッセージ内で答えを識別しようとします。

予測を実行するには、次の手順を実行します。

1. モデルを実行し、回答の特定にかかった時間（`inferenceTime`）を測定する `answer` 関数を作成します。サンプルアプリケーションでは、`answer` 関数は [BertQaHelper.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/bert_qa/android/app/src/main/java/org/tensorflow/lite/examples/bertqa/BertQaHelper.kt#L78-L98) にあります。

    ```
    fun answer(contextOfQuestion: String, question: String) {
        if (bertQuestionAnswerer == null) {
            setupBertQuestionAnswerer()
        }

        var inferenceTime = SystemClock.uptimeMillis()

        val answers = bertQuestionAnswerer?.answer(contextOfQuestion, question)
        inferenceTime = SystemClock.uptimeMillis() - inferenceTime
        answererListener?.onResults(answers, inferenceTime)
    }
    ```

2. `answer` から結果をリスナーオブジェクトに渡します。

    ```
    interface AnswererListener {
        fun onError(error: String)
        fun onResults(
            results: List<QaAnswer>?,
            inferenceTime: Long
        )
    }
    ```

### モデル出力の処理

質問を入力すると、モデルはパッセージ内で最大 5 つの可能な回答を提供します。

モデルから結果を取得するには、次の手順を実行します。

1. 出力を処理するリスナーオブジェクトの `onResult` 関数を作成します。サンプルアプリケーションでは、リスナーオブジェクトは [BertQaHelper.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/bert_qa/android/app/src/main/java/org/tensorflow/lite/examples/bertqa/BertQaHelper.kt#L92-L98) にあります。

    ```
    interface AnswererListener {
        fun onError(error: String)
        fun onResults(
            results: List<QaAnswer>?,
            inferenceTime: Long
        )
    }
    ```

2. 結果に基づいてパッセージのセクションを強調表示します。サンプルアプリケーションでは、これは [QaFragment.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/bert_qa/android/app/src/main/java/org/tensorflow/lite/examples/bertqa/fragments/QaFragment.kt#L199-L208) にあります。

    ```
    override fun onResults(results: List<QaAnswer>?, inferenceTime: Long) {
        results?.first()?.let {
            highlightAnswer(it.text)
        }

        fragmentQaBinding.tvInferenceTime.text = String.format(
            requireActivity().getString(R.string.bottom_view_inference_time),
            inferenceTime
        )
    }
    ```

モデルが一連の結果を返すと、アプリケーションはユーザーに結果を提示するか、追加のロジックを実行することで、これらの予測に基づいて行動できます。

## 次のステップ

- [TensorFlow Lite Model Maker を使用した質問回答](https://www.tensorflow.org/lite/models/modify/model_maker/question_answer)チュートリアルを使用して、モデルを最初からトレーニングして実装します。
- その他の [TensorFlow 用のテキスト処理ツール](https://www.tensorflow.org/text)を考察します。
- [TensorFlow Hub](https://tfhub.dev/google/collections/bert/1) で他の BERT モデルをダウンロードします。
- [例](../../examples)で TensorFlow Lite のさまざまな使用方法を考察します。
