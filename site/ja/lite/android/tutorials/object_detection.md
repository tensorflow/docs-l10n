# Android での物体検出

このチュートリアルでは、TensorFlow Lite を使用して Android アプリを構築し、デバイスのカメラによって取り込まれたフレームで物体を継続的に検出する方法について説明します。このアプリケーションは、物理 Android デバイス向けです。既存のプロジェクトを更新する場合は、コードサンプルを参考にして、[プロジェクトの修正](#add_dependencies)手順に進むことができます。

![Object detection animated demo](https://storage.googleapis.com/download.tensorflow.org/tflite/examples/obj_detection_cat.gif){: .attempt-right width="250px"}

## 物体検出の概要

*物体検出*は、画像内の物体の複数のクラスの存在または位置を特定する機械学習タスクです。物体検知モデルは、既知の物体のセットを含むデータセットでトレーニングされます。

トレーニングされたモデルは、画像フレームを入力として受け取り、認識するようにトレーニングされた既知のクラスのセットから画像の項目を分類します。物体検出では、画像フレームごとに、検出された物体のリスト、各物体のバウンディングボックスの位置、および物体分類の正確性に対する信頼度を示すスコアが出力されます。

## モデルとデータセット

このチュートリアルでは、[COCO データセット](http://cocodataset.org/) を使用してトレーニングされたモデルが使用されます。COCO は、330K の画像、150 万の物体インスタンス、80 の物体カテゴリを含む大規模な物体検出データセットです。

次のトレーニング済みモデルのいずれかを使用します。

- [EfficientDet-Lite0](https://tfhub.dev/tensorflow/lite-model/efficientdet/lite0/detection/metadata/1) *[推奨]* - BiFPN 特徴抽出、共有ボックス予測、焦点損失を備えた、軽量物体検出モデル。COCO 2017 検証データセットの mAP (mean Average Precision: 平均適合率) は 25.69% です。

- [EfficientDet-Lite1](https://tfhub.dev/tensorflow/lite-model/efficientdet/lite1/detection/metadata/1) - 中程度のサイズの EfficientDet 物体検出モデル。COCO 2017 検証データセットの mAP は 30.55% です。

- [EfficientDet-Lite2](https://tfhub.dev/tensorflow/lite-model/efficientdet/lite2/detection/metadata/1) - 大きい EfficientDet 物体検出モデル。COCO 2017 検証データセットの mAP は 33.97% です。

- [MobileNetV1-SSD](https://tfhub.dev/tensorflow/lite-model/ssd_mobilenet_v1/1/metadata/2) - 物体検出で TensorFlow Lite と連携するように最適化された非常に軽量のモデル。COCO 2017 の検証データセットの mAP は 21% です。

このチュートリアルでは、*EfficientDet-Lite0* モデルが、サイズと精度のバランスにおいて最適です。

モデルのダウンロード、抽出、アセットフォルダへの配置は、ビルド時に実行される、`download.gradle` ファイルによって自動的に管理されています。手動で TFLite モデルをプロジェクトにダウンロードする必要はありません。

## セットアップと実行の例

物体検知アプリをセットアップするには、[GitHub](https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection/android) からサンプルをダウンロードし、[Android Studio](https://developer.android.com/studio/) を使用して実行します。このチュートリアルの次のセクションでは、コードサンプルの関連するセクションを考察し、独自の Android アプリに応用できるようにします。

### システム要件

- <a>Android Studio</a> バージョン 2021.1.1 (Bumblebee) 以上
- Android SDK バージョン 31 以上
- OS バージョン SDK 24 (Android 7.0 - Nougat) 以上が搭載された Android デバイス (開発者モードが有効であること)

注意: この例ではカメラを使用するため、物理 Android デバイスで実行する必要があります。

### サンプルコードの取得

サンプルコードのローカルコピーを作成します。このコードを使用して、Android Studio でプロジェクトを作成し、サンプルアプリケーションを実行します。

サンプルコードを複製してセットアップするには、次の手順を実行します。

1. git リポジトリを複製します。
    <pre class="devsite-click-to-copy">    git clone https://github.com/tensorflow/examples.git
        </pre>
2. 任意で、sparse checkout を使用するように git インスタンスを構成します。これで、物体検出サンプルアプリのファイルのみを取得できます。
    <pre class="devsite-click-to-copy">    cd examples
        git sparse-checkout init --cone
        git sparse-checkout set lite/examples/object_detection/android
        </pre>

### プロジェクトのインポートと実行

ダウンロードしたサンプルコードからプロジェクトを作成し、プロジェクトをビルドして、実行します。

サンプルコードプロジェクトをインポートしてビルドするには、次の手順を実行します。

1. [Android Studio](https://developer.android.com/studio) を起動します。
2. Android Studio で、**[File] &gt; [New] &gt; [Import Project]** を選択します。
3. `build.gradle` ファイルがあるサンプルコードディレクトリ (`.../examples/lite/examples/object_detection/android/build.gradle`) に移動し、ディレクトリを選択します。
4. Android Studio で Gradle Sync が要求される場合は、[OK] をクリックします。
5. Android デバイスがコンピュータに接続され、開発者モードが有効であることを確認します。緑色の `Run` 矢印をクリックします。

正しいディレクトリを選択すると、Android Studio で新しいプロジェクトが作成、ビルドされます。Android Studio を他のプロジェクトでも使用している場合、コンピューターの速度によっては、この処理に数分かかる場合があります。ビルドが完了すると、Android Studio の <strong>[Build Output]</strong> ステータスパネルに <code>BUILD SUCCESSFUL</code> メッセージが表示されます。

注意: サンプルコードは Android Studio 4.2.2 で作成されますが、それよりも前のバージョンの Studio でも動作します。前のバージョンの Android Studio を使用している場合は、Studio をアップグレードせずに、Android プラグインのバージョン番号を調整し、ビルドを完了させることができます。

**任意:** Android プラグインバージョンを更新してビルドエラーを修正するには、次の手順を実行します。

1. プロジェクトディレクトリで build.gradle ファイルを開きます。

2. 次のように、Android ツールバージョンを変更します。

    ```
    // from: classpath
    'com.android.tools.build:gradle:4.2.2'
    // to: classpath
    'com.android.tools.build:gradle:4.1.2'
    ```

3. **[File] &gt; [Sync Project with Gradle Files]** を選択して、プロジェクトを同期します。

プロジェクトを実行するには、次の手順を実行します。

1. Android Studio で **[Run] &gt; [Run…]** を選択して、プロジェクトを実行します。
2. 接続されたカメラを搭載した Android デバイスを選択し、アプリをテストします。

次のセクションでは、このサンプルアプリを参考にして、この機能を独自のアプリに追加するために、既存のプロジェクトに行う必要がある修正について説明します。

## プロジェクト依存関係の追加 {:#add_dependencies}

独自のアプリケーションで、TensorFlow Lite 機械学習モデルを実行するための特定のプロジェクト依存関係を追加する必要があります。また、画像などのデータを、使用中のモデルで処理できるテンソルデータに変換するユーティリティ関数にアクセスする必要があります。

サンプルアプリでは、TensorFlow Lite [Task library for vision](../../inference_with_metadata/task_library/overview#supported_tasks) を使用して、物体検出機会学習モデルの実行を可能にします。次の手順では、必要な依存関係を Android アプリプロジェクトに追加する方法について説明します。

次の手順では、必要なプロジェクトおよびモデル依存関係を Android アプリプロジェクトに追加する方法について説明します。

モジュール依存関係を追加するには、次の手順を実行します。

1. TensorFlow Lite を使用するモジュールで、モジュールの `build.gradle` ファイルを更新し、次の依存関係を追加します。サンプルコードでは、このファイルは次の場所にあります。`...examples/lite/examples/object_detection/android/app/build.gradle` ([コードリファレンス](https://github.com/tensorflow/examples/blob/master/lite/examples/object_detection/android/app/build.gradle))

    ```
    dependencies {
      ...
      implementation 'org.tensorflow:tensorflow-lite-task-vision:0.4.0'
      // Import the GPU delegate plugin Library for GPU inference
      implementation 'org.tensorflow:tensorflow-lite-gpu-delegate-plugin:0.4.0'
      implementation 'org.tensorflow:tensorflow-lite-gpu:2.9.0'
    }
    ```

    プロジェクトには、Vision タスクライブラリ (`tensorflow-lite-task-vision`) を含める必要があります。グラフィックス処理装置 (GPU) ライブラリ (`tensorflow-lite-gpu-delegate-plugin`) は、GPU でアプリを実行するためのインフラストラクチャを提供します。デリゲート (`tensorflow-lite-gpu`) は、互換性リストを提供します。

2. Android Studio で、**[File] &gt; [Sync Project with Gradle Files]** を選択して、プロジェクト依存関係を同期します。

## ML モデルの初期化

Android アプリで、モデルで予測を実行する前に、TensorFlow Lite 機械学習モデルを初期化する必要があります。これらの初期化パラメータは物体検出モデル全体で同じであり、予測の最低精度しきい値といった設定を含めることができます。

TensorFlow Lite モデルには、モデルコードを含む `.tflite` ファイルがあります。また、多くの場合、モデルで予測されるクラスの名前を含むラベルファイルもあります。物体検出の場合、クラスは、人、犬、猫、車などの物体です。

この例では、`download_models.gradle` で指定された複数のモデルをダウンロードします。`ObjectDetectorHelper` クラスは、モデルのセレクタを提供します。

```
val modelName =
  when (currentModel) {
    MODEL_MOBILENETV1 -> "mobilenetv1.tflite"
    MODEL_EFFICIENTDETV0 -> "efficientdet-lite0.tflite"
    MODEL_EFFICIENTDETV1 -> "efficientdet-lite1.tflite"
    MODEL_EFFICIENTDETV2 -> "efficientdet-lite2.tflite"
    else -> "mobilenetv1.tflite"
  }
```

要点: モデルは、開発プロジェクトの `src/main/assets` ディレクトリに格納されます。モデルファイル名を指定すると、TensorFlow Lite タスクライブラリによって自動的にこのディレクトリがチェックされます。

アプリでモデルを初期化するには、次の手順を実行します。

1. `.tflite` モデルファイルを開発プロジェクトの `src/main/assets` ディレクトリに追加します。例: [EfficientDet-Lite0](https://tfhub.dev/tensorflow/lite-model/efficientdet/lite0/detection/metadata/1)。

2. モデルファイル名の静的変数を設定します。サンプルアプリでは、`modelName` 変数を `MODEL_EFFICIENTDETV0` に設定して、EfficientDet-Lite0 検出モデルを使用します。

3. 予測しきい値、結果セットサイズ、ハードウェアアクセラレーションデリゲート (任意) などのモデルのオプションを設定します。

    ```
    val optionsBuilder =
      ObjectDetector.ObjectDetectorOptions.builder()
        .setScoreThreshold(threshold)
        .setMaxResults(maxResults)
    ```

4. このオブジェクトの設定を使用して、モデルを含む TensorFlow Lite [`ObjectDetector`](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/task/vision/detector/ObjectDetector#createFromFile(Context,%20java.lang.String)) オブジェクトを作成します。

    ```
    objectDetector =
      ObjectDetector.createFromFileAndOptions(
        context, modelName, optionsBuilder.build())
    ```

`setupObjectDetector` では、次のモデルパラメータが設定されます。

- 検出しきい値
- 検出結果の最大数
- (`BaseOptions.builder().setNumThreads(numThreads)`) を使用する処理しきい値数
- 実際のモデル (`modelName`)
- ObjectDetector オブジェクト (`objectDetector`)

### ハードウェアアクセラレータの構成

アプリケーションで TensorFlow Lite モデルを初期化するときには、ハードウェアアクセラレーション機能を使用して、モデルの予測計算を高速化できます。

TensorFlow Lite *デリゲート*は、グラフィックス処理装置 (GPU)、テンソル処理装置 (TPU)、デジタルシグナルプロセッサ (DSP) といったモバイルデバイスの特殊な処理ハードウェアを使用して、機械学習モデルの実行を高速化するソフトウェアモジュールです。TensorFlow Lite モデルの実行では、デリゲートを使用することをお勧めします。ただし必須ではありません。

The object detector is initialized using the current settings on the thread that is using it. You can use CPU and [NNAPI](../../android/delegates/nnapi) delegates with detectors that are created on the main thread and used on a background thread, but the thread that initialized the detector must use the GPU delegate.

デリゲートは、`ObjectDetectionHelper.setupObjectDetector()` 関数内で設定されます。

```
when (currentDelegate) {
    DELEGATE_CPU -> {
        // Default
    }
    DELEGATE_GPU -> {
        if (CompatibilityList().isDelegateSupportedOnThisDevice) {
            baseOptionsBuilder.useGpu()
        } else {
            objectDetectorListener?.onError("GPU is not supported on this device")
        }
    }
    DELEGATE_NNAPI -> {
        baseOptionsBuilder.useNnapi()
    }
}
```

TensorFlow Lite でのハードウェアアクセラレーションデリゲートの使用の詳細については、[TensorFlow Lite デリゲート](../../performance/delegates)を参照してください。

## モデルのデータの準備

Android アプリでは、コードによって、画像フレームなどの既存のデータが、モデルで処理できる<a>テンソル</a>データ形式に変換されて、モデルに入力され、解釈されます。 テンソルのデータには、モデルをトレーニングするために使用されたデータ形式と一致する固有の次元または形状が必要です。

このコードサンプルで使用される [EfficientDet-Lite0](https://tfhub.dev/tensorflow/lite-model/efficientdet/lite0/detection/metadata/1) モデルでは、320 x 320 の寸法で、ピクセルごとに 3 つのチャネル (赤、青、緑) がある画像を表すテンソルを入力できます。テンソルの各値は、0 ～ 255 の範囲のシングルバイトです。このため、新しい画像で予測を実行するには、アプリで、画像データを、そのサイズと形状のテンソルデータオブジェクトに変換する必要があります。データ変換は、TensorFlow Lite Task Library Vision API によって処理されます。

アプリでは、[`ImageAnalysis`](https://developer.android.com/training/camerax/analyze) オブジェクトを使用して、カメラから画像を取得できます。このオブジェクトでは、カメラからのビットマップを使用して、`detectObject` 関数を呼び出します。データは、`ImageProcessor` によって、自動的にサイズが変更され、回転されるため、モデルの画像データ要件が満たされます。その後、画像は [`TensorImage`](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/support/image/TensorImage) オブジェクトに変換されます。

ML モデルで処理されるカメラサブシステムからデータを準備するには、次の手順を実行します。

1. `ImageAnalysis` オブジェクトをビルドし、必要な形式で画像を抽出します。

    ```
    imageAnalyzer =
        ImageAnalysis.Builder()
            .setTargetAspectRatio(AspectRatio.RATIO_4_3)
            .setTargetRotation(fragmentCameraBinding.viewFinder.display.rotation)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build()
            ...
    ```

2. 分析器をカメラサブシステムに接続し、カメラから受信したデータを格納するビットマップバッファを作成します。

    ```
    .also {
      it.setAnalyzer(cameraExecutor) {
        image -> if (!::bitmapBuffer.isInitialized)
        { bitmapBuffer = Bitmap.createBitmap( image.width, image.height,
        Bitmap.Config.ARGB_8888 ) } detectObjects(image)
        }
      }
    ```

3. モデルで必要な特定の画像データを抽出し、画像回転情報を渡します。

    ```
    private fun detectObjects(image: ImageProxy) {
      //Copy out RGB bits to the shared bitmap buffer
      image.use {bitmapBuffer.copyPixelsFromBuffer(image.planes[0].buffer) }
        val imageRotation = image.imageInfo.rotationDegrees
        objectDetectorHelper.detect(bitmapBuffer, imageRotation)
      }
    ```

4. サンプルアプリの `ObjectDetectorHelper.detect()` メソッドで示すように、最終データ変換を完了し、画像データを `TensorImage` オブジェクトに追加します。

    ```
    val imageProcessor = ImageProcessor.Builder().add(Rot90Op(-imageRotation / 90)).build()
    // Preprocess the image and convert it into a TensorImage for detection.
    val tensorImage = imageProcessor.process(TensorImage.fromBitmap(image))
    ```

注記: Android カメラシステムから画像情報を抽出するときには、必ず画像を RGB 形式で取得します。この形式は、モデル分析用に画像を準備するために使用する TensorFlow Lite <a>ImageProcessor</a> クラスで必要です。RGB 形式の画像には、透明なデータが無視されるアルファチャネルが含まれています。

## 予測の実行

Android アプリでは、正しい形式の画像データを使用して、TensorImage オブジェクトを作成すると、そのデータに対してモデルを実行し、予測または*推論*を生成できます。

サンプルアプリの `fragments/CameraFragment.kt` クラスでは、アプリがカメラに接続されると、`bindCameraUseCases` 関数の `imageAnalyzer` オブジェクトによって、自動的にデータがモデルに渡されて、予測が実行されます。

アプリは、`cameraProvider.bindToLifecycle()` メソッドを使用して、カメラセレクタ、プレビューウィンドウ、ML モデル処理を扱います。`ObjectDetectorHelper.kt` クラスは、画像データをモデルに渡します。モデルを実行し、画像データから予測を生成するには、次の手順を実行します。

- 画像データを予測関数に渡して、予測を実行します。

    ```
    val results = objectDetector?.detect(tensorImage)
    ```

TensorFlow Lite Interpreter オブジェクトはこのデータを受け取り、それをモデルに対して実行し、予測のリストを生成します。モデルによるデータ処理を連続的に実行するために、`runForMultipleInputsOutputs()` メソッドを使用して、Interpreter オブジェクトが作成されず、予測実行のたびにシステムによって削除されるようにします。

## モデル出力の処理

Android アプリで、物体検出モデルに対して画像データを実行した後、追加のビジネスロジックの実行、ユーザーへの結果の表示、または他のアクションの実行によって、アプリコードで処理する必要がある予測のリストが生成されます。

特定の TensorFlow Lite モデルの出力は、生成される予測の数 (1 つまたは複数) と各予測の記述情報という点で異なります。物体検出モデルの場合、一般的に、物体が画像で検出される場所を示すバウンディングボックスのデータが予測に含まれます。このサンプルコードでは、結果が、物体検出プロセスで DetectorListener として機能する、`CameraFragment.kt` の `onResults` 関数に渡されます。

```
interface DetectorListener {
  fun onError(error: String)
  fun onResults(
    results: MutableList<Detection>?,
    inferenceTime: Long,
    imageHeight: Int,
    imageWidth: Int
  )
}
```

この例で使用されるモデルの場合、各予測には、物体のバウンディングボックスの位置、物体のラベル、および予測の信頼度を表す 0 ～ 1 の浮動小数点数の予測スコア (1 が最高の信頼度評価) が含まれます。一般的に、予測のスコアが 50% (0.5) 未満の場合、決定的ではないと見なされます。ただし、低い値の予測結果を処理する方法は、開発者やアプリケーションのニーズによって異なります。

モデル予測結果を処理するには、次の手順に従います。

1. リスナーパターンを使用して、結果をアプリコードまたはユーザーインターフェイスオブジェクトに渡します。サンプルアプリは、このパターンを使用して、`ObjectDetectorHelper` オブジェクトの検出結果を `CameraFragment` オブジェクトに渡します。

    ```
    objectDetectorListener.onResults(
    // instance of CameraFragment
        results,
        inferenceTime,
        tensorImage.height,
        tensorImage.width)
    ```

2. ユーザーに予測を表示するといったように、結果に対して処理を実行します。例では、CameraPreview オブジェクトにオーバーレイが描画され、結果を示します。

    ```
    override fun onResults(
      results: MutableList<Detection>?,
      inferenceTime: Long,
      imageHeight: Int,
      imageWidth: Int
    ) {
        activity?.runOnUiThread {
            fragmentCameraBinding.bottomSheetLayout.inferenceTimeVal.text =
                String.format("%d ms", inferenceTime)

            // Pass necessary information to OverlayView for drawing on the canvas
            fragmentCameraBinding.overlay.setResults(
                results ?: LinkedList<Detection>(),
                imageHeight,
                imageWidth
            )

            // Force a redraw
            fragmentCameraBinding.overlay.invalidate()
        }
    }
    ```

予測結果がモデルによって返されると、結果がユーザーに表示されるか、追加のロジックが実行され、アプリケーションでその予測に対して処理が実行されます。サンプルコードの場合、特定された物体の周りにバウンディングボックスが描画され、画面にクラス名が表示されます。

## 次のステップ

- [例](../../examples)を使って、TensorFlow Lite のさまざまな使用方法を考察します。
- [モデル](../../models)セクションで、TensorFlow Lite の機械学習モデルの使用方法について詳細に説明します。
- [TensorFlow Lite 開発者ガイド](../../guide)で、モデルアプリケーションでの機械学習の実装について詳細に説明します。
