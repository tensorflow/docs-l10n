# Android 版クイックスタート

このページでは、TensorFlow Lite を使用して Android アプリをビルドし、ライブカメラフィードを分析してオブジェクトを識別する方法を説明します。この機械学習のユースケースは、*オブジェクト検出*と呼ばれます。サンプルアプリでは、[Google Play サービス](./play_services)を介して TensorFlow Lite の[ビジョン用 Task Library](../inference_with_metadata/task_library/overview#supported_tasks) を使用し、オブジェクト検出機械学習モデルを実行できるようにします。これは、TensorFlow Lite で ML アプリケーションを構築するための推奨されるアプローチです。

<aside class="note"><b>利用規約:</b> Google Play サービス API で TensorFlow Lite にアクセスまたは使用することにより、<a href="./play_services#tos">利用規約</a>に同意したことになります。API にアクセスする前に、該当するすべての条件とポリシーを読み、理解してください。</aside>

![Object detection animated demo](https://storage.googleapis.com/download.tensorflow.org/tflite/examples/obj_detection_cat.gif){: .attempt-right width="250px"}

## サンプルのセットアップと実行

この演習の最初の部分では、[サンプルコード](https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection/android_play_services)を GitHub からダウンロードし、[Android Studio](https://developer.android.com/studio/) を使用して実行します。このドキュメントの次のセクションでは、コードサンプルの関連セクションを考察し、独自の Android アプリに応用できるようにします。これらのツールの次のバージョンがインストールされている必要があります。

- Android Studio 4.2 以上
- Android SDK バージョン 21 以上

注意: この例ではカメラを使用するため、物理 Android デバイスで実行する必要があります。

### サンプルコードの取得

サンプルコードのローカルコピーを作成して、ビルドして実行できるようにします。

サンプルコードを複製してセットアップするには、次の手順を実行します。

1. git リポジトリを複製します。
    <pre class="devsite-click-to-copy">    git clone https://github.com/tensorflow/examples.git
        </pre>
2. sparse checkout を使用するように git インスタンスを構成します。これで、物体検出サンプルアプリのファイルのみを取得できます。
    <pre class="devsite-click-to-copy">    cd examples
        git sparse-checkout init --cone
        git sparse-checkout set lite/examples/object_detection/android_play_services
        </pre>

### プロジェクトのインポートと実行

Android Studio を使用して、ダウンロードしたサンプルコードからプロジェクトを作成し、プロジェクトをビルドして実行します。

サンプルコードプロジェクトをインポートしてビルドするには、次の手順を実行します。

1. [Android Studio](https://developer.android.com/studio) を起動します。
2. Android Studio の **[Welcome]** ページで **[Import Project]** を選択するか、**[File] &gt; [New] &gt; [Import Project]** を選択します。
3. build.gradle ファイル（`...examples/lite/examples/object_detection/android_play_services/build.gradle`）があるサンプルコードディレクトリに移動し、ディレクトリを選択します。

このディレクトリを選択すると、Android Studio によって新しいプロジェクトが作成され、ビルドされます。ビルドが完了すると、Android Studio は<strong>ビルド出力</strong>ステータスパネルに <code>BUILD SUCCESSFUL</code> メッセージを表示します。

プロジェクトを実行するには、次の手順を実行します。

1. Android Studio で **Run &gt; Run…**、**MainActivity** を選択して、プロジェクトを実行します。
2. 接続されたカメラを搭載した Android デバイスを選択し、アプリをテストします。

## サンプルアプリの仕組み

このサンプルアプリでは、Android デバイスのカメラのライブビデオストリームのオブジェクトに、[mobilenetv1.tflite](https://tfhub.dev/tensorflow/lite-model/ssd_mobilenet_v1/1/metadata/2?lite-format=tflite) など、TensorFlow Lite 形式の事前にトレーニングされたオブジェクト検出モデルを使用します。この機能のコードは、主に次のファイルにあります。

- [ObjectDetectorHelper.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/object_detection/android_play_services/app/src/main/java/org/tensorflow/lite/examples/objectdetection/ObjectDetectorHelper.kt) - ランタイム環境を初期化し、ハードウェアアクセラレーションを有効にして、オブジェクト検出 ML モデルを実行します。
- [CameraFragment.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/object_detection/android_play_services/app/src/main/java/org/tensorflow/lite/examples/objectdetection/fragments/CameraFragment.kt) - カメラ画像データストリームをビルドして、モデルのデータを準備し、オブジェクト検出結果を表示します。

注意: このサンプルアプリでは、一般的な機械学習操作を実行するための使いやすいタスク固有の API を提供する TensorFlow Lite [Task Library](../inference_with_metadata/task_library/overview#supported_tasks) を使用します。より具体的なニーズとカスタマイズされた ML 関数を持つアプリの場合は、[Interpreter API](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/InterpreterApi) の使用を検討してください。

次のセクションでは、これらのコードファイルの主要なコンポーネントを示します。これにより、Android アプリを変更してこの機能を追加できます。

## アプリのビルド {:#build_app}

以下のセクションでは、独自の Android アプリをビルドし、サンプルアプリに示されているモデルを実行するための主要な手順について説明します。これらの手順では、参照ポイントとして前に示したサンプルアプリを使用します。

注意: これらの手順に沿って独自のアプリを作成するには、Android Studio を使用して[基本的な Android プロジェクト](https://developer.android.com/studio/projects/create-project)を作成します。

### プロジェクト依存関係の追加 {:#add_dependencies}

基本的な Android アプリで、TensorFlow Lite 機械学習モデルを実行し、ML データユーティリティ関数にアクセスするためのプロジェクトの依存関係を追加します。これらのユーティリティ関数は、画像などのデータをモデルで処理できるテンソルデータ形式に変換します。

このサンプルアプリでは、[Google Play サービス](./play_services)の　TensorFlow Lite [Task Library（ビジョン用）](../inference_with_metadata/task_library/overview#supported_tasks)を使用して、オブジェクト検出機械学習モデルを実行できるようにします。次の手順では、必要なライブラリの依存関係を独自の Android アプリプロジェクトに追加する方法について説明します。

モジュール依存関係を追加するには、次の手順を実行します。

1. TensorFlow Lite を使用する Android アプリモジュールで、モジュールの `build.gradle` ファイルを更新して、次の依存関係を追加します。サンプルコードでは、このファイルは次の場所にあります: `...examples/lite/examples/object_detection/android_play_services/app/build.gradle`
    ```
    ...
    dependencies {
    ...
        // Tensorflow Lite dependencies
        implementation 'org.tensorflow:tensorflow-lite-task-vision-play-services:0.4.2'
        implementation 'com.google.android.gms:play-services-tflite-gpu:16.0.0'
    ...
    }
    ```
2. Android Studio で、**[File] &gt; [Sync Project with Gradle Files]** を選択して、プロジェクト依存関係を同期します。

### Google Play サービスの初期化

[Google Play サービス](./play_services)を使用して TensorFlow Lite モデルを実行する場合、サービスを使用する前に初期化する必要があります。 GPU アクセラレーションなどのハードウェアアクセラレーションサポートをサービスで使用する場合は、この初期化の一部としてそのサポートも有効にします。

TensorFlow Lite を Google Play サービスで初期化するには、以下の手順を実行します。

1. `TfLiteInitializationOptions` オブジェクトを作成し、変更して GPU サポートを有効にします。

    ```
    val options = TfLiteInitializationOptions.builder()
        .setEnableGpuDelegateSupport(true)
        .build()
    ```

2. `TfLiteVision.initialize()` メソッドを使用して Play サービスランタイムの使用を有効にし、リスナーを設定して正常にロードされたことを確認します。

    ```
    TfLiteVision.initialize(context, options).addOnSuccessListener {
        objectDetectorListener.onInitialized()
    }.addOnFailureListener {
        // Called if the GPU Delegate is not supported on the device
        TfLiteVision.initialize(context).addOnSuccessListener {
            objectDetectorListener.onInitialized()
        }.addOnFailureListener{
            objectDetectorListener.onError("TfLiteVision failed to initialize: "
                    + it.message)
        }
    }
    ```

### ML モデルインタープリタの初期化

モデルファイルを読み込んでモデルパラメーターを設定して、TensorFlow Lite 機械学習モデルインタープリタを初期化します。 TensorFlow Lite モデルには、モデルコードを含む `.tflite` ファイルが含まれています。モデルは、開発プロジェクトの `src/main/assets` ディレクトリに保存する必要があります。次に例を示します。

```
.../src/main/assets/mobilenetv1.tflite`
```

ヒント: ファイルパスを指定しない場合、タスクライブラリインタープリタコードは `src/main/assets` ディレクトリ内のモデルを自動的に検索します。

モデルを初期化するには、以下の手順を実行します。

1. <a>ssd_mobilenet_v1</a> などの開発プロジェクトの `src/main/assets` ディレクトリに <code>.tflite</code> モデルファイルを追加します。
2. `modelName` 変数を設定して、ML モデルのファイル名を指定します。
    ```
    val modelName = "mobilenetv1.tflite"
    ```
3. 予測しきい値や結果セットのサイズなど、モデルのオプションを設定します。
    ```
    val optionsBuilder =
        ObjectDetector.ObjectDetectorOptions.builder()
            .setScoreThreshold(threshold)
            .setMaxResults(maxResults)
    ```
4. オプションを使用して GPU アクセラレーションを有効にし、アクセラレーションがデバイスでサポートされていない場合はコードが正常に失敗するようにします。
    ```
    try {
        optionsBuilder.useGpu()
    } catch(e: Exception) {
        objectDetectorListener.onError("GPU is not supported on this device")
    }

    ```
5. このオブジェクトの設定を使用して、モデルを含む TensorFlow Lite [`ObjectDetector`](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/task/vision/detector/ObjectDetector#createFromFile(Context,%20java.lang.String)) オブジェクトを作成します。
    ```
    objectDetector =
        ObjectDetector.createFromFileAndOptions(
            context, modelName, optionsBuilder.build())
    ```

TensorFlow Lite でのハードウェアアクセラレーションデリゲートの使用の詳細については、[TensorFlow Lite デリゲート](../performance/delegates)を参照してください。

### モデルのデータの準備

モデルで処理できるように、画像などの既存のデータを[テンソル](../api_docs/java/org/tensorflow/lite/Tensor)データ形式に変換して、モデルによる解釈用のデータを準備します。テンソル内のデータには、モデルのトレーニングに使用されるデータの形式と一致する特定の次元または形状が必要です。使用するモデルによっては、モデルが期待するものに適合するようにデータを変換する必要がある場合があります。サンプルアプリは [`ImageAnalysis`](https://developer.android.com/reference/androidx/camera/core/ImageAnalysis) オブジェクトを使用して、カメラサブシステムから画像フレームを抽出します。

モデルで処理するデータを準備するには、以下の手順を実行します。

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
            it.setAnalyzer(cameraExecutor) { image ->
                if (!::bitmapBuffer.isInitialized) {
                    bitmapBuffer = Bitmap.createBitmap(
                        image.width,
                        image.height,
                        Bitmap.Config.ARGB_8888
                    )
                }
                detectObjects(image)
            }
        }
    ```
3. モデルで必要な特定の画像データを抽出し、画像回転情報を渡します。
    ```
    private fun detectObjects(image: ImageProxy) {
        // Copy out RGB bits to the shared bitmap buffer
        image.use { bitmapBuffer.copyPixelsFromBuffer(image.planes[0].buffer) }
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

### 予測の実行

正しい形式の画像データを使用して [TensorImage](../api_docs/java/org/tensorflow/lite/support/image/TensorImage) オブジェクトを作成したら、そのデータに対してモデルを実行して、予測または*推論*を生成できます。サンプルアプリでは、このコードは `ObjectDetectorHelper.detect()` メソッドに含まれています。

モデルを実行し、画像データから予測を生成するには、以下の手順を実行します。

- 画像データを予測関数に渡して、予測を実行します。
    ```
    val results = objectDetector?.detect(tensorImage)
    ```

### モデル出力の処理

オブジェクト検出モデルに対して画像データを実行すると、追加のビジネスロジックを実行し、ユーザーに結果を表示し、その他のアクションを実行することにより、アプリコードで処理する必要がある予測結果のリストが生成されます。サンプルアプリのオブジェクト検出モデルは、検出されたオブジェクトの予測と境界ボックスのリストを生成します。サンプルアプリでは、予測結果がリスナーオブジェクトに渡されてさらに処理され、ユーザーに表示されます。

モデル予測結果を処理するには、次の手順に従います。

1. リスナーパターンを使用して、結果をアプリコードまたはユーザーインターフェイスオブジェクトに渡します。サンプルアプリは、このパターンを使用して、`ObjectDetectorHelper` オブジェクトの検出結果を `CameraFragment` オブジェクトに渡します。
    ```
    objectDetectorListener.onResults( // instance of CameraFragment
        results,
        inferenceTime,
        tensorImage.height,
        tensorImage.width)
    ```
2. ユーザーに予測を表示するなど、結果に対して処理を実行します。サンプルアプリは `CameraPreview` オブジェクトにオーバーレイを描画して結果を表示します。
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

## Next steps

- [Task Library API](../inference_with_metadata/task_library/overview#supported_tasks) の詳細
- [Interpreter API](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/InterpreterApi) の詳細
- [例](../examples)を使って、TensorFlow Lite のさまざまな使用方法を考察します。
- [モデル](../models)セクションで、TensorFlow Lite の機械学習モデルの使用およびビルド方法について詳細に説明します。
- [TensorFlow Lite 開発者ガイド](../guide)で、モデルアプリケーションでの機械学習の実装について詳細に説明します。
