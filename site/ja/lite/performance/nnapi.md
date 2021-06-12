# TensorFlow Lite NNAPI デレゲート

[Android Neural Networks API (NNAPI)](https://developer.android.com/ndk/guides/neuralnetworks) は、Android 8.1 (API レベル 27) 以降を実行しているすべての Android デバイスで使用でき、以下を含むサポートされているハードウェアアクセラレータを備えた Android デバイス上で TensorFlow Lite モデルのアクセラレーションを提供します。

- GPU (グラフィックス プロセッシング ユニット)
- DSP (デジタルシグナル プロセッサ)
- NPU (ニューラルプロセッシングユニット)

パフォーマンスは、デバイスで使用可能な特定のハードウェアに応じて異なります。

このページでは、Java および Kotlin の TensorFlow Lite インタープリタ で NNAPI デリゲートを使用する方法について説明します。Android C API については、[Android Native Developer Kit のドキュメント](https://developer.android.com/ndk/guides/neuralnetworks)をご覧ください。

## NNAPI デリゲートを独自のモデルで試す

### Gradle インポート

NNAPI デリゲートは、TensorFlow Lite Android インタープリタ、リリース 1.14.0 以降の一部です。次のコードをモジュールグラドルファイルに追加することで、プロジェクトにインポートできます。

```groovy
dependencies {
   implementation 'org.tensorflow:tensorflow-lite:2.0.0'
}
```

### NNAPI デリゲートの初期化

TensorFlow Lite インタープリタを初期化する前に、NNAPI デリゲートを初期化するコードを追加します。

注意: NNAPI は API レベル 27 (Android Oreo MR1) からサポートされていますが、演算のサポートは API レベル 28 (Android Pie) 以降で大幅に改善されました。そのため、ほとんどのシナリオでは、Android Pie 以上の NNAPI デリゲートを使用することをお勧めします。

```java
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.nnapi.NnApiDelegate;

Interpreter.Options options = (new Interpreter.Options());
NnApiDelegate nnApiDelegate = null;
// Initialize interpreter with NNAPI delegate for Android Pie or above
if(Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
    nnApiDelegate = new NnApiDelegate();
    options.addDelegate(nnApiDelegate);
}

// Initialize TFLite interpreter
try {
    tfLite = new Interpreter(loadModelFile(assetManager, modelFilename), options);
} catch (Exception e) {
    throw new RuntimeException(e);
}

// Run inference
// ...

// Unload delegate
tfLite.close();
if(null != nnApiDelegate) {
    nnApiDelegate.close();
}
```

## ベストプラクティス

### デプロイする前にパフォーマンスをテストする

実行時のパフォーマンスは、モデルのアーキテクチャ、サイズ、演算、ハードウェアの可用性、および実行時のハードウェアの使用状況によって大幅に異なる可能性があります。たとえば、アプリがレンダリングに GPU を頻繁に使用する場合、NNAPI アクセラレーションはリソースの競合のためにパフォーマンスを改善しない可能性があります。推論時間を測定するには、デバッグロガーを使用して簡単なパフォーマンステストを実行することをお勧めします。本番環境で NNAPI を有効にする前に、ユーザーベースを代表する異なるチップセット (同じメーカーのメーカーまたはモデル) を搭載した複数のモバイルデバイスでテストを実行します。

また、TensorFlow Lite は上級開発者向けに [Android 向けのモデルベンチマークツール](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark)も提供しています。

### デバイス除外リストを作成する

本番環境では、NNAPI が期待どおりに動作しない場合があるので、NNAPI アクセラレーションを特定のモデルと組み合わせて使用しないようにデバイスのリストを維持することをお勧めします。このリストは、`"ro.board.platform"`の値に基づいて作成できます。次のコードスニペットを使用して取得できます。

```java
String boardPlatform = "";

try {
    Process sysProcess =
        new ProcessBuilder("/system/bin/getprop", "ro.board.platform").
        redirectErrorStream(true).start();

    BufferedReader reader = new BufferedReader
        (new InputStreamReader(sysProcess.getInputStream()));
    String currentLine = null;

    while ((currentLine=reader.readLine()) != null){
        boardPlatform = line;
    }
    sysProcess.destroy();
} catch (IOException e) {}

Log.d("Board Platform", boardPlatform);
```

上級開発者には、リモート構成システムを介してこのリストを維持することをお勧めします。TensorFlow チームは、最適な NNAPI 構成の検出と適用を簡素化および自動化する方法に積極的に取り組んでいます。

### 量子化

量子化では、計算に 32 ビットの浮動小数点数ではなく、8 ビットの整数または 16 ビットの浮動小数点数を使用することにより、モデルのサイズを縮小します。8 ビット整数モデルのサイズは、32 ビット浮動小数点バージョンの 4 分の 1 です。16 ビット浮動小数点数のサイズは半分です。量子化によってパフォーマンスが大幅に向上しますが、その処理過程でモデルの精度が低下することがあります。

様々な種類のトレーニング後の量子化手法を利用できますが、現在のハードウェアで最大限のサポートと高速化を実現するには、[完全整数量子化](post_training_quantization#full_integer_quantization_of_weights_and_activations)をお勧めします。このアプローチは、重みと演算の両方を整数に変換します。この量子化プロセスが機能するには、代表的なデータセットが必要です。

### サポートされているモデルと演算を使用する

NNAPI デリゲートがモデル内の一部の演算子またはパラメータの組み合わせをサポートしていない場合、フレームワークは、アクセラレータでグラフがサポートされている部分のみを実行します。残りは CPU で実行されるため、分割実行になります。CPU/アクセラレータの同期には高いコストがかかるため、CPU だけでネットワーク全体を実行するよりもパフォーマンスが低下する可能性があります。

NNAPI は、モデルが[サポートされている演算子](https://developer.android.com/ndk/guides/neuralnetworks#model)のみを使用する場合に最適に機能します。以下のモデルは、NNAPI と互換性があることが確認されています。

- [MobileNet v1 (224x224) 画像分類（浮動小数点数モデルのダウンロード）](https://ai.googleblog.com/2017/06/mobilenets-open-source-models-for.html) [(量子化モデルのダウンロード)](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224_quant.tgz) <br>(モバイルおよび組み込みベースのビジョンアプリケーション向けに設計された画像分類モデル)
- [MobileNet SSD 物体検出](https://ai.googleblog.com/2018/07/accelerated-training-and-inference-with.html)[（ダウンロード）](https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/mobile_ssd_v2_float_coco.tflite) <br>*(バウンディングボックスで複数のオブジェクトを検出する画像分類モデル)*
- [MobileNet v1(300x300) Single Shot Detector (SSD) 物体検出](https://ai.googleblog.com/2018/07/accelerated-training-and-inference-with.html) [(ダウンロード)] (https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip)
- [ポーズ推定のための PoseNet](https://github.com/tensorflow/tfjs-models/tree/master/posenet) [（ダウンロード）](https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/multi_person_mobilenet_v1_075_float.tflite) <br><i>(画像または動画内の人物のポーズを推定するビジョンモデル)</i>

また、モデルに動的サイズの出力が含まれている場合も、NNAPI アクセラレーションはサポートされません。この場合、次のような警告が表示されます。

```none
ERROR: Attempting to use a delegate that only supports static-sized tensors with a graph that has dynamic-sized tensors.
```

### NNAPI CPU の実装を有効にする

アクセラレータで完全に処理できないグラフは、NNAPI  CPU 実装にフォールバックできます。ただし、これは通常、TensorFlow インタープリタよりもパフォーマンスが低いため、Android 10（API レベル 29）以降の NNAPI デリゲートでは、このオプションはデフォルトで無効になっています。この動作をオーバーライドするには、`NnApi Delegate.Options`オブジェクトで`setUseNnapiCpu`を`true`に設定します。
