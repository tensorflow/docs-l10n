# TensorFlow Lite デリゲート

## はじめに

**デリゲート**は、GPU や[デジタルシグナルプロセッサ（DSP）](https://en.wikipedia.org/wiki/Digital_signal_processor)などのデバイス上のアクセラレータを活用して、TensorFlow Lite モデルのハードウェアアクセラレーションを有効にします。

デフォルトでは、TensorFlow Lite は [ARM Neon](https://developer.arm.com/documentation/dht0002/a/Introducing-NEON/NEON-architecture-overview/NEON-instructions) 命令セット用に最適化された CPU カーネルを利用します。ただし、CPU は多目的プロセッサであり、機械学習モデルで一般的に使用される大きな演算（畳み込みや密なレイヤーに関連する行列計算）に必ずしも最適化されているわけではありません。

一方、最近のほとんどのスマートフォンには、これらの大きな演算の処理に優れたチップが搭載されています。それらをニューラルネットワークの演算に利用すると、レイテンシと電力効率の点で大きなメリットがあります。たとえば、GPU ではレイテンシを最大[ 5 倍高速化](https://blog.tensorflow.org/2020/08/faster-mobile-gpu-inference-with-opencl.html)でき、[Qualcomm® Hexagon DSP](https://developer.qualcomm.com/software/hexagon-dsp-sdk/dsp-processor)では、実験では消費電力を最大 75 ％削減できることが示されています。

これらの各アクセラレータには、モバイル GPU の場合は [OpenCL](https://www.khronos.org/opencl/) または [OpenGL ES](https://www.khronos.org/opengles/)、DSP の場合は [Qualcomm® Hexagon SDK](https://developer.qualcomm.com/software/hexagon-dsp-sdk) など、カスタム計算を可能にする API が関連付けられています。通常、これらのインターフェイスを介してニューラルネットワークを実行するには、多くのカスタムコードを作成する必要があります。各アクセラレータには長所と短所があり、ニューラルネットワークのすべての演算を実行できるわけではないことを考えると、さらに複雑になります。TensorFlow Lite の Delegate API は、TFLite ランタイムとこれらの低レベル API の間のブリッジとして機能することにより、この問題を解決します。

![Original graph](../images/performance/tflite_delegate_graph_1.png "Original Graph")

## デレゲートの選択

TensorFlow Lite は複数のデリゲートをサポートしており、各デリゲートは特定のプラットフォームおよび特定のタイプのモデル用に最適化されています。通常、2 つの主要な基準（ターゲットとする*プラットフォーム*（Android または iOS）、高速化する*モデル型*（浮動小数点または量子化））に応じて、ユースケースに適用できる複数のデリゲートがあります。

### プラットフォーム別のデリゲート

#### クロスプラットフォーム（Android および iOS）

- **GPU デリゲート** - Android と iOS の両方で使用できます。GPU が利用可能な場合に 32 ビットおよび 16 ビットの浮動小数点数ベースのモデルを実行するように最適化されています。また、8 ビットの量子化されたモデルもサポートし、浮動小数点数モデルと同等の GPU パフォーマンスを提供します。GPU デリゲートに関する詳細については、[GPU の TensorFlow Lite](gpu_advanced.md) を参照してください。Android および iOS の GPU デリゲートの使用に関する段階的なチュートリアルについては、[TensorFlow Lite GPU デリゲートチュートリアル](gpu.md)を参照してください。

#### Android

- **新しい Android デバイス用 NNAPI デリゲート** - NNAPI デリゲートは、GPU、DSP および/または NPU が利用可能な Android デバイスでのモデルの高速化に使用することができます。Android 8.1 (API 27+) 以上で利用可能です。NNAPI デリゲートの概要、段階的な手順、ベストプラクティスについては、[TensorFlow Lite NNAPI デリゲート](nnapi.md)を参照してください。
- **古い Android デバイス用 Hexagon デリゲート** - Hexagon デリゲートは、Qualcomm Hexagon DSP を搭載した Android デバイスでのモデルの高速化に使用できます。NNAPI を完全にサポートしていない古いバージョンの Android OS デバイスで使用することができます。詳細については、[TensorFlow Lite Hexagon デリゲート](hexagon_delegate.md) を参照してください。

#### iOS

- **新しい iPhone と iPad 用 Core ML デリゲート** - Neural Engine が利用可能な最近の iPhone や iPad では、Core ML デリゲートを使用して単精度浮動小数点数または半精度浮動小数点数を使用するモデルの推論を高速化することができます。Neural Engine は、A12 SoC 以上の Apple モバイルデバイスで利用できます。Core ML デリゲートの概要と段階的な手順については、[TensorFlow Lite Core ML デリゲート](coreml_delegate.md)をご覧ください。

### モデルタイプ別のデリゲート

各アクセラレータは、特定のビット幅のデータを念頭に置いて設計されています。8 ビットの量子化演算のみをサポートするデリゲート（[Hexagon デリゲート](hexagon_delegate.md)など）に浮動小数点モデルを提供すると、すべての演算が拒否され、モデルは完全に CPU で実行されます。このようなことを避けるために、以下の表は、モデル型に基づいたデリゲートサポートの概要を示しています。

**モデルタイプ** | **GPU** | **NNAPI** | **Hexagon** | **CoreML**
--- | --- | --- | --- | ---
単精度浮動小数点（32ビット） | 〇 | 〇 | ✕ | 〇
トレーニング後の float16 量子化 | 〇 | ✕ | ✕ | 〇
トレーニング後のダイナミックレンジ量子化 | 〇 | 〇 | ✕ | ✕
トレーニング後の整数量子化 | 〇 | 〇 | 〇 | ✕
量子化認識トレーニング | 〇 | 〇 | 〇 | ✕

### パフォーマンスの検証

このセクションの情報は、アプリケーションを改善するためのデリゲートを選択するためのガイドラインです。ただし、各デリゲートには、サポートする一連の事前定義された演算があり、モデルとデバイスによって実行が異なる場合があることに注意する必要があります。たとえば、[NNAPI デリゲート](nnapi.md)は、Pixel スマートフォンでは Google Edge-TPU を使用し、別のデバイスでは DSP を使用できます。したがって、通常は、ベンチマークを実行して、デリゲートがニーズをどの程度満たすかを評価することをお勧めします。これは、TensorFlow Lite ランタイムへのデリゲートのアタッチに関連するバイナリサイズの増加を正当化するのにも役立ちます。

TensorFlow Lite には、開発者が自信をもってアプリケーションでデリゲートを使用できるように、広範なパフォーマンスと精度評価ツールが用意されています。これらのツールについては、次のセクションで説明します。

## 評価のためのツール

### レイテンシとメモリフットプリント

TensorFlow Lite の[ベンチマークツール](https://www.tensorflow.org/lite/performance/measurement)を適切なパラメータとともに使用して、平均推論レイテンシ、初期化オーバーヘッド、メモリフットプリントなどのモデルパフォーマンスを推定できます。このツールは、モデルに最適なデリゲート構成を把握するための複数のフラグをサポートしています。たとえば、`--gpu_backend=gl`を`--use_gpu`で指定して、OpenGL で GPU の実行を測定できます。サポートされているデリゲートパラメータの完全なリストは、[詳細なドキュメント](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/tools/delegates/README.md#tflite-delegate-registrar)で定義されています。

`adb`を介して GPU を使用した量子化モデルの実行例を次に示します。

```
adb shell /data/local/tmp/benchmark_model \
  --graph=/data/local/tmp/mobilenet_v1_224_quant.tflite \
  --use_gpu=true
```

Android 64 ビット ARM アーキテクチャ用のツールのビルド済みバージョンは[こちら](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_aarch64_benchmark_model.apk)からダウンロードできます（[詳細はこちら](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark/android)）。

### 精度と正確性

デリゲートは通常、対応する CPU とは異なる精度で計算を実行します。その結果、ハードウェアアクセラレーションにデリゲートを利用することに関連して、（通常はマイナーな）精度のトレードオフが発生します。これは*常に*発生するわけではありません。たとえば、GPU は浮動小数点精度を使用して量子化モデルを実行するため、わずかな精度の向上が見られる場合があります（1% 未満の ILSVRC 画像分類の上位 5 の向上）。

TensorFlow Lite には、デリゲートが特定のモデルに対してどの程度正確に動作するかを測定するための 2 種類のツール（*タスクベースのツール*と*タスクに依存しないツール*）があります。このセクションで説明するすべてのツールは、前のセクションのベンチマークツールで使用された[高度なデレゲーションパラメータ](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/tools/delegates/README.md#tflite-delegate-registrar)をサポートしています。以下のサブセクションでは、モデル評価（モデル自体はタスクに適しているか？）ではなく、*デリゲート評価*（デリゲートは CPU と同じように実行されているか？）に焦点を当てていることに注意してください。

#### タスクベースの評価

TensorFlow Lite には、2 つの画像ベースのタスクの正確さを評価するためのツールがあります。

- [ILSVRC 2012](http://image-net.org/challenges/LSVRC/2012/)（画像分類）[ top-K 精度](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Precision_at_K)

- [COCO 物体検出（バウンディングボックス付き）](https://cocodataset.org/#detection-2020) [MAP（mean Average Precision）付き](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Mean_average_precision)

これらのツール（Android、64 ビット ARM アーキテクチャ）のビルド済みバイナリとドキュメントは、以下にあります。

- [ImageNet 画像分類](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_aarch64_eval_imagenet_image_classification) ([詳細はこちら](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/evaluation/tasks/imagenet_image_classification))
- [COCO 物体検出](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_aarch64_eval_coco_object_detection) ([詳細はこちら](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/evaluation/tasks/coco_object_detection))

以下の例は、Pixel 4 で Google の Edge-TPU を利用する NNAPI を使用した[画像分類評価](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/evaluation/tasks/imagenet_image_classification)を示しています。

```
adb shell /data/local/tmp/run_eval \
  --model_file=/data/local/tmp/mobilenet_quant_v1_224.tflite \
  --ground_truth_images_path=/data/local/tmp/ilsvrc_images \
  --ground_truth_labels=/data/local/tmp/ilsvrc_validation_labels.txt \
  --model_output_labels=/data/local/tmp/model_output_labels.txt \
  --output_file_path=/data/local/tmp/accuracy_output.txt \
  --num_images=0 # Run on all images. \
  --use_nnapi=true \
  --nnapi_accelerator_name=google-edgetpu
```

期待される出力は、1 から 10 までの Top-K メトリックのリストです。

```
Top-1 Accuracy: 0.733333
Top-2 Accuracy: 0.826667
Top-3 Accuracy: 0.856667
Top-4 Accuracy: 0.87
Top-5 Accuracy: 0.89
Top-6 Accuracy: 0.903333
Top-7 Accuracy: 0.906667
Top-8 Accuracy: 0.913333
Top-9 Accuracy: 0.92
Top-10 Accuracy: 0.923333
```

#### タスクにとらわれない評価

確立されたデバイス上の評価ツールがないタスクの場合、またはカスタムモデルを実験している場合、TensorFlow Lite には[ Inference Diff](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/evaluation/tasks/inference_diff) ツールがあります。（Android、64 ビット ARM バイナリアーキテクチャバイナリは[こちら](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_aarch64_eval_inference_diff)からダウンロードできます）

Inference Diff は、次の 2 つの設定で TensorFlow Lite の実行を（レイテンシと出力値の偏差の観点から）比較します。

- シングルスレッド CPU 推論
- ユーザー定義の推論-[これらのパラメータ](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/tools/delegates/README.md#tflite-delegate-registrar)により定義されます

そのために、ツールはランダムなガウスデータを生成し、それを 2 つの TFLite インタープリタに渡します。1 つはシングルスレッド CPU カーネルを実行し、もう 1 つはユーザーの引数によりパラメータ化されます。

両方のレイテンシと、各インタープリタからの出力テンソル間の絶対差を要素ごとに測定します。

単一の出力テンソルを持つモデルの場合、出力は次のようになります。

```
Num evaluation runs: 50
Reference run latency: avg=84364.2(us), std_dev=12525(us)
Test run latency: avg=7281.64(us), std_dev=2089(us)
OutputDiff[0]: avg_error=1.96277e-05, std_dev=6.95767e-06
```

これは、インデックス`0`の出力テンソルの場合、CPU 出力の要素は、デリゲート出力とは平均して`1.96e-05`異なることを意味します。

これらの数値を解釈するには、モデルと、各出力テンソルが何を意味するかについてのより深い知識が必要です。ある種のスコアまたは埋め込みを決定する単純な回帰の場合、差は小さいはずです（そうでない場合は、デリゲートのエラーです）。ただし、SSD モデルからの「検出クラス」のような出力は、解釈が少し難しくなります。たとえば、このツールを使用すると違いが表示される場合がありますが、デリゲートに実際に問題があることを意味するわけではありません。「テレビ（ID：10）」、「モニター（ID：20）」の 2 つの（偽の）クラスを検討してください。デリゲートが正解から少し外れていて、テレビの代わりにモニターを表示している場合、このテンソルの出力差分は高くなる可能性があります（20-10 = 10）。
