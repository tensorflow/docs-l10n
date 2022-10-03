# TensorFlow Lite の GPU デリゲート

グラフィックス処理装置（GPU）を使用して機械学習（ML）モデルを実行すると、モデルのパフォーマンスと ML 対応アプリケーションのユーザーエクスペリエンスが大幅に向上します。 TensorFlow Lite では、[*デリゲート*](./delegates)と呼ばれるハードウェアドライバーを介して、GPU やその他の専用プロセッサを使用できます。TensorFlow Lite ML アプリケーションで GPU の使用を有効にすると、次のメリットがあります。

- **速度** - GPU は、大規模に実行する並行化可能なワークロードで高い処理能力を実現するように構築されています。そのため、この設計は多数の演算で構成されるディープニューラルネットにネットワークに適しています。各演算は並行処理できる入力テンソルで機能するため、通常レイテンシが低くなります。最良のシナリオでは、GPU でモデルを実行すると、以前は不可能だったリアルタイムアプリケーションが利用可能になる速度で実行できる可能性があります。
- **電力効率** - GPU は、非常に効率的かつ最適化された方法で ML 計算を実行します。通常、CPU で実行される同じタスクよりも消費電力と発熱が少なくなります。

このドキュメントでは、TensorFlow Lite での GPU サポートの概要と、GPU プロセッサの高度な使用法について説明します。特定のプラットフォームでの GPU サポートの実装に関する具体的な情報については、次のガイドを参照してください。

- [Android の GPU サポート](../android/delegates/gpu)
- [iOS の GPU サポート](../ios/delegates/gpu)

## GPU ML 操作のサポート {:#supported_ops}

TensorFlow Lite GPU デリゲートによって高速化できる TensorFlow ML 演算（*ops*）には、いくつかの制限があります。デリゲートは、16 ビットおよび 32 ビットの浮動小数点精度で次の操作をサポートします。

- `ADD`
- `AVERAGE_POOL_2D`
- `CONCATENATION`
- `CONV_2D`
- `DEPTHWISE_CONV_2D v1-2`
- `EXP`
- `FULLY_CONNECTED`
- `LOGISTIC`
- `LSTM v2（基本的な LSTM のみ）`
- `MAX_POOL_2D`
- `MAXIMUM`
- `MINIMUM`
- `MUL`
- `PAD`
- `PRELU`
- `RELU`
- `RELU6`
- `RESHAPE`
- `RESIZE_BILINEAR v1-3`
- `SOFTMAX`
- `STRIDED_SLICE`
- `SUB`
- `TRANSPOSE_CONV`

デフォルトでは、すべての演算はバージョン 1 でのみサポートされます。[量子化サポート](#quantized-models)を有効にすると、ADD v2 などの適切なバージョンが有効になります。

### GPU サポートのトラブルシューティング

一部の演算が GPU デリゲートでサポートされていない場合、フレームワークは GPU でグラフの一部のみを実行し、残りの部分を CPU で実行します。CPU と GPU 同期のコストは高いため、このような分割実行モードでは、ネットワーク全体が CPU のみで実行されている場合よりもパフォーマンスが遅くなることがよくあります。この場合、アプリケーションは次のような警告を生成します。

```none
WARNING: op code #42 cannot be handled by this delegate.
```

これは実際のランタイムエラーではないため、このタイプのエラーに対するコールバックはありません。GPU デリゲートを使用してモデルの実行をテストするときは、これらの警告に注意する必要があります。これらの警告の数が多い場合は、モデルが GPU アクセラレーションの使用に最適ではないことを示している可能性があり、モデルのリファクタリングが必要になる場合があります。

## モデル例

次のサンプルモデルは、TensorFlow Lite で GPU アクセラレーションを利用するために構築されており、参照とテスト用に提供されています。

- [MobileNet v1（224x224）画像分類](https://ai.googleblog.com/2017/06/mobilenets-open-source-models-for.html) - モバイルおよび組み込みベースのビジョンアプリケーション向けに設計された画像分類モデル。（[モデル](https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/classification/5)）
- [DeepLab セグメンテーション（257x257）](https://ai.googleblog.com/2018/03/semantic-image-segmentation-with.html) - 犬、猫、車などのセマンティックラベルを入力画像のすべてのピクセルに割り当てる画像セグメンテーションモデル。（[モデル](https://tfhub.dev/tensorflow/lite-model/deeplabv3/1/default/1)）
- [MobileNet SSD オブジェクト検出](https://ai.googleblog.com/2018/07/accelerated-training-and-inference-with.html) - バウンディングボックスで複数のオブジェクトを検出する画像分類モデル。（[モデル](https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/mobile_ssd_v2_float_coco.tflite)）
- [ポーズ推定のための PoseNet](https://github.com/tensorflow/tfjs-models/tree/master/pose-detection) - 画像または動画内の人物のポーズを推定するビジョンモデル。（[モデル](https://tfhub.dev/tensorflow/lite-model/posenet/mobilenet/float/075/1/default/1)）

## GPU の最適化

次の手法は、TensorFlow Lite GPU デリゲートを使用して GPU ハードウェアでモデルを実行する際のパフォーマンスを向上させるのに役立ちます。

- CPU において簡単な一部の演算は、モバイルデバイスの GPU には高コストとなる可能性があります。特に形状変更を行う演算の実行コストは高く、こういった演算には、<code>BATCH_TO_SPACE</code>、`SPACE_TO_BATCH`、`SPACE_TO_DEPTH` などがあります。形状変更を行う演算の使用を十分に調べ、データ探索やモデルの早期のイテレーションのみに適用されていないか検討することをお勧めします。こういった演算を除去することで、パフォーマンスを大幅に改善できます。

- **画像データチャンネル** - GPU では、テンソルデータは 4 チャンネルにスライスされるため、形状 `[B,H,W,5]` のテンソルでの計算は、形状 `[B,H,W,8]` のテンソルでの計算とほぼ同程度のパフォーマンスを見せますが、`[B,H,W,4]` に比べれば、著しく悪化します。使用しているカメラハードウェアが RGBA の画像フレームをサポートしている場合、その 4 チャネル入力の供給は、3 チャネル RGB から 4 チャネル RGBX へのメモリコピーが回避されるため、はるかに高速になります。

- **モバイル向けに最適化されたモデル** - 最高のパフォーマンスを得るには、モバイル向けに最適化されたネットワークアーキテクチャを使用して分類器を再トレーニングすることを検討する必要があります。デバイス上の推論を最適化によって、モバイルハードウェアの機能を利用し、レイテンシと電力消費量を大幅に削減できます。

## 高度な GPU サポート

GPU 処理で追加の高度な手法を使用して、量子化やシリアル化など、モデルのパフォーマンスをさらに向上させることができます。以下のセクションでは、これらの手法について詳しく説明します。

### 量子化モデルの使用 {:#quantized-models}

このセクションでは、GPU デリゲートが 8 ビットの量子化モデルを高速化する方法について説明します。

- [量子化認識トレーニング](https://www.tensorflow.org/model_optimization/guide/quantization/training)でトレーニングされたモデル
- トレーニング後の[ダイナミックレンジ量子化](https://www.tensorflow.org/lite/performance/post_training_quant)
- トレーニング後の[完全整数量子化](https://www.tensorflow.org/lite/performance/post_training_integer_quant)

パフォーマンスを最適化するには、浮動小数点入出力テンソルを持つモデルを使用します。

#### 仕組み

GPU バックエンドは浮動小数点の実行のみをサポートするため、元のモデルの「浮動小数点ビュー」を与えて量子化モデルを実行します。上位レベルで、次のような手順が含まれます。

- 定数テンソル（重み/バイアスなど）は、GPU メモリに一度逆量子化されます。この演算は、デリゲートが  TensorFlow Lite に対して有効である場合に発生します。

- 8 ビット量子化されている場合、GPU プログラムへの*入出力*は、推論ごとにそれぞれ逆量子化および量子化されます。この演算は、TensorFlow FLite の最適化されたカーネルを使用して CPU 上で行われます。

- 演算間に*量子化シミュレータ*が挿入され、量子化された動作が模倣されます。このアプローチは、演算が量子化中に学習した範囲に活性化が続くことを期待するモデルに必要です。

GPU デリゲートでこの機能を有効にする方法については、次を参照してください。

- [Android の GPU で量子化されたモデル](../android/delegates/gpu#quantized-models)を使用する
- [iOS の GPU で量子化されたモデル](../ios/delegates/gpu#quantized-models)を使用する

### シリアル化による初期化時間の削減 {:#delegate_serialization}

GPU デリゲート機能を使用すると、事前にコンパイルされたカーネルコードと、シリアル化されて以前の実行からディスクに保存されたモデルデータから読み込むことができます。このアプローチは再コンパイルを回避し、起動時間を最大 90% 短縮できます。この改善は、時間を節約するためにディスク領域を交換することで達成されます。次のコード例に示すように、いくつかの構成オプションを使用してこの機能を有効にすることができます。

<div>
  <devsite-selector>
    <section>
      <h3>C++</h3>
      <p></p>
<pre class="prettyprint lang-cpp">    TfLiteGpuDelegateOptionsV2 options = TfLiteGpuDelegateOptionsV2Default();
    options.experimental_flags |= TFLITE_GPU_EXPERIMENTAL_FLAGS_ENABLE_SERIALIZATION;
    options.serialization_dir = kTmpDir;
    options.model_token = kModelToken;

    auto* delegate = TfLiteGpuDelegateV2Create(options);
    if (interpreter-&gt;ModifyGraphWithDelegate(delegate) != kTfLiteOk) return false;
      </pre>
    </section>
    <section>
      <h3>Java</h3>
      <p></p>
<pre class="prettyprint lang-java">    GpuDelegate delegate = new GpuDelegate(
      new GpuDelegate.Options().setSerializationParams(
        /* serializationDir= */ serializationDir,
        /* modelToken= */ modelToken));

    Interpreter.Options options = (new Interpreter.Options()).addDelegate(delegate);
      </pre>
    </section>
  </devsite-selector>
</div>

シリアル化機能を使用する場合、コードが以下の実装ルールでコンパイルすることを確認してください。

- シリアル化データを他のアプリがアクセスできないディレクトリに保存します。Android デバイスでは、現在のアプリケーションに非公開の場所にポイントする getCodeCacheDir() を使用します。
- モデルトークンは、特定のモデルのデバイスに一意である必要があります。モデルトークンは、[`farmhash::Fingerprint64`](https://github.com/google/farmhash) など、モデルデータからフィンガープリントを生成することで計算できます。

注意: このシリアル化機能を使用するには、[OpenCL SDK](https://github.com/KhronosGroup/OpenCL-SDK) が必要です。
