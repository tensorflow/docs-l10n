# セグメンテーション

Image segmentation is the process of partitioning a digital image into multiple segments (sets of pixels, also known as image objects). The goal of segmentation is to simplify and/or change the representation of an image into something that is more meaningful and easier to analyze.

TensorFlow Lite を初めて使用する場合、Android または iOS を使用する場合は、以下のサンプルアプリをご覧ください。


<img src="../images/segmentation.png" class="attempt-right">

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/image_segmentation/android">Android の例</a>

## はじめに

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/image_segmentation/ios">iOS example</a>

You can leverage the out-of-box API from [TensorFlow Lite Task Library](../../inference_with_metadata/task_library/image_segmenter) to integrate image segmentation models within just a few lines of code. You can also integrate the model using the [TensorFlow Lite Interpreter Java API](../../guide/inference#load_and_run_a_model_in_java).

以下の Android の例では、両方のメソッドをそれぞれ [lib_task_api](https://github.com/tensorflow/examples/tree/master/lite/examples/image_segmentation/android/lib_task_api) および [lib_interpreter](https://github.com/tensorflow/examples/tree/master/lite/examples/image_segmentation/android/lib_interpreter) として実装しています。

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/image_segmentation/android">Android の例を見る</a>

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/image_segmentation/ios">View iOS example</a>

Android 以外のプラットフォームを使用する場合、または、すでに <a href="https://www.tensorflow.org/api_docs/python/tf/lite">TensorFlow Lite API</a> に精通している場合は、画像セグメンテーション スターターモデルをダウンロードしてください。

スターターモデルをダウンロードする

## モデルの説明

*DeepLab* は、セマンティック画像セグメンテーションの最先端のディープラーニングモデルであり、入力画像の各ピクセルにセマンティックラベル（例えば、人、犬、猫など）を割り当てることを目的としています。

### 使い方

セマンティック画像セグメンテーションは、画像の各ピクセルが特定のクラスに関連付けられているかどうかを予測します。これは、矩形の領域でオブジェクトを検出する<a href="../object_detection/overview.md">オブジェクト検出</a>、および画像全体を分類する<a href="../image_classification/overview.md">画像分類</a>とは対照的です。

現在の実装は、以下のような機能を備えています。

<ol>
  <li>DeepLabv1: Atrous 畳み込みを使用して解像度を明示的に制御し、深層畳み込みニューラルネットワーク内で特徴応答を計算します。</li>
  <li>DeepLabv2: Atrous 空間ピラミッドプーリング (ASPP) を使用して、複数のサンプリングレートと有効な視野でフィルタを用い、複数のスケールでオブジェクトをロバストにセグメント化します。</li>
  <li>DeepLabv3: ASPP モジュールを画像レベルの特徴 [5, 6] で拡張して、より広い範囲の情報を取得します。また、バッチ正規化 [7] パラメータをインクルードして、トレーニングを容易にします。特に、トレーニングと評価の際には、Atrous 畳み込みを適用し、それぞれ異なる出力ストライドで出力特徴を抽出します。これにより、出力ストライド = 16 で効率的にバッチ正規化トレーニングを行い、評価時には出力ストライド = 8 で高い性能を発揮することが可能になります。</li>
  <li>DeepLabv3+: DeepLabv3 を拡張し、特にオブジェクト境界に沿ってセグメンテーションの結果を洗練させる、シンプルかつ効果的なデコーダモジュールを追加しました。さらに、このエンコーダ/デコーダ構造では、Atrous 畳み込みで抽出されるエンコーダ特徴の解像度を任意に制御して、精度とランタイムをトレードオフすることができます。</li>
</ol>

## パフォーマンスベンチマーク

パフォーマンスベンチマークの数値は、[ここで説明する](https://www.tensorflow.org/lite/performance/benchmarks)ツールで生成されます。

<table>
  <thead>
    <tr>
      <th>モデル名</th>
      <th>モデルサイズ</th>
      <th>デバイス</th>
      <th>GPU</th>
      <th>CPU</th>
    </tr>
  </thead>
  <tr>
    <td rowspan="3">
      <a href="https://tfhub.dev/tensorflow/lite-model/deeplabv3/1/metadata/2?lite-format=tflite">Deeplab v3</a>
    </td>
    <td rowspan="3">       2.7 Mb</td>
    <td>Pixel 3 (Android 10)</td>
    <td>16ms</td>
    <td>37ms*</td>
  </tr>
   <tr>
     <td>Pixel 4 (Android 10)</td>
    <td>20ms</td>
    <td>23ms*</td>
  </tr>
   <tr>
     <td>iPhone XS (iOS 12.4.1)</td>
     <td>16ms</td>
    <td>25ms**</td>
  </tr>
</table>

* 4 threads used.

** 最高のパフォーマンス結果を得るために、iPhone では 2 つのスレッドを使用。

## Further reading and resources

<ul>
  <li><a href="https://ai.googleblog.com/2018/03/semantic-image-segmentation-with.html">Semantic Image Segmentation with DeepLab in TensorFlow</a></li>
  <li><a href="https://medium.com/tensorflow/tensorflow-lite-now-faster-with-mobile-gpus-developer-preview-e15797e6dee7">TensorFlow Lite Now Faster with Mobile GPUs (Developer Preview)</a></li>
  <li><a href="https://github.com/tensorflow/models/tree/master/research/deeplab">DeepLab: Deep Labelling for Semantic Image Segmentation</a></li>
</ul>
