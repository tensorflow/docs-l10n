# セグメンテーション

<img src="../images/segmentation.png" class="attempt-right">

## はじめに

*DeepLab* は、セマンティック画像セグメンテーションの最先端のディープラーニングモデルであり、入力画像の各ピクセルにセマンティックラベル（例えば、人、犬、猫など）を割り当てることを目的としています。

TensorFlow Lite を初めて使用する場合、Android または iOS を使用する場合は、以下のサンプルアプリをご覧ください。

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/image_segmentation/android">Android の例</a>

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/image_segmentation/ios">iOS example</a>

Android 以外のプラットフォームを使用する場合、または、すでに <a href="https://www.tensorflow.org/api_docs/python/tf/lite">TensorFlow Lite API</a> に精通している場合は、画像セグメンテーション スターターモデルをダウンロードしてください。

<a class="button button-primary" href="https://tfhub.dev/tensorflow/lite-model/deeplabv3/1/metadata/2?lite-format=tflite">スターターモデルをダウンロードする</a>

## 使い方

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
    <td rowspan="3"><a href="https://tfhub.dev/tensorflow/lite-model/deeplabv3/1/metadata/2?lite-format=tflite">Deeplab v3</a></td>
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

* 4 つのスレッドを使用。

** 最高のパフォーマンス結果を得るために、iPhone では 2 つのスレッドを使用。

## 出力例

モデルは、ターゲットオブジェクトの上に高精度でマスクを作成します。

 <img alt="Animation showing image segmentation" src="https://github.com/tensorflow/docs-l10n/blob/master/site/ja/lite/models/segmentation/images/segmentation.gif?raw=true" class="">

## セグメンテーションについてもっと読む

<ul>
  <li><p data-md-type="paragraph"><a href="https://ai.googleblog.com/2018/03/semantic-image-segmentation-with.html">TensorFlow で DeepLab を使用したセマンティック画像セグメンテーション</a></p></li>
  <li><p data-md-type="paragraph"><a href="https://medium.com/tensorflow/tensorflow-lite-now-faster-with-mobile-gpus-developer-preview-e15797e6dee7">モバイル GPU で高速化した TensorFlow Lite（開発者プレビュー）</a></p></li>
  <li><p data-md-type="paragraph"><a href="https://github.com/tensorflow/models/tree/master/research/deeplab">DeepLab: セマンティック画像セグメンテーションのためのディープラベリング</a></p></li>
</ul>
