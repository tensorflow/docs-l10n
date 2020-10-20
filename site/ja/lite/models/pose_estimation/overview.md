# ポーズ推定

<img src="../images/pose.png" class="attempt-right">

## はじめに

*PoseNet* は、主要な体の関節の位置を推定することにより、画像や動画内の人物のポーズを推定するために使用できるビジョンモデルです。

<a class="button button-primary" href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite">スターターモデルをダウンロードする</a>

Web ブラウザ上で実験する場合には、<a href="https://github.com/tensorflow/tfjs-models/tree/master/posenet">TensorFlow.js GitHub リポジトリ</a>をご覧ください。

### サンプルアプリとガイド

PoseNet モデルのデモが可能な TensorFlow Lite アプリの例を Android と iOS の両方に提供しています。

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/posenet/android">Android の例</a>

## 使い方

ポーズ推定とは、画像や動画内の人物を検出するコンピュータビジョン手法のことで、例えば、画像内の人物の肘の位置などを特定することができます。

正確には、この技術は画像の中に誰がいるかを認識できるわけではありません。アルゴリズムは単に主要な体の関節がどこにあるかを推定するものです。

検出されたキーポイントは「部位 ID」でインデックス化され、信頼度スコアは 0.0 ～ 1.0 で、1.0 が最も高い信頼度となります。

<table style="width: 30%;">
  <thead>
    <tr>
      <th>ID</th>
      <th>部位</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>鼻</td>
    </tr>
    <tr>
      <td>1</td>
      <td>左目</td>
    </tr>
    <tr>
      <td>2</td>
      <td>右目</td>
    </tr>
    <tr>
      <td>3</td>
      <td>左耳</td>
    </tr>
    <tr>
      <td>4</td>
      <td>右耳</td>
    </tr>
    <tr>
      <td>5</td>
      <td>左肩</td>
    </tr>
    <tr>
      <td>6</td>
      <td>右肩</td>
    </tr>
    <tr>
      <td>7</td>
      <td>左ひじ</td>
    </tr>
    <tr>
      <td>8</td>
      <td>右ひじ</td>
    </tr>
    <tr>
      <td>9</td>
      <td>左手首</td>
    </tr>
    <tr>
      <td>10</td>
      <td>右手首</td>
    </tr>
    <tr>
      <td>11</td>
      <td>左腰</td>
    </tr>
    <tr>
      <td>12</td>
      <td>右腰</td>
    </tr>
    <tr>
      <td>13</td>
      <td>左ひざ</td>
    </tr>
    <tr>
      <td>14</td>
      <td>右ひざ</td>
    </tr>
    <tr>
      <td>15</td>
      <td>左足首</td>
    </tr>
    <tr>
      <td>16</td>
      <td>右足首</td>
    </tr>
  </tbody>
</table>

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
      <p data-md-type="paragraph">PoseNet</p>
    </td>
    <td rowspan="3">       12.7 Mb</td>
    <td>Pixel 3 (Android 10)</td>
    <td>12ms</td>
    <td>31ms*</td>
  </tr>
   <tr>
     <td>Pixel 4 (Android 10)</td>
    <td>12ms</td>
    <td>19ms*</td>
  </tr>
   <tr>
     <td>iPhone XS (iOS 12.4.1)</td>
     <td>4.8ms</td>
    <td>22ms**</td>
  </tr>
</table>

* 4 つのスレッドを使用。

** 最高のパフォーマンス結果を得るために、iPhone では 2 つのスレッドを使用。

## 出力例

 <img alt="Animation showing pose estimation" src="https://www.tensorflow.org/images/lite/models/pose_estimation.gif" class="">

## パフォーマンス

使用するデバイスと出力ストライド（ヒートマップとオフセットベクトル）によって、パフォーマンスは異なります。PoseNet モデルは画像サイズが不変であるため、画像サイズがスケールダウンされても、元の画像と同じスケールでポーズの位置を予測することができます。これはパフォーマンスと引き換えにして高い精度が出せるように PoseNet を設定できることを意味します。

出力ストライドは、入力画像サイズに対してどれだけ出力をスケールダウンするかを決定します。これはレイヤーとモデル出力のサイズに影響を与えます。出力ストライドが大きくなれば、ネットワーク内のレイヤーと出力の解像度は小さくなり、それに応じて精度が低下します。この実装では、出力ストライドの値を 8、16、32 のいずれかで設定できます。言い換えると、出力ストライドが 32 の場合は最速のパフォーマンスが得られますが、精度は最も低くなります。出力ストライドが 8 の場合は最高の精度が得られますが、パフォーマンス速度は最も遅くなります。まずは 16 から始めることをお勧めします。

次の図では、入力画像サイズに対して、出力ストライドがどのように出力のスケールダウン値を決定するかを示しています。出力ストライドが大きければ高速になりますが、精度は低下するということになります。

 <img alt="Output stride and heatmap resolution" src="https://github.com/tensorflow/docs-l10n/blob/master/site/ja/lite/models/images/output_stride.png?raw=true" class="">

## ポーズ推定についてもっと読む

<ul>
  <li><p data-md-type="paragraph"><a href="https://medium.com/tensorflow/real-time-human-pose-estimation-in-the-browser-with-tensorflow-js-7dd0bc881cd5">ブログ記事: TensorFlow.js によるブラウザでのリアルタイム人物ポーズ推定</a></p></li>
  <li><p data-md-type="paragraph"><a href="https://github.com/tensorflow/tfjs-models/tree/master/posenet">TF.js GitHub: ブラウザでのポーズ検出: PoseNet モデル</a></p></li>
   <li><p data-md-type="paragraph"><a href="https://medium.com/tensorflow/track-human-poses-in-real-time-on-android-with-tensorflow-lite-e66d0f3e6f9e">ブログ記事: TensorFlow Lite を使用した Android によるリアルタイム人物ポーズ追跡</a></p></li>
</ul>

### ユースケース

<ul>
  <li><p data-md-type="paragraph"><a href="https://vimeo.com/128375543">‘PomPom Mirror’</a></p></li>
  <li><p data-md-type="paragraph"><a href="https://youtu.be/I5__9hq-yas">あなたを鳥に変える驚きのアートインスタレーション | Chris Milk "The Treachery of Sanctuary"</a></p></li>
  <li><p data-md-type="paragraph"><a href="https://vimeo.com/34824490">パペットパレード - インタラクティブ キネクト パペット</a></p></li>
  <li><p data-md-type="paragraph"><a href="https://vimeo.com/2892576">メッサ ディ ヴォーチェ（パフォーマンス）、抜粋</a></p></li>
  <li><p data-md-type="paragraph"><a href="https://www.instagram.com/p/BbkKLiegrTR/">拡張現実 (AR)</a></p></li>
  <li><p data-md-type="paragraph"><a href="https://www.instagram.com/p/Bg1EgOihgyh/">インタラクティブ アニメーション</a></p></li>
  <li><p data-md-type="paragraph"><a href="https://www.runnersneed.com/expert-advice/gear-guides/gait-analysis.html">歩行分析</a></p></li>
</ul>
