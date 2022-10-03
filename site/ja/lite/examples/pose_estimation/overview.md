# ポーズ推定

 <img alt="Animation showing pose estimation" src="https://www.tensorflow.org/images/lite/models/pose_estimation.gif">

ポーズ推定は、ML モデルを使用して、主要な体の関節 (キーポイント) の空間的な位置を推定することで、画像または動画から人のポーズを推定するタスクです。

## はじめに

<a class="button button-primary" href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite">スターターモデルをダウンロードする</a>

<a data-md-type="raw_html" class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/pose_estimation/android">Android の例</a> <a data-md-type="raw_html" class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/pose_estimation/ios">iOS の例</a>

[TensorFlow Lite API](https://www.tensorflow.org/api_docs/python/tf/lite) に慣れている場合は、スターター MoveNet ポーズ推定モデルと追加ファイルをダウンロードしてください。

<a class="button button-primary" href="https://tfhub.dev/s?q=movenet"> スターターモデルをダウンロードする</a>

Web ブラウザでポーズ推定を試す場合は、<a href="https://storage.googleapis.com/tfjs-models/demos/pose-detection/index.html?model=movenet">TensorFlow JS デモ</a>を参照してください。

## モデルの説明

### 使い方

ポーズ推定は、コンピュータビジョン手法を参照して、画像や動画の人物を検出するため、たとえば、誰かのひじが画像に現れる場所を判定できます。ポーズ推定では、主要な体の関節の場所を推定することはほとんどなく、画像や動画の人物が誰なのかを認識することもないという事実を理解することが重要です。

ポーズ推定モデルは、処理済みのカメラ画像を入力として受け取り、キーポイントに関する情報を出力します。検出されたキーポイントは、信頼度スコア 0.0 ～ 1.0 のパーツ ID によってインデックス付けされます。信頼度スコアは、キーポイントがその位置に存在する確率を示します。

次の TensorFlow Lite の 2 つのポーズ推定モデルについて、実装の参考情報が提供されています。

- MoveNet: Lighting と Thunder という 2 つのバージョンで提供されている最先端のポーズ推定モデル。この 2 つの比較については、以下のセクションを参照してください。
- PoseNet: 2017 年にリリースされた前の世代のポーズ推定モデル。

ポーズ推定モデルで検出されたさまざまな体の関節は、次の表のとおりです。

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

次に、出力の例を示します。

 <img alt="Animation showing pose estimation" src="https://storage.googleapis.com/download.tensorflow.org/example_images/movenet_demo.gif" class="">

## パフォーマンスベンチマーク

MoveNet は次の 2 つのバージョンで提供されています。

- MoveNet.Lightning は小さく高速ですが、Thunder バージョンよりも精度が劣ります。最新のスマートフォンでリアルタイムで実行できます。
- MoveNet.Thunder は精度が高いバージョンですが、Lightning よりもサイズが大きく低速です。高い精度が求められるユースケースで有用です。

MoveNet は、さまざまなデータセットに対して、PoseNet よりも優れています。特に、フィットネスアクション画像を含む画像で優れています。このため、PoseNet よりも MoveNet を使用することをお勧めします。

パフォーマンスベンチマークの数値は、[こちらで説明されている](../../performance/measurement)ツールを使用して生成されています。精度 (mAP) は、各画像が 1 人の人物だけを含むようにフィルタおよび切り取りされた [COCO データセット](https://cocodataset.org/#home)のサブセットに対して測定されます。

<table>
<thead>
  <tr>
    <th rowspan="2">モデル</th>
    <th rowspan="2">サイズ (MB)</th>
    <th rowspan="2">mAP</th>
    <th colspan="3">レイテンシ (ms)</th>
  </tr>
  <tr>
    <td>Pixel 5 - CPU 4 スレッド</td>
    <td>Pixel 5 - GPU</td>
    <td>Raspberry Pi 4 - CPU 4 スレッド</td>
  </tr>
</thead>
<tbody>
  <tr>
    <td>       <a href="https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/tflite/float16/4">MoveNet.Thunder (FP16 量子化)</a> </td>
    <td>12.6MB</td>
    <td>72.0</td>
    <td>155ms</td>
    <td>45ms</td>
    <td>594ms</td>
  </tr>
  <tr>
    <td>       <a href="https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/tflite/int8/4">MoveNet.Thunder (INT8 量子化)</a> </td>
    <td>7.1MB</td>
    <td>68.9</td>
    <td>100ms</td>
    <td>52ms</td>
    <td>251ms</td>
  </tr>
  <tr>
    <td>       <a href="https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/float16/4">MoveNet.Lightning (FP16 量子化)</a> </td>
    <td>4.8MB</td>
    <td>63.0</td>
    <td>60ms</td>
    <td>25ms</td>
    <td>186ms</td>
  </tr>
  <tr>
    <td>       <a href="https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/int8/4">MoveNet.Lightning (INT8 量子化)</a> </td>
    <td>2.9MB</td>
    <td>57.4</td>
    <td>52ms</td>
    <td>28ms</td>
    <td>95ms</td>
  </tr>
  <tr>
    <td>       <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite">PoseNet(MobileNetV1 バックボーン、FP32)</a> </td>
    <td>13.3MB</td>
    <td>45.6</td>
    <td>80ms</td>
    <td>40ms</td>
    <td>338ms</td>
  </tr>
</tbody>
</table>

## その他の資料とリソース

- MoveNet と TensorFlow Lite を使用したポーズ推定の詳細については、こちらの[ブログ投稿](https://blog.tensorflow.org/2021/08/pose-estimation-and-classification-on-edge-devices-with-MoveNet-and-TensorFlow-Lite.html)をお読みください。
- Web でのポーズ推定の詳細については、こちらの[ブログ投稿](https://blog.tensorflow.org/2021/05/next-generation-pose-detection-with-movenet-and-tensorflowjs.html)をお読みください。
- TensorFlow Hub のモデルを使用した Python での MoveNet の実行の詳細については、こちらの[チュートリアル](https://www.tensorflow.org/hub/tutorials/movenet)を参照してください。
- Coral/EdgeTPU では、エッジデバイスでポーズ推定を大幅に高速化して実行できます。詳細については、[EdgeTPU 最適化モデル](https://coral.ai/models/pose-estimation/)を参照してください。
- [こちら](https://arxiv.org/abs/1803.08225)から PoseNet ドキュメントをお読みください。

次のポーズ推定のユースケースも確認してください。

<ul>
  <li><a href="https://vimeo.com/128375543">‘PomPom Mirror’</a></li>
  <li><a href="https://youtu.be/I5__9hq-yas">Amazing Art Installation Turns You Into A Bird | Chris Milk "The Treachery of Sanctuary"</a></li>
  <li><a href="https://vimeo.com/34824490">Puppet Parade - Interactive Kinect Puppets</a></li>
  <li><a href="https://vimeo.com/2892576">Messa di Voce (Performance), Excerpts</a></li>
  <li><a href="https://www.instagram.com/p/BbkKLiegrTR/">Augmented reality</a></li>
  <li><a href="https://www.instagram.com/p/Bg1EgOihgyh/">対話型アニメーション</a></li>
  <li><a href="https://www.runnersneed.com/expert-advice/gear-guides/gait-analysis.html">歩容解析</a></li>
</ul>
