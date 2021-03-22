# レコメンド

個別のレコメンドは、メディアコンテンツの検索、ショッピングの商品提案、アプリの次のレコメンドなど、モバイルデバイス上のさまざまなユースケースで広く使用されています。ユーザーのプライバシーを尊重しながらアプリで個別のレコメンドを提供することに興味がある場合には、以下の例とツールキットの検討をお勧めします。

## はじめに

<img src="images/screenshot.gif" class="attempt-right" style="max-width: 300px">

Android 上でユーザーに関連するアイテムを推奨する方法を実演する TensorFlow Lite のサンプルアプリを提供しています。

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/recommendation/android">Android の例</a>

Android 以外のプラットフォームを使用する場合、または、すでに TensorFlow Lite API に精通している場合は、レコメンドスターターモデルをダウンロードしてください。

<a class="button button-primary" href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/recommendation/20200720/recommendation.tar.gz">スターターモデルをダウンロードする</a>

また、独自のモデルをトレーニングするためのトレーニングスクリプトも GitHub に用意しています。

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/recommendation/ml">トレーニング用コード</a>

## モデルアーキテクチャの理解

Sequential にユーザー履歴をエンコードするコンテキストエンコーダと、予測されたレコメンドをエンコードするラベルエンコーダから成る、デュアルエンコーダ モデルアーキテクチャを活用しています。コンテキストエンコーディングとラベルエンコーディング間の類似性は、予測された候補がユーザーのニーズを満たす可能性を表現するために使用されます。

このコードベースでは、Sequential なユーザー履歴のエンコーディング手法を 3 種類提供しています。

- Bag of Words エンコーダ (BOW): コンテキストの順序を考慮しないでユーザーアクティビティの埋め込みを平均化します。
- 畳み込みニューラルネットワークエンコーダ (CNN): 畳み込みニューラルネットワークの複数のレイヤーを適用してコンテキストエンコーディングを生成します。
- 再帰ニューラルネットワークエンコーダ (RNN): 再帰ニューラルネットワークを適用してコンテキストシーケンスをエンコードします。

*注意: 本モデルは、研究目的のために [MovieLens](https://grouplens.org/datasets/movielens/1m/) データセットを使用してトレーニングしています。

## 使用例

入力 ID:

- マトリックス (ID: 260)
- プライベート・ライアン (ID: 2028)
- （その他）

出力 ID:

- スター・ウォーズ: エピソード 6 - ジェダイの帰還 (ID: 1210)
- （その他）

## パフォーマンスベンチマーク

パフォーマンスベンチマークの数値は、[ここで説明する](https://www.tensorflow.org/lite/performance/benchmarks)ツールで生成されます。

<table>
  <thead>
    <tr>
      <th>モデル名</th>
      <th>モデルサイズ</th>
      <th>デバイス</th>
      <th>CPU</th>
    </tr>
  </thead>
  <tr>
    <td rowspan="3"><a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/recommendation/20200720/model.tar.gz">レコメンド</a></td>
    <td rowspan="3">       0.52 Mb</td>
    <td>Pixel 3</td>
    <td>0.09ms*</td>
  </tr>
   <tr>
     <td>Pixel 4</td>
    <td>0.05ms*</td>
  </tr>
</table>

* 4 つのスレッドを使用。

## 独自のトレーニングデータを使用する

トレーニング済みモデルに加え、オープンソースの[ツールキットを GitHub](https://github.com/tensorflow/examples/tree/master/lite/examples/recommendation/ml) で提供しているので、独自のデータを使用してモデルをトレーニングすることができます。本チュートリアルでは、ツールキットの使い方、そしてトレーニング済みモデルを独自のモバイルアプリにデプロイする方法を学ぶことができます。

独自のデータセットを用いてレコメンドモデルをトレーニングする場合は、この[チュートリアル](https://github.com/tensorflow/examples/tree/master/lite/examples/recommendation/ml/ondevice_recommendation.ipynb)に従い、ここで使用したのと同じ手法を適用してください。

## 独自のデータによるモデルカスタマイズのヒント

このデモアプリケーションに統合されている事前トレーニング済みモデルは [MovieLens](https://grouplens.org/datasets/movielens/1m/) データセットでトレーニングしていますが、独自のデータに基づいて、語彙サイズ、埋め込み次元、入力コンテキストの長さなどのモデル構成を変更することができます。以下にヒントを示します。

- 入力コンテキストの長さ: 入力コンテキストの最適な長さはデータセットによって異なります。ラベルのイベントの長期的関心と短期的コンテキストとの相関関係に基づいて、入力コンテキストの長さを選択することをお勧めします。

- エンコーダタイプの選択: 入力コンテキストの長さに基づいて適切なエンコーダのタイプを選択することをお勧めします。Bag of Words エンコーダは入力コンテキストの長さが短い場合（例えば 10 未満）でよく機能しますが、CNN や RNN エンコーダは入力コンテキストの長さが長い場合により優れた要約力を発揮します。
