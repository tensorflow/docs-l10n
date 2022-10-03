# 強化学習

エージェントを相手にボードゲームをします。エージェントは、強化学習を使用してトレーニングされ、TensorFlow Lite でデプロイされています。

## はじめに


<img src="images/screenshot.gif" class="attempt-right" style="max-width: 300px">

TensorFlow Lite を初めて使用する場合、Android を使用する場合は、以下のサンプルアプリをご覧ください。

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/reinforcement_learning/android">Android の例</a>

Android 以外のプラットフォームを使用する場合、または、すでに [TensorFlow Lite API](https://www.tensorflow.org/api_docs/python/tf/lite) に精通している場合は、トレーニング済みモデルをダウンロードできます。

<a class="button button-primary" href="https://github.com/tensorflow/examples/blob/master/lite/examples/reinforcement_learning/android/app/src/main/assets/planestrike_tf.tflite">モデルのダウンロード</a>

## 使い方

ゲームエージェントが「Plane Strike」という小さいボードゲームをするためのモデルが構築されています。このゲームの簡単な概要とルールについては、[README](https://github.com/tensorflow/examples/tree/master/lite/examples/reinforcement_learning/android) を参照してください。

アプリの UI の基盤には、人間のプレイヤーと対戦するエージェントが構築されています。エージェントは 3 層 MLP であり、ボードの状態を入力値として受け取り、64 のボードのマスのそれぞれに対して予測されたスコアを出力します。モデルは、方策勾配 (REINFORCE) を使用してトレーニングされています。トレーニングコードについては、[こちら](https://github.com/tensorflow/examples/blob/master/lite/examples/reinforcement_learning/ml)を参照してください。エージェントをトレーニングした後は、モデルを TFLite に変換し、Android アプリでデプロイします。

Android アプリでの実際のゲーム中、エージェントの番になると、エージェントは人間の対戦相手のボード状態 (下部のボード) を確認します。このボード状態には、以前の成功と失敗 (当たりと外れ) に関する情報が含まれています。そして、トレーニング済みのモデルを使用して、次に狙う場所を予測し、人間の対戦相手よりも早くゲームを終わらせることができるようにします。

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
    <td rowspan="2">       <a href="https://github.com/tensorflow/examples/blob/master/lite/examples/reinforcement_learning/android/app/src/main/assets/planestrike.tflite">方策勾配</a> </td>
    <td rowspan="2">       84 Kb</td>
    <td>Pixel 3 (Android 10)</td>
    <td>0.01ms*</td>
  </tr>
   <tr>
     <td>Pixel 4 (Android 10)</td>
    <td>0.01ms*</td>
  </tr>
</table>

* 1 つのスレッドを使用。

## 入力

モデルでは、(1, 8, 8) の 3-D `float32` テンソルをボード状態として入力できます。

## 出力

モデルは、64 の考えられる候補それぞれに対して、形状 (1,64) の 2-D `float32` テンソルを予測スコアとして返します。

## モデルのトレーニング

[トレーニングコード](https://github.com/tensorflow/examples/blob/master/lite/examples/reinforcement_learning/ml)で `BOARD_SIZE` パラメータを変更すると、大きい/小さいボードの独自のモデルをトレーニングできます。
