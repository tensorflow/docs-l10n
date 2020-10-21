# TensorFlow モデルの最適化の基礎

## 1. タスクに最適なモデルを選択する

タスクによっては、モデルの複雑さとサイズの間でトレードオフを行う必要があります。高い精度が要求されるタスクの場合は、大規模かつ複雑なモデルが必要になる可能性があります。 さほど精度を必要としないタスクの場合は、小規模モデルを使用した方がディスク容量とメモリの使用量が少なくて済むだけでなく、一般的に高速でエネルギー効率が良くなります。

## 2. 事前に最適化されたモデル

既存の [TensorFlow Lite の最適化済みモデル](https://www.tensorflow.org/lite/models)によってアプリケーションが必要とする効率性を得られるかどうかを確認します。

## 3. ポストトレーニングのツール

アプリケーションでトレーニング済みのモデルを使用できない場合は、[TensorFlow Lite への変換](https://www.tensorflow.org/lite/convert)中に、[TensorFlow Lite のポストトレーニング量子化ツール](./quantization/post_training)を使用してみてください。すでにトレーニングされた TensorFlow モデルを最適化することができます。

詳細は、[ポストトレーニング量子化のチュートリアル](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/performance/post_training_quant.ipynb)をご覧ください。

## 次のステップ: トレーニング時のツール

上記の簡単なソリューションではニーズを満たせない場合は、トレーニング時間最適化手法が必要かもしれません。トレーニング時間ツールを使って[さらに最適化](optimize_further.md)し、掘り下げてみましょう。
