# 訓練後の量子化

訓練後の量子化は、わずかなモデルの精度の低下を伴いますが、モデルの大きさを削減することができ、さらにCPUとハードウェアアクセラレータのレイテンシーを改善する変換手法です。
TensorFlow Lite 形式に変換するときに、こと、訓練済みの浮動小数点数の TensorFlow モデルを使うこれらの手法を実行できます。

注意: このページの関数はTensorFlow 1.15 以上が必要です。

### 最適化オプション

選択可能な訓練後の量子化のいくつかのオプションがあります。
これは選択肢の概要の一覧表とその効果です。


| 手法                      | 効果                  | ハードウェア            |
| ------------------------- | ------------------------- | ------------------- |
| ダイナミックレンジ         | 4倍小型化, 2～3倍高速化、精度 | CPU                 |
: 量子化                    : accuracy                  :                     :
| 整数量子化                | 4倍小型化、3倍以上高速化   | CPU、Edge TPU、など |
| 半精度浮動小数点数量子化   | 2倍小型化, 潜在的なGPU | CPU/GPU             |
:                           : アクセラレーション              :                     :

この決定木は、どの訓練後の量子化方法があなたのユースケースに最適化を決める手助けになるでしょう。

![post-training optimization options](images/optimization.jpg)

Alternatively, you might achieve higher accuracy if you perform
もう1つの方法として、
[quantization-aware training](
https://github.com/tensorflow/tensorflow/tree/r1.14/tensorflow/contrib/quantize) 
を実行するのであれば、より高い精度を実現できるかもしれません。
しかし、そうするには、偽の量子化ノードを追加するために、モデルをいくつか修正する必要があります。
なお、このページの訓練後の量子化手法は、すでにある訓練済みのモデルを使用します。

### ダイナミックレンジの量子化

訓練後の量子化のもっとも単純な形式は、静的に重みのみを浮動小数点数から8ビット精度に量子化します。
この手法は、
[TensorFlow Lite 変換器](../convert/) のオプションとして利用できます。

```
import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant_model = converter.convert()
```

推論時に、重みは8ビット精度から浮動小数点数に変換され、浮動小数点数カーネルを使って計算されます。
この変換は一度だけおこなわれ、レイテンシーを減らすためにキャッシュされます。

レイテンシーをさらに改善するために、"ダイナミックレンジ" 演算子は、8ビットの範囲に基づき動的に活性化を量子化し、また8ビット重みと活性化を用いて計算を実施します。
この量子化は、レイテンシーを完全な固定小数点数の推論に近づけます。
しかし、出力はまだ浮動小数点数を用いて保持されているので、ダイナミックレンジ演算子による高速化度合いは完全な固定小数点数よりも小さいです。
ダイナミックレンジ演算子はネットワーク内の多くの数値計算演算子に適用可能です。

*  [tf.contrib.layers.fully_connected](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/fully_connected)
*  [tf.nn.conv2d](https://www.tensorflow.org/api_docs/python/tf/nn/conv2d)
*  [tf.nn.embedding_lookup](https://www.tensorflow.org/api_docs/python/tf/nn/embedding_lookup)
*  [BasicRNN](https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/BasicRNNCell)
*  [tf.nn.bidirectional_dynamic_rnn for BasicRNNCell type](https://www.tensorflow.org/api_docs/python/tf/nn/bidirectional_dynamic_rnn)
*  [tf.nn.dynamic_rnn for LSTM and BasicRNN Cell types](https://www.tensorflow.org/api_docs/python/tf/nn/dynamic_rnn)

### 重みと活性化の完全な整数量子化

モデル内のすべての計算が量子化を確実に行うことで、
さらにレイテンシー改善、ピーク時のメモリ使用量を削減し、
整数演算のみに対応したハードウェア・アクセラレータを利用できるようになります。

こうするためには、代表的なデータセットを与えることによって、活性化と入力の範囲を計測する必要があります。
簡単に入力データ生成器を作って、それを変換器への入力とすることができます。
例えば、

```
import tensorflow as tf

def representative_dataset_gen():
  for _ in range(num_calibration_steps):
    # Get sample input data as a numpy array in a method of your choosing.
    # あなたの選択した関数内で、サンプル入力をnumpy配列として取得する
    yield [input]

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
tflite_quant_model = converter.convert()
```

結果として得られるモデルは完全に量子化されているべきですが、
量子化の実装を持たないいくつの演算子は浮動小数点数として残されます。
これは変換を簡単に行えますが、
モデルは完全な整数量子化を必要とするアクセラレータと互換性がないでしょう。

そのうえ、モデルは便宜上まだ浮動小数点数の入力と出力を使用します。

いくつかのアクセラレータ(Coral Edge TPUのような)と互換性を保つためには、
すべての演算子に完全な整数化量子化を強制し、
変換する前に以下の行を追加することで整数の入力と出力を使用することができます。

```
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
```

最初の行は、もし現在は量子化できない演算に遭遇したら、変換器にエラーを投げさせます。

注意: `target_spec.supported_ops` は以前はPython APIでは `target_ops` でした。

### 重みの半精度浮動小数点数量子化

重みを、16ビット浮動小数点数のIEEE標準である半精度浮動小数点数に量子化することで
浮動小数点数モデルの大きさを削減することができます。

* モデルの大きさを半分にまで削減できます(すべての重みがオリジナルのサイズの半分になるので)
* 最小限の精度劣化
* いくつの委譲(たとえば、GPUへの委譲)は半精度浮動小数点数データで直接演算でき、結果として単精度浮動小数点数の計算より速く実行されます。

もし最大の性能を必要とするのであれば、この量子化は良い選択ではないかもしれません
(その場合は、固定小数点数への量子化の方がより良いでしょう)。
重みの半精度固定小数点数の量子化を有効にするには、
上記の "DEFAULT" 最適化を明記し、それから半精度浮動小数点数が target_spec に対してサポートされた型であることを明記します。

```
import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.lite.constants.FLOAT16]
tflite_quant_model = converter.convert()
```

デフォルトでは、半精度浮動小数点数の量子化モデルは、CPUで実行するときに重みの値を単精度浮動小数点数に"逆量子化する"でしょう。
GPU委譲はこの逆量子化は行わないでしょう、その理由は半精度浮動小数点数のままで演算できるからです。

### モデル精度

重みが訓練後に量子化されるので、特に小さなネットワークでは、精度劣化が発生するかもしれません。
訓練前に完全に量子化されたモデルは、個別のネットワークごとに
[TensorFlow Lite モデル・リポジトリ](../models/) で提供されています。
量子化後のモデルの精度を検査し、精度劣化が許容範囲内であるか検証することが大切です。
評価するツールは
[TensorFlow Lite モデル精度](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/tools/accuracy/ilsvrc/README.md){:.external}.
にあります。

精度劣化が大きい場合には、
[量子化を考慮した訓練](https://github.com/tensorflow/tensorflow/tree/r1.13/tensorflow/contrib/quantize){:.external}.
を使用することを検討してください。

### 量子化されたテンソルの表現

8ビット量子化は浮動小数点数を以下の式で近似します。
`real_value = (int8_value - zero_point) * scale`.

その表現はおおきく2つの部分があります。

* ゼロ点は0に等しく、-127以上127以下の8ビットの2の補数で表現された軸毎(つまりチャンネル毎)、あるいはテンソルごとの重み

* ゼロ点は-128以上127以下のどこかにあり、-128以上127以下の8ビットの2の補数で表現されたテンソルごとの活性化と入力

量子化スキームの詳細は、
[quantization spec](./quantization_spec.md) を見てください。
TensorFlow Lite の委譲インターフェースにつながりたいハードウェアベンダーは、
そこで説明した量子化スキームを実装することが推奨されています。
