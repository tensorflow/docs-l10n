# 訓練後の量子化

訓練後の量子化は、わずかなモデルの精度の低下を伴いますが、
モデルの大きさを削減することができ、さらにCPUとハードウェアアクセラレータのレイテンシーを改善する変換手法です。
[TensorFlow Lite Converter](../convert/)
を使って TensorFlow Lite 形式に変換するときに、訓練済みの浮動小数点数の TensorFlow モデルを使ってこれらの手法を実行できます。

注意: このページの関数はTensorFlow 1.15 以上が必要です。

### 最適化手法

訓練後の量子化にはいくつか選択肢があります。
これは選択肢の概要の一覧表とその効果です。

| 手法                      | 効果                      | ハードウェア            |
| ------------------------- | ------------------------- | ------------------- |
| ダイナミックレンジ         | 4倍小型化, 2～3倍高速化   | CPU                 |
: 量子化                    :                          :                     :
| 完全な整数                | 4倍小型化、3倍以上高速化   | CPU、Edge TPU       |
: 量子化                    :                           : マイクロコントローラ :
| 半精度浮動小数点数量子化   | 2倍小型化, GPU    | CPU, GPU             |
:                           : アクセラレーションの可能性 :                     :

この決定木は、どの訓練後の量子化方法があなたのユースケースに最適であるかを決めることへの手助けになるでしょう:

![post-training optimization options](images/optimization.jpg)

### ダイナミックレンジの量子化

訓練後の量子化のもっとも単純な形式は静的に、重みのみを浮動小数点数から8ビット精度の整数に量子化します。

<pre>
import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
<b>converter.optimizations = [tf.lite.Optimize.DEFAULT]</b>
tflite_quant_model = converter.convert()
</pre>

推論時に、重みは8ビット精度から浮動小数点数に変換され、浮動小数点数カーネルを使って計算されます。
この変換は一度だけおこなわれ、レイテンシーを減らすためにキャッシュされます。

レイテンシーをさらに改善するために、"ダイナミックレンジ" 演算子は、8ビットに合わせた範囲に基づき動的に活性化を量子化し、8ビット重みと活性化を用いて計算を実施します。
この最適化は、完全な固定小数点数の推論に近いレイテンシーを提供します。
しかし、出力は依然として浮動小数点数を用いて保持されているので、ダイナミックレンジ演算子による高速化度合いは、完全な固定小数点数よりも小さいです。
ダイナミックレンジ演算子は、ネットワーク内の多くの数値計算演算子に適用可能です:

*   `tf.keras.layers.Dense`
*   `tf.keras.layers.Conv2D`
*   `tf.keras.layers.LSTM`
*   `tf.nn.embedding_lookup`
*   `tf.compat.v1.nn.rnn_cell.BasicRNNCell`
*   `tf.compat.v1.nn.bidirectional_dynamic_rnn`
*   `tf.compat.v1.nn.dynamic_rnn`

### 完全な整数量子化

モデル内のすべての計算の量子化を実施することで、
さらにレイテンシー改善したり、ピーク時のメモリ使用量を削減したり、
整数演算のみに対応したハードウェアデバイスやアクセラレータを利用できるようになります。

こうするためにはサンプル入力データを変換器に与え、活性化と入力の範囲を計測する必要があります。
以下のコードで使用されている `representative_dataset_gen()` 関数を参照してください

#### 浮動小数点数の代替をともなう整数 (元々の浮動小数点数の入力と出力を使用する)

完全に整数量子化されたモデルを使うが、整数に対応した(整数への変換が円滑にできることが確かめれらた)実装がない浮動小数点演算を使用するときには、以下のステップを使ってください。

<pre>
import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
<b>converter.optimizations = [tf.lite.Optimize.DEFAULT]
def representative_dataset_gen():
  for _ in range(num_calibration_steps):
    # あなたの選択した関数内で、サンプル入力をnumpy配列として取得する
    yield [input]
converter.representative_dataset = representative_dataset_gen</b>
tflite_quant_model = converter.convert()
</pre>

注意: これは、整数のみのデバイス(8ビットマイクロコントローラのような)やアクセラレータ(Coral Edge TPUのような)と互換性がないでしょう。
推論時の利便性を考慮し、元の浮動小数点数のみのモデルとおなじインターフェースを保持するために、入力と出力は浮動小数点数のままです。

#### 整数のみ

*これは
[マイクロコントローラ向け TensorFlow Lite](https://www.tensorflow.org/lite/microcontrollers)
と [Coral Edge TPUs](https://coral.ai/) に対する共通のユースケースです*

さらに、整数のみのデバイス(8ビットマイクロコントローラのような)やアクセラレータ(Coral Edge TPUのような)と互換性を保つためには、
以下のステップにしたがって、入力と出力を含むすべての演算子に対して完全な整数量子化を実施することができます。

<pre>
import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
def representative_dataset_gen():
  for _ in range(num_calibration_steps):
    # あなたが選択した関数内で、サンプル入力データをnumpy配列として取得する。
    yield [input]
converter.representative_dataset = representative_dataset_gen
<b>converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]</b>
<b>converter.inference_input_type = tf.int8</b>  # or tf.uint8
<b>converter.inference_output_type = tf.int8</b>  # or tf.uint8
tflite_quant_model = converter.convert()
</pre>

注意: 現在は量子化できない演算があった場合は、変換器はエラーを投げるでしょう。

### 半精度浮動小数点数量子化

16ビット浮動小数点数のIEEE標準である半精度浮動小数点数に重みを量子化することで、
浮動小数点数モデルの大きさを削減することができます。重みの半精度浮動小数点数量子化を行うには、以下のステップを実施してください。

<pre>
import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
<b>converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.lite.constants.FLOAT16]</b>
tflite_quant_model = converter.convert()
</pre>

この量子化の利点は以下のとおりです。

* モデルの大きさを半分にまで削減します(すべての重みがオリジナルのサイズの半分になるので)
* 最小限の精度劣化
* いくつのデリゲート(たとえば、GPUへのデリゲート)は、半精度浮動小数点数データで直接演算でき、結果として単精度浮動小数点数の計算より速く実行されます。

この量子化の欠点は以下のとおりです。

* 最大の性能を必要とするのであれば、この量子化はよい選択ではありません (その場合は、固定小数点数への量子化の方がより良いでしょう)。
* デフォルトでは、半精度浮動小数点数の量子化モデルは、CPUで実行するときに重みの値を単精度浮動小数点数に"逆量子化する"でしょう。
(GPUデリゲートはこの逆量子化は行わないでしょう、その理由は半精度浮動小数点数のままで演算できるからです。)

### モデル精度

重みが訓練後に量子化されるので、特に小さなネットワークでは、精度劣化が発生するかもしれません。
訓練前に完全に量子化されたモデルは、個別のネットワークごとに
[TensorFlow Lite モデル・リポジトリ](../models/) で提供されています。
量子化後のモデルの精度を検査し、精度劣化が許容範囲内であるか検証することが大切です。
評価するツールは
[TensorFlow Lite モデル精度](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/tools/accuracy/ilsvrc/README.md){:.external}.
にあります。

ほかの方法として、精度劣化が大きい場合には、
[量子化を考慮した訓練](https://github.com/tensorflow/tensorflow/tree/r1.13/tensorflow/contrib/quantize){:.external}.
を使用することを検討してください。
しかし、そうすることは、モデルの訓練時に偽の量子化ノードを追加するために修正を行う必要があります。なお、このページの訓練後の量子化手法は、既存の訓練済みモデルを使用します。

### 量子化されたテンソルの表現

8ビット量子化は浮動小数点数を以下の式で近似します。

$$real\_value = (int8\_value - zero\_point) \times scale$$

その表現はおおきく2つの部分があります。

* ゼロ点が0に等しく、-127以上127以下の8ビットの2の補数で表現された軸毎(つまりチャンネル毎)、あるいはテンソルごとの重み

* ゼロ点が-128以上127以下のどこかにあり、-128以上127以下の8ビットの2の補数で表現されたテンソルごとの活性化と入力

量子化スキームの詳細は、
[quantization spec](./quantization_spec.md) を見てください。
TensorFlow Lite のデリゲート・インターフェースに接続したいハードウェアベンダーは、
そこで説明されている量子化スキームを実装することが推奨されています。
