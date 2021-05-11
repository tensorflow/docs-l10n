# トレーニング後の量子化

トレーニング後の量子化は、わずかなモデルの精度の低下を伴いますが、 モデルの大きさを削減することができ、さらに CPU とハードウェアアクセラレータのレイテンシを改善する変換手法です。 [TensorFlow Lite Converter](../convert/) を使って TensorFlow Lite 形式に変換する場合、トレーニング済みの浮動小数点数の TensorFlow モデルを使ってこれらの手法を実行できます。

注：このページの関数は TensorFlow 1.15 以上が必要です。

### 最適化手法

トレーニング後の量子化にはいくつか選択肢があります。 これは選択肢の概要の一覧表とその効果です。

手法 | 効果 | ハードウェア
--- | --- | ---
ダイナミックレンジ | 4倍小型化、2～3倍高速化 | CPU
: 量子化                    :                          :                     : |  |
完全な整数 | 4倍小型化、3倍以上高速化 | CPU、Edge TPU、
: 量子化                    :                           : マイクロコントローラ : |  |
半精度浮動小数点数量子化 | 2倍小型化、GPU | CPU、GPU
:                           : アクセラレーションの可能性 :                     : |  |

この決定木は、ユースケースに最適なトレーニング後の量子化手法を選択するのに役立ちます。

![post-training optimization options](images/optimization.jpg)

### ダイナミックレンジの量子化

トレーニング後の量子化の最も単純な形式は静的に、重みのみを浮動小数点数から 8 ビット精度の整数に量子化します。

<pre>
import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
&lt;b&gt;converter.optimizations = [tf.lite.Optimize.DEFAULT]&lt;/b&gt;
tflite_quant_model = converter.convert()
</pre>

推論時に、重みは 8 ビット精度から浮動小数点数に変換され、浮動小数点数カーネルを使って計算されます。この変換は一度だけおこなわれ、レイテンシを低減ためにキャッシュされます。

レイテンシをさらに改善するために、「ダイナミックレンジ」演算子は、活性化関数をその値域に基づいて 8 ビットへ動的に量子化し、8 ビット化された重みと活性化関数を用いて計算を実施します。 この最適化は、完全な固定小数点数の推論に近いレイテンシを提供します。しかし、出力は依然として浮動小数点数を用いて保持されているので、ダイナミックレンジ演算子による高速化度合いは、完全な固定小数点数よりも小さいです。ダイナミックレンジ演算子は、ネットワーク内の多くの数値計算演算子に適用可能です。

### 完全な整数量子化

モデル内のすべての計算の量子化を実施することで、 さらにレイテンシを改善したり、ピーク時のメモリ使用量を削減したり、 整数演算のみに対応したハードウェアデバイスやアクセラレータを利用できるようになります。

こうするためにはサンプルの入力データを変換器に与え、活性化関数の入力と出力の範囲を計測する必要があります。 以下のコードで使用されている `representative_dataset_gen()` 関数を参照してください

<pre>def representative_dataset():
  for data in tf.data.Dataset.from_tensor_slices((images)).batch(1).take(100):
    yield [tf.dtypes.cast(data, tf.float32)]
</pre>

テストのためには、次のようにダミーデータセットを使用できます。

<pre>def representative_dataset():
    for _ in range(100):
      data = np.random.rand(1, 244, 244, 3)
      yield [data.astype(np.float32)]
 </pre>

#### 浮動小数点数の代替をともなう整数 (元々の浮動小数点数の入力と出力を使用する)

モデルを完全に整数量子化し、整数の実装がない場合に浮動小数点数の演算子を使用するには、次の手順に従います（変換がスムーズに行われます）。

<pre>import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
&lt;b&gt;converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset&lt;/b&gt;
tflite_quant_model = converter.convert()
</pre>

注：これは、整数のみのデバイス(8 ビットマイクロコントローラのような)やアクセラレータ(Coral Edge TPU のような)と互換性がありません。 推論時の利便性を考慮し、元の浮動小数点数のみのモデルとおなじインターフェイスを保持するために、入力と出力は浮動小数点数のままです。

#### 整数のみ

*整数のみのモデルの作成は、[マイクロコントローラ向け TensorFlow Lite](https://www.tensorflow.org/lite/microcontrollers) および [Coral Edge TPUs](https://coral.ai/) の一般的な使用例です。*

注：TensorFlow 2.3.0 以降、`instance_input_type`属性と`inference_output_type `属性がサポートされています。

さらに、整数のみのデバイス (8 ビットマイクロコントローラのような)やアクセラレータ (Coral Edge TPU など)と互換性を保つためには、 以下の手順に従い、入力と出力を含むすべての演算子に対して完全な整数量子化を実施することができます。

<pre>import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
&lt;b&gt;converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]&lt;/b&gt;
&lt;b&gt;converter.inference_input_type = tf.int8&lt;/b&gt;  # or tf.uint8
&lt;b&gt;converter.inference_output_type = tf.int8&lt;/b&gt;  # or tf.uint8
tflite_quant_model = converter.convert()
</pre>

注：現在量子化できない演算がある場合、コンバータは、エラーをスローします。

### 半精度浮動小数点数量子化

重みを半精度浮動小数点（16 ビット浮動小数点数の IEEE 標準）に量子化することにより、浮動小数点モデルのサイズを縮小できます。 重みの半精度浮動小数点量子化を有効にするには、次の手順に従います。

<pre>import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
&lt;b&gt;converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]&lt;/b&gt;
tflite_quant_model = converter.convert()
</pre>

この量子化の利点は以下のとおりです。

- 最小限の精度劣化
- 精度の低下は最小限に抑えられます。
- 半精度浮動小数点データを直接演算できる一部のデリゲート（GPU デリゲートなど）をサポートしているため、32 ビットの単精度浮動小数点の計算よりも高速に実行できます。

この量子化の欠点は以下のとおりです。

- レイテンシは固定小数点演算への量子化ほど減少しません。
- モデルの大きさを半分にまで削減します(すべての重みがオリジナルのサイズの半分になるので)

### モデルの精度

重みがトレーニング後に量子化されるので、特に小さなネットワークでは、精度劣化が発生する可能性があります。トレーニング前に完全に量子化されたモデルは、個別のネットワークごとに [TensorFlow Lite モデル・リポジトリ](../models/)で提供されています。 量子化後のモデルの精度を検査し、精度劣化が許容範囲内であるか検証することが大切です。評価するツールは [TensorFlow Lite モデル精度](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/tools/accuracy/ilsvrc/README.md){:.external}. にあります。

ほかの方法として、精度劣化が大きい場合には、 [量子化を考慮したトレーニング](https://www.tensorflow.org/model_optimization/guide/quantization/training)を使用することを検討してください。しかし、そうすることは、モデルのトレーニング時に偽の量子化ノードを追加するために修正を行う必要があります。なお、このページのトレーニング後の量子化手法は、既存のトレーニング済みモデルを使用します。

<pre>import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.representative_dataset = representative_dataset
&lt;b&gt;converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]&lt;/b&gt;
tflite_quant_model = converter.convert()
</pre>

8 ビット量子化は浮動小数点数を以下の式で近似します。

<pre>import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.representative_dataset = representative_dataset
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8,
&lt;b&gt;tf.lite.OpsSet.TFLITE_BUILTINS&lt;/b&gt;]
tflite_quant_model = converter.convert()
</pre>

$$real_value = (int8_value - zero_point) \times scale$$

その表現の欠点は以下のとおりです。

- 現在、最適化されたカーネル実装がないため、推論は 8 ビットの完全整数よりも著しく遅くなります。
- 現在、既存のハードウェアアクセラレータを使用する TF Lite デリゲートとは互換性がありません。

量子化スキーマの詳細は、 [量子化の仕様](./quantization_spec.md)をご覧ください。TensorFlow Lite のデリゲートインターフェイスに接続するハードウェアベンダーには、 そこに説明されている量子化スキーマを実装することが推奨されています。

この量子化モードのチュートリアルは、[こちら](post_training_integer_quant_16x8.ipynb)からご覧ください。

### 量子化されたテンソルの表現

重みはトレーニング後に量子化されるため、特に小規模なネットワークでは、精度が低下する可能性があります。トレーニング済みの完全に量子化されたモデルは、[TensorFlow Lite モデルリポジトリ](../models/)の特定のネットワーク用に提供されています。量子化されたモデルの精度をチェックして、精度の低下が許容範囲内にあることを確認することが重要です。[TensorFlow Lite モデルの精度](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/evaluation/tasks){:.external}を評価するためのツールがあります。

または、精度の低下が大きすぎる場合は、[量子化対応トレーニング](https://www.tensorflow.org/model_optimization/guide/quantization/training)の使用を検討してください。ただし、これを行うには、モデルトレーニング中に偽の量子化ノードを追加するために変更する必要がありますが、このページのトレーニング後の量子化手法では、既存の事前トレーニング済みモデルを使用します。

### 量子化されたテンソルの表現

8-bit quantization approximates floating point values using the following formula.

$$real_value = (int8_value - zero_point) \times scale$$

表現には以下の2つの主要な部分があります。

- 軸ごと（チャネルごと）またはテンソルごとの重みは、ゼロ点が 0 に等しく、範囲が [-127、127] の int8 の 2 の補数値で表されます。

- テンソルごとのアクティベーション/入力は、[-128, 127]の範囲のゼロ点、 [-127、127] の範囲の int8 の 2 の補数値で表されます。

量子化スキームの詳細については、[量子化仕様](./quantization_spec.md)を参照してください。TensorFlow Lite のデリゲートインターフェイスにプラグインするハードウェアベンダーには、ここに説明されている量子化スキームを実装することをお勧めします。
