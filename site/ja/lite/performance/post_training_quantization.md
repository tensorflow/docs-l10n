# トレーニング後の量子化

トレーニング後の量子化は、わずかなモデルの精度の低下を伴いますが、モデルの大きさを削減し、CPU とハードウェアアクセラレータのレイテンシを改善する変換手法です。[TensorFlow Lite Converter](../convert/) を使って TensorFlow Lite 形式に変換する場合、トレーニング済みの浮動小数点数の TensorFlow モデルを使ってこれらの手法を実行できます。

注意：このページの手法には TensorFlow 1.15 以上が必要です。

### 最適化手法

トレーニング後の量子化にはいくつか選択肢があります。以下は選択肢の概要の一覧表とその効果です。

手法 | 効果 | ハードウェア
--- | --- | ---
ダイナミックレンジ | 4倍小型化、2～3倍高速化 | CPU
: 量子化                    :                          :                     : |  |
完全な整数 | 4倍小型化、3倍以上高速化 | CPU、Edge TPU、
: 量子化                    :                           : マイクロコントローラ : |  |
半精度浮動小数点数の量子化 | 2倍小型化、GPU | CPU、GPU
:                           : アクセラレーション :                     : |  |

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

推論時に、重みは 8 ビット精度から浮動小数点数に変換され、浮動小数点数カーネルを使って計算されます。この変換は一度だけ行われ、レイテンシ低減のためにキャッシュされます。

レイテンシをさらに改善するために、「ダイナミックレンジ」演算子は、活性化関数をその値域に基づいて 8 ビットへ動的に量子化し、8 ビット化された重みと活性化関数を用いて計算を実施します。この最適化は、完全な固定小数点数の推論に近いレイテンシを提供します。しかし、出力は依然として浮動小数点数を用いて保持されているので、ダイナミックレンジ演算子による高速化の度合いは、完全な固定小数点数よりも小さくなります。ダイナミックレンジ演算子は、ネットワーク内の多くの数値計算演算子に適用可能です。

### 完全な整数量子化

モデル内のすべての計算の量子化を実施することで、さらにレイテンシを改善したり、ピーク時のメモリ使用量を削減したり、整数演算のみに対応したハードウェアデバイスやアクセラレータを利用できるようになります。

こうするためにはサンプルの入力データを変換器に与え、活性化関数の入力と出力の範囲を計測する必要があります。重みやバイアスなどの一定テンソルとは異なり、モデル入力、活性化（中間層の出力）、モデル出力などの可変テンソルは、いくつかの推論サイクルを実行しない限り、キャリブレーションできません。そのため、コンバータはそれらをキャリブレーションするために代表的なデータセットを必要とします。このデータセットには、トレーニングデータまたは検証データの小さなサブセット（約100〜500サンプル）を使用できます。以下のコードで使用されている `representative_dataset_gen()` 関数を参照してください。

From TensorFlow 2.7 version, you can specify the representative dataset through a [signature](/lite/guide/signatures) as the following example:

<pre>def representative_dataset():
  for data in dataset:
    yield {
      "image": data.image,
      "bias": data.bias,
    }
</pre>

If there are more than one signature in the given TensorFlow model, you can specify the multiple dataset by specifying the signature keys:

<pre>def representative_dataset():
  # Feed data set for the "encode" signature.
  for data in encode_signature_dataset:
    yield (
      "encode", {
        "image": data.image,
        "bias": data.bias,
      }
    )

  # Feed data set for the "decode" signature.
  for data in decode_signature_dataset:
    yield (
      "decode", {
        "image": data.image,
        "hint": data.hint,
      },
    )
</pre>

You can generate the representative dataset by providing an input tensor list:

<pre>def representative_dataset():
  for data in tf.data.Dataset.from_tensor_slices((images)).batch(1).take(100):
    yield [tf.dtypes.cast(data, tf.float32)]
</pre>

Since TensorFlow 2.7 version, we recommend using the signature-based approach over the input tensor list-based approach because the input tensor ordering can be easily flipped.

テストのためには、次のようにダミーデータセットを使用できます。

<pre>def representative_dataset():
    for _ in range(100):
      data = np.random.rand(1, 244, 244, 3)
      yield [data.astype(np.float32)]
 </pre>

#### 浮動小数点数の代替をともなう整数 (デフォルトの浮動小数点数の入力と出力を使用する)

モデルを完全に整数量子化し、整数の実装がない場合に浮動小数点数の演算子を使用するには、次の手順に従います（変換がスムーズに行われます）。

<pre>import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
&lt;b&gt;converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset&lt;/b&gt;
tflite_quant_model = converter.convert()
</pre>

注意：この`tflite_quant_model`は、整数のみのデバイス(8 ビットマイクロコントローラのような)やアクセラレータ(Coral Edge TPU のような)と互換性がありません。推論時の利便性を考慮し、元の浮動小数点数のみのモデルと同じインターフェイスを保持するために、入力と出力は浮動小数点数のままです。

#### 整数のみ

*整数のみのモデルの作成は、[マイクロコントローラ向け TensorFlow Lite](https://www.tensorflow.org/lite/microcontrollers) および [Coral Edge TPU](https://coral.ai/) の一般的な使用例です。*

注意：TensorFlow 2.3.0 以降、`instance_input_type`属性と`inference_output_type `属性がサポートされています。

さらに、整数のみのデバイス (8 ビットマイクロコントローラなど)やアクセラレータ (Coral Edge TPU など)との互換性を保つためには、以下の手順に従い、入力と出力を含むすべての演算子に対して完全な整数量子化を実施することができます。

<pre>import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
&lt;b&gt;converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]&lt;/b&gt;
&lt;b&gt;converter.inference_input_type = tf.int8&lt;/b&gt;  # or tf.uint8
&lt;b&gt;converter.inference_output_type = tf.int8&lt;/b&gt;  # or tf.uint8
tflite_quant_model = converter.convert()
</pre>

注意：現在量子化できない演算がある場合、コンバータは、エラーをスローします。

### 半精度浮動小数点数の量子化

重みを半精度浮動小数点（16 ビット浮動小数点数の IEEE 標準）に量子化することにより、浮動小数点モデルのサイズを縮小できます。重みの半精度浮動小数点量子化を有効にするには、次の手順に従います。

<pre>import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
&lt;b&gt;converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]&lt;/b&gt;
tflite_quant_model = converter.convert()
</pre>

この量子化の利点は以下のとおりです。

- モデルのサイズを最大半分に縮小します（すべての重みが元のサイズの半分になるため）。
- 精度の低下は最小限に抑えられます。
- 半精度浮動小数点データを直接演算できる一部のデリゲート（GPU デリゲートなど）をサポートしているため、単精度の浮動小数点の計算よりも高速に実行できます。

半精度浮動小数点数の量子化の欠点は以下のとおりです。

- レイテンシは固定小数点量子化ほど減少しません。
- デフォルトでは、半精度浮動小数点数の量子化モデルは、CPU で実行されると、重み値を単精度浮動小数点数に「逆量子化」します。（GPU デリゲートは半精度浮動小数点数のデータを演算できるため、この逆量子化を実行しないでください。）

### 整数のみ：8ビットの重みを使用した16ビットの活性化（実験的）

これは実験的な量子化スキームで、「整数のみ」のスキームに似ていますが、活性化は16ビットまでの範囲に基づいて量子化されます。重みは8ビット整数、バイアスは64ビット整数に量子化されます。これは16x8量子化と呼ばれます。

この量子化の主な利点は、モデルサイズをわずかに増やすだけで精度を大幅に向上できることです。

<pre>import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.representative_dataset = representative_dataset
&lt;b&gt;converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]&lt;/b&gt;
tflite_quant_model = converter.convert()
</pre>

モデル内の一部の演算子で16x8量子化がサポートされていない場合でも、モデルは量子化できます。その場合、サポートされていない演算子は浮動小数点に保持されます。これを可能にするには、target_spec に次のオプションを追加する必要があります。

<pre>import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.representative_dataset = representative_dataset
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8,
&lt;b&gt;tf.lite.OpsSet.TFLITE_BUILTINS&lt;/b&gt;]
tflite_quant_model = converter.convert()
</pre>

この量子化スキームにより*超解像、*ノイズキャンセリングやビームフォーミングなどのオーディオ信号処理、*画像のノイズ除去、*単一画像からの HDR 再構成などの使用例で精度が向上します。

この量子化の欠点は以下のとおりです。

- 現在、最適化されたカーネル実装がないため、推論は 8 ビットの完全整数よりも著しく遅くなります。
- 現在、既存のハードウェアアクセラレータを使用する TF Lite デリゲートとは互換性がありません。

注意：これは実験的機能です。

この量子化モードのチュートリアルは、[こちら](post_training_integer_quant_16x8.ipynb)からご覧ください。

### モデルの精度

重みはトレーニング後に量子化されるため、特に小規模なネットワークでは、精度が低下する可能性があります。トレーニング済みの完全に量子化されたモデルは、[TensorFlow Lite モデルリポジトリ](../models/)の特定のネットワーク用に提供されています。量子化されたモデルの精度をチェックして、精度の低下が許容範囲内にあることを確認することが重要です。[TensorFlow Lite モデルの精度](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/evaluation/tasks){:.external}を評価するためのツールがあります。

または、精度の劣化が大きすぎる場合は、[量子化対応トレーニング](https://www.tensorflow.org/model_optimization/guide/quantization/training)の使用を検討してください。ただし、これを行うには、モデルトレーニング中に偽の量子化ノードを追加するために変更する必要がありますが、このページのトレーニング後の量子化手法では、既存の事前トレーニング済みモデルを使用します。

### 量子化されたテンソルの表現

8 ビット量子化は、次の方程式により、浮動小数点値を概算します。

$$real_value = (int8_value - zero_point) \times scale$$

表現には以下の2つの主要な部分があります。

- 軸ごと（チャネルごと）またはテンソルごとの重みは、ゼロ点が 0 に等しく、範囲が [-127、127] の int8 の 2 の補数値で表されます。

- テンソルごとの活性化/入力は、[-128, 127]の範囲のゼロ点、 [-127、127] の範囲の int8 の 2 の補数値で表されます。

量子化スキームの詳細については、[量子化仕様](./quantization_spec.md)を参照してください。TensorFlow Lite のデリゲートインターフェイスにプラグインするハードウェアベンダーには、ここに説明されている量子化スキームを実装することをお勧めします。
