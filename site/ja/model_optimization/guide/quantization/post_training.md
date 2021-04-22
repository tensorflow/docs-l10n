# ポストトレーニング量子化

ポストトレーニング量子化には、モデルの精度に影響をほとんど与えることなく、CPU とハードウェアアクセラレータのレイテンシ、処理、電力、およびモデルサイズを小さくする一般的なテクニックが含まれます。こういったテクニックは、トレーニング済みの浮動小数点 TensorFlow モデルに実行することができ、TensorFlow Lite 変換中に適用することができます。[TensorFlow Lite コンバータ](https://www.tensorflow.org/lite/convert/)のオプションとして、有効化することができます。

早速、エンドツーエンドの例を確認するには、次のチュートリアルをご覧ください。

- [ポストトレーニングのダイナミックレンジ量子化](https://www.tensorflow.org/lite/performance/post_training_quant)
- [ポストトレーニングの完全な整数量子化](https://www.tensorflow.org/lite/performance/post_training_integer_quant)
- [ポストトレーニングの float16 量子化](https://www.tensorflow.org/lite/performance/post_training_float16_quant)

## 重みを量子化する

重みは、16 ビット浮動小数点または 8 ビット整数など、精度を落とした型に変換することができます。通常、GPU アクセラレーションでは 16 ビット浮動小数点、CPU 実行では 8 ビット整数をお勧めします。

たとえば、8 ビット整数の重み量子化を指定するには、次のように行います。

```
import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant_model = converter.convert()
```

推論時、最も重要な部分は、浮動小数点ではなく 8 ビットで計算されます。以下の重みとアクティベーションの両方を量子化することに比べて、推論時のパフォーマンスのオーバーヘッドがいくつかあります。

詳細は、TensorFlow Lite [ポストトレーニング量子化](https://www.tensorflow.org/lite/performance/post_training_quantization)ガイドをご覧ください。

## 重みとアクティベーションの完全な整数量子化

重みとアクティベーションの両方が量子化されていることを確認することで、レイテンシ、処理、電力使用量を改善し、整数のみのハードウェアアクセラレータにアクセスできます。これには小さな代表データセットが必要です。

```
import tensorflow as tf

def representative_dataset_gen():
  for _ in range(num_calibration_steps):
    # Get sample input data as a numpy array in a method of your choosing.
    yield [input]

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
tflite_quant_model = converter.convert()
```

結果として得られるモデルは、便宜上、浮動小数点入力と出力を受け取ります。

詳細は、TensorFlow Lite [ポストトレーニング量子化](https://www.tensorflow.org/lite/performance/post_training_quantization#full_integer_quantization_of_weights_and_activations)ガイドをご覧ください。
