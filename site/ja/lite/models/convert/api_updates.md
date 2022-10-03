# API の更新 <a name="api_updates"></a>

このページでは、TensorFlow 2.x の `tf.lite.TFLiteConverter` [Python API](index.md) に適用された更新に関する情報を提供しています。

注意: この変更で何らかの懸念事項が生じた場合は、[GitHub 課題](https://github.com/tensorflow/tensorflow/issues/new?template=60-tflite-converter-issue.md)を提出してください。

- TensorFlow 2.3

    - Support integer (previously, only float) input/output type for integer quantized models using the new `inference_input_type` and `inference_output_type` attributes. Refer to this [example usage](../../performance/post_training_quantization.md#integer_only).
    - 動的な次元によるモデルの変換とサイズ変更のサポート
    - 16 ビットアクティベーションと 8 ビットの重みによる量子化モードを実験的に追加

- TensorFlow 2.2

    - デフォルトで、機械学習向けの Google の最先端コンパイラテクノロジーである [MLIR ベース変換](https://mlir.llvm.org/) を使用。これにより、Mask R-CNN や Mobile BERT などの新しいモデルのクラスの変換が可能になり、Functional 制御フローを伴うモデルがサポートされています。

- TensorFlow 2.0 と TensorFlow 1.x の比較

    - `target_ops` 属性の名前が `target_spec.supported_ops` に変更されました。
    - 次の属性が削除されています。
        - *quantization*: `inference_type`, `quantized_input_stats`, `post_training_quantize`, `default_ranges_stats`, `reorder_across_fake_quant`, `change_concat_input_ranges`, `get_input_arrays()`. Instead, [quantize aware training](https://www.tensorflow.org/model_optimization/guide/quantization/training) is supported through the `tf.keras` API and [post training quantization](../../performance/post_training_quantization.md) uses fewer attributes.
        - *視覚化*: `output_format`、`dump_graphviz_dir`、`dump_graphviz_video`。代わりに、TensorFlow Lite モデルの視覚化には、[visualize.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/tools/visualize.py) を使用する方法をお勧めします。
        - *凍結グラフ*: `drop_control_dependency`。TensorFlow 2.x では凍結グラフはサポートされていません。
    - `tf.lite.toco_convert` や `tf.lite.TocoConverter` などのその他のコンバータ API が削除されています。
    - `tf.lite.OpHint` や `tf.lite.constants` などのその他の関連 API が削除されています（`tf.lite.constants.*` 型は、`tf.*` の TensorFlow データ型にマッピングされ、重複が減らされています）。
