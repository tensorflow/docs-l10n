# API の更新 <a name="api_updates"></a>

このページでは、TensorFlow 2.x の `tf.lite.TFLiteConverter` [Python API](index.md) に適用された更新に関する情報を提供しています。

注意: この変更で何らかの懸念事項が生じた場合は、[GitHub 課題](https://github.com/tensorflow/tensorflow/issues/new?template=60-tflite-converter-issue.md)を提出してください。

- TensorFlow 2.3

    - `inference_input_type` と `inference_output_type` 属性を使用する整数の入力型/出力型の量子化モデルのサポート（以前は浮動小数点のみをサポート）。こちらの[使用例](../performance/post_training_quantization.md#integer_only)をご覧ください。
    - 動的な次元によるモデルの変換とサイズ変更のサポート
    - 16 ビットアクティベーションと 8 ビットの重みによる量子化モードを実験的に追加

- TensorFlow 2.2

    - デフォルトで、機械学習向けの Google の最先端コンパイラテクノロジーである [MLIR ベース変換](https://mlir.llvm.org/) を使用。これにより、Mask R-CNN や Mobile BERT などの新しいモデルのクラスの変換が可能になり、Functional 制御フローを伴うモデルがサポートされています。

- TensorFlow 2.0 と TensorFlow 1.x の比較

    - `target_ops` 属性の名前が `target_spec.supported_ops` に変更されました。
    - 次の属性が削除されています。
        - *量子化*: `inference_type`、`quantized_input_stats`、`post_training_quantize`、`default_ranges_stats`、`reorder_across_fake_quant`、`change_concat_input_ranges`、`get_input_arrays()`。代わりに、`tf.keras` API で[量子化を意識したトレーニング](https://www.tensorflow.org/model_optimization/guide/quantization/training)がサポートされ、[ポストトレーニング量子化](../performance/post_training_quantization.md)に使用する属性数が少なくなっています。
        - *視覚化*: `output_format`、`dump_graphviz_dir`、`dump_graphviz_video`。代わりに、TensorFlow Lite モデルの視覚化には、[visualize.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/tools/visualize.py) を使用する方法をお勧めします。
        - *凍結グラフ*: `drop_control_dependency`。TensorFlow 2.x では凍結グラフはサポートされていません。
    - `tf.lite.toco_convert` や `tf.lite.TocoConverter` などのその他のコンバータ API が削除されています。
    - `tf.lite.OpHint` や `tf.lite.constants` などのその他の関連 API が削除されています（`tf.lite.constants.*` 型は、`tf.*` の TensorFlow データ型にマッピングされ、重複が減らされています）。
