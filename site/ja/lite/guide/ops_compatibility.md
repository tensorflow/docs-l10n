# TensorFlow Lite と TensorFlow 演算子の互換性

TensorFlow Lite は、一般的な推論モデルで使用される多数の TensorFlow 演算をサポートしています。演算は TensorFlow Lite Optimizing Converter で処理されるため、サポートされている演算が TensorFlow Lite の相当する演算にマッピングされる前に、省略または融合されることがあります。

TensorFlow Lite のビルトイン演算子ライブラリがサポートする TensorFlow 演算子は制限されているため、すべてのモデルが互換しているわけではありません。サポートされている演算であっても、パフォーマンスの理由により、非常に特定的な使用パターンが期待されることがあります。TensorFlow Lite の将来のリリースでは、演算のサポート範囲を拡大したいと考えています。

TensorFlow Lite で使用できる TensorFlow モデルを構築する方法を理解する最善の方法は、このプロセスで課せられる制限とともに演算子がどのように変換されて最適化されるのかをよく考察することです。

## サポートされている型

Most TensorFlow Lite operations target both floating-point (`float32`) and quantized (`uint8`, `int8`) inference, but many ops do not yet for other types like `tf.float16` and strings.

浮動小数点数モデルと量子化モデルの間には、異なるバージョンの演算を使用するというだけでなく、ほかにも変換方法に違いがあります。量子化変換には、テンソルのダイナミックレンジ情報が必要です。これには、モデルのトレーニング中に「偽量子化」が必要で、較正データセット経由でレンジ情報を取得するか、「その場で」レンジ推測を行う必要があります。[量子化](../performance/model_optimization.md)をご覧ください。

## サポートされている演算と制約

TensorFlow Lite は、いくつかの制約を伴う TensorFlow のサブセットをサポートしています。演算と制限の全リストについては、[TF Lite 演算子のページ](https://www.tensorflow.org/mlir/tfl_ops)をご覧ください。

## 明瞭な変換、定数畳み込み、および融合

多くの TensorFlow 演算は、直接相当する演算がない場合でも、TensorFlow Lite によって処理することができます。これは、単純にグラフから除外して（`tf.identity`）テンソルで置換（`tf.placeholder`）できる演算か、より複雑な演算に融合（`tf.nn.bias_add`）できる演算の場合です。一部のサポートされている演算であっても、これらのプロセスの一部で除外されてしまうことがあります。

次は、完全なリストではありませんが、一般的にグラフから除外される TensorFlow 演算です。

- `tf.add`
- `tf.check_numerics`
- `tf.constant`
- `tf.div`
- `tf.divide`
- `tf.fake_quant_with_min_max_args`
- `tf.fake_quant_with_min_max_vars`
- `tf.identity`
- `tf.maximum`
- `tf.minimum`
- `tf.multiply`
- `tf.no_op`
- `tf.placeholder`
- `tf.placeholder_with_default`
- `tf.realdiv`
- `tf.reduce_max`
- `tf.reduce_min`
- `tf.reduce_sum`
- `tf.rsqrt`
- `tf.shape`
- `tf.sqrt`
- `tf.square`
- `tf.subtract`
- `tf.tile`
- `tf.nn.batch_norm_with_global_normalization`
- `tf.nn.bias_add`
- `tf.nn.fused_batch_norm`
- `tf.nn.relu`
- `tf.nn.relu6`

注意: こういった演算の多くには TensorFlow Lite 相当の演算がないため、省略または融合されない場合、対応するモデルには互換性が与えられません。

## 実験的な演算

次の TensorFlow Lite 演算は存在しますが、カスタムモデルにはまだ使用できません。

- `CALL`
- `CONCAT_EMBEDDINGS`
- `CUSTOM`
- `EMBEDDING_LOOKUP_SPARSE`
- `HASHTABLE_LOOKUP`
- `LSH_PROJECTION`
- `SKIP_GRAM`
- `SVDF`
