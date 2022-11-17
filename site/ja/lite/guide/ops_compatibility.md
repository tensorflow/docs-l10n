# TensorFlow Lite と TensorFlow 演算子の互換性

モデルで使用する機械学習（ML）演算子は、TensorFlow モデルを TensorFlow Lite 形式に変換するプロセスに影響を与える可能性があります。TensorFlow Lite コンバータは、一般的な推論モデルで使用される限られた数の TensorFlow 演算をサポートします。つまり、すべてのモデルが直接変換できるわけではありません。コンバータツールを使用すると、追加の演算子を含めることができますが、この方法でモデルを変換するには、モデルの実行に使用する TensorFlow Lite ランタイム環境を変更する必要もあります。これにより、[Google Play サービス](../android/play_services)などの標準ランタイムでのデプロイオプションを使用する能力が制限される可能性があります。

TensorFlow Lite コンバータは、モデル構造を分析し、直接サポートされている演算子と互換性を持たせるために最適化を適用するように設計されています。例えば、モデル内の ML 演算子に応じて、コンバータはそれらの演算子を[省略または融合](../models/convert/operation_fusion)して、それらの演算子を TensorFlow Lite の対応する演算子にマッピングする場合があります。

サポートされている演算であっても、パフォーマンス上の理由から、特定の使用パターンが予想される場合があります。 TensorFlow Lite で使用できる TensorFlow モデルを構築する方法を理解する最善の方法は、このプロセスで課せられる制限とともに演算がどのように変換され最適化されるのかをよく考察することです。

## サポートされている演算子

TensorFlow Lite 組み込み演算子は、TensorFlow コアライブラリの一部である演算子のサブセットです。TensorFlow モデルには、複合演算子またはユーザーが定義した新しい演算子の形式でカスタム演算子を含めることもできます。以下の図は、これらの演算子間の関係を示しています。

![TensorFlow operators](../images/convert/tf_operators_relationships.png)

この範囲の ML モデル演算子から、変換プロセスでサポートされるモデルには 3 つのタイプがあります。

1. TensorFlow Lite 組み込み演算子のみのモデル。（**推奨**）
2. 組み込み演算子と一部の TensorFlow コア 演算子を使用したモデル。
3. 組み込み演算子、TensorFlow コア演算子、カスタム演算子、またはそれらすべてを使用したモデル。

モデルに TensorFlow Lite でネイティブにサポートされている演算のみが含まれている場合は、変換するための追加のフラグは必要ありません。このタイプのモデルはスムーズに変換され、デフォルトの TensorFlow Lite ランタイムを使用して最適化と実行がより簡単になるため、これが推奨されるパスです。また、[Google Play サービス](../android/play_services)など、より多くのモデルのデプロイオプションがあります。[TensorFlow Lite コンバータガイド](../models/convert/convert_models)で始めることができます。組み込み演算子のリストについては、[TensorFlow Lite Ops ページ](https://www.tensorflow.org/mlir/tfl_ops)をご覧ください。

コアライブラリから一部の TensorFlow 演算を含める必要がある場合は、変換時にそれを指定し、ランタイムにそれらの演算が含まれるようにする必要があります。詳細な手順については、[TensorFlow 演算子の選択](ops_select.md)のトピックをご覧ください。

可能な限り、変換されたモデルにカスタム演算子を含めるという最後のオプションは避けてください。[カスタム演算子](https://www.tensorflow.org/guide/create_op)は、複数のプリミティブな TensorFlow コア演算子を組み合わせて作成された演算子か、まったく新しい演算子を定義したものです。カスタム演算子が変換されると、組み込みの TensorFlow Lite ライブラリの外部に依存関係が発生するため、モデル全体のサイズが大きくなる可能性があります。カスタム演算は、モバイルまたはデバイスのデプロイ用に特別に作成されていない場合、リソースに制約のあるデバイスにデプロイすると、サーバー環境と比較してパフォーマンスが低下する可能性があります。最後に、一部の TensorFlow コア演算子を含めるのと同様に、カスタム演算子では[モデルのランタイム環境を変更する](ops_custom#create_and_register_the_operator)必要があり、[Google Play サービス](../android/play_services)などの標準ランタイムサービスを利用することが制限されます。

## サポートされている型

ほとんどの TensorFlow Lite 演算は、浮動小数点数（`float32`）と量子化（`uint8`、`int8`）推論の両方をターゲットしていますが、多くの演算子は、`tf.float16` や文字列といったほかの型をまだターゲットしていません。

浮動小数点数モデルと量子化モデルの間には、異なるバージョンの演算を使用するというだけでなく、他にも変換方法に違いがあります。量子化変換には、テンソルのダイナミックレンジ情報が必要です。これには、モデルのトレーニング中に「偽量子化」が必要で、較正データセット経由でレンジ情報を取得するか、「その場で」レンジ推測を行う必要があります。詳しくは、[量子化](../performance/model_optimization.md)をご覧ください。

## 明瞭な変換、定数畳み込み、および融合

多くの TensorFlow 演算は、直接相当する演算がない場合でも、TensorFlow Lite で処理できます。これは、単純にグラフから除外して（`tf.identity`）、テンソルに置換（`tf.placeholder`）できる演算か、より複雑な演算に融合（`tf.nn.bias_add`）できる演算の場合です。一部のサポートされている演算であっても、これらのプロセスのいずれかによって除外される場合があります。

次は、完全なリストではありませんが、一般的にグラフから除外される TensorFlow 演算です。

- `tf.add`
- `tf.debugging.check_numerics`
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
