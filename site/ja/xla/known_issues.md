# 既知の問題

XLA を使用してコンパイルすると、プログラムのパフォーマンスを大幅に向上させることができますが、TensorFlow 相互運用性にはいくつかの既知の問題があります。

## 異なるデバイス上の `tf.Variable`

*エラーメッセージ*: `INVALID_ARGUMENT: Trying to access resource <Variable> (defined @ <Loc>) located in device CPU:0 from device GPU:0`

XLA クラスタは 1 台のデバイスでしか実行できません。別のデバイスにある `tf.Variable` に対して読み取りまたは書き込みを行うことはできません。通常、このエラーメッセージは、変数が最初から適切なデバイスに配置されていないことを示しています。エラーメッセージは、問題のある変数の場所を正確に指定します。

注意: `int32` 型の `tf.Variable` は常にホストに配置され、GPU に配置することはできません。回避策として、`int64` を使用できます。

## TensorArray TF/XLA 相互変換はサポートされていない

*エラーメッセージ*: `Support for TensorList crossing the XLA/TF boundary is not implemented`

XLA は `tf.TensorArray` をサポートしていますが、TF と XLA 間の*相互変換*はまだ実装されていません。このエラーは、コンパイルされたブロック内で `TensorArray` が使用されていて、導関数が外部で使用されている場合によく発生します。

*回避策*: 導関数をとっている最も外側のスコープをコンパイルします。

## TensorFlow while ループを制限する必要がある（または backprop を無効にする）

*エラーメッセージ*: `XLA compilation requires a fixed tensor list size. Set the max number of elements. This could also happen if you're using a TensorArray in a while loop that does not have its maximum_iteration set, you can fix this by setting maximum_iteration to a suitable value`

`tf.while_loop` を使用して作成された TF while [ループ](https://www.tensorflow.org/api_docs/python/tf/while_loop)は、すべての中間結果を `TensorArray` に累積することでバックプロパゲーションをサポートしますが、XLA は制限付き `TensorArray` のみをサポートします。

*回避策*: コンパイルされたすべての while ループでは、`maximum_iterations` パラメータをコンパイル時に既知の定数値に設定するか、`back_prop=False` を使用してバックプロパゲーションを無効にする必要があります。

## 動的な `tf.TensorArray` はサポートされていない

`tf.TensorArray(..., dynamic_size=True)` への書き込みは、配列が元の境界を超えたときに不明な数の再割り当てが必要になるため、XLA ではコンパイルできません。

*回避策*: 静的な既知の配列の制限を提供します。

## 乱数生成は TF シードを無視する

XLA は現在、ランダム演算に対する TF シードを無視しています。これは、`tf.random.normal` や `tf.nn.dropout` などのステートフル TF ランダム演算に影響を与えます。XLA は、実行ごとに新しい一意のシードがコンパイルにシードされたかのように動作します。

*回避策*: `tf.random.stateless_uniform` または、`tf.random.Generator` などの[推薦される RNG](https://www.tensorflow.org/guide/random_numbers#stateless_rngs) を直接使用します。
