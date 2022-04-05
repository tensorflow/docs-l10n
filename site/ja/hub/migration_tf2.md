<!--* freshness: { owner: 'maringeo' reviewed: '2022-01-12' } *-->

# TensorFlow Hub を使用したまま TF1 から TF2 に移行する

このページでは、TensorFlow コードを TensorFlow 1 から TensorFlow 2 に移行する間に、TensoFlow Hub を使用し続ける方法を説明します。TensorFlow の一般的な[移行ガイド](https://www.tensorflow.org/guide/migrate)を補足する内容です。

TF2 では、TF Hub は、`tf.contrib.v1.layers` などが行うような `tf.compat.v1.Graph` を構築するためのレガシー `hub.Module` API を排除しました。その代わり、ほかの Keras レイヤーとともに使用する、`tf.keras.Model` を（通常、TF2 の新しい [Eager execution 環境](https://www.tensorflow.org/guide/eager_)で）構築するための `hub.KerasLayer` と、低レベル TensorFlow コード用の基盤の `hub.load()` メソッドが追加されています。

`hub.Module` API は、TF1 と TF2 の TF1 互換モードで使用できるように、`tensorflow_hub` ライブラリに残されていますが、[TF1 Hub 形式](tf1_hub_module.md)のモデルのみを読み込むことができます。

`hub.load()` と `hub.KerasLayer` の新しい API は、TensorFlow 1.15（eager およびグラフモード）と TensorFlow 2 で機能します。この API は、新しい [TF2 SavedModel](tf2_saved_model.md) アセットと、[モデルの互換性ガイド](model_compatibility.md)に説明される制限のもと、TF1 Hub 形式のレガシーモデルを読み込むことができます。

一般的に、できる限り新しい API を使用することが推奨されます

## 新しい API の要約

`hub.load()` は、TensorFlow Hub（または互換性のあるサービス）から SavedModel を読み込む新しい低レベル関数です。TF2 の `tf.saved_model.load()` をラッピングします。TensorFlow の [SavedModel ガイド](https://www.tensorflow.org/guide/saved_model)には、その結果何を行えるかが説明されています。

```python
m = hub.load(handle)
outputs = m(inputs)
```

`hub.KerasLayer` クラスは、`hub.load()` を呼び出して、ほかの Keras レイヤーとともに Keras で使用できるように結果を適合させます。（ほかの方法で使用される読み込み済みの SavedModels では便利なラッパーとして使用できる可能性もあります。）

```python
model = tf.keras.Sequential([
    hub.KerasLayer(handle),
    ...])
```

多くのチュートリアルで、上記の API が実際に使用される様子を紹介しています。特に、次の項目をご覧ください。

- [Text classification example notebook](https://github.com/tensorflow/hub/blob/master/examples/colab/tf2_text_classification.ipynb)
- [Image classification example notebook](https://github.com/tensorflow/hub/blob/master/examples/colab/tf2_image_retraining.ipynb)

### Estimator トレーニングで新しい API を使用する

パラメータサーバーを使った（または、リモートデバイスに変数が配置された TF1 セッションでの）トレーニングにおいて、Estimator で TF2 SavedModel を使用する場合、tf.Session の ConfigProto に `experimental.share_cluster_devices_in_session` を設定する必要があります。設定しない場合、"Assigned device '/job:ps/replica:0/task:0/device:CPU:0' does not match any device." のようなエラーが発生します。

必要なオプションは、次のように設定できます。

```python
session_config = tf.compat.v1.ConfigProto()
session_config.experimental.share_cluster_devices_in_session = True
run_config = tf.estimator.RunConfig(..., session_config=session_config)
estimator = tf.estimator.Estimator(..., config=run_config)
```

TF2.2 より、このオプションは実験的ではなくなっているため、`.experimental` の部分を削除することができます。

## TF1 Hub 形式のレガシーモデルを読み込む

使用事例では、新しい TF2 SavedModel をまだ使用できない場合もあり、TF1 Hub 形式でレガシーモデルを読み込まなければならない場合もあります。`tensorflow_hub` リリース 0.7 より、TF1 Hub 形式のレガシーモデルを、次に示す `hub.KerasLayer` とともに使用できるようになっています。

```python
m = hub.KerasLayer(handle)
tensor_out = m(tensor_in)
```

また、`KerasLayer` には、より具体的な TF1 Hub 形式のレガシーモデルとレガシー SavedModel の使用方法に、`tags`、`signature`、`output_key`、および `signature_outputs_as_dict` を指定する機能が備わっています。

TF1 Hub 形式の互換性に関する詳細は、[モデルの互換性ガイド](model_compatibility.md)をご覧ください。

## 低レベル API を使用する

レガシー TF1 Hub 形式のモデルは、次に示す方法の代わりに、`tf.saved_model.load` を使って読み込むことができます。

```python
# DEPRECATED: TensorFlow 1
m = hub.Module(handle, tags={"foo", "bar"})
tensors_out_dict = m(dict(x1=..., x2=...), signature="sig", as_dict=True)
```

次の方法を使用することが推奨されています。

```python
# TensorFlow 2
m = hub.load(path, tags={"foo", "bar"})
tensors_out_dict = m.signatures["sig"](x1=..., x2=...)
```

上記の例の `m.signatures` は、シグネチャ名でキーが設定された TensorFlow [具象関数](https://www.tensorflow.org/tutorials/customization/performance#tracing)の dict です。こういった関数を呼び出すと、未使用であっても、すべての出力が計算されます（TF1 のグラフモードの遅延評価とは異なります）。
