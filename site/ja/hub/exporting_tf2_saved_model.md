<!--* freshness: { owner: 'maringeo' reviewed: '2021-10-10' review_interval: '6 months' } *-->

# SavedModel をエクスポートする

このページでは、TensorFlow プログラムのモデルを [TensorFlow 2 の SavedModel 形式](https://www.tensorflow.org/guide/saved_model)でエクスポートする方法について説明します。この形式は、TensorFlow Hub 上で事前トレーニングされたモデルとモデルピースの共有に推奨される方法です。これは以前の [TF1 Hub 形式](tf1_hub_module.md)に置き換わるもので、新しい API セットが付属しています。TF1 Hub 形式モデルのエクスポートに関する詳細は、[TF1 Hub 形式のエクスポート](exporting_hub_format.md)をご覧ください。TensorFlow Hub で共有するために  SavedModel を圧縮する方法については、[こちら](writing_documentation.md#model-specific_asset_content)をご覧ください。

一部のモデル構築ツールキットでは、これを行うためのツールがすでに提供されています（[TensorFlow Model Garden](#tensorflow-model-garden) について以下をご覧ください）。

## 概要

SavedModel は TensorFlow のトレーニング済みモデルまたはモデルピース用の標準的なシリアル化形式です。モデルのトレーニング済みの重みとともに、計算を実行するための正確な TensorFlow 演算が含まれるため、それを作成したコードに依存することなく使用することが可能です。特に、TensorFlow 演算は共通の基本言語であるため、Keras のような高レベルのモデル構築 API で再利用できます。

## Keras から保存する

TensorFlow 2 より、`tf.keras.Model.save()` と `tf.keras.models.save_model()` は、デフォルトで SavedModel 形式（HDF5 ではありません）になります。保存された SavedModel は `hub.load()`、`hub.KerasLayer`、およびほかの高レベル API 用の類似アダプタで使用することができます。

`include_optimizer=False` を使って保存するだけで、完全な Keras モデルを共有することができます。

Keras モデルの一部を共有する場合は、モデルのピースを作成してからそれを保存します。このようなコードは、次のように始めから作成することができます。

```python
piece_to_share = tf.keras.Model(...)
full_model = tf.keras.Sequential([piece_to_share, ...])
full_model.fit(...)
piece_to_share.save(...)
```

または、その後で共有する部分を切り取ることもできます（フルモデルのレイヤー構造と一致する場合）。

```python
full_model = tf.keras.Model(...)
sharing_input = full_model.get_layer(...).get_output_at(0)
sharing_output = full_model.get_layer(...).get_output_at(0)
piece_to_share = tf.keras.Model(sharing_input, sharing_output)
piece_to_share.save(..., include_optimizer=False)
```

[TensorFlow Models](https://github.com/tensorflow/models) は、BERT に前者のアプローチを使用しています（[nlp/tools/export_tfhub_lib.py](https://github.com/tensorflow/models/blob/master/official/nlp/tools/export_tfhub_lib.py) をご覧の上、エクスポート用の `core_model` とチェックポイント復元用の `pretrainer` で分割されているところに注意してください）。また、ResNet には後者のアプローチを使用しています（[legacy/image_classification/tfhub_export.py](https://github.com/tensorflow/models/blob/master/official/legacy/image_classification/resnet/tfhub_export.py) をご覧ください）。

## 低レベル TensorFlow から保存する

この操作には、TensorFlow の [SavedModel ガイド](https://www.tensorflow.org/guide/saved_model)を熟知している必要があります。

推論されるシグネチャ以上のものを提供する場合は、[再利用可能な SavedModel のインターフェース](reusable_saved_models.md)を実装する必要があります。概念的には、次のように記述されます。

```python
class MyMulModel(tf.train.Checkpoint):
  def __init__(self, v_init):
    super().__init__()
    self.v = tf.Variable(v_init)
    self.variables = [self.v]
    self.trainable_variables = [self.v]
    self.regularization_losses = [
        tf.function(input_signature=[])(lambda: 0.001 * self.v**2),
    ]

  @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
  def __call__(self, inputs):
    return tf.multiply(inputs, self.v)

tf.saved_model.save(MyMulModel(2.0), "/tmp/my_mul")

layer = hub.KerasLayer("/tmp/my_mul")
print(layer([10., 20.]))  # [20., 40.]
layer.trainable = True
print(layer.trainable_weights)  # [2.]
print(layer.losses)  # 0.004
```

## SavedModel 作成者に対する注意事項

TensorFlow Hub で共有する SavedModel を作成する場合は、コンシューマがファインチューニングを行うのか、どのように行うのか、ということを予め考え、ドキュメントにガイダンスを提供するようにしてください。

Keras モデルから保存すると、ファインチューニングのすべての仕組みを機能させることができます（重み正則化損失の保存、トレーニング可能な変数の宣言、`training=True` と `training=False` の両方の `__call__` のトレースなど）。

ソフトマックス確率やトップ k 予測の代わりにロジットを出力するなど、勾配のフローとうまく連携するモデルインターフェースを選択してください。

モデルでドロップアウト、バッチ正規化、またはハイパーパラメータを使用する類似のトレーニングテクニックが使用されている場合は、期待される多数のターゲット問題やバッチサイズに合理的な値に設定してください。（これを執筆している時点では、Keras から保存すると、消費する側でこれらを簡単に調整することはできません。）

各レイヤーの重み正則化器は保存されますが（正則化の強度計数とともに）、オプティマイザ内の重み正則化（`tf.keras.optimizers.Ftrl.l1_regularization_strength=...)` など）は失われます。SavedModel のコンシューマに適宜アドバイスを提供してください。

<a name="tensorflow-model-garden"></a>

## TensorFlow Model Garden

[TensorFlow Model Garden](https://github.com/tensorflow/models/tree/master/official) リポジトリには、[tfhub.dev](https://tfhub.dev/) にアップロードするために再利用可能な TF2 SavedModel を作成している、多くの例があります。

## コミュニティリクエスト

TensorFlow Hub チームは、tfhub.dev で利用可能なアセットのごく一部を生成しています。モデルの作成は、主に Google や Deepmind の研究者、企業や学術研究機関、ML 愛好家の方々を頼りとしています。そのため、特定のアセットに対するコミュニティの要求を満たす保証や、新しいアセットが利用可能になるまでの時間の見積もりはできません。

以下の[コミュニティモデルリクエスト マイルストーン](https://github.com/tensorflow/hub/milestone/1)には、コミュニティからの特定のアセットのリクエストが含まれています。ご自身やお知り合いでアセットを制作して tfhub.dev で共有したいという方があれば、ぜひご提供ください！
