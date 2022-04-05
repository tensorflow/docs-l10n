<!--* freshness: { owner: 'maringeo' reviewed: '2022-01-12'} *-->

# TensorFlow 2 における TF Hub の SavedModel

TensorFlow Hub でトレーニング済みのモデルやモデルピースを共有するには、[TensorFlow 2 の SavedModel 形式](https://www.tensorflow.org/guide/saved_model)が推奨されます。古い [TF1 Hub 形式](tf1_hub_module.md)に置き換わるものであり、新しい API が含まれます。

このページでは、`hub.load()` API とその `hub.KerasLayer` ラッパーを使用して、TensorFlow 2 プログラムで TF2 SavedModels を再利用する方法を説明します。（通常、`hub.KerasLayer` はほかの `tf.keras.layers` と組み合わせて、Keras モデルまたは TF2 Estimator の `model_fn` を構築します。）これらの API は、TF1 Hub 形式のレガシーモデルを読み込むこともできます。[互換性ガイド](model_compatibility.md)をご覧ください。

TensorFlow 1 のユーザーは、TF 1.15 に更新すると、同じ API を使用できるようになります。これより古いバージョンの TF1 は機能しません。

## TF Hub から SavedModels を使用する

### Keras で SavedModel を使用する

[Keras](https://www.tensorflow.org/guide/keras/) は、Keras レイヤーオブジェクトを合成することでディープラーニングモデルを構築するための高レベル API です。`tensorflow_hub` ライブラリには、SavedModel の URL（またはファイルシステムパス）を使って初期化された後にトレーニング済みの重みを含む SavedModel の計算を提供する、 `hub.KerasLayer` クラスがあります。

次に、トレーニング済みのテキスト埋め込みの例を示します。

```python
import tensorflow as tf
import tensorflow_hub as hub

hub_url = "https://tfhub.dev/google/nnlm-en-dim128/2"
embed = hub.KerasLayer(hub_url)
embeddings = embed(["A long sentence.", "single-word", "http://example.com"])
print(embeddings.shape, embeddings.dtype)
```

上記から、通常の Keras の方法でテキスト分類器を構築することができます。

```python
model = tf.keras.Sequential([
    embed,
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])
```

[Text classification colab](https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/tf2_text_classification.ipynb) は、こういった分類器をトレーニングして評価する方法を示す完全なサンプルです。

`hub.KerasLayer` のモデルの重みは、デフォルトでトレーニング対象外に設定されています。これを変更する方法は、以下のファインチューニングのセクションで説明されています。Keras では通常通り、重みは同じレイヤーのすべてのアプリケーションで共有されます。

### Estimator で SavedModel を使用する

分散型トレーニングでは、TensorFlow の [Estimator](https://www.tensorflow.org/tutorials/distribute/multi_worker_with_estimator) API ユーザーは、ほかの `tf.keras.layers` のほかに `hub.KerasLayer` の観点で `model_fn` を記述することで、TF Hub から SavedModel を使用することができます。

### 舞台裏の処理: SavedModel のダウンロードとキャッシュ

TensorFlow Hub から（または [ホスティング](hosting.md)プロトコルを実装するほかの HTTPS サーバーから）SavedModel を使用すると、すでに存在しない限りその SavedModel をローカルファイルシステムにダウンロードして解凍します。ダウンロードして解凍される SavedModel のキャッシュに使用するデフォルトの一時的な場所は、環境変数 `TFHUB_CACHE_DIR` を設定してオーバーライドすることができます。詳細は、[キャッシング](caching.md)をご覧ください。

### 低レベルの TensorFlow で SavedModel を使用する

関数 `hub.load(handle)` は、（`handle` がファイルシステムにすでに存在しない場合は）SavedModel をダウンロードして解凍し、TensorFlow のビルトイン関数 `tf.saved_model.load()` で読み込んだ結果を返します。したがって、`hub.load()` はあらゆる有効な SavedModel を処理することができます（以前の TF1 の `hub.Module` とは異なります）。

#### 高度なトピック: 読み込み後の SavedModel に期待されること

SavedModel のコンテンツによっては、`obj = hub.load(...)` の結果をさまざまな方法で呼び出すことができます（TensorFlow の [SavedModel ガイド](https://www.tensorflow.org/guide/saved_model) にはより詳細に説明されています）。

- SavedModel のサービングシグネチャ（存在する場合）は、具象関数のディクショナリとして表され、`tensors_out = obj.signatures["serving_default"](**tensors_in)` のように呼び出すことができます。テンソルのディクショナリは、対応する入力名と出力名をキーとし、シグネチャの形状と dtype の制限に従います。

- [`@tf.function`](https://www.tensorflow.org/api_docs/python/tf/function) でデコレートされた、保存されたオブジェクトのメソッド（存在する場合）は、tf.function オブジェクトとしてリストアすることができます。このオブジェクトは、保存される前に tf.function によって[トレース](https://www.tensorflow.org/tutorials/customization/performance#tracing)された、テンソル型引数と非テンソル型引数のすべての組み合わせて呼び出すことができます。特に、適切なトレースを使った `obj.__call__` 　メソッドがある場合、`obj` そのものを Python 関数として呼び出すことができます。単純な例は、`output_tensor = obj(input_tensor, training=False)` のようになります。

これにより、SavedModel が実装できるインターフェースに大きな自由が生まれます。`obj` の [Reusable SavedModel インターフェース](reusable_saved_models.md)は、`hub.KerasLayer` のようなアダプタなど、クライアントコードが SavedModel の使用方法を理解できるように規則を確立します。

SavedModel の中には、特により大きなモデルで再利用されることを目的としていないモデル全体など、その規則に従わないものもあり、サービングシグネチャのみを提供することがあります。

SavedModel のトレーニング対象変数は、トレーニング対象として再読み込みされ、`tf.GradientTape` によってデフォルトで監視されます。注意事項については、以下のファインチューニングのセクションをご覧の上、スターターではこれを避けることを検討してください。ファインチューニングを行うとしても、`obj.trainable_variables` が、もともとトレーニング対象の変数のサブセットのみを再トレーニングするように推奨しているかどうかを確認するようにしてください。

## TF Hub 用の SavedModel を作成する

### 概要

SavedModel は TensorFlow のトレーニング済みモデルまたはモデルピース用の標準的なシリアル化形式です。モデルのトレーニング済みの重みとともに、計算を実行するための正確な TensorFlow 演算が含まれるため、それを作成したコードに依存することなく使用することが可能です。特に、TensorFlow 演算は共通の基本言語であるため、Keras のような高レベルのモデル構築 API で再利用できます。

### Keras から保存する

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

### 低レベル TensorFlow から保存する

この操作では、TensorFlow の [SavedModel ガイド](https://www.tensorflow.org/guide/saved_model)を熟知している必要があります。

サービングシグネチャ以上のものを提供する場合は、[Reusable SavedModel のインターフェース](reusable_saved_models.md)を実装する必要があります。概念的には、次のように記述されます。

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

## ファインチューニング

インポートされた SavedModel のトレーニング済みの変数とそれに関するモデルのトレーニング済みの変数を合わせてトレーニングことを、SavedModel の*ファインチューニング*と呼びます。これにより品質が改善されますが、多くの場合トレーニングがより困難になります（特に CNN では、時間が長引く、オプティマイザとハイパーパラメータによさらにり依存する、過適合のリスクが高まる、データセットの増加が必要となる、といった問題が生まれる可能性があります）。SavedModel のコンシューマーは、適切なトレーニングシステムを整えた後で、SavedModel のパブリッシャーが推奨する場合に限り、ファインチューニングを検討することをお勧めします。

ファインチューニングよにって、トレーニングされる「連続的な」モデルパラメータが変更されて今います。テキスト入力のトークン化や埋め込み行列の対応エントリへのトークンのマッピングなど、ハードコードされた変換は変更されません。

### SavedModel コンシューマーに対する注意事項

`hub.KerasLayer` を次のように作成します。

```python
layer = hub.KerasLayer(..., trainable=True)
```

上記のようにすることで、レイヤーが読み込む SavedModel をファインチューニングすることができます。SavedModel に宣言されたトレーニング対象の重みと重み正則化器を Keras モデルに追加し、SavedModel の計算をトレーニングモードで実行します（ドロップアウトなど）。

[image classification colab](https://github.com/tensorflow/hub/blob/master/examples/colab/tf2_image_retraining.ipynb) には、エンドツーエンドの例が、オプションのファインチューニングとともに含まれています。

#### ファインチューニングの結果を再エクスポートする

高度なユーザーは、ファインチューニングの結果を SavedModel に保存し直し、元の読み込まれた SavedModel の代わりに使用することがあるかもしれません。これは、以下のようなコードを使って行うことができます。

```python
loaded_obj = hub.load("https://tfhub.dev/...")
hub_layer = hub.KerasLayer(loaded_obj, trainable=True, ...)

model = keras.Sequential([..., hub_layer, ...])
model.compile(...)
model.fit(...)

export_module_dir = os.path.join(os.getcwd(), "finetuned_model_export")
tf.saved_model.save(loaded_obj, export_module_dir)
```

### SavedModel 作成者に対する注意事項

TensorFlow Hub で共有する SavedModel を作成する場合は、それを消費するユーザーがファインチューンを行うのか、どのように行うのか、ということを予め考え、ドキュメントにガイダンスを提供するようにしてください。

Keras モデルから保存すると、ファインチューニングのすべての仕組みを機能させることができます（重み正則化損失の保存、トレーニング対象変数の宣言、`training=True` と `training=False` の両方の `__call__` のトレースなど）。

ソフトマックス確率やトップ k 予測の代わりにロジットを出力するなど、勾配のフローとうまく連携するモデルインターフェースを選択してください。

モデルでドロップアウト、バッチ正規化、またはハイパーパラメータを使用する類似のトレーニングテクニックが使用されている場合は、期待される多数のターゲット問題やバッチサイズに合理的な値に設定してください。（これを執筆している時点では、Keras から保存すると、消費する側でこれらを簡単に調整することはできません。）

各レイヤーの重み正則化器は保存されますが（正則化の強度計数とともに）、オプティマイザ内の重み正則化（`tf.keras.optimizers.Ftrl.l1_regularization_strength=...)` など）は失われます。SavedModel のコンシューマに適宜アドバイスを提供してください。
