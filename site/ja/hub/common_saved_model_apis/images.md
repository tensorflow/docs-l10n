<!--* freshness: { owner: 'mroff' reviewed: '2021-03-09'  } *-->

# 画像タスクの一般的な SavedModel API

このページでは、画像関連のタスクに使用する [TF2 SavedModel](../tf2_saved_model.md) が [Reusable SavedModel API](../reusable_saved_models.md) をどのように実装しているかを説明します。（これは、[画像の共通シグネチャ](../common_signatures/images.md)に置き換わります。[TF1 Hub 形式](../tf1_hub_module)は使用廃止となっています。）

<a name="feature-vector"></a>

## 画像特徴量ベクトル

### 使い方の概要

**画像特徴量ベクトル**は、画像全体を表す密な 1 次元テンソルで、通常、単純なフィードフォワード分類器によってコンシューマーモデルで使用されます。（従来の CNN の観点では、これは、空間範囲がプールされたかフラット化された後にボトルネックとなる部分で、分類が行われる前にはボトルネックではありません。これについては、以下の [画像の分類](#classification)をご覧ください。）

画像特徴量抽出の Reusable SavedModel には、ルートオブジェクトに対する `__call__` メソッドがあり、画像のバッチを特徴量ベクトルのバッチにマッピングします。たとえば、次のように使用することができます。

```python
obj = hub.load("path/to/model")  # That's tf.saved_model.load() after download.
images = ...  # A batch of images with shape [batch_size, height, width, 3].
features = obj(images)   # A batch with shape [batch_size, num_features].
```

Keras では、次のようになります。

```python
features = hub.KerasLayer("path/to/model")(images)
```

入力は、[画像の入力](#input)の一般的な規則に従います。入力の `height` と `width` の許容可能な範囲は、モデルのドキュメントに示されています。

出力は、dtype `float32` と形状 `[batch_size, num_features]` の単一のテンソルです。`batch_size` は入力と同じですが、`num_features` は、入力サイズに関係なく、モジュール固有の定数です。

### API の詳細

[Reusable SavedModel API](../reusable_saved_models.md) は、`obj.variables` のリストも提供しています（Eager でロードしていない場合に初期化するなどの目的で）。

ファインチューニングをサポートするモデルは、`obj.trainable_variables` のリストを提供します。トレーニングモードで実行するには、`training=True` を渡す必要がある場合があります（ドロップアウトなどのため）。一部のモデルではオプションの引数が許可されており、ハイパーパラメータをオーバーライドすることができます（ドロップアウト率など。モデルのドキュメントを参照）。また、`obj.regularization_losses` のリストも提供する場合があります。詳細は、[Reusable SavedModel API](../reusable_saved_models.md) をご覧ください。

Keras では、これは `hub.KerasLayer` が処理します。ファインチューニングを有効にするために、`trainable=True` と（hparam のオーバーライドが適用されるまれなケースでは）`arguments=dict(some_hparam=some_value, ...))` で初期化します。

### 補足

ドロップアウトを出力の特徴量に適用するかどうかは、モデルの消費者に任せる必要があります。SavedModel 自体が実際の出力でドロップアウトを実行するべきではありません（ほかの場所で内部的にドロップアウトを使用している場合でも）。

### 例

画像特徴量ベクトルの Reusable SavedModel は次のリンク先で使用されています。

- Colab チュートリアル「[Retraining an Image Classifier](https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/tf2_image_retraining.ipynb)」
- コマンドラインツールの [make_image_classifier](https://github.com/tensorflow/hub/tree/master/tensorflow_hub/tools/make_image_classifier)

<a name="classification"></a>

## 画像分類

### 使い方の概要

**画像分類**は、*モジュールパブリッシャーによって選択された*分類のクラスのメンバーシップについて、画像のピクセルを線形スコア（ロジット）にマッピングします。これにより、モデルの消費者はパブリッシャーモジュールによって学習された特定の分類から結論を導き出すことができます。（新しいクラスのセットを使った画像分類では、代わりに新しい分類器で[画像特徴量ベクトル](#feature-vector)を再利用するのが一般的です。）

画像分類の Reusable SavedModel には、ルートオブジェクトに対する `__call__` メソッドがあり、画像のバッチをロジットのバッチにマッピングします。たとえば、次のように使用することができます。

```python
obj = hub.load("path/to/model")  # That's tf.saved_model.load() after download.
images = ...  # A batch of images with shape [batch_size, height, width, 3].
logits = obj(images)   # A batch with shape [batch_size, num_classes].
```

Keras では、次のようになります。

```python
logits = hub.KerasLayer("path/to/model")(images)
```

入力は、[画像の入力](#input)の一般的な規則に従います。入力の `height` と `width` の許容可能な範囲は、モデルのドキュメントに示されています。

出力 `logits` は、dtype `float32` と形状 `[batch_size, num_classes]` の単一のテンソルです。`batch_size` は入力と同じですが、`num_classes` は分類のクラスの数で、モデル固有の定数です。

値 `logits[i, c]` は、インデックス `c` を持つクラス内の例 `i` のメンバーシップを予測するスコアです。

これらのスコアがソフトマックス（相互に排他的なクラスの場合）、シグモイド（直交クラスの場合）、または他の何かで使用されることを意図しているかどうかは、基本的な分類に依存します。モジュールのドキュメントでこれを説明し、クラスインデックスの定義を参照する必要があります。

### API の詳細

[Reusable SavedModel API](../reusable_saved_models.md) は、`obj.variables` のリストも提供しています（Eager でロードしていない場合に初期化するなどの目的で）。

ファインチューニングをサポートするモデルは、`obj.trainable_variables` のリストを提供します。トレーニングモードで実行するには、`training=True` を渡す必要がある場合があります（ドロップアウトなどのため）。一部のモデルではオプションの引数が許可されており、ハイパーパラメータをオーバーライドすることができます（ドロップアウト率など。モデルのドキュメントを参照）。また、`obj.regularization_losses` のリストも提供する場合があります。詳細は、[Reusable SavedModel API](../reusable_saved_models.md) をご覧ください。

Keras では、これは `hub.KerasLayer` が処理します。ファインチューニングを有効にするために、`trainable=True` と（hparam のオーバーライドが適用されるまれなケースでは）`arguments=dict(some_hparam=some_value, ...))` で初期化します。

<a name="input"></a>

## 画像入力

以下の内容は、すべてのタイプの画像モデルに共通です。

画像のバッチを入力として受け取るモデルは、その入力を dtype `float32` および要素が [0, 1] の範囲に正規化されたピクセルの RGB カラー値になっている形状 `[batch_size, height, width, 3]` の密な 4 次元テンソルとして受け入れます。これは、`tf.image.convert_image_dtype(..., tf.float32)` が後に続く `tf.image.decode_*()` から取得されるものです。

モデルはすべての `batch_size` を受け入れます。`height` と `width` の許容可能な範囲は、モデルのドキュメントに説明されています。最後の次元は、3 つの RGB チャネルに固定されています。

モデルが一貫してテンソルの `channels_last`（または `NHWC`）レイアウトを使用し、必要に応じて `channels_first`（または `NCHW`）への書き換えを TensorFlow のグラフオプティマイザに任せることを推奨します。
