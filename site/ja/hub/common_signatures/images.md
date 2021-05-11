<!--* freshness: { owner: 'mroff' reviewed: '2021-03-09' } *-->

# 画像の一般的なシグネチャ

This page describes common signatures that should be implemented by modules in the [TF1 Hub format](../tf1_hub_module.md) for image-related tasks. (For the [TF2 SavedModel format](../tf2_saved_model.md), see the analogous [SavedModel API](../common_saved_model_apis/images.md).)

Some modules can be used for more than one task (e.g., image classification modules tend to do some feature extraction on the way). Therefore, each module provides (1) named signatures for all the tasks anticipated by the publisher, and (2) a default signature `output = m(images)` for its designated primary task.

<a name="feature-vector"></a>

## Image Feature Vector

### 使い方の概要

An **image feature vector** is a dense 1-D tensor that represents a whole image, typically for classification by the consumer model. (Unlike the intermediate activations of CNNs, it does not offer a spatial breakdown. Unlike [image classification](#classification), it discards the classification learned by the publisher model.)

A module for image feature extraction has a default signature that maps a batch of images to a batch of feature vectors. It can be used like so:

```python
  module_spec = hub.load_module_spec("path/to/module")
  height, width = hub.get_expected_image_size(module_spec)
  images = ...  # A batch of images with shape [batch_size, height, width, 3].
  module = hub.Module(module_spec)
  features = module(images)   # A batch with shape [batch_size, num_features].
```

このシグネチャは、対応する名前付きシグネチャも定義します。

### シグネチャの仕様

The named signature for extracting image feature vectors is invoked as

```python
  outputs = module(dict(images=images), signature="image_feature_vector",
                   as_dict=True)
  features = outputs["default"]
```

入力は、[画像入力](#input)に関する一般的な規則に従います。

outputs ディクショナリには dtype `float32` および形状 `[batch_size, num_features]` の `"default"` 出力が含まれています。`batch_size` は入力と同じですが、グラフの作成時には不明です。`num_features` は入力サイズに依存しないモジュール固有の既知の定数です。

These feature vectors are meant to be usable for classification with a simple feed-forward classifier (like the pooled features from the topmost convolutional layer in a typical CNN for image classification).

Applying dropout to the output features (or not) should be left to the module consumer. The module itself should not perform dropout on the actual outputs (even if it uses dropout internally in other places).

The outputs dictionary may provide further outputs, for example, the activations of hidden layers inside the module. Their keys and values are module-dependent. It is recommended to prefix architecture-dependent keys with an architecture name (e.g., to avoid confusing the intermediate layer `"InceptionV3/Mixed_5c"` with the topmost convolutional layer `"InceptionV2/Mixed_5c"`).

<a name="classification"></a>

## 画像分類

### 使い方の概要

**画像分類**は、*モジュールパブリッシャーによって選択された分類のクラス*のメンバーシップについて、画像のピクセルを線形スコア（ロジット）にマッピングします。これにより、コンシューマーは基本的な特徴（[画像特徴ベクトル](#feature-vector)を参照）だけでなく、パブリッシャーモジュールによって学習された特定の分類から結論を出すことができます。

A module for image feature extraction has a default signature that maps a batch of images to a batch of logits. It can be used like so:

```python
  module_spec = hub.load_module_spec("path/to/module")
  height, width = hub.get_expected_image_size(module_spec)
  images = ...  # A batch of images with shape [batch_size, height, width, 3].
  module = hub.Module(module_spec)
  logits = module(images)   # A batch with shape [batch_size, num_classes].
```

このシグネチャは、対応する名前付きシグネチャも定義します。

### シグネチャの仕様

The named signature for extracting image feature vectors is invoked as

```python
  outputs = module(dict(images=images), signature="image_classification",
                   as_dict=True)
  logits = outputs["default"]
```

入力は、[画像入力](#input)に関する一般的な規則に従います。

outputs ディクショナリには dtype `float32` および形状 `[batch_size, num_classes]` の `"default"` 出力が含まれています。`batch_size` は入力と同じですが、グラフの作成時には不明です。`num_classes` は入力サイズに依存しないモジュール固有の既知の定数です。

Evaluating `outputs["default"][i, c]` yields a score predicting the membership of example `i` in the class with index `c`.

これらのスコアがソフトマックス（相互に排他的なクラスの場合）、シグモイド（直交クラスの場合）、または他の何かで使用されることを意図しているかどうかは、基本的な分類に依存します。モジュールのドキュメントでこれを説明し、クラスインデックスの定義を参照する必要があります。

The outputs dictionary may provide further outputs, for example, the activations of hidden layers inside the module. Their keys and values are module-dependent. It is recommended to prefix architecture-dependent keys with an architecture name (e.g., to avoid confusing the intermediate layer `"InceptionV3/Mixed_5c"` with the topmost convolutional layer `"InceptionV2/Mixed_5c"`).

<a name="input"></a>

## 画像入力

以下の内容は、すべてのタイプの画像モジュールと画像シグネチャに共通です。

画像のバッチを入力として受け取るシグネチャは、その入力を dtype `float32` および要素が [0, 1] の範囲に正規化されたピクセルの RGB カラー値になっている形状 `[batch_size, height, width, 3]` の密な 4 次元テンソルとして受け入れます。これは、`tf.image.convert_image_dtype(..., tf.float32)` が後続する `tf.image.decode_*()` から取得されるものです。

厳密に 1 つ（または 1 つの主な）画像入力を持つモジュールは、この入力に `"images"` という名前を使用します。

このモジュールは任意の `batch_size` を受け入れ、それに応じて TensorInfo.tensor_shape の最初の次元を "unknown"（不明）に設定します。最後の次元は RGB チャネルの数である `3` に固定されています。次元 `height` および `width` は入力画像の期待サイズに固定されています（今後の作業により、完全な畳み込みモジュールではこの制限が撤廃される可能性があります）。

モジュールのコンシューマーは形状を直接検査するのではなく、モジュールまたはモジュールの仕様で hub.get_expected_image_size() を呼び出してサイズ情報を取得し、それに応じて（通常はバッチ処理前/処理中に）入力画像のサイズを変更する必要があります。

簡単にするため、TF-Hub モジュールはテンソルの `channels_last`（または `NHWC`）レイアウトを使用し、必要に応じて TensorFlow のグラフオプティマイザに `channels_first`（または `NCHW`）への書き換えを任せます。TensorFlow バージョン 1.7 移行はこの動作がデフォルトになっています。
