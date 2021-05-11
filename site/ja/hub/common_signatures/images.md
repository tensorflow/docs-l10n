<!--* freshness: { owner: 'mroff' reviewed: '2021-03-09' } *-->

# 画像の一般的なシグネチャ

このページでは、画像関連タスクにおいて、[TF1 Hub 形式](../tf1_hub_module.md) のモジュ―ルで実装すべき一般的なシグネチャを説明します。（[TF2 SavedModel 形式](../tf2_saved_model.md)については、同様の [SavedModel API](../common_saved_model_apis/images.md) をご覧ください。）

一部のモジュールは複数のタスクに使用できます（たとえば、画像分類モジュールは途中で特徴量抽出を実行する傾向があります）。このため、各モジュールは (1) パブリッシャーが期待するすべてのタスクに名前付きシグネチャを、(2) 指定のプライマリタスクにデフォルトのシグネチャ `output = m(images)` を提供します。

<a name="feature-vector"></a>

## 画像特徴量ベクトル

### 使い方の概要

**画像特徴量ベクトル**は、主にコンシューマーモデルによる分類のために画像全体を表現する密な 1 次元テンソルです（CNN の中間活性化とは異なり、空間分解は行われません。[画像分類](#classification)とは異なり、パブリッシャーモデルによって学習された分類は破棄されます）。

画像特徴量抽出のモジュールには、画像のバッチを特徴量ベクトルのバッチにマッピングするデフォルトのシグネチャがあります。たとえば、次のように使用できます。

```python
  module_spec = hub.load_module_spec("path/to/module")
  height, width = hub.get_expected_image_size(module_spec)
  images = ...  # A batch of images with shape [batch_size, height, width, 3].
  module = hub.Module(module_spec)
  features = module(images)   # A batch with shape [batch_size, num_features].
```

このシグネチャは、対応する名前付きシグネチャも定義します。

### シグネチャの仕様

画像特徴量ベクトルを抽出する名前付きシグネチャは、次のように呼び出されます。

```python
  outputs = module(dict(images=images), signature="image_feature_vector",
                   as_dict=True)
  features = outputs["default"]
```

入力は、[画像入力](#input)に関する一般的な規則に従います。

outputs ディクショナリには dtype `float32` および形状 `[batch_size, num_features]` の `"default"` 出力が含まれています。`batch_size` は入力と同じですが、グラフの作成時には不明です。`num_features` は入力サイズに依存しないモジュール固有の既知の定数です。

これらの特徴量ベクトルは、（画像分類用の典型的な CNN の最上位の畳み込みレイヤーからプールされた特徴のように）単純なフィードフォワード分類器での分類に使用できるように作られています。

ドロップアウトを出力の特徴量に適用するかどうかは、モジュールのコンシューマーに任せる必要があります。モジュール自体が実際の出力でドロップアウトを実行するべきではありません（ほかの場所で内部的にドロップアウトを使用している場合でも）。

outputs ディクショナリはモジュール内の非表示レイヤーの活性化など、さらなる出力を提供できます。それらのキーと値はモジュールに依存しています。アーキテクチャ依存キーの前にアーキテクチャ名を付けることをお勧めします（たとえば、中間レイヤー `"InceptionV3/Mixed_5c"` と最上位の畳み込みレイヤー `"InceptionV2/Mixed_5c"` の混同を回避します）。

<a name="classification"></a>

## 画像分類

### 使い方の概要

**画像分類**は、*モジュールパブリッシャーによって選択された分類のクラス*のメンバーシップについて、画像のピクセルを線形スコア（ロジット）にマッピングします。これにより、コンシューマーは基本的な特徴（[画像特徴ベクトル](#feature-vector)を参照）だけでなく、パブリッシャーモジュールによって学習された特定の分類から結論を出すことができます。

画像特徴量抽出のモジュールには、画像のバッチをロジットのバッチにマッピングするデフォルトのシグネチャがあります。たとえば、次のように使用できます。

```python
  module_spec = hub.load_module_spec("path/to/module")
  height, width = hub.get_expected_image_size(module_spec)
  images = ...  # A batch of images with shape [batch_size, height, width, 3].
  module = hub.Module(module_spec)
  logits = module(images)   # A batch with shape [batch_size, num_classes].
```

このシグネチャは、対応する名前付きシグネチャも定義します。

### シグネチャの仕様

画像特徴量ベクトルを抽出する名前付きシグネチャは、次のように呼び出されます。

```python
  outputs = module(dict(images=images), signature="image_classification",
                   as_dict=True)
  logits = outputs["default"]
```

入力は、[画像入力](#input)に関する一般的な規則に従います。

outputs ディクショナリには dtype `float32` および形状 `[batch_size, num_classes]` の `"default"` 出力が含まれています。`batch_size` は入力と同じですが、グラフの作成時には不明です。`num_classes` は入力サイズに依存しないモジュール固有の既知の定数です。

`outputs["default"][i, c]` を評価すると、インデックス `c` を持つクラス内のサンプル `i` のメンバーシップを予測するスコアが得られます。

これらのスコアがソフトマックス（相互に排他的なクラスの場合）、シグモイド（直交クラスの場合）、または他の何かで使用されることを意図しているかどうかは、基本的な分類に依存します。モジュールのドキュメントでこれを説明し、クラスインデックスの定義を参照する必要があります。

outputs ディクショナリはモジュール内の非表示レイヤーの活性化など、さらなる出力を提供できます。それらのキーと値はモジュールに依存しています。アーキテクチャ依存キーの前にアーキテクチャ名を付けることをお勧めします（たとえば、中間レイヤー `"InceptionV3/Mixed_5c"` と最上位の畳み込みレイヤー `"InceptionV2/Mixed_5c"` の混同を回避します）。

<a name="input"></a>

## 画像入力

以下の内容は、すべてのタイプの画像モジュールと画像シグネチャに共通です。

画像のバッチを入力として受け取るシグネチャは、その入力を dtype `float32` および要素が [0, 1] の範囲に正規化されたピクセルの RGB カラー値になっている形状 `[batch_size, height, width, 3]` の密な 4 次元テンソルとして受け入れます。これは、`tf.image.convert_image_dtype(..., tf.float32)` が後続する `tf.image.decode_*()` から取得されるものです。

厳密に 1 つ（または 1 つの主な）画像入力を持つモジュールは、この入力に `"images"` という名前を使用します。

このモジュールは任意の `batch_size` を受け入れ、それに応じて TensorInfo.tensor_shape の最初の次元を "unknown"（不明）に設定します。最後の次元は RGB チャネルの数である `3` に固定されています。次元 `height` および `width` は入力画像の期待サイズに固定されています（今後の作業により、完全な畳み込みモジュールではこの制限が撤廃される可能性があります）。

モジュールのコンシューマーは形状を直接検査するのではなく、モジュールまたはモジュールの仕様で hub.get_expected_image_size() を呼び出してサイズ情報を取得し、それに応じて（通常はバッチ処理前/処理中に）入力画像のサイズを変更する必要があります。

簡単にするため、TF-Hub モジュールはテンソルの `channels_last`（または `NHWC`）レイアウトを使用し、必要に応じて TensorFlow のグラフオプティマイザに `channels_first`（または `NCHW`）への書き換えを任せます。TensorFlow バージョン 1.7 移行はこの動作がデフォルトになっています。
