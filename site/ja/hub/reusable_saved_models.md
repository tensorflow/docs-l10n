<!--* freshness: { owner: 'akhorlin' reviewed: '2021-11-22' } *-->

# Reusable SavedModel

## はじめに

TensorFlow Hub は、ほかのアセットともに TensorFlow 2 の SavedModels をホストしており、`obj = hub.load(url)` を使って Python プログラムに読み込み直すことができます（[詳細](tf2_saved_model)）。返される `obj` は、`tf.saved_model.load()` の結果です（TensorFlow の [SavedModel ガイド](https://www.tensorflow.org/guide/saved_model)を参照）。このオブジェクトは、tf.functions、tf.Variables（トレーニング済みの値から初期化）、その他のリソースなどのオブジェクトである任意の属性を持つことができます。

このガイドでは、TensorFlow Python プログラムで*再利用*するために読み込まれた `obj` によって実装されるインターフェースを説明します。このインターフェースに適合する SavedModel は *Reusable SavedModel* と呼ばれます。

再利用するということは、ファインチューニングできる能力を含めて、`obj` に関するより大規模なモデルを構築するということです。ファインチューニングとは、周囲のモデルの一環として読み込まれた `obj` の重みをさらにトレーニングすることを指します。損失関数とオプティマイザは周囲のモデルによって決定されます。`obj` は、入力と出力活性化のマッピング（フォワードパス）のみを定義し、ドロップアウトまたはバッチ正規化などのテクニックが含まれる可能性があります。

**The TensorFlow Hub チームは、上述の意味で再利用されるすべての SavedModel に Reusable SavedModel インターフェースの実装を推奨しています**。`tensorflow_hub` ライブラリに含まれる多数のユーティリティ、特に `hub.KerasLayer` では、SavedModel を使ってインターフェースを実装することを要件としています。

### SignatureDef との関係

tf.function とその他の TF2 機能に関しては、このインターフェースは TF1 から利用可能で、TF2 でも引き続き推論（SavedModel を TF Serving または TF Lite にデプロイするなど）に使用されています。推論に使うシグネチャはファインチューニングをサポートできるほどの表現力がないため、再利用されるモデルには、[`tf.function`](https://www.tensorflow.org/api_docs/python/tf/function) がより自然で表現力の豊かな [Python API](https://www.tensorflow.org/tutorials/customization/performance) を提供しています。

### model-building ライブラリとの関係

Resuable SavedModel は、Keras や Sonnet などの特定の model-building ライブラリとは別に、TensorFlow 2 のプリミティブ型のみを使用します。このため、元のモデル構築コードとの依存関係を持つことなく、model-building ライブラリで簡単に再利用することができます。

Reusable SavedModel を読み込んだり、特定の model-buiding ライブラリから保存したりするには、ある程度の調整が必要となります。Keras の場合、<code>hub.KerasLayer</code> で読み込みを行い、Keras の SavedModel 形式によるビルトインの保存機能は、このインターフェースのスーパーセットとなることを目標に、TF2 向けに再設計されています（2019 年 5 月の [RFC](https://github.com/tensorflow/community/blob/master/rfcs/20190509-keras-saved-model.md) をご覧ください）。

### タスク固有の「Common SavedModel API」との関係

このページに記載のインターフェースの定義では、任意の数と型の入力と出力を使用できます。[TF Hub の Common SavedModel API](common_saved_model_apis/index.md) は、モデルを簡単に交換できるようにするために、この汎用インターフェースを特定のタスクの使用規則で調整しています。

## インターフェースの定義

### 属性

Reusable SavedModel は、`obj = tf.saved_model.load(...)` が次の属性を持つオブジェクトを返す TensorFlow 2 SavedModel です。

- `__call__`: 必須。以下の仕様に基づいてモデルの計算を実装する（フォワードパス）tf.function です。

- `variables`: tf.Variable オブジェクトのリスト。トレーニング対象とトレーニング対象外の変数を含む、すべての可能な `__call__` 呼び出しによって使用されるすべての変数をリストします。

    リストが空である場合は、省略できます。

    注意: この名前は TF1 SavedModel を読み込んで `GLOBAL_VARIABLES` コレクションを表す際に`tf.saved_model.load(...)` によって合成される属性と一致しているため便利です。

- `trainable_variables`: すべての要素の `v.trainable` が true である tf.Variable オブジェクトのリスト。この変数は、`variables` のサブセットである必要があり、オブジェクトをファインチューニングする際にトレーニングされる変数です。SavedModel の作成者は、もともとトレーニング対象であった変数を省略して、ファインチューニング中に変更してはいけないことを示すことができます。

    リストが空である場合、特に SavedModel がファインチューニングをサポートしない場合は、省略できます。

- `regularization_losses`: ゼロの入力を取って単一のスカラー浮動小数点数テンソルを返す tf.functions のリスト。ファインチューニングするには、SavedModel ユーザーは、（以降のスケーリングを必要としない最も単純なケースにおいて）追加の正規化項としてこれらを損失に含めることが推奨されます。通常、重みの正則化器を表現するために使用されます。（入力がない場合、tf.functions は行動の正則化器を表現できません。）

    リストが空である場合、特に SavedModel がファインチューニングをサポートしない場合や重み正則化器を記述しない場合は、省略できます。

### `__call__` 関数

Restored SavedModel `obj` には、リストアされた tf.function である `obj.__call__` 属性を指定できるようになっており、次のように `obj` を呼び出すことができます。

概要（疑似コード）:

```python
outputs = obj(inputs, trainable=..., **kwargs)
```

#### 引数

引数は次のとおりです。

- SavedModel の入力活性化のバッチに必要な引数には、1 つの位置指定引数があります。その型は次のいずれかです。

    - 単一入力の場合は、単一のテンソル
    - 名前なし入力の整列されたシーケンスの場合は、テンソルのリスト
    - 特定の入力名のセットでキーが設定されたテンソルの dict 型。

    （将来的にこのインターフェースが改善されると、より一般的なネストが可能になるかもしれません。）SavedModel の作成者は、上記のいずれかとテンソルの形状、および dtype を選択することができます。有用であれば、形状の次元を未定義にする必要があります（特にバッチサイズ）。

- Python のブール値である `True` または `False` を受け取るオプションのキーワード引数 `training` がある場合があります。デフォルトは `False` です。モデルのファインチューニングがサポートされており、2 つの計算が相違する場合（ドロップアウトとバッチ正規化）には、その違いをこの引数で実装します。そうでない場合は、この引数は存在しない場合があります。

    `__call__` がテンソル値の `training` 引数を受け入れる必要はありません。それらの間でディスパッチする必要がある場合、`tf.cond()` を使用するのは呼び出し側に依存します。

- SavedModel の作成者は、特定の名前の `kwargs` オプションをさらに多く受け入れるように選択できます。

    - テンソル値の引数については、SavedModel の作成者は許容される dtype と形状を定義します。`tf.function` は、tf.TensorSpec 入力でトレースされる引数で、Python デフォルト値を受け入れます。このような引数は、`__call__` に関わる数値ハイパーパラメータのカスタマイズを可能にするために使用できます（ドロップアウト率など）。

    - Python 値の引数の場合、SavedModel の作成者が許容値を定義します。このような引数は、トレースされる関数で個別の選択を行うためのフラグとして使用できます（ただし、組み合わせによるトレースの爆発的な増加に注意してください）。

リストアされた `__call__` 関数は、すべての許容可能な引数の組み合わせのトレースを提供する必要があります。`training` の `True` と `False` を切り替えても、引数の許容範囲は変更されません。

#### 結果

`obj` の呼び出しによる `outputs` は、次のようになります。

- 単一出力の場合は、単一のテンソル
- 名前なし出力の整列されたシーケンスの場合は、テンソルのリスト
- 特定の出力名のセットでキーが設定されたテンソルの dict 型。

（このインターフェースの次期レビジョンでは、より全般的なネストが可能になる場合があります。）戻り値の型は、Python 値の kwargs によって異なる場合があるため、フラグがは追加の出力を生成できます。出力の dtype と形状、および入力への依存関係は、SavedModel の作成者が定義します。

### 名前付きのコーラブル

Reusable SavedModel は、上記に説明したように、`obj.foo`、`obj.bar`、などのように複数のモデルピースを名前付きのサブオブジェクトに入れて提供することができます。それぞれのサブオブジェクトには、モデルピースに特化した変数などに関する `__call__` メソッドと補足属性があります。上記の例では、`obj.foo.__call__`、 `obj.foo.variables`、などのようになります。

このインターフェースには、ベア tf.function を `tf.foo` として直接追加するアプローチは*ない*ことに注意してください。

Reusable SavedModel を使用する人は、1 つの階層のネストのみを処理できます（`obj.bar` を処理できますが、`obj.bar.baz` は処理できません）。（このインターフェースの将来的なレビジョンでは、より深い階層のネストを使用できるようになり、最上位オブジェクト自体がコーラブルである必要があるという要件が排除される可能性があります。）

## 最後に

### in-process API との関係

このドキュメントでは、tf.function や tf.Variable などのプリミティブ型で構成される Python クラスのインターフェースについて説明しました。これらは、`tf.saved_model.save()` と `tf.saved_model.load()` によるシリアル化が実施されるラウンドトリップに耐えることができます。ただし、インターフェースは、`tf.saved_model.save()` に渡された元のオブジェクトにすでに存在していました。このインターフェースを調整することで、単一の TensorFlow プログラム内の model-building API で、モデルピースのやり取りが可能となります。
