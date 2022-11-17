# TF1.x から TF2 への移行の概要

TensorFlow 2 は、いくつかの点で TF1.x とは根本的に異なります。次のようにすると、未変更の TF1.x コード（[contrib を除く](https://github.com/tensorflow/community/blob/master/rfcs/20180907-contrib-sunset.md)）を TF2 バイナリインストールで引き続き実行できます。

```python
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
```

ただし、これは TF2 の動作と API を*実行していない*ため、TF2 用に記述されたコードでは期待どおりに動作しない可能性があります。TF2 の動作をアクティブにして実行していない場合は、TF2 インストールの上で TF1.x を実行していることになります。TF2 と TF1.x の違いの詳細については、[TF1 と TF2 の動作ガイド](./tf1_vs_tf2.ipynb)を参照してください。

このガイドでは、TF1.x コードを TF2 に移行するプロセスの概要を説明します。これにより、新機能および今後の機能改善を利用できるようになります。また、コードがよりシンプルになり、パフォーマンスが向上し、メンテナンスが容易になります。

`tf.keras` の高レベル API を使用し、`model.fit` のみでトレーニングしている場合、コードは次の場合を除いて、TF2 とほぼ完全に互換性があるはずです。

- TF2 には、Keras オプティマイザの新しい[デフォルトの学習率](../../guide/effective_tf2.ipynb#optimizer_defaults)があります。
- TF2 では、指標がログされる「名前」が[変更されている可能性があります](../../guide/effective_tf2.ipynb#keras_metric_names)。

## TF2 移行プロセス

移行する前に、[ガイド](./tf1_vs_tf2.ipynb)を読んで、TF1.x と TF2 の動作と API の違いについて学習してください。

1. 自動スクリプトを実行して、TF1.x API の使用の一部を `tf.compat.v1` に変換します。
2. 古い `tf.contrib` シンボルを削除します（[TF Addons](https://github.com/tensorflow/addons) と [TF-Slim](https://github.com/google-research/tf-slim) を確認してください）。
3. Eager execution を有効にして、TF1.x モデルのフォワードパスを TF2 で実行します。
4. トレーニングループとモデルの保存/読み込み用の TF1.x コードを TF2 の同等のものにアップグレードします。
5. （オプション）TF2 互換の `tf.compat.v1` API を慣用的な TF2 API に移行します。

次のセクションでは、上記の手順を詳しく説明します。

## シンボル変換スクリプトを実行する

これは、TF 2.x バイナリに対して実行するようにコードシンボルを書き換える際に初期パスを実行しますが、これによりコードが TF 2.x に対して慣用的になったり、自動的にコードが TF2 の動作と互換性を持つことはありません。

多くの場合、コードは、プレースホルダー、セッション、コレクション、およびその他の TF1.x スタイルの機能にアクセスするために、`tf.compat.v1` エンドポイントを引き続き使用します。

シンボル変換スクリプトを使用するためのベストプラクティスの詳細については、[ガイド](./upgrade.ipynb)を参照してください。

## `tf.contrib` の使用を削除

`tf.contrib` モジュールは廃止され、そのサブモジュールのいくつかが Core TF2 API に統合されました。他のサブモジュールは、[TF IO](https://github.com/tensorflow/io) や [TF Addons](https://www.tensorflow.org/addons/overview) などの他のプロジェクトにスピンオフされました。

古い TF1.x コードの多くは、`tf.contrib.layers` として TF1.x にパッケージ化された [Slim](https://ai.googleblog.com/2016/08/tf-slim-high-level-library-to-define.html) ライブラリを使用しています。Slim コードを TF2 に移行する場合は、使用する Slim API が [tf-slim pip パッケージ](https://pypi.org/project/tf-slim/)を指すように切り替えます。その後、[モデルマッピングガイド](https://tensorflow.org/guide/migrate/model_mapping#a_note_on_slim_and_contriblayers)を読み、Slim コードを変換する方法を学びます。

Slim 事前トレーニング済みモデルを使用する場合は、`tf.keras.applications` から Keras 事前トレーニング済みモデル、または元の Slim コードからエクスポートされた [TF Hub](https://tfhub.dev/s?tf-version=tf2&q=slim) の TensorFlow 2 `SavedModel` をお試しください。

## TF2 動作を有効にして TF1.x モデルのフォワードパスを実行する

### 変数と損失を追跡する

[TF2 はグローバルコレクションをサポートしていません。](./tf1_vs_tf2.ipynb#no_more_globals)

TF2 の Eager execution は、`tf.Graph` コレクションベースの API をサポートしていません。これは、変数の作成方法と追跡方法に影響します。

新しい TF2 コードでは、`v1.get_variable` の代わりに `tf.Variable` を使用し、`tf.compat.v1.variable_scope` の代わりに Python オブジェクトを使用して変数を収集および追跡します。通常、これは次のいずれかになります。

- `tf.keras.layers.Layer`
- `tf.keras.Model`
- `tf.Module`

`tf.Graph.get_collection(tf.GraphKeys.VARIABLES)` などの変数のリストを集める必要がある場合には、`Layer`、`Module`、および `Model` オブジェクトの `.variables` と `.trainable_variables` 属性を使用します。

これら `Layer` と `Model` クラスは、グローバルコレクションの必要性を除去した別のプロパティを幾つか実装します。`.losses` プロパティは、`tf.GraphKeys.LOSSES` コレクション使用の置き換えとなります。

TF2 コードモデリング shim を使用して既存の `get_variable` および `variable_scope` ベースのコードを  `Layers`、`Models`、および `Modules` 内に埋め込む方法の詳細については、[モデルマッピングガイド](./model_mapping.ipynb)を参照してください。これにより、大幅な書き換えを行うことなく、Eager execution を有効にしてフォワードパスを実行できます。

### 他の動作の変更への適応

モデルフォワードパスを実行して他の動作変更を実行するのに[モデルマッピングガイド](./model_mapping.ipynb)だけでは不十分な場合は、[TF1.x と TF2 の動作の比較](./tf1_vs_tf2.ipynb)に関するガイドを参照してください。他の動作の変更と、それらに適応する方法について学べます。詳細については、[サブクラス化ガイドによる新しいレイヤーとモデルの作成](https://tensorflow.org/guide/keras/custom_layers_and_models.ipynb)も参照してください。

### 結果の検証

イーガー実行が有効な場合にモデルが正しく動作していることを （数値的に）検証する方法に関する簡単なツールとガイダンスについては、[モデル検証ガイド](./validate_correctness.ipynb)を参照してください。[モデルマッピングガイド](./model_mapping.ipynb)と合わせて参照することをお勧めします。

## トレーニング、評価、インポート/エクスポートコードのアップグレード

`v1.Session` スタイルの `tf.estimator.Estimator` およびその他のコレクションベースのアプローチで構築された TF1.x トレーニングループは、TF2 の新しい動作と互換性がありません。TF2 コードと組み合わせると予期しない動作が発生する可能性があるため、すべての TF1.x トレーニングコードを移行することが重要です。

これを行うには、いくつかの方法から選択できます。

最高レベルのアプローチは、`tf.keras` を使用することです。Keras の高レベル関数は、独自のトレーニングループを記述する場合に見逃しやすい多くの低レベルの詳細を管理します。たとえば、それらは自動的に正規化損失を収集し、モデルを呼び出すときに `training=True` 引数を設定します。

`tf.estimator.Estimator` のコードを移行して[バニラ](./migrating_estimator.ipynb#tf2_keras_training_api)および[カスタム](./migrating_estimator.ipynb#tf2_keras_training_api_with_custom_training_step) `tf.keras` トレーニングループを使用する方法については [Estimator 移行ガイド](./migrating_estimator.ipynb)を参照してください。

カスタムトレーニングループを使用すると、個々のレイヤーの重みを追跡するなど、モデルをより細かく制御できます。`tf.GradientTape` を使用してモデルの重みを取得し、それらを使用してモデルを更新する方法については、[ゼロからトレーニングループを構築する](https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch)ガイドを参照してください。

### TF1.x オプティマイザを Keras オプティマイザに変換する

[Adam オプティマイザ](https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/AdamOptimizer)や [勾配降下オプティマイザ](https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/GradientDescentOptimizer)などの `tf.compat.v1.train` 内のオプティマイザには、`tf.keras.optimizers` 内に同等のものをもちます。

以下の表は、これらのレガシーオプティマイザを Keras の同等のものに変換する方法をまとめたものです。追加の手順（[デフォルトの学習率の更新](../../guide/effective_tf2.ipynb#optimizer_defaults)など）が必要でない限り、TF1.x バージョンを TF2 バージョンに直接置き換えることができます。

オプティマイザをアップグレードすると、古いチェックポイントに互換性がなくなる可能性があるので注意してください。

<table>
  <tr>
    <th>TF1.x</th>
    <th>TF2</th>
    <th>追加の手順</th>
  </tr>
  <tr>
    <td>`tf.v1.train.GradientDescentOptimizer`</td>
    <td>`tf.keras.optimizers.SGD`</td>
    <td>なし</td>
  </tr>
  <tr>
    <td>`tf.v1.train.MomentumOptimizer`</td>
    <td>`tf.keras.optimizers.SGD`</td>
    <td>`momentum` 引数を含める</td>
  </tr>
  <tr>
    <td>`tf.v1.train.AdamOptimizer`</td>
    <td>`tf.keras.optimizers.Adam`</td>
    <td>`beta1` および `beta2` 引数の名前を `beta_1` および `beta_2` に変更する</td>
  </tr>
  <tr>
    <td>`tf.v1.train.RMSPropOptimizer`</td>
    <td>`tf.keras.optimizers.RMSprop`</td>
    <td>`decay` 引数の名前を `rho` に変更する</td>
  </tr>
  <tr>
    <td>`tf.v1.train.AdadeltaOptimizer`</td>
    <td>`tf.keras.optimizers.Adadelta`</td>
    <td>なし</td>
  </tr>
  <tr>
    <td>`tf.v1.train.AdagradOptimizer`</td>
    <td>`tf.keras.optimizers.Adagrad`</td>
    <td>なし</td>
  </tr>
  <tr>
    <td>`tf.v1.train.FtrlOptimizer`</td>
    <td>`tf.keras.optimizers.Ftrl`</td>
    <td>`accum_name` および `linear_name` 引数を削除する</td>
  </tr>
  <tr>
    <td>`tf.contrib.AdamaxOptimizer`</td>
    <td>`tf.keras.optimizers.Adamax`</td>
    <td>`beta1` および `beta2` 引数の名前を `beta_1` および `beta_2` に変更する</td>
  </tr>
  <tr>
    <td>`tf.contrib.Nadam`</td>
    <td>`tf.keras.optimizers.Nadam`</td>
    <td>`beta1` および `beta2` 引数の名前を `beta_1` および `beta_2` に変更する</td>
  </tr>
</table>

注意: TF2 では、すべてのイプシロン（数値安定定数）のデフォルトが `1e-8` ではなく `1e-7` になりました。ほとんどの場合、この違いは無視できます。

### データ入力パイプラインをアップグレードする

データをtf.kerasモデルに供給するには多くの方法があります。それらは入力として Python ジェネレータと Numpy 配列を受け取ります。

モデルへのデータ供給方法として推奨しているのは、データ操作用の高パフォーマンスなクラスのコレクションを含んだ `tf.data` パッケージの使用です。`tf.data` に属する `dataset` は効率的で表現力があり、TF2 と統合できます。

次のように、tf.keras.Model.fit メソッドに直接渡すことができます。

```python
model.fit(dataset, epochs=5)
```

また、標準的な Python で直接イテレートすることもできます。

```python
for example_batch, label_batch in dataset:
    break
```

依然としてtf.queueを使用している場合、これらは入力パイプラインとしてではなく、データ構造としてのみ対応されます。

`tf.feature_columns` を使用する特徴前処理コードもすべて移行する必要があります。詳細については、[移行ガイド](./migrating_feature_columns.ipynb)を参照してください。

### モデルの保存と読み込み

TF2 は、オブジェクトベースのチェックポイントを使用します。名前ベースの TF1.x チェックポイントからの移行についての詳細は、[チェックポイント移行ガイド](./migrating_checkpoints.ipynb)を参照してください。TensorFlow Core ドキュメントの[チェックポイントガイド](https://www.tensorflow.org/guide/checkpoint)も参照してください。

保存されたモデルには、重大な互換性の問題はありません。TF1.x の `SavedModel` を TF2 に移行する方法についての詳細は、<a href="./saved_model.ipynb" data-md-type="link">`SavedModel` ガイド</a>を参照してください。一般に、

- TF1.x の saved_models は TF2 で機能します。
- すべての ops がサポートされている場合、TF2 saved_models は TF1.x で機能します。

`Graph.pb` および `Graph.pbtxt` オブジェクトの操作の詳細については、`SavedModel` 移行ガイドの [`GraphDef` セクション](./saved_model.ipynb#graphdef_and_metagraphdef)も参照してください。

## （オプション）`tf.compat.v1` シンボルを移行する

`tf.compat.v1` モジュールには、元のセマンティクスを持つ完全な TF1.x API が含まれています。

上記の手順に従って、すべての TF2 動作と完全に互換性のあるコードになった後でも、TF2 と互換性のある `compat.v1` API について多くの言及がある可能性があります。これらの従来の `compat.v1` API は、作成済みのコードに対して引き続き機能しますが、新しくコードを作成する場合は使用しないでください。

ただし、既存の使用法を非レガシー TF2 API に移行することがあるかもしれません。多くの場合、個々の `compat.v1` シンボルのドキュメント文字列は、それらを非レガシー TF2 API に移行する方法を説明しています。また、[慣用的な TF2 API への増分移行に関するモデルマッピングガイドのセクション](./model_mapping.ipynb#incremental_migration_to_native_tf2)も参照してください。

## リソースとその他の文献

前述のとおり、すべての TF1.x コードを TF2 に移行することを推薦します。詳細については、TensorFlow ガイドの [TF2 の移行セクション](https://tensorflow.org/guide/migrate)を参照してください。
