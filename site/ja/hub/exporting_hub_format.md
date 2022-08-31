<!--* freshness: { owner: 'maringeo' reviewed: '2022-04-09' review_interval: '6 months' } *-->

# TF1 Hub 形式でモデルをエクスポートする

この形式の詳細については [TF1 Hub 形式](tf1_hub_module.md)をご覧ください。

## 互換性に関するメモ

TF1 Hub 形式は、TensorFlow 1 に対応するように作成されています。TensorFlow 2 の TF Hub では一部しかサポートされません。[モデルのエクスポート](exporting_tf2_saved_model)に従って、新しい [TF2 SavedModel](tf2_saved_model.md) 形式での公開をぜひ検討してください。

TF1 Hub 形式は、TensorFlow 1 の SavedModel 形式と構文レベルで似ていますが（同じファイル名とプロトコルメッセージ）、セマンティックレベルでは、モジュールの再利用、合成および再トレーニングが可能という点で異なります（リソースイニシャライザのストレージが異なる、メタグラフのタグ規則が異なるなど）。ディスク上で区別するには、<code>tfhub_module.pb</code>ファイルの有無を確認することが最も簡単です。

## 一般的なアプローチ

新しいモジュールを定義するには、パブリッシャーは `module_fn` 関数を使用して `hub.create_module_spec()` を呼び出します。この関数は、呼び出し元が指定する入力に `tf.placeholder()` を使用して、モデルの内部構造を表すグラフを構築し、`hub.add_signature(name, inputs, outputs)` を 1 回以上呼び出してシグネチャを定義します。

例を示します。

```python
def module_fn():
  inputs = tf.placeholder(dtype=tf.float32, shape=[None, 50])
  layer1 = tf.layers.dense(inputs, 200)
  layer2 = tf.layers.dense(layer1, 100)
  outputs = dict(default=layer2, hidden_activations=layer1)
  # Add default signature.
  hub.add_signature(inputs=inputs, outputs=outputs)

...
spec = hub.create_module_spec(module_fn)
```

特定の TensorFlow グラフ内のオブジェクトをインスタンス化するには、パスの代わりに `hub.create_module_spec()` の結果が使用される場合があります。この場合、チェックポイントはなく、モジュールインスタンスは代わりに変数イニシャライザを使用します。

モジュールインスタンスは、`export(path, session)` メソッドを使用してディスクにシリアル化されます。モジュールをエクスポートすると、その定義は `session` にある変数のその時点の状態とともに、渡されるパスにシリアル化されます。これは、初めてモジュールをエクスポートするときだけでなく、ファインチューニングしたモジュールをエクスポートする際にも使用できます。

TensorFlow Estimator との互換性を得るため、`tf.estimator.LatestExporter` が最新のチェックポイントからモデル全体をエクスポートするのと同様に、`hub.LatestModuleExporter` も最新のチェックポイントからモジュールをエクスポートします。

モジュールのパブリッシャーは、可能な限り[共通シグネチャ](common_signatures/index.md)を実装することで、消費者がモジュールを簡単に交換して、問題に最適なものを見つけ出せるようにする必要があります。

## 実際の例

[テキスト埋め込みモジュールのエクスポーター](https://github.com/tensorflow/hub/blob/master/examples/text_embeddings/export.py)をご覧ください。一般的なテキスト埋め込み形式からモジュールを生成する実世界の例を紹介しています。

## パブリッシャー対象の注意事項

コンシューマがファインチューニングを簡単に行えるように、次のことに注意してください。

- ファインチューニングには正則化が必要です。モデルは `REGULARIZATION_LOSSES` コレクションとともにエクスポートされており、これによって、`tf.layers.dense(..., kernel_regularizer=...)` などの選択肢が、コンシューマが `tf.losses.get_regularization_losses()` から取得するものへと変化します。L1/L2 正則化損失を定義するこの方法をお勧めします。

- パブリッシャーのモデルでは、`tf.train.FtrlOptimizer`、`tf.train.ProximalGradientDescentOptimizer`、およびその他の近似オプティマイザの `l1_` と`l2_regularization_strength` パラメータを使って L1/L2 正則化を定義しないようにしてください。これらはモジュールとともにエクスポートされないため、正則化の強度をグローバルに設定することはコンシューマに適していない場合があります。ワイド（スパース線形）またはワイド＆ディープモデルでの L1 正則化を除き、代わりに個別の正則化損失を使用することが可能なはずです。

- ドロップアウト、バッチ正規化、または似たようなトレーニング技法を使用する場合、ハイパーパラメータを多くの必要な使用例全体で意味がある値に設定してください。ドロップアウト率は、オーバーフィットする傾向があるターゲットの問題に対して調整する必要がある場合があります。バッチ正規化では、運動量（崩壊定数としても知られています）は小さなデータセット、大きなバッチ、またはその両方でファインチューニングできるよう小さくする必要があります。高度なコンシューマは、重要なハイパーパラメータをコントロールする署名の追加を検討してください。
