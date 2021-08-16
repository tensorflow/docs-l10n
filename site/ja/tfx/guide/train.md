# TFX 向けの TensorFlow モデリングコードを設計する

TFX 用の TensorFlow モデリングコードを設計する場合、モデリング API の選択など、注意すべき点がいくつかあります。

- 入力: [Transform](transform.md) の SavedModel と [ExampleGen](examplegen.md) のデータ
- 出力: SavedModel 形式のトレーニング済みモデル

<aside class="note" id="tf2-support"><b>注意:</b> TFX は、小さな例外を除き、ほぼすべての TensorFlow 2.X をサポートしています。TFX は TensorFlow 1.15 も完全にサポートしています。</aside>

<ul>
  <li>新しい TFX パイプラインは、<a href="https://github.com/tensorflow/community/blob/master/rfcs/20200117-tfx-generic-trainer.md">汎用の Trainer</a> を介して Keras モデルで TensorFlow 2.x を使用する必要があります。</li>
  <li>tf.distribute のサポート改善を含み、TensorFlow 2.X のフルサポートは次期リリースで段階的に追加される予定です。</li>
  <li>以前の TFX パイプラインは引き続き TensorFlow 1.15 を使用できます。TensorFlow 2.X に切り替えるには、<a href="https://www.tensorflow.org/guide/migrate">TensorFlow の移行ガイド</a>をご覧ください。</li>
</ul>

TFX リリースに関する最新情報は、<a href="https://github.com/tensorflow/tfx/blob/master/ROADMAP.md">TFX OSS ロードマップ</a> と <a href="https://blog.tensorflow.org/search?label=TFX&amp;max-results=20">TFX ブログ</a>をご確認ください。また、<a href="https://services.google.com/fb/forms/tensorflow/">TensorFlow ニュースレター</a>もご購読ください。




モデルの入力レイヤーは、[Transform](transform.md) コンポーネントが作成した SavedModel から消費する必要があります。また、SavedModel と EvalSavedModel をエクスポートする際に [Transform](transform.md) コンポーネントが作成した変換が含まれるように、Transform モデルのレイヤーがモデルに含まれている必要があります。

TFX 向けの典型的な TensorFlow モデルの設計は次のようになります。

```python
def _build_estimator(tf_transform_dir,                      config,                      hidden_units=None,                      warm_start_from=None):   """Build an estimator for predicting the tipping behavior of taxi riders.    Args:     tf_transform_dir: directory in which the tf-transform model was written       during the preprocessing step.     config: tf.contrib.learn.RunConfig defining the runtime environment for the       estimator (including model_dir).     hidden_units: [int], the layer sizes of the DNN (input layer first)     warm_start_from: Optional directory to warm start from.    Returns:     Resulting DNNLinearCombinedClassifier.   """   metadata_dir = os.path.join(tf_transform_dir,                               transform_fn_io.TRANSFORMED_METADATA_DIR)   transformed_metadata = metadata_io.read_metadata(metadata_dir)   transformed_feature_spec = transformed_metadata.schema.as_feature_spec()    transformed_feature_spec.pop(_transformed_name(_LABEL_KEY))    real_valued_columns = [       tf.feature_column.numeric_column(key, shape=())       for key in _transformed_names(_DENSE_FLOAT_FEATURE_KEYS)   ]   categorical_columns = [       tf.feature_column.categorical_column_with_identity(           key, num_buckets=_VOCAB_SIZE + _OOV_SIZE, default_value=0)       for key in _transformed_names(_VOCAB_FEATURE_KEYS)   ]   categorical_columns += [       tf.feature_column.categorical_column_with_identity(           key, num_buckets=_FEATURE_BUCKET_COUNT, default_value=0)       for key in _transformed_names(_BUCKET_FEATURE_KEYS)   ]   categorical_columns += [       tf.feature_column.categorical_column_with_identity(           key, num_buckets=num_buckets, default_value=0)       for key, num_buckets in zip(           _transformed_names(_CATEGORICAL_FEATURE_KEYS),  #           _MAX_CATEGORICAL_FEATURE_VALUES)   ]   return tf.estimator.DNNLinearCombinedClassifier(       config=config,       linear_feature_columns=categorical_columns,       dnn_feature_columns=real_valued_columns,       dnn_hidden_units=hidden_units or [100, 70, 50, 25],       warm_start_from=warm_start_from)
```
