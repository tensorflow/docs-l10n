# 为 TFX 设计 TensorFlow 建模代码

在为 TFX 设计 TensorFlow 建模代码时，有一些事项需要注意，包括建模 API 的选择。

- 使用：来自 [Transform](transform.md) 的 SavedModel 和来自 [ExampleGen](examplegen.md) 的数据
- 发出：SavedModel 格式的训练模型

<aside class="note" id="tf2-support"><b>注</b>：除了少数例外，TFX 支持几乎所有 TensorFlow 2.X。TFX 还完全支持 TensorFlow 1.15。</aside>

<ul>
  <li>新的 TFX 流水线应通过<a href="https://github.com/tensorflow/community/blob/master/rfcs/20200117-tfx-generic-trainer.md">通用 Trainer</a> 将 TensorFlow 2.x 与 Keras 模型一起使用。</li>
  <li>对 TensorFlow 2.X 的全面支持（包括对 tf.distribute 的改进支持）将在接下来的版本中逐步添加。</li>
  <li>之前的 TFX 流水线可以继续使用 TensorFlow 1.15。要将它们切换到 TensorFlow 2.X，请参阅 <a href="https://www.tensorflow.org/guide/migrate">TensorFlow 迁移指南</a>。</li>
</ul>

要随时掌握 TFX 版本的最新消息，请参阅 <a href="https://github.com/tensorflow/tfx/blob/master/ROADMAP.md">TFX OSS 路线图</a>，阅读 <a href="https://blog.tensorflow.org/search?label=TFX&amp;max-results=20">TFX 博客</a>并订阅 <a href="https://services.google.com/fb/forms/tensorflow/">TensorFlow 简报</a>。




您的模型的输入层应使用由 [Transform](transform.md) 组件创建的 SavedModel，并且应将 Transform 模型的层包含在您的模型中，以便在导出 SavedModel 和 EvalSavedModel 时，它们将包含由 [Transform](transform.md) 组件创建的转换。

TFX 的典型 TensorFlow 模型设计如下所示：

```python
def _build_estimator(tf_transform_dir,                      config,                      hidden_units=None,                      warm_start_from=None):   """Build an estimator for predicting the tipping behavior of taxi riders.    Args:     tf_transform_dir: directory in which the tf-transform model was written       during the preprocessing step.     config: tf.contrib.learn.RunConfig defining the runtime environment for the       estimator (including model_dir).     hidden_units: [int], the layer sizes of the DNN (input layer first)     warm_start_from: Optional directory to warm start from.    Returns:     Resulting DNNLinearCombinedClassifier.   """   metadata_dir = os.path.join(tf_transform_dir,                               transform_fn_io.TRANSFORMED_METADATA_DIR)   transformed_metadata = metadata_io.read_metadata(metadata_dir)   transformed_feature_spec = transformed_metadata.schema.as_feature_spec()    transformed_feature_spec.pop(_transformed_name(_LABEL_KEY))    real_valued_columns = [       tf.feature_column.numeric_column(key, shape=())       for key in _transformed_names(_DENSE_FLOAT_FEATURE_KEYS)   ]   categorical_columns = [       tf.feature_column.categorical_column_with_identity(           key, num_buckets=_VOCAB_SIZE + _OOV_SIZE, default_value=0)       for key in _transformed_names(_VOCAB_FEATURE_KEYS)   ]   categorical_columns += [       tf.feature_column.categorical_column_with_identity(           key, num_buckets=_FEATURE_BUCKET_COUNT, default_value=0)       for key in _transformed_names(_BUCKET_FEATURE_KEYS)   ]   categorical_columns += [       tf.feature_column.categorical_column_with_identity(           key, num_buckets=num_buckets, default_value=0)       for key, num_buckets in zip(           _transformed_names(_CATEGORICAL_FEATURE_KEYS),  #           _MAX_CATEGORICAL_FEATURE_VALUES)   ]   return tf.estimator.DNNLinearCombinedClassifier(       config=config,       linear_feature_columns=categorical_columns,       dnn_feature_columns=real_valued_columns,       dnn_hidden_units=hidden_units or [100, 70, 50, 25],       warm_start_from=warm_start_from)
```
