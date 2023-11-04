# Projetando código de modelagem do TensorFlow para o TFX

Ao projetar seu código de modelagem do TensorFlow para o TFX, há alguns itens que você deve conhecer, incluindo a escolha de uma API de modelagem.

- Consome: SavedModel de [Transform](transform.md) e dados de [ExampleGen](examplegen.md)
- Produz: modelo treinado no formato SavedModel

<aside class="note" id="tf2-support"><b>Observação:</b> o TFX oferece suporte a quase todo o TensorFlow 2.X, com pequenas exceções. O TFX também oferece suporte total ao TensorFlow 1.15.</aside>

<ul>
  <li>Novos pipelines do TFX devem usar o TensorFlow 2.x com modelos Keras por meio do <a href="https://github.com/tensorflow/community/blob/master/rfcs/20200117-tfx-generic-trainer.md">Generic Trainer</a>.</li>
  <li>O suporte total ao TensorFlow 2.X, incluindo suporte aprimorado para tf.distribute, será adicionado de forma incremental nas próximas versões.</li>
  <li>Os pipelines anteriores do TFX podem continuar usando o TensorFlow 1.15. Para migrar para o TensorFlow 2.X, veja o <a href="https://www.tensorflow.org/guide/migrate">guia de migração do TensorFlow</a>.</li>
</ul>

Para ficar em dia quando aos lançamentos do TFX, consulte o <a href="https://github.com/tensorflow/tfx/blob/master/ROADMAP.md">TFX OSS Roadmap</a>, leia o <a href="https://blog.tensorflow.org/search?label=TFX&amp;max-results=20">blog do TFX</a> e assine o <a href="https://services.google.com/fb/forms/tensorflow/">boletim informativo do TensorFlow</a>.




A camada de entrada do seu modelo deve consumir do SavedModel que foi criado por um componente [Transform](transform.md), e as camadas do modelo Transform devem ser incluídas no seu modelo de forma que, quando você exportar seu SavedModel e EvalSavedModel, eles incluam as transformações que foram criadas pelo componente [Transform](transform.md).

Um típico projeto de modelo do TensorFlow para TFX está mostrado a seguir:

```python
def _build_estimator(tf_transform_dir,                      config,                      hidden_units=None,                      warm_start_from=None):   """Build an estimator for predicting the tipping behavior of taxi riders.    Args:     tf_transform_dir: directory in which the tf-transform model was written       during the preprocessing step.     config: tf.contrib.learn.RunConfig defining the runtime environment for the       estimator (including model_dir).     hidden_units: [int], the layer sizes of the DNN (input layer first)     warm_start_from: Optional directory to warm start from.    Returns:     Resulting DNNLinearCombinedClassifier.   """   metadata_dir = os.path.join(tf_transform_dir,                               transform_fn_io.TRANSFORMED_METADATA_DIR)   transformed_metadata = metadata_io.read_metadata(metadata_dir)   transformed_feature_spec = transformed_metadata.schema.as_feature_spec()    transformed_feature_spec.pop(_transformed_name(_LABEL_KEY))    real_valued_columns = [       tf.feature_column.numeric_column(key, shape=())       for key in _transformed_names(_DENSE_FLOAT_FEATURE_KEYS)   ]   categorical_columns = [       tf.feature_column.categorical_column_with_identity(           key, num_buckets=_VOCAB_SIZE + _OOV_SIZE, default_value=0)       for key in _transformed_names(_VOCAB_FEATURE_KEYS)   ]   categorical_columns += [       tf.feature_column.categorical_column_with_identity(           key, num_buckets=_FEATURE_BUCKET_COUNT, default_value=0)       for key in _transformed_names(_BUCKET_FEATURE_KEYS)   ]   categorical_columns += [       tf.feature_column.categorical_column_with_identity(           key, num_buckets=num_buckets, default_value=0)       for key, num_buckets in zip(           _transformed_names(_CATEGORICAL_FEATURE_KEYS),  #           _MAX_CATEGORICAL_FEATURE_VALUES)   ]   return tf.estimator.DNNLinearCombinedClassifier(       config=config,       linear_feature_columns=categorical_columns,       dnn_feature_columns=real_valued_columns,       dnn_hidden_units=hidden_units or [100, 70, 50, 25],       warm_start_from=warm_start_from)
```
