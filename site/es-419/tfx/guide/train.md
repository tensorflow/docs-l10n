# Diseño de código de modelado de  TensorFlow para TFX

Al diseñar su código de modelado de TensorFlow para TFX, hay algunos elementos que debe tener en cuenta, incluida la elección de una API de modelado.

- Consume: SavedModel de [Transform](transform.md) y datos de [ExampleGen](examplegen.md)
- Emite: modelo entrenado en formato SavedModel

<aside class="note" id="tf2-support"><b>Nota:</b> TFX es casi completamente compatible con TensorFlow 2.X, con mínimas excepciones. TFX también es totalmente compatible con TensorFlow 1.15.</aside>

<ul>
  <li>Las nuevas canalizaciones de TFX deben usar TensorFlow 2.x con modelos Keras a través del <a href="https://github.com/tensorflow/community/blob/master/rfcs/20200117-tfx-generic-trainer.md">Trainer genérico</a>.</li>
  <li>La compatibilidad total con TensorFlow 2.X, incluida la compatibilidad mejorada con tf.distribute, se agregará gradualmente en las próximas versiones.</li>
  <li>Las canalizaciones de TFX anteriores pueden seguir usando TensorFlow 1.15. Para cambiarse a TensorFlow 2.X, consulte la <a href="https://www.tensorflow.org/guide/migrate">guía de migración de TensorFlow</a>.</li>
</ul>

Para estar al día con las últimas actualizaciones de TFX, consulte la <a href="https://github.com/tensorflow/tfx/blob/master/ROADMAP.md">hoja de ruta OSS de TFX </a>, lea el <a href="https://blog.tensorflow.org/search?label=TFX&amp;max-results=20">blog de TFX</a> y suscríbase al <a href="https://services.google.com/fb/forms/tensorflow/">boletín informativo de TensorFlow</a>.




La capa de entrada de su modelo debe consumir del SavedModel que fue creado por un componente [Transform](transform.md), y las capas del modelo Transform deben incluirse con su modelo para que cuando exporte su SavedModel y su EvalSavedModel incluyan las transformaciones que creó el componente [Transform](transform.md).

Un diseño de modelo típico de TensorFlow para TFX se ve así:

```python
def _build_estimator(tf_transform_dir,                      config,                      hidden_units=None,                      warm_start_from=None):   """Build an estimator for predicting the tipping behavior of taxi riders.    Args:     tf_transform_dir: directory in which the tf-transform model was written       during the preprocessing step.     config: tf.contrib.learn.RunConfig defining the runtime environment for the       estimator (including model_dir).     hidden_units: [int], the layer sizes of the DNN (input layer first)     warm_start_from: Optional directory to warm start from.    Returns:     Resulting DNNLinearCombinedClassifier.   """   metadata_dir = os.path.join(tf_transform_dir,                               transform_fn_io.TRANSFORMED_METADATA_DIR)   transformed_metadata = metadata_io.read_metadata(metadata_dir)   transformed_feature_spec = transformed_metadata.schema.as_feature_spec()    transformed_feature_spec.pop(_transformed_name(_LABEL_KEY))    real_valued_columns = [       tf.feature_column.numeric_column(key, shape=())       for key in _transformed_names(_DENSE_FLOAT_FEATURE_KEYS)   ]   categorical_columns = [       tf.feature_column.categorical_column_with_identity(           key, num_buckets=_VOCAB_SIZE + _OOV_SIZE, default_value=0)       for key in _transformed_names(_VOCAB_FEATURE_KEYS)   ]   categorical_columns += [       tf.feature_column.categorical_column_with_identity(           key, num_buckets=_FEATURE_BUCKET_COUNT, default_value=0)       for key in _transformed_names(_BUCKET_FEATURE_KEYS)   ]   categorical_columns += [       tf.feature_column.categorical_column_with_identity(           key, num_buckets=num_buckets, default_value=0)       for key, num_buckets in zip(           _transformed_names(_CATEGORICAL_FEATURE_KEYS),  #           _MAX_CATEGORICAL_FEATURE_VALUES)   ]   return tf.estimator.DNNLinearCombinedClassifier(       config=config,       linear_feature_columns=categorical_columns,       dnn_feature_columns=real_valued_columns,       dnn_hidden_units=hidden_units or [100, 70, 50, 25],       warm_start_from=warm_start_from)
```
