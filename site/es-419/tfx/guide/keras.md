# TensorFlow 2.x en TFX

[TensorFlow 2.0 se publicó en 2019](https://blog.tensorflow.org/2019/09/tensorflow-20-is-now-available.html), con [una estrecha integración de Keras](https://www.tensorflow.org/guide/keras/overview), [ejecución eager](https://www.tensorflow.org/guide/eager) predeterminada y [ejecución de funciones Pythónicas](https://www.tensorflow.org/guide/function), entre otras [nuevas características y mejoras](https://www.tensorflow.org/guide/effective_tf2#a_brief_summary_of_major_changes).

Esta guía ofrece una descripción técnica completa de TF 2.x en TFX.

## ¿Qué versión debemos usar?

TFX es compatible con TensorFlow 2.x y las API de alto nivel que existían en TensorFlow 1.x (en particular, Estimators) continúan funcionando.

### Cómo comenzar nuevos proyectos en TensorFlow 2.x

Dado que TensorFlow 2.x conserva las capacidades de alto nivel de TensorFlow 1.x, no hay ninguna ventaja en usar la versión anterior en proyectos nuevos, incluso si no planea usar las nuevas funciones.

Por lo tanto, si está comenzando un nuevo proyecto de TFX, le recomendamos que use TensorFlow 2.x. Quizás desee actualizar su código más adelante cuando esté disponible la compatibilidad completa con Keras y otras nuevas características, y el alcance de los cambios será mucho más limitado si comienza con TensorFlow 2.x, en lugar de intentar actualizar desde TensorFlow 1.x en el futuro.

### Cómo convertir proyectos existentes a TensorFlow 2.x

El código escrito para TensorFlow 1.x es ampliamente compatible con TensorFlow 2.x y seguirá funcionando en TFX.

Sin embargo, si desea aprovechar las mejoras y nuevas características a medida que estén disponibles en TF 2.x, puede seguir las [instrucciones para migrar a TF 2.x.](https://www.tensorflow.org/guide/migrate)

## Estimator

La API Estimator se mantuvo en TensorFlow 2.x, pero no es el foco de las nuevas características ni del desarrollo. El código escrito en TensorFlow 1.x o 2.x que use instancias de Estimator seguirá funcionando como se espera en TFX.

A continuación, se muestra un ejemplo de TFX de extremo a extremo donde se usa un Estimator puro: [Ejemplo de taxi (Estimator)](https://github.com/tensorflow/tfx/blob/r0.21/tfx/examples/chicago_taxi_pipeline/taxi_utils.py)

## Keras con `model_to_estimator`

Los modelos de Keras se pueden envolver con la función `tf.keras.estimator.model_to_estimator`, que les permite trabajar como si fueran instancias de Estimator. Para usar esto, siga estos pasos:

1. Compile un modelo de Keras.
2. Pase el modelo compilado a `model_to_estimator`.
3. Use el resultado de `model_to_estimator` en Trainer, de la misma manera que normalmente utilizaría Estimator.

```py
# Build a Keras model.
def _keras_model_builder():
  """Creates a Keras model."""
  ...

  model = tf.keras.Model(inputs=inputs, outputs=output)
  model.compile()

  return model


# Write a typical trainer function
def trainer_fn(trainer_fn_args, schema):
  """Build the estimator, using model_to_estimator."""
  ...

  # Model to estimator
  estimator = tf.keras.estimator.model_to_estimator(
      keras_model=_keras_model_builder(), config=run_config)

  return {
      'estimator': estimator,
      ...
  }
```

Salvo por el archivo del módulo de usuario de Trainer, el resto del proceso permanece sin cambios.

## Keras nativo (es decir, Keras sin `model_to_estimator`)

Nota: La plena compatibilidad con todas las características de Keras se encuentra en fase de desarrollo, en la mayoría de los casos, Keras funciona correctamente en TFX. Todavía no funciona con características dispersas para FeatureColumns.

### Ejemplos y Colab

Estos son varios ejemplos con Keras nativo:

- [Penguin](https://github.com/tensorflow/tfx/blob/master/tfx/examples/penguin/penguin_pipeline_local.py) ([archivo de módulo](https://github.com/tensorflow/tfx/blob/master/tfx/examples/penguin/penguin_utils_keras.py)): Ejemplo de 'Hola mundo' de extremo a extremo.
- [MNIST](https://github.com/tensorflow/tfx/blob/master/tfx/examples/mnist/mnist_pipeline_native_keras.py) ([archivo de módulo](https://github.com/tensorflow/tfx/blob/master/tfx/examples/mnist/mnist_utils_native_keras.py)): ejemplo de imagen y TFLite de extremo a extremo.
- [Taxi](https://github.com/tensorflow/tfx/blob/master/tfx/examples/chicago_taxi_pipeline/taxi_pipeline_native_keras.py) ([archivo de módulo](https://github.com/tensorflow/tfx/blob/master/tfx/examples/chicago_taxi_pipeline/taxi_utils_native_keras.py)): ejemplo con uso avanzado de Transform de extremo a extremo.

También tenemos un [Keras Colab](https://www.tensorflow.org/tfx/tutorials/tfx/components_keras) por componente.

### Componentes de TFX

Las siguientes secciones explican cómo los componentes de TFX relacionados admiten Keras nativo.

#### Transform

Actualmente, Transform ofrece compatibilidad experimental para los modelos Keras.

El componente Transform en sí se puede usar sin cambios en Keras nativos. La definición de `preprocessing_fn` sigue siendo la misma y usa las operaciones [TensorFlow](https://www.tensorflow.org/api_docs/python/tf) y [tf.Transform](https://www.tensorflow.org/tfx/transform/api_docs/python/tft).

La función de servicio y la función de evaluación se cambian para Keras nativos. Los detalles se discutirán en las siguientes secciones de Trainer y Evaluator.

Nota: Las transformaciones dentro de `preprocessing_fn` no se pueden aplicar a la función de etiqueta para entrenamiento o evaluación.

#### Trainer

Para configurar Keras nativo, se debe configurar `GenericExecutor` para que el componente Trainer reemplace el ejecutor predeterminado basado en Estimator. Si desea obtener más información, consulte [aquí](trainer.md#configuring-the-trainer-component-to-use-the-genericexecutor).

##### Archivo del módulo Keras con Transform

El archivo del módulo de entrenamiento debe contener una `run_fn` que será llamada por `GenericExecutor`; una `run_fn` típica de Keras se vería así:

```python
def run_fn(fn_args: TrainerFnArgs):
  """Train the model based on given args.

  Args:
    fn_args: Holds args used to train the model as name/value pairs.
  """
  tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

  # Train and eval files contains transformed examples.
  # _input_fn read dataset based on transformed schema from tft.
  train_dataset = _input_fn(fn_args.train_files, fn_args.data_accessor,
                            tf_transform_output.transformed_metadata.schema)
  eval_dataset = _input_fn(fn_args.eval_files, fn_args.data_accessor,
                           tf_transform_output.transformed_metadata.schema)

  model = _build_keras_model()

  model.fit(
      train_dataset,
      steps_per_epoch=fn_args.train_steps,
      validation_data=eval_dataset,
      validation_steps=fn_args.eval_steps)

  signatures = {
      'serving_default':
          _get_serve_tf_examples_fn(model,
                                    tf_transform_output).get_concrete_function(
                                        tf.TensorSpec(
                                            shape=[None],
                                            dtype=tf.string,
                                            name='examples')),
  }
  model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)
```

En la `run_fn` que se mostró anteriormente, se necesita una firma de servicio al exportar el modelo entrenado para que el modelo pueda tomar ejemplos sin procesar para la predicción. Una función de servicio típica se vería así:

```python
def _get_serve_tf_examples_fn(model, tf_transform_output):
  """Returns a function that parses a serialized tf.Example."""

  # the layer is added as an attribute to the model in order to make sure that
  # the model assets are handled correctly when exporting.
  model.tft_layer = tf_transform_output.transform_features_layer()

  @tf.function
  def serve_tf_examples_fn(serialized_tf_examples):
    """Returns the output to be used in the serving signature."""
    feature_spec = tf_transform_output.raw_feature_spec()
    feature_spec.pop(_LABEL_KEY)
    parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)

    transformed_features = model.tft_layer(parsed_features)

    return model(transformed_features)

  return serve_tf_examples_fn
```

En la función de servicio anterior, las transformaciones tf.Transform se deben aplicar a los datos sin procesar para hacer inferencias, con la capa [`tft.TransformFeaturesLayer`](https://www.tensorflow.org/tfx/transform/api_docs/python/tft/TransformFeaturesLayer). La `_serving_input_receiver_fn` anterior que era necesaria para las instancias de Estimator ya no será necesaria con Keras.

##### Archivo del módulo Keras sin Transform

Esto es similar al archivo del módulo que se muestra arriba, pero sin las transformaciones:

```python
def _get_serve_tf_examples_fn(model, schema):

  @tf.function
  def serve_tf_examples_fn(serialized_tf_examples):
    feature_spec = _get_raw_feature_spec(schema)
    feature_spec.pop(_LABEL_KEY)
    parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
    return model(parsed_features)

  return serve_tf_examples_fn


def run_fn(fn_args: TrainerFnArgs):
  schema = io_utils.parse_pbtxt_file(fn_args.schema_file, schema_pb2.Schema())

  # Train and eval files contains raw examples.
  # _input_fn reads the dataset based on raw data schema.
  train_dataset = _input_fn(fn_args.train_files, fn_args.data_accessor, schema)
  eval_dataset = _input_fn(fn_args.eval_files, fn_args.data_accessor, schema)

  model = _build_keras_model()

  model.fit(
      train_dataset,
      steps_per_epoch=fn_args.train_steps,
      validation_data=eval_dataset,
      validation_steps=fn_args.eval_steps)

  signatures = {
      'serving_default':
          _get_serve_tf_examples_fn(model, schema).get_concrete_function(
              tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')),
  }
  model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)
```

##### [tf.distribute.Strategy](https://www.tensorflow.org/guide/distributed_training)

Por el momento, TFX solo admite estrategias de trabajador único (por ejemplo, [MirroredStrategy](https://www.tensorflow.org/api_docs/python/tf/distribute/MirroredStrategy), [OneDeviceStrategy](https://www.tensorflow.org/api_docs/python/tf/distribute/OneDeviceStrategy)).

Para utilizar una estrategia de distribución, cree una tf.distribute.Strategy adecuada y mueva la creación y la compilación del modelo de Keras dentro del alcance de una estrategia.

Por ejemplo, reemplace `model = _build_keras_model()` por:

```python
  mirrored_strategy = tf.distribute.MirroredStrategy()
  with mirrored_strategy.scope():
    model = _build_keras_model()

  # Rest of the code can be unchanged.
  model.fit(...)
```

Para verificar qué dispositivo (CPU/GPU) usa `MirroredStrategy`, habilite el registro de tensorflow a nivel de información:

```python
import logging
logging.getLogger("tensorflow").setLevel(logging.INFO)
```

y debería poder ver `Using MirroredStrategy with devices (...)` en el registro.

Nota: Quizás necesite la variable de entorno `TF_FORCE_GPU_ALLOW_GROWTH=true` para un problema de falta de memoria de la GPU. Para obtener más información, consulte la [guía de GPU de tensorflow](https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth).

#### Evaluator

En TFMA v0.2x, se combinaron ModelValidator y Evaluator en un único [componente Evaluator nuevo](https://github.com/tensorflow/community/blob/master/rfcs/20200117-tfx-combining-model-validator-with-evaluator.md). El nuevo componente Evaluator puede llevar a cabo tanto la evaluación de un solo modelo como la validación del modelo actual al compararlo con los modelos anteriores. Con este cambio, el componente Pusher ahora consume un resultado de aprobación de Evaluator en lugar de ModelValidator.

El nuevo Evaluator es compatible con modelos de Keras y Estimator. `_eval_input_receiver_fn` y el modelo guardado de evaluación que se requerían anteriormente ya no serán necesarios con Keras, ya que Evaluator ahora se basa en el mismo `SavedModel` que se usa para servir.

[Si desea obtener más información, consulte Evaluator](evaluator.md).
