# Использование TensorFlow 2.x в рамках TFX

[Библиотека TensorFlow 2.0 появилась в 2019 году](https://blog.tensorflow.org/2019/09/tensorflow-20-is-now-available.html) и, помимо прочих нововведений, представила такие [улучшения](https://www.tensorflow.org/guide/effective_tf2#a_brief_summary_of_major_changes), как [тесная интеграция с Keras](https://www.tensorflow.org/guide/keras/overview), использование [режима eager execution](https://www.tensorflow.org/guide/eager) по умолчанию и [выполнение функций с использованием естественного синтаксиса Python](https://www.tensorflow.org/guide/function).

В этом руководстве подробно рассмотрены технические особенности работы с TF 2.x в среде TFX.

## Совместимость версий

Платформа TFX совместима с TensorFlow 2.x и высокоуровневыми API из TensorFlow 1.x (в частности, с Estimator).

### Создание проектов в TensorFlow 2.x

Так как в TensorFlow 2.x остались высокоуровневые возможности из TensorFlow 1.x, использовать старую версию в новых проектах нецелесообразно, даже если новые функции не нужны.

Соответственно, при создании проекта на TFX лучше сразу перейти на TensorFlow 2.x. В будущем, когда появится полная поддержка Keras и другие важные функции, вам придется внести гораздо меньше изменений в код, если он изначально создавался под TensorFlow 2.x, а не TensorFlow 1.x.

### Перенос уже имеющихся проектов на TensorFlow 2.x

Код, написанный для TensorFlow 1.x, в основном совместим с TensorFlow 2.x и продолжит работать на TFX.

Однако если вы хотите использовать возможности и улучшения новой версии, выполните [миграцию на TF 2.x](https://www.tensorflow.org/guide/migrate).

## Estimator

API Estimator остался в TensorFlow 2.x, однако нововведения его не коснулись. Код, написанный для TensorFlow 1.x или 2.x с использованием Estimator, будет работать на TFX как и прежде.

Вот законченный пример использования Estimator в чистом виде на TFX: [Taxi (Estimator)](https://github.com/tensorflow/tfx/blob/r0.21/tfx/examples/chicago_taxi_pipeline/taxi_utils.py).

## Keras с функцией `model_to_estimator`

Функция `tf.keras.estimator.model_to_estimator` позволяет преобразовать модель Keras в объект Estimator и работать с ней так же, как с любым другим объектом данного типа. Для этого:

1. Постройте модель Keras.
2. Передайте скомпилированную модель в функцию `model_to_estimator`.
3. Используйте результат функции `model_to_estimator` в компоненте Trainer в качестве объекта Estimator.

```py
# Построение модели Keras.
def _keras_model_builder():
  """Создание модели Keras."""
  ...

  model = tf.keras.Model(inputs=inputs, outputs=output)
  model.compile()

  return model


# Создание типовой функции trainer
def trainer_fn(trainer_fn_args, schema):
  """Построение estimator с помощью функции model_to_estimator."""
  ...

  # Преобразование модели в estimator
  estimator = tf.keras.estimator.model_to_estimator(
      keras_model=_keras_model_builder(), config=run_config)

  return {
      'estimator': estimator,
      ...
  }
```

Изменяется только пользовательский модуль компонента Trainer; остальная часть конвейера остается прежней. Вот законченный пример использования Keras с функцией model_to_estimator на TFX: [Iris (model_to_estimator)](https://github.com/tensorflow/tfx/blob/r0.21/tfx/examples/iris/iris_utils.py).

## Нативный Keras (Keras без функции `model_to_estimator`)

Примечание. Хотя полная поддержка всех возможностей Keras еще не реализована, в большинстве случаев Keras на TFX работает корректно. Разреженные признаки при работе с FeatureColumns пока что не поддерживаются.

### Примеры и учебник в Colab

Вот несколько примеров с использованием нативного Keras:

- [Iris](https://github.com/tensorflow/tfx/blob/master/tfx/examples/iris/iris_pipeline_native_keras.py) ([файл модуля](https://github.com/tensorflow/tfx/blob/master/tfx/examples/iris/iris_utils_native_keras.py)): простейшая законченная программа.
- [MNIST](https://github.com/tensorflow/tfx/blob/master/tfx/examples/mnist/mnist_pipeline_native_keras.py) ([файл модуля](https://github.com/tensorflow/tfx/blob/master/tfx/examples/mnist/mnist_utils_native_keras.py)): законченный пример классификации изображений с использованием TFLite.
- [Taxi](https://github.com/tensorflow/tfx/blob/master/tfx/examples/chicago_taxi_pipeline/taxi_pipeline_native_keras.py) ([файл модуля](https://github.com/tensorflow/tfx/blob/master/tfx/examples/chicago_taxi_pipeline/taxi_utils_native_keras.py)): законченный пример использования расшире

Вы также можете ознакомиться с [учебником по Keras в Colab](https://www.tensorflow.org/tfx/tutorials/tfx/components_keras), где подробно описан каждый компонент TFX.

### Компоненты TFX

Далее в статье мы расскажем о поддержке нативного Keras релевантными компонентами TFX.

#### Transform

В библиотеке Transform в настоящее время реализована экспериментальная поддержка моделей Keras.

Сам компонент Transform при работе с нативным Keras можно использовать так же, как обычно. Определение функции предварительной обработки `preprocessing_fn` остается прежним: используются операторы [TensorFlow](https://www.tensorflow.org/api_docs/python/tf) и [tf.Transform](https://www.tensorflow.org/tfx/transform/api_docs/python/tft).

Функции обработки запроса и оценки при использовании нативного Keras работают иначе. Подробности описаны в разделах Trainer и Evaluator ниже.

Примечание. Преобразования внутри функции `preprocessing_fn` нельзя применять к признаку label («метка») при обучении или оценке.

#### Trainer

Чтобы настроить Trainer на работу с нативным Keras, необходимо заменить исполнитель на основе Estimator, который используется по умолчанию, на `GenericExecutor`. Подробнее об этом см. [здесь](trainer.md#configuring-the-trainer-component-to-use-the-genericexecutor).

##### Файл модуля Keras с использованием Transform

Файл модуля обучения должен содержать функцию `run_fn`, которую будет вызывать исполняющий модуль `GenericExecutor`. Типовая функция `run_fn` для Keras выглядит следующим образом:

```python
def run_fn(fn_args: TrainerFnArgs):
  """Обучение модели на основе полученных аргументов.

  Аргументы:
    fn_args: содержит аргументы для обучения модели в виде пар «имя/значение».
  """
  tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

  # Файлы для обучения и оценки содержат преобразованные образцы.
  # Функция _input_fn считывает набор данных на основе спецификации признаков, преобразованной с помощью модуля tft.
  train_dataset = _input_fn(fn_args.train_files, tf_transform_output, 40)
  eval_dataset = _input_fn(fn_args.eval_files, tf_transform_output, 40)

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

В приведенной выше функции `run_fn` при экспорте обученной модели необходимо указать сигнатуру для обработки запросов, чтобы позволить модели принимать необработанные образцы для прогнозирования. Типичная функция обработки запросов выглядит следующим образом:

```python
def _get_serve_tf_examples_fn(model, tf_transform_output):
  """Возвращает функцию разбора сериализованного tf.Example."""

  # слой добавляется в модель как атрибут, чтобы
  # ресурсы модели корректно обрабатывались при экспорте.
  model.tft_layer = tf_transform_output.transform_features_layer()

  @tf.function
  def serve_tf_examples_fn(serialized_tf_examples):
    """Возвращает результат, используемый в сигнатуре для обработки запроса."""
    feature_spec = tf_transform_output.raw_feature_spec()
    feature_spec.pop(_LABEL_KEY)
    parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)

    transformed_features = model.tft_layer(parsed_features)

    return model(transformed_features)

  return serve_tf_examples_fn
```

В приведенной выше функции обработки запроса необходимо применить преобразования tf.Transform к необработанным данным для получения результата, используя слой [`tft.TransformFeaturesLayer`](https://github.com/tensorflow/transform/blob/master/docs/api_docs/python/tft/TransformFeaturesLayer.md). Функция `_serving_input_receiver_fn`, которая требовалась для работы с Estimator, при использовании Keras больше не нужна.

##### Файл модуля Keras без использования Transform

Здесь все аналогично приведенному выше файлу модуля, но без преобразований:

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

  # Файлы для обучения и оценки содержат необработанные образцы.
  # Функция _input_fn считывает набор данных на основе необработанной спецификации признаков из переменной schema.
  train_dataset = _input_fn(fn_args.train_files, schema, 40)
  eval_dataset = _input_fn(fn_args.eval_files, schema, 40)

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

#####

[tf.distribute.Strategy](https://www.tensorflow.org/guide/distributed_training)

На данный момент TFX поддерживает стратегии распределения только на одной машине (например, [MirroredStrategy](https://www.tensorflow.org/api_docs/python/tf/distribute/MirroredStrategy), [OneDeviceStrategy](https://www.tensorflow.org/api_docs/python/tf/distribute/OneDeviceStrategy)).

Чтобы использовать стратегию распределения, создайте подходящий элемент tf.distribute.Strategy и перенесите создание и компиляцию модели Keras в область действия стратегии.

Например, в приведенном выше коде замените `model = _build_keras_model()` на:

```python
  mirrored_strategy = tf.distribute.MirroredStrategy()
  with mirrored_strategy.scope():
    model = _build_keras_model()

  # Остальной код изменять не нужно.
  model.fit(...)
```

Чтобы узнать, какое устройство (CPU/GPU) используется стратегией `MirroredStrategy`, включите регистрацию событий tensorflow на уровне info:

```python
import logging
logging.getLogger("tensorflow").setLevel(logging.INFO)
```

В журнале должна появиться запись `Using MirroredStrategy with devices (...)`.

Примечание. В случае нехватки видеопамяти может потребоваться переменная среды `TF_FORCE_GPU_ALLOW_GROWTH=true`. Подробнее об этом см. в [руководстве по tensorflow на GPU](https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth).

#### Evaluator

В TFMA 0.2x компоненты ModelValidator и Evaluator объединены в [новый компонент Evaluator](https://github.com/tensorflow/community/blob/master/rfcs/20200117-tfx-combining-model-validator-with-evaluator.md), который может выполнять как оценку отдельной модели, так и валидацию текущей модели путем ее сравнения с предыдущими. В связи с этим изменением компонент Pusher теперь получает результат одобрения модели от Evaluator, а не от ModelValidator, как раньше.

Новый компонент Evaluator поддерживает как модели Keras, так и модели Estimator. При работе с Keras больше не требуется функция `_eval_input_receiver_fn` и модель типа EvalSavedModel, как раньше, поскольку Evaluator теперь работает с той же моделью `SavedModel`, которая используется для обработки запросов.

[Подробнее об этом см. в разделе Evaluator](evaluator.md).
