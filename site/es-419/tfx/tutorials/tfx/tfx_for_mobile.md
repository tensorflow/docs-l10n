# TFX para móviles

## Introducción

Esta guía demuestra cómo Tensorflow Extended (TFX) puede crear y evaluar modelos de aprendizaje automático que se implementarán en el dispositivo. TFX ahora proporciona soporte nativo para [TFLite](https://www.tensorflow.org/lite), lo que permite realizar inferencias altamente eficientes en dispositivos móviles.

En esta guía se le indicarán los cambios que se pueden realizar en cualquier canalización para generar y evaluar modelos de TFLite. [Aquí](https://github.com/tensorflow/tfx/blob/master/tfx/examples/mnist/mnist_pipeline_native_keras.py) le proporcionamos un ejemplo completo que demuestra cómo TFX puede entrenar y evaluar modelos de TFLite que se entrenan a partir del conjunto de datos [MNIST](http://yann.lecun.com/exdb/mnist/). Además, le mostramos cómo se puede usar la misma canalización para exportar simultáneamente tanto el [SavedModel](https://www.tensorflow.org/guide/saved_model) estándar basado en Keras como el TFLite, lo que permite a los usuarios comparar la calidad de los dos.

Asumimos que está familiarizado con TFX, sus componentes y canalizaciones. Si no es así, consulte este [tutorial](https://www.tensorflow.org/tfx/tutorials/tfx/components).

## Pasos

Solo se requieren dos pasos para crear y evaluar un modelo TFLite en TFX. El primer paso es invocar el reescritor TFLite dentro del contexto del [TFX Trainer](https://www.tensorflow.org/tfx/guide/trainer) para convertir el modelo TensorFlow entrenado en uno TFLite. El segundo paso es configurar Evaluator para evaluar los modelos TFLite. Ahora analizaremos cada uno de ellos.

### Cómo invocar el reescritor TFLite dentro de Trainer

TFX Trainer espera que se especifique una `run_fn` definida por el usuario en un archivo de módulo. Esta `run_fn` define el modelo que se va a entrenar, lo entrena durante el número especificado de iteraciones y exporta el modelo entrenado.

En lo que resta de esta sección, proporcionamos fragmentos de código donde se muestran los cambios necesarios para invocar la reescritura de TFLite y exportar un modelo de TFLite. Todo este código se encuentra en `run_fn` del [módulo MNIST TFLite](https://github.com/tensorflow/tfx/blob/master/tfx/examples/mnist/mnist_utils_native_keras_lite.py).

Como se muestra en el código siguiente, primero debemos crear una firma que tome un `Tensor` para cada característica como entrada. Tenga en cuenta que esto es una desviación de la mayoría de los modelos existentes en TFX, que toman protocolos [tf.Example](https://www.tensorflow.org/api_docs/python/tf/train/Example) serializados como entrada.

```python
 signatures = {
      'serving_default':
          _get_serve_tf_examples_fn(
              model, tf_transform_output).get_concrete_function(
                  tf.TensorSpec(
                      shape=[None, 784],
                      dtype=tf.float32,
                      name='image_floats'))
  }
```

A continuación, el modelo Keras se guarda como SavedModel del mismo modo que se hace normalmente.

```python
  temp_saving_model_dir = os.path.join(fn_args.serving_model_dir, 'temp')
  model.save(temp_saving_model_dir, save_format='tf', signatures=signatures)
```

Finalmente, creamos una instancia del reescritor TFLite (`tfrw`) y la invocamos en SavedModel para obtener el modelo TFLite. Almacenamos este modelo TFLite en el `serving_model_dir` proporcionado por la persona que llama a `run_fn`. De esta manera, el modelo TFLite se almacena en la ubicación donde todos los componentes posteriores de TFX esperarán encontrar el modelo.

```python
  tfrw = rewriter_factory.create_rewriter(
      rewriter_factory.TFLITE_REWRITER, name='tflite_rewriter')
  converters.rewrite_saved_model(temp_saving_model_dir,
                                 fn_args.serving_model_dir,
                                 tfrw,
                                 rewriter.ModelType.TFLITE_MODEL)
```

### Cómo evaluar el modelo TFLite

[TFX Evaluator](https://www.tensorflow.org/tfx/guide/evaluator) brinda la capacidad de analizar modelos entrenados para comprender su calidad en una amplia gama de métricas. Además de analizar SavedModels, TFX Evaluator ahora también puede analizar modelos TFLite.

El siguiente fragmento de código (reproducido de la [canalización MNIST](https://github.com/tensorflow/tfx/blob/master/tfx/examples/mnist/mnist_pipeline_native_keras.py)) muestra cómo configurar un Evaluator que analiza un modelo TFLite.

```python
  # Informs the evaluator that the model is a TFLite model.
  eval_config_lite.model_specs[0].model_type = 'tf_lite'

  ...

  # Uses TFMA to compute the evaluation statistics over features of a TFLite
  # model.
  model_analyzer_lite = Evaluator(
      examples=example_gen.outputs['examples'],
      model=trainer_lite.outputs['model'],
      eval_config=eval_config_lite,
  ).with_id('mnist_lite')
```

Como se muestra arriba, el único cambio que debemos hacer es establecer el campo `model_type` en `tf_lite`. No se requieren otros cambios de configuración para analizar el modelo TFLite. Independientemente de si se analiza un modelo TFLite o un SavedModel, la salida del `Evaluator` tendrá exactamente la misma estructura.

Sin embargo, tenga en cuenta que Evaluator asume que el modelo TFLite está guardado en un archivo llamado `tflite` dentro de trainer_lite.outputs['model'].
