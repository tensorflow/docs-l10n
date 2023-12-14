# El componente de canalización Trainer TFX

El componente de canalización Trainer TFX entrena un modelo de TensorFlow.

## Trainer y TensorFlow

Trainer hace un uso extensivo de la API de [TensorFlow](https://www.tensorflow.org) para Python con el fin de entrenar modelos.

Nota: TFX es compatible con TensorFlow 1.15 y 2.x.

## Componente

Trainer toma:

- tf.Examples que usa para entrenamiento y evaluación.
- Un archivo de módulo proporcionado por el usuario que define la lógica del entrenador.
- Definición de [Protobuf](https://developers.google.com/protocol-buffers) de argumentos de entrenamiento y argumentos de evaluación.
- (Opcional) Un esquema de datos creado por un componente de canalización SchemaGen y, opcionalmente, modificado por el desarrollador.
- (Opcional) gráfico de transformación producido por un componente Transform ascendente.
- (Opcional) modelos previamente entrenados que se usan para escenarios como el arranque en tibio.
- (Opcionales) hiperparámetros, que se pasarán a la característica del módulo de usuario. Los detalles de la integración con Tuner se pueden encontrar [aquí](tuner.md).

Trainer emite: al menos un modelo para inferencia/servicio (normalmente en SavedModelFormat) y, opcionalmente, otro modelo para evaluación (generalmente un EvalSavedModel).

Brindamos soporte para formatos de modelos alternativos como [TFLite](https://www.tensorflow.org/lite) a través de la [Model Rewriting Library](https://github.com/tensorflow/tfx/blob/master/tfx/components/trainer/rewriting/README.md). Consulte el enlace a Model Rewriting Library para ver ejemplos de cómo convertir los modelos Estimator y Keras.

## Trainer genérico

El Trainer genérico permite a los desarrolladores usar cualquier API modelo de TensorFlow con el componente Trainer. Además de Estimator de TensorFlow, los desarrolladores pueden utilizar modelos Keras o bucles de entrenamiento personalizados. Para obtener más información, consulte el [RFC para Trainer genérico](https://github.com/tensorflow/community/blob/master/rfcs/20200117-tfx-generic-trainer.md).

### Cómo configurar un componente Trainer

El código DSL de canalización típico para Trainer genérico se vería así:

```python
from tfx.components import Trainer

...

trainer = Trainer(
    module_file=module_file,
    examples=transform.outputs['transformed_examples'],
    transform_graph=transform.outputs['transform_graph'],
    train_args=trainer_pb2.TrainArgs(num_steps=10000),
    eval_args=trainer_pb2.EvalArgs(num_steps=5000))
```

Trainer invoca un módulo de entrenamiento, que se especifica en el parámetro `module_file`. En lugar de `trainer_fn`, se requiere una `run_fn` en el archivo del módulo si `GenericExecutor` se especifica en `custom_executor_spec`. `trainer_fn` fue responsable de crear el modelo. Además de eso, `run_fn` también necesita manejar la parte de entrenamiento y enviar el modelo entrenado a la ubicación deseada que proporciona [FnArgs](https://github.com/tensorflow/tfx/blob/master/tfx/components/trainer/fn_args_utils.py):

```python
from tfx.components.trainer.fn_args_utils import FnArgs

def run_fn(fn_args: FnArgs) -> None:
  """Build the TF model and train it."""
  model = _build_keras_model()
  model.fit(...)
  # Save model to fn_args.serving_model_dir.
  model.save(fn_args.serving_model_dir, ...)
```

Este es un [archivo de módulo de ejemplo](https://github.com/tensorflow/tfx/blob/master/tfx/examples/penguin/penguin_utils_keras.py) con `run_fn`.

Tenga en cuenta que, si el componente Transform no se usa en el proceso, Trainer tomará los ejemplos de ExampleGen directamente:

```python
trainer = Trainer(
    module_file=module_file,
    examples=example_gen.outputs['examples'],
    schema=infer_schema.outputs['schema'],
    train_args=trainer_pb2.TrainArgs(num_steps=10000),
    eval_args=trainer_pb2.EvalArgs(num_steps=5000))
```

Hay más detalles disponibles en la [referencia de la API de Trainer](https://www.tensorflow.org/tfx/api_docs/python/tfx/v1/components/Trainer).
