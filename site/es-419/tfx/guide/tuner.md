# El componente de canalización Tuner TFX

El componente Tuner sirve para ajustar los hiperparámetros del modelo.

## Componente Tuner y biblioteca KerasTuner

El componente Tuner hace un uso extensivo de la API de [KerasTuner](https://www.tensorflow.org/tutorials/keras/keras_tuner) para Python con el fin de ajustar los hiperparámetros.

Nota: La biblioteca KerasTuner se puede usar para el ajuste de hiperparámetros independientemente de la API de modelado, no solo para los modelos Keras.

## Componente

Tuner toma:

- tf.Examples que usa para entrenamiento y evaluación.
- Un archivo de módulo proporcionado por el usuario (o módulo fn) que define la lógica de ajuste, incluida la definición del modelo, el espacio de búsqueda de hiperparámetros, el objetivo, etc.
- Definición de [Protobuf](https://developers.google.com/protocol-buffers) de argumentos de entrenamiento y argumentos de evaluación.
- (Opcional) Definición de [Protobuf](https://developers.google.com/protocol-buffers) de argumentos de ajuste.
- (Opcional) Gráfico de transformación producido por un componente Transform ascendente.
- (Opcional) Un esquema de datos creado por un componente de canalización de SchemaGen y, opcionalmente, modificado por el desarrollador.

Con los datos, el modelo y el objetivo proporcionados, Tuner ajusta los hiperparámetros y emite el mejor resultado.

## Instrucciones

Se requiere una función de módulo de usuario `tuner_fn` con la siguiente firma para Tuner:

```python
...
from keras_tuner.engine import base_tuner

TunerFnResult = NamedTuple('TunerFnResult', [('tuner', base_tuner.BaseTuner),
                                             ('fit_kwargs', Dict[Text, Any])])

def tuner_fn(fn_args: FnArgs) -> TunerFnResult:
  """Build the tuner using the KerasTuner API.
  Args:
    fn_args: Holds args as name/value pairs.
      - working_dir: working dir for tuning.
      - train_files: List of file paths containing training tf.Example data.
      - eval_files: List of file paths containing eval tf.Example data.
      - train_steps: number of train steps.
      - eval_steps: number of eval steps.
      - schema_path: optional schema of the input data.
      - transform_graph_path: optional transform graph produced by TFT.
  Returns:
    A namedtuple contains the following:
      - tuner: A BaseTuner that will be used for tuning.
      - fit_kwargs: Args to pass to tuner's run_trial function for fitting the
                    model , e.g., the training and validation dataset. Required
                    args depend on the above tuner's implementation.
  """
  ...
```

En esta función, se definen los espacios de búsqueda del modelo y del hiperparámetro, y se elige el objetivo y el algoritmo para el ajuste. El componente Tuner toma el código de este módulo como entrada, ajusta los hiperparámetros y emite el mejor resultado.

Trainer puede tomar los hiperparámetros de salida de Tuner como entrada y usarlos en el código de su módulo de usuario. La definición de canalización se ve así:

```python
...
tuner = Tuner(
    module_file=module_file,  # Contains `tuner_fn`.
    examples=transform.outputs['transformed_examples'],
    transform_graph=transform.outputs['transform_graph'],
    train_args=trainer_pb2.TrainArgs(num_steps=20),
    eval_args=trainer_pb2.EvalArgs(num_steps=5))

trainer = Trainer(
    module_file=module_file,  # Contains `run_fn`.
    examples=transform.outputs['transformed_examples'],
    transform_graph=transform.outputs['transform_graph'],
    schema=schema_gen.outputs['schema'],
    # This will be passed to `run_fn`.
    hyperparameters=tuner.outputs['best_hyperparameters'],
    train_args=trainer_pb2.TrainArgs(num_steps=100),
    eval_args=trainer_pb2.EvalArgs(num_steps=5))
...
```

Quizás no quiera ajustar los hiperparámetros cada vez que vuelve a entrenar su modelo. Una vez que haya usado Tuner para determinar un buen conjunto de hiperparámetros, puede eliminar Tuner de su canalización y usar `ImporterNode` para importar el artefacto Tuner de una ejecución de entrenamiento anterior para cargarlo a Trainer.

```python
hparams_importer = Importer(
    # This can be Tuner's output file or manually edited file. The file contains
    # text format of hyperparameters (keras_tuner.HyperParameters.get_config())
    source_uri='path/to/best_hyperparameters.txt',
    artifact_type=HyperParameters,
).with_id('import_hparams')

trainer = Trainer(
    ...
    # An alternative is directly use the tuned hyperparameters in Trainer's user
    # module code and set hyperparameters to None here.
    hyperparameters = hparams_importer.outputs['result'])
```

## Cómo hacer ajustes en Google Cloud Platform (GCP)

Cuando se ejecuta en Google Cloud Platform (GCP), el componente Tuner puede aprovechar dos servicios:

- [AI Platform Vizier](https://cloud.google.com/ai-platform/optimizer/docs/overview) (a través de la implementación de CloudTuner)
- [AI Platform Training](https://cloud.google.com/ai-platform/training/docs) (como gestor de grupos para el ajuste distribuido)

### AI Platform Vizier como backend del ajuste de hiperparámetros

[AI Platform Vizier](https://cloud.google.com/ai-platform/optimizer/docs/overview) es un servicio administrado que ejecuta optimización de caja negra, basado en la tecnología [Google Vizier](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/bcb15507f4b52991a0783013df4222240e942381.pdf).

[CloudTuner](https://github.com/tensorflow/cloud/blob/master/src/python/tensorflow_cloud/tuner/tuner.py) es una implementación de [KerasTuner](https://www.tensorflow.org/tutorials/keras/keras_tuner) que se comunica con el servicio AI Platform Vizier como backend del estudio. Dado que CloudTuner es una subclase de `keras_tuner.Tuner`, se puede usar como reemplazo directo en el módulo `tuner_fn` y ejecutar como parte del componente TFX Tuner.

A continuación, se muestra un fragmento de código que muestra cómo usar `CloudTuner`. Tenga en cuenta que la configuración de `CloudTuner` requiere elementos que son específicos de GCP, como `project_id` y `region`.

```python
...
from tensorflow_cloud import CloudTuner

...
def tuner_fn(fn_args: FnArgs) -> TunerFnResult:
  """An implementation of tuner_fn that instantiates CloudTuner."""

  ...
  tuner = CloudTuner(
      _build_model,
      hyperparameters=...,
      ...
      project_id=...,       # GCP Project ID
      region=...,           # GCP Region where Vizier service is run.
  )

  ...
  return TuneFnResult(
      tuner=tuner,
      fit_kwargs={...}
  )

```

### Ajuste paralelo en Cloud AI Platform para entrenar grupo de trabajadores distribuidos

El marco KerasTuner como implementación subyacente del componente Tuner tiene la capacidad de buscar hiperparámetros en paralelo. Si bien el componente Tuner estándar no tiene la capacidad de ejecutar más de un trabajador de búsqueda en paralelo, al utilizar el [componente Tuner de la extensión de Google Cloud AI Platform](https://github.com/tensorflow/tfx/blob/master/tfx/extensions/google_cloud_ai_platform/tuner/component.py), ofrece la capacidad de ejecutar ajustes en paralelo, mediante un trabajo de entrenamiento de AI Platform como gestor de un grupo de trabajadores distribuidos. [TuneArgs](https://github.com/tensorflow/tfx/blob/master/tfx/proto/tuner.proto) es la configuración específica de este componente. Este es un reemplazo directo del componente Tuner estándar.

```python
tuner = google_cloud_ai_platform.Tuner(
    ...   # Same kwargs as the above stock Tuner component.
    tune_args=proto.TuneArgs(num_parallel_trials=3),  # 3-worker parallel
    custom_config={
        # Configures Cloud AI Platform-specific configs . For for details, see
        # https://cloud.google.com/ai-platform/training/docs/reference/rest/v1/projects.jobs#traininginput.
        TUNING_ARGS_KEY:
            {
                'project': ...,
                'region': ...,
                # Configuration of machines for each master/worker in the flock.
                'masterConfig': ...,
                'workerConfig': ...,
                ...
            }
    })
...

```

El comportamiento y la salida del componente Tuner de extensión es el mismo que el componente Tuner estándar, excepto que se ejecutan múltiples búsquedas de hiperparámetros en paralelo en diferentes máquinas de trabajo y, como resultado, `num_trials` se completa más rápido. Esto es particularmente efectivo cuando el algoritmo de búsqueda es increíblemente paralelizable, como `RandomSearch`. Sin embargo, si el algoritmo de búsqueda usa información de resultados de pruebas anteriores, como lo hace el algoritmo Google Vizier que se implementa en AI Platform Vizier, una búsqueda excesivamente paralela afectaría negativamente la eficacia de la acción.

Nota: Cada prueba en cada búsqueda paralela se ejecuta en una sola máquina en el grupo de trabajadores, es decir, cada prueba no aprovecha el entrenamiento distribuido entre varios trabajadores. Si desea aplicar la distribución entre varios trabajadores para cada prueba, consulte [`DistributingCloudTuner`](https://github.com/tensorflow/cloud/blob/b9c8752f5c53f8722dfc0b5c7e05be52e62597a8/src/python/tensorflow_cloud/tuner/tuner.py#L384-L676), en lugar de `CloudTuner`.

Nota: Tanto `CloudTuner` como el componente Tuner de extensiones de Google Cloud AI Platform se pueden usar juntos, en cuyo caso permite el ajuste paralelo distribuido respaldado por el algoritmo de búsqueda de hiperparámetros de AI Platform Vizier. Sin embargo, para poder hacerlo, el trabajo de Cloud AI Platform debe tener acceso al servicio AI Platform Vizier. Consulte esta [guía](https://cloud.google.com/ai-platform/training/docs/custom-service-account#custom) para configurar una cuenta de servicio personalizada. Después de eso, debe especificar la cuenta de servicio personalizada para su trabajo de entrenamiento en el código de canalización. Para obtener más información al respecto, consulte el [Ejemplo de E2E CloudTuner en GCP](https://github.com/tensorflow/tfx/blob/master/tfx/examples/penguin/penguin_pipeline_kubeflow.py).

## Enlaces

[Ejemplo de E2E](https://github.com/tensorflow/tfx/blob/master/tfx/examples/penguin/penguin_pipeline_local.py)

[Ejemplo de E2E CloudTuner en GCP](https://github.com/tensorflow/tfx/blob/master/tfx/examples/penguin/penguin_pipeline_kubeflow.py)

[Tutorial de KerasTuner](https://www.tensorflow.org/tutorials/keras/keras_tuner)

[Tutorial de CloudTuner](https://github.com/GoogleCloudPlatform/ai-platform-samples/blob/master/notebooks/samples/optimizer/ai_platform_vizier_tuner.ipynb)

[Propuesta](https://github.com/tensorflow/community/blob/master/rfcs/20200420-tfx-tuner-component.md)

Hay más detalles disponibles en la [referencia de la API de Tuner](https://www.tensorflow.org/tfx/api_docs/python/tfx/v1/components/Tuner).
