# El componente de canalización Evaluator TFX

El componente de canalización Evaluator TFX ejecuta un análisis profundo de los resultados del entrenamiento de sus modelos, para ayudarlo a comprender cómo se desempeña su modelo en subconjuntos de sus datos. Evaluator también le ayuda a validar sus modelos exportados, para garantizar que sean "lo suficientemente buenos" para pasarse a producción.

Cuando la validación está habilitada, Evaluator compara nuevos modelos con una línea base (como el modelo que se ofrece actualmente) para determinar si son "lo suficientemente buenos" en relación con la línea base. Para esto, evalúa ambos modelos en un conjunto de datos de evaluación y calcula su rendimiento en métricas (por ejemplo, AUC, pérdida). Si las métricas del nuevo modelo cumplen con los criterios especificados por el desarrollador para el modelo de línea base (por ejemplo, el AUC no es inferior), el modelo está "aprobado" (se marca como bueno), lo que indica al [Pusher](pusher.md) que puede pasar el modelo a producción.

- Consume:
    - Una división de evaluación de [ExampleGen](https://www.tensorflow.org/tfx/guide/examplegen)
    - Un modelo entrenado de [Trainer](trainer.md)
    - Un modelo previamente aprobado (si se va a ejecutar una validación)
- Emite:
    - Resultados del análisis de [ML Metadata](mlmd.md)
    - Resultados de la validación de [ML Metadata](mlmd.md) (si se va a ejecutar una validación)

## Evaluator y TensorFlow Model Analysis

Evaluator aprovecha la biblioteca [TensorFlow Model Analysis](tfma.md) para ejecutar el análisis, que a su vez usa [Apache Beam](beam.md) para acceder a un procesamiento escalable.

## Uso del componente Evaluator

Un componente de canalización Evaluator suele ser muy fácil de implementar y requiere muy poca personalización, ya que el componente Evaluator TFX hace la mayor parte del trabajo.

Para configurar el evaluador, se necesita la siguiente información:

- Métricas para configurar (solo son necesarias si se agregan métricas adicionales además de las guardadas con el modelo). Consulte [Métricas de Tensorflow Model Analysis](https://github.com/tensorflow/model-analysis/blob/master/g3doc/metrics.md) para obtener más información.
- Segmentos para configurar (si no se ofrecen segmentos, se agregará un segmento "general" de forma predeterminada). Consulte [Configuración de Tensorflow Model Analysis](https://github.com/tensorflow/model-analysis/blob/master/g3doc/setup.md) para obtener más información.

Si desea incluir la validación, necesita la siguiente información adicional:

- Con qué modelo comparar (último aprobado, etc.).
- Validaciones del modelo (umbrales) a verificar. Consulte [Validaciones de Tensorflow Model Analysis](https://github.com/tensorflow/model-analysis/blob/master/g3doc/model_validations.md) para obtener más información.

Cuando esté habilitada, la validación se ejecutará con todas las métricas y los segmentos que se definieron.

El código típico se ve así:

```python
import tensorflow_model_analysis as tfma
...

# For TFMA evaluation

eval_config = tfma.EvalConfig(
    model_specs=[
        # This assumes a serving model with signature 'serving_default'. If
        # using estimator based EvalSavedModel, add signature_name='eval' and
        # remove the label_key. Note, if using a TFLite model, then you must set
        # model_type='tf_lite'.
        tfma.ModelSpec(label_key='<label_key>')
    ],
    metrics_specs=[
        tfma.MetricsSpec(
            # The metrics added here are in addition to those saved with the
            # model (assuming either a keras model or EvalSavedModel is used).
            # Any metrics added into the saved model (for example using
            # model.compile(..., metrics=[...]), etc) will be computed
            # automatically.
            metrics=[
                tfma.MetricConfig(class_name='ExampleCount'),
                tfma.MetricConfig(
                    class_name='BinaryAccuracy',
                    threshold=tfma.MetricThreshold(
                        value_threshold=tfma.GenericValueThreshold(
                            lower_bound={'value': 0.5}),
                        change_threshold=tfma.GenericChangeThreshold(
                            direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                            absolute={'value': -1e-10})))
            ]
        )
    ],
    slicing_specs=[
        # An empty slice spec means the overall slice, i.e. the whole dataset.
        tfma.SlicingSpec(),
        # Data can be sliced along a feature column. In this case, data is
        # sliced along feature column trip_start_hour.
        tfma.SlicingSpec(feature_keys=['trip_start_hour'])
    ])

# The following component is experimental and may change in the future. This is
# required to specify the latest blessed model will be used as the baseline.
model_resolver = Resolver(
      strategy_class=dsl.experimental.LatestBlessedModelStrategy,
      model=Channel(type=Model),
      model_blessing=Channel(type=ModelBlessing)
).with_id('latest_blessed_model_resolver')

model_analyzer = Evaluator(
      examples=examples_gen.outputs['examples'],
      model=trainer.outputs['model'],
      baseline_model=model_resolver.outputs['model'],
      # Change threshold will be ignored if there is no baseline (first run).
      eval_config=eval_config)
```

El evaluador produce un [EvalResult](https://www.tensorflow.org/tfx/model_analysis/api_docs/python/tfma/EvalResult) (y opcionalmente un [ValidationResult](https://www.tensorflow.org/tfx/model_analysis/api_docs/python/tfma/ValidationResult) si se usó validación) que se puede cargar con [TFMA](tfma.md). El siguiente es un ejemplo de cómo cargar los resultados en un bloc de notas Jupyter:

```
import tensorflow_model_analysis as tfma

output_path = evaluator.outputs['evaluation'].get()[0].uri

# Load the evaluation results.
eval_result = tfma.load_eval_result(output_path)

# Visualize the metrics and plots using tfma.view.render_slicing_metrics,
# tfma.view.render_plot, etc.
tfma.view.render_slicing_metrics(tfma_result)
...

# Load the validation results
validation_result = tfma.load_validation_result(output_path)
if not validation_result.validation_ok:
  ...
```

Hay más detalles disponibles en la [referencia de la API de Evaluator](https://www.tensorflow.org/tfx/api_docs/python/tfx/v1/components/Evaluator).
