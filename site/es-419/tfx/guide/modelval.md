# El componente de canalización ModelValidator TFX (obsoleto)

El componente ModelValidator se usó para comprobar si un modelo era lo suficientemente bueno para que se use en producción. Seguimos pensando que la validación es útil, pero como el [Evaluator](evaluator.md) del modelo ya calculó todas las métricas que desea validar, decidimos fusionar las dos funciones para que no tenga que duplicar los cálculos.

Si bien hemos dejado de usar ModelValidator y no recomendamos su uso, si necesita mantener un componente ModelValidator existente, puede tomar como ejemplo la siguiente configuración:

```python
import tfx
import tensorflow_model_analysis as tfma
from tfx.components.model_validator.component import ModelValidator

...

model_validator = ModelValidator(
      examples=example_gen.outputs['output_data'],
      model=trainer.outputs['model'])
```

Para aquellos que deseen migrar la configuración a Evaluator, una configuración similar para Evaluator sería la siguiente:

```python
from tfx import components
import tensorflow_model_analysis as tfma

...

eval_config = tfma.EvalConfig(
    model_specs=[
        # This assumes a serving model with signature 'serving_default'. If
        # using estimator based EvalSavedModel, add signature_name: 'eval' and
        # remove the label_key.
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

model_resolver = Resolver(
      strategy_class=latest_blessed_model_resolver.LatestBlessedModelResolver,
      model=Channel(type=Model),
      model_blessing=Channel(type=ModelBlessing)
).with_id('latest_blessed_model_resolver')

model_analyzer = components.Evaluator(
      examples=examples_gen.outputs['examples'],
      model=trainer.outputs['model'],
      baseline_model=model_resolver.outputs['model'],
      # Change threshold will be ignored if there is no baseline (first run).
      eval_config=eval_config)
```
