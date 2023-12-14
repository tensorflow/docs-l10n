# El componente de canalización BulkInferrer TFX

El componente BulkInferrer TFX ejecuta la inferencia por lotes sobre datos sin etiquetar. El InferenceResult ([tensorflow_serving.apis.prediction_log_pb2.PredictionLog](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/apis/prediction_log.proto)) que se genera contiene las características originales y los resultados de la predicción.

BulkInferrer consume esto:

- Un modelo entrenado en formato [SavedModel](https://www.tensorflow.org/guide/saved_model.md).
- tf.Examples sin etiqueta que contienen características.
- (Opcional) Resultado de la validación del componente [Evaluator](https://www.tensorflow.org/tfx/guide/evaluator.md).

BulkInferrer emite esto:

- [InferenceResult](https://github.com/tensorflow/tfx/blob/master/tfx/types/standard_artifacts.py)

## Cómo usar el componente BulkInferrer

Se usa un componente BulkInferrer TFX para ejecutar inferencia por lotes en tf.Examples sin etiquetar. Por lo general, se implementa después de un componente [Evaluator](https://www.tensorflow.org/tfx/guide/evaluator.md) para inferir con modelo validado, o después de un componente [Trainer](https://www.tensorflow.org/tfx/guide/trainer.md) para ejecutar la inferencia directamente en el modelo exportado.

Actualmente ejecuta inferencia de modelos en memoria e inferencia remota. Para ejecutar la inferencia remota el modelo debe estar alojado en Cloud AI Platform.

El código típico se ve así:

```python
bulk_inferrer = BulkInferrer(
    examples=examples_gen.outputs['examples'],
    model=trainer.outputs['model'],
    model_blessing=evaluator.outputs['blessing'],
    data_spec=bulk_inferrer_pb2.DataSpec(),
    model_spec=bulk_inferrer_pb2.ModelSpec()
)
```

Hay más detalles disponibles en la [referencia de la API de BulkInferrer](https://www.tensorflow.org/tfx/api_docs/python/tfx/v1/components/BulkInferrer).
