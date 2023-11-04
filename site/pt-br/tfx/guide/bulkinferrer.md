# O componente de pipeline BulkInferrer TFX

O componente BulkInferrer TFX realiza inferência em lote em dados não rotulados. O InferenceResult gerado ([tensorflow_serving.apis.prediction_log_pb2.PredictionLog](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/apis/prediction_log.proto)) contém as características originais e os resultados da previsão.

O BulkInferrer consome:

- Um modelo treinado no formato [SavedModel](https://www.tensorflow.org/guide/saved_model.md).
- tf.Examples não rotulados que contêm características.
- (Opcional) Resultado da validação do componente [Evaluator](https://www.tensorflow.org/tfx/guide/evaluator.md).

O BulkInferrer produz:

- [InferenceResult](https://github.com/tensorflow/tfx/blob/master/tfx/types/standard_artifacts.py)

## Usando o componente BulkInferrer

Um componente BulkInferrer TFX é usado para realizar inferência em lote em tf.Examples não rotulados. Normalmente é implantado depois de um componente [Evaluator](https://www.tensorflow.org/tfx/guide/evaluator.md) para realizar inferência com um modelo validado ou após um componente [Trainer](https://www.tensorflow.org/tfx/guide/trainer.md) para realizar inferência diretamente no modelo exportado.

Ele atualmente realiza inferência de modelo na memória e inferência remota. A inferência remota requer que o modelo seja hospedado no Cloud AI Platform.

Este é um exemplo de código típico:

```python
bulk_inferrer = BulkInferrer(
    examples=examples_gen.outputs['examples'],
    model=trainer.outputs['model'],
    model_blessing=evaluator.outputs['blessing'],
    data_spec=bulk_inferrer_pb2.DataSpec(),
    model_spec=bulk_inferrer_pb2.ModelSpec()
)
```

Mais detalhes estão disponíveis na [referência da API BulkInferrer](https://www.tensorflow.org/tfx/api_docs/python/tfx/v1/components/BulkInferrer).
