# O componente de pipeline Evaluator TFX

O componente de pipeline Evaluator TFX realiza análises profundas nos resultados de treinamento dos seus modelos, para ajudar você a entender o desempenho do seu modelo em subconjuntos de seus dados. O Evaluator também ajuda a validar seus modelos exportados, garantindo que eles sejam “bons o suficiente” para serem enviados para produção.

Quando a validação está habilitada, o Evaluator compara novos modelos com uma referência (como, por exemplo, o modelo atualmente em serviço) para determinar se eles são "bons o suficiente" em relação à referência. Isto é feito avaliando ambos os modelos em um dataset de avaliação e computando seu desempenho em métricas (por exemplo, AUC, perda). Se as métricas do novo modelo atenderem aos critérios especificados pelo desenvolvedor em relação ao modelo de referência (por exemplo, AUC não for inferior), o modelo será autorizado, ou "abençoado" (marcado como bom), indicando ao [Pusher](pusher.md) que não há problema em enviar o modelo para produção.

- Consome:
    - Uma divisão de avaliação de [ExampleGen](https://www.tensorflow.org/tfx/guide/examplegen)
    - Um modelo treinado do [Trainer](trainer.md)
    - Um modelo previamente abençoado (se a validação for realizada)
- Produz:
    - Resultados da análise para [ML Metadata](mlmd.md)
    - Resultados de validação para [ML Metadata](mlmd.md) (se a validação for realizada)

## Evaluator e o TensorFlow Model Analysis

O Evaluator aproveita a biblioteca [TensorFlow Model Analysis](tfma.md) para realizar a análise, que por sua vez usa o [Apache Beam](beam.md) para processamento escalonável.

## Usando o componente Evaluator

Um componente de pipeline Evaluator normalmente é muito fácil de implantar e requer pouca personalização, já que a maior parte do trabalho é feita pelo próprio componente Evaluator TFX.

Para configurar o evaluator são necessárias as seguintes informações:

- Métricas a serem configuradas (necessárias apenas se métricas adicionais estiverem sendo adicionadas fora daquelas salvas com o modelo). Consulte [Métricas do Tensorflow Model Analysis](https://github.com/tensorflow/model-analysis/blob/master/g3doc/metrics.md) para mais informações.
- Fatias a serem configuradas (se nenhuma fatia for fornecida, uma fatia "geral" será adicionada por padrão). Consulte [Configuração do Tensorflow Model Analysis](https://github.com/tensorflow/model-analysis/blob/master/g3doc/setup.md) para mais informações.

Se a validação for incluída, as seguintes informações adicionais serão necessárias:

- Qual modelo comparar (último abençoado, etc.).
- Validações de modelo (limites) a serem verificadas. Consulte [Validações de modelo do Tensorflow Model Analysis](https://github.com/tensorflow/model-analysis/blob/master/g3doc/model_validations.md) para mais informações.

Quando ativada, a validação será realizada em todas as métricas e fatias que foram definidas.

Este é um exemplo de código típico:

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

O evaluator produz um [EvalResult](https://www.tensorflow.org/tfx/model_analysis/api_docs/python/tfma/EvalResult) (e opcionalmente um [ValidationResult](https://www.tensorflow.org/tfx/model_analysis/api_docs/python/tfma/ValidationResult) se a validação tiver sido usada) que pode ser carregado usando o [TFMA](tfma.md). A seguir está um exemplo de como carregar os resultados em um notebook Jupyter:

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

Mais detalhes estão disponíveis na [Referência da API do Evaluator](https://www.tensorflow.org/tfx/api_docs/python/tfx/v1/components/Evaluator).
