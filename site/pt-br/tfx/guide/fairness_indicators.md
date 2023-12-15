# Fairness Indicators

A ferramenta Fairness Indicators foi projetada para apoiar as equipes na avaliação e melhoria de modelos para questões de equidade em parceria com o kit de ferramentas mais abrangente do Tensorflow. A ferramenta é usada internamente por muitos de nossos produtos e agora está disponível em BETA para você testar nos seus próprios casos de uso.

![Dashboard do Fairness Indicator](https://github.com/tensorflow/docs-l10n/blob/master/site/pt-br/tfx/guide/images/fairnessIndicators.png?raw=true)

## O que é o Fairness Indicators?

Fairness Indicators (indicadores de equidade) é uma biblioteca que permite a computação fácil de métricas de equidade frequentemente identificadas para classificadores binários e multiclasse. Muitas ferramentas existentes para avaliar questões de equidade não funcionam bem em datasets e modelos de grande escala. No Google, é importante que tenhamos ferramentas que possam funcionar em sistemas com bilhões de usuários. O Fairness Indicators permitire que você avalie casos de uso de qualquer tamanho.

Em particular, o Fairness Indicators inclui a capacidade de:

- Avaliar a distribuição de datasets
- Avaliar o desempenho de modelos, divididos em grupos de usuários definidos
    - Trazer confiança em relação aos seus resultados com intervalos de confiança e avaliações em múltiplod limites
- Mergulhar fundo em fatias individuais para explorar as causas raízes e oportunidades de melhoria

Este [estudo de caso](https://developers.google.com/machine-learning/practica/fairness-indicators), completo com [vídeos](https://www.youtube.com/watch?v=pHT-ImFXPQo) e exercícios de programação, demonstra como o Fairness Indicators pode ser usado ​​num de seus próprios produtos para avaliar preocupações quanto à equidade ao longo do tempo.

[](http://www.youtube.com/watch?v=pHT-ImFXPQo)

O download do pacote pip inclui:

- **[Tensorflow Data Validation (TFDV)](https://www.tensorflow.org/tfx/data_validation/get_started)**
- **[Tensorflow Model Analysis (TFMA)](https://www.tensorflow.org/tfx/model_analysis/get_started)**
    - **Fairness Indicators**
- **[The What-If Tool (WIT)](https://www.tensorflow.org/tensorboard/what_if_tool)**

## Usando Fairness Indicators com modelos Tensorflow

### Dados

Para executar Fairness Indicators com o TFMA, certifique-se de que o dataset de avaliação esteja rotulado para as características que você gostaria usar para fatiar. Se você não tiver as características exatas de fatiamento para suas questões de equidade, você pode tentar encontrar um conjunto de avaliações que tenha, ou considerar usar características intermediárias no seu conjunto de características que possam destacar disparidades dos resultados. Para orientação adicional, clique [aqui](https://tensorflow.org/responsible_ai/fairness_indicators/guide/guidance).

### Modelos

Você pode usar a classe Tensorflow Estimator para construir seu modelo. O suporte para modelos Keras chegará em breve ao TFMA. Se você deseja executar o TFMA num modelo Keras, consulte a seção “TFMA independente de modelo” abaixo.

Depois do treinamento do seu Estimator, você precisará exportar um modelo salvo para fins de avaliação. Para saber mais, veja o [guia do TFMA](/tfx/model_analysis/get_started).

### Configurando fatias

Em seguida, defina as fatias que você gostaria de avaliar:

```python
slice_spec = [
  tfma.slicer.SingleSliceSpec(columns=[‘fur color’])
]
```

Se quiser avaliar fatias interseccionais (por exemplo, cor e altura do pelo), você pode definir o seguinte:

```python
slice_spec = [
  tfma.slicer.SingleSliceSpec(columns=[‘fur_color’, ‘height’])
]`
```

### Computação de métricas de equidade

Adicione um callback do Fairness Indicators à lista `metrics_callback`. No retorno de chamada, você poderá definir uma lista de limites nos quais o modelo será avaliado.

```python
from tensorflow_model_analysis.addons.fairness.post_export_metrics import fairness_indicators

# Build the fairness metrics. Besides the thresholds, you also can config the example_weight_key, labels_key here. For more details, please check the api.
metrics_callbacks = \
    [tfma.post_export_metrics.fairness_indicators(thresholds=[0.1, 0.3,
     0.5, 0.7, 0.9])]

eval_shared_model = tfma.default_eval_shared_model(
    eval_saved_model_path=tfma_export_dir,
    add_metrics_callbacks=metrics_callbacks)
```

Antes de executar a configuração, determine se deseja ou não ativar a computação de intervalos de confiança. Os intervalos de confiança são calculados usando inicialização de Poisson e requerem recomputação em 20 amostras.

```python
compute_confidence_intervals = True
```

Execute o pipeline de avaliação do TFMA:

```python
validate_dataset = tf.data.TFRecordDataset(filenames=[validate_tf_file])

# Run the fairness evaluation.
with beam.Pipeline() as pipeline:
  _ = (
      pipeline
      | beam.Create([v.numpy() for v in validate_dataset])
      | 'ExtractEvaluateAndWriteResults' >>
       tfma.ExtractEvaluateAndWriteResults(
                 eval_shared_model=eval_shared_model,
                 slice_spec=slice_spec,
                 compute_confidence_intervals=compute_confidence_intervals,
                 output_path=tfma_eval_result_path)
  )
eval_result = tfma.load_eval_result(output_path=tfma_eval_result_path)
```

### Renderização de Fairness Indicators

```python
from tensorflow_model_analysis.addons.fairness.view import widget_view

widget_view.render_fairness_indicator(eval_result=eval_result)
```

![Fairness Indicators](images/fairnessIndicators.png)

Dicas para usar Fairness Indicators:

- **Selecione as métricas a serem exibidas** marcando as caixas do lado esquerdo. Grafos individuais para cada uma das métricas aparecerão no widget, em ordem.
- **Altere a fatia de referência**, a primeira barra do grafo, usando o menu dropdown. Os deltas serão calculados com base neste valor de referência.
- **Selecione limites** usando o menu dropdown. Você pode visualizar vários limites no mesmo grafo. Os limites selecionados estarão em negrito e você pode clicar sobre um limite em negrito para deselecioná-lo.
- **Passe o mouse sobre uma barra** para ver as métricas dessa fatia.
- **Identifique disparidades com a linha de referência** usando a coluna "Diff w. baseline" ("Diferença com linha de referência"), que identifica a diferença percentual entre a fatia atual e a referência.
- **Explore detalhadamente os pontos de dados de uma fatia** usando a [ferramenta What-If](https://pair-code.github.io/what-if-tool/). Veja [aqui](https://github.com/tensorflow/fairness-indicators/) um exemplo.

#### Renderizando Fairness Indicators para múltiplos modelos

Os Fairness Indicators também podem ser usados ​​para comparar modelos. Em vez de passar um único eval_result, passe um objeto multi_eval_results, que é um dicionário que mapeia dois nomes de modelos para objetos eval_result.

```python
from tensorflow_model_analysis.addons.fairness.view import widget_view

eval_result1 = tfma.load_eval_result(...)
eval_result2 = tfma.load_eval_result(...)
multi_eval_results = {"MyFirstModel": eval_result1, "MySecondModel": eval_result2}

widget_view.render_fairness_indicator(multi_eval_results=multi_eval_results)
```

![Fairness Indicators - comparação de modelos](https://github.com/tensorflow/docs-l10n/blob/master/site/pt-br/tfx/guide/images/fairnessIndicators.png?raw=true)

A comparação de modelos pode ser usada junto com a comparação de limites. Por exemplo, você pode comparar dois modelos em dois conjuntos de limites para encontrar a combinação ideal para suas métricas de equidade.

## Usando Fairness Indicators com modelos não TensorFlow

Para melhor atender aos clientes que possuem diferentes modelos e workflows, desenvolvemos uma biblioteca de avaliação que independe do modelo que está sendo avaliado.

Qualquer pessoa que queira avaliar seu sistema de aprendizado de máquina pode usá-lo, especialmente se você tiver modelos não baseados no TensorFlow. Usando o Apache Beam Python SDK, você pode criar um binário de avaliação TFMA independente e executá-lo para analisar seu modelo.

### Dados

Esta etapa tem como objetivo fornecer o dataset no qual você deseja que as avaliações sejam executadas. Deve estar no formato proto [tf.Example](https://www.tensorflow.org/tutorials/load_data/tfrecord) com rótulos, previsões e outras caraterísticas que você pode querer dividir.

```python
tf.Example {
    features {
        feature {
          key: "fur_color" value { bytes_list { value: "gray" } }
        }
        feature {
          key: "height" value { bytes_list { value: "tall" } }
        }
        feature {
          key: "prediction" value { float_list { value: 0.9 } }
        }
        feature {
          key: "label" value { float_list { value: 1.0 } }
        }
    }
}
```

### Modelos

Em vez de especificar um modelo, você cria um extrator e configuração de avaliação independente do modelo para processar e fornecer os dados que o TFMA precisa para computar as métricas. A especificação [ModelAgnosticConfig](https://github.com/tensorflow/model-analysis/blob/master/tensorflow_model_analysis/model_agnostic_eval/model_agnostic_predict.py) define as características, previsões e rótulos a serem usados ​​nos exemplos de entrada.

Para isso, crie um mapa de características com chaves que representam todas as características, incluindo chaves de rótulo e previsão e valores que representam o tipo de dados da característica.

```python
feature_map[label_key] = tf.FixedLenFeature([], tf.float32, default_value=[0])
```

Crie uma configuração independente de modelo usando chaves de rótulo, chaves de previsão e mapa de características.

```python
model_agnostic_config = model_agnostic_predict.ModelAgnosticConfig(
    label_keys=list(ground_truth_labels),
    prediction_keys=list(predition_labels),
    feature_spec=feature_map)
```

### Configuração de um Extractor independente de modelos

O [Extractor](https://github.com/tensorflow/model-analysis/blob/master/tensorflow_model_analysis/model_agnostic_eval/model_agnostic_extractor.py) é usado para extrair as características, rótulos e previsões da entrada usando uma configuração independente de modelos (model agnostic). E se quiser fatiar seus dados, você também precisará definir a [slice key spec](https://github.com/tensorflow/model-analysis/tree/master/tensorflow_model_analysis/slicer), contendo informações sobre as colunas que deseja fatiar.

```python
model_agnostic_extractors = [
    model_agnostic_extractor.ModelAgnosticExtractor(
        model_agnostic_config=model_agnostic_config, desired_batch_size=3),
    slice_key_extractor.SliceKeyExtractor([
        slicer.SingleSliceSpec(),
        slicer.SingleSliceSpec(columns=[‘height’]),
    ])
]
```

### Computação de métricas de equidade

Como parte do [EvalSharedModel](https://www.tensorflow.org/tfx/model_analysis/api_docs/python/tfma/types/EvalSharedModel), você pode fornecer todas as métricas nas quais deseja que seu modelo seja avaliado. As métricas são fornecidas na forma de retornos de chamada de métricas, como os definidos em [post_export_metrics](https://github.com/tensorflow/model-analysis/blob/master/tensorflow_model_analysis/post_export_metrics/post_export_metrics.py) ou [fairness_indicators](https://github.com/tensorflow/model-analysis/blob/master/tensorflow_model_analysis/addons/fairness/post_export_metrics/fairness_indicators.py).

```python
metrics_callbacks.append(
    post_export_metrics.fairness_indicators(
        thresholds=[0.5, 0.9],
        target_prediction_keys=[prediction_key],
        labels_key=label_key))
```

Ele também recebe um `construct_fn` que é usado para criar um grafo tensorflow para realizar a avaliação.

```python
eval_shared_model = types.EvalSharedModel(
    add_metrics_callbacks=metrics_callbacks,
    construct_fn=model_agnostic_evaluate_graph.make_construct_fn(
        add_metrics_callbacks=metrics_callbacks,
        fpl_feed_config=model_agnostic_extractor
        .ModelAgnosticGetFPLFeedConfig(model_agnostic_config)))
```

Depois que tudo estiver configurado, use uma das funções `ExtractEvaluate` ou `ExtractEvaluateAndWriteResults` fornecidas por [model_eval_lib](https://github.com/tensorflow/model-analysis/blob/master/tensorflow_model_analysis/api/model_eval_lib.py) para avaliar o modelo.

```python
_ = (
    examples |
    'ExtractEvaluateAndWriteResults' >>
        model_eval_lib.ExtractEvaluateAndWriteResults(
        eval_shared_model=eval_shared_model,
        output_path=output_path,
        extractors=model_agnostic_extractors))

eval_result = tensorflow_model_analysis.load_eval_result(output_path=tfma_eval_result_path)
```

Finalmente, renderize Fairness Indicators usando as instruções da seção "Renderizando Fairness Indicators" acima.

## Mais exemplos

O [diretório de exemplos d Fairness Indicators](https://github.com/tensorflow/fairness-indicators/blob/master/g3doc/tutorials/) contém vários exemplos:

- [Fairness_Indicators_Example_Colab.ipynb](https://github.com/tensorflow/fairness-indicators/blob/master/g3doc/tutorials/Fairness_Indicators_Example_Colab.ipynb) fornece uma visão geral do Fairness Indicators no [TensorFlow Model Analysis](https://www.tensorflow.org/tfx/guide/tfma) e como usá-lo com um dataset de verdade. Este notebook também aborda o [TensorFlow Data Validation](https://www.tensorflow.org/tfx/data_validation/get_started) e a ferramenta [What-If](https://pair-code.github.io/what-if-tool/), duas ferramentas para analisar modelos do TensorFlow que vêm com indicadores de equidade.
- [Fairness_Indicators_on_TF_Hub.ipynb](https://github.com/tensorflow/fairness-indicators/blob/master/g3doc/tutorials/Fairness_Indicators_on_TF_Hub_Text_Embeddings.ipynb) demonstra como usar Fairness Indicators para comparar modelos treinados em diferentes [embeddings de texto](https://en.wikipedia.org/wiki/Word_embedding). Este notebook usa embeddings de texto do [TensorFlow Hub](https://www.tensorflow.org/hub), a biblioteca do TensorFlow para publicar, descobrir e reutilizar componentes do modelo.
- [Fairness_Indicators_TensorBoard_Plugin_Example_Colab.ipynb](https://github.com/tensorflow/fairness-indicators/blob/master/g3doc/tutorials/Fairness_Indicators_TensorBoard_Plugin_Example_Colab.ipynb) demonstra como visualizar Fairness Indicators no TensorBoard.
