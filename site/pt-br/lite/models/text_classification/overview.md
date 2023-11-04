# Classificação de texto

Use um modelo do TensorFlow Lite para categorizar um parágrafo em grupos predefinidos.

Observação: (1) para integrar um modelo existente, use a [TensorFlow Lite Task Library](https://www.tensorflow.org/lite/inference_with_metadata/task_library/nl_classifier) (biblioteca de tarefas do TensorFlow Lite). (2) Para personalizar um modelo, use o [TensorFlow Lite Model Maker](https://www.tensorflow.org/lite/models/modify/model_maker/text_classification) (criador de modelos do TensorFlow Lite).

## Como começar

<img src="images/screenshot.gif" class="attempt-right" style="max-width: 300px">

Se você for iniciante no TensorFlow Lite e estiver trabalhando com o Android, recomendamos conferir o guia da [TensorFlow Lite Task Library](../../inference_with_metadata/task_library/nl_classifier) (biblioteca de tarefas do TensorFlowLite) para integrar modelos de classificação de texto com somente algumas linhas de código. Além disso, você pode integrar o modelo usando a [TensorFlow Lite Interpreter Java API](../../guide/inference#load_and_run_a_model_in_java) (API Java Interpreter do TensorFlow Lite).

O exemplo do Android abaixo demonstra a implementação dos dois métodos como a [lib_task_api](https://github.com/tensorflow/examples/tree/master/lite/examples/text_classification/android/lib_task_api) e a [lib_interpreter](https://github.com/tensorflow/examples/tree/master/lite/examples/text_classification/android/lib_interpreter), respectivamente.

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/text_classification/android">Exemplo do Android</a>

Se você estiver usando outra plataforma que não o Android ou se já conhecer bem as APIs do TensorFlow Lite, pode baixar nosso modelo inicial de classificação de texto.

<a class="button button-primary" href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/text_classification/text_classification_v2.tflite">Baixar modelo inicial</a>

## Como funciona

A classificação de texto categoriza um parágrafo em grupos predeterminados com base no conteúdo.

O modelo pré-treinado prevê se o sentimento de um parágrafo é positivo ou negativo. Ele foi treinado usando o [Large Movie Review Dataset v1.0](http://ai.stanford.edu/~amaas/data/sentiment/) de Mass et al, composto por avaliações de fillmes do IMDB rotulados como positivas ou negativas.

Veja as etapas para classificar um parágrafo usando o modelo:

1. Tokenize o parágrafo e converta-o em uma lista de IDs de palavras usando um vocabulário predefinido.
2. Alimente o modelo do TensorFlow Lite com a lista.
3. Calcule a probabilidade de o parágrafo ser positivo ou negativo usando as saídas do modelo.

### Observações

- Só há suporte ao idioma inglês.
- O modelo foi treinado com o dataset de avaliações de filmes, então pode haver redução da exatidão ao classificar textos de outros domínios de conhecimento.

## Referenciais de desempenho

Os referenciais de desempenho são gerados com a ferramenta [descrita aqui](https://www.tensorflow.org/lite/performance/benchmarks).

<table>
  <thead>
    <tr>
      <th>Nome do modelo</th>
      <th>Tamanho do modelo</th>
      <th>Dispositivo</th>
      <th>CPU</th>
    </tr>
  </thead>
  <tr>
    <td rowspan="3">       <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/text_classification/text_classification_v2.tflite">Classificação de texto</a>
</td>
    <td rowspan="3">       0,6 MB</td>
    <td>Pixel 3 (Android 10)</td>
    <td>0,05 ms*</td>
  </tr>
   <tr>
     <td>Pixel 4 (Android 10)</td>
    <td>0,05 ms*</td>
  </tr>
   <tr>
     <td>iPhone XS (iOS 12.4.1)</td>
    <td>0,025 ms**</td>
  </tr>
</table>

* 4 threads usados.

** 2 threads usados no iPhone para o resultado com maior desempenho.

## Exemplo de saída

Texto | Negativo (0) | Positivo (1)
--- | --- | ---
O melhor filme que vi nos últimos | 25,3% | 74,7%
: anos. Recomendo muito!              :              :              : |  |
Que desperdício de tempo. | 72,5% | 27,5%

## Use seu dataset de treinamento

Confira este [tutorial](https://www.tensorflow.org/lite/models/modify/model_maker/text_classification) para aplicar a mesma técnica utilizada aqui para treinar um modelo de classificação de texto usando seus próprios datasets. Com o dataset adequado, você pode criar um modelo para outros casos de uso, como categorização de documentos ou detecção de comentários tóxicos.

## Leia mais sobre classificação de texto

- [Embeddings de palavras e tutorial para treinar este modelo](https://www.tensorflow.org/tutorials/text/word_embeddings)
