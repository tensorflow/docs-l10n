# O componente de pipeline Transform TFX

O componente de pipeline Transform TFX executa engenharia de características em tf.Examples produzidos por um componente [ExampleGen](examplegen.md), usando um esquema de dados criado por um componente [SchemaGen](schemagen.md) e produz um SavedModel, bem como estatísticas sobre dados pré-transformação e pós-transformação. Quando executado, o SavedModel aceitará tf.Examples produzido por um componente ExampleGen e produzirá os dados da característica transformada.

- Consome: tf.Examples de um componente ExampleGen e um esquema de dados de um componente SchemaGen.
- Produz: Um SavedModel para um componente Trainer, estatísticas de pré-transformação e pós-transformação.

## Configurando um componente Transform

Depois que seu `preprocessing_fn` for escrito, ele precisará ser definido num módulo Python que será fornecido ao componente Transform como entrada. Esse módulo será carregado por Transform e a função chamada `preprocessing_fn` será localizada e usada por Transform para construir o pipeline de pré-processamento.

```
transform = Transform(
    examples=example_gen.outputs['examples'],
    schema=schema_gen.outputs['schema'],
    module_file=os.path.abspath(_taxi_transform_module_file))
```

Além disso, você talvez queira fornecer opções para a computação de estatísticas pré-transformação ou pós-transformação baseadas no [TFDV](tfdv.md). Para fazer isso, defina `stats_options_updater_fn` dentro do mesmo módulo.

## Transform e TensorFlow Transform

O Transform faz uso extensivo do [TensorFlow Transform](tft.md) para realizar engenharia de recursos no seu dataset. O TensorFlow Transform é uma ótima ferramenta para transformar dados de características antes que sejam enviados para o seu modelo e como parte do processo de treinamento. As transformações de características comuns incluem:

- **Embedding**: conversão de características esparsas (como os IDs inteiros produzidos por um vocabulário) em características densas, encontrando um mapeamento significativo do espaço de alta dimensão para o espaço de baixa dimensão. Veja [Unidade de Embeddings no Machine-learning Crash Course](https://developers.google.com/machine-learning/crash-course/embedding) para uma introdução aos embeddings.
- **Geração de vocabulário**: conversão de strings ou outras características não numéricas para inteiras, criando um vocabulário que mapeia cada valor exclusivo para um número de identificação.
- **Normalização de valores**: transformando características numéricas para que todos caiam num intervalo semelhante.
- **Bucketização**: conversão de características de valor contínuo para características categóricas, atribuindo valores a intervalos discretos.
- **Enriquecimento de características textuais**: produção de textuais a partir de dados brutos, como tokens, n-gramas, entidades, sentimentos, etc., para enriquecer o conjunto de características.

O TensorFlow Transform oferece suporte para estas e muitos outros tipos de transformações:

- Gerar automaticamente um vocabulário a partir de seus dados mais recentes.

- Realizar transformações arbitrárias nos seus dados antes de enviá-los ao seu modelo. O TensorFlow Transform cria transformações no grafo do TensorFlow para seu modelo para que as mesmas transformações sejam realizadas no momento do treinamento e da inferência. Você pode definir transformações que se referem a propriedades globais dos dados, como o valor máximo de uma característica em todas as instâncias de treinamento.

Você pode transformar seus dados como quiser antes de executar o TFX. Mas se você fizer isto no TensorFlow Transform, as transformações se tornarão parte do grafo do TensorFlow. Essa abordagem ajuda a evitar desvios de treinamento/serviço.

As transformações dentro do seu código de modelagem usam FeatureColumns. Usando FeatureColumns, você pode definir bucketizações, transformações em inteiro que usam vocabulários predefinidos ou quaisquer outras transformações que possam ser definidas sem examinar os dados.

Por outro lado, o TensorFlow Transform foi projetado para transformações que exigem uma passagem completa dos dados para computar valores que não são conhecidos antecipadamente. Por exemplo, a geração de vocabulário requer uma passagem completo por todos os dados.

Observação: essas computações são implementadas internamente no [Apache Beam](https://beam.apache.org/) .

Além de computar valores usando o Apache Beam, o TensorFlow Transform permite que os usuários incorporem esses valores em um grafo do TensorFlow, que pode então ser carregado no grafo de treinamento. Por exemplo, ao normalizar características, a função `tft.scale_to_z_score` calculará a média e o desvio padrão de uma característica e também uma representação, num grafo do TensorFlow, da função que subtrai a média e divide pelo desvio padrão. Ao produzir um grafo do TensorFlow, não apenas estatísticas, o TensorFlow Transform simplifica o processo de criação do pipeline de pré-processamento.

Ja que o pré-processamento é expresso num grafo, ele pode acontecer no servidor e tem garantia de consistência entre o treinamento e o serviço. Essa consistência elimina uma fonte de desvio do treinamento/serviço.

O TensorFlow Transform permite que os usuários especifiquem seu pipeline de pré-processamento usando código do TensorFlow. Isto significa que um pipeline é construído da mesma maneira que um grafo do TensorFlow. Se apenas as operações do TensorFlow fossem usadas neste grafo, o pipeline seria um mapa puro que aceita lotes de entrada e retorna lotes de saída. Tal pipeline seria equivalente a colocar este grafo dentro de seu `input_fn` ao usar a API `tf.Estimator`. Para especificar operações de passo completo, como quantis de computação, o TensorFlow Transform fornece funções especiais chamadas `analyzers` que aparecem como ops do TensorFlow, mas que na verdade especificam uma computação adiada que será feita pelo Apache Beam e que terá a saída inserida no grafo como um constante. Embora uma op comum do TensorFlow receba um único lote como entrada, execute algumas computações apenas nesse lote e produza um lote, um `analyzer` realizará uma redução global (implementada no Apache Beam) sobre todos os lotes e retornará o resultado.

Ao combinar operações comuns do TensorFlow e analisadores do TensorFlow Transform, os usuários podem criar pipelines complexos para pré-processar seus dados. Por exemplo, a função `tft.scale_to_z_score` recebe um tensor de entrada e retorna esse tensor normalizado para obter a média `0` e variância `1`. Ele faz isso chamando os analisadores `mean` e `var` nos bastidores, o que irá efetivamente gerar constantes no grafo iguais à média e à variância do tensor de entrada. Em seguida, ele usará operações do TensorFlow para subtrair a média e dividir pelo desvio padrão.

## A função `preprocessing_fn` do TensorFlow Transform

O componente TFX Transform simplifica o uso do Transform manipulando as chamadas de API relacionadas à leitura e gravação de dados e gravando a saída SavedModel no disco. Como usuário do TFX, você só precisa definir uma única função chamada `preprocessing_fn`. Em `preprocessing_fn` você define uma série de funções que manipulam o dict de tensores de entrada para produzir o dict de tensores de saída. Você encontrará funções helper como scale_to_0_1 e compute_and_apply_vocabulary na [API do TensorFlow Transform](/tfx/transform/api_docs/python/tft) ou usar funções regulares do TensorFlow conforme mostrado abaixo.

```python
def preprocessing_fn(inputs):
  """tf.transform's callback function for preprocessing inputs.

  Args:
    inputs: map from feature keys to raw not-yet-transformed features.

  Returns:
    Map from string feature key to transformed feature operations.
  """
  outputs = {}
  for key in _DENSE_FLOAT_FEATURE_KEYS:
    # If sparse make it dense, setting nan's to 0 or '', and apply zscore.
    outputs[_transformed_name(key)] = transform.scale_to_z_score(
        _fill_in_missing(inputs[key]))

  for key in _VOCAB_FEATURE_KEYS:
    # Build a vocabulary for this feature.
    outputs[_transformed_name(
        key)] = transform.compute_and_apply_vocabulary(
            _fill_in_missing(inputs[key]),
            top_k=_VOCAB_SIZE,
            num_oov_buckets=_OOV_SIZE)

  for key in _BUCKET_FEATURE_KEYS:
    outputs[_transformed_name(key)] = transform.bucketize(
        _fill_in_missing(inputs[key]), _FEATURE_BUCKET_COUNT)

  for key in _CATEGORICAL_FEATURE_KEYS:
    outputs[_transformed_name(key)] = _fill_in_missing(inputs[key])

  # Was this passenger a big tipper?
  taxi_fare = _fill_in_missing(inputs[_FARE_KEY])
  tips = _fill_in_missing(inputs[_LABEL_KEY])
  outputs[_transformed_name(_LABEL_KEY)] = tf.where(
      tf.is_nan(taxi_fare),
      tf.cast(tf.zeros_like(taxi_fare), tf.int64),
      # Test if the tip was > 20% of the fare.
      tf.cast(
          tf.greater(tips, tf.multiply(taxi_fare, tf.constant(0.2))), tf.int64))

  return outputs
```

### Entendendo as entradas para preprocessing_fn

O `preprocessing_fn` descreve uma série de operações em tensores (ou seja, objetos `Tensor`, `SparseTensor` ou `RaggedTensor`). Para definir corretamente o `preprocessing_fn` é necessário entender como os dados são representados como tensores. A entrada para `preprocessing_fn` é determinada pelo esquema. Um [proto `Schema`](https://github.com/tensorflow/metadata/blob/master/tensorflow_metadata/proto/v0/schema.proto#L72) será em algum momento convertido numa "feature spec" (especificação de característica), às vezes chamada de "parsing spec" (especificação de processamento) que é usada para o processamento dos dados. Veja mais detalhes sobre a lógica de conversão [aqui](https://github.com/tensorflow/metadata/blob/master/tfx_bsl/docs/schema_interpretation.md).

## Usando o TensorFlow Transform para lidar com rótulos em formato string

Normalmente, deseja-se usar o TensorFlow Transform para gerar um vocabulário e aplicar esse vocabulário para converter strings em números inteiros. Ao seguir este workflow, o `input_fn` construído no modelo produzirá como saída a string como inteiro. No entanto, os rótulos são uma exceção, porque para que o modelo seja capaz de mapear os rótulos de saída (inteiros) de volta para strings, o modelo precisa que o `input_fn` gere um rótulo em formato string, junto com uma lista de valores possíveis para o rótulo. Por exemplo, se os rótulos forem `cat` e `dog`, então a saída de `input_fn` deve consistir dessas strings brutas, e as chaves `["cat", "dog"]` precisam ser passadas para o estimador como um parâmetro (veja detalhes abaixo).

Para lidar com o mapeamento de rótulos em formato string para números inteiros, você deve usar o TensorFlow Transform para gerar um vocabulário. Demonstramos isto no trecho de código abaixo:

```python
def _preprocessing_fn(inputs):
  """Preprocess input features into transformed features."""

  ...


  education = inputs[features.RAW_LABEL_KEY]
  _ = tft.vocabulary(education, vocab_filename=features.RAW_LABEL_KEY)

  ...
```

A função de pré-processamento acima recebe a característica de entrada bruta (que também será retornada como parte da saída da função de pré-processamento) e chama `tft.vocabulary` nela. Isto tem como resultado a geração de um vocabulário para `education` que poderá ser acessado no modelo.

O exemplo também mostra como transformar um rótulo e depois gerar um vocabulário para o rótulo transformado. Em particular, ele recebe o rótulo bruto `education` e converte todos os rótulos, exceto os 5 principais (por frequência), em `UNKNOWN`, sem converter o rótulo em número inteiro.

No código do modelo, o classificador deve receber o vocabulário gerado por `tft.vocabulary` como argumento de `label_vocabulary`. Isto é feito primeiro lendo este vocabulário como uma lista através de uma função helper. Isto é mostrado no trecho abaixo. Observe que o código de exemplo usa o rótulo transformado discutido acima, mas aqui mostramos o código para o uso do rótulo bruto.

```python
def create_estimator(pipeline_inputs, hparams):

  ...

  tf_transform_output = trainer_util.TFTransformOutput(
      pipeline_inputs.transform_dir)

  # vocabulary_by_name() returns a Python list.
  label_vocabulary = tf_transform_output.vocabulary_by_name(
      features.RAW_LABEL_KEY)

  return tf.contrib.learn.DNNLinearCombinedClassifier(
      ...
      n_classes=len(label_vocab),
      label_vocabulary=label_vocab,
      ...)
```

## Configurando estatísticas de pré-transformação e pós-transformação

Conforme mencionado acima, o componente Transform chama o TFDV para calcular estatísticas de pré- e pós-transformação. O TFDV recebe como entrada um objeto [StatsOptions](https://github.com/tensorflow/data-validation/blob/master/tensorflow_data_validation/statistics/stats_options.py) opcional. Os usuários podem querer configurar este objeto para permitir certas estatísticas adicionais (por exemplo, estatísticas de NLP) ou para definir limites que sejam validados (por exemplo, frequência mínima/máxima do token). Para isto, defina `stats_options_updater_fn` no arquivo do módulo.

```python
def stats_options_updater_fn(stats_type, stats_options):
  ...
  if stats_type == stats_options_util.StatsType.PRE_TRANSFORM:
    # Update stats_options to modify pre-transform statistics computation.
    # Most constraints are specified in the schema which can be accessed
    # via stats_options.schema.
  if stats_type == stats_options_util.StatsType.POST_TRANSFORM
    # Update stats_options to modify post-transform statistics computation.
    # Most constraints are specified in the schema which can be accessed
    # via stats_options.schema.
  return stats_options
```

As estatísticas de pós-transformação geralmente tiram proveito do conhecimento do vocabulário usado para pré-processar uma característica. O nome do vocabulário para mapeamento de caminhos é fornecido ao StatsOptions (e, portanto, ao TFDV) para cada vocabulário gerado pelo TFT. Além disso, mapeamentos para vocabulários criados externamente podem ser adicionados (i) modificando diretamente o dicionário `vocab_paths` em StatsOptions ou (ii) usando `tft.annotate_asset`.
