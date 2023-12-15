# APIs de SavedModel comuns para tarefas com texto

Esta página descreve como os [SavedModels do TF2](../tf2_saved_model.md) para tarefas relacionadas a texto implementam a [API Reusable SavedModel](../reusable_saved_models.md) (isso substitui as [Assinaturas comuns para texto](../common_signatures/text.md) do [formato TF1 Hub](../tf1_hub_module) descontinuado).

## Visão geral

Existem diversas APIs para computar **embeddings de texto** (também conhecidos como representações densas de texto ou vetores de características de texto).

- A API de *embeddings de texto a partir de entradas de texto* é implementada por um SavedModel que mapeia um lote de strings para um lote de vetores de embeddings. Isso é muito fácil de usar, e diversos modelos do TF Hub fizeram essa implementação. Porém, isso não permite que sejam feitos ajustes finos no modelo em TPUs.

- A API para *embeddings de texto com entradas pré-processadas* resolve a mesma tarefa, mas é implementada por dois SavedModels distintos:

    - um *pré-processador*, que pode ser executado dentro de um pipeline de entrada tf.data e converte strings e outros dados de tamanho variável em Tensores numéricos.
    - um *encoder*, que recebe os resultados do pré-processador e faz a parte treinável da computação de embeddings.

    Com essa divisão, as entradas podem ser pré-processadas de maneira assíncrona antes de serem alimentadas no loop de treinamento. Especificamente, isso permite criar encoders que podem ser executados e passar por ajustes finos em [TPUs](https://www.tensorflow.org/guide/tpu).

- A API de *embeddings de texto com encoders transformadores* estende a API de embeddings de texto para uma que não usa entradas pré-processadas em um caso específico de BERT e outros encoders transformadores.

    - O *pré-processador* é estendido para criar entradas do encoder a partir de mais de um segmento de texto de entrada.
    - O *encoder transformador* expõe os embeddings (com reconhecimento do contexto) de tokens individuais.

Em cada caso, as entradas de texto são strings com codificação UTF-8, geralmente texto sem formatação, a menos que a documentação do modelo indique o contrário.

Independentemente da API, modelos diferentes foram pré-treinados com texto de idiomas e domínios diferentes, e com tarefas diferentes em mente. Portanto, nem todo modelo de embedding de texto é adequado para todos os problemas.

<a name="feature-vector"></a>
<a name="text-embeddings-from-text"></a>

## Embedding de texto a partir de entradas de texto

Um SavedModel para **embeddings de texto a partir de entradas de texto** recebe um lote de entradas em um Tensor de strings de formato `[batch_size]` (tamanho do lote) e mapeia essas strings para um Tensor float32 de formato `[batch_size, dim]` (tamanho do lote, dimensão) com representações densas (vetores de características) das entradas.

### Sinopse do uso

```python
obj = hub.load("path/to/model")
text_input = ["A long sentence.",
              "single-word",
              "http://example.com"]
embeddings = obj(text_input)
```

Lembre-se de que, na [API Reusable SavedModel](../reusable_saved_models.md), executar o modelo no modo de treinamento (por exemplo, dropout) pode exigir um argumento palavra-chave `obj(..., training=True)` e que `obj` fornece atributos `.variables`, (variáveis) `.trainable_variables` (variáveis treináveis) e `.regularization_losses` (perdas de regularização) conforme aplicável.

No Keras, isso é feito assim:

```python
embeddings = hub.KerasLayer("path/to/model", trainable=...)(text_input)
```

### Treinamento distribuído

Se o embedding de texto for usado como parte de um modelo que é treinado com uma estratégia de distribuição, a chamada a `hub.load("path/to/model")` ou `hub.KerasLayer("path/to/model", ...)`, respectivamente, deve acontecer dentro do escopo de DistributionStrategy para criar as variáveis do modelo na maneira distribuída. Por exemplo:

```python
  with strategy.scope():
    ...
    model = hub.load("path/to/model")
    ...
```

### Exemplos

- Tutorial [Classificação de texto com avaliações de filmes](https://colab.research.google.com/github/tensorflow/docs/blob/master/g3doc/en/hub/tutorials/tf2_text_classification.ipynb) no Colab.

<a name="text-embeddings-preprocessed"></a>

## Embeddings de texto com entradas pré-processadas

Um **embedding de texto com entradas pré-processadas/strong0} é implementado por dois SavedModels distintos:**

- um **pré-processador** que mapeia um Tensor de strings de formato `[batch_size]` (tamanho do lote) para um dicionário de Tensores numéricos.
- um **encoder** que recebe um dicionário de Tensores retornado pelo pré-processador, faz a parte treinável da computação de embedding e retorna um dicionário de saídas. A saída com chave `"default"` é um Tensor float32 de formato `[batch_size, dim]` (tamanho do lote, dimensão).

Isso permite executar o pré-processador em um pipeline de entrada, mas permite fazer os ajustes finos dos embeddings computados pelo encoder como parte de um modelo maior. Especificamente, permite criar encoders que podem ser executados e passar por ajustes finos em [TPUs](https://www.tensorflow.org/guide/tpu).

É um detalhe da implementação definir quais Tensores são contidos na saída do pré-processador e quais (se houver) Tensores adicionais além do `"default"` são contidos na saída do encoder.

A documentação do encoder precisa especificar qual pré-processador deve ser usado com ele. Geralmente, há exatamente uma escolha certa.

### Sinopse do uso

```python
text_input = tf.constant(["A long sentence.",
                          "single-word",
                          "http://example.com"])
preprocessor = hub.load("path/to/preprocessor")  # Must match `encoder`.
encoder_inputs = preprocessor(text_input)

encoder = hub.load("path/to/encoder")
enocder_outputs = encoder(encoder_inputs)
embeddings = enocder_outputs["default"]
```

Lembre-se de que, na [API Reusable SavedModel](../reusable_saved_models.md), executar o encoder no modo de treinamento (por exemplo, dropout) pode exigir um argumento palavra-chave `encoder(..., training=True)` e que `encoder` fornece atributos `.variables`, (variáveis) `.trainable_variables` (variáveis treináveis) e `.regularization_losses` (perdas de regularização) conforme aplicável.

O modelo de `preprocessor` pode ter `.variables`, mas elas não devem ser treinadas ainda mais. O pré-processamento não depende do modo: se `preprocessor()` tiver um argumento `training=...`, ele não tem efeito nenhum.

No Keras, isso é feito assim:

```python
encoder_inputs = hub.KerasLayer("path/to/preprocessor")(text_input)
encoder_outputs = hub.KerasLayer("path/to/encoder", trainable=True)(encoder_inputs)
embeddings = encoder_outputs["default"]
```

### Treinamento distribuído

Se o encoder for usado como parte de um modelo que é treinado com uma estratégia de distribuição, a chamada a `hub.load("path/to/encoder")` ou `hub.KerasLayer("path/to/encoder", ...)`, respectivamente, deve acontecer dentro de:

```python
  with strategy.scope():
    ...
```

para recriar as variáveis do encoder na maneira distribuída.

Da mesma forma, se o pré-processador fizer parte do modelo treinado (como no exemplo simples acima), ele também precisa ser carregado dentro do escopo de estratégia distribuída. Porém, se o pré-processador for usado em um pipeline de entrada (por exemplo, em um callable passado para `tf.data.Dataset.map()`), o carregamento deve ocorrer *fora* do escopo da estratégia de distribuição para colocar suas variáveis (se houver) na CPU do host.

### Exemplos

- Tutorial [Classificar texto com BERT](https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/text/classify_text_with_bert.ipynb) no Colab.

<a name="transformer-encoders"></a>

## Embeddings de texto com encoders transformadores

Os encoders transformadores para texto fazem operações nas sequências de entrada de um lote, em que cada sequência é composta por *n* ≥ 1 segmentos de texto tokenizado, dentro de um limite específico do modelo em *n*. Para BERT e muitas de suas extensões, esse limite é 2, então aceitam segmentos únicos e pares de segmentos.

A API de <em>embeddings de texto com encoders transformadores</em> estende a API de embeddings de texto com entradas pré-processadas com essa configuração.

### Pré-processador

Um SavedModel pré-processador para embeddings de texto com encoders transformadores implementa a API de um SavedModel pré-processador para embeddings de texto com entradas pré-processadas (veja acima), que conta com uma forma de mapear entradas de texto de elemento único diretamente para entradas do encoder.

Além disso, o SavedModel pré-processador fornece subobjetos `tokenize` chamáveis  para tokenização (separadamente por segmento) e `bert_pack_inputs` para agrupar *n* segmentos tokenizados em uma sequência de entrada para o encoder. Cada subobjeto segue a [API Reusable SavedModel](../reusable_saved_models.md).

#### Sinopse do uso

Vejamos um exemplo concreto de dois segmentos de texto: uma tarefa de consequência lógica que pergunta se uma premissa (primeiro segmento) implica ou não uma hipótese (segundo segmento).

```python
preprocessor = hub.load("path/to/preprocessor")

# Tokenize batches of both text inputs.
text_premises = tf.constant(["The quick brown fox jumped over the lazy dog.",
                             "Good day."])
tokenized_premises = preprocessor.tokenize(text_premises)
text_hypotheses = tf.constant(["The dog was lazy.",  # Implied.
                               "Axe handle!"])       # Not implied.
tokenized_hypotheses = preprocessor.tokenize(text_hypotheses)

# Pack input sequences for the Transformer encoder.
seq_length = 128
encoder_inputs = preprocessor.bert_pack_inputs(
    [tokenized_premises, tokenized_hypotheses],
    seq_length=seq_length)  # Optional argument.
```

No Keras, essa computação pode ser expressada como:

```python
tokenize = hub.KerasLayer(preprocessor.tokenize)
tokenized_hypotheses = tokenize(text_hypotheses)
tokenized_premises = tokenize(text_premises)

bert_pack_inputs = hub.KerasLayer(
    preprocessor.bert_pack_inputs,
    arguments=dict(seq_length=seq_length))  # Optional argument.
encoder_inputs = bert_pack_inputs([tokenized_premises, tokenized_hypotheses])
```

#### Detalhes de `tokenize`

Uma chamada a `preprocessor.tokenize()` recebe um Tensor de strings de formato `[batch_size]` (tamanho do lote) e retorna um [RaggedTensor](https://www.tensorflow.org/guide/ragged_tensor) de formato `[batch_size, ...]` cujos valores são IDs de token int32 representando as strings da entrada. Pode haver *r* ≥ 1 dimensões irregulares após `batch_size`, mas nenhuma outra dimensão uniforme.

- Se *r*=1, o formato é `[batch_size, (tokens)]`, e cada entrada é tokenizada em uma sequência plana de tokens.
- Se *r*&gt;1, há *r*-1 níveis adicionais de agrupamento. Por exemplo, [tensorflow_text.BertTokenizer](https://github.com/tensorflow/text/blob/v2.3.0/tensorflow_text/python/ops/bert_tokenizer.py#L138) usa *r*=2 para agrupar tokens por palavras e gera o formato `[batch_size, (words), (tokens_per_word)]`. Cabe ao modelo em questão quantos desses níveis extras existem, se houver algum, e quais agrupamentos eles representam.

O usuário pode (mas não precisa) modificar entradas tokenizadas, por exemplo, para acomodar o limite seq_length (comprimento da sequência) que será imposto ao agrupar as entradas do encoder. As dimensões extras na saída do tokenizador podem ajudar (por exemplo, para respeitar as fronteiras de palavras), mas se tornam inúteis no próximo passo.

Quanto à [API Reusable SavedModel](../reusable_saved_models.md), o objeto `preprocessor.tokenize` pode ter `.variables` (variáveis), mas elas não devem ser treinadas ainda mais. A tokenization não depende do modo: se `preprocessor.tokenize()` tiver um argumento `training=...`, ele não tem efeito nenhum.

#### Detalhes de `bert_pack_inputs`

Uma chamada a `preprocessor.bert_pack_inputs()` recebe uma lista Python de entradas tokenizadas (divididas em lotes separadamente para cada segmento de entrada) e retorna um dicionário de Tensores representando um lote de sequências de entradas de tamanho fixo para o modelo de encoder transformador.

Cada entrada tokenizada é um RaggedTensor int32 de formato `[batch_size, ...]`, (tamanho do lote), em que o número *r* de dimensões irregulares após batch_size é 1 ou o mesmo que da saída de `preprocessor.tokenize().` (esse último é apenas por conveniência; as dimensões extras são niveladas antes do agrupamento).

O agrupamento adiciona tokens especiais ao redor dos segmentos de entrada, conforme esperado pelo encoder. A chamada `bert_pack_inputs()` implementa exatamente o esquema de agrupamento usado pelos modelos BERT originais e muitas de suas extensões: a sequência agrupada começa com um token de início da sequência, seguida pelos segmentos tokenizados, em que cada um termina com um token de fim de segmento. As posições remanescentes até  seq_length (comprimento da sequência), se houver, são preenchidas com tokens de preenchimento.

Se uma sequência agrupada excedesse seq_length (comprimento da sequência), `bert_pack_inputs()` truncaria seus segmentos para prefixos de tamanhos aproximadamente iguais para que a sequência agrupada caiba exatamente dentro de seq_length.

O agrupamento não depende do modo: se `preprocessor.bert_pack_inputs()` tiver um argumento `training=...`, ele não tem efeito nenhum. Além disso, não se espera que `preprocessor.bert_pack_inputs` tenha variáveis ou tenha suporte a ajustes finos.

### Encoder

O encoder é chamado no dicionário de `encoder_inputs` da mesma forma que na API de embeddings de texto com entradas pré-processadas (confira acima), incluindo as disposições da [API Reusable SavedModel](../reusable_saved_models.md).

#### Sinopse do uso

```python
enocder = hub.load("path/to/encoder")
enocder_outputs = encoder(encoder_inputs)
```

Ou o equivalente no Keras:

```python
encoder = hub.KerasLayer("path/to/encoder", trainable=True)
encoder_outputs = encoder(encoder_inputs)
```

#### Detalhes

`encoder_outputs` (saídas do encoder) são um dicionário de Tensores com as seguintes chaves.

<!-- TODO(b/172561269): More guidance for models trained without poolers. -->

- `"sequence_output"` (saída de sequências): um Tensor float32 de formato `[batch_size, seq_length, dim]` (tamanho do lote, comprimento da sequência, dimensão), com o embedding (com reconhecimento de contexto) de cada token de toda sequência de entradas agrupada.
- `"pooled_output"` (saída em pool): um Tensor float32 de formato `[batch_size, dim]` (tamanho do lote, dimensão), com o embedding de cada sequência de entradas como um todo, derivada de sequence_output (saída de sequências) em alguma forma treinável.
- `"default"` (padrão), conforme exigido pela API de embeddings de texto com entradas pré-processadas: um Tensor float32 de formato `[batch_size, dim]` (tamanho do lote, dimensão), com o embedding de cada sequência de entradas (pode ser apenas um alias de pooled_output).

O conteúdo de `encoder_inputs` (entradas do encoder) não é estritamente exigido pela definição dessa API. Porém, para encoders que usam entradas com o estilo de BERT, recomenda-se usar os seguintes nomes (do [Kit de ferramentas de modelagem de NLP para o TensorFlow Model Garden](https://github.com/tensorflow/models/tree/master/official/nlp)) para minimizar o atrito ao intercambiar encoders e reutilizar modelos de pré-processador:

- `"input_word_ids"` (IDs de palavras de entrada): um Tensor int32 de formato `[batch_size, seq_length]` (tamanho do lote, comprimento da sequência) com os IDs de token da sequência de entradas agrupada (ou seja, inclui um token de início de sequência, tokens de fim de segmento e preenchimento).
- `"input_mask"` (máscara de entrada): um Tensor int32 de formato `[batch_size, seq_length]` (tamanho do lote, comprimento da sequência), com valor 1 na posição de todos os tokens de entrada presentes antes do preenchimento e valor 0 para os tokens de preenchimento.
- `"input_type_ids"` (IDs de tipo de entrada): um Tensor int32 de formato `[batch_size, seq_length]` (tamanho do lote, comprimento da sequência), com o índice do segmento de entrada que deu origem ao token de entrada na respectiva posição. O primeiro segmento de entrada (índice 0) inclui o token de início de sequência e seu token de fim de segmento. O segundo segmento e posteriores (se presentes) incluem seu respectivo token de fim de segmento. Os tokens de preenchimento recebem o índice 0 novamente.

### Treinamento distribuído

Para carregar os objetos pré-processador e encoder dentro ou fora de um escopo de estratégia de distribuição, aplicam-se as mesmas regras da API de embeddings de texto com entradas pré-processadas (confira acima).

### Exemplos

- Tutorial [Resolva tarefas GLUE usando BERT em TPUs](https://colab.research.google.com/github/tensorflow/docs-l10n/blob/master/site/pt-br/text/tutorials/bert_glue.ipynb) no Colab.
