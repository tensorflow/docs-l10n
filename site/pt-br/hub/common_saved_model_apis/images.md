# APIs de SavedModel comuns para tarefas com imagens

Esta página descreve como os [SavedModels do TF2](../tf2_saved_model.md) para tarefas relacionadas a imagens implementam a [API Reusable SavedModel](../reusable_saved_models.md) (isso substitui as [Assinaturas comuns para imagens](../common_signatures/images.md) do [formato TF1 Hub](../tf1_hub_module) descontinuado).

<a name="feature-vector"></a>

## Vetor de características de imagens

### Resumo do uso

Um **vetor de características de imagens** é um tensor unidimensional denso que representa uma imagem inteira, geralmente para uso por um classificador simples de alimentação para frente no modelo de consumidor (em CNNs clássicas, esse é o valor de gargalo após a extensão especial ter sido agrupada ou nivelada, mas antes de a classificação ser feita. Confira [classificação de imagens](#classification) abaixo).

Um SavedModel reutilizável para extração de características de imagens tem um método `__call__` no objeto raiz que mapeia um lote de imagens para um lote de vetores de características. Esse método pode ser usado da seguinte forma:

```python
obj = hub.load("path/to/model")  # That's tf.saved_model.load() after download.
images = ...  # A batch of images with shape [batch_size, height, width, 3].
features = obj(images)   # A batch with shape [batch_size, num_features].
```

No Keras, o equivalente é:

```python
features = hub.KerasLayer("path/to/model")(images)
```

A entrada segue a convenção geral para [entrada de imagens](#input). A documentação do modelo especifica o intervalo admissível de `height` (altura) e `width` (largura) da entrada.

A saída é um único tensor com dtype `float32` e formato `[batch_size, num_features]` (tamanho do lote, número de características). `batch_size` é o mesmo que na entrada. `num_features` é uma constante específica ao módulo independente do tamanho da entrada.

### Detalhes da API

A [API Reusable SavedModel](../reusable_saved_models.md) também fornece uma lista de `obj.variables` (por exemplo, para inicialização quando não estiver fazendo o carregamento adiantado – eager).

Um modelo com suporte a ajustes finos fornece uma lista de `obj.trainable_variables` (variáveis treináveis). Ele pode exigir que você passe `training=True` para executar no modo de treinamento (por exemplo, dropout). Alguns modelos permitem argumentos opcionais para sobrescrever hiperparâmetros (por exemplo, taxa de dropout; isso é descrito na documentação do modelo). O modelo também pode fornecer uma lista de `obj.regularization_losses` (perdas de regularização). Confira mais detalhes na [API Reusable SavedModel](../reusable_saved_models.md).

No Keras, isso é feito por `hub.KerasLayer`: inicialize com `trainable=True` para permitir os ajustes finos e (no caso raro de sobrescrita de hparam) com `arguments=dict(some_hparam=some_value, ...))`.

### Observações

Aplicar dropout às características de saída (ou não) deve ser deixado a cargo do consumidor do modelo. O SavedModel em si não deve fazer dropout nas saídas (mesmo se usar dropout internamente em outros lugares).

### Exemplos

SavedModels reutilizáveis para vetores de características de imagens são usados em:

- tutorial do Colab [Retreinando um classificador de imagens](https://colab.research.google.com/github/tensorflow/docs/blob/master/g3doc/en/hub/tutorials/tf2_image_retraining.ipynb).

<a name="classification"></a>

## Classificação de imagens

### Resumo do uso

A **classificação de imagens** mapeia os pixels de uma imagem para pontuações lineares (logits) a fim de fazer associação nas classes de uma taxonomia *selecionada pelo publicador do modelo*. Dessa forma, os consumidores do modelo podem tirar conclusões a partir da classificação específica aprendida pelo módulo do publicador (para classificação de imagens com um novo conjunto de classes, é comum reutilizar um modelo de [vetor de classificação de imagens](#feature-vector) com um novo classificador em seu lugar).

Um SavedModel reutilizável para classificação de imagens tem um método `__call__` no objeto raiz que mapeia um lote de imagens para um lote de logits. Esse método pode ser usado da seguinte forma:

```python
obj = hub.load("path/to/model")  # That's tf.saved_model.load() after download.
images = ...  # A batch of images with shape [batch_size, height, width, 3].
logits = obj(images)   # A batch with shape [batch_size, num_classes].
```

No Keras, o equivalente é:

```python
logits = hub.KerasLayer("path/to/model")(images)
```

A entrada segue a convenção geral para [entrada de imagens](#input). A documentação do modelo especifica o intervalo admissível de `height` (altura) e `width` (largura) da entrada.

A saída `logits` é um único tensor com dtype `float32` e formato `[batch_size, num_classes]` (tamanho do lote, número de classes). `batch_size` é o mesmo que na entrada. `num_classes` é o número de classes na classificação, que é uma constante específica ao modelo.

O valor `logits[i, c]` é uma pontuação prevendo a associação do exemplo `i` na classe com índice `c`.

O fato de essas pontuações deverem ser usadas com softmax (para classes mutuamente excludentes), sigmoide (para classes ortogonais) ou alguma outra coisa depende da classificação subjacente. A documentação do módulo deve descrever isso e incluir uma definição dos índices da classe.

### Detalhes da API

A [API Reusable SavedModel](../reusable_saved_models.md) também fornece uma lista de `obj.variables` (por exemplo, para inicialização quando não estiver fazendo o carregamento adiantado – eager).

Um modelo com suporte a ajustes finos fornece uma lista de `obj.trainable_variables` (variáveis treináveis). Ela pode exigir que você passe `training=True` para executar no modo de treinamento (por exemplo, dropout). Alguns modelos permitem argumentos opcionais para sobrescrever hiperparâmetros (por exemplo, taxa de dropout, isso é descrito na documentação do modelo). O modelo também pode fornecer uma lista de `obj.regularization_losses` (perdas de regularização). Confira mais detalhes na [API Reusable SavedModel](../reusable_saved_models.md).

No Keras, isso é feito por `hub.KerasLayer`: inicialize com `trainable=True` para permitir os ajustes finos e (no caso raro de sobrescrita de hparam) com `arguments=dict(some_hparam=some_value, ...))`.

<a name="input"></a>

## Imagem como entrada

É comum a todos os tipos de modelos de imagem.

Um modelo que receba um lote de imagens como entrada as recebe como um tensor denso de 4 dimensões com dtype `float32` e formato `[batch_size, height, width, 3]` (tamanho do lote, altura, largura, 3), cujos elementos são valores de cor em RGB dos pixels normalizados para o intervalo [0, 1]. Isso é o que você obtém com `tf.image.decode_*()`seguido por `tf.image.convert_image_dtype(..., tf.float32)`.

O modelo aceita qualquer `batch_size`. A documentação do modelo especifica o intervalo admissível de `height` e `width`. A última dimensão é fixa: 3 canais RGB.

É recomendável que os modelos usem o layout de Tensores `channels_last` (ou `NHWC`) em todo o código e deixem o otimizador de grafo do TensorFLow reescrever para `channels_first` (ou `NCHW`), se necessário.
