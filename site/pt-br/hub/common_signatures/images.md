# Assinaturas comuns para imagens

Esta página descreve assinaturas comuns que devem ser implementadas por módulos no [formato TF1 Hub](../tf1_hub_module.md) para tarefas relacionadas a imagens (para o [formato SavedModel do TF2](../tf2_saved_model.md), confira a [API SavedModel](../common_saved_model_apis/images.md) análoga).

Alguns módulos podem ser usados para mais de uma tarefa (por exemplo, módulos de classificação de imagens também costumam fazer extração de características). Portanto, cada módulo fornece (1) assinaturas nomeadas para todas as tarefas imaginadas pelo publicador e (2) uma assinatura padrão `output = m(images)` para sua tarefa principal concebida.

<a name="feature-vector"></a>

## Vetor de características de imagens

### Resumo do uso

Um **vetor de características de imagens** é um tensor unidimensional denso que representa uma imagem inteira, geralmente para classificação pelo modelo do consumidor (ao contrário das ativações intermediárias de CNNs, não oferece um detalhamento espacial. Ao contrário da [classificação de imagens](#classification), descarta a classificação aprendida pelo modelo do publicador).

Um módulo para extração de características de imagens tem uma assinatura padrão que mapeia um lote de imagens para um lote de vetores de características. Essa assinatura pode ser usada da seguinte forma:

```python
  module_spec = hub.load_module_spec("path/to/module")
  height, width = hub.get_expected_image_size(module_spec)
  images = ...  # A batch of images with shape [batch_size, height, width, 3].
  module = hub.Module(module_spec)
  features = module(images)   # A batch with shape [batch_size, num_features].
```

Além disso, define a assinatura nomeada correspondente.

### Especificação da assinatura

A assinatura nomeada para extração de vetores de características de imagens é invocada como:

```python
  outputs = module(dict(images=images), signature="image_feature_vector",
                   as_dict=True)
  features = outputs["default"]
```

A entrada segue a convenção geral para [entrada de imagens](#input).

O dicionário de saídas contém uma saída `"default"` com dtype `float32` e formato `[batch_size, num_features]` (tamanho do lote, número de características). `batch_size` é o mesmo que na entrada, mas não é conhecido no momento da construção do grafo. `num_features` é uma constante específica ao módulo independente do tamanho da entrada.

Esses vetores de características devem ser usáveis para a classificação com um classificador simples de feed-forward (como as características em pool da camada convolucional superior em uma CNN típica para classificação de imagens).

Aplicar dropout às características de saída (ou não) deve ser deixado a cargo do consumidor do modelo. O módulo em si não deve fazer dropout nas saídas (mesmo se usar dropout internamente em outros lugares).

O dicionário de saídas pode fornecer outras saídas, por exemplo, as ativações de camadas ocultas dentro do modelo. Suas chaves e valores são dependentes do módulo. Recomenda-se adicionar um prefixo às chaves dependentes de arquitetura com um nome de arquitetura (por exemplo, para evitar confundir a camada intermediária `"InceptionV3/Mixed_5c"` com a camada convolucional superior `"InceptionV2/Mixed_5c"`).

<a name="classification"></a>

## Classificação de imagens

### Resumo do uso

A **classificação de imagens** mapeia os pixels de uma imagem para pontuações lineares (logits) a fim de fazer associação nas classes de uma taxonomia *selecionada pelo publicador do modelo*. Dessa forma, os consumidores do modelo podem tirar conclusões a partir da classificação específica aprendida pelo módulo do publicador, e não apenas suas características subjacentes (confira [Vetor de características de imagens](#feature-vector)).

Um módulo para extração de características de imagens tem uma assinatura padrão que mapeia um lote de imagens para um lote de logits. Essa assinatura pode ser usada da seguinte forma:

```python
  module_spec = hub.load_module_spec("path/to/module")
  height, width = hub.get_expected_image_size(module_spec)
  images = ...  # A batch of images with shape [batch_size, height, width, 3].
  module = hub.Module(module_spec)
  logits = module(images)   # A batch with shape [batch_size, num_classes].
```

Além disso, define a assinatura nomeada correspondente.

### Especificação da assinatura

A assinatura nomeada para extração de vetores de características de imagens é invocada como:

```python
  outputs = module(dict(images=images), signature="image_classification",
                   as_dict=True)
  logits = outputs["default"]
```

A entrada segue a convenção geral para [entrada de imagens](#input).

O dicionário de saídas contém uma saída `"default"` com dtype `float32` e formato `[batch_size, num_classes]` (tamanho do lote, número de classes). `batch_size` é o mesmo que na entrada, mas não é conhecido no momento da construção do grafo. `num_classes` é o número de classes na classificação, que é uma constante conhecida independente do tamanho da entrada.

Avaliar `outputs["default"][i, c]` gera uma pontuação prevendo a associação do exemplo `i` na classe com índice `c`.

O fato de essas pontuações deverem ser usadas com softmax (para classes mutuamente excludentes), sigmoide (para classes ortogonais) ou alguma outra coisa depende da classificação subjacente. A documentação do módulo deve descrever isso e incluir uma definição dos índices da classe.

O dicionário de saídas pode fornecer outras saídas, por exemplo, as ativações de camadas ocultas dentro do modelo. Suas chaves e valores são dependentes do módulo. Recomenda-se adicionar um prefixo às chaves dependentes de arquitetura com um nome de arquitetura (por exemplo, para evitar confundir a camada intermediária `"InceptionV3/Mixed_5c"` com a camada convolucional superior `"InceptionV2/Mixed_5c"`).

<a name="input"></a>

## Imagem como entrada

É comum a todos os tipos de módulos de imagem e assinaturas de imagem.

Uma assinatura que receba um lote de imagens como entrada as recebe como um tensor denso de 4 dimensões com dtype `float32` e formato `[batch_size, height, width, 3]` (tamanho do lote, altura, largura, 3), cujos elementos são valores de cor em RGB dos pixels normalizados para o intervalo [0, 1]. Isso é o que você obtém com `tf.image.decode_*()`seguido por `tf.image.convert_image_dtype(..., tf.float32)`.

Um módulo com exatamente uma entrada (ou uma entrada principal) de imagens usa o nome `"images"` para essa entrada.

O módulo aceita qualquer `batch_size` e define a primeira dimensão de TensorInfo.tensor_shape como "unknown" (desconhecida). A última dimensão é fixada com o número `3` de canais RGB. As dimensões `height` (altura) e `width` (largura) são fixadas com o tamanho esperado das imagens de entrada (trabalhos futuros poderão remover essa restrição para módulos totalmente convolucionais).

Os consumidores do módulo não devem inspecionar o formato diretamente, mas sim obter a informação de tamanho chamando hub.get_expected_image_size() no módulo ou na especificação do módulo e devem redimensionar as imagens de entrada (geralmente, antes/durante a divisão em lotes).

Por questões de simplicidade, os módulos do TF Hub usam o layout de Tensores `channels_last` (ou `NHWC`) e deixam o otimizador de grafo do TensorFlow reescrever para `channels_first` (ou `NCHW`), se necessário. Tem sido assim por padrão desde a versão 1.7 do TensorFlow.
