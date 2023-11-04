# SavedModels do TF Hub no TensorFlow 2

O [formato SavedModel do TensorFlow 2](https://www.tensorflow.org/guide/saved_model) é a forma recomendada de compartilhar modelos e partes de modelos pré-treinados no TensorFlow Hub. Ele substitui o [formato antigo TF1 Hub](tf1_hub_module.md) e conta com um novo conjunto de APIs.

Esta página explica como reutilizar SavedModels do TF2 em um programa do TensorFlow 2 com a API de baixo nível `hub.load()` e seu encapsulador `hub.KerasLayer` (geralmente, `hub.KerasLayer`é combinado com outras camadas `tf.keras.layers` para criar um modelo do Keras ou a função `model_fn` de um Estimator do TF2). Essas APIs também podem carregar modelos legados no formato TF1 Hub, mas há certos limites. Confira o [guia de compatibilidade](model_compatibility.md).

Usuários do TensorFlow 1 podem atualizar para o TF 1.15 e usar as mesmas APIs. Versões mais antigas do TF1 não funcionam.

## Usando SavedModels do TF Hub

### Usando um SavedModel no Keras

O [Keras](https://www.tensorflow.org/guide/keras/) é uma API de alto nível do TensorFlow para criar modelos de aprendizado profundo por meio da composição de objetos Layer do Keras. A biblioteca `tensorflow_hub` conta com a classe `hub.KerasLayer`, que é inicializada com a URL (ou o caminho no sistema de arquivos) de um SavedModel e depois oferece a computação do SavedModel, incluindo os pesos pré-treinados.

Veja um exemplo de como usar um embedding de texto pré-treinado:

```python
import tensorflow as tf
import tensorflow_hub as hub

hub_url = "https://tfhub.dev/google/nnlm-en-dim128/2"
embed = hub.KerasLayer(hub_url)
embeddings = embed(["A long sentence.", "single-word", "http://example.com"])
print(embeddings.shape, embeddings.dtype)
```

Com base nisso, podemos criar um classificador de texto como sempre usando o Keras:

```python
model = tf.keras.Sequential([
    embed,
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])
```

O [Colab de classificação de texto](https://colab.research.google.com/github/tensorflow/docs/blob/master/g3doc/en/hub/tutorials/tf2_text_classification.ipynb) é um exemplo completo de como treinar e avaliar um classificador como esse.

Os pesos do modelo em uma camada `hub.KerasLayer` são definidos como não treináveis por padrão. Confira a seção abaixo sobre ajustes finos para ver como mudar isso. Como sempre, os pesos são compartilhados entre todas as aplicações do mesmo objeto de camada no Keras.

### Usando um SavedModel em um Estimator

Usuários da API [Estimator](https://www.tensorflow.org/tutorials/distribute/multi_worker_with_estimator) do TensorFlow para fazer treinamento distribuído podem usar SavedModels do TF Hub escrevendo a função `model_fn` utilizando  `hub.KerasLayer`, entre outras camadas `tf.keras.layers`.

### Por trás dos bastidores: download e cache de SavedModel

Ao usar um SavedModel do TensorFlow Hub (ou de outros servidores HTTPS que implementam seu protocolo de [hospedagem](hosting.md)), ele é baixado e descompactado no sistema de arquivos local se já não estiver presente. A variável de ambiente `TFHUB_CACHE_DIR` pode ser definida para sobrescrever o local temporário padrão para fazer cache de SavedModels baixados e descompactados. Confira mais detalhes em [Como fazer cache](caching.md).

### Usando um SavedModel no TensorFlow de baixo nível

#### Identificadores do modelo

Os SavedModels podem ser carregados a partir de um `handle` (identificador) especificado, em que `handle` é um caminho no sistema de arquivos, uma URL do modelo válida em TFhub.dev (por exemplo, "https://tfhub.dev/..."). As URLs de modelos do Kaggle espelham os identificadores de TFhub.dev de acordo com nossos termos e a licença associada aos ativos do modelo, como "https://www.kaggle.com/...". Os identificadores de modelos do Kaggle são equivalentes ao seu identificador correspondente em TFhub.dev.

A função `hub.load(handle)` baixa e descompacta um SavedModel (a menos que `handle` já seja um caminho no sistema de arquivos) e depois retorna o resultado do carregamento com a função integrada `tf.saved_model.load()` do TensorFlow. Portanto, `hub.load()` pode tratar qualquer SavedModel válido (diferentemente de seu `hub.Module` predecessor no TF1).

#### Tópico avançado: o que esperar do SavedModel após o carregamento

Dependendo do conteúdo do SavedModel, o resultado de `obj = hub.load(...)` pode ser invocado de diversas formas (conforme explicado mais detalhadamente no [guia sobre o SavedModel](https://www.tensorflow.org/guide/saved_model) do TensorFlow):

- As assinaturas de serviço do SavedModel (se existirem) são representadas como um dicionário de funções concretas e podem ser chamadas da seguinte forma: `tensors_out = obj.signatures["serving_default"](**tensors_in)`, com dicionários de tensores cujas chaves são os nomes respectivos de entrada e saída, sujeitos às restrições de formato e dtype da assinatura.

- Os métodos do objeto salvo decorados com [`@tf.function`](https://www.tensorflow.org/api_docs/python/tf/function) (se existirem) são restaurados como objetos tf.function que podem ser chamados por todas as combinações de argumentos tensores e não tensores para os quais foi feito [tracing](https://www.tensorflow.org/tutorials/customization/performance#tracing) de tf.function antes do salvamento. Especificamente, se houver um método `obj.__call__` com tracings adequados, `obj` pode ser chamado como uma função do Python. Veja um exemplo simples: `output_tensor = obj(input_tensor, training=False)`.

Isso proporciona grande liberdade em relação às interfaces que os SavedModels podem implementar. A [interface Reusable SavedModels](reusable_saved_models.md) (SavedModels Reutilizáveis) para `obj` define convenções de forma que o código do cliente, incluindo adaptadores como `hub.KerasLayer`, saiba como usar o SavedModel.

Alguns SavedModels podem não seguir essa convenção, especialmente cujos modelos não foram criados para serem reutilizados em modelos inteiros e apenas para fornecerem assinaturas de serviço.

As variáveis treináveis em um SavedModel são recarregadas como treináveis, e `tf.GradientTape` vai acompanhá-las por padrão. Confira algumas ressalvas na seção sobre ajustes finos abaixo e considere evitar isso inicialmente. Mesmo se você quiser fazer os ajustes finos, é bom conferir se `obj.trainable_variables` orienta a fazer o retreinamento somente de um subconjunto das variáveis treináveis originalmente.

## Criando SavedModels para o TF Hub

### Visão geral

O SavedModel é o formato de serialização padrão do TensorFlow para modelos ou partes de modelos treinados. Ele armazena os pesos treinados do modelo juntamente com as operações exatas do TensorFlow para realizar a sua computação. Pode ser usado independentemente do código que o criou. Especificamente, pode ser reutilizado em diferentes APIs de construção de modelos de alto nível, como o Keras, porque as operações do TensorFlow são sua linguagem básica comum.

### Como salvar usando o Keras

A partir do TensorFlow 2, o formato padrão do `tf.keras.Model.save()` e `tf.keras.models.save_model()` é SavedModel (e não HDF5). Os SavedModels resultantes podem ser usados com `hub.load()`, `hub.KerasLayer` e adaptadores similares em outras APIs de alto nível à medida que forem disponibilizadas.

Para compartilhar um modelo completo do Keras, basta salvá-lo com `include_optimizer=False`.

Para compartilhar uma parte de um modelo do Keras, individualize essa parte do modelo e depois salve. Você pode fazer isso no código desde o começo...

```python
piece_to_share = tf.keras.Model(...)
full_model = tf.keras.Sequential([piece_to_share, ...])
full_model.fit(...)
piece_to_share.save(...)
```

...ou cortar um pedaço a ser compartilhado posteriormente (se ele estiver alinhado às camadas do modelo completo):

```python
full_model = tf.keras.Model(...)
sharing_input = full_model.get_layer(...).get_output_at(0)
sharing_output = full_model.get_layer(...).get_output_at(0)
piece_to_share = tf.keras.Model(sharing_input, sharing_output)
piece_to_share.save(..., include_optimizer=False)
```

Os [Modelos do TensorFlow](https://github.com/tensorflow/models) no GitHub usam a primeira estratégia para BERT (confira [nlp/tools/export_tfhub_lib.py](https://github.com/tensorflow/models/blob/master/official/nlp/tools/export_tfhub_lib.py), observe a divisão entre `core_model` para exportação e `pretrainer` para restauração do checkpoint), e a segunda estratégia para ResNet (confira [legacy/image_classification/tfhub_export.py](https://github.com/tensorflow/models/blob/master/official/legacy/image_classification/resnet/tfhub_export.py)).

### Como salvar usando o TensorFlow de baixo nível

Isso requer um bom conhecimento do [Guia do SavedModel](https://www.tensorflow.org/guide/saved_model) do TensorFlow.

Se você quiser fornecer mais do que apenas uma assinatura como serviço, deve implementar a [interface reutilizável do SavedModel](reusable_saved_models.md). Confira conceitualmente:

```python
class MyMulModel(tf.train.Checkpoint):
  def __init__(self, v_init):
    super().__init__()
    self.v = tf.Variable(v_init)
    self.variables = [self.v]
    self.trainable_variables = [self.v]
    self.regularization_losses = [
        tf.function(input_signature=[])(lambda: 0.001 * self.v**2),
    ]

  @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
  def __call__(self, inputs):
    return tf.multiply(inputs, self.v)

tf.saved_model.save(MyMulModel(2.0), "/tmp/my_mul")

layer = hub.KerasLayer("/tmp/my_mul")
print(layer([10., 20.]))  # [20., 40.]
layer.trainable = True
print(layer.trainable_weights)  # [2.]
print(layer.losses)  # 0.004
```

## Ajustes finos

Treinar as variáveis já treinadas de um SavedModel importado juntamente com as do modelo subjacente é chamado de *fazer os ajustes finos* do SavedModel. Isso pode aumentar a qualidade, mas geralmente deixa o treinamento mais exigente (o que pode levar tempo, depender mais do otimizador e de seus hiperparâmetros, aumentar o risco de overfitting e exigir ampliação do dataset, especialmente para CNNs). Sugerimos que consumidores de SavedModel considerem fazer os ajustes finos somente após terem um bom processo de treinamento e somente se o publicador do SavedModel recomendar.

Ao fazer os ajustes finos, os parâmetros "contínuos" do modelo são treinados. Isso não muda as transformações embutidas no código, como a tokenização de entrada de texto e o mapeamento dos tokens às suas respectivas entradas em uma matriz de embeddings.

### Para consumidores de SavedModel

Ao criar uma camada `hub.KerasLayer` da seguinte forma

```python
layer = hub.KerasLayer(..., trainable=True)
```

é possível fazer os ajustes do SavedModel carregado pela camada. Os pesos treináveis e os regularizadores de pesos declarados no SavedModel são adicionados ao modelo do Keras, e a computação do SavedModel é executada no modo de treinamento (pense em dropout, etc.).

O [Colab de classificação de imagens](https://github.com/tensorflow/docs/blob/master/g3doc/en/hub/tutorials/tf2_image_retraining.ipynb) tem um exemplo completo com ajustes finos opcionais do início ao fim.

#### Nova exportação dos resultados com ajustes finos

Usuários avançados podem querer salvar os resultados com ajustes finos em um SavedModel para que possa ser usado em vez daquele carregado originalmente. Isso pode ser feito da seguinte forma:

```python
loaded_obj = hub.load("https://tfhub.dev/...")
hub_layer = hub.KerasLayer(loaded_obj, trainable=True, ...)

model = keras.Sequential([..., hub_layer, ...])
model.compile(...)
model.fit(...)

export_module_dir = os.path.join(os.getcwd(), "finetuned_model_export")
tf.saved_model.save(loaded_obj, export_module_dir)
```

### Para criadores de SavedModel

Ao criar um SavedModel para compartilhar no TensorFlow Hub, pense com antecedência se e como os consumidores devem fazer o ajuste fino, e forneça orientações na documentação.

Ao salvar um modelo do Keras, todas as mecânicas de ajuste fino devem funcionar (salvar perdas de regularização de pesos, declarar variáveis treináveis, fazer o tracing de `__call__` tanto com `training=True` quanto com `training=False`, etc.).

Escolha uma interface de modelo que funcione bem com o fluxo de gradientes, por exemplo, gerar como saída logits em vez de probabilidades softmax ou previsões de top-k.

Se o modelo usasr dropout, regularização de lote ou técnicas de treinamento similares que envolvam hiperparâmetros, defina-os como valores que façam sentido para os diversos problemas e tamanhos de lote esperados (no momento da escrita deste documento, salvar usando o Keras não facilita o ajuste pelos consumidores).

Os reguladores de pesos para camadas individuais são salvos (com seus coeficientes de força de regularização), mas a regularização de pesos dentro do otimizador (como `tf.keras.optimizers.Ftrl.l1_regularization_strength=...)`) é perdida. Oriente os consumidores do seu SavedModel adequadamente.
