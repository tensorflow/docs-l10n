# Como exportar um SavedModel

Esta página apresenta os detalhes de como exportar (salvar) um modelo de um programa do TensorFlow para o [formato SavedModel do TensorFlow 2](https://www.tensorflow.org/guide/saved_model). Esse formato é a forma recomendada de compartilhar modelos e partes de modelos pré-treinados no TensorFlow Hub. Ele substitui o [formato TF1 Hub](tf1_hub_module.md) antigo e conta com um novo conjunto de APIs. Confira mais informações sobre como exportar modelos para o formato TF1 Hub no guia [Exportar para o formato TF1 Hub](exporting_hub_format.md). Confira mais detalhes de como compactar o SavedModel para compartilhá-lo no TensorFlow Hub [aqui](writing_documentation.md#model-specific_asset_content).

Alguns kits de ferramentas de criação de modelos já contam com ferramentas para fazer isso (confira abaixo o caso [TensorFlow Model Garden](#tensorflow-model-garden)).

## Visão geral

O SavedModel é o formato de serialização padrão do TensorFlow para modelos ou partes de modelos treinados. Ele armazena os pesos treinados do modelo juntamente com as operações exatas do TensorFlow para realizar a sua computação. Pode ser usado independentemente do código que o criou. Especificamente, pode ser reutilizado em diferentes APIs de construção de modelos de alto nível, como o Keras, porque as operações do TensorFlow são sua linguagem básica comum.

## Como salvar usando o Keras

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

## Como salvar usando o TensorFlow de baixo nível

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

## Conselho para criadores de SavedModel

Ao criar um SavedModel para compartilhar no TensorFlow Hub, pense com antecedência se e como os consumidores devem fazer o ajuste fino, e forneça orientações na documentação.

Ao salvar um modelo do Keras, todas as mecânicas de ajuste fino devem funcionar (salvar perdas de regularização de pesos, declarar variáveis treináveis, fazer o tracing de `__call__` tanto com `training=True` quanto com `training=False`, etc.).

Escolha uma interface de modelo que funcione bem com o fluxo de gradientes, por exemplo, gerar como saída logits em vez de probabilidades softmax ou previsões de top-k.

Se o modelo usar dropout, regularização de lote ou técnicas de treinamento similares que envolvam hiperparâmetros, defina-os como valores que façam sentido para os diversos problemas e tamanhos de lote esperados (no momento da escrita deste documento, salvar usando o Keras não facilita o ajuste pelos consumidores).

Os reguladores de pesos para camadas individuais são salvos (com seus coeficientes de força de regularização), mas a regularização de pesos dentro do otimizador (como `tf.keras.optimizers.Ftrl.l1_regularization_strength=...)`) é perdida. Oriente os consumidores do seu SavedModel adequadamente.

<a name="tensorflow-model-garden"></a>

## TensorFlow Model Garden

O repositório [TensorFlow Model Garden](https://github.com/tensorflow/models/tree/master/research/official) contém diversos exemplos da criação de SavedModelos do TF2 reutilizáveis a serem carregados em [tfhub.dev](https://tfhub.dev/).

## Solicitações da comunidade

A equipe do TensorFlow Hub gera somente uma pequena fração dos ativos disponíveis em tfhub.dev. Contamos com os pesquisadores do Google e DeepMind, instituições de pesquisa corporativas e acadêmicas, além de entusiastas de aprendizado de máquina para criar modelos. Dessa forma, não podemos garantir o atendimento a solicitações de ativos específicos feitas pela comunidade e não podemos fornecer estimativas de tempo para a disponibilização de novos ativos.

O [marco de solicitações de modelos feitas pela comunidade](https://github.com/tensorflow/hub/milestone/1) contém solicitações de ativos específicos feitas pela comunidade. Se você ou alguém que você conheça tenha interesse em criar algum ativo e compartilhá-lo em tfhub.dev, agradecemos o envio!
