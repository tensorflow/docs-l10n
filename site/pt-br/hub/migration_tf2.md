# Migrando do TF1 para o TF2 com o TensorFlow Hub

Esta página explica como continuar usando o TensorFlow Hub ao migrar seu código do TensorFlow 1 para o TensorFlow 2. Ela complementa o [guia geral de migração](https://www.tensorflow.org/guide/migrate) do TensorFlow.

Para o TF2, TF Hub parou de usar a API legada `hub.Module` para criar um `tf.compat.v1.Graph` como as `tf.contrib.v1.layers` fazem. Em vez disso, agora existe uma `hub.KerasLayer` para uso juntamente com outras camadas do Keras para criar um `tf.keras.Model` (geralmente no novo [ambiente de execução adiantada (eager)](https://www.tensorflow.org/api_docs/python/tf/executing_eagerly)) do TF2 e seu método subjacente `hub.load()` para código de baixo nível do TensorFlow.

A API `hub.Module` permanece disponível na biblioteca `tensorflow_hub` para uso no TF1 e no modo de compatibilidade com o TF1 do TF2. Ela só pode carregar modelos no [formato TF1 Hub](tf1_hub_module.md).

A nova API de `hub.load()` e `hub.KerasLayer` funciona para o TensorFlow 1.15 (nos modos eager e grafo) e no TensorFlow 2. Essa nova API pode carregar os novos ativos de [SavedModel do TF2](tf2_saved_model.md) e, com as restrições descritas no [guia de compatibilidade de modelos](model_compatibility.md), os modelos legados no formato TF1 Hub.

De forma geral, recomenda-se usar a nova API sempre que possível.

## Resumo da nova API

`hub.load()` é a nova função de baixo nível para carregar um SavedModel do TensorFlow Hub (ou serviços compatíveis). Ela encapsula o `tf.saved_model.load()` do TF2; o [Guia sobre SavedModel](https://www.tensorflow.org/guide/saved_model) do TensorFlow descreve o que você pode fazer com o resultado.

```python
m = hub.load(handle)
outputs = m(inputs)
```

A classe `hub.KerasLayer` chama `hub.load()` e adapta o resultado para uso no Keras juntamente com outras camadas do Keras (ela pode até mesmo ser um encapsulador conveniente para SavedModels carregados usados de várias formas).

```python
model = tf.keras.Sequential([
    hub.KerasLayer(handle),
    ...])
```

Diversos tutoriais mostram o uso dessas APIs. Confira especificamente:

- [Notebook de exemplo de classificação de texto](https://github.com/tensorflow/docs/blob/master/g3doc/en/hub/tutorials/tf2_text_classification.ipynb)
- [Notebook de exemplo de classificação de imagens](https://github.com/tensorflow/docs/blob/master/g3doc/en/hub/tutorials/tf2_image_retraining.ipynb)

### Usando a nova API no treinamento de Estimators

Se você usar um SavedModel do TF2 em um Estimator para treinamento com servidores de parâmetros (ou em uma sessão do TF1 com variáveis colocadas em dispositivos remotos), precisa definir `experimental.share_cluster_devices_in_session` no ConfigProto de tf.Session, ou então gerá gerado um erro do tipo "Assigned device '/job:ps/replica:0/task:0/device:CPU:0' does not match any device" (O dispositivo atribuído '/job:ps/replica:0/task:0/device:CPU:0' não coincide com nenhum dispositivo).

A opção necessária pode ser definida desta forma:

```python
session_config = tf.compat.v1.ConfigProto()
session_config.experimental.share_cluster_devices_in_session = True
run_config = tf.estimator.RunConfig(..., session_config=session_config)
estimator = tf.estimator.Estimator(..., config=run_config)
```

A partir do TF2.2, esta opção não é mais experimental, e a parte `.experimental` pode ser retirada.

## Carregando modelos legados no formato TF1 Hub

Há casos em que um novo SavedModel do TF2 ainda não esteja disponível para seu caso de uso e você precisa carregar um modelo legado com o formato TF1 Hub. A partir da versão 0.7 de `tensorflow_hub`, você pode usar um modelo legado com o formato TF1 Hub junto com `hub.KerasLayer` confirme exibido abaixo:

```python
m = hub.KerasLayer(handle)
tensor_out = m(tensor_in)
```

Além disso, `KerasLayer` expõe a funcionalidade de especificar `tags` (etiquetas), `signature` (assinatura), `output_key` (chave de saída) e `signature_outputs_as_dict` (saídas de assinatura como dicionário) para usos mais específicos dos modelos legados com o formato TF1 Hub e SavedModels legados.

Confira mais informações sobre a compatibilidade com o formato TF1 Hub no [guia de compatibilidade de modelo](model_compatibility.md).

## Usando APIs de nível mais baixo

Os modelos com o formato legado TF1 Hub podem ser carregados via `tf.saved_model.load` em vez de:

```python
# DEPRECATED: TensorFlow 1
m = hub.Module(handle, tags={"foo", "bar"})
tensors_out_dict = m(dict(x1=..., x2=...), signature="sig", as_dict=True)
```

Recomenda-se usar:

```python
# TensorFlow 2
m = hub.load(path, tags={"foo", "bar"})
tensors_out_dict = m.signatures["sig"](x1=..., x2=...)
```

Nesse exemplos, `m.signatures` é um dicionário (dict) de [funções concretas](https://www.tensorflow.org/tutorials/customization/performance#tracing) do TensorFlow com nomes de assinatura como chave. Ao chamar uma função como essa, todas as saídas são computadas, mesmo se não usadas [isso é diferente da avaliação lazy (postergada) do modo grafo do TF1].
