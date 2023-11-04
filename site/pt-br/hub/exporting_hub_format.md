# Como exportar modelos no formato TF1 Hub

Saiba mais sobre esse formato em [formato TF1 Hub](tf1_hub_module.md).

## Aviso de compatibilidade

O formato TF1 Hub é adequado para o TensorFlow 1. O TF Hub no TensorFlow 2 tem apenas suporte limitado a esse formato. Considere publicar no novo formato [SavedModel do TF2](tf2_saved_model.md), o que é explicado no guia [Exportando um modelo](exporting_tf2_saved_model).

O formato TF1 Hub é similar ao formato SavedModel do TensorFlow 1 em um nível sintático (mesmos nomes de arquivo e mensagens de protocolo), mas é diferente semanticamente para permitir a reutilização de modelos, a composição e o retreinamento (por exemplo, armazenamento diferente de inicializadores de recursos e convenções de etiquetas diferentes para metagrafos). A maneira mais fácil de diferenciá-los no disco é a presença ou ausência do arquivo `tfhub_module.pb`.

## Estratégia geral

Para definir um novo módulo, um publicador chama `hub.create_module_spec()` com uma função `module_fn`. Essa função constrói um grafo que representa a estrutura interna do modelo usando `tf.placeholder()` para as entradas a serem fornecidas pelo chamador. Em seguida, define assinaturas chamando `hub.add_signature(name, inputs, outputs)` uma ou mais vezes.

Por exemplo:

```python
def module_fn():
  inputs = tf.placeholder(dtype=tf.float32, shape=[None, 50])
  layer1 = tf.layers.dense(inputs, 200)
  layer2 = tf.layers.dense(layer1, 100)
  outputs = dict(default=layer2, hidden_activations=layer1)
  # Add default signature.
  hub.add_signature(inputs=inputs, outputs=outputs)

...
spec = hub.create_module_spec(module_fn)
```

O resultado de `hub.create_module_spec()` pode ser usado no lugar de um caminho para instanciar um objeto do modelo dentro de um grafo específico do TensorFlow. Nesse caso, não há um checkpoint, e a instância do modelo usará os inicializadores de variável em seu lugar.

Qualquer instância do modelo pode ser serializada no disco por seu método `export(path, session)`. Ao exportar um módulo, sua definição é serializada juntamente com o estado atual de suas variáveis na `session` (sessão) no caminho passado. Isso pode ser usado ao exportar um módulo pela primeira vez, bem como ao exportar um módulo após um ajuste fino.

Por questões de compatibilidade com os Estimators do TensorFlow, `hub.LatestModuleExporter` exporta módulos do checkpoint mais recente da mesma forma que `tf.estimator.LatestExporter` exporta o modelo inteiro usando o último checkpoint.

Os publicadores de modelos devem implementar uma [assinatura comum](common_signatures/index.md) quando possível para que os consumidores possam trocar facilmente os módulos e encontrar o melhor para seu próprio problema.

## Exemplo real

Dê uma olhada em nosso [exportador de modelos de embedding de texto](https://github.com/tensorflow/hub/blob/master/examples/text_embeddings/export.py) para ver um exemplo real de como criar um modelo usando um formato comum de embedding de texto.

## Conselho para publicadores

Para facilitar o ajuste fino para os consumidores, tenha em mente o seguinte:

- O ajuste fino requer regularização. Seu módulo é exportado com a coleção `REGULARIZATION_LOSSES` (perdas de regularização), que é o que coloca sua escolha de `tf.layers.dense(..., kernel_regularizer=...)`, etc. naquilo que o consumidor recebe de `tf.losses.get_regularization_losses()`. Opte por essa forma de definir perda de regularização L1/L2.

- No modelo de publicador, evite definir regularização L1/L2 usando os parâmetros `l1_` e `l2_regularization_strength` de `tf.train.FtrlOptimizer`, `tf.train.ProximalGradientDescentOptimizer` e outros otimizadores proximais. Eles não são exportados junto com o módulo, e definir as forças de regularização globalmente pode não ser apropriado para o consumidor. Exceto para a regularização L1 em modelos amplos (ou seja, lineares esparsos) ou amplos e profundos, deve ser possível usar as perdas de regularização individuais.

- Se você usar dropout, normalização de lote ou outras técnicas de treinamento similares, defina os hiperparâmetros como os valores que fazem sentido para os diversos usos esperados. Talvez seja preciso ajustar a taxa de dropout de acordo com a propensão a overfitting do problema em questão. Na normalização de lote, o momento (também chamado de coeficiente de decaimento) deve ser pequeno o suficiente para permitir o ajuste fino com datasets pequenos e/ou lotes grandes. Para consumidores avançados, considere adicionar uma assinatura que exponha o controle dos hiperparâmetros críticos.
