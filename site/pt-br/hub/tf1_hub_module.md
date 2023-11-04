# Formato TF1 Hub

Em seu lançamento em 2018, o TensorFlow Hub oferecia um único tipo de ativo: formato TF1 Hub para importação em programas do TensorFlow 1.

Esta página explica como usar o formato TF1 Hub no TF1 (ou no modo de compatibilidade com o TF1 do TF2) utilizando a classe `hub.Module` e as APIs associadas (o uso típico é criar um `tf.Graph`, possivelmente dentro de um `Estimator` do TF1, ou combinando um ou mais modelos no formato TF1 Hub com `tf.compat.layers` ou `tf.layers`).

Usuários do TensorFlow 2 (que não estejam usando o modo de compatibilidade com o TF1) devem usar a [nova API com `hub.load()` ou `hub.KerasLayer`](tf2_saved_model.md). A nova API carrega o novo tipo de ativo SavedModel do TF2, mas também tem [suporte limitado para o carregamento do formato TF1 Hub no TF2](migration_tf2.md).

## Usando um modelo no formato TF1 Hub

### Instanciando um modelo no formato TF1 Hub

Um modelo no formato TF1 Hub é importado para um programa do TensorFlow criando-se um objeto `hub.Module` a partir de uma string com sua URL ou caminho no sistema de arquivos, como:

```python
m = hub.Module("path/to/a/module_dir")
```

**Observação:** veja mais informações sobre outros tipos de identificadores válidos [aqui](tf2_saved_model.md#model_handles).

Dessa forma, as variáveis do módulo são adicionadas ao grafo atual do TensorFlow. Ao executar os inicializadores, os valores pré-treinados serão lidos no disco. Da mesma forma, tabelas e outros estados são adicionados ao grafo.

### Como fazer cache de módulos

Ao criar um módulo a partir de uma URL, seu conteúdo é baixado, e é feito cache dele no diretório temporário do sistema local. O local onde os módulos ficam em cache pode ser sobrescrito usando a variável de ambiente `TFHUB_CACHE_DIR`. Confira mais detalhes em [Como fazer cache](caching.md).

### Aplicando um módulo

Após instanciado, um módulo `m` pode ser chamado nenhuma ou mais vezes como uma função do Python de entradas do tensor para saídas do tensor:

```python
y = m(x)
```

Cada chamada como essa adiciona operações ao grafo atual do TensorFlow para computar `y` a partir de `x`. Se isso envolver variáveis com pesos treinados, eles são compartilhados entre todas as aplicações.

Os módulos podem definir diversas *assinaturas* com nome para permitir a aplicação de mais de uma forma (similar a como objetos do Python têm *métodos*). A documentação do modelo deve descrever as assinaturas disponíveis. A chamada acima aplica a assinatura chamada `"default"` (padrão). Qualquer assinatura pode ser selecionada passando-se seu nome ao argumento opcional `signature=`.

Se uma assinatura tiver várias entradas, elas devem ser passadas como um dicionário, com as chaves definidas pela assinatura. Igualmente, se uma assinatura tiver várias saídas, elas podem ser recuperadas como um dicionário passando-se `as_dict=True`, de acordo com as chaves definidas pela assinatura (a chave `"default"` é para a única saída retornada se `as_dict=False`). Portanto, a forma mais geral de aplicar um modelo é:

```python
outputs = m(dict(apples=x1, oranges=x2), signature="fruit_to_pet", as_dict=True)
y1 = outputs["cats"]
y2 = outputs["dogs"]
```

Um chamador precisa fornecer todas as entradas definidas por uma assinatura, mas não há exigência de usar todas as saídas de um módulo. O TensorFlow vai executar somente as partes do módulo que acabam como dependências de um alvo em `tf.Session.run()`. De fato, a maioria dos publicadores podem optar por fornecer diversas saídas para usuários avançados (como ativações ou camadas intermediárias) junto com as saídas principais. Os consumidores do módulo devem tratar as saídas adicionais de forma elegante.

### Experimentando módulos alternativos

Sempre que houver diversos módulos para a mesma tarefa, o TensorFlow Hub aconselha fornecer a eles assinaturas (interfaces) compatíveis de tal forma que experimentar assinaturas diferentes seja tão fácil quanto variar o identificador do módulo como um hiperparâmetro cujo valor é uma string.

Para esse fim, mantemos uma coleção de [assinaturas comuns](common_signatures/index.md) recomendadas para tarefas populares.

## Criando um novo módulo

### Aviso de compatibilidade

O formato TF1 Hub é adequado para o TensorFlow 1 e tem somente suporte limitado pelo TF Hub no TensorFlow 2. Considere publicar no novo formato [SavedModel do TF2](tf2_saved_model.md).

O formato TF1 Hub é similar ao formato SavedModel do TensorFlow 1 em um nível sintático (mesmos nomes de arquivo e mensagens de protocolo), mas é diferente semanticamente para permitir a reutilização de modelos, a composição e o retreinamento (por exemplo, armazenamento diferente de inicializadores de recursos e convenções de etiquetas diferentes para metagrafos). A maneira mais fácil de diferenciá-los no disco é a presença ou ausência do arquivo `tfhub_module.pb`.

### Estratégia geral

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

### Exemplo real

Dê uma olhada em nosso [exportador de modelos de embedding de texto](https://github.com/tensorflow/hub/blob/master/examples/text_embeddings/export.py) para ver um exemplo real de como criar um modelo usando um formato comum de embedding de texto.

## Ajustes finos

Treinar as variáveis de um módulo importado junto com as variáveis do modelo adjacente é chamado de *ajustes finos*. Fazer os ajustes finos pode aumentar a qualidade, mas traz novas complicações. Orientamos que os consumidores façam ajustes finos somente após explorar ajustes de qualidade mais simples e somente se o publicador do módulo recomendar.

### Para consumidores

Para fazer os ajustes finos, instancie o módulo com `hub.Module(..., trainable=True)` para tornar suas variáveis treináveis e importe as `REGULARIZATION_LOSSES` (perdas de regularização) do TensorFlow. Se o módulo tiver várias variantes do grafo, escolha aquele adequado para o treinamento. Geralmente, é aquele com a etiqueta `{"train"}`.

Escolha uma forma de treinamento que não arruíne os pesos pré-treinados. Por exemplo, uma taxa de aprendizado mais baixa do que ao fazer o treinamento do zero.

### Para publicadores

Para facilitar o ajuste fino para os consumidores, tenha em mente o seguinte:

- O ajuste fino requer regularização. Seu módulo é exportado com a coleção `REGULARIZATION_LOSSES` (perdas de regularização), que é o que coloca sua escolha de `tf.layers.dense(..., kernel_regularizer=...)`, etc. naquilo que o consumidor recebe de `tf.losses.get_regularization_losses()`. Opte por essa forma de definir perda de regularização L1/L2.

- No modelo de publicador, evite definir regularização L1/L2 usando os parâmetros `l1_` e `l2_regularization_strength` de `tf.train.FtrlOptimizer`, `tf.train.ProximalGradientDescentOptimizer` e outros otimizadores proximais. Eles não são exportados junto com o módulo, e definir as forças de regularização globalmente pode não ser apropriado para o consumidor. Exceto para a regularização L1 em modelos amplos (ou seja, lineares esparsos) ou amplos e profundos, deve ser possível usar as perdas de regularização individuais.

- Se você usar dropout, normalização de lote ou outras técnicas de treinamento similares, defina os hiperparâmetros como os valores que fazem sentido para os diversos usos esperados. Talvez seja preciso ajustar a taxa de dropout de acordo com a propensão a overfitting do problema em questão. Na normalização de lote, o momento (também chamado de coeficiente de decaimento) deve ser pequeno o suficiente para permitir o ajuste fino com datasets pequenos e/ou lotes grandes. Para consumidores avançados, considere adicionar uma assinatura que exponha o controle dos hiperparâmetros críticos.
