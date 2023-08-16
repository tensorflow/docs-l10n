# Visão geral da migração TF1.x -&gt; TF2

O TensorFlow 2 é fundamentalmente diferente do TF1.x de várias maneiras. Você ainda pode executar código TF1.x não modificado ([exceto para contrib](https://github.com/tensorflow/community/blob/master/rfcs/20180907-contrib-sunset.md)) em instalações binárias TF2 da seguinte forma:

```python
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
```

No entanto, isto *não* executa APIs nem comportamentos do TF2 e pode não funcionar como esperado com código escrito para o TF2. Se você não estiver executando com comportamentos ativos do TF2, estará na prática executando o TF1.x sobre uma instalação TF2. Leia o [Guia de comportamentos TF1 vs TF2](./tf1_vs_tf2.ipynb) para obter mais sobre como TF2 difere do TF1.x.

Este guia fornece uma visão geral do processo para migrar seu código de TF1.x para TF2. Isto vai permitir que você aproveite as melhorias dos recursos novos e futuros e também vai deixar seu código mais simples, com melhor desempenho e mais fácil de manter.

Se você estiver usando as APIs de alto nível `tf.keras` e treinando exclusivamente com `model.fit`, seu código deve ser mais ou menos compatível com TF2, exceto pelas seguintes ressalvas:

- O TF2 tem novas [taxas de aprendizado padrão](../../guide/effective_tf2.ipynb#optimizer_defaults) para otimizadores Keras.
- O TF2 [pode ter alterado](../../guide/effective_tf2.ipynb#keras_metric_names) o "nome" onde as métricas são registradas em log.

## Processo de migração do TF2

Antes de migrar, aprenda sobre o comportamento e as diferenças de API entre TF1.x e TF2 lendo o [guia](./tf1_vs_tf2.ipynb).

1. Execute o script automatizado para converter parte do uso da sua API TF1.x para `tf.compat.v1`.
2. Remova símbolos `tf.contrib` antigos (verifique [TF Addons](https://github.com/tensorflow/addons) e [TF-Slim](https://github.com/google-research/tf-slim)).
3. Faça com que os passos para frente do modelo TF1.x sejam executados no TF2 com a execução antecipada (eager) ativada.
4. Faça upgrade do seu código TF1.x para loops de treinamento e salvamento/carga de modelos para seus equivalentes no TF2.
5. (Opcional) Migre suas APIs `tf.compat.v1`, compatíveis com TF2, para APIs TF2 idiomáticas.

As seções a seguir expandem as etapas descritas acima.

## Execute o script de conversão de símbolos

Isto executa um passo inicial para reescrever seus símbolos de código para executar em binários TF 2.x, mas não vai deixar seu código idiomático para o TF 2.x nem automaticamente tornará seu código compatível com os comportamentos do TF2.

Seu código provavelmente ainda fará uso de endpoints `tf.compat.v1` para acessar placeholders, sessões, coleções e outras funcionalidades do estilo TF1.x.

Leia o [guia](./upgrade.ipynb) para saber mais sobre as práticas recomendadas para usar o script de conversão de símbolos.

## Remova o uso de `tf.contrib`

O módulo `tf.contrib` foi desativado e vários de seus submódulos foram integrados ao core da API TF2. Os outros submódulos agora são derivados de outros projetos como [TF IO](https://github.com/tensorflow/io) e [TF Addons](https://www.tensorflow.org/addons/overview).

Uma grande quantidade de código TF1.x mais antigo usa a biblioteca [Slim](https://ai.googleblog.com/2016/08/tf-slim-high-level-library-to-define.html), que foi empacotada com TF1.x como `tf.contrib.layers`. Ao migrar seu código Slim para TF2, mude seus usos da API Slim para apontar para o [pacote pip tf-slim](https://pypi.org/project/tf-slim/). Em seguida, leia o [Guia de mapeamento de modelos](https://tensorflow.org/guide/migrate/model_mapping#a_note_on_slim_and_contriblayers) para saber como converter o código Slim.

Como alternativa, se você usar modelos pré-treinados do Slim, considere experimentar os modelos pré-treinados do Keras em `tf.keras.applications` ou o TF2 <code>SavedModel</code> do <a>TF Hub</a>, exportado do código Slim original.

## Faça com que os passos para frente do modelo TF1.x sejam executados com os comportamentos TF2 ativados

### Rastreie variáveis ​​e perdas

[O TF2 não oferece suporte a coleções globais.](./tf1_vs_tf2.ipynb#no_more_globals)

A execução antecipada (eager) no TF2 não oferece suporte a APIs baseadas em coleções `tf.Graph`. Isto afeta como você constrói e controla as variáveis.

Para o novo código TF2, você usaria `tf.Variable` em vez de `v1.get_variable` e usaria objetos Python para coletar e rastrear variáveis ​​em vez de `tf.compat.v1.variable_scope`. Normalmente, isto seria um dos seguintes:

- `tf.keras.layers.Layer`
- `tf.keras.Model`
- `tf.Module`

Agregue listas de variáveis ​​(como `tf.Graph.get_collection(tf.GraphKeys.VARIABLES)`) com os atributos `.variables` e `.trainable_variables` dos objetos `Layer`, `Module` ou `Model`.

As classes `Layer` e `Model` implementam várias outras propriedades que eliminam a necessidade de coleções globais. A propriedade `.losses` pode substituir o uso da coleção `tf.GraphKeys.LOSSES`.

Leia o [guia de mapeamento de modelos](./model_mapping.ipynb) para saber mais sobre como usar os shims de modelagem de código TF2 para incorporar seu código existente baseado em `get_variable` e `variable_scope` dentro de `Layers`, `Models` e `Modules`. Isso permitirá que você execute passes avançados com execução antecipada (eager) ativada sem precisar fazer grandes alterações.

### Adaptação a outras mudanças de comportamento

Se o [guia de mapeamento de modelos](./model_mapping.ipynb) por si só for insuficiente para que seu modelo passe adiante executando outras mudanças de comportamento que podem ser mais detalhadas, consulte o guia sobre [comportamentos TF1.x vs TF2](./tf1_vs_tf2.ipynb) para aprender sobre as outras mudanças de comportamento e como você pode se adaptar a elas. Verifique também o [guia de criação de novas camadas e modelos por meio de subclasses](https://tensorflow.org/guide/keras/custom_layers_and_models.ipynb) para mais detalhes.

### Validando seus resultados

Consulte o [guia de validação de modelos](./validate_correctness.ipynb) para obter ferramentas fáceis de usar e orientação sobre como você pode (numericamente) validar se seu modelo está se comportando corretamente quando a execução antecipada (eager) é ativada. Você pode achar isso especialmente útil quando combinado com o [guia de mapeamento de modelos](./model_mapping.ipynb).

## Atualize o treinamento, avaliação e código de importação/exportação

Os loops de treinamento do TF1.x criados com objetos `tf.estimator.Estimator` em estilo `v1.Session` e outras abordagens baseadas em coleções, não são compatíveis com os novos comportamentos do TF2. É importante que você migre todo o seu código de treinamento TF1.x, pois combiná-lo com o código TF2 pode causar comportamentos inesperados.

Você pode escolher dentre várias estratégias.

A abordagem de nível mais alto é usar `tf.keras`. As funções de alto nível em Keras gerenciam muitos detalhes de baixo nível que podem ser fáceis de perder se você escrever seu próprio loop de treinamento. Por exemplo, eles coletam automaticamente as perdas de regularização e definem o argumento `training=True` ao chamar o modelo.

Consulte o [Guia de migração do Estimator](./migrating_estimator.ipynb) para saber como migrar o código `tf.estimator.Estimator` para usar os loops de treinamento [vanilla](./migrating_estimator.ipynb#tf2_keras_training_api) e <code>tf.keras</code> <a>personalizados</a>.

Os loops de treinamento personalizados oferecem um controle mais preciso sobre seu modelo, como o rastreamento dos pesos de camadas individuais. Leia o guia sobre como [criar loops de treinamento do zero](https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch) para aprender a usar `tf.GradientTape` para recuperar pesos de modelo e usá-los para atualizar o modelo.

### Converta otimizadores TF1.x em otimizadores Keras

Os otimizadores em `tf.compat.v1.train`, como o [otimizador Adam](https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/AdamOptimizer) e o [otimizador de método do gradiente descendente](https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/GradientDescentOptimizer), têm equivalentes em `tf.keras.optimizers`.

A tabela abaixo resume como você pode converter esses otimizadores herdados aos seus equivalentes Keras. Você pode substituir diretamente a versão TF1.x pela versão TF2, a menos que passos adicionais (como [atualizar a taxa de aprendizado padrão](../../guide/effective_tf2.ipynb#optimizer_defaults)) sejam obrigatórios.

Observe que a conversão de seus otimizadores [pode deixar os checkpoints antigos incompatíveis](./migrating_checkpoints.ipynb).

<table>
  <tr>
    <th>TF1.x</th>
    <th>TF2</th>
    <th>Passos adicionais</th>
  </tr>
  <tr>
    <td>`tf.v1.train.GradientDescentOptimizer`</td>
    <td>`tf.keras.optimizers.SGD`</td>
    <td>Nenhum</td>
  </tr>
  <tr>
    <td>`tf.v1.train.MomentumOptimizer`</td>
    <td>`tf.keras.optimizers.SGD`</td>
    <td>Inclua o argumento `momentum`</td>
  </tr>
  <tr>
    <td>`tf.v1.train.AdamOptimizer`</td>
    <td>`tf.keras.optimizers.Adam`</td>
    <td>Renomeie os argumentos `beta1` e `beta2` para `beta_1` e `beta_2`</td>
  </tr>
  <tr>
    <td>`tf.v1.train.RMSPropOptimizer`</td>
    <td>`tf.keras.optimizers.RMSprop`</td>
    <td>Renomeie o argumento `decay` para `rho`</td>
  </tr>
  <tr>
    <td>`tf.v1.train.AdadeltaOptimizer`</td>
    <td>`tf.keras.optimizers.Adadelta`</td>
    <td>Nenhum</td>
  </tr>
  <tr>
    <td>`tf.v1.train.AdagradOptimizer`</td>
    <td>`tf.keras.optimizers.Adagrad`</td>
    <td>Nenhum</td>
  </tr>
  <tr>
    <td>`tf.v1.train.FtrlOptimizer`</td>
    <td>`tf.keras.optimizers.Ftrl`</td>
    <td>Remova os argumentos `accum_name` e `linear_name`</td>
  </tr>
  <tr>
    <td>`tf.contrib.AdamaxOptimizer`</td>
    <td>`tf.keras.optimizers.Adamax`</td>
    <td>Renomeie os argumentos `beta1` e `beta2` para `beta_1` e `beta_2`</td>
  </tr>
  <tr>
    <td>`tf.contrib.Nadam`</td>
    <td>`tf.keras.optimizers.Nadam`</td>
    <td>Renomeie os argumentos `beta1` e `beta2` para `beta_1` e `beta_2`</td>
  </tr>
</table>

Observação: No TF2, todos os epsilons (constantes de estabilidade numérica) agora são padronizados para `1e-7` em vez de `1e-8`. Essa diferença é insignificante na maioria dos casos de uso.

### Faça upgrade dos pipelines de entrada de dados

Há muitas maneiras de alimentar dados para um modelo `tf.keras`. Eles aceitarão geradores Python e matrizes Numpy como entrada.

A maneira recomendada de alimentar um modelo com dados é usar o pacote `tf.data`, que contém uma coleção de classes de alto desempenho para manipulação de dados. Os `dataset` pertencentes a `tf.data` são eficientes, expressivos e integram-se bem com o TF2.

Eles podem ser passados ​​diretamente para o método `tf.keras.Model.fit`.

```python
model.fit(dataset, epochs=5)
```

Você pode iterar sobre eles diretamente no Python padrão:

```python
for example_batch, label_batch in dataset:
    break
```

Se você ainda estiver usando `tf.queue`, eles agora são suportados apenas como estruturas de dados, não como pipelines de entrada.

Você também deve migrar todo o código de pré-processamento de características que usa `tf.feature_columns`. Leia o [guia de migração](./migrating_feature_columns.ipynb) para mais detalhes.

### Salvando e carregando modelos

O TF2 usa checkpoints baseados em objeto. Leia o [guia de migração de checkpoints](./migrating_checkpoints.ipynb) para saber mais sobre a migração de checkpoints TF1.x baseados em nomes. Leia também o [guia de checkpoints](https://www.tensorflow.org/guide/checkpoint) nos documentos principais do TensorFlow.

Não há questões significativas de compatibilidade para os modelos salvos. Leia o [guia `SavedModel`](./saved_model.ipynb) para mais informações sobre como migrar `SavedModel`s do TF1.x para TF2. Em geral,

- Os saved_models do TF1.x funcionam no TF2.
- Os saved_models do TF2 funcionam no TF1.x se todas as operações forem suportadas.

Consulte também a [seção `GraphDef`](./saved_model.ipynb#graphdef_and_metagraphdef) no guia de migração `SavedModel` para mais informações sobre como trabalhar com objetos `Graph.pb` e `Graph.pbtxt`.

## (Opcional) Migre símbolos `tf.compat.v1`

O módulo `tf.compat.v1` contém a API TF1.x completa, com sua semântica original.

Mesmo depois de seguir as etapas acima e terminar com um código totalmente compatível com todos os comportamentos do TF2, é provável que ainda haja muitas menções a APIs `compat.v1` que acabam sendo compatíveis com o TF2. Você deve evitar usar essas APIs `compat.v1` legadas para qualquer novo código que escrever, embora elas continuem funcionando para seu código já escrito.

No entanto, você pode optar por migrar os usos existentes para as novas APIs do TF2. As docstrings de símbolos `compat.v1` individuais geralmente explicam como migrá-los para as novas APIs do TF2. Além disso, a [seção do guia de mapeamento de modelos sobre migração incremental para APIs TF2 idiomáticas](./model_mapping.ipynb#incremental_migration_to_native_tf2) também pode ser útil.

## Recursos e leitura adicional

Conforme mencionado anteriormente, é uma boa prática migrar todo o seu código TF1.x para TF2. Leia os guias na seção [Como migrar para o TF2](https://tensorflow.org/guide/migrate) do guia do TensorFlow para saber mais.
