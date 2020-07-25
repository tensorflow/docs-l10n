# TensorFlow 2 efetivo

Há várias mudanças no TensorFlow 2.0 para tornar os usuários do TensorFlow mais produtivos. O TensorFlow 2.0 remove [APIs redundantes](https://github.com/tensorflow/community/blob/master/rfcs/20180827-api-names.md), deixa as APIs mais consistentes ([RNNs unificadas](https://github.com/tensorflow/community/blob/master/rfcs/20180920-unify-rnn-interface.md), [otimizadores unificados](https://github.com/tensorflow/community/blob/master/rfcs/20181016-optimizer-unification.md)) e se integra melhor ao runtime do Python com a [Eager execution](https://www.tensorflow.org/guide/eager).

Diversas [RFCs](https://github.com/tensorflow/community/pulls?utf8=%E2%9C%93&q=is%3Apr) explicaram as mudanças que resultaram no TensorFlow 2.0. Este guia apresenta uma visão de como deve ser o desenvolvimento no TensorFlow 2.0. Supõe-se que você tenha algum grau de familiaridade com o TensorFlow 1.x.

## Breve resumo das principais mudanças

### Limpeza de APIs

Diversas APIs foram [removidas ou movidas de lugar](https://github.com/tensorflow/community/blob/master/rfcs/20180827-api-names.md) no TF 2.0. Entre as principais mudanças, temos: remoção de `tf.app`, `tf.flags` e `tf.logging` para dar lugar à nova API de código aberto [absl-py](https://github.com/abseil/abseil-py), transferência dos projetos que ficavam em `tf.contrib` e limpeza do namespace principal `tf.*`, através da transferência de funções menos usadas para subpacotes como `tf.math`. Algumas APIs foram substituídas por suas equivalentes na versão 2.0: `tf.summary`, `tf.keras.metrics` e `tf.keras.optimizers`. A maneira mais fácil de aplicar automaticamente essas renomeações é usando o [script de atualização v2](upgrade.md).

### Eager execution

O TensorFlow 1.X requer que os usuários criem manualmente uma [árvore sintática abstrata](https://en.wikipedia.org/wiki/Abstract_syntax_tree) (o grafo) fazendo chamadas à API `tf.*`. Em seguida, requer que os usuários compilem manualmente a árvore sintática abstrata passando um conjunto de tensores de saída e de entrada para uma chamada da função `session.run()`. O TensorFlow 2.0 usa eager execution (como o Python já faz normalmente) e, na versão 2.0, os grafos e as sessões devem parecer detalhes de implementação.

Um subproduto notável da Eager execution é que `tf.control_dependencies()` não é mais necessário, pois todas as linhas de código são executadas sequencialmente (dentro de uma `tf.function`, os códigos com efeitos colaterais são executados na ordem em que foram escritos).

### O fim das variáveis globais

O TensorFlow 1.X dependia muito de namespaces implicitamente globais. Quando você chamava `tf.Variable()`, ele era colocado no grafo padrão e permanecia lá, mesmo se você perdesse a variável do Python que apontava para ele. Depois, você poderia recuperar esse `tf.Variable`, mas somente se soubesse o nome com que foi criado originalmente. Isto era difícil de fazer se você não tivesse controle sobre a criação da variável. Como resultado, houve uma proliferação de mecanismos para tentar ajudar os usuários a encontrar suas variáveis novamente e para os frameworks encontrarem variáveis criadas por usuários: escopos de variáveis, coleções globais, métodos auxiliares, como `tf.get_global_step()` e `tf.global_variables_initializer()`, otimizadores computando gradientes implicitamente sobre todas as variáveis treináveis e assim por diante. O TensorFlow 2.0 elimina todos esses mecanismos ([RFC Variables 2.0](https://github.com/tensorflow/community/pull/11)), dando lugar ao mecanismo padrão: manter o controle de suas variáveis! Se você perder o controle de uma `tf.Variable`, ela é recolhida pelo coletor de lixo.

A necessidade de rastrear variáveis cria algum trabalho adicional para o usuário, mas, com objetos do Keras (veja abaixo), esse trabalho é minimizado.

### Funções, não sessões

Uma chamada a `session.run()` é quase igual a uma chamada de função: você especifica as entradas e a função a ser chamada, e recebe de volta um conjunto de saídas. No TensorFlow 2.0, você pode decorar uma função do Python usando `tf.function()` que irá marcá-la para a compilação JIT, de forma que seja executada como um único grafo pelo TensorFlow ([RFC Functions 2.0](https://github.com/tensorflow/community/pull/20)). Este mecanismo permite que o TensorFlow 2.0 ganhe todas as vantagens do modo de grafo:

- Desempenho: a função pode ser otimizada (poda de nós, fusão de kernels, etc.)
- Portabilidade: a função pode ser exportada/reimportada ([RFC SavedModel 2.0](https://github.com/tensorflow/community/pull/34)), permitindo que os usuários reusem e compartilhem funções modulares do TensorFlow.

```python
# TensorFlow 1.X
outputs = session.run(f(placeholder), feed_dict={placeholder: input})
# TensorFlow 2.0
outputs = f(input)
```

Com o poder de interpor livremente código do Python e do TensorFlow, os usuários podem tirar proveito da expressividade do Python. Mas o TensorFlow portátil executa em contextos que não possuem um interpretador de Python, como em mobile, C++ e JavaScript. Para ajudar os usuários a evitar ter que reescrever seu código ao adicionar `@tf.function`, o [AutoGraph](function.ipynb) converte um subconjunto de estruturas de código Python em equivalentes do TensorFlow:

- `for`/`while` -> `tf.while_loop` (há suporte para `break` e `continue`)
- `if` -> `tf.cond`
- `for _ in dataset` -> `dataset.reduce`

O AutoGraph suporta aninhamentos arbitrários de fluxos de controle, o que possibilita implementar de maneira concisa e com bom desempenho diversos programas de ML complexos, como modelos de sequência, aprendizagem por reforço, loops de treinamento personalizados e mais.

## Recomendações para o TensorFlow 2.0 idiomático

### Refatore seu código em funções menores

Um padrão de uso comum no TensorFlow 1.X era a estratégia "kitchen sink" ("pia de cozinha"), em que a união de todas as possíveis computações era traçada antecipadamente e, em seguida, os tensores selecionados eram avaliados por `session.run()`. No TensorFlow 2.0, os usuários devem refatorar seus códigos em funções menores, que são chamadas conforme necessário. Em geral, não é necessário decorar cada uma dessas funções menores com `tf.function`. Somente use `tf.function` para decorar computações de alto nível, como, por exemplo, uma única etapa do treinamento ou em uma fase de propagação (forward pass) de seu modelo.

### Use camadas e modelos do Keras para gerenciar variáveis

Os modelos e camadas do Keras oferecem as convenientes propriedades `variables` e `trainable_variables`, que reúnem recursivamente todas as variáveis dependentes. Assim, fica mais fácil gerenciar variáveis localmente, no lugar onde estão sendo usadas.

Compare:

```python
def dense(x, W, b):
  return tf.nn.sigmoid(tf.matmul(x, W) + b)

@tf.function
def multilayer_perceptron(x, w0, b0, w1, b1, w2, b2 ...):
  x = dense(x, w0, b0)
  x = dense(x, w1, b1)
  x = dense(x, w2, b2)
  ...

# Você ainda precisa gerenciar w_i e b_i, e suas formas são definidas bem longe do código.
```

com a versão em Keras:

```python
# Cada camada pode ser chamada, com uma assinatura equivalente a linear(x)
layers = [tf.keras.layers.Dense(hidden_size, activation=tf.nn.sigmoid) for _ in range(n)]
perceptron = tf.keras.Sequential(layers)

# layers[3].trainable_variables => retorna [w3, b3]
# perceptron.trainable_variables => retorna [w0, b0, ...]
```

Os modelos/camadas do Keras herdam de `tf.train.Checkpointable` e estão integrados a `@tf.function`, o que permite capturar pontos de verificação diretamente ou exportar SavedModels a partir de objetos do Keras. Você não precisa necessariamente usar a API `.fit()` do Keras para aproveitar essas integrações.

Veja abaixo um exemplo de transferência de aprendizagem que demonstra como o Keras facilita a coleta de um subconjunto de variáveis relevantes. Digamos que você esteja treinando um modelo multicabeças com um tronco compartilhado:

```python
trunk = tf.keras.Sequential([...])
head1 = tf.keras.Sequential([...])
head2 = tf.keras.Sequential([...])

path1 = tf.keras.Sequential([trunk, head1])
path2 = tf.keras.Sequential([trunk, head2])

# Treinar o conjunto de dados principal
for x, y in main_dataset:
  with tf.GradientTape() as tape:
    # training=True somente é necessário se houver camadas com comportamento
    # diferente durante treinamento versus inferência (exemplo, Dropout).
    prediction = path1(x, training=True)
    loss = loss_fn_head1(prediction, y)
  # Otimiza simultaneamente os pesos de trunk (tronco) e head1 (cabeça 1).
  gradients = tape.gradient(loss, path1.trainable_variables)
  optimizer.apply_gradients(zip(gradients, path1.trainable_variables))

# Faz o ajuste fino da segunda cabeça, reusando o tronco
for x, y in small_dataset:
  with tf.GradientTape() as tape:
    # training=True somente é necessário se houver camadas com comportamento
    # diferente durante treinamento versus inferência (exemplo, Dropout).
    prediction = path2(x, training=True)
    loss = loss_fn_head2(prediction, y)
  # Otimiza somente os pesos de head2 (cabeça 2), e não os pesos de trunk (tronco)
  gradients = tape.gradient(loss, head2.trainable_variables)
  optimizer.apply_gradients(zip(gradients, head2.trainable_variables))

# Você pode publicar somente a computação do trunk (tronco) para que outras pessoas possam reusar.
tf.saved_model.save(trunk, output_path)
```

### Combine tf.data.Datasets e @tf.function

Ao iterar sobre dados de treinamento que cabem na memória, fique à vontade para usar os recursos básicos de iteração do Python. Caso contrário, `tf.data.Dataset` é a melhor maneira de transmitir dados de treinamento a partir do disco. Conjuntos de dados são [iteráveis (e não iteradores)](https://docs.python.org/3/glossary.html#term-iterable) e funcionam como qualquer outro iterável do Python no modo Eager. Você pode utilizar totalmente os recursos assíncronos de pré-busca/streaming dos conjuntos de dados ao encapsular seu código em `tf.function()`, que substitui uma iteração do Python pelas operações equivalentes de grafo usando o AutoGraph.

```python
@tf.function
def train(model, dataset, optimizer):
  for x, y in dataset:
    with tf.GradientTape() as tape:
      # training=True somente é necessário se houver camadas com comportamento
      # diferente durante treinamento versus inferência (exemplo, Dropout).
      prediction = model(x, training=True)
      loss = loss_fn(prediction, y)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
```

Se você usar a API `.fit()` do Keras, não precisará se preocupar com a iteração de conjuntos de dados.

```python
model.compile(optimizer=optimizer, loss=loss_fn)
model.fit(dataset)
```

### Aproveite o AutoGraph com fluxo de controle do Python

O AutoGraph fornece uma maneira de converter fluxos de controle dependentes de dados em equivalentes no modo grafo, como `tf.cond` e `tf.while_loop`.

Um caso comum de uso de fluxos de controle dependentes de dados são os modelos sequenciais. `tf.keras.layers.RNN` encapsula uma célula de RNN, permitindo que você desdobre a recorrência de maneira estática ou dinâmica. Para fins de demonstração, você poderia implementar novamente o desdobramento dinâmico da seguinte maneira:

```python
class DynamicRNN(tf.keras.Model):

  def __init__(self, rnn_cell):
    super(DynamicRNN, self).__init__(self)
    self.cell = rnn_cell

  def call(self, input_data):
    # [batch, time, features] -> [time, batch, features]
    input_data = tf.transpose(input_data, [1, 0, 2])
    outputs = tf.TensorArray(tf.float32, input_data.shape[0])
    state = self.cell.zero_state(input_data.shape[1], dtype=tf.float32)
    for i in tf.range(input_data.shape[0]):
      output, state = self.cell(input_data[i], state)
      outputs = outputs.write(i, output)
    return tf.transpose(outputs.stack(), [1, 0, 2]), state
```

Para uma visão mais detalhada dos recursos do AutoGraph, consulte [o guia](./function.ipynb).

### tf.metrics agrega os dados, e tf.summary os registra

Para registrar resumos, use `tf.summary.(scalar|histogram|...)` e redirecione-o para um gravador de dados usando o gerenciador de contexto. (Se você omitir o gerenciador de contexto, nada acontecerá). Diferentemente do TF 1.X, os resumos são emitidos diretamente ao gravador de dados; não existe um operador "merge" separado nem uma chamada separada a `add_summary()`, ou seja, o valor de `step` precisa ser fornecido no momento da chamada.

```python
summary_writer = tf.summary.create_file_writer('/tmp/summaries')
with summary_writer.as_default():
  tf.summary.scalar('loss', 0.1, step=42)
```

Para agregar os dados antes de registrá-los como resumos, use `tf.metrics`. Métricas são stateful: elas acumulam valores e retornam um resultado cumulativo quando você chama `.result()`. Use `.reset_states()` para limpar os valores acumulados.

```python
def train(model, optimizer, dataset, log_freq=10):
  avg_loss = tf.keras.metrics.Mean(name='loss', dtype=tf.float32)
  for images, labels in dataset:
    loss = train_step(model, optimizer, images, labels)
    avg_loss.update_state(loss)
    if tf.equal(optimizer.iterations % log_freq, 0):
      tf.summary.scalar('loss', avg_loss.result(), step=optimizer.iterations)
      avg_loss.reset_states()

def test(model, test_x, test_y, step_num):
  # training=False is only needed if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  loss = loss_fn(model(test_x, training=False), test_y)
  tf.summary.scalar('loss', loss, step=step_num)

train_summary_writer = tf.summary.create_file_writer('/tmp/summaries/train')
test_summary_writer = tf.summary.create_file_writer('/tmp/summaries/test')

with train_summary_writer.as_default():
  train(model, optimizer, dataset)

with test_summary_writer.as_default():
  test(model, test_x, test_y, optimizer.iterations)
```

Visualize os resumos gerados apontando o TensorBoard para o diretório de registros de resumos:

```
tensorboard --logdir /tmp/summaries
```

### Use tf.config.experimental_run_functions_eagerly() ao depurar

No TensorFlow 2.0, a Eager execution permite executar o código passo a passo para inspecionar formas, tipos de dados e valores. Determinadas APIs, como `tf.function`, `tf.keras`, etc., foram concebidas para usar a execução de grafo por questões de desempenho e portabilidade. Ao depurar, use `tf.config.experimental_run_functions_eagerly(True)` para utilizar a Eager execution dentro deste código.

Por exemplo:

```python
@tf.function
def f(x):
  if x > 0:
    import pdb
    pdb.set_trace()
    x = x + 1
  return x

tf.config.experimental_run_functions_eagerly(True)
f(tf.constant(1))
```

```
>>> f()
-> x = x + 1
(Pdb) l
  6  	@tf.function
  7  	def f(x):
  8  	  if x > 0:
  9  	    import pdb
 10  	    pdb.set_trace()
 11  ->	    x = x + 1
 12  	  return x
 13
 14  	tf.config.experimental_run_functions_eagerly(True)
 15  	f(tf.constant(1))
[EOF]
```

Isso também funciona dentro dos modelos do Keras e de outras APIs que têm suporte à Eager execution:

```
class CustomModel(tf.keras.models.Model):

  @tf.function
  def call(self, input_data):
    if tf.reduce_mean(input_data) > 0:
      return input_data
    else:
      import pdb
      pdb.set_trace()
      return input_data // 2


tf.config.experimental_run_functions_eagerly(True)
model = CustomModel()
model(tf.constant([-2, -4]))
```

```
>>> call()
-> return input_data // 2
(Pdb) l
 10  	    if tf.reduce_mean(input_data) > 0:
 11  	      return input_data
 12  	    else:
 13  	      import pdb
 14  	      pdb.set_trace()
 15  ->	      return input_data // 2
 16
 17
 18  	tf.config.experimental_run_functions_eagerly(True)
 19  	model = CustomModel()
 20  	model(tf.constant([-2, -4]))
```
