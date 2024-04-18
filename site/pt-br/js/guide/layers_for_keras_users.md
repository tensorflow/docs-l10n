# API Layers do TensorFlow.js para usuários do Keras

A API Layers do TensorFlow.js é modelada de acordo com o Keras, e nos esforçamos para deixar a [API Layers](https://js.tensorflow.org/api/latest/#Layers) o mais similar possível ao Keras, dadas as diferenças entre o JavaScript e o Python. Dessa forma, os usuários com experiência em desenvolvimento de modelos do Keras no Python têm mais facilidade de migrar para TensorFlow.js Layers no JavaScript. Por exemplo, o seguinte código do Keras é traduzido em JavaScript:

```python
# Python:
import keras
import numpy as np

# Cria e compila modelo.
model = keras.Sequential()
model.add(keras.layers.Dense(units=1, input_shape=[1]))
model.compile(optimizer='sgd', loss='mean_squared_error')

# Gera alguns dados sintéticos para o treinamento.
xs = np.array([[1], [2], [3], [4]])
ys = np.array([[1], [3], [5], [7]])

# Treina modelo com fit().
model.fit(xs, ys, epochs=1000)

# Executa inferência com predict().
print(model.predict(np.array([[5]])))
```

```js
// JavaScript:
import * as tf from '@tensorflow/tfjs';

// Build and compile model.
const model = tf.sequential();
model.add(tf.layers.dense({units: 1, inputShape: [1]}));
model.compile({optimizer: 'sgd', loss: 'meanSquaredError'});

// Generate some synthetic data for training.
const xs = tf.tensor2d([[1], [2], [3], [4]], [4, 1]);
const ys = tf.tensor2d([[1], [3], [5], [7]], [4, 1]);

// Train model with fit().
await model.fit(xs, ys, {epochs: 1000});

// Run inference with predict().
model.predict(tf.tensor2d([[5]], [1, 1])).print();
```

Entretanto, gostaríamos de destacar e explicar algumas diferenças neste documento. Quando você entender essas diferenças e a motivação por trás delas, a migração de Python para JavaScript (ou vice-versa) deverá ser relativamente simples.

## Construtores recebem objetos do JavaScript como configurações

Compare as seguintes linhas do Python e JavaScript com o exemplo acima – ambas criam uma camada [Dense](https://keras.io/api/layers/core_layers/dense).

```python
# Python:
keras.layers.Dense(units=1, inputShape=[1])
```

```js
// JavaScript:
tf.layers.dense({units: 1, inputShape: [1]});
```

As funções do JavaScript não têm um equivalente aos argumentos palavra-chave nas funções do Python. Queremos evitar a implementação de opções de construtores como argumentos posicionais no JavaScript, o que seria especialmente complicado de se lembrar e usar para construtores com um grande número de argumentos palavra-chave (como [LSTM](https://keras.io/api/layers/recurrent_layers/lstm)). É por isso que usamos objetos de configuração do JavaScript, que oferecem o mesmo nível de invariância posicional e flexibilidade que os argumentos palavra-chave do Python.

Alguns métodos da classe Model, como [`Model.compile()`](https://keras.io/models/model/#model-class-api), também recebem um objeto de configuração do JavaScript como entrada. Entretanto, lembre-se de que `Model.fit()`, `Model.evaluate()` e `Model.predict()` são ligeiramente diferentes. Como esses métodos recebem dados obrigatórios `x` (características) e `y` (rótulos ou alvos) como entrada; `x` e `y` são argumentos posicionais separados do objeto de configuração subsequente que cumpre o papel de argumentos palavra-chave. Por exemplo:

```js
// JavaScript:
await model.fit(xs, ys, {epochs: 1000});
```

## Model.fit() é assíncrono

`Model.fit()` é o principal método usado pelos usuários para fazer o treinamento de modelos no TensorFlow.js. Esse método costuma ter execução longa, podendo durar segundos ou minutos. Portanto, utilizamos o recurso `async` da linguagem JavaScript para que essa função possa ser usada de uma forma que não bloqueie o thread principal de interface gráfica ao executar no navegador. Isso é semelhante a outras funções potencialmente de execução longa no JavaScript, como a `async` [fetch](https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API). Observe que `async` é um constructo que não existe no Python. Enquanto o método [`fit()`](https://keras.io/models/model/#model-class-api) no Keras retorna um objeto History, a contrapartida do método `fit()` no JavaScript retorna uma [Promise](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Promise) do History, que pode ser tratada com [await](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Operators/await) (como no exemplo acima) ou usando o método then().

## Sem NumPy para o TensorFlow.js

Os usuários do Keras (Python) usam [NumPy](http://www.numpy.org/) para realizar operações numéricas e com arrays básicas, como a geração de tensores bidimensionais no exemplo acima.

```python
# Python:
xs = np.array([[1], [2], [3], [4]])
```

No TensorFlow.js, esse tipo de operação numérica básica é feito com o próprio pacote. Por exemplo:

```js
// JavaScript:
const xs = tf.tensor2d([[1], [2], [3], [4]], [4, 1]);
```

O namespace `tf.*` também conta com diversas outras funções para operações com arrays e de álgebra linear, como multiplicação de matrizes. Confira mais informações na [documentação do TensorFlow.js Core](https://js.tensorflow.org/api/latest/).

## Use métodos de fábrica, não construtores

Esta linha no Python (do exemplo acima) é uma chamada ao construtor:

```python
# Python:
model = keras.Sequential()
```

Se traduzida estritamente para JavaScript, a chamada ao construtor equivalente seria assim:

```js
// JavaScript:
const model = new tf.Sequential();  // !!! NÃO FAÇA ISSO !!!
```

Entretanto, decidimos não criar objetos usando construtores com "new" porque: 1) a palavra-chave “new” deixaria o código mais confuso e 2) a criação de objetos com "new" é considerada uma “parte ruim” do JavaScript: uma possível armadilha, como discutido em [*JavaScript: the Good Parts*](https://www.oreilly.com/library/view/javascript-the-good/9780596517748/) (O melhor do JavaScript). Para criar modelos e camadas no TensorFlow.js, você deve chamar os métodos de fábrica, com nomes em lowerCamelCase, como:

```js
// JavaScript:
const model = tf.sequential();

const layer = tf.layers.batchNormalization({axis: 1});
```

## Valores de strings de opções usam o formato lowerCamelCase, não snake_case

No JavaScript, é mais comum usar Camel Case para nomes de símbolos (confira o [Guia de estilo de JavaScript do Google](https://google.github.io/styleguide/jsguide.html#naming-camel-case-defined)), em comparação ao Python, em que o Snake Case é comum (como no Keras). Portanto, decidimos usar lowerCamelCase para valores de strings, incluindo os seguintes:

- DataFormat, como **`channelsFirst`** em vez `channels_first`
- Inicializador, como **`glorotNormal`** em vez de `glorot_normal`
- Perda e métricas, como **`meanSquaredError`** em vez de `mean_squared_error` e **`categoricalCrossentropy`** em vez de `categorical_crossentropy`.

Por exemplo, como no exemplo acima:

```js
// JavaScript:
model.compile({optimizer: 'sgd', loss: 'meanSquaredError'});
```

Em relação à serialização e desserialização de modelos, não se preocupe. O mecanismo interno do TensorFlow.js garante que o Snake Case em objetos JSON seja tratado corretamente, como ao carregar modelos pré-treinados a partir do Keras no Python.

## Execute objetos Layer com apply(), não chamando-os como funções

No Keras, um objeto Layer tem o método `__call__` definido. Portanto, o usuário pode invocar a lógica da camada chamando o objeto como uma função. Por exemplo:

```python
# Python:
my_input = keras.Input(shape=[2, 4])
flatten = keras.layers.Flatten()

print(flatten(my_input).shape)
```

A sintaxe "sugar" do Python é implementada como o método apply() no TensorFlow.js:

```js
// JavaScript:
const myInput = tf.input({shape: [2, 4]});
const flatten = tf.layers.flatten();

console.log(flatten.apply(myInput).shape);
```

## Layer.apply() tem suporte à avaliação imperativa (eager) de tensores concretos

Atualmente, no Keras, o método **call** pode operar somente em objetos `tf.Tensor` do TensorFlow (Python) (pressupondo o back-end do TensorFlow), que são simbólicos e não armazenam valores numéricos reais. Isso foi exibido no exemplo da seção anterior. Entretanto, no TensorFlow.js, o método apply() das camadas pode operar tanto no modo simbólico quanto no imperativo. Se `apply()` for invocado com um SymbolicTensor (uma analogia próxima a tf.Tensor), o valor retornado será um SymbolicTensor. Tipicamente, isso ocorre durante a construção do modelo. Porém, se `apply()` for invocado com um valor de Tensor concreto real, retornará um Tensor concreto. Por exemplo:

```js
// JavaScript:
const flatten = tf.layers.flatten();

flatten.apply(tf.ones([2, 3, 4])).print();
```

Esse recurso nos faz lembrar da [execução eager](https://www.tensorflow.org/guide/eager) do TensorFlow (Python) e permite maior interatividade e capacidade de depuração durante o desenvolvimento do modelo, além de abrir as portas para compor redes neurais dinâmicas.

## Os otimizadores estão em train.*, não em optimizers*

No Keras, os construtores de objetos Optimizer estão sob o namespace `keras.optimizers.*`. No TensorFlow.js Layers, os métodos de fábrica dos otimizadores estão sob o namespace `tf.train.*`. Por exemplo:

```python
# Python:
my_sgd = keras.optimizers.sgd(lr=0.2)
```

```js
// JavaScript:
const mySGD = tf.train.sgd({lr: 0.2});
```

## loadLayersModel() carrega a partir de uma URL, não um arquivo HDF5

No Keras, geralmente os modelos são [salvos](https://keras.io/getting_started/faq/#what-are-my-options-for-saving-models) como um arquivo HDF5 (.h5), que pode ser carregado posteriormente usando o método `keras.models.load_model()`. Esse método recebe um caminho para o arquivo .h5. A contrapartida de `load_model()` no TensorFlow.js é [`tf.loadLayersModel()`](https://js.tensorflow.org/api/latest/#loadLayersModel). Como o HDF5 não é um formato de arquivo otimizado para navegadores, `tf.loadLayersModel()` recebe um formato específico do TensorFlow.js. `tf.loadLayersModel()` recebe um arquivo model.json como argumento de entrada, que pode ser convertido a partir de um arquivo HDF5 do Keras usando o pacote pip tensorflowjs.

```js
// JavaScript:
const model = await tf.loadLayersModel('https://foo.bar/model.json');
```

Observe também que `tf.loadLayersModel()` retorna uma [`Promise`](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Promise) de [`tf.Model`](https://js.tensorflow.org/api/latest/#class:Model).

De forma geral, salvar e carregar `tf.Model`s no TensorFlow.js é feito usando-se os métodos `tf.Model.save` e `tf.loadLayersModel`, respectivamente. Desenvolvemos essas APIs de maneira similar à [API de salvamento e carregamento de modelos](https://keras.io/getting_started/faq/#what-are-my-options-for-saving-models) do Keras. Porém, o ambiente de navegadores é bem diferente do ambiente de back-end no qual frameworks básicos de aprendizado profundo, como o Keras, são executados, especialmente na gama de rotas para fazer a persistência e transmissão de dados. Portanto, veja algumas diferenças interessantes entre as APIs de salvamento/carregamento no TensorFlow.js e no Keras. Confira mais informações no tutorial [Como salvar e carregar tf.Model](./save_load.md).

## Use `fitDataset()` para treinar modelos usando objetos `tf.data.Dataset`

No tf.keras do TensorFlow (Python), um modelo pode ser treinado usando um objeto [Dataset](https://www.tensorflow.org/guide/datasets). O método `fit()` do modelo aceita um objeto como esse diretamente. Um modelo do TensorFlow.js também pode ser treinado com o equivalente dos objetos Dataset no JavaScript (confira [a documentação da API tf.data no TensorFlow.js](https://js.tensorflow.org/api/latest/#Data)). Entretanto, diferentemente do Python, o treinamento baseado em Datasets é feito por meio de um método dedicado, [fitDataset](https://js.tensorflow.org/api/0.15.1/#tf.Model.fitDataset). O método [fit()](https://js.tensorflow.org/api/latest/#tf.Model.fitDataset) serve apenas para treinamento de modelos baseados em tensores.

## Gerenciamento de memória de objetos Layer e Model

O TensorFlow.js é executado no WebGL no navegador, em que os pesos dos objetos Layer e Model são texturas do WebGL. Porém, o WebGL não tem suporte à coleta de lixo integrada. Os objetos Layer e Model gerenciam internamente a memória dos tensores para o usuário durante as chamadas de inferência e treinamento, mas eles também permitem que o usuário os descartem para liberar a memória do WebGL ocupada, o que é útil nos casos em que diversas instâncias do modelo são criadas e liberadas durante um único carregamento de página. Para descartar um objeto Layer ou Model, use o método `dispose()`.
