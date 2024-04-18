# Treinamento de modelos

Este guia pressupõe que você já tenha lido o guia [Modelos e camadas](models_and_layers.md).

No TensorFlow.js, há duas maneiras de treinar um modelo de aprendizado de máquina:

1. Usando a API Layers, com <code>&lt;a href="https://js.tensorflow.org/api/latest/#tf.Model.fit" data-md-type="link"&gt;LayersModel.fit()&lt;/a&gt;</code> ou <code>&lt;a href="https://js.tensorflow.org/api/latest/#tf.Model.fitDataset" data-md-type="link"&gt;LayersModel.fitDataset()&lt;/a&gt;</code>.
2. Usando a API Core, com <code>&lt;a href="https://js.tensorflow.org/api/latest/#tf.train.Optimizer.minimize" data-md-type="link"&gt;Optimizer.minimize()&lt;/a&gt;</code>.

Primeiros, veremos a API Layers, uma API de alto nível para criar modelos. Em seguida, mostraremos como treinar o mesmo modelo usando a API Core.

## Introdução

Um *modelo* de aprendizado de máquina é uma função com parâmetros aprendíveis que mapeia uma entrada para uma saída desejada. Os parâmetros ideias são obtidos ao treinar o modelo com dados.

O treinamento é composto por várias etapas:

- Obter um [lote](https://developers.google.com/machine-learning/glossary/#batch) de dados para o modelo.
- Pedir para o modelo fazer uma previsão.
- Comparar essa previsão com o valor "verdadeiro".
- Decidir o quanto alterar cada parâmetro para que o modelo consiga fazer uma previsão melhor no futuro para esse lote.

Um modelo bem treinado fornecerá um mapeamento exato da entrada para a saída desejada.

## Parâmetros do modelo

Vamos definir um modelo simples com duas camadas usando a API Layers:

```js
const model = tf.sequential({
 layers: [
   tf.layers.dense({inputShape: [784], units: 32, activation: 'relu'}),
   tf.layers.dense({units: 10, activation: 'softmax'}),
 ]
});
```

Nos bastidores, os modelos têm parâmetros (geralmente chamados de *pesos*) que são aprendidos ao fazer o treinamento com dados. Vamos exibir via print os nomes dos pesos associados a este modelo e seus formatos:

```js
model.weights.forEach(w => {
 console.log(w.name, w.shape);
});
```

Vamos obter a seguinte saída:

```
> dense_Dense1/kernel [784, 32]
> dense_Dense1/bias [32]
> dense_Dense2/kernel [32, 10]
> dense_Dense2/bias [10]
```

Há quatro pesos no total, dois por camada densa. Isso é o esperado, já que camadas densas representam uma função que mapeia o tensor de entrada `x` para um tensor de saída `y` por meio da equação `y = Ax + b`, em que `A` (o kernel) e `b` (o bias) são os parâmetros da camada densa.

> OBSERVAÇÃO: por padrão, as camadas densas incluem um bias, mas você pode excluí-lo especificando `{useBias: false}` nas opções ao criar uma camada densa.

`model.summary()` é um método útil se você quiser obter uma visão geral do modelo e ver o número total de parâmetros:

<table>
  <tr>
   <td>Camada (tipo)</td>
   <td>Formato da saída</td>
   <td>Nº de parâm.</td>
  </tr>
  <tr>
   <td>dense_Dense1 (Densa)</td>
   <td>[null,32]    </td>
   <td>25.120</td>
  </tr>
  <tr>
   <td>dense_Dense2 (Densa)</td>
   <td>[null,10]    </td>
   <td>330    </td>
  </tr>
  <tr>
   <td colspan="3">Total de parâmetros: 25.450<br>Parâmetros treináveis: 25.450<br>Parâmetros não treináveis: 0</td>
  </tr>
</table>

Cada peso no modelo tem um respectivo objeto <code>&lt;a href="https://js.tensorflow.org/api/0.14.2/#class:Variable" data-md-type="link"&gt;Variable&lt;/a&gt;</code>. No TensorFlow.js, uma <code>Variable</code> é um <code>Tensor</code> de ponto flutuante com um método adicional <code>assign()</code> usado para atualizar seus valores. A API Layers inicializa os pesos automaticamente usando práticas recomendadas. Para fins de demonstração, poderíamos sobrescrever os pesos chamando <code>assign()</code> nas variáveis subjacentes:

```js
model.weights.forEach(w => {
  const newVals = tf.randomNormal(w.shape);
  // w.val é uma instância de tf.Variable
  w.val.assign(newVals);
});
```

## Otimizador, perda e métrica

Antes de fazer qualquer treinamento, você precisa tomar decisões sobre três aspectos:

1. **Um otimizador**. O trabalho do otimizador é decidir o quanto alterar cada parâmetro no modelo dada a previsão atual do modelo. Ao usar a API Layers, você pode fornecer um identificador string ou um otimizador existente (como `'sgd'` ou `'adam'`), ou ainda uma instância da classe <code>&lt;a href="https://js.tensorflow.org/api/latest/#Training-Optimizers" data-md-type="link"&gt;Optimizer&lt;/a&gt;</code>.
2. <strong>Uma função de perda</strong>. Um valor que o modelo tentará minimizar. O objetivo é fornecer um único número do "quão errada" a previsão do modelo estava. A perda é computada em cada lote de dados para que o modelo possa atualizar seus pesos. Ao usar a API Layers, você pode fornecer um identificador string de uma função de perda existente (como <code>'categoricalCrossentropy'</code>) ou qualquer função que receba um valor previsto e um verdadeiro, e retorne uma perda. Confira a [lista de perdas disponíveis](https://js.tensorflow.org/api/latest/#Training-Losses) na documentação da API.
3. <strong>Lista de métricas.</strong> Similares às perdas, as métricas computam um único número que resume o desempenho do modelo. Geralmente, as métricas são computadas para todos os dados no final de cada época. No mínimo, vamos querer monitorar se a perda está diminuindo ao longo do tempo. Entretanto, é comum querermos uma métrica mais amigável, como exatidão. Ao usar a API Layers, você pode fornecer um identificador string ou uma métrica existente (como <code>'accuracy'</code>) ou qualquer função que receba um valor previsto e um verdadeiro, e retorne uma pontuação. Confira a [lista de métricas disponíveis](https://js.tensorflow.org/api/latest/#Metrics) na documentação da API.

Após você se decidir, compile um <code>LayersModel</code> chamando <code>model.compile()</code> com as opções fornecidas:

```js
model.compile({
  optimizer: 'sgd',
  loss: 'categoricalCrossentropy',
  metrics: ['accuracy']
});
```

Durante a compilação, o modelo fará uma validação para garantir que as opções escolhidas sejam compatíveis entre si.

## Treinamento

Existem duas maneiras de treinar um `LayersModel`:

- Usando `model.fit()` e fornecendo os dados como um grande tensor.
- Usando `model.fitDataset()` e fornecendo os dados por um objeto `Dataset`.

### model.fit()

Se o dataset couber na memória principal e estiver disponível como um único tensor, você pode treinar um modelo chamando o método `fit()`:

```js
// Gera dados aleatórios.
const data = tf.randomNormal([100, 784]);
const labels = tf.randomUniform([100, 10]);

function onBatchEnd(batch, logs) {
  console.log('Precisão', logs.acc);
}

// Treina por 5 épocas com tamanho de lote 32.
model.fit(data, labels, {
   epochs: 5,
   batchSize: 32,
   callbacks: {onBatchEnd}
 }).then(info => {
   console.log('Precisão final', info.history.acc);
 });
```

Nos bastidores, `model.fit()` pode fazer muita coisa:

- Divide os dados em datasets de treinamento e validação, e usa o dataset de validação para mensurar o progresso durante o treinamento.
- Mistura os dados, porém somente após a divisão. Por segurança, você deve pré-misturar os dados antes de passá-los a `fit()`.
- Divide o tensor grande de dados em tensores menores de tamanho igual a `batchSize`.
- Chama `optimizer.minimize()` ao computar a perda do modelo com relação ao lote de dados.
- Notifica você sobre o começo e fim de cada época ou lote. No nosso caso, recebemos uma notificação no fim de cada lote usando a opção `callbacks.onBatchEnd `. Confira outras opções: `onTrainBegin`, `onTrainEnd`, `onEpochBegin`, `onEpochEnd` e `onBatchBegin`.
- Cede ao thread principal para garantir que as tarefas enfileiradas no loop de eventos do JS possam ser tratadas em tempo hábil.

Confira mais informações na [documentação](https://js.tensorflow.org/api/latest/#tf.Sequential.fit) de `fit()`. Observe que, se você optar por usar a API Core, precisará implementar essa lógica por conta própria.

### model.fitDataset()

Se os dados não couberem inteiramente na memória ou estiverem sendo transmitidos, é possível treinar um modelo chamando `fitDataset()`, que recebe um objeto `Dataset`. Aqui, temos o mesmo código de treinamento, porém com um dataset que encapsula uma função geradora:

```js
function* data() {
 for (let i = 0; i < 100; i++) {
   // Gera uma amostra de cada vez.
   yield tf.randomNormal([784]);
 }
}

function* labels() {
 for (let i = 0; i < 100; i++) {
   // Gera uma amostra de cada vez.
   yield tf.randomUniform([10]);
 }
}

const xs = tf.data.generator(data);
const ys = tf.data.generator(labels);
// Nós compactamos os dados e os rótulos juntos, embaralhamos e agrupamos 32 amostras por vez.
const ds = tf.data.zip({xs, ys}).shuffle(100 /* bufferSize */).batch(32);

// Treina o modelo por 5 épocas.
model.fitDataset(ds, {epochs: 5}).then(info => {
 console.log('Accuracy', info.history.acc);
});
```

Confira mais informações sobre datasets na [documentação](https://js.tensorflow.org/api/latest/#tf.Model.fitDataset) de `model.fitDataset()`.

## Previsão de novos dados

Após o modelo ser treinado, você pode chamar `model.predict()` para fazer previsões com dados nunca vistos:

```js
// Prediz 3 amostras aleatórias.
const prediction = model.predict(tf.randomNormal([3, 784]));
prediction.print();
```

Observação: como mencionamos no guia [Modelos e camadas](models_and_layers), o `LayersModel` espera que a dimensão mais externa da entrada seja o tamanho do lote. No exemplo anterior, o tamanho do lote é 3.

## API Core

Anteriormente, mencionamos que existem duas maneiras de treinar um modelo de aprendizado de máquina no TensorFlow.js.

A regra geral é sempre tentar usar a API Layers primeiro, já que ela é modelada de acordo com a API Keras, amplamente adotada. A API Layers também oferece diversas soluções prontas para uso, como inicialização de pesos, serialização de modelos, monitoramento de treinamento, portabilidade e checagem de segurança.

Talvez você queira usar a API Core sempre que:

- Precisar de flexibilidade ou controle máximos.
- Não precisar de serialização ou puder implementar sua própria lógica de serialização.

Confira mais informações sobre essa API na seção "API Core" do guia [Modelos e camadas](models_and_layers.md).

O mesmo modelo acima escrito usando a API Core fica da seguinte forma:

```js
// Os pesos e bias para as duas camadas densas.
const w1 = tf.variable(tf.randomNormal([784, 32]));
const b1 = tf.variable(tf.randomNormal([32]));
const w2 = tf.variable(tf.randomNormal([32, 10]));
const b2 = tf.variable(tf.randomNormal([10]));

function model(x) {
  return x.matMul(w1).add(b1).relu().matMul(w2).add(b2);
}
```

Além da API Layers, a API Data também funciona de maneira integrada com a API Core. Vamos reutilizar o dataset definido anteriormente na seção [model.fitDataset()](#model.fitDataset()), que faz a mistura dos dados e divisão em lotes:

```js
const xs = tf.data.generator(data);
const ys = tf.data.generator(labels);
// Nós compactamos os dados e os rótulos juntos, embaralhamos e agrupamos 32 amostras por vez.
const ds = tf.data.zip({xs, ys}).shuffle(100 /* bufferSize */).batch(32);
```

Vamos treinar o modelo:

```js
const optimizer = tf.train.sgd(0.1 /* learningRate */);
// Treina por 5 épocas.
for (let epoch = 0; epoch < 5; epoch++) {
  await ds.forEachAsync(({xs, ys}) => {
    optimizer.minimize(() => {
      const predYs = model(xs);
      const loss = tf.losses.softmaxCrossEntropy(ys, predYs);
      loss.data().then(l => console.log('Perda', l));
      return loss;
    });
  });
  console.log('Época', epoch);
}
```

O código acima é a receita padrão ao treinar um modelo com a API Core:

- Faça um loop com o número de épocas.
- Dentro de cada época, faça um loop dos lotes de dados. Ao usar um `Dataset`, <code>&lt;a href="https://js.tensorflow.org/api/0.15.1/#tf.data.Dataset.forEachAsync" data-md-type="link"&gt;dataset.forEachAsync()&lt;/a&gt;</code> é uma maneira conveniente de fazer o loop dos lotes.
- Para cada lote, chame <code>&lt;a href="https://js.tensorflow.org/api/latest/#tf.train.Optimizer.minimize" data-md-type="link"&gt;optimizer.minimize(f)&lt;/a&gt;</code>, que executa <code>f</code> e minimiza sua saída computando gradientes em relação às quatro variáveis que definimos anteriormente.
- <code>f</code> computa a perda. Ele chama uma das funções de perda predefinidas usando a previsão do modelo e o valor verdadeiro.
