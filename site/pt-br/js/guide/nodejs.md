# TensorFlow.js no Node

## TensorFlow CPU

O pacote TensorFlow CPU pode ser importado da seguinte forma:

```js
import * as tf from '@tensorflow/tfjs-node'
```

Ao importar o TensorFlow.js por meio desse pacote, o módulo obtido será acelerado pelo binário C do TensorFlow e executado na CPU. O TensorFlow na CPU usa aceleração de hardware para aumentar a velocidade da computação de álgebra linear nos bastidores.

Esse pacote funciona nas plataformas Linux, Windows e Mac com as quais o TensorFlow é compatível.

> Observação: você não precisa importar '@tensorflow/tfjs' ou adicioná-lo ao seu package.json. Este pacote é importado indiretamente pela biblioteca do Node.

## TensorFlow GPU

O pacote TensorFlow GPU pode ser importado da seguinte forma:

```js
import * as tf from '@tensorflow/tfjs-node-gpu'
```

Assim como o pacote de CPU, o módulo obtido será acelerado pelo binário C do TensorFlow. Entretanto, ele executará operações dos tensores na GPU com CUDA e, portanto, somente no Linux. Esse binding pode ser pelo menos uma ordem de magnitude mais rápido do que outras opções de binding.

> Observação: esse pacote funciona somente com o CUDA. Você precisa ter o CUDA instalado na máquina com uma placa gráfica da NVIDIA antes de seguir por este caminho.

> Observação: você não precisa importar '@tensorflow/tfjs' ou adicioná-lo ao seu package.json. Este pacote é importado indiretamente pela biblioteca do Node.

## CPU vanilla

A versão do TensorFlow.js que executa operações de CPU vanilla pode ser importada da seguinte forma:

```js
import * as tf from '@tensorflow/tfjs'
```

Este pacote é o mesmo que você usaria no navegador. Nele, as operações são executados no JavaScript vanilla na CPU. Este pacote é muito menor do que os outros porque não precisa do binário do TensorFlow, porém é muito mais lento.

Como este pacote não depende do TensorFlow, pode ser usado em mais dispositivos com suporte ao Node.js do que somente Linux, Windows e Mac.

## Considerações de produção

Os bindings do Node.js fornecem um back-end para o TensorFlow.js que implementa as operações de forma síncrona. Portanto, quando você chama uma operação, como `tf.matMul(a, b)`, ela bloqueará o thread principal até que a operação seja concluída.

Por esse motivo, os bindings são adequados atualmente para scripts e para tarefas offline. Se você deseja usar os bindings do Node.js em uma aplicação em produção, como um servidor web, deve configurar uma fila de trabalhos ou threads workers para que o código do TensorFlow.js não bloqueie o thread principal.

## APIs

Após importar o pacote como tf em qualquer uma das opções acima, todos os símbolos normais do TensorFlow.js aparecerão no módulo importado.

### tf.browser

No pacote normal do TensorFlow.js, os símbolos no namespace `tf.browser.*` não serão utilizáveis no Node.js, pois usam APIs específicas de navegador.

Atualmente, são estes:

- tf.browser.fromPixels
- tf.browser.toPixels

### tf.node

Os dois pacotes do Node.js também fornecem um namespace, `tf.node`, que contém APIs específicas do Node.

O TensorBoard é um exemplo notável de APIs específicas do Node.js.

Veja um exemplo de exportação de resumos para o TensorBoard no Node.js:

```js
const model = tf.sequential();
model.add(tf.layers.dense({ units: 1, inputShape: [200] }));
model.compile({
  loss: 'meanSquaredError',
  optimizer: 'sgd',
  metrics: ['MAE']
});


// Gera alguns dados falsos aleatórios com propósito demonstrativo.
const xs = tf.randomUniform([10000, 200]);
const ys = tf.randomUniform([10000, 1]);
const valXs = tf.randomUniform([1000, 200]);
const valYs = tf.randomUniform([1000, 1]);


// Inicial o processo de treinamento do modelo.
async function train() {
  await model.fit(xs, ys, {
    epochs: 100,
    validationData: [valXs, valYs],
    // Adiciona o callback do tensorBoard aqui.
    callbacks: tf.node.tensorBoard('/tmp/fit_logs_1')
  });
}
train();
```
