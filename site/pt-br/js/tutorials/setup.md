# Configuração

## Configuração do navegador

Existem duas maneiras principais de colocar o TensorFlow.js em seus projetos para navegador:

- Usando [tags script](https://developer.mozilla.org/en-US/docs/Learn/HTML/Howto/Use_JavaScript_within_a_webpage).

- Instalando pelo [NPM](https://www.npmjs.com) e usando uma ferramenta de compilação, como [Parcel](https://parceljs.org/), [WebPack](https://webpack.js.org/) ou [Rollup](https://rollupjs.org/guide/en).

Se você for iniciante no desenvolvimento web ou nunca tiver ouvido falar de ferramentas como WebPack ou Parcel, *recomendamos o uso da estratégia de tags script*. Se você tiver mais experiência ou quiser escrever programas maiores, pode valer a pena usar ferramentas de build.

### Uso via tag script

Adicione a seguinte tag script ao seu arquivo HTML principal.

```html
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@2.0.0/dist/tf.min.js"></script>
```

<section class="expandable">
  <h4 class="showalways">Veja o exemplo de código de configuração da tag script</h4>
  <pre class="prettyprint"> // Define a model for linear regression. const model = tf.sequential(); model.add(tf.layers.dense({units: 1, inputShape: [1]})); </pre></section>

model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

// Generate some synthetic data for training. const xs = tf.tensor2d([1, 2, 3, 4], [4, 1]); const ys = tf.tensor2d([1, 3, 5, 7], [4, 1]);

// Train the model using the data. model.fit(xs, ys, {epochs: 10}).then(() =&gt; { // Use the model to do inference on a data point the model hasn't seen before: model.predict(tf.tensor2d([5], [1, 1])).print(); // Open the browser devtools to see the output });




### Instalação via NPM

Você pode usar a ferramenta [npm cli](https://docs.npmjs.com/cli/npm) ou [yarn](https://yarnpkg.com/en/) para instalar o TensorFlow.js.

```
yarn add @tensorflow/tfjs
```

*ou*

```
npm install @tensorflow/tfjs
```

<section class="expandable">
  <h4 class="showalways">Veja o exemplo de código para instalação via NPM</h4>
  <pre class="prettyprint"> import * as tf from '@tensorflow/tfjs'; </pre></section>

// Define a model for linear regression. const model = tf.sequential(); model.add(tf.layers.dense({units: 1, inputShape: [1]}));

model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

// Generate some synthetic data for training. const xs = tf.tensor2d([1, 2, 3, 4], [4, 1]); const ys = tf.tensor2d([1, 3, 5, 7], [4, 1]);

// Train the model using the data. model.fit(xs, ys, {epochs: 10}).then(() =&gt; { // Use the model to do inference on a data point the model hasn't seen before: model.predict(tf.tensor2d([5], [1, 1])).print(); // Open the browser devtools to see the output });




## Configuração do Node.js

Você pode usar a ferramenta [npm cli](https://docs.npmjs.com/cli/npm) ou [yarn](https://yarnpkg.com/en/) para instalar o TensorFlow.js.

**Opção 1:** instale o TensorFlow.js com as vinculações C++ nativas.

```
yarn add @tensorflow/tfjs-node
```

*ou*

```
npm install @tensorflow/tfjs-node
```

**Opção 2** (somente Linux): se o seu sistema tiver uma CPU NVIDIA® GPU com [suporte a CUDA](https://www.tensorflow.org/install/install_linux#NVIDIARequirements), use o pacote GPU para maior desempenho.

```
yarn add @tensorflow/tfjs-node-gpu
```

*ou*

```
npm install @tensorflow/tfjs-node-gpu
```

**Opção 3:** instale a versão puramente em JavaScript. É a opção mais lenta em termos de desempenho.

```
yarn add @tensorflow/tfjs
```

*ou*

```
npm install @tensorflow/tfjs
```

<section class="expandable">
  <h4 class="showalways">Veja o código de exemplo de uso do Node.js</h4>
  <pre class="prettyprint"> const tf = require('@tensorflow/tfjs'); </pre></section>

// Optional Load the binding: // Use '@tensorflow/tfjs-node-gpu' if running with GPU. require('@tensorflow/tfjs-node');

// Train a simple model: const model = tf.sequential(); model.add(tf.layers.dense({units: 100, activation: 'relu', inputShape: [10]})); model.add(tf.layers.dense({units: 1, activation: 'linear'})); model.compile({optimizer: 'sgd', loss: 'meanSquaredError'});

const xs = tf.randomNormal([100, 10]); const ys = tf.randomNormal([100, 1]);

model.fit(xs, ys, { epochs: 100, callbacks: { onEpochEnd: (epoch, log) =&gt; console.log(`Epoch ${epoch}: loss = ${log.loss}`) } });




### TypeScript

Ao usar o TypeScript, talvez você precise definir `skipLibCheck: true` no arquivo `tsconfig.json` caso seu projeto use verificação estrita de null ou se houver erros durante a compilação.
