# Setup

## Browser Setup

There are two main ways to get TensorFlow.js in your browser based projects:

  - Using [script tags](https://developer.mozilla.org/en-US/docs/Learn/HTML/Howto/Use_JavaScript_within_a_webpage).

  - Installation from [NPM](https://www.npmjs.com) and using a build tool like [Parcel](https://parceljs.org/), [WebPack](https://webpack.js.org/), or [Rollup](https://rollupjs.org/guide/en).

If you are new to web development, or have never heard of tools like webpack or parcel, _we recommend you use the script tag approach_. If you are more experienced or want to write larger programs it might be worthwhile to explore using build tools.

### Usage via Script Tag

Add the following script tag to your main HTML file.

```html
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@2.0.0/dist/tf.min.js"></script>
```

<section class="expandable">
  <h4 class="showalways">See code sample for script tag setup</h4>
  <pre class="prettyprint">
// Define a model for linear regression.
const model = tf.sequential();
model.add(tf.layers.dense({units: 1, inputShape: [1]}));

model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

// Generate some synthetic data for training.
const xs = tf.tensor2d([1, 2, 3, 4], [4, 1]);
const ys = tf.tensor2d([1, 3, 5, 7], [4, 1]);

// Train the model using the data.
model.fit(xs, ys, {epochs: 10}).then(() => {
  // Use the model to do inference on a data point the model hasn't seen before:
  model.predict(tf.tensor2d([5], [1, 1])).print();
  // Open the browser devtools to see the output
});
  </pre>
</section>

### Installation from NPM

You can use either the [npm cli](https://docs.npmjs.com/cli/npm) tool or [yarn](https://yarnpkg.com/en/) to install TensorFlow.js.

```
yarn add @tensorflow/tfjs
```

_or_

```
npm install @tensorflow/tfjs
```

<section class="expandable">
  <h4 class="showalways">See sample code for installation via NPM</h4>
  <pre class="prettyprint">
import * as tf from '@tensorflow/tfjs';

// Define a model for linear regression.
const model = tf.sequential();
model.add(tf.layers.dense({units: 1, inputShape: [1]}));

model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

// Generate some synthetic data for training.
const xs = tf.tensor2d([1, 2, 3, 4], [4, 1]);
const ys = tf.tensor2d([1, 3, 5, 7], [4, 1]);

// Train the model using the data.
model.fit(xs, ys, {epochs: 10}).then(() => {
  // Use the model to do inference on a data point the model hasn't seen before:
  model.predict(tf.tensor2d([5], [1, 1])).print();
  // Open the browser devtools to see the output
});
  </pre>
</section>


## Node.js Setup

You can use either the [npm cli](https://docs.npmjs.com/cli/npm) tool or [yarn](https://yarnpkg.com/en/) to install TensorFlow.js.

**Option 1:** Install TensorFlow.js with native C++ bindings.

```
yarn add @tensorflow/tfjs-node
```

_or_

```
npm install @tensorflow/tfjs-node
```

**Option 2:** (Linux Only) If your system has a NVIDIA® GPU with [CUDA support](https://www.tensorflow.org/install/install_linux#NVIDIARequirements), use the GPU package even for higher performance.

```
yarn add @tensorflow/tfjs-node-gpu
```

_or_

```
npm install @tensorflow/tfjs-node-gpu
```

**Option 3:** Install the pure JavaScript version. This is the slowest option performance wise.

```
yarn add @tensorflow/tfjs
```

_or_

```
npm install @tensorflow/tfjs
```


<section class="expandable">
  <h4 class="showalways">See sample code for Node.js usage</h4>
  <pre class="prettyprint">
const tf = require('@tensorflow/tfjs');

// Optional Load the binding:
// Use '@tensorflow/tfjs-node-gpu' if running with GPU.
require('@tensorflow/tfjs-node');

// Train a simple model:
const model = tf.sequential();
model.add(tf.layers.dense({units: 100, activation: 'relu', inputShape: [10]}));
model.add(tf.layers.dense({units: 1, activation: 'linear'}));
model.compile({optimizer: 'sgd', loss: 'meanSquaredError'});

const xs = tf.randomNormal([100, 10]);
const ys = tf.randomNormal([100, 1]);

model.fit(xs, ys, {
  epochs: 100,
  callbacks: {
    onEpochEnd: (epoch, log) => console.log(`Epoch ${epoch}: loss = ${log.loss}`)
  }
});
  </pre>
</section>

### TypeScript

When using TypeScript you may need to set `skipLibCheck: true` in your `tsconfig.json` file if your project makes use of strict null checking or you will run into errors during compilation.
