# TensorFlow.js in Node

## TensorFlow CPU

The TensorFlow CPU package can be imported as follows:


```js
import * as tf from '@tensorflow/tfjs-node'
```


When importing TensorFlow.js from this package, the module that you get will be accelerated by the TensorFlow C binary and run on the CPU. TensorFlow on the CPU uses hardware acceleration to accelerate the linear algebra computation under the hood.

This package will work on Linux, Windows, and Mac platforms where TensorFlow is supported.

> Note: You do not have to import '@tensorflow/tfjs' or add it to your package.json. This is indirectly imported by the node library.


## TensorFlow GPU

The TensorFlow GPU package can be imported as follows:


```js
import * as tf from '@tensorflow/tfjs-node-gpu'
```


Like the CPU package, the module that you get will be accelerated by the TensorFlow C binary, however it will run tensor operations on the GPU with CUDA and thus only linux. This binding can be at least an order of magnitude faster than the other binding options.

> Note: this package currently only works with CUDA. You need to have CUDA installed on your machine with an NVIDIA graphics card before going this route.

> Note: You do not have to import '@tensorflow/tfjs' or add it to your package.json. This is indirectly imported by the node library.


## Vanilla CPU

The version of TensorFlow.js running with vanilla CPU operations can be imported as follows:


```js
import * as tf from '@tensorflow/tfjs'
```


This package is the same package as what you would use in the browser. In this package, the operations are run in vanilla JavaScript on the CPU. This package is much smaller than the others because it doesn't need the TensorFlow binary, however it is much slower.

Because this package doesn't rely on TensorFlow, it can be used in more devices that support Node.js than just Linux, Windows, and Mac.


## Production considerations

The Node.js bindings provide a backend for TensorFlow.js that implements operations synchronously. That means when you call an operation, e.g. `tf.matMul(a, b)`, it will block the main thread until the operation has completed.

For this reason, the bindings currently are well suited for scripts and offline tasks. If you want to use the Node.js bindings in a production application, like a webserver, you should set up a job queue or set up worker threads so your TensorFlow.js code will not block the main thread.


## APIs

Once you import the package as tf in any of the options above, all of the normal TensorFlow.js symbols will appear on the imported module.


### tf.browser

In the normal TensorFlow.js package, the symbols in the `tf.browser.*` namespace will not be usable in Node.js as they use browser-specific APIs.

Currently, these are:

*   tf.browser.fromPixels
*   tf.browser.toPixels


### tf.node

The two Node.js packages also provide a namespace, `tf.node`, which contain node-specific APIs.

TensorBoard is a notable example of Node.js-specific APIs.

An example of exporting summaries to TensorBoard in Node.js:

```js
const model = tf.sequential();
model.add(tf.layers.dense({ units: 1, inputShape: [200] }));
model.compile({
  loss: 'meanSquaredError',
  optimizer: 'sgd',
  metrics: ['MAE']
});


// Generate some random fake data for demo purpose.
const xs = tf.randomUniform([10000, 200]);
const ys = tf.randomUniform([10000, 1]);
const valXs = tf.randomUniform([1000, 200]);
const valYs = tf.randomUniform([1000, 1]);


// Start model training process.
async function train() {
  await model.fit(xs, ys, {
    epochs: 100,
    validationData: [valXs, valYs],
    // Add the tensorBoard callback here.
    callbacks: tf.node.tensorBoard('/tmp/fit_logs_1')
  });
}
train();
```
