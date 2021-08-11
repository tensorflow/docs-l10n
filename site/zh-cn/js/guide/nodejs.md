# Node 中的 TensorFlow.js

## TensorFlow CPU

TensorFlow CPU 软件包可以按如下方式导入：

```js
import * as tf from '@tensorflow/tfjs-node'
```

从此软件包导入 TensorFlow.js 时，您导入的模块将由 TensorFlow C 二进制文件加速并在 CPU 上运行。CPU 上的 TensorFlow 使用硬件加速来加速后台的线性代数运算。

此软件包可以在支持 TensorFlow 的 Linux、Windows 和 Mac 平台上运行。

> 注：您不必导入 '@tensorflow/tfjs' 或者将其添加到您的 package.json 中。它由 Node 库间接导入。

## TensorFlow GPU

TensorFlow GPU 软件包可以按如下方式导入：

```js
import * as tf from '@tensorflow/tfjs-node-gpu'
```

与 CPU 软件包一样，您导入的模块将由 TensorFlow C 二进制文件加速，但是它将在支持 CUDA 的 GPU 上运行张量运算，因此只能在 Linux 平台上运行。此绑定比其他绑定选项至少快一个数量级。

> 注：此软件包目前仅适用于 CUDA。在选择本方案之前，您需要在带有 NVIDIA 显卡的的计算机上安装 CUDA。

> 注：您不必导入 '@tensorflow/tfjs' 或者将其添加到您的 package.json 中。它由 Node 库间接导入。

## 普通 CPU

使用普通 CPU 运算运行的 TensorFlow.js 版本可以按如下方式导入：

```js
import * as tf from '@tensorflow/tfjs'
```

此软件包与您在浏览器中使用的软件包相同。在此软件包中，运算在 CPU 上以原生 JavaScript 运行。此软件包比其他软件包小得多，因为它不需要 TensorFlow 二进制文件，但是速度要慢得多。

由于此软件包不依赖于 TensorFlow，因此它可用于支持 Node.js 的更多设备，而不仅仅是 Linux、Windows 和 Mac 平台。

## 生产考量因素

Node.js 绑定为 TensorFlow.js 提供了一个同步执行运算的后端。这意味着当您调用一个运算（例如 `tf.matMul(a, b)`）时，它将阻塞主线程，直到运算完成。

因此，绑定当前非常适合脚本和离线任务。如果您要在正式应用（例如网络服务器）中使用 Node.js 绑定，应设置一个作业队列或设置一些工作进程线程，以便您的 TensorFlow.js 代码不会阻塞主线程。

## API

一旦您在上面的任何选项中将软件包作为 tf 导入，所有普通的 TensorFlow.js 符号都将出现在导入的模块上。

### tf.browser

在普通的 TensorFlow.js 软件包中，`tf.browser.*` 命名空间中的符号将在 Node.js 中不可用，因为它们使用浏览器特定的 API。

目前，存在以下 API：

- tf.browser.fromPixels
- tf.browser.toPixels

### tf.node

两个 Node.js 软件包还提供了一个名为 `tf.node` 的命名空间，其中包含 Node 特定的 API。

TensorBoard 是一个值得注意的 Node.js 特定的 API 示例。

在 Node.js 中将摘要导出到 TensorBoard 的示例：

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
