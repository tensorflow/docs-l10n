# 训练模型

本指南假定您已阅读[模型和层](models_and_layers.md)指南。

在 TensorFlow.js 中，您可以通过以下两种方式训练机器学习模型：

1. 使用 Layers API 与 <code>&lt;a href="https://js.tensorflow.org/api/latest/#tf.Model.fit" data-md-type="link"&gt;LayersModel.fit()&lt;/a&gt;</code> 或 <code>&lt;a href="https://js.tensorflow.org/api/latest/#tf.Model.fitDataset" data-md-type="link"&gt;LayersModel.fitDataset()&lt;/a&gt;</code>。
2. 使用 Core API 与 <code>&lt;a href="https://js.tensorflow.org/api/latest/#tf.train.Optimizer.minimize" data-md-type="link"&gt;Optimizer.minimize()&lt;/a&gt;</code>。

首先，我们将了解 Layers API，它是一种用于构建和训练模型的高级 API。然后，我们将展示如何使用 Core API 训练相同的模型。

## 简介

机器学习*模型*是一种具有可学习参数的函数，可将输入映射到所需输出。基于数据训练模型可以获得最佳参数。

训练涉及多个步骤：

- 获取一[批次](https://developers.google.com/machine-learning/glossary/#batch)数据来训练模型。
- 让模型做出预测。
- 将该预测与“真实”值进行对比。
- 确定每个参数的更改幅度，使模型在未来能够针对该批次数据做出更好的预测。

训练得当的模型将提供从输入到所需输出的准确映射。

## 模型参数

让我们使用 Layers API 来定义一个简单的 2 层模型：

```js
const model = tf.sequential({
 layers: [
   tf.layers.dense({inputShape: [784], units: 32, activation: 'relu'}),
   tf.layers.dense({units: 10, activation: 'softmax'}),
 ]
});
```

模型以可学习参数（常称为*权重*）为基础，基于数据进行训练。让我们打印与此模型及其形状关联的权重的名称：

```js
model.weights.forEach(w => {
 console.log(w.name, w.shape);
});
```

我们得到以下输出：

```
> dense_Dense1/kernel [784, 32]
> dense_Dense1/bias [32]
> dense_Dense2/kernel [32, 10]
> dense_Dense2/bias [10]
```

共有 4 个权重，每个密集层 2 个。这是可以预期的，因为密集层表示一个函数，通过等式 `y = Ax + b` 将输入张量 `x` 映射到输出张量 `y`，其中 `A`（内核）和 `b`（偏差）为密集层参数。

> 注：默认情况下，密集层将包含偏差，但您可以通过在创建密集层时的选项中指定 `{useBias: false}` 将其排除。

如果您想简要了解模型并查看参数总数，`model.summary()` 是一种实用的方法：

<table>
  <tr>
   <td>层（类型）</td>
   <td>输出形状</td>
   <td>参数数量</td>
  </tr>
  <tr>
   <td>dense_Dense1（密集）</td>
   <td>[null,32]</td>
   <td>25120</td>
  </tr>
  <tr>
   <td>dense_Dense2（密集）</td>
   <td>[null,10]</td>
   <td>330</td>
  </tr>
  <tr>
   <td colspan="3">参数总数：25450<br>可训练参数：25450<br>不可训练参数：0</td>
  </tr>
</table>

模型中的每个权重均由 <code>&lt;a href="https://js.tensorflow.org/api/0.14.2/#class:Variable" data-md-type="link"&gt;Variable&lt;/a&gt;</code> 对象提供支持。在 TensorFlow.js 中，<code>Variable</code> 为浮点型 <code>Tensor</code>，具有一个用于更新值的附加方法 <code>assign()</code>。Layers API 会使用最佳做法自动初始化权重。出于演示目的，我们可以通过在基础变量上调用 <code>assign()</code> 来覆盖权重：

```js
model.weights.forEach(w => {
  const newVals = tf.randomNormal(w.shape);
  // w.val is an instance of tf.Variable
  w.val.assign(newVals);
});
```

## 优化器、损失和指标

进行任何训练之前，您需要确定以下三项内容：

1. **优化器**。优化器的作用是在给定当前模型预测的情况下，决定对模型中每个参数实施更改的幅度。使用 Layers API 时，您可以提供现有优化器的字符串标识符（例如 `'sgd'` 或 `'adam'`），也可以提供 <code>&lt;a href="https://js.tensorflow.org/api/latest/#Training-Optimizers" data-md-type="link"&gt;Optimizer&lt;/a&gt;</code> 类的实例。
2. <strong>损失函数</strong>。模型将以最小化损失作为目标。该函数旨在将模型预测的“误差程度”量化为具体数字。损失以每一批次数据为基础计算，因此模型可以更新其权重。使用 Layers API 时，您可以提供现有损失函数的字符串标识符（例如 <code>'categoricalCrossentropy'</code>），也可以提供任何采用预测值和真实值并返回损失的函数。请参阅我们的 API 文档中的[可用损失列表](https://js.tensorflow.org/api/latest/#Training-Losses)。
3. <strong>指标列表。</strong>与损失类似，指标也会计算一个数字，用于总结模型的运作情况。通常要在每个周期结束时基于整体数据来计算指标。至少，我们要监控损失是否随着时间推移而下降。但是，我们经常需要准确率等更人性化的指标。使用 Layers API 时，您可以提供现有指标的字符串标识符（例如 <code>'accuracy'</code>），也可以提供任何采用预测值和真实值并返回分数的函数。请参阅我们的 API 文档中的[可用指标列表](https://js.tensorflow.org/api/latest/#Metrics)。

确定后，使用提供的选项调用 <code>model.compile()</code> 来编译 <code>LayersModel</code>：

```js
model.compile({
  optimizer: 'sgd',
  loss: 'categoricalCrossentropy',
  metrics: ['accuracy']
});
```

在编译过程中，模型将进行一些验证以确保您所选择的选项彼此兼容。

## 训练

您可以通过以下两种方式训练 `LayersModel`：

- 使用 `model.fit()` 并以一个大型张量形式提供数据。
- 使用 `model.fitDataset()` 并通过 `Dataset` 对象提供数据。

### model.fit()

如果您的数据集适合装入主内存，并且可以作为单个张量使用，则您可以通过调用 `fit()` 方法来训练模型：

```js
// Generate dummy data.
const data = tf.randomNormal([100, 784]);
const labels = tf.randomUniform([100, 10]);

function onBatchEnd(batch, logs) {
  console.log('Accuracy', logs.acc);
}

// Train for 5 epochs with batch size of 32.
model.fit(data, labels, {
   epochs: 5,
   batchSize: 32,
   callbacks: {onBatchEnd}
 }).then(info => {
   console.log('Final accuracy', info.history.acc);
 });
```

`model.fit()` 在后台可以完成很多操作：

- 将数据拆分为训练集和验证集，并使用验证集衡量训练期间的进度。
- 打乱数据顺序（仅在拆分后）。为了安全起见，您应该在将数据传递至 `fit()` 之前预先打乱数据顺序。
- 将大型数据张量拆分成大小为 `batchSize` 的小型张量。
- 在计算相对于一批次数据的模型损失的同时，调用 `optimizer.minimize()`。
- 可以在每个周期或批次的开始和结尾为您提供通知。我们的示例使用 `callbacks.onBatchEnd` 选项在每个批次的结尾提供通知。其他选项包括：`onTrainBegin`、`onTrainEnd`、`onEpochBegin`、`onEpochEnd` 和 `onBatchBegin`。
- 受制于主线程，确保 JS 事件循环中排队的任务可以得到及时处理。

有关更多信息，请参阅 `fit()` 的[文档](https://js.tensorflow.org/api/latest/#tf.Sequential.fit)。请注意，如果您选择使用 Core API，则必须自行实现此逻辑。

### model.fitDataset()

如果您的数据不能完全装入内存或进行流式传输，则您可以通过调用 `fitDataset()` 来训练模型，它会获取一个 `Dataset` 对象。以下为相同的训练代码，但具有包装生成器函数的数据集：

```js
function* data() {
 for (let i = 0; i < 100; i++) {
   // Generate one sample at a time.
   yield tf.randomNormal([784]);
 }
}

function* labels() {
 for (let i = 0; i < 100; i++) {
   // Generate one sample at a time.
   yield tf.randomUniform([10]);
 }
}

const xs = tf.data.generator(data);
const ys = tf.data.generator(labels);
// We zip the data and labels together, shuffle and batch 32 samples at a time.
const ds = tf.data.zip({xs, ys}).shuffle(100 /* bufferSize */).batch(32);

// Train the model for 5 epochs.
model.fitDataset(ds, {epochs: 5}).then(info => {
 console.log('Accuracy', info.history.acc);
});
```

有关数据集的更多信息，请参阅 `model.fitDataset()` [文档](https://js.tensorflow.org/api/latest/#tf.Model.fitDataset)。

## 预测新数据

在模型完成训练后，您可以调用 `model.predict()`，基于未见过的数据进行预测：

```js
// Predict 3 random samples.
const prediction = model.predict(tf.randomNormal([3, 784]));
prediction.print();
```

注：正如我们在[模型和层](models_and_layers)指南中所讲，`LayersModel` 期望输入的最外层维度为批次大小。在上例中，批次大小为 3。

## Core API

之前，我们提到您可以通过两种方式在 TensorFlow.js 中训练机器学习模型。

根据常规经验法则，可以首先尝试使用 Layers API，因为它是由广为采用的 Keras API 建模而成。Layers API 还提供了各种现成的解决方案，例如权重初始化、模型序列化、监控训练、可移植性和安全性检查。

在以下情况下，您可以使用 Core API：

- 您需要最大的灵活性或控制力。
- 并且您不需要序列化，或者可以实现自己的序列化逻辑。

有关此 API 的更多信息，请参阅[模型和层](models_and_layers.md)指南中的“Core API”部分。

使用 Core API 编写上述相同模型，方法如下：

```js
// The weights and biases for the two dense layers.
const w1 = tf.variable(tf.randomNormal([784, 32]));
const b1 = tf.variable(tf.randomNormal([32]));
const w2 = tf.variable(tf.randomNormal([32, 10]));
const b2 = tf.variable(tf.randomNormal([10]));

function model(x) {
  return x.matMul(w1).add(b1).relu().matMul(w2).add(b2);
}
```

除了 Layers API 以外，Data API 也可与 Core API 无缝协作。让我们重用先前在 [model.fitDataset()](#model.fitDataset()) 部分中定义的数据集，该数据集已完成打乱顺序和批处理操作：

```js
const xs = tf.data.generator(data);
const ys = tf.data.generator(labels);
// Zip the data and labels together, shuffle and batch 32 samples at a time.
const ds = tf.data.zip({xs, ys}).shuffle(100 /* bufferSize */).batch(32);
```

让我们训练模型：

```js
const optimizer = tf.train.sgd(0.1 /* learningRate */);
// Train for 5 epochs.
for (let epoch = 0; epoch < 5; epoch++) {
  await ds.forEachAsync(({xs, ys}) => {
    optimizer.minimize(() => {
      const predYs = model(xs);
      const loss = tf.losses.softmaxCrossEntropy(ys, predYs);
      loss.data().then(l => console.log('Loss', l));
      return loss;
    });
  });
  console.log('Epoch', epoch);
}
```

以上代码是使用 Core API 训练模型时的标准方法：

- 循环周期数。
- 在每个周期内，循环各批次数据。使用 `Dataset` 时，<code>&lt;a href="https://js.tensorflow.org/api/0.15.1/#tf.data.Dataset.forEachAsync" data-md-type="link"&gt;dataset.forEachAsync()&lt;/a&gt; </code> 可方便地循环各批次数据。
- 针对每个批次，调用 <code>&lt;a href="https://js.tensorflow.org/api/latest/#tf.train.Optimizer.minimize" data-md-type="link"&gt;optimizer.minimize(f)&lt;/a&gt;</code>，它可以执行 <code>f</code> 并通过计算相对于我们先前定义的四个变量的梯度来最小化其输出。
- <code>f</code> 可计算损失。它使用模型的预测和真实值调用预定义的损失函数之一。
