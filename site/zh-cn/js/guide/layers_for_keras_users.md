# 适用于 Keras 用户的 TensorFlow.js 层 API

TensorFlow.js 的 Layers API 以 Keras 为模型。考虑到 JavaScript 与 Python 之间的差异，我们努力使 [Layers API](https://js.tensorflow.org/api/latest/) 与 Keras 类似。这样，具有使用 Python 开发 Keras 模型经验的用户可以更轻松地迁移到使用 JavaScript 编写的 TensorFlow.js 层。例如，以下 Keras 代码可以转换为 JavaScript：

```python
# Python:
import keras
import numpy as np

# Build and compile model.
model = keras.Sequential()
model.add(keras.layers.Dense(units=1, input_shape=[1]))
model.compile(optimizer='sgd', loss='mean_squared_error')

# Generate some synthetic data for training.
xs = np.array([[1], [2], [3], [4]])
ys = np.array([[1], [3], [5], [7]])

# Train model with fit().
model.fit(xs, ys, epochs=1000)

# Run inference with predict().
print(model.predict(np.array([[5]])))
```

```js
// JavaScript:
import * as tf from '@tensorlowjs/tfjs';

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

但是，我们希望在本文档中说明并解释一些差异。一旦理解了这些差异及其背后的基本原理，将您的程序从Python 迁移到JavaScript（或反向迁移）应该会是一种相对平稳的体验。

## 构造函数将 JavaScript 对象作为配置

比较上例中的以下 Python 和 JavaScript 代码：它们都可以创建一个[密集](https://keras.io/layers/core/#dense)层。

```python
# Python:
keras.layers.Dense(units=1, inputShape=[1])
```

```js
// JavaScript:
tf.layers.dense({units: 1, inputShape: [1]});
```

JavaScript 函数在 Python 函数中没有等效的关键字参数。我们希望避免在 JavaScript 中将构造函数选项作为位置参数实现，这对于记忆和使用具有大量关键字参数的构造函数（例如 [LSTM](https://keras.io/layers/recurrent/#lstm)）来说尤其麻烦。这就是我们使用 JavaScript 配置对象的原因。这些对象提供与 Python 关键字参数相同的位置不变性和灵活性。

Model 类的一些方法（例如 [`Model.compile()`](https://keras.io/models/model/#model-class-api)）也将 JavaScript 配置对象作为输入。但是请记住，<code>Model.fit()</code>、<code>Model.evaluate()</code> 和 <code>Model.predict()</code> 略有不同。因为这些方法将强制 <code>x</code>（特征）和 <code>y</code>（标签或目标）数据作为输入；<code>x</code> 和 <code>y</code> 是与后续配置对象分开的位置参数，属于关键字参数。例如：

```js
// JavaScript:
await model.fit(xs, ys, {epochs: 1000});
```

## Model.fit() 是异步方法

`Model.fit()` 是用户在 TensorFlow.js 中执行模型训练的主要方法。此方法通常可以长时间运行（持续数秒或数分钟）。因此，我们利用 JavaScript 语言的 `async` 特性，因此在浏览器中运行时，能够以不阻塞主界面线程的方式使用此函数。这与 JavaScript 中其他可能长时间运行的函数类似，例如 `async` [获取](https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API)。请注意，`async` 是一个在 Python 中不存在的构造。Keras 中的 [`fit()`](https://keras.io/models/model/#model-class-api) 方法返回一个 History 对象，而 `fit()` 方法在 JavaScript 中的对应项则返回 History 的 [Promise](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Promise)，这个响应可以[等待](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Operators/await)（如上例中所示），也可与 then() 方法一起使用。

## TensorFlow.js 中没有 NumPy

Python Keras 用户经常使用 [NumPy](http://www.numpy.org/) 来执行基本的数值和数组运算，例如在上例中生成二维张量。

```python
# Python:
xs = np.array([[1], [2], [3], [4]])
```

在 TensorFlow.js 中，这种基本的数值运算是使用软件包本身完成的。例如：

```js
// JavaScript:
const xs = tf.tensor2d([[1], [2], [3], [4]], [4, 1]);
```

<code>tf.*</code> 命名空间还为数组和线性代数运算（例如矩阵乘法）提供了大量其他函数。有关更多信息，请参阅 [TensorFlow.js Core 文档](https://js.tensorflow.org/api/latest/)。

## 使用工厂方法，而不是构造函数

Python 中的这一行（来自上例）是一个构造函数调用：

```python
# Python:
model = keras.Sequential()
```

如果严格转换为 JavaScript，则等效构造函数调用将如下所示：

```js
// JavaScript:
const model = new tf.Sequential();  // !!! DON'T DO THIS !!!
```

不过，我们决定不使用“new”构造函数，因为 1)“new”关键字会使代码更加膨胀；2)“new”构造函数被视为 JavaScript 的“不良部分”：一个潜在的陷阱，如 [*JavaScript: the Good Parts*](http://archive.oreilly.com/pub/a/javascript/excerpts/javascript-good-parts/bad-parts.html) 中所讨论。要在 TensorFlow.js 中创建模型和层，可以调用具有 lowerCamelCase（小驼峰式命名法）名称的工厂方法，例如：

```js
// JavaScript:
const model = tf.sequential();

const layer = tf.layers.batchNormalization({axis: 1});
```

## 选项字符串值为小驼峰式命名法，而不是蛇形命名法

在 JavaScript 中，更常见的是为符号名称使用驼峰命名法（例如，请参阅 [Google JavaScript 样式指南](https://google.github.io/styleguide/jsguide.html#naming-camel-case-defined)），而在 Python 中，蛇形命名法很常见（例如，在 Keras 中）。因此，我们决定使用小驼峰式命名法作为选项的字符串值，包括：

- DataFormat，例如，channelsFirst 而不是 channels_first
- 初始值设定项，例如，**`glorotNormal`** 而不是 `glorot_normal`
- 损失和指标，例如，**`meanSquaredError`** 而不是 `mean_squared_error`，**`categoricalCrossentropy`** 而不是 `categorical_crossentropy`。

例如，如上例所示：

```js
// JavaScript:
model.compile({optimizer: 'sgd', loss: 'meanSquaredError'});
```

对于模型序列化和反序列化，请放心。TensorFlow.js 的内部机制可以确保正确处理 JSON 对象中的蛇形命名法，例如，在从 Python Keras 加载预训练模型时。

## 使用 apply() 运行 Layer 对象，而不是将其作为函数调用

在 Keras 中，Layer 对象定义了 `__call__` 方法。因此，用户可以通过将对象作为函数调用来调用层的逻辑，例如：

```python
# Python:
my_input = keras.Input(shape=[2, 4])
flatten = keras.layers.Flatten()

print(flatten(my_input).shape)
```

这个 Python 语法糖在 TensorFlow.js 中作为 apply() 方法实现：

```js
// JavaScript:
const myInput = tf.input({shape: [2, 4]});
const flatten = tf.layers.flatten();

console.log(flatten.apply(myInput).shape);
```

## Layer.apply() 支持对具体张量进行命令式 (Eager) 执行

目前，在 Keras 中，<strong>调用</strong>方法只能在 (Python) TensorFlow 的 `tf.Tensor` 对象上运行（假设 TensorFlow 是后端），这些对象是符号对象并且不包含实际数值。这就是上一部分中的示例所显示的内容。但是，在 TensorFlow.js 中，层的 `apply()` 方法可以在符号和命令模式下运行。如果使用 SymbolicTensor（类似于 tf.Tensor）调用 `apply()`，返回值将为 SymbolicTensor。这通常发生在模型构建期间。但是，如果使用实际的具体张量值调用 `apply()`，将返回一个具体的张量。例如：

```js
// JavaScript:
const flatten = tf.layers.flatten();

flatten.apply(tf.ones([2, 3, 4])).print();
```

这个特性让人联想到 (Python) TensorFlow 的 [Eager Execution](https://tensorflow.google.cn/guide/eager)。它在模型开发期间提供了更出色的交互性和可调试性，并且为构建动态神经网络打开了大门。

## 优化器在 train. 下，*而不是在 optimizers. 下*

在 Keras 中，Optimizer 对象的构造函数位于 <code>keras.optimizers.*</code> 命名空间下。在 TensorFlow.js Layers 中，Optimizer 的工厂方法位于 <code>tf.train.*</code> 命名空间下。例如：

```python
# Python:
my_sgd = keras.optimizers.sgd(lr=0.2)
```

```js
// JavaScript:
const mySGD = tf.train.sgd({lr: 0.2});
```

## loadLayersModel() 从网址而不是 HDF5 文件加载

在 Keras 中，模型通常[保存](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model)为 HDF5 (.h5) 文件，然后可以使用 `keras.models.load_model()` 方法加载。该方法采用 .h5 文件的路径。`load_model()` 在 TensorFlow.js 中的对应项是 [`tf.loadLayersModel()`](https://js.tensorflow.org/api/latest/#loadLayersModel)。由于 HDF5 文件格式对浏览器并不友好，因此 `tf.loadLayersModel()` 采用 TensorFlow.js 特定的格式。`tf.loadLayersModel()` 将 model.json 文件作为其输入参数。可以使用 tensorflowjs 的 pip 软件包从 Keras HDF5 文件转换 model.json。

```js
// JavaScript:
const model = await tf.loadLayersModel('https://foo.bar/model.json');
```

还要注意，`tf.loadLayersModel()` 返回 [`tf.Model`](https://js.tensorflow.org/api/latest/#class:Model) 的 [`Promise`](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Promise)。

一般来说，在 TensorFlow.js 中分别使用 `tf.Model.save` 和 <code>tf.loadLayersModel</code> 方法保存和加载 `tf.Model`。我们将这些 API 设计为类似于 Keras 的 [save_model 和 load_model API](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model)。但是，浏览器环境与 Keras 等主要深度学习框架运行的后端环境完全不同，特别是在用于持久化和传输数据的路由数组中。因此，TensorFlow.js 和 Keras 中的保存/加载 API 之间存在一些有趣的差异。有关更多详细信息，请参阅我们有关[保存和加载 tf.Model](./save_load.md) 的教程。

## 利用 `fitDataset()` 训练使用 `tf.data.Dataset` 对象的模型

在 Python TensorFlow 的 tf.keras 中，模型可以使用 [Dataset](https://tensorflow.google.cn/guide/datasets) 对象进行训练。模型的 `fit()` 方法直接接受此类对象。TensorFlow.js 模型可以使用 Dataset 对象的 JavaScript 对应项进行训练（请参阅 [TensorFlow.js 中的 tf.data API 文档](https://js.tensorflow.org/api/latest/#Data)。不过，与 Python 不同，基于 Dataset 的训练是通过一个名为 [fitDataset](https://js.tensorflow.org/api/0.15.1/#tf.Model.fitDataset) 的专用方法完成的。[fit()](https://js.tensorflow.org/api/latest/#tf.Model.fitDataset) 方法仅适用于基于张量的模型训练。

## Layer 对象和 Model 对象的内存管理

TensorFlow.js 在浏览器中的 WebGL 上运行，其中 Layer 对象和 Model 对象的权重由 WebGL 纹理支持。不过，WebGL 不支持内置垃圾收集。在推断和训练调用过程中，Layer 对象和 Model 对象为用户在内部管理张量内存。但是，它们也允许用户清理以释放占用的 WebGL 内存。对于在单页加载中创建和释放许多模型实例的情况，这样做很有用。要想清理 Layer 对象或 Model 对象，请使用 `dispose()` 方法。
