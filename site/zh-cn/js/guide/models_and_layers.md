# 模型和层

在机器学习中，*模型*是一个带有*可学习*[参数](https://developers.google.com/machine-learning/glossary/#parameter)的函数，可将输入映射至输出。通过在数据上训练模型获得最佳参数。训练好的模型可以提供从输入到所需输出的准确映射。

在 TensorFlow.js 中，您可以通过两种方式创建机器学习模型：

1. 使用 Layers API（使用*层*构建模型）
2. 使用 Core API（借助低级运算，例如 `tf.matMul()`、`tf.add()` 等）

首先，我们会了解 Layers API，Layers API 是用于构建模型的高级 API。然后，我们将演示如何使用 Core API 构建相同的模型。

## 使用 Layers API 创建模型

你可以通过两种方式使用 Layers API 创建模型：*序贯*模型和*函数式*模型。下面两部分将详细介绍两种类型。

### 序贯模型

最常见的模型是 <code>&lt;a href="https://js.tensorflow.org/api/0.15.1/#class:Sequential" data-md-type="link"&gt;Sequential&lt;/a&gt;</code> 模型，序贯模型是层的线性堆叠。您可以通过将层列表传递到 <code>&lt;a href="https://js.tensorflow.org/api/0.15.1/#sequential" data-md-type="link"&gt;sequential()&lt;/a&gt;</code> 函数来创建 <code>Sequential</code> 模型：

```js
const model = tf.sequential({
 layers: [
   tf.layers.dense({inputShape: [784], units: 32, activation: 'relu'}),
   tf.layers.dense({units: 10, activation: 'softmax'}),
 ]
});
```

或通过 `add()` 方法：

```js
const model = tf.sequential();
model.add(tf.layers.dense({inputShape: [784], units: 32, activation: 'relu'}));
model.add(tf.layers.dense({units: 10, activation: 'softmax'}));
```

> 重要提示：模型的第一层需要 `inputShape`。提供 `inputShape` 时请确保排除批次大小。例如，创建模型时，如果您计划馈送形状为 `[B, 784]`（其中 `B` 可为任何批次大小）的模型张量，请将 `inputShape` 指定为 `[784]`。

您可以通过 `model.layers` 访问模型的层，更具体而言为 `model.inputLayers` 和 `model.outputLayers`。

### 函数式模型

创建 `LayersModel` 的另一种方式是通过 `tf.model()` 函数。`tf.model()` 和 `tf.sequential()` 的主要区别为，`tf.model()` 可用于创建层的任意计算图，前提是层没有循环。

以下代码段可以使用 `tf.model()` API 定义与上文相同的模型：

```js
// Create an arbitrary graph of layers, by connecting them
// via the apply() method.
const input = tf.input({shape: [784]});
const dense1 = tf.layers.dense({units: 32, activation: 'relu'}).apply(input);
const dense2 = tf.layers.dense({units: 10, activation: 'softmax'}).apply(dense1);
const model = tf.model({inputs: input, outputs: dense2});
```

我们在每一层调用 `apply()` 以将其连接到另一个层的输出。在这种情况下，`apply()` 的结果是一个 `SymbolicTensor`，后者类似于 `Tensor`，但不包含任何具体值。

请注意，与序贯模型不同，我们通过 `tf.input()` 创建 `SymbolicTensor`，而非向第一层提供 `inputShape`。

如果您向 `apply()` 传递一个具体 `Tensor`，它也会为您提供一个具体 `Tensor`：

```js
const t = tf.tensor([-2, 1, 0, 5]);
const o = tf.layers.activation({activation: 'relu'}).apply(t);
o.print(); // [0, 1, 0, 5]
```

这对于单独测试层并查看它们的输出非常有用。

与在序贯模型中一样，您可以通过 `model.layers` 访问模型的层，更具体而言为 `model.inputLayers` 和 `model.outputLayers`。

## 验证

序贯模型和函数式模型都是 `LayersModel` 类的实例。使用 `LayersModels` 的一个主要优势是验证：它会强制您指定输入形状，并稍后将其用于验证您的输入。`LayersModel` 还会在数据流经层时自动推断形状。提前了解形状后，模型就可以自动创建它的参数，并告知您两个相邻的层是否相互兼容。

## 模型摘要

调用 `model.summary()` 以打印模型的实用摘要，其中包括：

- 模型中所有层的名称和类型
- 每个层的输出形状
- 每个层的权重参数数量
- 每个层接收的输入（如果模型具有一般拓扑，下文将讨论）
- 模型的可训练和不可训练参数总数

对于上面定义的模型，我们在控制台上获取以下输出：

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

注意层的输出形状中的 `null` 值：这表示模型希望输入的批次大小为最外层维度，在这种情况下，由于 `null` 值，批次大小比较灵活。

## 序列化

在较低级别的 API 上使用 `LayersModel` 的一个主要优势是能够保存和加载模型。`LayersModel` 了解：

- 模型的架构，让您可以创新创建模型
- 模型的权重
- 训练配置（损失、优化器和指标）
- 优化器的状态，让您可以恢复训练

保存或加载模型只需要 1 行代码：

```js
const saveResult = await model.save('localstorage://my-model-1');
const model = await tf.loadLayersModel('localstorage://my-model-1');
```

上面的示例可将模型保存到浏览器的本地存储空间中。请参阅 <code>&lt;a href="https://js.tensorflow.org/api/latest/#tf.Model.save" data-md-type="link"&gt;model.save() 文档&lt;/a&gt;</code>和[保存并加载](save_load.md)指南，了解如何保存到不同的媒介（例如，文件存储空间、<code>IndexedDB</code>、触发浏览器下载等）。

## 自定义层

层是模型的基本要素。如果您的模型需要进行自定义计算，您可以定义一个自定义层，它可以与层的其他部分很好地交互。我们在下面定义的自定义层可以计算正方形总数：

```js
class SquaredSumLayer extends tf.layers.Layer {
 constructor() {
   super({});
 }
 // In this case, the output is a scalar.
 computeOutputShape(inputShape) { return []; }

 // call() is where we do the computation.
 call(input, kwargs) { return input.square().sum();}

 // Every layer needs a unique name.
 getClassName() { return 'SquaredSum'; }
}
```

要对其进行测试，我们可以调用包含具体张量的 `apply()` 方法：

```js
const t = tf.tensor([-2, 1, 0, 5]);
const o = new SquaredSumLayer().apply(t);
o.print(); // prints 30
```

> 重要提示：如果添加自定义层，将无法序列化模型。

## 使用 Core API 创建模型

在本指南开头处，我们提到可以通过两种方式在 TensorFlow.js 中创建机器学习模型。

一般来说，您始终应当先尝试使用 Layers API，因为它基于被广泛使用的 Keras API，后者[遵循最佳做法并降低了认知负担](https://keras.io/why-use-keras/)。Layers API 还提供了各种现成的解决方案，如权重初始化、模型序列化、训练监视、概率和安全检查。

在以下情况下，您可能需要使用 Core API：

- 您需要最大程度的灵活性和控制
- 您不需要序列化或可以实现自己的序列化逻辑

使用 Core API 创建的模型是以一个或多个 `Tensor` 作为输入并输出 `Tensor` 的函数。使用 Core API 编写的上面同一个模型如下所示：

```js
// The weights and biases for the two dense layers.
const w1 = tf.variable(tf.randomNormal([784, 32]));
const b1 = tf.variable(tf.randomNormal([32]));
const w2 = tf.variable(tf.randomNormal([32, 10]));
const b2 = tf.variable(tf.randomNormal([10]));

function model(x) {
  return x.matMul(w1).add(b1).relu().matMul(w2).add(b2).softmax();
}
```

请注意，在 Core API 中，我们需要创建和初始化模型的权重。每个权重都由一个 `Variable` 支持，变量可以告知 TensorFlow.js 这些张量是可学习张量。您可以使用 [tf.variable()](https://js.tensorflow.org/api/latest/#variable) 并传入现有 `Tensor` 来创建 `Variable`。

本文介绍了如何使用 Layers API 和 Core API 创建模型。接下来，请参阅[训练模型](train_models.md)指南了解如何训练模型。
