# 张量和运算

TensorFlow.js 是一种框架，用于以 JavaScript 定义和运行使用张量的计算。*张量*是向量和矩阵向更高维度的泛化。

## 张量

TensorFlow.js 中数据的中央单元为 `tf.Tensor`：一组形状为一维或多维数组的值。`tf.Tensor` 与多维数组非常相似。

`tf.Tensor` 还包含以下属性：

- `rank`：定义张量包含的维数
- `shape`：定义数据每个维度的大小
- `dtype`：定义张量的数据类型

注：我们会将“维度”一词与秩互换使用。有时在机器学习中，张量的“维数”也可以指特定维度的大小（例如，形状为 [10, 5] 的矩阵为 2 秩张量或二维张量。第一维的维数为 10。这可能会造成混淆，但由于您可能会遇到该术语的这两种说法，因此我们在此提供了此注释）。

可以使用 `tf.tensor()` 方法从数组创建 `tf.Tensor`：

```js
// Create a rank-2 tensor (matrix) matrix tensor from a multidimensional array.
const a = tf.tensor([[1, 2], [3, 4]]);
console.log('shape:', a.shape);
a.print();

// Or you can create a tensor from a flat array and specify a shape.
const shape = [2, 2];
const b = tf.tensor([1, 2, 3, 4], shape);
console.log('shape:', b.shape);
b.print();
```

默认情况下，`tf.Tensor` 将具有 `float32` `dtype`。也可以使用 bool、int32、complex64 和字符串数据类型创建 `tf.Tensor`：

```js
const a = tf.tensor([[1, 2], [3, 4]], [2, 2], 'int32');
console.log('shape:', a.shape);
console.log('dtype', a.dtype);
a.print();
```

TensorFlow.js 还提供了一组便捷的方法，用于创建随机张量、填充特定值的张量、`HTMLImageElement` 中的张量，以及[此处](https://js.tensorflow.org/api/latest/#Tensors-Creation)所列的更多张量。

#### 更改张量的形状

`tf.Tensor` 中元素的数量是其形状大小的乘积。由于通常可以有多个具有相同大小的形状，因此将 `tf.Tensor` 重塑为具有相同大小的其他形状通常非常实用。这可以通过 `reshape()` 方法实现：

```js
const a = tf.tensor([[1, 2], [3, 4]]);
console.log('a shape:', a.shape);
a.print();

const b = a.reshape([4, 1]);
console.log('b shape:', b.shape);
b.print();
```

#### 从张量获取值

您还可以使用 `Tensor.array()` 或 `Tensor.data()` 方法从 `tf.Tensor` 中获取值：

```js
 const a = tf.tensor([[1, 2], [3, 4]]);
 // Returns the multi dimensional array of values.
 a.array().then(array => console.log(array));
 // Returns the flattened data that backs the tensor.
 a.data().then(data => console.log(data));
```

我们还提供了这些方法的同步版本，这些版本更易于使用，但会在您的应用中引起性能问题。在生产应用中，您应始终优先使用异步方法。

```js
const a = tf.tensor([[1, 2], [3, 4]]);
// Returns the multi dimensional array of values.
console.log(a.arraySync());
// Returns the flattened data that backs the tensor.
console.log(a.dataSync());
```

## 运算

张量可用于存储数据，而运算则可用于操作该数据。TensorFlow.js 还提供了可对张量执行的适用于线性代数和机器学习的多种运算。

示例：对 `tf.Tensor` 中的所有元素执行 x<sup>2</sup> 计算：

```js
const x = tf.tensor([1, 2, 3, 4]);
const y = x.square();  // equivalent to tf.square(x)
y.print();
```

示例：对两个 `tf.Tensor` 的元素执行逐元素相加：

```js
const a = tf.tensor([1, 2, 3, 4]);
const b = tf.tensor([10, 20, 30, 40]);
const y = a.add(b);  // equivalent to tf.add(a, b)
y.print();
```

由于张量是不可变的，因此这些运算不会改变其值。相反，return 运算总会返回新的 `tf.Tensor`。

> 注：大多数运算都会返回 `tf.Tensor`，但结果实际上可能并未准备就绪。这意味着您获得的 `tf.Tensor` 实际上是计算的句柄。当您调用 `Tensor.data()` 或 `Tensor.array()` 时，这些方法将返回仅在计算完成时才解析值的 promise。在界面上下文（例如浏览器应用）中运行时，应始终首选这些方法的异步版本而非同步版本，以免在计算完成之前阻塞界面线程。

您可以在[此处](https://js.tensorflow.org/api/latest/#Operations)找到 TensorFlow.js 所支持运算的列表。

## 内存

使用 WebGL 后端时，必须显式管理 `tf.Tensor` 内存（即使 `tf.Tensor` 超出范围也**不足以**释放其内存）。

要销毁 tf.Tensor 的内存，您可以使用 `dispose()` 方法或 `tf.dispose()`：

```js
const a = tf.tensor([[1, 2], [3, 4]]);
a.dispose(); // Equivalent to tf.dispose(a)
```

在应用中将多个运算链接在一起十分常见。保持对用于处置这些运算的所有中间变量的引用会降低代码的可读性。为了解决这个问题，TensorFlow.js 提供了 `tf.tidy()` 方法，可清理执行函数后未被该函数返回的所有 `tf.Tensor`，类似于执行函数时清理局部变量的方式：

```js
const a = tf.tensor([[1, 2], [3, 4]]);
const y = tf.tidy(() => {
  const result = a.square().log().neg();
  return result;
});
```

在此示例中，`square()` 和 `log()` 的结果将被自动处置。`neg()` 的结果不会被处置，因为它是 tf.tidy() 的返回值。

您还可以获取 TensorFlow.js 跟踪的张量数量：

```js
console.log(tf.memory());
```

`tf.memory()` 打印的对象将包含有关当前分配了多少内存的信息。您可以在[此处](https://js.tensorflow.org/api/latest/#memory)查找更多信息。
