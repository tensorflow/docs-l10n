# Tensors and operations

TensorFlow.js is a framework to define and run computations using tensors in JavaScript. A *tensor* is a generalization of vectors and matrices to higher dimensions.

## Tensors

The central unit of data in TensorFlow.js is the `tf.Tensor`: a set of values shaped into an array of one or more dimensions. `tf.Tensor`s are very similar to multidimensional arrays.

A `tf.Tensor` also contains the following properties:

*   `rank`: defines how many dimensions the tensor contains
*   `shape`: which defines the size of each dimension of the data
*   `dtype`: which defines the data type of the tensor.

Note: We will use the term "dimension" interchangeably with the rank. Sometimes in machine learning, "dimensionality" of a tensor can also refer to the size of a particular dimension (e.g. a matrix of shape [10, 5] is a rank-2 tensor, or a 2-dimensional tensor. The dimensionality of the first dimension is 10. This can be confusing, but we put this note here because you will likely come across these dual uses of the term).

A `tf.Tensor` can be created from an array with the `tf.tensor()` method:


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


By default, `tf.Tensor`s will have a `float32` `dtype.` `tf.Tensor`s can also be created with bool, int32, complex64, and string dtypes:


```js
const a = tf.tensor([[1, 2], [3, 4]], [2, 2], 'int32');
console.log('shape:', a.shape);
console.log('dtype', a.dtype);
a.print();
```


TensorFlow.js also provides a set of convenience methods for creating random tensors, tensors filled with a particular value, tensors from `HTMLImageElement`s, and many more which you can find [here](https://js.tensorflow.org/api/latest/#Tensors-Creation).


#### Changing the shape of a Tensor

The number of elements in a `tf.Tensor` is the product of the sizes in its shape. Since often times there can be multiple shapes with the same size, it's often useful to be able to reshape a `tf.Tensor` to another shape with the same size. This can be achieved with the `reshape()` method:


```js
const a = tf.tensor([[1, 2], [3, 4]]);
console.log('a shape:', a.shape);
a.print();

const b = a.reshape([4, 1]);
console.log('b shape:', b.shape);
b.print();
```



#### Getting values from a Tensor

You can also get the values from a `tf.Tensor` using the `Tensor.array()` or `Tensor.data()` methods:


```js
 const a = tf.tensor([[1, 2], [3, 4]]);
 // Returns the multi dimensional array of values.
 a.array().then(array => console.log(array));
 // Returns the flattened data that backs the tensor.
 a.data().then(data => console.log(data));
```


We also provide synchronous versions of these methods which are simpler to use, but will cause performance issues in your application. You should always prefer the asynchronous methods in production applications.


```js
const a = tf.tensor([[1, 2], [3, 4]]);
// Returns the multi dimensional array of values.
console.log(a.arraySync());
// Returns the flattened data that backs the tensor.
console.log(a.dataSync());
```



## Operations

While tensors allow you to store data, operations (ops) allow you to manipulate that data. TensorFlow.js also provides a wide variety of ops suitable for linear algebra and machine learning that can be performed on tensors.

Example: computing x<sup>2</sup> of all elements in a `tf.Tensor`:


```js
const x = tf.tensor([1, 2, 3, 4]);
const y = x.square();  // equivalent to tf.square(x)
y.print();
```


Example: adding elements of two `tf.Tensor`s element-wise:


```js
const a = tf.tensor([1, 2, 3, 4]);
const b = tf.tensor([10, 20, 30, 40]);
const y = a.add(b);  // equivalent to tf.add(a, b)
y.print();
```


Because tensors are immutable, these ops do not change their values. Instead, ops return always return new `tf.Tensor`s.

> Note: most operations return `tf.Tensor`s, however the result may not actually be ready yet. This means the `tf.Tensor` that you get is actually a handle to the computation. When you call `Tensor.data()` or `Tensor.array()`, these methods return promises that resolve with values only when computation is finished. When running in a UI context (such as browser app), you should always prefer the asynchronous versions of these methods instead of their synchronous counterparts to avoid blocking the UI thread until the computation completes.

You can find a list of the operations TensorFlow.js supports [here](https://js.tensorflow.org/api/latest/#Operations).


## Memory

When using the WebGL backend, `tf.Tensor` memory must be managed explicitly (it is **not sufficient** to let a `tf.Tensor` go out of scope for its memory to be released).

To destroy the memory of a tf.Tensor, you can use the `dispose() `method or `tf.dispose()`:


```js
const a = tf.tensor([[1, 2], [3, 4]]);
a.dispose(); // Equivalent to tf.dispose(a)
```


It is very common to chain multiple operations together in an application. Holding a reference to all of the intermediate variables to dispose them can reduce code readability. To solve this problem, TensorFlow.js provides a `tf.tidy()` method which cleans up all `tf.Tensor`s that are not returned by a function after executing it, similar to the way local variables are cleaned up when a function is executed:


```js
const a = tf.tensor([[1, 2], [3, 4]]);
const y = tf.tidy(() => {
  const result = a.square().log().neg();
  return result;
});
```


In this example, the result of `square()` and `log()` will automatically be disposed. The result of `neg()` will not be disposed as it is the return value of the tf.tidy().

You can also get the number of Tensors tracked by TensorFlow.js:


```js
console.log(tf.memory());
```


The object printed by `tf.memory()` will contain information about how much memory is currently allocated. You can find more information [here](https://js.tensorflow.org/api/latest/#memory).
