# TensorFlow.js layers API for Keras users

The Layers API of TensorFlow.js is modeled after Keras and we strive to make the
[Layers API](https://js.tensorflow.org/api/latest/) as
similar to Keras as reasonable given the differences between JavaScript and
Python. This makes it easier for users with experience developing Keras models
in Python to migrate to TensorFlow.js Layers in JavaScript. For example, the
following Keras code translates into JavaScript:

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
import * as tf from '@tensorflow/tfjs';

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

However, there are some differences we’d like to call out and explain in this
document. Once you understand these differences and the rationale behind them,
your Python-to-JavaScript migration (or migration in the reverse direction)
should be a relatively smooth experience.

## Constructors take JavaScript Objects as configurations

Compare the following Python and JavaScript lines from the example above: they
both create a [Dense](https://keras.io/api/layers/core_layers/dense) layer.

```python
# Python:
keras.layers.Dense(units=1, inputShape=[1])
```

```js
// JavaScript:
tf.layers.dense({units: 1, inputShape: [1]});
```

JavaScript functions don’t have an equivalent of the keyword arguments in Python
functions. We want to avoid implementing constructor options as positional
arguments in JavaScript, which would be especially cumbersome to remember and
use for constructors with a large number of keyword arguments (e.g.,
[LSTM](https://keras.io/api/layers/recurrent_layers/lstm)). This
is why we use JavaScript configuration objects. Such objects provide the same
level of positional invariance and flexibility as Python keyword arguments.

Some methods of the Model class, e.g.,
[`Model.compile()`](https://keras.io/models/model/#model-class-api), also take a
JavaScript configuration object as the input. However, keep in mind that
`Model.fit()`, `Model.evaluate()` and `Model.predict()` are slightly different.
Since these method take obligatory `x` (features) and `y` (labels or targets)
data as inputs; `x` and `y` are positional arguments separate from the ensuing
configuration object that plays the role of the keyword arguments. For example:

```js
// JavaScript:
await model.fit(xs, ys, {epochs: 1000});
```

## Model.fit() is async

`Model.fit()` is the primary method with which users perform model training in
TensorFlow.js. This method can often be long-running, lasting for seconds or
minutes. Therefore, we utilize the `async` feature of the JavaScript language,
so that this function can be used in a way that does not block the main UI
thread when running in the browser.
This is similar to other potentially long-running functions in JavaScript, such
as the `async`
[fetch](https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API).
Note that `async` is a construct that does not exist in Python. While the
[`fit()`](https://keras.io/models/model/#model-class-api)
method in Keras returns a History object, the counterpart of the `fit()` method
in JavaScript returns a
[Promise](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Promise)
of History, which can be
[await](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Operators/await)ed
(as in the example above) or used with the then() method.

## No NumPy for TensorFlow.js

Python Keras users often use [NumPy](http://www.numpy.org/) to perform basic
numeric and array operations, such as generating 2D tensors in the example
above.

```python
# Python:
xs = np.array([[1], [2], [3], [4]])
```

In TensorFlow.js, this kind of basic numeric operations are done with the
package itself. For example:

```js
// JavaScript:
const xs = tf.tensor2d([[1], [2], [3], [4]], [4, 1]);
```

The `tf.*` namespace also provides a number of other functions for array and
linear algebra operations such as matrix multiplication. See the
[TensorFlow.js Core documentation](https://js.tensorflow.org/api/latest/) for more
information.

## Use factory methods, not constructors

This line in Python (from the example above) is a constructor call:

```python
# Python:
model = keras.Sequential()
```

If translated strictly into JavaScript, the equivalent constructor call would
look like the following:

```js
// JavaScript:
const model = new tf.Sequential();  // !!! DON'T DO THIS !!!
```

However, we decided not to use the “new” constructors because 1) the “new”
keyword would make the code more bloated and 2) the “new” constructor is
regarded as a “bad part” of JavaScript: a potential pitfall, as is argued in
[*JavaScript: the Good Parts*](https://www.oreilly.com/library/view/javascript-the-good/9780596517748/).
To create models and layers in TensorFlow.js, you call factory methods, which
have lowerCamelCase names, for example:

```js
// JavaScript:
const model = tf.sequential();

const layer = tf.layers.batchNormalization({axis: 1});
```

## Option string values are lowerCamelCase, not snake_case

In JavaScript, it is more common to use camel case for symbol names (e.g., see
[Google JavaScript Style Guide](https://google.github.io/styleguide/jsguide.html#naming-camel-case-defined)),
as compared with Python, where snake case is common (e.g., in Keras). As such,
we decided to use lowerCamelCase for string values for options including the
following:

- DataFormat, e.g., **`channelsFirst`** instead of `channels_first`
- Initializer, e.g., **`glorotNormal`** instead of `glorot_normal`
- Loss and metrics, e.g., **`meanSquaredError`** instead of
  `mean_squared_error`, **`categoricalCrossentropy`** instead of
  `categorical_crossentropy`.

For example, as in the example above:

```js
// JavaScript:
model.compile({optimizer: 'sgd', loss: 'meanSquaredError'});
```

With regard to model serialization and deserialization, rest assured.
TensorFlow.js’s internal mechanism ensure that snake cases in JSON objects are
handled correctly, e.g., when loading pretrained models from Python Keras.

## Run Layer objects with apply(), not by calling them as functions

In Keras, a Layer object has the `__call__` method defined. Therefore the user can
invoke the layer’s logic by calling the object as a function, e.g.,

```python
# Python:
my_input = keras.Input(shape=[2, 4])
flatten = keras.layers.Flatten()

print(flatten(my_input).shape)
```

This Python syntax sugar is implemented as the apply() method in TensorFlow.js:

```js
// JavaScript:
const myInput = tf.input({shape: [2, 4]});
const flatten = tf.layers.flatten();

console.log(flatten.apply(myInput).shape);
```

## Layer.apply() supports imperative (eager) evaluation on concrete tensors

Currently, in Keras, the __call__ method can only operate on (Python)
TensorFlow’s `tf.Tensor`
objects (assuming TensorFlow backend), which are symbolic and do not hold actual
numeric values. This is what’s shown in the example in the previous section.
However, in TensorFlow.js, the apply() method of layers can operate in both
symbolic and imperative modes. If `apply()` is invoked with a SymbolicTensor (a
close analogy of tf.Tensor), the return value will be a SymbolicTensor. This
happens typically during model building. But if `apply()` is invoked with an
actual concrete Tensor value, it will return a concrete Tensor. For example:

```js
// JavaScript:
const flatten = tf.layers.flatten();

flatten.apply(tf.ones([2, 3, 4])).print();
```

This feature is reminiscent of (Python) TensorFlow’s
[Eager Execution](https://www.tensorflow.org/guide/eager).
It affords greater interactivity and debuggability during model development, in
addition to opening doors to composing dynamic neural networks.

## Optimizers are under train.*, not optimizers.*

In Keras, the constructors for Optimizer objects are under the
`keras.optimizers.*` namespace. In TensorFlow.js Layers, the factory methods for
Optimizers are under the `tf.train.*` namespace. For example:

```python
# Python:
my_sgd = keras.optimizers.sgd(lr=0.2)
```

```js
// JavaScript:
const mySGD = tf.train.sgd({lr: 0.2});
```

## loadLayersModel() loads from a URL, not an HDF5 file

In Keras, models are usually [saved](https://keras.io/getting_started/faq/#what-are-my-options-for-saving-models)
as a HDF5 (.h5) file, which can be later loaded using the
`keras.models.load_model()` method. The method takes a path to the .h5 file. The
counterpart of `load_model()` in TensorFlow.js is
[`tf.loadLayersModel()`](https://js.tensorflow.org/api/latest/#loadLayersModel). Since HDF5 is not a
browser-friendly file format, `tf.loadLayersModel()` takes a TensorFlow.js-specific
format. `tf.loadLayersModel()` takes a model.json file as its input argument. The
model.json can be converted from a Keras HDF5 file using the tensorflowjs pip
package.

```js
// JavaScript:
const model = await tf.loadLayersModel('https://foo.bar/model.json');
```

Also note that `tf.loadLayersModel()` returns a
[`Promise`](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Promise)
of [`tf.Model`](https://js.tensorflow.org/api/latest/#class:Model).

In general, saving and loading `tf.Model`s in TensorFlow.js is done using the
`tf.Model.save` and `tf.loadLayersModel` methods, respectively. We designed these APIs to be similar to
[the save and load_model API](https://keras.io/getting_started/faq/#what-are-my-options-for-saving-models)
of Keras. But the browser environment is quite different from the backend environment
on which staple deep learning frameworks like Keras run, particularly in the
array of routes for persisting and transmitting data. Hence there are
some interesting differences between the save/load APIs in TensorFlow.js and in Keras.
See our tutorial on [Saving and Loading tf.Model](./save_load.md) for more
details.

## Use `fitDataset()` to train models using `tf.data.Dataset` objects

In Python TensorFlow's tf.keras, a model can be trained using a
[Dataset](https://www.tensorflow.org/guide/datasets) object. The model's `fit()`
method accepts such an object directly. A TensorFlow.js model can be trained
with the JavaScript equivalent of the Dataset objects as well (see
[the documentation of the tf.data API in TensorFlow.js](https://js.tensorflow.org/api/latest/#Data)).
However, unlike in Python, Dataset-based training is done through a dedicated
method, namely [fitDataset](https://js.tensorflow.org/api/0.15.1/#tf.Model.fitDataset).
The [fit()](https://js.tensorflow.org/api/latest/#tf.Model.fitDataset) method is
only for tensor-based model training.

## Memory management of Layer and Model objects

TensorFlow.js runs on WebGL in the browser, where the weights of Layer and Model
objects are backed by WebGL textures. However, WebGL has no built-in garbage-collection
support. Layer and Model objects internally manage tensor memory for the user
during their inference and training calls. But they also allows the user to
dispose them in order to free the WebGL memory that they occupy. This is useful
in cases where many model instances are created and released within a single
page load. To dispose a Layer or Model object, use the `dispose()` method.
