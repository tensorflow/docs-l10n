# Importing a Keras model into TensorFlow.js

Keras models (typically created via the Python API) may be saved in [one of several formats](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model).  The "whole model" format can be converted to TensorFlow.js Layers format, which can be loaded directly into TensorFlow.js for inference or for further training.

The target TensorFlow.js Layers format is a directory containing a `model.json` file and a set of sharded weight files in binary format.  The `model.json` file contains both the model topology (aka "architecture" or "graph": a description of the layers and how they are connected) and a manifest of the weight files.

## Requirements

The conversion procedure requires a Python environment; you may want to keep an isolated one using [pipenv](https://github.com/pypa/pipenv) or [virtualenv](https://virtualenv.pypa.io).  To install the converter, use `pip install tensorflowjs`.

Importing a Keras model into TensorFlow.js is a two-step process. First, convert an existing Keras model to TF.js Layers format, and then load it into TensorFlow.js.

## Step 1. Convert an existing Keras model to TF.js Layers format

Keras models are usually saved via `model.save(filepath)`, which produces a single HDF5 (.h5) file containing both the model topology and the weights.  To convert such a file to TF.js Layers format, run the following command, where _`path/to/my_model.h5`_ is the source Keras .h5 file and _`path/to/tfjs_target_dir`_ is the target output directory for the TF.js files:


```sh
# bash

tensorflowjs_converter --input_format keras \
                       path/to/my_model.h5 \
                       path/to/tfjs_target_dir
```

## Alternative: Use the Python API to export directly to TF.js Layers format

If you have a Keras model in Python, you can export it directly to the TensorFlow.js Layers format as follows:

```py
# Python

import tensorflowjs as tfjs

def train(...):
    model = keras.models.Sequential()   # for example
    ...
    model.compile(...)
    model.fit(...)
    tfjs.converters.save_keras_model(model, tfjs_target_dir)
```

## Step 2: Load the model into TensorFlow.js

Use a web server to serve the converted model files you generated in Step 1.  Note that you may need to configure your server to [allow Cross-Origin Resource Sharing (CORS)](https://enable-cors.org/), in order to allow fetching the files in JavaScript.

Then load the model into TensorFlow.js by providing the URL to the model.json file:

```js
// JavaScript

import * as tf from '@tensorflow/tfjs';

const model = await tf.loadLayersModel('https://foo.bar/tfjs_artifacts/model.json');
```

Now the model is ready for inference, evaluation, or re-training.  For instance, the loaded model can be immediately used to make a prediction:

```js
// JavaScript

const example = tf.fromPixels(webcamElement);  // for example
const prediction = model.predict(example);
```

Many of the [TensorFlow.js Examples](https://github.com/tensorflow/tfjs-examples) take this approach, using pretrained models that have been converted and hosted on Google Cloud Storage.

Note that you refer to the entire model using the `model.json` filename.  `loadModel(...)` fetches `model.json`, and then makes additional HTTP(S) requests to obtain the sharded weight files referenced in the `model.json` weight manifest.  This approach allows all of these files to be cached by the browser (and perhaps by additional caching servers on the internet), because the `model.json` and the weight shards are each smaller than the typical cache file size limit.  Thus a model is likely to load more quickly on subsequent occasions.

## Supported features

TensorFlow.js Layers currently only supports Keras models using standard Keras constructs.
Models using unsupported ops or layers—e.g. custom layers, Lambda layers, custom losses, or custom metrics—cannot be automatically imported, because they depend on Python code that cannot be reliably translated into JavaScript.