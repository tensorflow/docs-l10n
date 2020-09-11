# Model conversion

TensorFlow.js comes with a variety of pre-trained models that are ready to use in the browser - they can be found in our [models repo](https://github.com/tensorflow/tfjs-models). However you may have found or authored a TensorFlow model elsewhere that you’d like to use in your web application. TensorFlow.js provides a model [converter](https://github.com/tensorflow/tfjs/tree/master/tfjs-converter) for this purpose. The TensorFlow.js converter has two components:

1. A command line utility that converts Keras and TensorFlow models for use in TensorFlow.js.
2. An API for loading and executing the model in the browser with TensorFlow.js.

## Convert your model

The TensorFlow.js converter works with several different model formats:

**SavedModel**: This is the default format in which TensorFlow models are saved. The SavedModel format is documented [here](https://www.tensorflow.org/guide/saved_model).

**Keras model**: Keras models are generally saved as an HDF5 file. More information about saving Keras models can be found [here](https://keras.io/getting-started/faq/#savingloading-whole-models-architecture-weights-optimizer-state).

**TensorFlow Hub module**: These are models that have been packaged for distribution on TensorFlow Hub, a platform for sharing and discovering models. The model library can be found [here](https://tfhub.dev/).

Depending on which type of model you’re trying to convert, you’ll need to pass different arguments to the converter. For example, let’s say you have saved a Keras model named `model.h5` to your `tmp/` directory. To convert your model using the TensorFlow.js converter, you can run the following command:

    $ tensorflowjs_converter --input_format=keras /tmp/model.h5 /tmp/tfjs_model

This will convert the model at `/tmp/model.h5` and output a `model.json` file along with binary weight files to your `tmp/tfjs_model/` directory.

More details about the command line arguments corresponding to different model formats can be found at the TensorFlow.js converter [README](https://github.com/tensorflow/tfjs/tree/master/tfjs-converter).

During the conversion process we traverse the model graph and check that each operation is supported by TensorFlow.js. If so, we write the graph into a format that the browser can consume. We try to optimize the model for being served on the web by sharding the weights into 4MB files - that way they can be cached by browsers. We also attempt to simplify the model graph itself using the open source [Grappler](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/core/grappler) project. Graph simplifications include folding together adjacent operations, eliminating common subgraphs, etc. These changes have no effect on the model’s output. For further optimization, users can pass in an argument that instructs the converter to quantize the model to a certain byte size. Quantization is a technique for reducing model size by representing weights with fewer bits. Users must be careful to ensure that their model maintains an acceptable degree of accuracy after quantization.

If we encounter an unsupported operation during conversion, the process fails and we print out the name of the operation for the user. Feel free to submit an issue on our [GitHub](https://github.com/tensorflow/tfjs/issues) to let us know about it - we try to implement new operations in response to user demand.

### Best practices

Although we make every effort to optimize your model during conversion, often the best way to ensure your model performs well is to build it with resource-constrained environments in mind. This means avoiding overly complex architectures and minimizing the number of parameters (weights) when possible.

## Run your model

Upon successfully converting your model, you’ll end up with a set of weight files and a model topology file. TensorFlow.js provides model loading APIs that you can use to fetch these model assets and run inference in the browser.

Here’s what the API looks like for a converted TensorFlow SavedModel or TensorFlow Hub module:

```js
const model = await tf.loadGraphModel(‘path/to/model.json’);
```

And here’s what it looks like for a converted Keras model:

```js
const model = await tf.loadLayersModel(‘path/to/model.json’);
```

The `tf.loadGraphModel` API returns a `tf.FrozenModel`, which means that the parameters are fixed and you will not be able to fine tune your model with new data. The `tf.loadLayersModel` API returns a tf.Model, which can be trained. For information on how to train a tf.Model, refer to the [train models guide](train_models.md).

After conversion, it’s a good idea to run inference a few times and benchmark the speed of your model. We have a standalone benchmarking page that can be used for this purpose: https://tensorflow.github.io/tfjs/e2e/benchmarks/local-benchmark/index.html. You may notice that we discard measurements from an initial warmup run - this is because (in general) your model’s first inference will be several times slower than subsequent inferences due to the overhead of creating textures and compiling shaders.



