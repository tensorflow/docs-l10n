# Importing a TensorFlow GraphDef based Models into TensorFlow.js

TensorFlow GraphDef based models (typically created via the Python API) may be saved in one of following formats:
1. TensorFlow [SavedModel](https://www.tensorflow.org/tutorials/keras/save_and_load)
2. Frozen Model
3. [Tensorflow Hub module](https://www.tensorflow.org/hub/)

All of the above formats can be converted by the [TensorFlow.js converter](https://github.com/tensorflow/tfjs/tree/master/tfjs-converter) into a format that can be loaded directly into TensorFlow.js for inference.

(Note: TensorFlow has deprecated the session bundle format, please migrate your models to the SavedModel format.)

## Requirements

The conversion procedure requires a Python environment; you may want to keep an isolated one using [pipenv](https://github.com/pypa/pipenv) or [virtualenv](https://virtualenv.pypa.io).  To install the converter, run the following command:

```bash
 pip install tensorflowjs
```

Importing a TensorFlow model into TensorFlow.js is a two-step process. First, convert an existing model to the TensorFlow.js web format, and then load it into TensorFlow.js.

## Step 1. Convert an existing TensorFlow model to the TensorFlow.js web format

Run the converter script provided by the pip package:

Usage:
SavedModel example:

```bash
tensorflowjs_converter \
    --input_format=tf_saved_model \
    --output_node_names='MobilenetV1/Predictions/Reshape_1' \
    --saved_model_tags=serve \
    /mobilenet/saved_model \
    /mobilenet/web_model
```

Frozen model example:

```bash
tensorflowjs_converter \
    --input_format=tf_frozen_model \
    --output_node_names='MobilenetV1/Predictions/Reshape_1' \
    /mobilenet/frozen_model.pb \
    /mobilenet/web_model
```

Tensorflow Hub module example:

```bash
tensorflowjs_converter \
    --input_format=tf_hub \
    'https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/classification/1' \
    /mobilenet/web_model
```

|Positional Arguments | Description |
|---|---|
|`input_path`  | Full path of the saved model directory, session bundle directory, frozen model file or TensorFlow Hub module handle or path.|
|`output_path` | Path for all output artifacts.|

| Options | Description
|---|---|
|`--input_format`     | The format of input model, use tf_saved_model for SavedModel, tf_frozen_model for frozen model, tf_session_bundle for session bundle, tf_hub for TensorFlow Hub module and keras for Keras HDF5. |
|`--output_node_names`| The names of the output nodes, separated by commas.|
|`--saved_model_tags` | Only applicable to SavedModel conversion, Tags of the MetaGraphDef to load, in comma separated format. Defaults to `serve`.|
|`--signature_name`   | Only applicable to TensorFlow Hub module conversion, signature to load. Defaults to `default`. See https://www.tensorflow.org/hub/common_signatures/.|

Use following command to get a detailed help message:

```bash
tensorflowjs_converter --help
```

### Converter generated files

The conversion script above produces two types of files:

* `model.json` (the dataflow graph and weight manifest)
* `group1-shard\*of\*` (collection of binary weight files)

For example, here is the output from converting MobileNet v2:

```html
  output_directory/model.json
  output_directory/group1-shard1of5
  ...
  output_directory/group1-shard5of5
```

## Step 2: Loading and running in the browser

1. Install the tfjs-converter npm package

`yarn add @tensorflow/tfjs` or `npm install @tensorflow/tfjs`

2. Instantiate the [FrozenModel class](https://github.com/tensorflow/tfjs-converter/blob/master/src/executor/frozen_model.ts) and run inference.

```js
import * as tf from '@tensorflow/tfjs';
import {loadGraphModel} from '@tensorflow/tfjs-converter';

const MODEL_URL = 'model_directory/model.json';

const model = await loadGraphModel(MODEL_URL);
const cat = document.getElementById('cat');
model.execute(tf.browser.fromPixels(cat));
```

Check out our [MobileNet demo](https://github.com/tensorflow/tfjs/tree/master/tfjs-converter/demo/mobilenet).

The `loadGraphModel` API accepts an additional `LoadOptions` parameter, which can be used to send credentials or custom headers along with the request. Please see the [loadGraphModel() documentation](https://js.tensorflow.org/api/1.0.0/#loadGraphModel) for more details.

## Supported operations

Currently TensorFlow.js supports a limited set of TensorFlow ops. If your model uses an unsupported op, the `tensorflowjs_converter` script will fail and print out a list of the unsupported ops in your model. Please file an [issue](https://github.com/tensorflow/tfjs/issues) for each op to let us know which ops you need support for.

## Loading the weights only

If you prefer to load the weights only, you can use the following code snippet.

```js
import * as tf from '@tensorflow/tfjs';

const weightManifestUrl = "https://example.org/model/weights_manifest.json";

const manifest = await fetch(weightManifestUrl);
this.weightManifest = await manifest.json();
const weightMap = await tf.io.loadWeights(
        this.weightManifest, "https://example.org/model");
