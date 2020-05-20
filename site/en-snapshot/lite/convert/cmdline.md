# Converter command line reference

This page describes how to use the [TensorFlow Lite converter](index.md) using
the command line tool. However, the [Python API](python_api.md) is recommended
for the majority of cases.

Note: This only contains documentation on the command line tool in TensorFlow 2.
Documentation on using the command line tool in TensorFlow 1 is available on
GitHub
([reference](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/r1/convert/cmdline_reference.md),
[example](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/r1/convert/cmdline_examples.md)).

## High-level overview

The TensorFlow Lite Converter has a command line tool named `tflite_convert`,
which supports basic models. Use the [Python API](python_api.md) for any
conversions involving optimizations, or any additional parameters (e.g.
signatures in [SavedModels](https://www.tensorflow.org/guide/saved_model) or
custom objects in
[Keras models](https://www.tensorflow.org/guide/keras/overview)).

## Usage

The following example shows a SavedModel being converted:

```bash
tflite_convert \
  --saved_model_dir=/tmp/mobilenet_saved_model \
  --output_file=/tmp/mobilenet.tflite
```

The inputs and outputs are specified using the following commonly used flags:

*   `--output_file`. Type: string. Specifies the full path of the output file.
*   `--saved_model_dir`. Type: string. Specifies the full path to the directory
    containing the SavedModel generated in 1.X or 2.X.
*   `--keras_model_file`. Type: string. Specifies the full path of the HDF5 file
    containing the `tf.keras` model generated in 1.X or 2.X.

To use all of the available flags, use the following command:

```bash
tflite_convert --help
```

The following flag can be used for compatibility with the TensorFlow 1.X version
of the converter CLI:

*   `--enable_v1_converter`. Type: bool. Enables user to enable the 1.X command
    line flags instead of the 2.X flags. The 1.X command line flags are
    specified
    [here](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/r1/convert/cmdline_reference.md).

## Installing the converter CLI

To obtain the latest version of the TensorFlow Lite converter CLI, we recommend
installing the nightly build using
[pip](https://www.tensorflow.org/install/pip):

```bash
pip install tf-nightly
```

Alternatively, you can
[clone the TensorFlow repository](https://www.tensorflow.org/install/source) and
use `bazel` to run the command:

```
bazel run //tensorflow/lite/python:tflite_convert -- \
  --saved_model_dir=/tmp/mobilenet_saved_model \
  --output_file=/tmp/mobilenet.tflite
```

### Custom ops in the new converter

There is a behavior change in how models containing
[custom ops](https://www.tensorflow.org/lite/guide/ops_custom) (those for which
users use to set --allow\_custom\_ops before) are handled in the
[new converter](https://github.com/tensorflow/tensorflow/blob/917ebfe5fc1dfacf8eedcc746b7989bafc9588ef/tensorflow/lite/python/lite.py#L81).

**Built-in TensorFlow op**

If you are converting a model with a built-in TensorFlow op that does not exist
in TensorFlow Lite, you should set --allow\_custom\_ops argument (same as
before), explained [here](https://www.tensorflow.org/lite/guide/ops_custom).

**Custom op in TensorFlow**

If you are converting a model with a custom TensorFlow op, it is recommended
that you write a [TensorFlow kernel](https://www.tensorflow.org/guide/create_op)
and [TensorFlow Lite kernel](https://www.tensorflow.org/lite/guide/ops_custom).
This ensures that the model is working end-to-end, from TensorFlow and
TensorFlow Lite. This also requires setting the --allow\_custom\_ops argument.

**Advanced custom op usage (not recommended)**

If the above is not possible, you can still convert a TensorFlow model
containing a custom op without a corresponding kernel. You will need to pass the
[OpDef](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/op_def.proto)
of the custom op in TensorFlow using --custom\_opdefs flag, as long as you have
the corresponding OpDef registered in the TensorFlow global op registry. This
ensures that the TensorFlow model is valid (i.e. loadable by the TensorFlow
runtime).

If the custom op is not part of the global TensorFlow op registry, then the
corresponding OpDef needs to be specified via the --custom\_opdefs flag. This is
a list of an OpDef proto in string that needs to be additionally registered.
Below is an example of an TFLiteAwesomeCustomOp with 2 inputs, 1 output, and 2
attributes:

```
--custom\_opdefs="name: 'TFLiteAwesomeCustomOp' input\_arg: { name: 'InputA'
type: DT\_FLOAT } input\_arg: { name: ‘InputB' type: DT\_FLOAT }
output\_arg: { name: 'Output' type: DT\_FLOAT } attr : { name: 'Attr1' type:
'float'} attr : { name: 'Attr2' type: 'list(float)'}"
```
