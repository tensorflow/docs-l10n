# 用于移动设备的 TFX

## 简介

本指南演示如何使用 TensorFlow Extended (TFX) 创建和评估在设备端部署的机器学习模型。现在，TFX 为 [TFLite](https://www.tensorflow.org/lite) 提供原生支持，这使得在移动设备上执行高效推断成为可能。

本指南将引导您对任何流水线进行更改以生成和评估 TFLite 模型。我们在[此处](https://github.com/tensorflow/tfx/blob/master/tfx/examples/mnist/mnist_pipeline_native_keras.py)提供了一个完整示例，演示 TFX 如何训练和评估使用 [MNIST](http://yann.lecun.com/exdb/mnist/) 数据集训练的 TFLite 模型。此外，我们还将展示如何使用同一个流水线同时导出标准的基于 Keras 的 [SavedModel](https://www.tensorflow.org/guide/saved_model) 和 TFLite SavedModel，使用户能够比较二者的质量。

我们假设您熟悉 TFX、相应组件和流水线。如果您不熟悉这些内容，请参阅此[教程](https://www.tensorflow.org/tfx/tutorials/tfx/components)。

## 步骤

在 TFX 中创建和评估 TFLite 模型只需两个步骤。第一步是在 [TFX Trainer](https://www.tensorflow.org/tfx/guide/trainer) 上下文中调用 TFLite 重写器，将训练的 TensorFlow 模型转换为 TFLite 模型。第二步是配置 Evaluator 以评估 TFLite 模型。现在，我们依次讨论这两个步骤。

### 在 Trainer 中调用 TFLite 重写器

TFX Trainer 要求在模块文件中指定用户定义的 `run_fn`。此 `run_fn` 定义要训练的模型，对其进行指定迭代次数的训练，并导出训练后的模型。

我们将在本节的其余部分提供代码段，展示调用 TFLite 重写器和导出 TFLite 模型所需的更改。所有这些代码都位于 [MNIST TFLite 模块](https://github.com/tensorflow/tfx/blob/master/tfx/examples/mnist/mnist_utils_native_keras_lite.py)的 `run_fn` 中。

如以下代码所示，我们必须首先创建一个签名，该签名会为每个特征接受 `Tensor` 作为输入。请注意，这与 TFX 中的大多数现有模型不同，现有模型接受序列化的 [tf.Example](https://www.tensorflow.org/api_docs/python/tf/train/Example) proto 作为输入。

```python
 signatures = {
      'serving_default':
          _get_serve_tf_examples_fn(
              model, tf_transform_output).get_concrete_function(
                  tf.TensorSpec(
                      shape=[None, 784],
                      dtype=tf.float32,
                      name='image_floats'))
  }
```

然后，使用与平时相同的方式将 Keras 模型保存为 SavedModel。

```python
  temp_saving_model_dir = os.path.join(fn_args.serving_model_dir, 'temp')
  model.save(temp_saving_model_dir, save_format='tf', signatures=signatures)
```

最后，我们创建 TFLite 重写器 (`tfrw`) 的实例，并在 SavedModel 上调用它以获取 TFLite 模型。我们将此 TFLite 模型存储在 `run_fn` 的调用者提供的 `serving_model_dir` 中。这样，TFLite 模型就会存储在所有下游 TFX 组件预期查找模型的位置。

```python
  tfrw = rewriter_factory.create_rewriter(
      rewriter_factory.TFLITE_REWRITER, name='tflite_rewriter')
  converters.rewrite_saved_model(temp_saving_model_dir,
                                 fn_args.serving_model_dir,
                                 tfrw,
                                 rewriter.ModelType.TFLITE_MODEL)
```

### 评估 TFLite 模型

[TFX Evaluator](https://www.tensorflow.org/tfx/guide/evaluator) 可以分析训练的模型，以了解模型在各种指标上的质量。除了分析 SavedModel 之外，TFX Evaluator 现在还可以分析 TFLite 模型。

以下代码段（从 [MNIST 流水线](https://github.com/tensorflow/tfx/blob/master/tfx/examples/mnist/mnist_pipeline_native_keras.py)复制）展示了如何配置分析 TFLite 模型的 Evaluator。

```python
  # Informs the evaluator that the model is a TFLite model.
  eval_config_lite.model_specs[0].model_type = 'tf_lite'

  ...

  # Uses TFMA to compute the evaluation statistics over features of a TFLite
  # model.
  model_analyzer_lite = Evaluator(
      examples=example_gen.outputs['examples'],
      model=trainer_lite.outputs['model'],
      eval_config=eval_config_lite,
      instance_name='mnist_lite')
```

如上所示，我们只需将 `model_type` 字段更改为 `tf_lite`。无需其他配置更改即可分析 TFLite 模型。无论是分析 TFLite 模型还是 SavedModel，`Evaluator` 的输出都将具有完全相同的结构。
