# TFX 中的 TensorFlow 2.x

[TensorFlow 2.0 于 2019 发布](https://blog.tensorflow.org/2019/09/tensorflow-20-is-now-available.html)，默认与 [Keras](https://www.tensorflow.org/guide/keras/overview) 和 [Eager Execution](https://www.tensorflow.org/guide/eager) 紧密集成，还有 [Python 风格的函数执行](https://www.tensorflow.org/guide/function)，以及其他[新功能和改进](https://www.tensorflow.org/guide/effective_tf2#a_brief_summary_of_major_changes)。

本指南提供了 TFX 中 TF 2.x 的全面技术概述。

## 选择版本

TFX 与 TensorFlow 2.x 兼容，而且 TensorFlow 1.x 中的高级 API（特别是 Estimator）也可以继续运行。

### 在 TensorFlow 2.x 中启动新项目

由于 TensorFlow 2.x 保留了 TensorFlow 1.x 的高级功能，因此即使您不打算使用新功能，在新项目上使用旧版本也没有优势。

因此，如果您正准备启动一个新的 TFX 项目，我们建议您使用 TensorFlow2.x。稍后全面支持 Keras 和其他新功能后，您可能需要更新代码，如果您从 TensorFlow 2.x 开始而不是将来尝试从 TensorFlow 1.x 升级，需要改动的范围将小得多。

### 将现有项目转换为 TensorFlow 2.x

针对 TensorFlow 1.x 编写的代码在很大程度上与 TensorFlow 2.x 兼容，并可在 TFX 中继续运行。

但是，如果想利用 TF 2.x 中提供的改进和新功能，您可以按照[迁移到 TF 2.x 的说明](https://www.tensorflow.org/guide/migrate)进行操作。

## Estimator

TensorFlow 2.x 中保留了 Estimator API，但它不是新功能和开发的重点。使用 Estimator 在 TensorFlow 1.x 或 2.x 中编写的代码将在 TFX 中继续按预期运行。

下面是一个使用纯 Estimator 的端到端 TFX 示例：[Taxi 示例 (Estimator)](https://github.com/tensorflow/tfx/blob/r0.21/tfx/examples/chicago_taxi_pipeline/taxi_utils.py)

## 带有 `model_to_estimator` 的 Keras

Keras 模型可以使用 `tf.keras.estimator.model_to_estimator` 函数进行封装，该函数可以使它们像 Estimator 一样工作。要使用该函数，请完成以下操作：

1. 构建 Keras 模型。
2. 将编译后的模型传递给 `model_to_estimator`。
3. 以您通常使用 Estimator 的方式在 Trainer 中使用 `model_to_estimator` 的结果。

```py
# Build a Keras model.
def _keras_model_builder():
  """Creates a Keras model."""
  ...

  model = tf.keras.Model(inputs=inputs, outputs=output)
  model.compile()

  return model


# Write a typical trainer function
def trainer_fn(trainer_fn_args, schema):
  """Build the estimator, using model_to_estimator."""
  ...

  # Model to estimator
  estimator = tf.keras.estimator.model_to_estimator(
      keras_model=_keras_model_builder(), config=run_config)

  return {
      'estimator': estimator,
      ...
  }
```

除了 Trainer 的用户模块文件外，流水线的其他部分保持不变。

## 原生 Keras（即不带 `model_to_estimator` 的 Keras）

注：对 Keras 中所有功能的全面支持正在开发。在大多数情况下，TFX 中的 Keras 将按预期运行。它尚不适用于 FeatureColumns 的稀疏特征。

### 示例和 Colab

下面是使用原生 Keras 的几个示例：

- [Penguin](https://github.com/tensorflow/tfx/blob/master/tfx/examples/penguin/penguin_pipeline_local.py)（[模块文件](https://github.com/tensorflow/tfx/blob/master/tfx/examples/penguin/penguin_utils.py)）：“Hello world”端到端示例。
- [MNIST](https://github.com/tensorflow/tfx/blob/master/tfx/examples/mnist/mnist_pipeline_native_keras.py)（[模块文件](https://github.com/tensorflow/tfx/blob/master/tfx/examples/mnist/mnist_utils_native_keras.py)）：图像和 TFLite 端到端示例。
- [Taxi](https://github.com/tensorflow/tfx/blob/master/tfx/examples/chicago_taxi_pipeline/taxi_pipeline_native_keras.py)（[模块文件](https://github.com/tensorflow/tfx/blob/master/tfx/examples/chicago_taxi_pipeline/taxi_utils_native_keras.py)）：具有高级 Transform 用法的端到端示例。

我们还有按组件的 [Keras Colab](https://www.tensorflow.org/tfx/tutorials/tfx/components_keras)。

### TFX 组件

以下各个部分说明了相关 TFX 组件如何支持原生 Keras。

#### Transform

Transform 目前为 Keras 模型提供实验性支持。

Transform 组件本身可直接用于原生 Keras，而无需更改。`preprocessing_fn` 定义保持不变，使用 [TensorFlow](https://www.tensorflow.org/api_docs/python/tf) 和 [tf.Transform](https://www.tensorflow.org/tfx/transform/api_docs/python/tft) 运算。

针对原生 Keras 更改了应用函数和评估函数。详细信息将在下面的 Trainer 和 Evaluator 部分中讨论。

注：`preprocessing_fn` 中的转换不能应用于训练或评估的标签特征。

#### Trainer

要配置原生 Keras，需要为 Trainer 组件设置 `GenericExecutor` 以替换默认的基于 Estimator 的执行器。有关详细信息，请查看[此处](trainer.md#configuring-the-trainer-component-to-use-the-genericexecutor)。

##### 带有 Transform 的 Keras Module 文件

训练模块文件必须包含 `run_fn`，后者将由 `GenericExecutor` 调用。典型的 Keras `run_fn` 如下所示：

```python
def run_fn(fn_args: TrainerFnArgs):
  """Train the model based on given args.

  Args:
    fn_args: Holds args used to train the model as name/value pairs.
  """
  tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

  # Train and eval files contains transformed examples.
  # _input_fn read dataset based on transformed schema from tft.
  train_dataset = _input_fn(fn_args.train_files, fn_args.data_accessor,
                            tf_transform_output.transformed_metadata.schema)
  eval_dataset = _input_fn(fn_args.eval_files, fn_args.data_accessor,
                           tf_transform_output.transformed_metadata.schema)

  model = _build_keras_model()

  model.fit(
      train_dataset,
      steps_per_epoch=fn_args.train_steps,
      validation_data=eval_dataset,
      validation_steps=fn_args.eval_steps)

  signatures = {
      'serving_default':
          _get_serve_tf_examples_fn(model,
                                    tf_transform_output).get_concrete_function(
                                        tf.TensorSpec(
                                            shape=[None],
                                            dtype=tf.string,
                                            name='examples')),
  }
  model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)
```

在上述 `run_fn` 中，导出训练后的模型时需要提供应用签名，以便模型可以获取原始样本进行预测。典型的应用函数如下所示：

```python
def _get_serve_tf_examples_fn(model, tf_transform_output):
  """Returns a function that parses a serialized tf.Example."""

  # the layer is added as an attribute to the model in order to make sure that
  # the model assets are handled correctly when exporting.
  model.tft_layer = tf_transform_output.transform_features_layer()

  @tf.function
  def serve_tf_examples_fn(serialized_tf_examples):
    """Returns the output to be used in the serving signature."""
    feature_spec = tf_transform_output.raw_feature_spec()
    feature_spec.pop(_LABEL_KEY)
    parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)

    transformed_features = model.tft_layer(parsed_features)

    return model(transformed_features)

  return serve_tf_examples_fn
```

在上述应用函数中，需要使用 [`tft.TransformFeaturesLayer`](https://github.com/tensorflow/transform/blob/master/docs/api_docs/python/tft/TransformFeaturesLayer.md) 层将 tf.Transform 转换应用于原始数据以进行推断。使用 Keras 时将不再需要 Estimator 所需的先前的 `_serving_input_receiver_fn`。

##### 不带 Transform 的 Keras Module 文件

这类似于上面展示的模块文件，但没有转换：

```python
def _get_serve_tf_examples_fn(model, schema):

  @tf.function
  def serve_tf_examples_fn(serialized_tf_examples):
    feature_spec = _get_raw_feature_spec(schema)
    feature_spec.pop(_LABEL_KEY)
    parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
    return model(parsed_features)

  return serve_tf_examples_fn


def run_fn(fn_args: TrainerFnArgs):
  schema = io_utils.parse_pbtxt_file(fn_args.schema_file, schema_pb2.Schema())

  # Train and eval files contains raw examples.
  # _input_fn reads the dataset based on raw data schema.
  train_dataset = _input_fn(fn_args.train_files, fn_args.data_accessor, schema)
  eval_dataset = _input_fn(fn_args.eval_files, fn_args.data_accessor, schema)

  model = _build_keras_model()

  model.fit(
      train_dataset,
      steps_per_epoch=fn_args.train_steps,
      validation_data=eval_dataset,
      validation_steps=fn_args.eval_steps)

  signatures = {
      'serving_default':
          _get_serve_tf_examples_fn(model, schema).get_concrete_function(
              tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')),
  }
  model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)
```

#####

[tf.distribute.Strategy](https://www.tensorflow.org/guide/distributed_training)

目前，TFX 仅支持单个工作进程策略（例如，[MirroredStrategy](https://www.tensorflow.org/api_docs/python/tf/distribute/MirroredStrategy) 和 [OneDeviceStrategy](https://www.tensorflow.org/api_docs/python/tf/distribute/OneDeviceStrategy)）。

要使用分布策略，请创建适当的 tf.distribute.Strategy 并将 Keras 模型的创建和编译移动到策略作用域内。

例如，将上面的 `model = _build_keras_model()` 替换为：

```python
  mirrored_strategy = tf.distribute.MirroredStrategy()
  with mirrored_strategy.scope():
    model = _build_keras_model()

  # Rest of the code can be unchanged.
  model.fit(...)
```

要验证由 `MirroredStrategy` 使用的设备 (CPU/GPU)，请启用信息级别的 TensorFlow 日志记录：

```python
import logging
logging.getLogger("tensorflow").setLevel(logging.INFO)
```

并且您应该能够在日志中看到 `Using MirroredStrategy with devices (...)`。

注：对于 GPU 内存不足问题，可能需要环境变量 `TF_FORCE_GPU_ALLOW_GROWTH=true`。有关详细信息，请参阅 [TensorFlow GPU 指南](https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth)。

#### Evaluator

在 TFMA v0.2x 中，ModelValidator 和 Evaluator 已合并成一个[新的 Evaluator 组件](https://github.com/tensorflow/community/blob/master/rfcs/20200117-tfx-combining-model-validator-with-evaluator.md)。与之前的模型相比，新的 Evaluator 组件不仅可以执行单个模型评估，还可以验证当前模型。进行此更改后，Pusher 组件现在可以使用来自 Evaluator 而非 ModelValidator 的祝福结果。

新的 Evaluator 支持 Keras 模型和 Estimator 模型。使用 Keras 时将不再需要先前需要的 `_eval_input_receiver_fn` 和已保存的评估模型，因为 Evaluator 现在基于用于应用的相同 `SavedModel`。

[如需了解详细信息，请参阅 Evaluator](evaluator.md)。
