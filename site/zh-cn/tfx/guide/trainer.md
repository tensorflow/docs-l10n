# Trainer TFX 流水线组件

Trainer TFX 流水线组件用于训练 TensorFlow 模型。

## Trainer 和 TensorFlow

Trainer 广泛使用 Python [TensorFlow](https://www.tensorflow.org) API 来训练模型。

注：TFX 支持 TensorFlow 1.15 和 2.x。

## 组件

Trainer 需要：

- 用于训练和评估的 tf.Examples。
- 由用户提供、用于定义 Trainer 逻辑的模块文件。
- 由 SchemaGen 流水线组件创建并且可由开发者选择性更改的数据架构。
- 训练参数和评估参数的 [Protobuf](https://developers.google.com/protocol-buffers) 定义。
- （可选）由上游 Transform 组件生成的转换计算图。
- （可选）用于热启动等场景的预训练模型。
- （可选）将传递给用户模块函数的超参数。可以在[此处](tuner.md)找到与 Tuner 集成的详细信息。

Trainer 发出：至少一个用于推断/应用的模型（通常在 SavedModelFormat 中），以及另一个用于评估的可选模型（通常是一个 EvalSavedModel）。

我们通过[模型重写库](https://github.com/tensorflow/tfx/blob/master/tfx/components/trainer/rewriting/README.md)为 [TFLite](https://www.tensorflow.org/lite) 等替代模型格式提供支持。有关如何同时转换 Estimator 和 Keras 模型的示例，请点击模型重写库的链接。

## 基于 Estimator 的 Trainer

要详细了解如何将基于 [Estimator](https://www.tensorflow.org/guide/estimator) 的模型与 TFX 和 Trainer 一起使用，请参阅[使用 tf.Estimator 为 TFX 设计 TensorFlow 建模代码](train.md)。

### 配置 Trainer 组件

典型的流水线 Python DSL 代码如下所示：

```python
from tfx.components import Trainer

...

trainer = Trainer(
      module_file=module_file,
      examples=transform.outputs['transformed_examples'],
      schema=infer_schema.outputs['schema'],
      base_models=latest_model_resolver.outputs['latest_model'],
      transform_graph=transform.outputs['transform_graph'],
      train_args=trainer_pb2.TrainArgs(num_steps=10000),
      eval_args=trainer_pb2.EvalArgs(num_steps=5000))
```

Trainer 调用一个在 `module_file` 参数中指定的训练模块。一个典型的训练模块如下所示：

```python
# TFX will call this function
def trainer_fn(trainer_fn_args, schema):
  """Build the estimator using the high level API.

  Args:
    trainer_fn_args: Holds args used to train the model as name/value pairs.
    schema: Holds the schema of the training examples.

  Returns:
    A dict of the following:
      - estimator: The estimator that will be used for training and eval.
      - train_spec: Spec for training.
      - eval_spec: Spec for eval.
      - eval_input_receiver_fn: Input function for eval.
  """
  # Number of nodes in the first layer of the DNN
  first_dnn_layer_size = 100
  num_dnn_layers = 4
  dnn_decay_factor = 0.7

  train_batch_size = 40
  eval_batch_size = 40

  tf_transform_output = tft.TFTransformOutput(trainer_fn_args.transform_output)

  train_input_fn = lambda: _input_fn(  # pylint: disable=g-long-lambda
      trainer_fn_args.train_files,
      tf_transform_output,
      batch_size=train_batch_size)

  eval_input_fn = lambda: _input_fn(  # pylint: disable=g-long-lambda
      trainer_fn_args.eval_files,
      tf_transform_output,
      batch_size=eval_batch_size)

  train_spec = tf.estimator.TrainSpec(  # pylint: disable=g-long-lambda
      train_input_fn,
      max_steps=trainer_fn_args.train_steps)

  serving_receiver_fn = lambda: _example_serving_receiver_fn(  # pylint: disable=g-long-lambda
      tf_transform_output, schema)

  exporter = tf.estimator.FinalExporter('chicago-taxi', serving_receiver_fn)
  eval_spec = tf.estimator.EvalSpec(
      eval_input_fn,
      steps=trainer_fn_args.eval_steps,
      exporters=[exporter],
      name='chicago-taxi-eval')

  run_config = tf.estimator.RunConfig(
      save_checkpoints_steps=999, keep_checkpoint_max=1)

  run_config = run_config.replace(model_dir=trainer_fn_args.serving_model_dir)
  warm_start_from = trainer_fn_args.base_models[
      0] if trainer_fn_args.base_models else None

  estimator = _build_estimator(
      # Construct layers sizes with exponetial decay
      hidden_units=[
          max(2, int(first_dnn_layer_size * dnn_decay_factor**i))
          for i in range(num_dnn_layers)
      ],
      config=run_config,
      warm_start_from=warm_start_from)

  # Create an input receiver for TFMA processing
  receiver_fn = lambda: _eval_input_receiver_fn(  # pylint: disable=g-long-lambda
      tf_transform_output, schema)

  return {
      'estimator': estimator,
      'train_spec': train_spec,
      'eval_spec': eval_spec,
      'eval_input_receiver_fn': receiver_fn
  }
```

## 通用 Trainer

通用 Trainer 使开发者可以将任何 TensorFlow 模型 API 与 Trainer 组件一起使用。除了 TensorFlow Estimator 外，开发者还可以使用 Keras 模型或自定义训练循环。有关详细信息，请参阅[通用 Trainer 的 RFC](https://github.com/tensorflow/community/blob/master/rfcs/20200117-tfx-generic-trainer.md)。

### 配置 Trainer 组件以使用 GenericExecutor

通用 Trainer 的典型流水线 DSL 代码如下所示：

```python
from tfx.components import Trainer
from tfx.components.base import executor_spec
from tfx.components.trainer.executor import GenericExecutor

...

trainer = Trainer(
    module_file=module_file,
    custom_executor_spec=executor_spec.ExecutorClassSpec(GenericExecutor),
    examples=transform.outputs['transformed_examples'],
    transform_graph=transform.outputs['transform_graph'],
    schema=infer_schema.outputs['schema'],
    train_args=trainer_pb2.TrainArgs(num_steps=10000),
    eval_args=trainer_pb2.EvalArgs(num_steps=5000))
```

Trainer 调用一个在 `module_file` 参数中指定的训练模块。如果在 `custom_executor_spec` 中指定了 `GenericExecutor`，则模块文件中需要 `run_fn` 而不是 `trainer_fn`。

如果流水线中未使用 Transform 组件，则 Trainer 将直接从 ExampleGen 中获取样本：

```python
trainer = Trainer(
    module_file=module_file,
    custom_executor_spec=executor_spec.ExecutorClassSpec(GenericExecutor),
    examples=example_gen.outputs['examples'],
    schema=infer_schema.outputs['schema'],
    train_args=trainer_pb2.TrainArgs(num_steps=10000),
    eval_args=trainer_pb2.EvalArgs(num_steps=5000))
```

这里提供了一个使用 `run_fn` 的[模块文件示例](https://github.com/tensorflow/tfx/blob/master/tfx/examples/iris/iris_utils_native_keras.py)。
