# Trainer TFX 流水线组件

Trainer TFX 流水线组件用于训练 TensorFlow 模型。

## Trainer 和 TensorFlow

Trainer 广泛使用 Python [TensorFlow](https://www.tensorflow.org) API 来训练模型。

注：TFX 支持 TensorFlow 1.15 和 2.x。

## 组件

Trainer 需要：

- 用于训练和评估的 tf.Examples。
- 由用户提供、用于定义 Trainer 逻辑的模块文件。
- 训练参数和评估参数的 [Protobuf](https://developers.google.com/protocol-buffers) 定义。
- （可选）由 SchemaGen 流水线组件创建并且可由开发者选择性更改的数据架构。
- （可选）由上游 Transform 组件生成的转换计算图。
- （可选）用于热启动等场景的预训练模型。
- （可选）将传递给用户模块函数的超参数。可以在[此处](tuner.md)找到与 Tuner 集成的详细信息。

Trainer 发出：至少一个用于推断/应用的模型（通常在 SavedModelFormat 中），以及另一个用于评估的可选模型（通常是一个 EvalSavedModel）。

我们通过[模型重写库](https://github.com/tensorflow/tfx/blob/master/tfx/components/trainer/rewriting/README.md)为 [TFLite](https://www.tensorflow.org/lite) 等替代模型格式提供支持。有关如何同时转换 Estimator 和 Keras 模型的示例，请点击模型重写库的链接。

## 通用 Trainer

通用 Trainer 使开发者可以将任何 TensorFlow 模型 API 与 Trainer 组件一起使用。除了 TensorFlow Estimator 外，开发者还可以使用 Keras 模型或自定义训练循环。有关详细信息，请参阅[通用 Trainer 的 RFC](https://github.com/tensorflow/community/blob/master/rfcs/20200117-tfx-generic-trainer.md)。

### 配置 Trainer 组件

通用 Trainer 的典型流水线 DSL 代码如下所示：

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

Trainer 调用一个训练模块，该模块在 `module_file` 参数中指定。如果在 `custom_executor_spec` 中指定了 `GenericExecutor`， 则模块文件中需要 `run_fn`，而不是 `trainer_fn`。`trainer_fn` 负责创建模型。除此之外，`run_fn` 还需要处理训练部分并将训练后的模型输出到 [FnArgs](https://github.com/tensorflow/tfx/blob/master/tfx/components/trainer/fn_args_utils.py) 给出的指定位置：

```python
from tfx.components.trainer.fn_args_utils import FnArgs

def run_fn(fn_args: FnArgs) -> None:
  """Build the TF model and train it."""
  model = _build_keras_model()
  model.fit(...)
  # Save model to fn_args.serving_model_dir.
  model.save(fn_args.serving_model_dir, ...)
```

这是一个带有 `run_fn` 的[示例模块文件](https://github.com/tensorflow/tfx/blob/master/tfx/examples/penguin/penguin_utils_keras.py)。

请注意，如果流水线中未使用 Transform 组件，则 Trainer 将直接从 ExampleGen 中获取样本：

```python
trainer = Trainer(
    module_file=module_file,
    examples=example_gen.outputs['examples'],
    schema=infer_schema.outputs['schema'],
    train_args=trainer_pb2.TrainArgs(num_steps=10000),
    eval_args=trainer_pb2.EvalArgs(num_steps=5000))
```

有关更多详细信息，请参阅 [Trainer API 参考](https://www.tensorflow.org/tfx/api_docs/python/tfx/v1/components/Trainer)。
