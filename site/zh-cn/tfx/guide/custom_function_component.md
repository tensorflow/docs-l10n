# 自定义 Python 函数组件

通过使用基于 Python 函数的组件定义，您可以省去定义组件规范类、执行器类和组件接口类的工作量，更轻松地创建 TFX 自定义组件。您需要在此组件定义样式中编写一个使用类型提示注解的函数。类型提示描述了组件的输入工件、输出工件和形参。

如以下示例所示，以此样式编写自定义组件非常简单。

```python
class MyOutput(TypedDict):
  accuracy: float

@component
def MyValidationComponent(
    model: InputArtifact[Model],
    blessing: OutputArtifact[Model],
    accuracy_threshold: Parameter[int] = 10,
) -> MyOutput:
  '''My simple custom model validation component.'''

  accuracy = evaluate_model(model)
  if accuracy >= accuracy_threshold:
    write_output_blessing(blessing)

  return {
    'accuracy': accuracy
  }
```

在底层，这定义了一个自定义组件，该组件是 [`BaseComponent`](https://github.com/tensorflow/tfx/blob/master/tfx/dsl/components/base/base_component.py) {: .external } 及其 Spec 和 Executor 类的子类。

注：下面描述的功能（基于 BaseBeamComponent 的组件，使用 `@component(use_beam=True)` 来注释函数）为实验性功能，没有公开的向后兼容性保证。

如果您想定义 [`BaseBeamComponent`](https://github.com/tensorflow/tfx/blob/master/tfx/dsl/components/base/base_beam_component.py) {: .external } 的子类，以便能够在编译流水线（[芝加哥出租车流水线示例](https://github.com/tensorflow/tfx/blob/master/tfx/examples/chicago_taxi_pipeline/taxi_pipeline_simple.py#L192){: .external }）时使用具有 TFX-pipeline-wise 共享配置的 Beam 流水线，即 `beam_pipeline_args`，则您可以在装饰器中设置 `use_beam=True` 并在函数中添加另一个默认值为 `None` 的 `BeamComponentParameter`，如下例所示：

```python
@component(use_beam=True)
def MyDataProcessor(
    examples: InputArtifact[Example],
    processed_examples: OutputArtifact[Example],
    beam_pipeline: BeamComponentParameter[beam.Pipeline] = None,
    ) -> None:
  '''My simple custom model validation component.'''

  with beam_pipeline as p:
    # data pipeline definition with beam_pipeline begins
    ...
    # data pipeline definition with beam_pipeline ends
```

如果您不熟悉 TFX 流水线，请[详细了解 TFX 流水线的核心概念](understanding_tfx_pipelines)。

## 输入、输出和形参

在 TFX 中，会将输入和输出作为 Artifact 对象进行跟踪，这些对象描述底层数据的位置以及与其相关的元数据属性。此信息存储在 ML Metadata 中。工件可以描述复杂或简单的数据类型，例如：int、float、byte 或 unicode 字符串。

形参是流水线构造时已知组件的实参（int、float、byte 或 unicode 字符串）。形参对于指定实参和超参数很有用，例如训练迭代计数、随机失活率以及组件的其他配置。在 ML Metadata 中进行跟踪时，形参会被存储为组件执行的属性。

注：目前，输出简单数据类型值不能用作形参，因为它们在执行时未知。同样，输入简单数据类型值目前无法接受在流水线构造时已知的具体值。我们可能会在未来版本的 TFX 中移除此限制。

## 定义

要创建自定义组件，请编写一个实现自定义逻辑的函数，并使用 `tfx.dsl.component.experimental.decorators` 模块中的 [`@component` 装饰器](https://github.com/tensorflow/tfx/blob/master/tfx/dsl/component/experimental/decorators.py){: .external } 对它进行装饰。要定义组件的输入和输出架构，请使用 `tfx.dsl.component.experimental.annotations` 模块中的注解对函数的实参进行注解并返回值：

- 对于每个**工件输入**，应用 `InputArtifact[ArtifactType]` 类型提示注解。将 `ArtifactType` 替换为工件的类型，该类型是 `tfx.types.Artifact` 的子类。这些输入可以是可选实参。

- 对于每个**输出工件**，应用 `OutputArtifact[ArtifactType]` 类型提示注解。将 `ArtifactType` 替换为工件的类型，该类型是 `tfx.types.Artifact` 的子类。组件输出工件应作为函数的输入实参传递，以便您的组件可以将输出写入系统管理的位置并设置适当的工件元数据属性。此实参可以是可选实参，也可以使用默认值进行定义。

- 对于每个**形参**，请使用类型提示注解 `Parameter[T]`。将 `T` 替换为形参的类型。我们目前仅支持原始 Python 类型：`bool`、`int`、`float`、`str` 或 `bytes`。

- 对于 **Beam 流水线**，使用类型提示注释 `BeamComponentParameter[beam.Pipeline]`。将默认值设置为 `None`。值 `None` 将由 <a><code>BaseBeamExecutor</code></a> {: .external } 的 `_make_beam_pipeline()` 创建的实例化 Beam 流水线所取代

- 对于每个在流水线构造时未知的**简单数据类型输入**（`int`、`float`、`str` 或 `bytes`），请使用类型提示 `T`。请注意，在 TFX 0.22 版本中，无法在流水线构造时为此类型的输入传递具体值（如前一个部分中所述，请使用 `Parameter` 注解）。此实参可以是可选实参，也可以使用默认值进行定义。如果您的组件具有简单数据类型输出（`int`、`float`、`str` 或 `bytes`），您可以使用 `OutputDict` 实例返回这些输出。将 <code>OutputDict</code> 类型提示应用为组件的返回值。

在函数体中，输入和输出工件会作为 `tfx.types.Artifact` 对象传递；您可以通过检查其 `.uri` 获得其系统管理的位置并读取/设置任何属性。输入形参和简单数据类型输入会作为指定类型的对象传递。简单数据类型输出应作为字典返回，其中键是适当的输出名称，值是所需的返回值。

完成后的函数组件如下所示：

```python
from typing import TypedDict
import tfx.v1 as tfx
from tfx.dsl.component.experimental.decorators import component

class MyOutput(TypedDict):
  loss: float
  accuracy: float

@component
def MyTrainerComponent(
    training_data: tfx.dsl.components.InputArtifact[tfx.types.standard_artifacts.Examples],
    model: tfx.dsl.components.OutputArtifact[tfx.types.standard_artifacts.Model],
    dropout_hyperparameter: float,
    num_iterations: tfx.dsl.components.Parameter[int] = 10
) -> MyOutput:
  '''My simple trainer component.'''

  records = read_examples(training_data.uri)
  model_obj = train_model(records, num_iterations, dropout_hyperparameter)
  model_obj.write_to(model.uri)

  return {
    'loss': model_obj.loss,
    'accuracy': model_obj.accuracy
  }

# Example usage in a pipeline graph definition:
# ...
trainer = MyTrainerComponent(
    examples=example_gen.outputs['examples'],
    dropout_hyperparameter=other_component.outputs['dropout'],
    num_iterations=1000)
pusher = Pusher(model=trainer.outputs['model'])
# ...
```

前面的示例将 `MyTrainerComponent` 定义为基于 Python 函数的自定义组件。该组件使用 `examples` 工件作为其输入，并生成 `model` 工件作为其输出。该组件使用 `artifact_instance.uri` 在其系统管理的位置读取或写入工件。该组件接受 `num_iterations` 输入形参和 `dropout_hyperparameter` 简单数据类型值，并将 `loss` 和 `accuracy` 指标输出为简单数据类型输出值。然后，`Pusher` 组件将使用输出 `model` 工件。
