# 构建完全自定义的组件

本指南介绍如何使用 TFX API 构建完全自定义的组件。完全自定义的组件使您可以通过定义组件规范、执行器和组件接口类来构建组件。您可以通过这种方式重用和扩展标准组件以满足您的需求。

如果您不熟悉 TFX 流水线，请[详细了解 TFX 流水线的核心概念](understanding_tfx_pipelines)。

## 自定义执行器或自定义组件

如果仅需要自定义处理逻辑，而组件的输入、输出和执行属性与现有组件相同，那么自定义执行器就足够了。当输入、输出或执行属性与任何现有 TFX 组件不同时，都需要完全自定义的组件。

## 如何创建自定义组件？

开发完全自定义的组件需要以下内容：

- 为新组件定义的一组输入和输出工件规范。特别地，输入工件的类型应与生成工件的组件的输出工件类型一致，并且输出工件的类型应与使用工件的组件（如果存在）的输入工件类型一致。
- 新组件所需的非工件执行参数。

### ComponentSpec

`ComponentSpec` 类通过定义组件的输入和输出工件以及用于组件执行的参数来定义组件协定。其中包括三个部分：

- *INPUTS*：进入组件执行器的输入工件的类型化参数字典。通常输入工件是来自上游组件的输出，因此共享同一类型。
- *OUTPUTS*：组件生成的输出工件的类型化参数字典。
- *PARAMETERS*：其他 [ExecutionParameter](https://github.com/tensorflow/tfx/blob/54aa6fbec6bffafa8352fe51b11251b1e44a2bf1/tfx/types/component_spec.py#L274) 项的字典，这些项将被传递到组件执行器。这些是我们希望在流水线 DSL 中灵活定义并传递给执行的非工件参数。

下面是一个 ComponentSpec 示例：

```python
class HelloComponentSpec(types.ComponentSpec):
  """ComponentSpec for Custom TFX Hello World Component."""

  PARAMETERS = {
      # These are parameters that will be passed in the call to
      # create an instance of this component.
      'name': ExecutionParameter(type=Text),
  }
  INPUTS = {
      # This will be a dictionary with input artifacts, including URIs
      'input_data': ChannelParameter(type=standard_artifacts.Examples),
  }
  OUTPUTS = {
      # This will be a dictionary which this component will populate
      'output_data': ChannelParameter(type=standard_artifacts.Examples),
  }
```

### 执行器

接下来，为新组件编写执行器代码。基本上，需要创建一个新的 `base_executor.BaseExecutor` 子类，并重写其 `Do` 函数。在 `Do` 函数中，传入的参数 `input_dict`、`output_dict` 和 `exec_properties` 会分别映射到在 ComponentSpec 中定义的 `INPUTS`、`OUTPUTS` 和 `PARAMETERS`。对于 `exec_properties`，可以通过字典查找直接提取该值。对于 `input_dict` 和 `output_dict` 中的工件，[artifact_utils](https://github.com/tensorflow/tfx/blob/41823f91dbdcb93195225a538968a80ba4bb1f55/tfx/types/artifact_utils.py) 类中提供了方便的函数，可用于提取工件实例或工件 URI。

```python
class Executor(base_executor.BaseExecutor):
  """Executor for HelloComponent."""

  def Do(self, input_dict: Dict[Text, List[types.Artifact]],
         output_dict: Dict[Text, List[types.Artifact]],
         exec_properties: Dict[Text, Any]) -> None:
    ...

    split_to_instance = {}
    for artifact in input_dict['input_data']:
      for split in json.loads(artifact.split_names):
        uri = artifact_utils.get_split_uri([artifact], split)
        split_to_instance[split] = uri

    for split, instance in split_to_instance.items():
      input_dir = instance
      output_dir = artifact_utils.get_split_uri(
          output_dict['output_data'], split)
      for filename in tf.io.gfile.listdir(input_dir):
        input_uri = os.path.join(input_dir, filename)
        output_uri = os.path.join(output_dir, filename)
        io_utils.copy_file(src=input_uri, dst=output_uri, overwrite=True)
```

#### 对自定义执行器进行单元测试

可以创建与[此](https://github.com/tensorflow/tfx/blob/r0.15/tfx/components/transform/executor_test.py)类似的自定义执行器单元测试。

### 组件接口

现在，最复杂的部分已经完成，下一步是将这些部件组装到组件接口中，以使组件能够在流水线中使用。包括几个步骤：

- 使组件接口成为 `base_component.BaseComponent` 的子类
- 使用先前定义的 `ComponentSpec` 类分配一个类变量 `SPEC_CLASS`
- 使用先前定义的 Executor 类分配一个类变量 `EXECUTOR_SPEC`
- 定义 `__init__()` 构造函数，方法为：使用函数的参数构造一个 ComponentSpec 类的实例，并使用该值和可选名称调用 super 函数

创建组件实例时，将调用 `base_component.BaseComponent` 类中的类型检查逻辑，以确保传入的参数与 `ComponentSpec` 类中定义的类型信息兼容。

```python
from tfx.types import standard_artifacts
from hello_component import executor

class HelloComponent(base_component.BaseComponent):
  """Custom TFX Hello World Component."""

  SPEC_CLASS = HelloComponentSpec
  EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(executor.Executor)

  def __init__(self,
               input_data: types.Channel = None,
               output_data: types.Channel = None,
               name: Optional[Text] = None):
    if not output_data:
      examples_artifact = standard_artifacts.Examples()
      examples_artifact.split_names = input_data.get()[0].split_names
      output_data = channel_utils.as_channel([examples_artifact])

    spec = HelloComponentSpec(input_data=input_data,
                              output_data=output_data, name=name)
    super(HelloComponent, self).__init__(spec=spec)
```

### 组装到 TFX 流水线

最后一步是将新的自定义组件插入 TFX 流水线。除了添加新组件的实例之外，还需要完成以下步骤：

- 将新组件与其上游和下游组件正确连接。实现方法是：在新组件中引用上游组件的输出并在下游组件中引用新组件的输出。
- 在构造流水线时，将新组件实例添加到组件列表。

下面的示例突出显示了上述更改。完整示例可以在 [TFX GitHub 仓库](https://github.com/tensorflow/tfx/tree/master/tfx/examples/custom_components/hello_world)中找到。

```python
def _create_pipeline():
  ...
  example_gen = CsvExampleGen(input_base=examples)
  hello = component.HelloComponent(
      input_data=example_gen.outputs['examples'], name='HelloWorld')
  statistics_gen = StatisticsGen(examples=hello.outputs['output_data'])
  ...
  return pipeline.Pipeline(
      ...
      components=[example_gen, hello, statistics_gen, ...],
      ...
  )
```

## 部署完全自定义的组件

为了正确运行流水线，除了代码更改以外，还需要在流水线运行环境中能够访问所有新添加的部分（`ComponentSpec`、`Executor`、组件接口）。
