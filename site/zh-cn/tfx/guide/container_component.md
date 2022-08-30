# 构建基于容器的组件

基于容器的组件可以灵活地将以任何语言编写的代码集成到您的流水线中，只要能在 Docker 容器中执行该代码即可。

如果您不熟悉 TFX 流水线，请[详细了解 TFX 流水线的核心概念](understanding_tfx_pipelines)。

## 创建基于容器的组件

基于容器的组件由容器化的命令行程序支持。如果已有容器镜像，您可以通过使用 [`create_container_component` 函数](https://github.com/tensorflow/tfx/blob/master/tfx/dsl/component/experimental/container_component.py){: .external } 声明输入和输出，利用 TFX 从该镜像创建组件。函数参数如下：

- **name**：组件的名称。
- **inputs**：将输入名称映射到类型的字典。输出：将输出名称映射到类型的字典。参数：将参数名称映射到类型的字典。
- **image**：容器镜像名称，以及可选的镜像标记。
- **command**：容器入口点命令行。不在 shell 内执行。命令行可以使用占位符对象，这些占位符对象会在编译时被替换为输入、输出或参数。占位符对象可以从 [`tfx.dsl.component.experimental.placeholders`](https://github.com/tensorflow/tfx/blob/master/tfx/dsl/component/experimental/placeholders.py){: .external } 导入。请注意，不支持 Jinja 模板。

**Return value**：从 base_component.BaseComponent 继承的 Component 类，可以在流水线内部实例化和使用。

### 占位符

对于具有输入或输出的组件，`command` 通常需要具有会在运行时被替换为实际数据的占位符。为此提供了几个占位符：

- `InputValuePlaceholder`：输入工件值的占位符。在运行时，此占位符将被替换为工件值的字符串表示。

- `InputUriPlaceholder`：输入工件参数的 URI 占位符。在运行时，此占位符将被替换为输入工件数据的 URI。

- `OutputUriPlaceholder`：输出工件参数的 URI 占位符。在运行时，此占位符将被替换为组件应存储输出工件数据的 URI。

详细了解 [TFX 组件命令行占位符](https://github.com/tensorflow/tfx/blob/master/tfx/dsl/component/experimental/placeholders.py){: .external }。

### 基于容器的组件示例

以下是一个下载、转换和上传数据的非 Python 组件示例：

```python
import tfx.v1 as tfx

grep_component = tfx.dsl.components.create_container_component(
    name='FilterWithGrep',
    inputs={
        'text': tfx.standard_artifacts.ExternalArtifact,
    },
    outputs={
        'filtered_text': tfx.standard_artifacts.ExternalArtifact,
    },
    parameters={
        'pattern': str,
    },
    # The component code uses gsutil to upload the data to Google Cloud Storage, so the
    # container image needs to have gsutil installed and configured.
    image='google/cloud-sdk:278.0.0',
    command=[
        'sh', '-exc',
        '''
          pattern="$1"
          text_uri="$3"/data  # Adding suffix, because currently the URI are "directories". This will be fixed soon.
          text_path=$(mktemp)
          filtered_text_uri="$5"/data  # Adding suffix, because currently the URI are "directories". This will be fixed soon.
          filtered_text_path=$(mktemp)

          # Getting data into the container
          gsutil cp "$text_uri" "$text_path"

          # Running the main code
          grep "$pattern" "$text_path" >"$filtered_text_path"

          # Getting data out of the container
          gsutil cp "$filtered_text_path" "$filtered_text_uri"
        ''',
        '--pattern', tfx.dsl.placeholders.InputValuePlaceholder('pattern'),
        '--text', tfx.dsl.placeholders.InputUriPlaceholder('text'),
        '--filtered-text', tfx.dsl.placeholders.OutputUriPlaceholder('filtered_text'),
    ],
)
```
