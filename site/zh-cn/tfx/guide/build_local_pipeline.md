# 在本地构建 TFX 流水线

通过 TFX 可以更轻松地将机器学习 (ML) 工作流编排为流水线，以实现以下功能：

- 使 ML 过程自动化，以便定期重新训练、评估和部署模型。
- 创建 ML 流水线，其中包括对模型性能的深入分析和对新训练的模型的验证，以确保性能和可靠性。
- 监视训练数据中的异常情况并消除训练-应用偏差
- 通过运行具有不同超参数集的流水线来提高实验速度。

典型的流水线开发过程从本地机器开始，进行数据分析和组件设置，然后再部署到生产环境中。本指南描述了在本地构建流水线的两种方法。

- 自定义 TFX 流水线模板以适合您的 ML 工作流的需求。TFX 流水线模板是预构建的工作流，它们演示了使用 TFX 标准组件的最佳做法。
- 使用 TFX 构建流水线。在此用例中，您无需从模板开始定义流水线。

在开发流水线时，您可以使用 `LocalDagRunner` 运行它。然后，一旦流水线各组件定义良好且经过测试，您便可以使用生产级编排器，例如 Kubeflow 或 Airflow。

## 开始之前

TFX 为 Python 软件包，因此您需要设置一个 Python 开发环境，例如虚拟环境或 Docker 容器。然后：

```bash
pip install tfx
```

如果您不熟悉 TFX 流水线，请在继续之前[详细了解 TFX 流水线的核心概念 ](understanding_tfx_pipelines)。

## 使用模板构建流水线

TFX 流水线模板通过提供预构建的一组流水线定义使流水线开发变得更加容易，您可针对自己的用例对这些预构建的流水线定义进行自定义。

以下各部分将介绍如何创建模板副本并对其进行自定义来满足您的需求。

### 创建流水线模板副本

1. 查看可用 TFX 流水线模板的列表：

    <pre class="devsite-click-to-copy devsite-terminal">    tfx template list
        </pre>

2. 从列表中选择模板

    <pre class="devsite-click-to-copy devsite-terminal">    tfx template copy --model=&lt;var&gt;template&lt;/var&gt; --pipeline_name=&lt;var&gt;pipeline-name&lt;/var&gt; \
        --destination_path=&lt;var&gt;destination-path&lt;/var&gt;
        </pre>

    替换以下内容：

    - <var>template</var>：要复制的模板的名称。
    - <var>pipeline-name</var>：要创建的流水线的名称。
    - <var>destination-path</var>：要将模板复制到的路径。

    详细了解 [`tfx template copy` 命令](cli#copy)。

3. 已在您指定的路径上创建了流水线模板的副本。

注：本指南的其余部分假设您选择了 `penguin` 模板。

### 探索流水线模板

本部分将概述通过模板创建的基架。

1. 探索已复制到流水线根目录的目录和文件

    - **pipeline** 目录，其中包含

        - `pipeline.py` - 定义流水线，并列出正在使用的组件
        - `configs.py` - 保存配置详细信息，例如数据来自何处或正在使用哪个编排器

    - **data** 目录

        - 此目录通常包含一个 `data.csv` 文件，该文件是 `ExampleGen` 的默认源。您可以在 `configs.py` 中更改数据源。

    - **models** 目录，包含预处理代码和模型实现。

    - 模板为本地环境和 Kubeflow 复制 DAG 运行程序。

    - 某些模板还包括 Python 笔记本，以便您可以使用 Machine Learning MetaData 来探索您的数据和工件。

2. 在流水线目录中运行以下命令：

    <pre class="devsite-click-to-copy devsite-terminal">    tfx pipeline create --pipeline_path local_runner.py
        </pre>

    <pre class="devsite-click-to-copy devsite-terminal">    tfx run create --pipeline_name &lt;var&gt;pipeline_name&lt;/var&gt;
        </pre>

    该命令使用 `LocalDagRunner` 创建流水线运行，将以下目录添加到流水线中：

    - **tfx_metadata** 目录，其中包含本地使用的 ML Metadata 库。
    - **tfx_pipeline_output** 目录，其中包含流水线的文件输出。

    注：`LocalDagRunner` 是 TFX 中支持的几种编排器之一。它特别适合在本地运行流水线以实现更快迭代，数据集也可能更小。`LocalDagRunner` 可能不适合生产使用，因为它在单台机器上运行，系统变得不可用时更容易丢失工作。TFX 还支持 Apache Beam、Apache Airflow 和 Kubeflow Pipeline 等编排器。如果要配合其他编排器使用 TFX，请为该编排器使用适当的 DAG 运行程序。

    注：在撰写本文时，`penguin` 模板中使用的是 `LocalDagRunner`，而 `taxi` 模板则使用 Apache Beam。`taxi` 模板的配置文件被设置为使用 Beam，而且 CLI 命令相同。

3. 打开流水线的 `pipeline/configs.py` 文件并查看内容。此脚本定义流水线使用的配置选项以及组件功能。您可以在此处指定数据源的位置或运行中的训练步骤数等内容。

4. 打开流水线的 `pipeline/pipeline.py` 文件并查看内容。该脚本创建 TFX 流水线。最初，流水线仅包含一个 `ExampleGen`  组件。

    - 请遵循 `pipeline.py` 的 **TODO** 注释中的说明，向流水线添加更多步骤。

5. 打开 `local_runner.py` 文件并查看内容。该脚本创建流水线运行并指定运行的*参数*（如 `data_path` 和 `preprocessing_fn`）。

6. 您已经查看了通过模板创建的基架，并使用 `LocalDagRunner` 创建了一个流水线运行。接下来，对模板进行自定义以满足您的要求。

### 自定义流水线

本部分将概述如何开始自定义模板。

1. 设计流水线。模板提供的基架可以帮助您使用 TFX 标准组件为表格数据实现流水线。若要将现有 ML 工作流移入流水线，则可能需要修改代码以充分利用 [TFX 标准组件](index#tfx_standard_components)。您可能还需要创建[自定义组件](understanding_custom_components)，以实现您的工作流所独有的功能或 TFX 标准组件尚不支持的功能。

2. 完成流水线设计后，使用下列过程以迭代方式自定义流水线。从将数据提取到流水线的组件开始，通常是 `ExampleGen` 组件。

    1. 自定义流水线或组件以适合您的用例。这些自定义可能包括如下更改：

        - 更改流水线参数。
        - 向流水线添加组件或移除组件。
        - 替换数据输入源。该数据源可以是文件，也可以是对 BigQuery 等服务的查询。
        - 更改流水线中组件的配置。
        - 更改组件的自定义函数。

    2. 使用 `local_runner.py` 脚本（如果您使用的是其他编排器，则使用其他合适的 DAG 运行程序）在本地运行组件。如果脚本失败，请对故障进行调试并尝试重试运行脚本。

    3. 该自定义正常工作后，继续下一个自定义。

3. 以迭代方式进行工作，您可以自定义模版工作流中的每个步骤以满足您的需求。

## 构建自定义流水线

使用以下说明详细了解如何在不使用模板的情况下构建自定义流水线。

1. 设计流水线。TFX 标准组件提供了经过验证的功能，以帮助您实现完整的 ML 工作流。如果要将现有的 ML 工作流移入流水线，则可能需要修改代码以充分利用 TFX 标准组件。您可能还需要创建[自定义组件](understanding_custom_components)来实现数据增强之类的功能。

    - 详细了解[标准 TFX 组件](index#tfx_standard_components)。
    - 详细了解[自定义组件](understanding_custom_components)。

2. 使用以下示例创建脚本文件以定义您的流水线。本指南将该文件命名为 `my_pipeline.py`。

    <pre class="devsite-click-to-copy prettyprint">    import os
        from typing import Optional, Text, List
        from absl import logging
        from ml_metadata.proto import metadata_store_pb2
        import tfx.v1 as tfx

        PIPELINE_NAME = 'my_pipeline'
        PIPELINE_ROOT = os.path.join('.', 'my_pipeline_output')
        METADATA_PATH = os.path.join('.', 'tfx_metadata', PIPELINE_NAME, 'metadata.db')
        ENABLE_CACHE = True

        def create_pipeline(
          pipeline_name: Text,
          pipeline_root:Text,
          enable_cache: bool,
          metadata_connection_config: Optional[
            metadata_store_pb2.ConnectionConfig] = None,
          beam_pipeline_args: Optional[List[Text]] = None
        ):
          components = []

          return tfx.dsl.Pipeline(
                pipeline_name=pipeline_name,
                pipeline_root=pipeline_root,
                components=components,
                enable_cache=enable_cache,
                metadata_connection_config=metadata_connection_config,
                beam_pipeline_args=beam_pipeline_args, &lt;!-- needed? --&gt;
            )

        def run_pipeline():
          my_pipeline = create_pipeline(
              pipeline_name=PIPELINE_NAME,
              pipeline_root=PIPELINE_ROOT,
              enable_cache=ENABLE_CACHE,
              metadata_connection_config=tfx.orchestration.metadata.sqlite_metadata_connection_config(METADATA_PATH)
              )

          tfx.orchestration.LocalDagRunner().run(my_pipeline)

        if __name__ == '__main__':
          logging.set_verbosity(logging.INFO)
          run_pipeline()
        </pre>

    在接下来的步骤中，您将在 `create_pipeline` 中定义流水线并使用本地运行程序在本地运行流水线。

    使用以下过程以迭代方式构建流水线。

    1. 自定义流水线或组件以适合您的用例。这些自定义可能包括如下更改：

        - 更改流水线参数。
        - 向流水线添加组件或移除组件。
        - 替换数据输入文件。
        - 更改流水线中组件的配置。
        - 更改组件的自定义函数。

    2. 使用本地运行程序或直接通过运行脚本来在本地运行组件。如果脚本失败，请对故障进行调试并尝试重试运行脚本。

    3. 该自定义正常工作后，继续下一个自定义。

    从流水线工作流中的第一个节点开始，通常第一个节点会将数据提取到流水线中。

3. 将工作流中的第一个节点添加到流水线。在此示例中，流水线使用 `ExampleGen` 标准组件从 `./data` 目录加载 CSV 文件。

    <pre class="devsite-click-to-copy prettyprint">    from tfx.components import CsvExampleGen

        DATA_PATH = os.path.join('.', 'data')

        def create_pipeline(
          pipeline_name: Text,
          pipeline_root:Text,
          data_path: Text,
          enable_cache: bool,
          metadata_connection_config: Optional[
            metadata_store_pb2.ConnectionConfig] = None,
          beam_pipeline_args: Optional[List[Text]] = None
        ):
          components = []

          example_gen = tfx.components.CsvExampleGen(input_base=data_path)
          components.append(example_gen)

          return tfx.dsl.Pipeline(
                pipeline_name=pipeline_name,
                pipeline_root=pipeline_root,
                components=components,
                enable_cache=enable_cache,
                metadata_connection_config=metadata_connection_config,
                beam_pipeline_args=beam_pipeline_args, &lt;!-- needed? --&gt;
            )

        def run_pipeline():
          my_pipeline = create_pipeline(
            pipeline_name=PIPELINE_NAME,
            pipeline_root=PIPELINE_ROOT,
            data_path=DATA_PATH,
            enable_cache=ENABLE_CACHE,
            metadata_connection_config=tfx.orchestration.metadata.sqlite_metadata_connection_config(METADATA_PATH)
            )

          tfx.orchestration.LocalDagRunner().run(my_pipeline)
        </pre>

    `CsvExampleGen` 使用指定数据路径的 CSV 中的数据创建序列化示例记录，方法是使用数据根设置 `CsvExampleGen` 组件的 `input_base` 参数。

4. 在与 `my_pipeline.py` 相同的目录中创建 `data` 目录。将一个小型 CSV 文件添加到 `data` 目录。

5. 使用以下命令运行 `my_pipeline.py` 脚本。

    <pre class="devsite-click-to-copy devsite-terminal">    python my_pipeline.py
        </pre>

    结果应大致如下所示：

    <pre>    INFO:absl:Component CsvExampleGen depends on [].
        INFO:absl:Component CsvExampleGen is scheduled.
        INFO:absl:Component CsvExampleGen is running.
        INFO:absl:Running driver for CsvExampleGen
        INFO:absl:MetadataStore with DB connection initialized
        INFO:absl:Running executor for CsvExampleGen
        INFO:absl:Generating examples.
        INFO:absl:Using 1 process(es) for Local pipeline execution.
        INFO:absl:Processing input csv data ./data/* to TFExample.
        WARNING:root:Couldn't find python-snappy so the implementation of _TFRecordUtil._masked_crc32c is not as fast as it could be.
        INFO:absl:Examples generated.
        INFO:absl:Running publisher for CsvExampleGen
        INFO:absl:MetadataStore with DB connection initialized
        INFO:absl:Component CsvExampleGen is finished.
        </pre>

6. 继续以迭代方式将组件添加到流水线。
