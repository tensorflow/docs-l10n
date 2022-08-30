# 在本地构建 TFX 流水线

通过 TFX 可以更轻松地将机器学习 (ML) 工作流编排为流水线，以实现以下功能：

- 使 ML 过程自动化，以便定期重新训练、评估和部署模型。
- Create ML pipelines which include deep analysis of model performance and validation of newly trained models to ensure performance and reliability.
- Monitor training data for anomalies and eliminate training-serving skew
- Increase the velocity of experimentation by running a pipeline with different sets of hyperparameters.

典型的流水线开发过程从本地机器开始，进行数据分析和组件设置，然后再部署到生产环境中。本指南描述了在本地构建流水线的两种方法。

- Customize a TFX pipeline template to fit the needs of your ML workflow. TFX pipeline templates are prebuilt workflows that demonstrate best practices using the TFX standard components.
- Build a pipeline using TFX. In this use case, you define a pipeline without starting from a template.

在开发流水线时，您可以使用 `LocalDagRunner` 运行它。然后，一旦流水线各组件定义良好且经过测试，您便可以使用生产级编排器，例如 Kubeflow 或 Airflow。

## 开始之前

TFX 为 Python 软件包，因此您需要设置一个 Python 开发环境，例如虚拟环境或 Docker 容器。然后：

```bash
pip install tfx
```

If you are new to TFX pipelines, [learn more about the core concepts for TFX pipelines](understanding_tfx_pipelines) before continuing.

## Build a pipeline using a template

TFX 流水线模板通过提供预构建的一组流水线定义使流水线开发变得更加容易，您可针对自己的用例对这些预构建的流水线定义进行自定义。

The following sections describe how to create a copy of a template and customize it to meet your needs.

### Create a copy of the pipeline template

1. 查看可用 TFX 流水线模板的列表：

    <pre class="devsite-click-to-copy devsite-terminal">    tfx template list
        </pre>

2. 从列表中选择模板

    <pre class="devsite-click-to-copy devsite-terminal">    tfx template copy --model=&lt;var&gt;template&lt;/var&gt; --pipeline_name=&lt;var&gt;pipeline-name&lt;/var&gt; \
        --destination_path=&lt;var&gt;destination-path&lt;/var&gt;
        </pre>

    Replace the following:

    - <var>template</var>: The name of the template you want to copy.
    - <var>pipeline-name</var>: The name of the pipeline to create.
    - <var>destination-path</var>：要将模板复制到的路径。

    Learn more about the [`tfx template copy` command](cli#copy).

3. A copy of the pipeline template has been created at the path you specified.

注：本指南的其余部分假设您选择了 `penguin` 模板。

### Explore the pipeline template

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
    - A **tfx_pipeline_output** directory which contains the pipeline's file outputs.

    注：`LocalDagRunner` 是 TFX 中支持的几种编排器之一。它特别适合在本地运行流水线以实现更快迭代，数据集也可能更小。`LocalDagRunner` 可能不适合生产使用，因为它在单台机器上运行，系统变得不可用时更容易丢失工作。TFX 还支持 Apache Beam、Apache Airflow 和 Kubeflow Pipeline 等编排器。如果要配合其他编排器使用 TFX，请为该编排器使用适当的 DAG 运行程序。

    注：在撰写本文时，`penguin` 模板中使用的是 `LocalDagRunner`，而 `taxi` 模板则使用 Apache Beam。`taxi` 模板的配置文件被设置为使用 Beam，而且 CLI 命令相同。

3. 打开流水线的 `pipeline/configs.py` 文件并查看内容。此脚本定义流水线使用的配置选项以及组件功能。您可以在此处指定数据源的位置或运行中的训练步骤数等内容。

4. 打开流水线的 `pipeline/pipeline.py` 文件并查看内容。该脚本创建 TFX 流水线。最初，流水线仅包含一个 `ExampleGen`  组件。

    - 请遵循 `pipeline.py` 的 **TODO** 注释中的说明，向流水线添加更多步骤。

5. 打开 `local_runner.py` 文件并查看内容。该脚本创建流水线运行并指定运行的*参数*（如 `data_path` 和 `preprocessing_fn`）。

6. 您已经查看了通过模板创建的基架，并使用 `LocalDagRunner` 创建了一个流水线运行。接下来，对模板进行自定义以满足您的要求。

### Customize your pipeline

本部分将概述如何开始自定义模板。

1. Design your pipeline. The scaffolding that a template provides helps you implement a pipeline for tabular data using the TFX standard components. If you are moving an existing ML workflow into a pipeline, you may need to revise your code to make full use of [TFX standard components](index#tfx_standard_components). You may also need to create [custom components](understanding_custom_components) that implement features which are unique to your workflow or that are not yet supported by TFX standard components.

2. Once you have designed your pipeline, iteratively customize the pipeline using the following process. Start from the component that ingests data into your pipeline, which is usually the `ExampleGen` component.

    1. Customize the pipeline or a component to fit your use case. These customizations may include changes like:

        - Changing pipeline parameters.
        - Adding components to the pipeline or removing them.
        - Replacing the data input source. This data source can either be a file or queries into services such as BigQuery.
        - Changing a component's configuration in the pipeline.
        - Changing a component's customization function.

    2. 使用 `local_runner.py` 脚本（如果您使用的是其他编排器，则使用其他合适的 DAG 运行程序）在本地运行组件。如果脚本失败，请对故障进行调试并尝试重试运行脚本。

    3. Once this customization is working, move on to the next customization.

3. Working iteratively, you can customize each step in the template workflow to meet your needs.

## Build a custom pipeline

Use the following instructions to learn more about building a custom pipeline without using a template.

1. Design your pipeline. The TFX standard components provide proven functionality to help you implement a complete ML workflow. If you are moving an existing ML workflow into a pipeline, you may need to revise your code to make full use of TFX standard components. You may also need to create [custom components](understanding_custom_components) that implement features such as data augmentation.

    - Learn more about [standard TFX components](index#tfx_standard_components).
    - Learn more about [custom components](understanding_custom_components).

2. Create a script file to define your pipeline using the following example. This guide refers to this file as `my_pipeline.py`.

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

    Iteratively build your pipeline using the following process.

    1. Customize the pipeline or a component to fit your use case. These customizations may include changes like:

        - Changing pipeline parameters.
        - Adding components to the pipeline or removing them.
        - Replacing a data input file.
        - Changing a component's configuration in the pipeline.
        - Changing a component's customization function.

    2. 使用本地运行程序或直接通过运行脚本来在本地运行组件。如果脚本失败，请对故障进行调试并尝试重试运行脚本。

    3. Once this customization is working, move on to the next customization.

    Start from the first node in your pipeline's workflow, typically the first node ingests data into your pipeline.

3. Add the first node in your workflow to your pipeline. In this example, the pipeline uses the `ExampleGen` standard component to load a CSV from a directory at `./data`.

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

4. Create a `data` directory in the same directory as `my_pipeline.py`. Add a small CSV file to the `data` directory.

5. 使用以下命令运行 `my_pipeline.py` 脚本。

    <pre class="devsite-click-to-copy devsite-terminal">    python my_pipeline.py
        </pre>

    The result should be something like the following:

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

6. Continue to iteratively add components to your pipeline.
