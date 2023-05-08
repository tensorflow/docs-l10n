# 使用 Google Cloud 为机器学习预处理数据

本教程介绍如何使用 [TensorFlow Transform](https://github.com/tensorflow/transform){: .external}（`tf.Transform` 库）为机器学习 (ML) 实现数据预处理。借助 TensorFlow 的 `tf.Transform` 库，您可以通过数据预处理流水线定义实例级和全通数据转换。这些流水线是使用 [Apache Beam](https://beam.apache.org/){: .external} 高效执行的，并且它们还会创建 TensorFlow 计算图作为副产品，以便在预测期间执行与应用模型时相同的转换。

本教程提供了一个使用 [Dataflow](https://cloud.google.com/dataflow/docs){: .external } 作为 Apache Beam 运行程序的端到端示例。本文假定您熟悉 [BigQuery](https://cloud.google.com/bigquery/docs){: .external }、Dataflow、[Vertex AI](https://cloud.google.com/vertex-ai/docs/start/introduction-unified-platform){: .external } 和 TensorFlow [Keras](https://www.tensorflow.org/guide/keras/overview) API。本文档还假定您具有使用 Jupyter 笔记本的经验，例如使用 Vertex AI Workbench{: .external }。

本教程还假定您熟悉 Google Cloud 上的预处理类型、挑战和选项概念，如[为机器学习预处理数据：选项和推荐](../../guide/tft_bestpractices)中所述。

## 目标

- 使用 `tf.Transform` 库实现 Apache Beam 流水线。
- 在 Dataflow 中运行流水线。
- 使用 `tf.Transform` 库实现 TensorFlow 模型。
- 训练并使用模型进行预测。

## 成本

本教程使用 Google Cloud 的以下收费组件：

- [Vertex AI](https://cloud.google.com/vertex-ai/pricing){: .external}
- [Cloud Storage](https://cloud.google.com/storage/pricing){: .external}
- [BigQuery](https://cloud.google.com/bigquery/pricing){: .external}
- [Dataflow](https://cloud.google.com/dataflow/pricing){: .external}

<!-- This doc uses plain text cost information because the pricing calculator is pre-configured -->

要估算运行本教程的成本，假设您将各项资源使用一整天，请使用预配置的[价格计算器](/products/calculator/#id=fad4d8-dd68-45b8-954e-5a56a5d148){: .external }。

## 准备工作

1. 在 Google Cloud 控制台的项目选择器页面上，选择或[创建一个 Google Cloud 项目](https://cloud.google.com/resource-manager/docs/creating-managing-projects)。

注：如果您不打算保留在此过程中创建的资源，请创建新的项目，而不要选择现有的项目。完成这些步骤后，您可以删除项目，并移除与该项目关联的所有资源。

[转到项目选择器](https://console.cloud.google.com/projectselector2/home/dashboard){: class="button button-primary" target="console" track-type="solution" track-name="consoleLink" track-metadata-position="body" }

1. 确保您的 Cloud 项目已启用结算功能。了解如何[检查项目是否已启用结算功能](https://cloud.google.com/billing/docs/how-to/verify-billing-enabled).

2. 启用 Dataflow、Vertex AI 和 Notebooks API。[启用 API](https://console.cloud.google.com/flows/enableapi?apiid=dataflow,aiplatform.googleapis.com,notebooks.googleapis.com){: class="button button-primary" target="console" track-type="solution" track-name="consoleLink" track-metadata-position="body" }

## 适用于此解决方案的 Jupyter 笔记本

以下 Jupyter 笔记本展示了实现示例：

- [笔记本 1](https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/blogs/babyweight_tft/babyweight_tft_keras_01.ipynb){: .external } 涵盖数据预处理。后面的[实现 Apache Beam 流水线](#implement-the-apache-beam-pipeline)部分提供了详细信息。
- [笔记本 2](https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/blogs/babyweight_tft/babyweight_tft_keras_02.ipynb){: .external } 涵盖模型训练。后面的[实现 TensorFlow 模型](#implement-the-tensorflow-model)部分提供了详细信息。

在以下部分中，您将克隆这些笔记本，然后执行笔记本以了解实现示例的工作原理。

## 启动用户管理的笔记本实例

1. 在 Google Cloud 控制台中，转到 **Vertex AI Workbench** 页面。

    [转到 Workbench](https://console.cloud.google.com/ai-platform/notebooks/list/instances){: class="button button-primary" target="console" track-type="solution" track-name="consoleLink" track-metadata-position="body" }

2. 在 **User-managed notebooks** 标签页上，点击 **+New notebook**。

3. 为实例类型选择 **TensorFlow Enterprise 2.8 (with LTS) without GPUs**。

4. 点击 **Create**。

创建笔记本后，等待 JupyterLab 的代理完成初始化。准备就绪后，笔记本名称旁边会显示 **Open JupyterLab**。

## 克隆笔记本

1. 在 **User-managed notebooks tab** 标签页上，点击笔记本名称旁边的 **Open JupyterLab**。JupyterLab 界面会在新标签页中打开。

    如果 JupyterLab 显示 **Build Recommended** 对话框，请点击 **Cancel** 来拒绝建议的构建。

2. 在 **Launcher** 标签页上，点击 **Terminal**。

3. 在终端窗口中，克隆笔记本：

    ```sh
    git clone https://github.com/GoogleCloudPlatform/training-data-analyst
    ```

## 实现 Apache Beam 流水线

本部分和下一部分[在 Dataflow 中运行流水线](#run-the-pipeline-in-dataflow){: track-type="solution" track-name="internalLink" track-metadata-position="body" } 为笔记本 1 提供了概览和上下文。笔记本提供了一个实际示例，描述了如何使用 `tf.Transform` 库来预处理数据。此示例使用了用于根据各种输入预测婴儿体重的 Natality 数据集。这些数据存储在 BigQuery 的公共[出生率](https://console.cloud.google.com/bigquery?p=bigquery-public-data&d=samples&t=natality&page=table&_ga=2.267763789.2122871960.1676620306-376763843.1676620306){: target="console" track-type="solution" track-name="consoleLink" track-metadata-position="body" }表中。

### 运行笔记本 1

1. 在 JupyterLab 界面中，点击 **File &gt; Open from path**，然后输入以下路径：

    ```sh
    training-data-analyst/blogs/babyweight_tft/babyweight_tft_keras_01.ipynb
    ```

2. 点击 **Edit &gt; Clear all outputs**。

3. 在 **Install required packages** 部分中，执行第一个单元以运行 `pip install apache-beam` 命令。

    输出的最后一部分如下所示：

    ```none{:.devsite-disable-click-to-copy}
    Successfully installed ...
    ```

    您可以忽略输出中的依赖项错误。无需重启内核。

4. 执行第二个单元以运行 `pip install tensorflow-transform` 命令。输出的最后一部分如下所示：

    ```none{:.devsite-disable-click-to-copy}
    Successfully installed ...
    Note: you may need to restart the kernel to use updated packages.
    ```

    您可以忽略输出中的依赖项错误。

5. 点击 **Kernel &gt; Restart Kernel**。

6. 执行 **Confirm the installed packages** 和 **Create setup.py to install packages to Dataflow containers** 部分中的单元。

7. 在 **Set global flags** 部分中，在 `PROJECT` 和 `BUCKET` 旁边，将 `your-project` 替换为您的 Cloud 项目 ID，然后执行该单元。

8. 执行其余所有单元，直到笔记本中的最后一个单元。有关在每个单元中执行操作的信息，请参阅笔记本中的说明。

### 流水线概述

在笔记本示例中，Dataflow 规模化运行 `tf.Transform` 流水线以准备数据并生成转换工件。本文档后面的几个部分说明了执行流水线中的每个步骤的函数。整个流水线步骤如下所示：

1. 从 BigQuery 读取训练数据。
2. 使用 `tf.Transform` 库分析和转换训练数据。
3. 以 [TFRecord](https://www.tensorflow.org/tutorials/load_data/tfrecord){: target="external" class="external" track-type="solution" track-name="externalLink" track-metadata-position="body" } 格式将转换后的训练数据写入 Cloud Storage。
4. 从 BigQuery 读取评估数据。
5. 使用第 2 步生成的 `transform_fn` 计算图转换评估数据。
6. 以 TFRecord 格式将转换后的训练数据写入 Cloud Storage。
7. 将转换工件写入 Cloud Storage，以便稍后用于创建和导出模型。

以下示例展示了整个流水线的 Python 代码。接下来的部分提供了每个步骤的说明和代码列表。

```py{:.devsite-disable-click-to-copy}
def run_transformation_pipeline(args):

    pipeline_options = beam.pipeline.PipelineOptions(flags=[], **args)

    runner = args['runner']
    data_size = args['data_size']
    transformed_data_location = args['transformed_data_location']
    transform_artefact_location = args['transform_artefact_location']
    temporary_dir = args['temporary_dir']
    debug = args['debug']

    # Instantiate the pipeline
    with beam.Pipeline(runner, options=pipeline_options) as pipeline:
        with impl.Context(temporary_dir):

            # Preprocess train data
            step = 'train'
            # Read raw train data from BigQuery
            raw_train_dataset = read_from_bq(pipeline, step, data_size)
            # Analyze and transform raw_train_dataset
            transformed_train_dataset, transform_fn = analyze_and_transform(raw_train_dataset, step)
            # Write transformed train data to sink as tfrecords
            write_tfrecords(transformed_train_dataset, transformed_data_location, step)

            # Preprocess evaluation data
            step = 'eval'
            # Read raw eval data from BigQuery
            raw_eval_dataset = read_from_bq(pipeline, step, data_size)
            # Transform eval data based on produced transform_fn
            transformed_eval_dataset = transform(raw_eval_dataset, transform_fn, step)
            # Write transformed eval data to sink as tfrecords
            write_tfrecords(transformed_eval_dataset, transformed_data_location, step)

            # Write transformation artefacts
            write_transform_artefacts(transform_fn, transform_artefact_location)

            # (Optional) for debugging, write transformed data as text
            step = 'debug'
            # Write transformed train data as text if debug enabled
            if debug == True:
                write_text(transformed_train_dataset, transformed_data_location, step)
```

### 从 BigQuery 读取原始训练数据{: id="read_raw_training_data"}

第一步是使用 `read_from_bq` 函数从 BigQuery 读取原始训练数据。此函数返回一个从 BigQuery 中提取的 `raw_dataset` 对象。您传递一个 `data_size` 值并传递 `train` 或 `eval` 的 `step` 值。BigQuery 源查询使用 `get_source_query` 函数构建，如以下示例所示：

```py{:.devsite-disable-click-to-copy}
def read_from_bq(pipeline, step, data_size):

    source_query = get_source_query(step, data_size)
    raw_data = (
        pipeline
        | '{} - Read Data from BigQuery'.format(step) >> beam.io.Read(
                           beam.io.BigQuerySource(query=source_query, use_standard_sql=True))
        | '{} - Clean up Data'.format(step) >> beam.Map(prep_bq_row)
    )

    raw_metadata = create_raw_metadata()
    raw_dataset = (raw_data, raw_metadata)
    return raw_dataset
```

在执行 `tf.Transform` 预处理之前，您可能需要执行基于 Apache Beam 的典型处理，包括映射、过滤、分组和窗口处理。在该示例中，代码使用 `beam.Map(prep_bq_row)` 方法清理从 BigQuery 读取的记录，其中 `prep_bq_row` 是自定义函数。此自定义函数将分类特征的数值代码转换为直观易懂的标签。

此外，要使用 `tf.Transform` 库分析和转换从 BigQuery 中提取的 `raw_data` 对象，您需要创建一个 `raw_dataset` 对象，即 `raw_data` 和 `raw_metadata` 对象的元组。`raw_metadata` 对象是使用 `create_raw_metadata` 函数创建的，如下所示：

```py{:.devsite-disable-click-to-copy}
CATEGORICAL_FEATURE_NAMES = ['is_male', 'mother_race']
NUMERIC_FEATURE_NAMES = ['mother_age', 'plurality', 'gestation_weeks']
TARGET_FEATURE_NAME = 'weight_pounds'

def create_raw_metadata():

    feature_spec = dict(
        [(name, tf.io.FixedLenFeature([], tf.string)) for name in CATEGORICAL_FEATURE_NAMES] +
        [(name, tf.io.FixedLenFeature([], tf.float32)) for name in NUMERIC_FEATURE_NAMES] +
        [(TARGET_FEATURE_NAME, tf.io.FixedLenFeature([], tf.float32))])

    raw_metadata = dataset_metadata.DatasetMetadata(
        schema_utils.schema_from_feature_spec(feature_spec))

    return raw_metadata
```

执行笔记本中定义此方法的单元后紧跟的单元时，系统会显示 `raw_metadata.schema` 对象的内容。它包括以下列：

- `gestation_weeks`（类型：`FLOAT`）
- `is_male`（类型：`BYTES`）
- `mother_age`（类型：`FLOAT`）
- `mother_race`（类型：`BYTES`）
- `plurality`（类型：`FLOAT`）
- `weight_pounds`（类型：`FLOAT`）

### 转换原始训练数据

假设您希望对训练数据的输入原始特征执行典型的预处理转换，以便为机器学习做好准备。这些转换包括全通和实例级操作，如下表所示：

<table>
<thead>
  <tr>
    <th>输入特征</th>
    <th>转换</th>
    <th>需要的统计信息</th>
    <th>类型</th>
    <th>输出特征</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td><code>weight_pound</code></td>
    <td>无</td>
    <td>无</td>
    <td>不适用</td>
    <td><code>weight_pound</code></td>
  </tr>
  <tr>
    <td><code>mother_age</code></td>
    <td>归一化</td>
    <td>平均值、方差</td>
    <td>全通</td>
    <td><code>mother_age_normalized</code></td>
  </tr>
  <tr>
    <td><code>mother_age</code></td>
    <td>等大小分桶</td>
    <td>分位数</td>
    <td>全通</td>
    <td><code>mother_age_bucketized</code></td>
  </tr>
  <tr>
    <td><code>mother_age</code></td>
    <td>计算日志</td>
    <td>无</td>
    <td>实例级</td>
    <td>
        <code>mother_age_log</code>
    </td>
  </tr>
  <tr>
    <td><code>plurality</code></td>
    <td>指示是单个还是多个婴儿</td>
    <td>无</td>
    <td>实例级</td>
    <td><code>is_multiple</code></td>
  </tr>
  <tr>
    <td><code>is_multiple</code></td>
    <td>将名义值转换为数值索引</td>
    <td>词汇表</td>
    <td>全通</td>
    <td><code>is_multiple_index</code></td>
  </tr>
  <tr>
    <td><code>gestation_weeks</code></td>
    <td>缩放到 0 到 1 之间</td>
    <td>最小值、最大值</td>
    <td>全通</td>
    <td><code>gestation_weeks_scaled</code></td>
  </tr>
  <tr>
    <td><code>mother_race</code></td>
    <td>将名义值转换为数值索引</td>
    <td>词汇表</td>
    <td>全通</td>
    <td><code>mother_race_index</code></td>
  </tr>
  <tr>
    <td><code>is_male</code></td>
    <td>将名义值转换为数值索引</td>
    <td>词汇表</td>
    <td>全通</td>
    <td><code>is_male_index</code></td>
  </tr>
</tbody>
</table>

这些转换在 `preprocess_fn` 函数中实现，该方法需要的是张量 (`input_features`) 的字典，返回已处理特征 (`output_features`) 的字典。

以下代码展示了使用 `tf.Transform` 全通转换 API（前缀为 `tft.`）以及 TensorFlow（前缀为 `tf.`）实例级操作来实现 `preprocess_fn` 函数的过程：

```py{:.devsite-disable-click-to-copy}
def preprocess_fn(input_features):

    output_features = {}

    # target feature
    output_features['weight_pounds'] = input_features['weight_pounds']

    # normalization
    output_features['mother_age_normalized'] = tft.scale_to_z_score(input_features['mother_age'])

    # scaling
    output_features['gestation_weeks_scaled'] =  tft.scale_to_0_1(input_features['gestation_weeks'])

    # bucketization based on quantiles
    output_features['mother_age_bucketized'] = tft.bucketize(input_features['mother_age'], num_buckets=5)

    # you can compute new features based on custom formulas
    output_features['mother_age_log'] = tf.math.log(input_features['mother_age'])

    # or create flags/indicators
    is_multiple = tf.as_string(input_features['plurality'] > tf.constant(1.0))

    # convert categorical features to indexed vocab
    output_features['mother_race_index'] = tft.compute_and_apply_vocabulary(input_features['mother_race'], vocab_filename='mother_race')
    output_features['is_male_index'] = tft.compute_and_apply_vocabulary(input_features['is_male'], vocab_filename='is_male')
    output_features['is_multiple_index'] = tft.compute_and_apply_vocabulary(is_multiple, vocab_filename='is_multiple')

    return output_features
```

除了先前示例中的转换之外，`tf.Transform` [框架](https://github.com/tensorflow/transform){: .external }还有多个其他转换，其中包括下表列出的转换：

<table>
<thead>
  <tr>
  <th>转换</th>
  <th>应用对象</th>
  <th>说明</th>
  </tr>
</thead>
<tbody>
    <tr>
    <td><code>scale_by_min_max</code></td>
    <td>数值特征</td>
    <td>       将数值列缩放到 [<code>output_min</code>,       <code>output_max</code>] 范围</td>
  </tr>
  <tr>
    <td><code>scale_to_0_1</code></td>
    <td>数值特征</td>
    <td>返回一列，该列为已缩放到 [<code>0</code>,<code>1</code>] 范围的输入列</td>
  </tr>
  <tr>
    <td><code>scale_to_z_score</code></td>
    <td>数值特征</td>
    <td>返回平均值为 0 且方差为 1 的标准化列</td>
  </tr>
  <tr>
    <td><code>tfidf</code></td>
    <td>文本特征</td>
    <td>将 <i>x</i> 中的字词映射到其词频 * 逆向文档频率</td>
  </tr>
  <tr>
    <td><code>compute_and_apply_vocabulary</code></td>
    <td>分类特征</td>
    <td>       生成分类特征的词汇表，并使用此词汇表将分类特征映射为一个整数</td>
  </tr>
  <tr>
    <td><code>ngrams</code></td>
    <td>文本特征</td>
    <td>创建 N 元语法的 <code>SparseTensor</code>
</td>
  </tr>
  <tr>
    <td><code>hash_strings</code></td>
    <td>分类特征</td>
    <td>对字符串进行哈希分桶操作</td>
  </tr>
  <tr>
    <td><code>pca</code></td>
    <td>数值特征</td>
    <td>使用偏协方差在数据集上计算 PCA</td>
  </tr>
  <tr>
    <td><code>bucketize</code></td>
    <td>数值特征</td>
    <td>       返回一个大小相等（基于分位数）的分桶列，且每个输入都分配有一个分桶索引</td>
  </tr>
</tbody>
</table>

要将 `preprocess_fn` 函数中实现的转换应用于流水线的上一步生成的 `raw_train_dataset` 对象，可以使用 `AnalyzeAndTransformDataset` 方法。此方法需要 `raw_dataset` 对象作为输入，应用 `preprocess_fn` 函数，并且会生成 `transformed_dataset` 对象和 `transform_fn` 计算图。以下代码说明了此处理过程：

```py{:.devsite-disable-click-to-copy}
def analyze_and_transform(raw_dataset, step):

    transformed_dataset, transform_fn = (
        raw_dataset
        | '{} - Analyze & Transform'.format(step) >> tft_beam.AnalyzeAndTransformDataset(
            preprocess_fn, output_record_batches=True)
    )

    return transformed_dataset, transform_fn
```

转换分两个阶段应用于原始数据：分析阶段和转换阶段。本文档后面的图 3 显示了如何将 `AnalyzeAndTransformDataset` 方法分解为 `AnalyzeDataset` 方法和 `TransformDataset` 方法。

#### 分析阶段

在分析阶段，原始训练数据在全通过程中进行分析，以计算转换所需的统计信息。这包括计算平均值、方差、最小值、最大值、分位数和词汇表。分析过程需要原始数据集（原始数据加上原始元数据），并且会生成两个输出：

- `transform_fn`：TensorFlow 计算图包含从分析阶段计算的统计信息和使用统计信息的转换逻辑作为实例级操作。如后面的[保存计算图](#save_the_graph){: track-type="solution" track-name="internalLink" track-metadata-position="body" }中所述，保存 `transform_fn` 计算图以将其附加到模型 `serving_fn` 函数。这样做便可对在线预测数据点应用相同的转换。
- `transform_metadata`：此对象描述转换后的数据需要的架构。

分析阶段如下面的图 1 所示：

<figure id="tf-transform-analyze-phase">
  <img src="images/data-preprocessing-for-ml-with-tf-transform-tf-transform-analyze-phase.svg"
    alt="The tf.Transform analyze phase.">
  <figcaption><b>Figure 1.</b> The <code>tf.Transform</code> analyze phase.</figcaption>
</figure>

`tf.Transform` [分析器](https://github.com/tensorflow/transform/blob/master/tensorflow_transform/beam/analyzer_impls.py){: target="github" class="external" track-type="solution" track-name="gitHubLink" track-metadata-position="body" }包括 `min`、`max`、`sum`、`size`、`mean`、`var`、`covariance`、`quantiles`、`vocabulary` 和 `pca`。

#### 转换阶段

在转换阶段，由分析阶段产生的 `transform_fn` 计算图用于在实例级过程中转换原始训练数据，以便产生转换后的训练数据。转换后的训练数据与转换后的元数据（由分析阶段产生）配对以生成 `transformed_train_dataset` 数据集。

转换阶段如下面的图 2 所示：

<figure id="tf-transform-transform-phase">
  <img src="images/data-preprocessing-for-ml-with-tf-transform-tf-transform-transform-phase.svg"
    alt="The tf.Transform transform phase.">
  <figcaption><b>Figure 2.</b> The <code>tf.Transform</code> transform phase.</figcaption>
</figure>

要预处理特征，请在 `preprocess_fn` 函数的实现中调用必需的 `tensorflow_transform` 转换（在代码中导入为 `tft`）。例如，在调用 `tft.scale_to_z_score` 操作时，`tf.Transform` 库将此函数调用转换为平均值和方差分析器，计算分析阶段的统计信息，然后应用这些统计信息来对转换阶段中的数值特征进行归一化。上述过程全部是通过调用 `AnalyzeAndTransformDataset(preprocess_fn)` 方法自动完成的。

此调用生成的 `transformed_metadata.schema` 实体包括以下列：

- `gestation_weeks_scaled`（类型：`FLOAT`）
- `is_male_index`（类型：`INT`、is_categorical：`True`）
- `is_multiple_index`（类型：`INT`、is_categorical：`True`）
- `mother_age_bucketized`（类型：`INT`、is_categorical：`True`）
- `mother_age_log`（类型：`FLOAT`）
- `mother_age_normalized`（类型：`FLOAT`）
- `mother_race_index`（类型：`INT`、is_categorical：`True`）
- `weight_pounds`（类型：`FLOAT`）

如本系列第一部分中的[预处理操作](data-preprocessing-for-ml-with-tf-transform-pt1#preprocessing_operations)中所述，特征转换将分类特征转换为数字表示法。转换后，分类特征由整数值表示。在 `transformed_metadata.schema` 实体中，`INT` 类型列的 `is_categorical` 标志指示该列代表分类特征还是真正数值特征。

### 写入转换后的训练数据{: id="step_3_write_transformed_training_data"}

在分析和转换阶段通过使用 `preprocess_fn` 函数预处理训练数据之后，您可将该数据写入接收器以用来训练 TensorFlow 模型。使用 Dataflow 执行 Apache Beam 流水线时，接收器是 Cloud Storage。在其他情况下，接收器是本地磁盘。虽然您可以将数据写为固定宽度格式的 CSV 文件，但 TensorFlow 数据集推荐使用的文件格式是 TFRecord 格式。这是一个简单的面向记录的二进制格式，由 `tf.train.Example` 协议缓冲区消息组成。

每个 `tf.train.Example` 记录包含一个或多个特征。它们被馈入模型进行训练时会转换为张量。以下代码将已转换的数据集写入指定位置的 TFRecord 文件中：

```py{:.devsite-disable-click-to-copy}
def write_tfrecords(transformed_dataset, location, step):
    from tfx_bsl.coders import example_coder

    transformed_data, transformed_metadata = transformed_dataset
    (
        transformed_data
        | '{} - Encode Transformed Data'.format(step) >> beam.FlatMapTuple(
                            lambda batch, _: example_coder.RecordBatchToExamples(batch))
        | '{} - Write Transformed Data'.format(step) >> beam.io.WriteToTFRecord(
                            file_path_prefix=os.path.join(location,'{}'.format(step)),
                            file_name_suffix='.tfrecords')
    )
```

### 读取、转换和写入评估数据

转换训练数据并生成 `transform_fn` 计算图后，您可以使用它来转换评估数据。首先，通过使用前面的[从 BigQuery 读取原始训练数据](#read-raw-training-data-from-bigquery){: track-type="solution" track-name="internalLink" track-metadata-position="body" }中所述的 `read_from_bq` 函数，并为 `step` 参数传递 `eval` 的值来读取和清理 BigQuery 中的评估数据。然后，使用以下代码将原始评估数据集 (`raw_dataset`) 转换为预期的转换后格式 (`transformed_dataset`)：

```py{:.devsite-disable-click-to-copy}
def transform(raw_dataset, transform_fn, step):

    transformed_dataset = (
        (raw_dataset, transform_fn)
        | '{} - Transform'.format(step) >> tft_beam.TransformDataset(output_record_batches=True)
    )

    return transformed_dataset
```

转换评估数据时，会使用 `transform_fn` 计算图中的逻辑以及从训练数据的分析阶段中计算的统计信息来仅应用实例级操作。换句话说，您不会将以全通方式分析评估数据计算得出的新统计信息（例如平均值和方差）用于对评估数据中的数值特征进行 z 评分归一化，而是使用从训练数据计算的统计信息，以实例级方式来转换评估数据。

因此，应在训练数据的上下文中使用 `AnalyzeAndTransform` 方法计算统计信息并转换数据。同时，在转换评估数据的上下文中使用 `TransformDataset` 方法以仅使用根据训练数据计算的统计信息来转换数据。

随后，使用 TFRecord 格式将数据写入接收器（Cloud Storage 或本地磁盘，具体取决于运行程序）以用于在训练过程中评估 TensorFlow 模型。为此，请使用[写入转换后的训练数据](#step_3_write_transformed_training_data){: track-type="solution" track-name="internalLink" track-metadata-position="body" }中讨论的 `write_tfrecords` 函数。下面的图 3 显示了如何使用在训练数据的分析阶段生成的 `transform_fn` 计算图来转换评估数据。

<figure id="transform-eval-data-using-transform-fn">
  <img src="images/data-preprocessing-for-ml-with-tf-transform-transforming-eval-data-using-transform_fn.svg"
    alt="Transforming evaluation data using the transform_fn graph.">
  <figcaption><b>Figure 3.</b> Transforming evaluation data using the <code>transform_fn</code> graph.</figcaption>
</figure>

### 保存计算图

`tf.Transform` 预处理流水线中的最后一步是存储工件，其中包括在训练数据的分析阶段中生成的 `transform_fn` 计算图。用于存储工件的代码显示在下面的 `write_transform_artefacts` 函数中：

```py{:.devsite-disable-click-to-copy}
def write_transform_artefacts(transform_fn, location):

    (
        transform_fn
        | 'Write Transform Artifacts' >> transform_fn_io.WriteTransformFn(location)
    )
```

这些工件稍后将用于训练模型以及导出模型进行应用。此步骤还会生成以下工件，如下一部分中所示：

- `saved_model.pb`：代表包含转换逻辑的 TensorFlow 计算图（`transform_fn` 计算图），此计算图将附加到模型应用接口以转换原始数据点的格式。
- `variables`：包括在训练数据的分析阶段计算的统计信息，用于 `saved_model.pb` 工件中的转换逻辑。
- `assets`：包含多个词汇表文件（使用 `compute_and_apply_vocabulary` 方法处理的每个分类特征各有一个），将在应用期间用于将输入原始名义值转换为数值索引。
- `transformed_metadata`：包含 `schema.json` 文件的目录，用于描述转换后数据的架构。

## 在 Dataflow 中运行流水线{:#run_the_pipeline_in_dataflow}

定义 `tf.Transform` 流水线后，使用 Dataflow 运行该流水线。下面的图 4 展示了示例中所述的 `tf.Transform` 流水线的 Dataflow 执行计算图。

<figure id="dataflow-execution-graph">
  <img src="images/data-preprocessing-for-ml-with-tf-transform-dataflow-execution-graph.png"
    alt="Dataflow execution graph of the tf.Transform pipeline." class="screenshot">
  <figcaption><b>Figure 4.</b> Dataflow execution graph
     of the <code>tf.Transform</code> pipeline.</figcaption>
</figure>

执行 Dataflow 流水线以预处理训练和评估数据后，您可以通过执行笔记本中的最后一个单元来浏览 Cloud Storage 中生成的对象。本部分中的代码段显示了结果，其中 <var><code>YOUR_BUCKET_NAME</code></var> 是您的 Cloud Storage 存储分区的名称。

TFRecord 格式的已转换训练和评估数据存储在以下位置：

```none{:.devsite-disable-click-to-copy}
gs://YOUR_BUCKET_NAME/babyweight_tft/transformed
```

生成的转换工件位于以下位置：

```none{:.devsite-disable-click-to-copy}
gs://YOUR_BUCKET_NAME/babyweight_tft/transform
```

以下列表是流水线的输出，显示了生成的数据对象和工件：

```none{:.devsite-disable-click-to-copy}
transformed data:
gs://YOUR_BUCKET_NAME/babyweight_tft/transformed/eval-00000-of-00001.tfrecords
gs://YOUR_BUCKET_NAME/babyweight_tft/transformed/train-00000-of-00002.tfrecords
gs://YOUR_BUCKET_NAME/babyweight_tft/transformed/train-00001-of-00002.tfrecords

transformed metadata:
gs://YOUR_BUCKET_NAME/babyweight_tft/transform/transformed_metadata/
gs://YOUR_BUCKET_NAME/babyweight_tft/transform/transformed_metadata/asset_map
gs://YOUR_BUCKET_NAME/babyweight_tft/transform/transformed_metadata/schema.pbtxt

transform artefact:
gs://YOUR_BUCKET_NAME/babyweight_tft/transform/transform_fn/
gs://YOUR_BUCKET_NAME/babyweight_tft/transform/transform_fn/saved_model.pb
gs://YOUR_BUCKET_NAME/babyweight_tft/transform/transform_fn/assets/
gs://YOUR_BUCKET_NAME/babyweight_tft/transform/transform_fn/variables/

transform assets:
gs://YOUR_BUCKET_NAME/babyweight_tft/transform/transform_fn/assets/
gs://YOUR_BUCKET_NAME/babyweight_tft/transform/transform_fn/assets/is_male
gs://YOUR_BUCKET_NAME/babyweight_tft/transform/transform_fn/assets/is_multiple
gs://YOUR_BUCKET_NAME/babyweight_tft/transform/transform_fn/assets/mother_race
```

## 实现 TensorFlow 模型{: id="implementing_the_tensorflow_model"}

本部分和下一部分[训练并使用模型进行预测](#train_and_use_the_model_for_predictions){: track-type="solution" track-name="internalLink" track-metadata-position="body" }为笔记本 2 提供了概览和上下文。此笔记本提供了一个用于预测婴儿体重的机器学习模型示例。在此示例中，TensorFlow 模型是使用 Keras API 实现的。此模型使用由先前所述的 `tf.Transform` 预处理流水线产生的数据和工件。

### 运行笔记本 2

1. 在 JupyterLab 界面中，点击 **File &gt; Open from path**，然后输入以下路径：

    ```sh
    training-data-analyst/blogs/babyweight_tft/babyweight_tft_keras_02.ipynb
    ```

2. 点击 **Edit &gt; Clear all outputs**。

3. 在 **Install required packages** 部分中，执行第一个单元以运行 `pip install tensorflow-transform` 命令。

    输出的最后一部分如下所示：

    ```none{:.devsite-disable-click-to-copy}
    Successfully installed ...
    Note: you may need to restart the kernel to use updated packages.
    ```

    您可以忽略输出中的依赖项错误。

4. 在 **Kernel** 菜单中，选择 **Restart Kernel**。

5. 执行 **Confirm the installed packages** 和 **Create setup.py to install packages to Dataflow containers** 部分中的单元。

6. 在 **Set global flags** 部分中，在 `PROJECT` 和 `BUCKET` 旁边，将 <code>your-project</code> 替换为您的 Cloud 项目 ID，然后执行该单元。

7. 执行所有剩余单元，直到笔记本中的最后一个单元。有关在每个单元中所执行操作的信息，请参阅笔记本中的说明。

### 模型创建概览

创建模型的步骤如下：

1. 使用存储在 `transformed_metadata` 目录中的架构信息创建特征列。
2. 使用 Keras API 创建 Wide &amp; Deep 模型，并使用特征列作为模型的输入。
3. 创建 `tfrecords_input_fn` 函数以使用转换工件读取并解析训练和评估数据。
4. 训练和评估模型。
5. 通过定义附加了 `transform_fn` 计算图的 `serving_fn` 函数导出经过训练的模型。
6. 使用 [`saved_model_cli`](https://www.tensorflow.org/guide/saved_model) 工具检查导出的模型。
7. 使用导出的模型进行预测。

本文档未介绍如何构建模型，因此也未详细讨论如何构建或训练模型。但是，以下部分介绍了如何使用 `tf.Transform` 过程生成的 `transform_metadata` 目录中存储的信息来创建模型的特征列。本文档还介绍了在导出模型以进行应用时如何在 `serving_fn` 函数中使用 `transform_fn` 计算图，此计算图同样由 `tf.Transform` 过程生成。

### 使用模型训练过程中生成的转换工件

训练 TensorFlow 模型时，您可以使用先前数据处理步骤中生成的已转换 `train` 和 `eval` 对象。这些对象以 TFRecord 格式存储为分片文件。上一步中生成的 `transformed_metadata` 目录中的架构信息可用于解析数据（`tf.train.Example` 对象）以提供模型进行训练和评估。

#### 解析数据

由于您读取 TFRecord 格式的文件以向模型馈送训练和评估数据，您需要解析文件中的每个 `tf.train.Example` 对象，以创建特征（张量）字典。这样可以确保使用用作模型训练和评估接口的特征列将特征映射到模型输入层。要解析数据，请使用从上一步中生成的工件创建的 `TFTransformOutput` 对象：

1. 根据先前预处理步骤中生成并保存的工件创建 `TFTransformOutput` 对象，如[保存计算图](#save_the_graph){: track-type="solution" track-name="internalLink" track-metadata-position="body" }部分中所述：

    ```py
    tf_transform_output = tft.TFTransformOutput(TRANSFORM_ARTEFACTS_DIR)
    ```

2. 从 `TFTransformOutput` 对象提取 `feature_spec` 对象：

    ```py
    tf_transform_output.transformed_feature_spec()
    ```

3. 如在 `tfrecords_input_fn` 函数中一样，使用 `feature_spec` 对象指定 `tf.train.Example` 对象中包含的特征：

    ```py
    def tfrecords_input_fn(files_name_pattern, batch_size=512):

        tf_transform_output = tft.TFTransformOutput(TRANSFORM_ARTEFACTS_DIR)
        TARGET_FEATURE_NAME = 'weight_pounds'

        batched_dataset = tf.data.experimental.make_batched_features_dataset(
            file_pattern=files_name_pattern,
            batch_size=batch_size,
            features=tf_transform_output.transformed_feature_spec(),
            reader=tf.data.TFRecordDataset,
            label_key=TARGET_FEATURE_NAME,
            shuffle=True).prefetch(tf.data.experimental.AUTOTUNE)

        return batched_dataset
    ```

#### 创建特征列

流水线在描述模型训练和评估所需的转换数据架构的 `transformed_metadata` 目录中生成架构信息。架构包含特征名称和数据类型，如下所示：

- `gestation_weeks_scaled`（类型：`FLOAT`）
- `is_male_index`（类型：`INT`、is_categorical：`True`）
- `is_multiple_index`（类型：`INT`、is_categorical：`True`）
- `mother_age_bucketized`（类型：`INT`、is_categorical：`True`）
- `mother_age_log`（类型：`FLOAT`）
- `mother_age_normalized`（类型：`FLOAT`）
- `mother_race_index`（类型：`INT`、is_categorical：`True`）
- `weight_pounds`（类型：`FLOAT`）

要查看此信息，请使用以下命令：

```sh
transformed_metadata = tft.TFTransformOutput(TRANSFORM_ARTEFACTS_DIR).transformed_metadata
transformed_metadata.schema
```

以下代码显示了如何使用特征名称创建特征列：

```py
def create_wide_and_deep_feature_columns():

    deep_feature_columns = []
    wide_feature_columns = []
    inputs = {}
    categorical_columns = {}

    # Select features you've checked from the metadata
    # Categorical features are associated with the vocabulary size (starting from 0)
    numeric_features = ['mother_age_log', 'mother_age_normalized', 'gestation_weeks_scaled']
    categorical_features = [('is_male_index', 1), ('is_multiple_index', 1),
                            ('mother_age_bucketized', 4), ('mother_race_index', 10)]

    for feature in numeric_features:
        deep_feature_columns.append(tf.feature_column.numeric_column(feature))
        inputs[feature] = layers.Input(shape=(), name=feature, dtype='float32')

    for feature, vocab_size in categorical_features:
        categorical_columns[feature] = (
            tf.feature_column.categorical_column_with_identity(feature, num_buckets=vocab_size+1))
        wide_feature_columns.append(tf.feature_column.indicator_column(categorical_columns[feature]))
        inputs[feature] = layers.Input(shape=(), name=feature, dtype='int64')

    mother_race_X_mother_age_bucketized = tf.feature_column.crossed_column(
        [categorical_columns['mother_age_bucketized'],
         categorical_columns['mother_race_index']],  55)
    wide_feature_columns.append(tf.feature_column.indicator_column(mother_race_X_mother_age_bucketized))

    mother_race_X_mother_age_bucketized_embedded = tf.feature_column.embedding_column(
        mother_race_X_mother_age_bucketized, 5)
    deep_feature_columns.append(mother_race_X_mother_age_bucketized_embedded)

    return wide_feature_columns, deep_feature_columns, inputs
```

此代码会为数值特征创建一个 `tf.feature_column.numeric_column` 列，并为分类特征创建一个 `tf.feature_column.categorical_column_with_identity` 列。

此外，您还可以创建扩展特征列，如本系列第一部分的[选项 C：TensorFlow](/architecture/data-preprocessing-for-ml-with-tf-transform-pt1#option_c_tensorflow){: track-type="solution" track-name="internalLink" track-metadata-position="body" } 中所述。在本系列采用的示例中，通过使用 `tf.feature_column.crossed_column` 特征列交叉 `mother_race` 和 `mother_age_bucketized` 特征创建了一个新特征 `mother_race_X_mother_age_bucketized`。此交叉特征的低维度密集表示法是通过使用 `tf.feature_column.embedding_column` 特征列创建的。

下面的图 5 显示了转换后的数据以及如何使用转换后的元数据来定义和训练 TensorFlow 模型：

<figure id="training-tf-with-transformed-data">
  <img src="images/data-preprocessing-for-ml-with-tf-transform-training-tf-model-with-transformed-data.svg"
    alt="Training the TensorFlow model with transformed data.">
  <figcaption><b>Figure 5.</b> Training the TensorFlow model with
    the transformed data.</figcaption>
</figure>

### 导出模型以应用预测

使用 Keras API 训练 TensorFlow 模型后，可以将经过训练的模型导出为 SavedModel 对象，这样它就能应用新的数据点来进行预测。导出模型时，必须定义其接口，即应用期间需要的输入特征架构。此输入特征架构是在 `serving_fn` 函数中定义的，如以下代码所示：

```py{:.devsite-disable-click-to-copy}
def export_serving_model(model, output_dir):

    tf_transform_output = tft.TFTransformOutput(TRANSFORM_ARTEFACTS_DIR)
    # The layer has to be saved to the model for Keras tracking purposes.
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serveing_fn(uid, is_male, mother_race, mother_age, plurality, gestation_weeks):
        features = {
            'is_male': is_male,
            'mother_race': mother_race,
            'mother_age': mother_age,
            'plurality': plurality,
            'gestation_weeks': gestation_weeks
        }
        transformed_features = model.tft_layer(features)
        outputs = model(transformed_features)
        # The prediction results have multiple elements in general.
        # But we need only the first element in our case.
        outputs = tf.map_fn(lambda item: item[0], outputs)

        return {'uid': uid, 'weight': outputs}

    concrete_serving_fn = serveing_fn.get_concrete_function(
        tf.TensorSpec(shape=[None], dtype=tf.string, name='uid'),
        tf.TensorSpec(shape=[None], dtype=tf.string, name='is_male'),
        tf.TensorSpec(shape=[None], dtype=tf.string, name='mother_race'),
        tf.TensorSpec(shape=[None], dtype=tf.float32, name='mother_age'),
        tf.TensorSpec(shape=[None], dtype=tf.float32, name='plurality'),
        tf.TensorSpec(shape=[None], dtype=tf.float32, name='gestation_weeks')
    )
    signatures = {'serving_default': concrete_serving_fn}

    model.save(output_dir, save_format='tf', signatures=signatures)
```

应用期间，模型需要原始格式的数据点（即转换前的原始特征）。因此，`serving_fn` 函数会接收原始特征并将其作为 Python 字典存储在 `features` 对象中。但是，如前面所述，经过训练的模型需要已转换架构中的数据点。要将原始特征转换为模型接口需要的 `transformed_features` 对象，请执行以下步骤将保存的 `transform_fn` 计算图应用于 `features` 对象：

1. 根据在先前预处理步骤中生成并保存的工件创建 `TFTransformOutput` 对象：

    ```py
    tf_transform_output = tft.TFTransformOutput(TRANSFORM_ARTEFACTS_DIR)
    ```

2. 根据 `TFTransformOutput` 对象创建 `TransformFeaturesLayer` 对象：

    ```py
    model.tft_layer = tf_transform_output.transform_features_layer()
    ```

3. 使用 `TransformFeaturesLayer` 对象应用 `transform_fn` 计算图：

    ```py
    transformed_features = model.tft_layer(features)
    ```

下面的图 6 说明了导出模型以进行应用的最后一步：

<figure id="exporting-model-for-serving-with-transform_fn">
  <img src="images/data-preprocessing-for-ml-with-tf-transform-exporting-model-for-serving-with-transform_fn.svg"
    alt="Exporting the model for serving with the transform_fn graph attached.">
  <figcaption><b>Figure 6.</b> Exporting the model for serving with the
    <code>transform_fn</code> graph attached.</figcaption>
</figure>

## 训练并使用模型进行预测

您可以通过执行笔记本的各个单元在本地训练模型。有关如何使用 Vertex AI Training 规模化打包代码和训练模型的示例，请参阅 Google Cloud [cloudml-samples](https://github.com/GoogleCloudPlatform/cloudml-samples){: .external } GitHub 仓库中的示例和指南。

使用 `saved_model_cli` 工具检查导出的 SavedModel 对象时，您会看到签名定义 `signature_def` 的 `inputs` 元素包含原始特征，如以下示例所示：

```py{:.devsite-disable-click-to-copy}
signature_def['serving_default']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['gestation_weeks'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1)
        name: serving_default_gestation_weeks:0
    inputs['is_male'] tensor_info:
        dtype: DT_STRING
        shape: (-1)
        name: serving_default_is_male:0
    inputs['mother_age'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1)
        name: serving_default_mother_age:0
    inputs['mother_race'] tensor_info:
        dtype: DT_STRING
        shape: (-1)
        name: serving_default_mother_race:0
    inputs['plurality'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1)
        name: serving_default_plurality:0
    inputs['uid'] tensor_info:
        dtype: DT_STRING
        shape: (-1)
        name: serving_default_uid:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['uid'] tensor_info:
        dtype: DT_STRING
        shape: (-1)
        name: StatefulPartitionedCall_6:0
    outputs['weight'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1)
        name: StatefulPartitionedCall_6:1
  Method name is: tensorflow/serving/predict
```

笔记本的其余单元介绍了如何使用导出的模型进行本地预测，以及如何使用 Vertex AI Prediction 将模型部署为微服务。需要强调的是，在这两种情况下，输入（示例）数据点都位于原始架构中。

## 清理

为避免因本教程中使用的资源导致您的 Google Cloud 帐号产生额外费用，请删除包含这些资源的项目。

### 删除项目

  <aside class="caution">     <strong>警告</strong>：删除项目会造成以下影响：     <ul>       <li>         <strong>项目中的所有内容都会被删除。</strong> 如果您将现有项目用于本教程，当您删除此项目后，也会删除您已在此项目中完成的任何其他工作。       </li>       <li>         <strong>自定义项目 ID 丢失。</strong>         创建此项目时，您可能创建了要在将来使用的自定义项目 ID。要保留使用该项目 ID 的网址（如 <code translate="no" dir="ltr">appspot.com</code> 网址），请删除项目内的选定资源，而不是删除整个项目。       </li>     </ul>     <p>       如果您计划浏览多个教程和快速入门，重复使用项目可以帮助您避免超出项目配额限制。     </p></aside>


1. 在 Google Cloud 控制台中，转到 **Manage resources** 页面。

    [转到 Manage resources](https://console.cloud.google.com/iam-admin/projects){: class="button button-primary" target="console" track-type="solution" track-name="consoleLink" track-metadata-position="body" }

2. 在项目列表中，选择要删除的项目，然后点击 **Delete**。

3. 在对话框中输入项目 ID，然后点击 **Shut down** 来删除项目。

## 后续步骤

- 要了解预处理数据以在 Google Cloud 上进行机器学习的概念、挑战和选项，请参阅本系列文章中的第一篇文章：[预处理数据以进行机器学习：选项和建议](../guide/tft_bestpractices)。
- 有关如何在 Dataflow 上实现、打包和运行 tf.Transform 流水线的更多信息 ，请参阅[使用 Census 数据集预测收入](https://github.com/GoogleCloudPlatform/cloudml-samples/tree/master/census/tftransformestimator){: .external }示例。
- 学习 Coursera 专项课程，了解如何[在 Google Cloud 上使用 TensorFlow](https://www.coursera.org/specializations/machine-learning-tensorflow-gcp){: .external } 进行机器学习。
- 阅读[机器学习规则](https://developers.google.com/machine-learning/guides/rules-of-ml/){: .external }，了解机器学习工程中的最佳做法。
- 如需查看更多参考架构、图表和最佳做法，请浏览[云架构中心](https://cloud.google.com/architecture)。
