# 使用 Apache Beam 生成大型数据集

有些数据集因数据量过大而无法在一台计算机上进行处理。`tfds` 支持使用 [Apache Beam](https://beam.apache.org/) 在多台计算机上生成数据。

本文档分为两部分：

- 面向想要生成现有 Beam 数据集的用户
- 面向想要创建新的 Beam 数据集的开发者

目录：

- [生成 Beam 数据集](#generating-a-beam-dataset)
    - [在 Google Cloud Dataflow 上生成](#on-google-cloud-dataflow)
    - [在本地生成](#locally)
    - [使用自定义脚本生成](#with-a-custom-script)
- [实现 Beam 数据集](#implementing-a-beam-dataset)
    - [前提条件](#prerequisites)
    - [说明](#instructions)
    - [示例](#example)
    - [运行您的流水线](#run-your-pipeline)

## 生成 Beam 数据集

以下是在云端或在本地生成 Beam 数据集的不同示例。

**警告**：使用 `tensorflow_datasets.scripts.download_and_prepare` 脚本生成数据集时，请确保指定要生成的数据集配置，否则将默认生成所有现有配置。例如，对于 [wikipedia](https://www.tensorflow.org/datasets/catalog/wikipedia)，请使用 `--dataset=wikipedia/20200301.en` 而非 `--dataset=wikipedia`。

### 在 Google Cloud Dataflow 上生成

要使用 [Google Cloud Dataflow](https://cloud.google.com/dataflow/) 运行流水线并利用分布式计算的优势，请首先遵循[快速入门说明](https://cloud.google.com/dataflow/docs/quickstarts/quickstart-python)。

设置好环境后，您可以使用 [GCS](https://cloud.google.com/storage/) 上的数据目录并为 `--beam_pipeline_options` 标记指定[所需的选项](https://cloud.google.com/dataflow/docs/guides/specifying-exec-params#configuring-pipelineoptions-for-execution-on-the-cloud-dataflow-service)来运行 `download_and_prepare` 脚本。

为了便于启动脚本，建议您使用自己的 GCP/GCS 设置和您要生成的数据集的实际值来定义以下变量：

```sh
DATASET_NAME=<dataset-name>
DATASET_CONFIG=<dataset-config>
GCP_PROJECT=my-project-id
GCS_BUCKET=gs://my-gcs-bucket
```

然后，您需要创建文件来告知 Dataflow 在工作进程上安装 `tfds`：

```sh
echo "tensorflow_datasets[$DATASET_NAME]" > /tmp/beam_requirements.txt
```

如果您使用的是 `tfds-nightly`，并且自上次发布以来数据集进行了更新，请确保从 `tfds-nightly` 回送数据。

```sh
echo "tfds-nightly[$DATASET_NAME]" > /tmp/beam_requirements.txt
```

最后，您可以使用以下命令启动作业：

```sh
python -m tensorflow_datasets.scripts.download_and_prepare \
  --datasets=$DATASET_NAME/$DATASET_CONFIG \
  --data_dir=$GCS_BUCKET/tensorflow_datasets \
  --beam_pipeline_options=\
"runner=DataflowRunner,project=$GCP_PROJECT,job_name=$DATASET_NAME-gen,"\
"staging_location=$GCS_BUCKET/binaries,temp_location=$GCS_BUCKET/temp,"\
"requirements_file=/tmp/beam_requirements.txt"
```

### 在本地生成

要使用默认的 Apache Beam 运行程序在本地运行脚本，该命令与其他数据集的命令相同：

```sh
python -m tensorflow_datasets.scripts.download_and_prepare \
  --datasets=my_new_dataset
```

**警告**：Beam 数据集可能非常**庞大**（太字节），并且生成数据集会占用大量资源（在本地计算机上可能需要数周）。建议使用分布式环境生成数据集。请参阅 [Apache Beam 文档](https://beam.apache.org/)以查看受支持的运行时列表。

### 使用自定义脚本生成

要在 Beam 上生成数据集，API 与其他数据集相同，但需要将 Beam 选项或运行程序传递给 `DownloadConfig`。

```py
# If you are running on Dataflow, Spark,..., you may have to set-up runtime
# flags. Otherwise, you can leave flags empty [].
flags = ['--runner=DataflowRunner', '--project=<project-name>', ...]

# To use Beam, you have to set at least one of `beam_options` or `beam_runner`
dl_config = tfds.download.DownloadConfig(
    beam_options=beam.options.pipeline_options.PipelineOptions(flags=flags)
)

data_dir = 'gs://my-gcs-bucket/tensorflow_datasets'
builder = tfds.builder('wikipedia/20190301.en', data_dir=data_dir)
builder.download_and_prepare(
    download_dir=FLAGS.download_dir,
    download_config=dl_config,
)
```

## 实现 Beam 数据集

### 前提条件

为了编写 Apache Beam 数据集，您应该熟悉以下概念：

- 熟悉 [`tfds` 数据集创建指南](https://github.com/tensorflow/datasets/tree/master/docs/add_dataset.md)，因为其中大部分内容仍适用于 Beam 数据集。
- 借助 [Beam 编程指南](https://beam.apache.org/documentation/programming-guide/)了解 Apache Beam。
- 如果要使用 Cloud Dataflow 生成数据集，请阅读 [Google Cloud 文档](https://cloud.google.com/dataflow/docs/quickstarts/quickstart-python)和 [Apache Beam 依赖项指南](https://beam.apache.org/documentation/sdks/python-pipeline-dependencies/)。

### 说明

如果您熟悉[数据集创建指南](https://github.com/tensorflow/datasets/tree/master/docs/add_dataset.md)，则仅需进行一些修改即可添加 Beam 数据集：

- 您的 `DatasetBuilder` 将继承自 `tfds.core.BeamBasedBuilder` 而非 `tfds.core.GeneratorBasedBuilder`。
- Beam 数据集应实现抽象方法 `_build_pcollection(self, **kwargs)` 而非 `_generate_examples(self, **kwargs)` 方法。`_build_pcollection` 应返回 `beam.PCollection` 以及与拆分相关联的示例。
- Beam 数据集与其他数据集的单元测试编写方法相同。

其他注意事项：

- 使用 `tfds.core.lazy_imports` 导入 Apache Beam。通过使用惰性依赖关系，用户在数据集生成后仍可以读取数据集，而不必安装 Beam。
- 使用 Python 闭包时要小心。在运行流水线时，使用 `pickle` 序列化 `beam.Map` 和 `beam.DoFn` 函数，并将其发送给所有工作进程。这会产生错误；例如，如果您在函数中使用了在函数外部声明的可变对象，则可能会遇到 `pickle` 错误或意外行为。解决方法通常是避免改变封闭的对象。
- 在 Beam 流水线中可以对 `DatasetBuilder` 使用方法。但是，在 pickle 过程中类被序列化的方式以及在创建过程中对特征所做的更改最多将被忽略。

### 示例

以下为 Beam 数据集的示例。要了解更为复杂的实际示例，请参见 [`Wikipedia` 数据集](https://github.com/tensorflow/datasets/tree/master/tensorflow_datasets/text/wikipedia.py)。

```python
class DummyBeamDataset(tfds.core.BeamBasedBuilder):

  VERSION = tfds.core.Version('1.0.0')

  def _info(self):
    return tfds.core.DatasetInfo(
        builder=self,
        features=tfds.features.FeaturesDict({
            'image': tfds.features.Image(shape=(16, 16, 1)),
            'label': tfds.features.ClassLabel(names=['dog', 'cat']),
        }),
    )

  def _split_generators(self, dl_manager):
    ...
    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            gen_kwargs=dict(file_dir='path/to/train_data/'),
        ),
        splits_lib.SplitGenerator(
            name=tfds.Split.TEST,
            gen_kwargs=dict(file_dir='path/to/test_data/'),
        ),
    ]

  def _build_pcollection(self, pipeline, file_dir):
    """Generate examples as dicts."""
    beam = tfds.core.lazy_imports.apache_beam

    def _process_example(filename):
      # Use filename as key
      return filename, {
          'image': os.path.join(file_dir, filename),
          'label': filename.split('.')[1],  # Extract label: "0010102.dog.jpeg"
      }

    return (
        pipeline
        | beam.Create(tf.io.gfile.listdir(file_dir))
        | beam.Map(_process_example)
    )
```

### 运行您的流水线

要运行流水线，请参阅以上部分内容。

**警告**：首次运行数据集以注​​册下载内容时，请勿忘记将注册校验和 `--register_checksums` 标记添加到 `download_and_prepare` 脚本中。

```sh
python -m tensorflow_datasets.scripts.download_and_prepare \
  --register_checksums \
  --datasets=my_new_dataset
```
