# 使用 TF Profiler 分析 `tf.data` 性能

## 概述

本文假定用户熟悉 TensorFlow [Profiler](https://tensorflow.google.cn/guide/profiler) 和 [`tf.data`](https://www.tensorflow.org/guide/data)，旨在提供包含示例的分步说明来帮助用户诊断和修复输入流水线性能问题。

首先，请收集您的 TensorFlow 作业的分析。有关如何执行此操作的说明适用于 [CPU/GPU](https://tensorflow.google.cn/guide/profiler#collect_performance_data) 和 [Cloud TPU](https://cloud.google.com/tpu/docs/cloud-tpu-tools#capture_profile)。

![TensorFlow Trace Viewer](images/data_performance_analysis/trace_viewer.png "The trace viewer page of the TensorFlow Profiler")

下面详细介绍的分析工作流侧重于 Profiler 中的 Trace Viewer 工具。此工具提供了一个时间线，可显示 TensorFlow 程序所执行运算的持续时间，并让您可以确定哪些运算的执行时间最长。有关 Trace Viewer 的更多信息，请查看 TF Profiler 指南的[此部分](https://tensorflow.google.cn/guide/profiler#trace_viewer)。通常，`tf.data` 事件会出现在主机 CPU 时间线上。

## 分析工作流

*请遵循下面的工作流。如果您想帮助我们改进此工作流，请使用“comp:data”标签[创建一个 Github 问题](https://github.com/tensorflow/tensorflow/issues/new/choose)。*

### 1. 您的 `tf.data` 流水线产生数据的速度是否足够快？

首先确定输入流水线是否为您的 TensorFlow 程序的瓶颈。

为此，请在 Trace Viewer 中查找 `IteratorGetNext::DoCompute` 运算。通常，您希望在步骤开始时就能看到这些信息。这些切片表示输入流水线在被请求时产生一批元素所用的时间。如果您在使用 Keras 或者在 `tf.function` 中迭代您的数据集，应当可以在 `tf_data_iterator_get_next` 线程中找到切片。

请注意，如果您正在使用[分配策略](https://tensorflow.google.cn/guide/distributed_training)，则可能会看到 <br>`IteratorGetNextAsOptional::DoCompute` 事件，而不是 `IteratorGetNext::DoCompute`（自 TF 2.3 起）。

![image](images/data_performance_analysis/get_next_fast.png "If your IteratorGetNext::DoCompute calls return quickly, `tf.data` is not your bottleneck.")

**如果调用迅速返回 (&lt;= 50 us)**，这表示您的数据在被请求时可用。输入流水线不是您的瓶颈；请参阅 [Profiler 指南](https://tensorflow.google.cn/guide/profiler)，获得更多常规性能分析提示。

![image](images/data_performance_analysis/get_next_slow.png "If your IteratorGetNext::DoCompute calls return slowly, `tf.data` is not producing data quickly enough.")

**如果调用缓慢返回**，则表示 `tf.data` 无法跟上使用者的请求速度。请继续阅读下一部分。

### 2. 您是否在预提取数据？

要提高输入流水线性能，最佳做法是在 `tf.data` 流水线的末尾插入 `tf.data.Dataset.prefetch` 转换。此转换会将输入流水线的预处理计算与模型计算的下一步重叠，当您训练模型时，需要该转换来实现最佳的输入流水线性能。如果您正在预提取数据，应当可以在 `IteratorGetNext::DoCompute` 运算所在的线程上看到 `Iterator::Prefetch` 切片。

![image](images/data_performance_analysis/prefetch.png "If you're prefetching data, you should see a `Iterator::Prefetch` slice in the same stack as the `IteratorGetNext::DoCompute` op.")

**如果您的流水线末尾没有 `prefetch`**，则应相应添加一个。有关 `tf.data` 性能建议的更多信息，请参阅 [tf.data 性能指南](https://tensorflow.google.cn/guide/data_performance#prefetching)。

**如果您已经在预提取数据**，并且输入流水线仍然是您的瓶颈，请继续阅读下一部分以进一步分析性能。

### 3. 您是否达到较高的 CPU 利用率？

`tf.data` 通过尝试最大程度地利用可用资源来实现高吞吐量。通常，即使在 GPU 或 TPU 等加速器上运行模型，`tf.data` 流水线仍在 CPU 上运行。您可以使用 [sar](https://linux.die.net/man/1/sar) 和 [htop](https://en.wikipedia.org/wiki/Htop) 等工具，或者在 [Cloud Monitoring 控制台](https://cloud.google.com/monitoring/docs/monitoring_in_console)中（如果在 GCP 上运行）检查 CPU 利用率。

**如果利用率较低**，这表明您的输入流水线可能没有充分利用主机 CPU。您应当参考 [tf.data 性能指南](https://tensorflow.google.cn/guide/data_performance)来了解最佳做法。如果您采用了最佳做法，但利用率和吞吐量仍然很低，请阅读下面的[瓶颈分析](#4_bottleneck_analysis)部分。

**如果利用率接近资源限制**，为了进一步提高性能，您需要提高输入流水线的效率（例如，避免不必要的计算）或者卸载计算。

通过避免在 `tf.data` 中进行不必要的计算，您可以提高输入流水线的效率。一种实现方法是在计算密集型工作之后插入 [`tf.data.Dataset.cache`](https://www.tensorflow.org/guide/data_performance#caching) 转换，前提是您的数据适合装入内存；这以增加内存使用量为代价减少了计算量。另外，在 `tf.data` 中禁用运算内并行有可能将效率提高 10% 以上，这可以通过在输入流水线上设置以下选项来实现：

```python
dataset = ...
options = tf.data.Options()
options.experimental_threading.max_intra_op_parallelism = 1
dataset = dataset.with_options(options)
```

### 4. 瓶颈分析

下一部分将详细介绍如何在 Trace Viewer 中读取 `tf.data` 事件，以了解瓶颈环节以及可能的缓解策略。

#### 了解 Profiler 中的 `tf.data` 事件

Profiler 中的每个 `tf.data` 事件都具有名称<br> `Iterator::<Dataset>`，其中 `<Dataset>` 是数据集源或转换的名称。此外，每个事件还具有长名称 `Iterator::<Dataset_1>::...::<Dataset_n>`，点击 `tf.data` 事件可以看到长名称。在长名称中，`<Dataset_n>` 与（短）名称中的 `<Dataset>` 匹配，长名称中的其他数据集表示下游转换。

![image](images/data_performance_analysis/map_long_name.png "tf.data.Dataset.range(10).map(lambda x: x).repeat(2).batch(5)")

例如，上面的屏幕截图通过以下代码生成：

```python
dataset = tf.data.Dataset.range(10)
dataset = dataset.map(lambda x: x)
dataset = dataset.repeat(2)
dataset = dataset.batch(5)
```

此处，`Iterator::Map` 事件的长名称为 `Iterator::BatchV2::FiniteRepeat::Map`。请注意，数据集名称可能与 Python API 略有不同（例如，名称为 FiniteRepeat 而非<br>Repeat），但应足够直观以方便解析。

##### 同步和异步转换

对于同步 `tf.data` 转换（例如 `Batch` 和 `Map`），您将在同一线程上看到来自上游转换的事件。在上面的示例中，由于使用的所有转换都是同步的，所有事件都出现在同一线程上。

对于异步转换（例如 `Prefetch`、`ParallelMap`、`ParallelInterleave` 和 `MapAndBatch`），来自上游转换的事件将位于其他线程上。在此类情况下，“长名称”可以帮助您确定事件对应于流水线中的哪个转换。

![image](images/data_performance_analysis/async_long_name.png "tf.data.Dataset.range(10).map(lambda x: x).repeat(2).batch(5).prefetch(1)")

例如，上面的屏幕截图通过以下代码生成：

```python
dataset = tf.data.Dataset.range(10)
dataset = dataset.map(lambda x: x)
dataset = dataset.repeat(2)
dataset = dataset.batch(5)
dataset = dataset.prefetch(1)
```

在这里，`Iterator::Prefetch` 事件位于 `tf_data_iterator_get_next` 线程上。由于 `Prefetch` 是异步的，其输入事件 (`BatchV2`) 将位于其他线程上，并且可以通过搜索长名称 `Iterator::Prefetch::BatchV2` 来定位。在这种情况下，它们位于 `tf_data_iterator_resource` 线程上。根据其长名称，您可以推断出 `BatchV2` 位于 `Prefetch` 的上游。此外，`BatchV2` 事件的 `parent_id` 将与 `Prefetch` 事件的 ID 相匹配。

#### 确定瓶颈

通常，要确定输入流水线中的瓶颈，需要检查整个输入流水线，从最外层转换一直到源。从流水线中的最终转换开始，递归到上游转换，直至找到缓慢的转换或到达源数据集，例如 `TFRecord`。在上面的示例中，您将从 `Prefetch` 开始，随后向上游递归到 `BatchV2`、`FiniteRepeat` 和 `Map`，最后是 `Range`。

一般而言，缓慢的转换对应于事件长但输入事件短的转换。下面给出了一些示例。

请注意，大多数主机输入流水线中的最终（最外层）转换都是 `Iterator::Model` 事件。`tf.data` 运行时会自动引入模型转换，该转换用于检测和自动调整输入流水线性能。

如果您的作业正在使用[分配策略](https://tensorflow.google.cn/guide/distributed_training)，则 Trace Viewer 将包含与设备输入流水线相对应的其他事件。设备流水线的最外层转换（嵌套在 `IteratorGetNextOp::DoCompute` 或 `IteratorGetNextAsOptionalOp::DoCompute` 下）将是一个具有上游 `Iterator::Generator` 事件的 `Iterator::Prefetch` 事件。您可以通过搜索 `Iterator::Model` 事件来找到相应的主机流水线。

##### 示例 1

![image](images/data_performance_analysis/example_1_cropped.png "Example 1")

上面的屏幕截图通过以下输入流水线生成：

```python
dataset = tf.data.TFRecordDataset(filename)
dataset = dataset.map(parse_record)
dataset = dataset.batch(32)
dataset = dataset.repeat()
```

在屏幕截图中，您可以看到 (1) `Iterator::Map` 事件很长，但 (2) 其输入事件 (`Iterator::FlatMap`) 迅速返回。这表明依序 Map 转换是瓶颈。

请注意，在屏幕截图中，`InstantiatedCapturedFunction::Run` 事件对应于执行映射函数所用的时间。

##### 示例 2

![image](images/data_performance_analysis/example_2_cropped.png "Example 2")

上面的屏幕截图通过以下输入流水线生成：

```python
dataset = tf.data.TFRecordDataset(filename)
dataset = dataset.map(parse_record, num_parallel_calls=2)
dataset = dataset.batch(32)
dataset = dataset.repeat()
```

此示例与上面的示例相似，但使用 ParallelMap 代替了 Map。我们在这里可以看到 (1) `Iterator::ParallelMap` 事件很长，但 (2) 其输入事件 `Iterator::FlatMap`（位于其他线程上，因为 ParallelMap 是异步的）很短。这表明 ParallelMap 转换是瓶颈。

#### 解决瓶颈

##### 源数据集

如果您已确定数据集源为瓶颈（例如，从 TFRecord 文件中读取），则可以通过并行处理数据提取来提高性能。为此，请确保将您的数据分片到多个文件中，并使用 `tf.data.Dataset.interleave`（将 `num_parallel_calls` 参数设置为 `tf.data.experimental.AUTOTUNE`）。如果确定性对您的程序不重要，则自 TF 2.2 起，您可以通过在 `tf.data.Dataset.interleave` 上设置 `deterministic=False` 标记来进一步提高性能。例如，如果您正在从 TFRecord 文件中读取，则可以执行以下操作：

```python
dataset = tf.data.Dataset.from_tensor_slices(filenames)
dataset = dataset.interleave(tf.data.TFRecordDataset,
  num_parallel_calls=tf.data.AUTOTUNE,
  deterministic=False)
```

请注意，分片文件应当足够大，以便分摊打开文件的开销。有关并行数据提取的更多详细信息，请参阅 `tf.data` 性能指南的[此部分](https://www.tensorflow.org/guide/data_performance#parallelizing_data_extraction)。

##### 转换数据集

如果您已确定中间 `tf.data` 转换为瓶颈，则可以通过并行处理转换或者[缓存计算](https://tensorflow.google.cn/guide/data_performance#caching)来解决此瓶颈，前提是您的数据适合装入内存并且适当。某些转换（例如 `Map`）具有并行的对应项；<a data-md-type="raw_html" href="https://tensorflow.google.cn/guide/data_performance#parallelizing_data_transformation">`tf.data` 性能指南演示了</a>如何并行处理这些对应项。其他转换（例如 `Filter`、`Unbatch` 和 `Batch`）本质上是依序的；您可以通过引入“外部并行”来并行处理它们。例如，假设您的输入流水线最初如下所示，其中瓶颈为 `Batch`：

```python
filenames = tf.data.Dataset.list_files(file_path, shuffle=is_training)
dataset = filenames_to_dataset(filenames)
dataset = dataset.batch(batch_size)
```

您可以通过在分片输入上运行输入流水线的多个副本并将结果组合来引入“外部并行”：

```python
filenames = tf.data.Dataset.list_files(file_path, shuffle=is_training)

def make_dataset(shard_index):
  filenames = filenames.shard(NUM_SHARDS, shard_index)
  dataset = filenames_to_dataset(filenames)
  Return dataset.batch(batch_size)

indices = tf.data.Dataset.range(NUM_SHARDS)
dataset = indices.interleave(make_dataset,
                             num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.prefetch(tf.data.AUTOTUNE)
```

## 其他资源

- 关于如何编写高性能 `tf.data` 输入流水线的 [tf.data 性能指南](https://tensorflow.google.cn/guide/data_performance)
- [TensorFlow 内部视频：`tf.data` 最佳做法](https://www.youtube.com/watch?v=ZnukSLKEw34)
- [Profiler 指南](https://tensorflow.google.cn/guide/profiler)
- [Colab 中的 Profiler 教程](https://tensorflow.google.cn/tensorboard/tensorboard_profiling_keras)
