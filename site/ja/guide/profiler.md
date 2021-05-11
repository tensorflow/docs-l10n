# プロファイラを使用した TensorFlow のパフォーマンス最適化

[TOC]

Use the tools available with the Profiler to track the performance of your TensorFlow models. See how your model performs on the host (CPU), the device (GPU), or on a combination of both the host and device(s).

Profiling helps you understand the hardware resource consumption (time and memory) of the various TensorFlow operations (ops) in your model and resolve performance bottlenecks and ultimately, make the model execute faster.

This guide will walk you through how to install the Profiler, the various tools available, the different modes of how the Profiler collects performance data, and some recommended best practices to optimize model performance.

If you want to profile your model performance on Cloud TPUs, refer to the [Cloud TPU guide](https://cloud.google.com/tpu/docs/cloud-tpu-tools#capture_profile).

## プロファイラのインストールと GPU の要件

Install the Profiler by downloading and running the [`install_and_run.py`](https://raw.githubusercontent.com/tensorflow/profiler/master/install_and_run.py) script from the [GitHub repository](https://github.com/tensorflow/profiler).

GPU 上でプロファイリングを実行するには、次の手順を行う必要があります。

1. Meet the NVIDIA® GPU drivers and CUDA® Toolkit requirements listed on [TensorFlow GPU support software requirements](https://www.tensorflow.org/install/gpu#linux_setup).

2. CUPTI がパスに存在することを確認します。

    ```shell
    /sbin/ldconfig -N -v $(sed 's/:/ /g' <<< $LD_LIBRARY_PATH) | \
    grep libcupti
    ```

If you don't have CUPTI on the path, prepend its installation directory to the `$LD_LIBRARY_PATH` environment variable by running:

```shell
export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
```

Run the `ldconfig` command above again to verify that the CUPTI library is found.

### 特権の問題を解消する

When you run profiling with CUDA® Toolkit in a Docker environment or on Linux, you may encounter issues related to insufficient CUPTI privileges (`CUPTI_ERROR_INSUFFICIENT_PRIVILEGES`). See the [NVIDIA Developer Docs](https://developer.nvidia.com/nvidia-development-tools-solutions-ERR_NVGPUCTRPERM-permission-issue-performance-counters){:.external} to learn more about how you can resolve these issues on Linux.

Docker 環境で CUPTI 特権の問題を解消するには、以下を実行してください。

```shell
docker run option '--privileged=true'
```

<a name="profiler_tools"></a>

## プロファイラツール

Access the Profiler from the **Profile** tab in TensorBoard which appears only after you have captured some model data.

Note: The Profiler requires internet access to load the [Google Chart libraries](https://developers.google.com/chart/interactive/docs/basic_load_libs#basic-library-loading). Some charts and tables may be missing if you run TensorBoard entirely offline on your local machine, behind a corporate firewall, or in a data center.

プロファイラには、次のようなパフォーマンス分析に役立つツールが含まれています。

- 概要ページ
- 入力パイプライン分析ツール
- TensorFlow 統計
- トレースビューア
- GPU カーネル統計
- Memory profile tool
- Pod viewer

<a name="overview_page"></a>

### 概要ページ

The overview page provides a top level view of how your model performed during a profile run. The page shows you an aggregated overview page for your host and all devices, and some recommendations to improve your model training performance. You can also select individual hosts in the Host dropdown.

概要ページには、次のようにデータが表示されます。

![image](./images/tf_profiler/overview_page.png)

- **Performance Summary -** モデルのパフォーマンスの概要が表示されます。パフォーマンスの概要は、次の2つの部分に分かれています。

    1. ステップ時間の内訳 - 平均ステップ時間を、時間を消費した場所に応じて複数のカテゴリに分類しています。

        - Compilation - カーネルのコンパイルに費やされた時間
        - Input - 入力データの読み込みに費やされた時間
        - Output - 出力データの読み込みに費やされた時間
        - Kernel Launch - ホストがカーネルを起動するのに費やされた時間
        - Host Compute Time - ホストの演算時間
        - Device to Device Time - デバイス間の通信時間
        - Device Compute Time - デバイス上の演算時間
        - Python のオーバーヘッドを含むその他すべての時間

    2. Device compute precisions - Reports the percentage of device compute time that uses 16 and 32-bit computations

- **Step-time Graph -** サンプリングされたすべてのステップのデバイスステップ時間（ミリ秒単位）のグラフを表示します。各ステップは、時間を費やしている箇所によって複数のカテゴリに（別々の色で）分かれています。赤い領域は、デバイスがホストからの入力データを待機してアイドル状態であったステップ時間の部分に対応しています。緑の領域は、デバイスが実際に動作していた時間の長さを示しています。

- **Top 10 TensorFlow operations on device -** Displays the on-device ops that ran the longest.

    各行には、演算に費やされた自己時間（すべての演算にかかった時間に占める割合）、累積時間、カテゴリ、名前が表示されます。

- **実行環境 -** 以下を含むモデルの実行環境の高度な概要が表示されます。

    - 使用されたホストの数
    - デバイスのタイプ（GPU/TPU）
    - デバイス コアの数

- **Recommendation for next steps -** Reports when a model is input bound and recommends tools you can use to locate and resolve model performance bottlenecks

<a name="input_pipeline_analyzer"></a>

### 入力パイプライン分析ツール

When a TensorFlow program reads data from a file it begins at the top of the TensorFlow graph in a pipelined manner. The read process is divided into multiple data processing stages connected in series, where the output of one stage is the input to the next one. This system of reading data is called the *input pipeline*.

ファイルからレコードを読み取るための一般的なパイプラインには、次のステージがあります。

1. ファイルの読み取り
2. ファイルの前処理（オプション）
3. ホストからデバイスへのファイル転送

An inefficient input pipeline can severely slow down your application. An application is considered **input bound** when it spends a significant portion of time in input pipeline. Use the insights obtained from the input pipeline analyzer to understand where the input pipeline is inefficient.

The input pipeline analyzer tells you immediately whether your program is input bound and walks you through device- and host-side analysis to debug performance bottlenecks at any stage in the input pipeline.

データ入力パイプラインを最適化するための推奨ベストプラクティスについては、入力パイプラインのパフォーマンスに関するガイダンスをご覧ください。

#### 入力パイプライン ダッシュボード

To open the input pipeline analyzer, select **Profile**, then select **input_pipeline_analyzer** from the **Tools** dropdown.

![image](./images/tf_profiler/input_pipeline_analyzer.png)

ダッシュボードには次の 3 つのセクションがあります。

1. **Summary -** Summarizes the overall input pipeline with information on whether your application is input bound and, if so, by how much
2. **Device-side analysis -** Displays detailed, device-side analysis results, including the device step-time and the range of device time spent waiting for input data across cores at each step
3. **Host-side analysis -** Shows a detailed analysis on the host side, including a breakdown of input processing time on the host

#### 入力パイプラインのサマリー

The Summary reports if your program is input bound by presenting the percentage of device time spent on waiting for input from the host. If you are using a standard input pipeline that has been instrumented, the tool reports where most of the input processing time is spent.

#### デバイス側の分析

The device-side analysis provides insights on time spent on the device versus on the host and how much device time was spent waiting for input data from the host.

1. **ステップ番号に対してプロットされたステップ時間 -** サンプリングされたすべてのステップのデバイス ステップ時間（ミリ秒単位）がグラフに表示されます。各ステップは、時間が費やされた箇所によって複数のカテゴリに（別々の色で）分かれています。赤色の領域は、デバイスがホストからの入力データを待機しているステップ時間に対応します。緑色の領域は、デバイスが実際に稼働していた時間の長さを表します。
2. **ステップ時間の統計 -** デバイス ステップ時間の平均、標準偏差、範囲（[最小、最大]）が報告されます。

#### ホスト側の分析

The host-side analysis reports a breakdown of the input processing time (the time spent on `tf.data` API ops) on the host into several categories:

- **Reading data from files on demand -** Time spent on reading data from files without caching, prefetching, and interleaving
- **Reading data from files in advance -** Time spent reading files, including caching, prefetching, and interleaving
- **データの前処理 -** 画像の圧縮など、前処理の演算に費やされた時間。
- **デバイスに転送するデータのエンキュー -** デバイスへの転送前にデータがインフィード キューに追加される際に費やされた時間。

Expand the **Input Op Statistics** to see the statistics for individual input ops and their categories broken down by execution time.

![image](./images/tf_profiler/tf_stats.png)

ソース データ テーブルには、次の情報を含む各エントリが表示されます。

1. **入力演算 -** 入力演算の TensorFlow 演算名が表示されます。
2. **Count -** Shows the total number of instances of op execution during the profiling period
3. **合計時間（ミリ秒） -** 各インスタンスに費やされた時間の累積合計が表示されます。
4. **Total Time % -** Shows the total time spent on an op as a fraction of the total time spent in input processing
5. **Total Self Time (in ms) -** Shows the cumulative sum of the self time spent on each of those instances. The self time here measures the time spent inside the function body, excluding the time spent in the function it calls.
6. **合計自己時間（%）** - 合計自己時間が、入力処理に費やされた合計時間との割合で表示されます。
7. **カテゴリ** - 入力演算の処理カテゴリが表示されます。

<a name="tf_stats"></a>

### TensorFlow 統計

TensorFlow Stats ツールには、プロファイリングセッション中にホストまたはデバイスで実行されるすべての TensorFlow演算（op）のパフォーマンスが表示されます。

![image](./images/tf_profiler/input_op_stats.png)

このツールでは 2 つのペインでパフォーマンス情報が表示されます。

- 上のペインには、最大 4 つの円グラフが表示されます。

    1. ホスト上の各演算の自己実行時間の分布
    2. ホスト上の各演算タイプの自己実行時間の分布
    3. デバイス上の各演算の自己実行時間の分布
    4. デバイス上の各演算タイプの自己実行時間の分布

- The lower pane shows a table that reports data about TensorFlow ops with one row for each op and one column for each type of data (sort columns by clicking the heading of the column). Click the Export as CSV button on the right side of the upper pane to export the data from this table as a CSV file.

    注意点:

    - 子の演算を持つ演算がある場合:

        - The total "accumulated" time of an op includes the time spent inside the child ops
        - 演算の合計「自己」時間の合計には、子の演算内で費やされた時間が含まれていません。

    - 演算がホスト上で実行される場合:

        - The percentage of the total self-time on device incurred by the op on will be 0
        - The cumulative percentage of the total self-time on device upto and including this op will be 0

    - 演算がデバイス上で実行される場合:

        - The percentage of the total self-time on host incurred by this op will be 0
        - The cumulative percentage of the total self-time on host upto and including this op will be 0

円グラフとテーブルにアイドル時間を含めるか除外するかを選択できます。

<a name="trace_viewer"></a>

### Trace viewer

The trace viewer displays a timeline that shows:

- TensorFlow モデルによって実行された演算の実行期間。
- Which part of the system (host or device) executed an op. Typically, the host executes input operations, preprocesses training data and transfers it to the device, while the device executes the actual model training

The trace viewer allows you to identify performance problems in your model, then take steps to resolve them. For example, at a high level, you can identify whether input or model training is taking the majority of the time. Drilling down, you can identify which ops take the longest to execute. Note that the trace viewer is limited to 1 million events per device.

#### Trace viewer interface

When you open the trace viewer, it appears displaying your most recent run:

![image](./images/tf_profiler/gpu_kernel_stats.png)

この画面には、次の主要な要素が表示されます。

1. **Timeline pane -** Shows ops that the device and the host executed over time
2. **Details pane -** Shows additional information for ops selected in the Timeline pane

The Timeline pane contains the following elements:

1. **上部バー -** さまざまな補助コントロールが含まれています。
2. **時間軸 -** トレースの開始時点を基準とした時間が表示されます。
3. **セクションとトラックラベル -** 各セクションには複数のトラックが含まれています。左側にある三角形をクリックすると、セクションの展開や折りたたみを行うことができます。システムで処理中の要素ごとに 1 つのセクションがあります。
4. **Tool selector -** Contains various tools for interacting with the trace viewer such as Zoom, Pan, Select, and Timing. Use the Timing tool to mark a time interval.
5. **Events -** These show the time during which an op was executed or the duration of meta-events, such as training steps

##### セクションとトラック

The trace viewer contains the following sections:

- **One section for each device node**, labeled with the number of the device chip and the device node within the chip (for example, `/device:GPU:0 (pid 0)`). Each device node section contains the following tracks:
    - **Step -** Shows the duration of the training steps that were running on the device
    - **TensorFlow Ops -** デバイス上で実行された演算が表示されます。
    - **XLA Ops -** [XLA](https://www.tensorflow.org/xla/) が使用されているコンパイラである場合にデバイス上で実行された XLA 演算が表示されます。1 つの TensorFlow 演算が 1 つ以上の XLA 演算に変換されます。XLA コンパイラにより、XLA 演算がデバイス上で実行されるコードに変換されます。
- **One section for threads running on the host machine's CPU,** labeled **"Host Threads"**. The section contains one track for each CPU thread. Note that you can ignore the information displayed alongside the section labels.

##### イベント

Events within the timeline are displayed in different colors; the colors themselves have no specific meaning.

The trace viewer can also display traces of Python function calls in your TensorFlow program. If you use the `tf.profiler.experimental.start()` API, you can enable Python tracing by using the `ProfilerOptions` namedtuple when starting profiling. Alternatively, if you use the sampling mode for profiling, you can select the level of tracing by using the dropdown options in the **Capture Profile** dialog.

![image](./images/tf_profiler/python_tracer.png)

<a name="gpu_kernel_stats"></a>

### GPU カーネル統計

This tool shows performance statistics and the originating op for every GPU accelerated kernel.

![image](./images/tf_profiler/trace_viewer.png)

このツールでは 2 つのペインで情報が表示されます。

- 上部のペインには、合計経過時間が最も長い CUDA カーネルを示す円グラフが表示されます。

- 下のペインには、一意のカーネルと演算のペアごとに次のデータを含むテーブルが表示されます。

    - A rank in descending order of total elapsed GPU duration grouped by kernel-op pair
    - 起動されたカーネルの名前
    - カーネルが使用している GPU レジスタの数
    - 使用されている共有（静的+動的共有）メモリの合計サイズ（バイト単位）
    - `blockDim.x, blockDim.y, blockDim.z` で表現されたブロックの次元
    - `gridDim.x, gridDim.y, gridDim.z` で表現されたグリッドの次元
    - 演算が TensorCore を使用可能かどうか
    - カーネルに TensorCore 命令が含まれているかどうか
    - このカーネルを起動した演算の名前
    - このカーネルと演算のペアが発生した数
    - 合計経過 GPU 時間（マイクロ秒）
    - 平均経過 GPU 時間（マイクロ秒）
    - 最短経過 GPU 時間（マイクロ秒）
    - 最長経過 GPU 時間（マイクロ秒）

<a name="memory_profile_tool"></a>

### Memory profile tool {: id = 'memory_profile_tool'}

The Memory Profile tool monitors the memory usage of your device during the profiling interval. You can use this tool to:

- Debug out of memory (OOM) issues by pinpointing peak memory usage and the corresponding memory allocation to TensorFlow ops. You can also debug OOM issues that may arise when you run [multi-tenancy](https://arxiv.org/pdf/1901.06887.pdf) inference
- Debug memory fragmentation issues

The memory profile tool displays data in three sections:

1. Memory Profile Summary
2. Memory Timeline Graph
3. Memory Breakdown Table

#### Memory profile summary

This section displays a high-level summary of the memory profile of your TensorFlow program as shown below:

&lt;img src="./images/tf_profiler/memory_profile_summary.png" width="400", height="450"&gt;

The memory profile summary has six fields:

1. Memory ID - Dropdown which lists all available device memory systems. Select the memory system you want to view from the dropdown
2. #Allocation - The number of memory allocations made during the profiling interval
3. #Deallocation - The number of memory deallocations in the profiling interval
4. Memory Capacity - The total capacity (in GiBs) of the memory system that you select
5. Peak Heap Usage - The peak memory usage (in GiBs) since the model started running
6. Peak Memory Usage - The peak memory usage (in GiBs) in the profiling interval. This field contains the following sub-fields:
    1. Timestamp - The timestamp of when the peak memory usage occurred on the Timeline Graph
    2. Stack Reservation - Amount of memory reserved on the stack (in GiBs)
    3. Heap Allocation - Amount of memory allocated on the heap (in GiBs)
    4. Free Memory - Amount of free memory (in GiBs). The Memory Capacity is the sum total of the Stack Reservation, Heap Allocation, and Free Memory
    5. Fragmentation - The percentage of fragmentation (lower is better). It is calculated as a percentage of (1 - Size of the largest chunk of free memory / Total free memory)

#### Memory timeline graph

This section displays a plot of the memory usage (in GiBs) and the percentage of fragmentation versus time (in ms).

![image](./images/tf_profiler/memory_timeline_graph.png)

The X-axis represents the timeline (in ms) of the profiling interval. The Y-axis on the left represents the memory usage (in GiBs) and the Y-axis on the right represents the percentage of fragmentation. At each point in time on the X-axis, the total memory is broken down into three categories: stack (in red), heap (in orange), and free (in green). Hover over a specific timestamp to view the details about the memory allocation/deallocation events at that point like below:

![image](./images/tf_profiler/memory_timeline_graph_popup.png)

The pop-up window displays the following information:

- timestamp(ms) - The location of the selected event on the timeline
- event - The type of event (allocation or deallocation)
- requested_size(GiBs) - The amount of memory requested. This will be a negative number for deallocation events
- allocation_size(GiBs) - The actual amount of memory allocated. This will be a negative number for deallocation events
- tf_op - The TensorFlow Op that requests the allocation/deallocation
- step_id - The training step in which this event occurred
- region_type - The data entity type that this allocated memory is for. Possible values are `temp` for temporaries, `output` for activations and gradients, and `persist`/`dynamic` for weights and constants
- data_type - The tensor element type (e.g., uint8 for 8-bit unsigned integer)
- tensor_shape - The shape of the tensor being allocated/deallocated
- memory_in_use(GiBs) - The total memory that is in use at this point of time

#### Memory breakdown table

This table shows the active memory allocations at the point of peak memory usage in the profiling interval.

![image](./images/tf_profiler/memory_breakdown_table.png)

There is one row for each TensorFlow Op and each row has the following columns:

- Op Name - The name of the TensorFlow op
- Allocation Size (GiBs) - The total amount of memory allocated to this op
- Requested Size (GiBs) - The total amount of memory requested for this op
- Occurrences - The number of allocations for this op
- Region type - The data entity type that this allocated memory is for. Possible values are `temp` for temporaries, `output` for activations and gradients, and `persist`/`dynamic` for weights and constants
- Data type - The tensor element type
- Shape - The shape of the allocated tensors

Note: You can sort any column in the table and also filter rows by op name.

<a name="pod_viewer"></a>

### Pod viewer

The Pod Viewer tool shows the breakdown of a training step across all workers.

![image](./images/tf_profiler/pod_viewer.png)

- The upper pane has slider for selecting the step number.
- The lower pane displays a stacked column chart. This is a high level view of broken down step-time categories placed atop one another. Each stacked column represents a unique worker.
- When you hover over a stacked column, the card on the left-hand side shows more details about the step breakdown.

<a name="tf_data_bottleneck_analysis"></a>

### tf.data bottleneck analysis

Warning: This tool is experimental. Please report [here](https://github.com/tensorflow/profiler/issues) if the analysis result seems off.

tf.data bottleneck analysis automatically detects bottlenecks in tf.data input pipelines in your program and provides recommendations on how to fix them. It works with any program using tf.data regardless of the platform (CPU/GPU/TPU) or the framework (TensorFlow/JAX). Its analysis and recommendations are based on this [guide](https://www.tensorflow.org/guide/data_performance_analysis).

It detects a bottleneck by following these steps:

1. Find the most input bound host.
2. Find the slowest execution of tf.data input pipeline.
3. Reconstruct the input pipeline graph from the profiler trace.
4. Find the critical path in the input pipeline graph.
5. Identify the slowest transformation on the critical path as a bottleneck.

The UI is divided into three sections: Performance Analysis Summary, Summary of All Input Pipelines and Input Pipeline Graph.

#### Performance analysis summary

![image](./images/tf_profiler/capture_profile.png)

This section provides the summary of the analysis. It tells whether a slow tf.data input pipeline is detected in the profile. If so, it shows the most input bound host and its slowest input pipeline with the max latency. And most importantly, it tells which part of the input pipeline is the bottleneck and how to fix it. The bottleneck information is provided with the iterator type and its long name.

##### How to read tf.data iterator's long name

A long name is formatted as `Iterator::<Dataset_1>::...::<Dataset_n>`. In the long name, `<Dataset_n>` matches the iterator type and the other datasets in the long name represent downstream transformations.

For example, consider the following input pipeline dataset:

```python
dataset = tf.data.Dataset.range(10).map(lambda x: x).repeat(2).batch(5)
```

The long names for the iterators from the above dataset will be:

Iterator Type | Long Name
:-- | :--
Range | Iterator::Batch::Repeat::Map::Range
Map | Iterator::Batch::Repeat::Map
Repeat | Iterator::Batch::Repeat
Batch | Iterator::Batch

#### Summary of All Input Pipelines

![image](./images/tf_profiler/tf_data_all_hosts.png)

This section provides the summary of all input pipelines across all hosts. Typically there is one input pipeline. When using the distribution strategy, there are one host input pipeline running the program's tf.data code and multiple device input pipelines retrieving data from the host input pipeline and transferring it to the devices.

For each input pipeline, it shows the statistics of its execution time. A call is counted as slow if it takes longer than 50 μs.

#### Input Pipeline Graph

![image](./images/tf_profiler/tf_data_graph_selector.png)

This section shows the input pipeline graph with the execution time information. You can use "Host" and "Input Pipeline" to choose which host and input pipeline to see. Executions of the input pipeline are sorted by the execution time in descending order which you can use "Rank" to choose.

![image](./images/tf_profiler/tf_data_graph.png)

The nodes on the critical path have bold outlines. The bottleneck node, which is the node with the longest self time on the critical path, has a red outline. The other non-critical nodes have gray dashed outlines.

In each node, "Start Time" indicates the start time of the execution. The same node may be executed multiple times, for example, if there is Batch in the input pipeline. If it is executed multiple times, it is the start time of the first execution.

"Total Duration" is the wall time of the execution. If it is executed multiple times, it is the sum of the wall times of all executions.

"Self Time" is "Total Time" without the overlapped time with its immediate child nodes.

"# Calls" is the number of times the input pipeline is executed.

<a name="collect_performance_data"></a>

## パフォーマンスデータの収集

The TensorFlow Profiler collects host activities and GPU traces of your TensorFlow model. You can configure the Profiler to collect performance data through either the programmatic mode or the sampling mode.

### プロファイリング API

次の API を使用してプロファイリングを実行できます。

- Programmatic mode using the TensorBoard Keras Callback (`tf.keras.callbacks.TensorBoard`)

    ```python
    # Profile from batches 10 to 15
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                 profile_batch='10, 15')

    # Train the model and use the TensorBoard Keras callback to collect
    # performance profiling data
    model.fit(train_data,
              steps_per_epoch=20,
              epochs=5,
              callbacks=[tb_callback])
    ```

- `tf.profiler` 関数 API を使用したプログラムモード

    ```python
    tf.profiler.experimental.start('logdir')
    # Train the model here
    tf.profiler.experimental.stop()
    ```

- コンテキストマネージャを使用したプログラムモード

    ```python
    with tf.profiler.experimental.Profile('logdir'):
        # Train the model here
        pass
    ```

Note: Running the Profiler for too long can cause it to run out of memory. It is recommended to profile no more than 10 steps at a time. Avoid profiling the first few batches to avoid inaccuracies due to initialization overhead.

<a name="sampling_mode"></a>

- Sampling mode - Perform on-demand profiling by using `tf.profiler.experimental.server.start()` to start a gRPC server with your TensorFlow model run. After starting the gRPC server and running your model, you can capture a profile through the **Capture Profile** button in the TensorBoard profile plugin. Use the script in the Install profiler section above to launch a TensorBoard instance if it is not already running.

    以下に例を示します。

    ```python
    # Start a profiler server before your model runs.
    tf.profiler.experimental.server.start(6009)
    # (Model code goes here).
    #  Send a request to the profiler server to collect a trace of your model.
    tf.profiler.experimental.client.trace('grpc://localhost:6009',
                                          'gs://your_tb_logdir', 2000)
    ```

    An example for profiling multiple workers:

    ```python
    # E.g. your worker IP addresses are 10.0.0.2, 10.0.0.3, 10.0.0.4, and you
    # would like to profile for a duration of 2 seconds.
    tf.profiler.experimental.client.trace(
        'grpc://10.0.0.2:8466,grpc://10.0.0.3:8466,grpc://10.0.0.4:8466',
        'gs://your_tb_logdir',
        2000)
    ```

<a name="capture_dialog"></a>

&lt;img src="./images/tf_profiler/capture_profile.png" width="400", height="450"&gt;

Use the **Capture Profile** dialog to specify:

- A comma delimited list of profile service URLs or TPU name.
- A profiling duration.
- The level of device, host, and Python function call tracing.
- How many times you want the Profiler to retry capturing profiles if unsuccessful at first.

### カスタムトレーニングループのプロファイリング

TensorFlow コードでカスタムトレーニングループのプロファイルを作成するには、`tf.profiler.experimental.Trace` API を使用してトレーニングループをインストルメント化し、プロファイラのステップの範囲を示してください。`name` 引数はステップ名の接頭辞として使用され、`step_num` キーワード引数はステップ名に追加され、`_r` キーワード引数はこのトレース イベントをステップ イベントとしてプロファイラに処理させます。

以下に例を示します。

```python
for step in range(NUM_STEPS):
    with tf.profiler.experimental.Trace('train', step_num=step, _r=1):
        train_data = next(dataset)
        train_step(train_data)
```

This will enable the Profiler's step-based performance analysis and cause the step events to show up in the trace viewer.

Ensure that you include the dataset iterator within the `tf.profiler.experimental.Trace` context for accurate analysis of the input pipeline.

以下のコードスニペットはアンチパターンです。

警告: このコードにより、不正確な入力パイプライン分析が実行されます。

```python
for step, train_data in enumerate(dataset):
    with tf.profiler.experimental.Trace('train', step_num=step, _r=1):
        train_step(train_data)
```

### プロファイリングの使用事例

プロファイラは、4 種類の軸に沿って多数の使用事例をカバーしています。これらの組み合わせの中には、現在サポートされているものもあれば、今後の追加が予定されているものもあります。次に一部の使用事例を示します。

- ローカルプロファイリングとリモートプロファイリング: これら 2 つは、プロファイリング環境を設定するための一般的な方法です。ローカルプロファイリングでは、モデルが実行されているのと同じマシン（GPU を備えたローカルのワークステーションなど）でプロファイリング API が呼び出されます。リモートプロファイリングでは、モデルが実行されているマシンとは異なるマシン（Cloud TPU 上など）でプロファイリング API が呼び出されます。
- 複数のワーカーのプロファイリング: TensorFlow の分散トレーニング機能を使用すると、複数のマシンをプロファイリングできます。
- ハードウェアプラットフォーム: CPU、GPU、TPU のプロファイリング。

The table below is a quick overview of which of the above use cases are supported by the various profiling APIs in TensorFlow:

<a name="profiling_api_table"></a>

| Profiling API                | Local     | Remote    | Multiple  | Hardware  | :                              :           :           : workers   : Platforms : | :--------------------------- | :-------- | :-------- | :-------- | :-------- | | **TensorBoard Keras          | Supported | Not       | Not       | CPU, GPU  | : Callback**                   :           : Supported : Supported :           : | **`tf.profiler.experimental` | Supported | Not       | Not       | CPU, GPU  | : start/stop [API](https://www.tensorflow.org/api_docs/python/tf/profiler/experimental#functions_2)**    :           : Supported : Supported :           : | **`tf.profiler.experimental` | Supported | Supported | Supported | CPU, GPU, | : client.trace [API](https://www.tensorflow.org/api_docs/python/tf/profiler/experimental/client/trace)**  :           :           :           : TPU       : | **Context manager API**      | Supported | Not       | Not       | CPU, GPU  | :                              :           : supported : Supported :           :

<a name="performance_best_practices"></a>

## 最適なモデルパフォーマンスのベストプラクティス

Use the following recommendations as applicable for your TensorFlow models to achieve optimal performance.

In general, perform all transformations on the device and ensure that you use the latest compatible version of libraries like cuDNN and Intel MKL for your platform.

### 入力データパイプラインの最適化

An efficient data input pipeline can drastically improve the speed of your model execution by reducing device idle time. Consider incorporating the following best practices as detailed [here](https://www.tensorflow.org/guide/data_performance) to make your data input pipeline more efficient:

- データをプリフェッチする
- データ抽出を並列化する
- データ変換を並列化する
- データをメモリにキャッシュする
- ユーザー定義関数をベクトル化する
- 変換を適用する際のメモリ使用量を削減する

Additionally, try running your model with synthetic data to check if the input pipeline is a performance bottleneck.

### デバイスのパフォーマンス改善

- Increase training mini-batch size (number of training samples used per device in one iteration of the training loop)
- TensorFlow 統計を使用し、デバイス上の演算がどの程度効率的に実行されているかを確認する。
- Use `tf.function` to perform computations and optionally, enable the `experimental_compile` flag
- Minimize host Python operations between steps and reduce callbacks. Calculate metrics every few steps instead of at every step
- デバイスの演算ユニットをビジー状態に保つ。
- 複数のデバイスに対して並列にデータを送信する。
- Optimize data layout to prefer channels first (e.g. NCHW over NHWC). Certain GPUs like the NVIDIA® V100 perform better with a NHWC data layout.
- Consider using 16-bit numerical representations such as `fp16`, the half-precision floating point format specified by IEEE or the Brain floating-point [bfloat16](https://cloud.google.com/tpu/docs/bfloat16) format
- Consider using the [Keras mixed precision API](https://www.tensorflow.org/guide/keras/mixed_precision)
- When training on GPUs, make use of the TensorCore. GPU kernels use the TensorCore when the precision is fp16 and input/output dimensions are divisible by 8 or 16 (for int8)

## 追加リソース

- See the end-to-end [TensorBoard profiler tutorial](https://www.tensorflow.org/tensorboard/tensorboard_profiling_keras) to implement the advice in this guide.
- Watch the [Performance profiling in TF 2](https://www.youtube.com/watch?v=pXHAQIhhMhI) talk from the TensorFlow Dev Summit 2020.

## Known limitations

### Profiling multiple GPUs on TensorFlow 2.2 and TensorFlow 2.3

TensorFlow 2.2 and 2.3 support multiple GPU profiling for single host systems only; multiple GPU profiling for multi-host systems is not supported. To profile multi-worker GPU configurations, each worker has to be profiled independently. On TensorFlow 2.4, multiple workers can be profiled using the [`tf.profiler.experimental.trace`](https://www.tensorflow.org/api_docs/python/tf/profiler/experimental/client/trace) API.

CUDA® Toolkit 10.2 or later is required to profile multiple GPUs. As TensorFlow 2.2 and 2.3 support CUDA® Toolkit versions only up to 10.1 , create symbolic links to `libcudart.so.10.1` and `libcupti.so.10.1`.

```shell
sudo ln -s /usr/local/cuda/lib64/libcudart.so.10.2 /usr/local/cuda/lib64/libcudart.so.10.1
sudo ln -s /usr/local/cuda/extras/CUPTI/lib64/libcupti.so.10.2 /usr/local/cuda/extras/CUPTI/lib64/libcupti.so.10.1
```
