# プロファイラを使用した TensorFlow のパフォーマンス最適化

[TOC]

This guide demonstrates how to use the tools available with the TensorFlow Profiler to track the performance of your TensorFlow models. You will learn how to understand how your model performs on the host (CPU), the device (GPU), or on a combination of both the host and device(s).

Profiling helps understand the hardware resource consumption (time and memory) of the various TensorFlow operations (ops) in your model and resolve performance bottlenecks and, ultimately, make the model execute faster.

このガイドでは、プロファイラのインストール方法、利用可能なさまざまなツール、プロファイラのさまざまなパフォーマンスデータ収集モード、およびモデルのパフォーマンスを最適化するために推奨されるベストプラクティスについて説明します。

Cloud TPU 上でモデルのパフォーマンスをプロファイリングする場合は、 [Cloud TPU のガイド](https://cloud.google.com/tpu/docs/cloud-tpu-tools#capture_profile)をご覧ください。

## プロファイラのインストールと GPU の要件

Install the Profiler plugin for TensorBoard with pip. Note that the Profiler requires the latest versions of TensorFlow and TensorBoard (&gt;=2.2).

```shell
pip install -U tensorboard_plugin_profile
```

GPU 上でプロファイリングを実行するには、次の手順を行う必要があります。

1. [TensorFlow GPU サポートソフトウェアの要件](https://www.tensorflow.org/install/gpu#linux_setup)に記載されている NVIDIA® GPU ドライバーと CUDA® Toolkit の要件を満たします。

2. Make sure the [NVIDIA® CUDA® Profiling Tools Interface](https://developer.nvidia.com/cupti) (CUPTI) exists on the path:

    ```shell
    /sbin/ldconfig -N -v $(sed 's/:/ /g' <<< $LD_LIBRARY_PATH) | \
    grep libcupti
    ```

パスに CUPTI が存在しない場合は、次のコマンドを実行してインストールディレクトリを `$LD_LIBRARY_PATH` 環境変数の前に追加します。

```shell
export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
```

Then, run the `ldconfig` command above again to verify that the CUPTI library is found.

### 特権の問題を解消する

When you run profiling with CUDA® Toolkit in a Docker environment or on Linux, you may encounter issues related to insufficient CUPTI privileges (`CUPTI_ERROR_INSUFFICIENT_PRIVILEGES`). Go to the [NVIDIA Developer Docs](https://developer.nvidia.com/nvidia-development-tools-solutions-ERR_NVGPUCTRPERM-permission-issue-performance-counters){:.external} to learn more about how you can resolve these issues on Linux.

Docker 環境で CUPTI 特権の問題を解消するには、以下を実行してください。

```shell
docker run option '--privileged=true'
```

<a name="profiler_tools"></a>

## プロファイラツール

Access the Profiler from the **Profile** tab in TensorBoard, which appears only after you have captured some model data.

注意: プロファイラは [Google Chart ライブラリ](https://developers.google.com/chart/interactive/docs/basic_load_libs#basic-library-loading)を読み込むため、インターネットアクセスを要求します。TensorBoard をローカルマシン上、企業内ファイアウォールの背後、またはデータセンターで完全にオフラインで実行する場合、一部のチャートやテーブルが表示されない場合があります。

プロファイラには、次のようなパフォーマンス分析に役立つツールが含まれています。

- Overview Page
- Input Pipeline Analyzer
- TensorFlow Stats
- Trace Viewer
- GPU Kernel Stats
- Memory Profile Tool
- Pod Viewer

<a name="overview_page"></a>

### 概要ページ

概要ページでは、プロファイリングを実行中にモデルがどのように動作したかが一番上に表示されます。このページには、ホストとすべてのデバイスの概要を集約したページと、モデルのトレーニングパフォーマンスを改善するためのいくつかの推奨事項が表示されます。Host ドロップダウンで個々のホストを選択することもできます。

概要ページには、次のようにデータが表示されます。

![image](./images/tf_profiler/overview_page.png)

- **Performance Summary**: Displays a high-level summary of your model performance. The performance summary has two parts:

    1. Step-time breakdown: Breaks down the average step time into multiple categories of where time is spent:

        - Compilation: Time spent compiling kernels.
        - Input: Time spent reading input data.
        - Output: Time spent reading output data.
        - Kernel launch: Time spent by the host to launch kernels
        - Host compute time..
        - Device-to-device communication time.
        - On-device compute time.
        - All others, including Python overhead.

    2. Device compute precisions - Reports the percentage of device compute time that uses 16 and 32-bit computations.

- **Step-time Graph**: Displays a graph of device step time (in milliseconds) over all the steps sampled. Each step is broken into the multiple categories (with different colors) of where time is spent. The red area corresponds to the portion of the step time the devices were sitting idle waiting for input data from the host. The green area shows how much of time the device was actually working.

- **Top 10 TensorFlow operations on device (e.g. GPU)**: Displays the on-device ops that ran the longest.

    各行には、演算に費やされた自己時間（すべての演算にかかった時間に占める割合）、累積時間、カテゴリ、名前が表示されます。

- **Run Environment**: Displays a high-level summary of the model run environment including:

    - Number of hosts used.
    - Device type (GPU/TPU).
    - Number of device cores.

- **Recommendation for Next Step**: Reports when a model is input bound and recommends tools you can use to locate and resolve model performance bottlenecks.

<a name="input_pipeline_analyzer"></a>

### 入力パイプライン分析ツール

TensorFlow プログラムがファイルからデータを読み込むと、TensorFlow グラフにパイプライン方式でデータが表示されます。読み取りプロセスは連続した複数のデータ処理ステージに分割され、1 つのステージの出力が次のステージの入力となります。この読み込み方式を*入力パイプライン*といいます。

ファイルからレコードを読み取るための一般的なパイプラインには、次のステージがあります。

1. File reading.
2. File preprocessing (optional).
3. File transfer from the host to the device.

An inefficient input pipeline can severely slow down your application. An application is considered **input bound** when it spends a significant portion of time in the input pipeline. Use the insights obtained from the input pipeline analyzer to understand where the input pipeline is inefficient.

入力パイプライン分析ツールは、プログラムで入力処理の負荷が高くなっているかどうかを即座に通知し、入力パイプラインの任意のステージでパフォーマンスボトルネックをデバッグするために、デバイス側とホスト側の分析を案内します。

Check the guidance on input pipeline performance for recommended best practices to optimize your data input pipelines.

#### 入力パイプライン ダッシュボード

入力パイプライン分析ツールを開くには、**Profile** を選択し、**Tools** プルダウンから **input_pipeline_analyzer** を選択します。

![image](./images/tf_profiler/tf_data_graph.png)

ダッシュボードには次の 3 つのセクションがあります。

1. **Summary**: Summarizes the overall input pipeline with information on whether your application is input bound and, if so, by how much.
2. **Device-side analysis**: Displays detailed, device-side analysis results, including the device step-time and the range of device time spent waiting for input data across cores at each step.
3. **Host-side analysis**: Shows a detailed analysis on the host side, including a breakdown of input processing time on the host.

#### 入力パイプラインのサマリー

サマリーは、ホストからの入力待ちに費やされたデバイス時間の割合が表示されます。これにより、プログラムで入力処理の負荷が高くなっているかどうかを確認できます。インストゥルメント化された標準の入力パイプラインを使用している場合は、ツールによって多くの入力処理時間が費やされている部分が報告されます。

#### デバイス側の分析

デバイス側の分析では、デバイスとホストの間で費やされた時間と、ホストからの入力データの待機に費やされたデバイス時間が表示されます。

1. **Step time plotted against step number**: Displays a graph of device step time (in milliseconds) over all the steps sampled. Each step is broken into the multiple categories (with different colors) of where time is spent. The red area corresponds to the portion of the step time the devices were sitting idle waiting for input data from the host. The green area shows how much of the time the device was actually working.
2. **Step time statistics**: Reports the average, standard deviation, and range ([minimum, maximum]) of the device step time.

#### ホスト側の分析

ホスト側の分析には、ホスト上での入力処理時間（`tf.data` API 演算に費やされた時間）の内訳が次のいくつかのカテゴリに分類されて表示されます。

- **Reading data from files on demand**: Time spent on reading data from files without caching, prefetching, and interleaving.
- **Reading data from files in advance**: Time spent reading files, including caching, prefetching, and interleaving.
- **Data preprocessing**: Time spent on preprocessing ops, such as image decompression.
- **Enqueuing data to be transferred to device**: Time spent putting data into an infeed queue before transferring the data to the device.

Expand **Input Op Statistics** to inspect the statistics for individual input ops and their categories broken down by execution time.

![image](./images/tf_profiler/input_op_stats.png)

A source data table will appear with each entry containing the following information:

1. **Input Op**: Shows the TensorFlow op name of the input op.
2. **Count**: Shows the total number of instances of op execution during the profiling period.
3. **Total Time (in ms)**: Shows the cumulative sum of time spent on each of those instances.
4. **Total Time %**: Shows the total time spent on an op as a fraction of the total time spent in input processing.
5. **Total Self Time (in ms)**: Shows the cumulative sum of the self time spent on each of those instances. The self time here measures the time spent inside the function body, excluding the time spent in the function it calls.
6. **Total Self Time %**. Shows the total self time as a fraction of the total time spent on input processing.
7. **Category**. Shows the processing category of the input op.

<a name="tf_stats"></a>

### TensorFlow 統計

TensorFlow Stats ツールには、プロファイリングセッション中にホストまたはデバイスで実行されるすべての TensorFlow演算（op）のパフォーマンスが表示されます。

![image](./images/tf_profiler/tf_data_graph_selector.png)

このツールでは 2 つのペインでパフォーマンス情報が表示されます。

- 上のペインには、最大 4 つの円グラフが表示されます。

    1. The distribution of self-execution time of each op on the host.
    2. The distribution of self-execution time of each op type on the host.
    3. The distribution of self-execution time of each op on the device.
    4. The distribution of self-execution time of each op type on the device.

- 下のペインには、TensorFlow 演算に関するデータを報告するテーブルが表示されており、各演算に 1 行、各タイプのデータに 1 列（列の見出しをクリックして列をソート）が割り当てられています。上のペインの右側にある Export as CSV ボタンをクリックすると、このテーブルのデータが CSV ファイルとしてエクスポートされます。

    注意点:

    - 子の演算を持つ演算がある場合:

        - The total "accumulated" time of an op includes the time spent inside the child ops.
        - The total "self" time of an op does not include the time spent inside the child ops.

    - 演算がホスト上で実行される場合:

        - The percentage of the total self-time on device incurred by the op on will be 0.
        - The cumulative percentage of the total self-time on device up to and including this op will be 0.

    - 演算がデバイス上で実行される場合:

        - The percentage of the total self-time on host incurred by this op will be 0.
        - The cumulative percentage of the total self-time on host up to and including this op will be 0.

円グラフとテーブルにアイドル時間を含めるか除外するかを選択できます。

<a name="trace_viewer"></a>

### トレースビューア

トレースビューアには次のタイムラインが表示されます。

- TensorFlow モデルによって実行された演算の実行期間。
- 演算を実行したシステムの部分（ホストまたはデバイス）。通常、ホストが入力演算を実行し、トレーニングデータを前処理してデバイスに転送し、デバイスは実際のモデルトレーニングを行います。

トレースビューアを使用して、モデル内のパフォーマンスの問題を特定し、この問題を解決する対策を講じることができます。たとえば、入力とモデルトレーニングのどちらに大部分の時間を費やしているかどうかを大まかに識別できます。さらに詳しく見ると、どの演算の実行に最も時間がかかっているかも識別できます。トレースビューアで表示できるのはデバイスごとに 100 万イベントまでです。

#### トレースビューアのインターフェース

トレースビューアを開くと、最新の実行結果が表示されます。

![image](./images/tf_profiler/gpu_kernel_stats.png)

この画面には、次の主要な要素が表示されます。

1. **Timeline pane**: Shows ops that the device and the host executed over time.
2. **Details pane**: Shows additional information for ops selected in the Timeline pane.

Timeline ペインには、次の要素が含まれます。

1. **Top bar**: Contains various auxiliary controls.
2. **Time axis**: Shows time relative to the beginning of the trace.
3. **Section and track labels**: Each section contains multiple tracks and has a triangle on the left that you can click to expand and collapse the section. There is one section for every processing element in the system.
4. **Tool selector**: Contains various tools for interacting with the trace viewer such as Zoom, Pan, Select, and Timing. Use the Timing tool to mark a time interval.
5. **Events**: These show the time during which an op was executed or the duration of meta-events, such as training steps.

##### セクションとトラック

トレースビューアには、次のセクションがあります。

- **One section for each device node**, labeled with the number of the device chip and the device node within the chip (for example, `/device:GPU:0 (pid 0)`). Each device node section contains the following tracks:
    - **Step**: Shows the duration of the training steps that were running on the device
    - **TensorFlow Ops**: Shows the ops executed on the device
    - **XLA Ops**: Shows [XLA](https://www.tensorflow.org/xla/) operations (ops) that ran on the device if XLA is the compiler used (each TensorFlow op is translated into one or several XLA ops. The XLA compiler translates the XLA ops into code that runs on the device).
- **ホストマシンの CPU 上で実行されるスレッドのセクション** - **「Host Threads」**というラベルが付いています。このセクションには、CPU スレッドごとに 1 つのトラックが含まれます。セクションラベルと一緒に表示される情報は無視してもかまいません。

##### イベント

タイムライン内のイベントは異なる色で表示されます。色自体には特別な意味はありません。

The trace viewer can also display traces of Python function calls in your TensorFlow program. If you use the `tf.profiler.experimental.start` API, you can enable Python tracing by using the `ProfilerOptions` namedtuple when starting profiling. Alternatively, if you use the sampling mode for profiling, you can select the level of tracing by using the dropdown options in the **Capture Profile** dialog.

![image](./images/tf_profiler/python_tracer.png)

<a name="gpu_kernel_stats"></a>

### GPU カーネル統計

このツールには、すべての GPU アクセラレータカーネルのパフォーマンス統計と元の演算を表示されます。

![image](./images/tf_profiler/tf_data_all_hosts.png)

このツールでは 2 つのペインで情報が表示されます。

- The upper pane displays a pie chart which shows the CUDA kernels that have the highest total time elapsed.

- The lower pane displays a table with the following data for each unique kernel-op pair:

    - A rank in descending order of total elapsed GPU duration grouped by kernel-op pair.
    - The name of the launched kernel.
    - The number of GPU registers used by the kernel.
    - The total size of shared (static + dynamic shared) memory used in bytes.
    - The block dimension expressed as `blockDim.x, blockDim.y, blockDim.z`.
    - The grid dimensions expressed as `gridDim.x, gridDim.y, gridDim.z`.
    - Whether the op is eligible to use [Tensor Cores](https://www.nvidia.com/en-gb/data-center/tensor-cores/).
    - Whether the kernel contains Tensor Core instructions.
    - The name of the op that launched this kernel.
    - The number of occurrences of this kernel-op pair.
    - The total elapsed GPU time in microseconds.
    - The average elapsed GPU time in microseconds.
    - The minimum elapsed GPU time in microseconds.
    - The maximum elapsed GPU time in microseconds.

<a name="memory_profile_tool"></a>

### メモリのプロファイリングツール {: id = 'memory_profile_tool'}

メモリプロファイルツールは、プロファイリング間のデバイスのメモリ使用状況を監視します。このツールを使用して、次のことを実行できます。

- Debug out of memory (OOM) issues by pinpointing peak memory usage and the corresponding memory allocation to TensorFlow ops. You can also debug OOM issues that may arise when you run [multi-tenancy](https://arxiv.org/pdf/1901.06887.pdf) inference.
- Debug memory fragmentation issues.

メモリプロファイルツールには、次の 3 つのセクションにデータが表示されます。

1. **メモリプロファイルのサマリー**
2. **メモリのタイムライングラフ**
3. **メモリの詳細テーブル**

#### メモリプロファイルのサマリー

このセクションには、以下に示されるように、TensorFlow プログラムの要約が表示されます。

&lt;img src="./images/tf_profiler/memory_profile_summary.png" width="400", height="450"&gt;

メモリプロファイルのサマリーには、次の 6 つのフィールドがあります。

1. **Memory ID**: Dropdown which lists all available device memory systems. Select the memory system you want to view from the dropdown.
2. **#Allocation**: The number of memory allocations made during the profiling interval.
3. **#Deallocation**: The number of memory deallocations in the profiling interval
4. **Memory Capacity**: The total capacity (in GiBs) of the memory system that you select.
5. **Peak Heap Usage**: The peak memory usage (in GiBs) since the model started running.
6. **Peak Memory Usage**: The peak memory usage (in GiBs) in the profiling interval. This field contains the following sub-fields:
    1. **Timestamp**: The timestamp of when the peak memory usage occurred on the Timeline Graph.
    2. **Stack Reservation**: Amount of memory reserved on the stack (in GiBs).
    3. **Heap Allocation**: Amount of memory allocated on the heap (in GiBs).
    4. **Free Memory**: Amount of free memory (in GiBs). The Memory Capacity is the sum total of the Stack Reservation, Heap Allocation, and Free Memory.
    5. **Fragmentation**: The percentage of fragmentation (lower is better). It is calculated as a percentage of `(1 - Size of the largest chunk of free memory / Total free memory)`.

#### メモリのタイムライングラフ

このセクションには、メモリ使用率（GiB）と断片率を時間（ms）比較した図が表示されます。

![image](https://github.com/tensorflow/docs-l10n/blob/master/site/ja/guide/images/tf_profiler/memory_timeline_graph.png?raw=true)

X 軸は、プロファイリングインターバルのタイムライン（ms）を表します。左の Y 軸はメモリ使用率（GiB）を、右の Y 軸は断片率を表します。合計メモリは、X 軸のある時点で、スタック（赤）、ヒープ（オレンジ）、空き（緑）の 3 つに分けて示されています。特定のタイムスタンプにマウスポインタを合わせると、以下のように、その時点でのメモリの割り当てと割り当て解除の詳細を確認できます。

![image](./images/tf_profiler/memory_timeline_graph_popup.png)

ポップアップウィンドウには、次の情報が表示されます。

- **timestamp(ms)**: The location of the selected event on the timeline.
- **event**: The type of event (allocation or deallocation).
- **requested_size(GiBs)**: The amount of memory requested. This will be a negative number for deallocation events.
- **allocation_size(GiBs)**: The actual amount of memory allocated. This will be a negative number for deallocation events.
- **tf_op**: The TensorFlow op that requests the allocation/deallocation.
- **step_id**: The training step in which this event occurred.
- **region_type**: The data entity type that this allocated memory is for. Possible values are `temp` for temporaries, `output` for activations and gradients, and `persist`/`dynamic` for weights and constants.
- **data_type**: The tensor element type (e.g., uint8 for 8-bit unsigned integer).
- **tensor_shape**: The shape of the tensor being allocated/deallocated.
- **memory_in_use(GiBs)**: The total memory that is in use at this point of time.

#### メモリの詳細テーブル

このテーブルには、プロファイリングインターバルのピークメモリ使用率の時点でアクティブなメモリの割り当てが示されます。

![image](./images/tf_profiler/input_pipeline_analyzer.png)

TensorFlow 演算ごとに 1 つの行があり、各行には次の列があります。

- **Op Name**: The name of the TensorFlow op.
- **Allocation Size (GiBs)**: The total amount of memory allocated to this op.
- **Requested Size (GiBs)**: The total amount of memory requested for this op.
- **Occurrences**: The number of allocations for this op.
- **Region type**: The data entity type that this allocated memory is for. Possible values are `temp` for temporaries, `output` for activations and gradients, and `persist`/`dynamic` for weights and constants.
- **Data type**: The tensor element type.
- **Shape**: The shape of the allocated tensors.

注意: テーブル内のすべての列は並べ替え可能で、演算名で行をフィルタできます。

<a name="pod_viewer"></a>

### Pod ビューア

Pod ビューアツールには、すべてのワーカーのトレーニングステップの詳細が表示されます。

![image](./images/tf_profiler/pod_viewer.png)

- The upper pane has a slider for selecting the step number.
- 下部のペインには、スタックされた列のグラフが表示されます。これは相互に重なったステップ時間カテゴリの詳細を示す要約です。各スタックの列は、一意のワーカーを表します。
- スタックされた列にマウスポインタを合わせると、左側のカードにそのステップの詳細に関するさらに詳しい情報が表示されます。

<a name="tf_data_bottleneck_analysis"></a>

### tf.data のボトルネック分析

Warning: This tool is experimental. Please open a [GitHub Issue](https://github.com/tensorflow/profiler/issues) if the analysis result seems incorrect.

The `tf.data` bottleneck analysis tool automatically detects bottlenecks in `tf.data` input pipelines in your program and provides recommendations on how to fix them. It works with any program using `tf.data` regardless of the platform (CPU/GPU/TPU). Its analysis and recommendations are based on this [guide](https://www.tensorflow.org/guide/data_performance_analysis).

次のステップで、ボトルネックを検出します。

1. 最も多い入力バウンドのホストを見つけます。
2. Find the slowest execution of a `tf.data` input pipeline.
3. プロファイラのトレースから入力パイプラインのグラフを再構築します。
4. 入力パイプライングラフの重要なパスを見つけます。
5. その重要なパスで最も遅い変換をボトルネックとして識別します。

UI は、パフォーマンス分析サマリー、全入力パイプラインのサマリー、入力パイプラインのグラフの 3 つのセクションに分かれています。

#### パフォーマンス分析サマリー

![image](./images/tf_profiler/capture_profile.png)

This section provides the summary of the analysis. It reports on slow `tf.data` input pipelines detected in the profile. This section also shows the most input bound host and its slowest input pipeline with the max latency. Most importantly, it identifies which part of the input pipeline is the bottleneck and how to fix it. The bottleneck information is provided with the iterator type and its long name.

##### tf.data イテレータのロング名の読み取り方

ロング名は、`Iterator::<Dataset_1>::...::<Dataset_n>` のような形式です。ロング名内の `<Dataset_n>` は、イテレータの種類に一致しており、ロング名の他のデータセットは、下流の変換を表します。

たとえば、次の入力パイプラインデータセットを見てみましょう。

```python
dataset = tf.data.Dataset.range(10).map(lambda x: x).repeat(2).batch(5)
```

上記のデータセットから、イテレータのロング名は次のように読み取れます。

イテレータの種類 | ロング名
:-- | :--
範囲 | Iterator::Batch::Repeat::Map::Range
マップ | Iterator::Batch::Repeat::Map
反復 | Iterator::Batch::Repeat
バッチ | Iterator::Batch

#### Summary of all input pipelines

![image](./images/tf_profiler/tf_stats.png)

This section provides the summary of all input pipelines across all hosts. Typically there is one input pipeline. When using the distribution strategy, there is one host input pipeline running the program's `tf.data` code and multiple device input pipelines retrieving data from the host input pipeline and transferring it to the devices.

入力パイプラインごとに、実行時間の統計が表示されます。50 μs より長くかかる呼び出しは、遅いと見なされます。

#### Input pipeline graph

![image](./images/tf_profiler/memory_breakdown_table.png)

This section shows the input pipeline graph with the execution time information. You can use "Host" and "Input Pipeline" to choose which host and input pipeline to see. Executions of the input pipeline are sorted by the execution time in descending order which you can choose using the **Rank** dropdown.

![image](./images/tf_profiler/trace_viewer.png)

重要なパスにあるノードは太いアウトラインで示されます。ボトルネックノードは重要なパスにある、それ自体の処理に最も時間のかかったノードで、赤いアウトラインで示されます。その他の重要でないノードは、グレーの破線で示されます。

In each node,**Start Time** indicates the start time of the execution. The same node may be executed multiple times, for example, if there is a `Batch` op in the input pipeline. If it is executed multiple times, it is the start time of the first execution.

**Total Duration** is the wall time of the execution. If it is executed multiple times, it is the sum of the wall times of all executions.

**Self Time** is **Total Time** without the overlapped time with its immediate child nodes.

「# Calls」は、入力パイプラインが実行された回数です。

<a name="collect_performance_data"></a>

## パフォーマンスデータの収集

TensorFlow プロファイラは、TensorFlow モデルのホストアクティビティと GPU トレースを収集します。プロファイラは、プログラムモードかサンプリングモードのいずれかでパフォーマンスデータを収集するように構成できます。

### プロファイリング API

次の API を使用してプロファイリングを実行できます。

- TensorBoard Keras のコールバックを使用したプログラムモード（`tf.keras.callbacks.TensorBoard`）

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

注意: プロファイラを長時間実行すると、メモリ不足になる可能性があります。一度にプロファイリングするのは 10 ステップまでにすることをお勧めします。初期化のオーバーヘッドによる精度低下を回避するため、最初の数バッチはプロファイリングを避けてください。

<a name="sampling_mode"></a>

- Sampling mode: Perform on-demand profiling by using `tf.profiler.experimental.server.start` to start a gRPC server with your TensorFlow model run. After starting the gRPC server and running your model, you can capture a profile through the **Capture Profile** button in the TensorBoard profile plugin. Use the script in the Install profiler section above to launch a TensorBoard instance if it is not already running.

    以下に例を示します。

    ```python
    # Start a profiler server before your model runs.
    tf.profiler.experimental.server.start(6009)
    # (Model code goes here).
    #  Send a request to the profiler server to collect a trace of your model.
    tf.profiler.experimental.client.trace('grpc://localhost:6009',
                                          'gs://your_tb_logdir', 2000)
    ```

    複数のワーカーのプロファイリング例を以下に示します。

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

以下の項目を指定するには、**Capture Profile** ダイアログを使用します。

- A comma-delimited list of profile service URLs or TPU names.
- プロファイリング期間
- デバイス、ホスト、Python 関数呼び出しのトレースレベル
- 初回失敗時にプロファイラにプロファイルのキャプチャを再試行させる回数

### カスタムトレーニングループのプロファイリング

To profile custom training loops in your TensorFlow code, instrument the training loop with the `tf.profiler.experimental.Trace` API to mark the step boundaries for the Profiler.

The `name` argument is used as a prefix for the step names, the `step_num` keyword argument is appended in the step names, and the `_r` keyword argument makes this trace event get processed as a step event by the Profiler.

以下に例を示します。

```python
for step in range(NUM_STEPS):
    with tf.profiler.experimental.Trace('train', step_num=step, _r=1):
        train_data = next(dataset)
        train_step(train_data)
```

これにより、プロファイラのステップごとのパフォーマンス分析が有効になり、ステップイベントがトレースビューアに表示されるようになります。

Make sure that you include the dataset iterator within the `tf.profiler.experimental.Trace` context for accurate analysis of the input pipeline.

以下のコードスニペットはアンチパターンです。

警告: このコードにより、不正確な入力パイプライン分析が実行されます。

```python
for step, train_data in enumerate(dataset):
    with tf.profiler.experimental.Trace('train', step_num=step, _r=1):
        train_step(train_data)
```

### プロファイリングの使用事例

プロファイラは、4 種類の軸に沿って多数の使用事例をカバーしています。これらの組み合わせの中には、現在サポートされているものもあれば、今後の追加が予定されているものもあります。次に一部の使用事例を示します。

- *Local vs. remote profiling*: These are two common ways of setting up your profiling environment. In local profiling, the profiling API is called on the same machine your model is executing, for example, a local workstation with GPUs. In remote profiling, the profiling API is called on a different machine from where your model is executing, for example, on a Cloud TPU.
- 複数のワーカーのプロファイリング: TensorFlow の分散トレーニング機能を使用すると、複数のマシンをプロファイリングできます。
- ハードウェアプラットフォーム: CPU、GPU、TPU のプロファイリング。

The table below provides a quick overview of the TensorFlow-supported use cases mentioned above:

<a name="profiling_api_table"></a>

| プロファイリング API                | ローカル     | リモート    | 複数  | ハードウェア  | :                              :           :           : ワーカー   : プラットフォーム : | :--------------------------- | :-------- | :-------- | :-------- | :-------- | | **TensorBoard Keras          | サポート対象 | サポート       | サポート       | CPU、GPU  | : コールバック**                   :           : 対象外 : 対象外 :           : | **`tf.profiler.experimental` | サポート対象 | サポート       | サポート       | CPU、GPU  | : start/stop [API](https://www.tensorflow.org/api_docs/python/tf/profiler/experimental#functions_2)**    :           : 対象外 : 対象外 :           : | **`tf.profiler.experimental` | サポート対象 | サポート対象 | サポート対象 | CPU、GPU, | : client.trace [API](https://www.tensorflow.org/api_docs/python/tf/profiler/experimental#functions_2)**  :           :           :           : TPU       : | **コンテキストマネージャ API**      | サポート対象 | サポート       | サポート       | CPU、GPU  | :                              :           : 対象外 : 対象外 :           :

<a name="performance_best_practices"></a>

## 最適なモデルパフォーマンスのベストプラクティス

TensorFlow モデルに適用可能な次の推奨事項を参照し、最適なパフォーマンスを実現してください。

一般的にはデバイス上ですべての変換を実行し、cuDNN や Intel MKL などのご使用のプラットフォームと互換性のあるライブラリの最新バージョンを使用するようにしてください。

### 入力データパイプラインの最適化

Use the data from the [#input_pipeline_analyzer] to optimize your data input pipeline. An efficient data input pipeline can drastically improve the speed of your model execution by reducing device idle time. Try to incorporate the best practices detailed in the [Better performance with the tf.data API](https://www.tensorflow.org/guide/data_performance) guide and below to make your data input pipeline more efficient.

- In general, parallelizing any ops that do not need to be executed sequentially can significantly optimize the data input pipeline.

- In many cases, it helps to change the order of some calls or to tune the arguments such that it works best for your model. While optimizing the input data pipeline, benchmark only the data loader without the training and backpropagation steps to quantify the effect of the optimizations independently.

- Try running your model with synthetic data to check if the input pipeline is a performance bottleneck.

- Use `tf.data.Dataset.shard` for multi-GPU training. Ensure you shard very early on in the input loop to prevent reductions in throughput. When working with TFRecords, ensure you shard the list of TFRecords and not the contents of the TFRecords.

- Parallelize several ops by dynamically setting the value of `num_parallel_calls` using `tf.data.AUTOTUNE`.

- Consider limiting the usage of `tf.data.Dataset.from_generator` as it is slower compared to pure TensorFlow ops.

- Consider limiting the usage of `tf.py_function` as it cannot be serialized and is not supported to run in distributed TensorFlow.

- Use `tf.data.Options` to control static optimizations to the input pipeline.

Also read the `tf.data` performance analysis [guide](https://www.tensorflow.org/guide/data_performance_analysis) for more guidance on optimizing your input pipeline.

#### Optimize data augmentation

When working with image data, make your [data augmentation](https://www.tensorflow.org/tutorials/images/data_augmentation) more efficient by casting to different data types <b><i>after</i></b> applying spatial transformations, such as flipping, cropping, rotating, etc.

Note: Some ops like `tf.image.resize` transparently change the `dtype` to `fp32`. Make sure you normalize your data to lie between `0` and `1` if its not done automatically. Skipping this step could lead to `NaN` errors if you have enabled [AMP](https://developer.nvidia.com/automatic-mixed-precision).

#### Use NVIDIA® DALI

In some instances, such as when you have a system with a high GPU to CPU ratio, all of the above optimizations may not be enough to eliminate bottlenecks in the data loader caused due to limitations of CPU cycles.

If you are using NVIDIA® GPUs for computer vision and audio deep learning applications, consider using the Data Loading Library ([DALI](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/examples/getting%20started.html)) to accelerate the data pipeline.

Check the [NVIDIA® DALI: Operations](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/supported_ops.html) documentation for a list of supported DALI ops.

### Use threading and parallel execution

Run ops on multiple CPU threads with the `tf.config.threading` API to execute them faster.

TensorFlow automatically sets the number of parallelism threads by default. The thread pool available for running TensorFlow ops depends on the number of CPU threads available.

Control the maximum parallel speedup for a single op by using `tf.config.threading.set_intra_op_parallelism_threads`. Note that if you run multiple ops in parallel, they will all share the available thread pool.

If you have independent non-blocking ops (ops with no directed path between them on the graph), use `tf.config.threading.set_inter_op_parallelism_threads` to run them concurrently using the available thread pool.

### Miscellaneous

When working with smaller models on NVIDIA® GPUs, you can set `tf.compat.v1.ConfigProto.force_gpu_compatible=True` to force all CPU tensors to be allocated with CUDA pinned memory to give a significant boost to model performance. However, exercise caution while using this option for unknown/very large models as this might negatively impact the host (CPU) performance.

### デバイスのパフォーマンス改善

Follow the best practices detailed here and in the [GPU performance optimization guide](https://www.tensorflow.org/guide/gpu_performance_analysis) to optimize on-device TensorFlow model performance.

If you are using NVIDIA GPUs, log the GPU and memory utilization to a CSV file by running:

```shell
nvidia-smi
--query-gpu=utilization.gpu,utilization.memory,memory.total,
memory.free,memory.used --format=csv
```

#### Configure data layout

When working with data that contains channel information (like images), optimize the data layout format to prefer channels last (NHWC over NCHW).

Channel-last data formats improve [Tensor Core](https://www.nvidia.com/en-gb/data-center/tensor-cores/) utilization and provide significant performance improvements especially in convolutional models when coupled with AMP. NCHW data layouts can still be operated on by Tensor Cores, but introduce additional overhead due to automatic transpose ops.

You can optimize the data layout to prefer NHWC layouts by setting `data_format="channels_last"` for layers such as `tf.keras.layers.Conv2D`, `tf.keras.layers.Conv3D`, and `tf.keras.layers.RandomRotation`.

Use `tf.keras.backend.set_image_data_format` to set the default data layout format for the Keras backend API.

#### Max out the L2 cache

When working with NVIDIA® GPUs, execute the code snippet below before the training loop to max out the L2 fetch granularity to 128 bytes.

```python
import ctypes

_libcudart = ctypes.CDLL('libcudart.so')
# Set device limit on the current device
# cudaLimitMaxL2FetchGranularity = 0x05
pValue = ctypes.cast((ctypes.c_int*1)(), ctypes.POINTER(ctypes.c_int))
_libcudart.cudaDeviceSetLimit(ctypes.c_int(0x05), ctypes.c_int(128))
_libcudart.cudaDeviceGetLimit(pValue, ctypes.c_int(0x05))
assert pValue.contents.value == 128
```

#### Configure GPU thread usage

The GPU thread mode decides how GPU threads are used.

Set the thread mode to `gpu_private` to make sure that preprocessing does not steal all the GPU threads. This will reduce the kernel launch delay during training. You can also set the number of threads per GPU. Set these values using environment variables.

```python
import os

os.environ['TF_GPU_THREAD_MODE']='gpu_private'
os.environ['TF_GPU_THREAD_COUNT']='1'
```

#### Configure GPU memory options

In general, increase the batch size and scale the model to better utilize GPUs and get higher throughput. Note that increasing the batch size will change the model’s accuracy so the model needs to be scaled by tuning hyperparameters like the learning rate to meet the target accuracy.

Also, use `tf.config.experimental.set_memory_growth` to allow GPU memory to grow to prevent all the available memory from being fully allocated to ops that require only a fraction of the memory. This allows other processes which consume GPU memory to run on the same device.

To learn more, check out the [Limiting GPU memory growth](https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth) guidance in the GPU guide to learn more.

#### Miscellaneous

- Increase the training mini-batch size (number of training samples used per device in one iteration of the training loop) to the maximum amount that fits without an out of memory (OOM) error on the GPU. Increasing the batch size impacts the model's accuracy—so make sure you scale the model by tuning hyperparameters to meet the target accuracy.

- Disable reporting OOM errors during tensor allocation in production code. Set `report_tensor_allocations_upon_oom=False` in `tf.compat.v1.RunOptions`.

- For models with convolution layers, remove bias addition if using batch normalization. Batch normalization shifts values by their mean and this removes the need to have a constant bias term.

- Use TF Stats to find out how efficiently on-device ops run.

- Use `tf.function` to perform computations and optionally, enable the `jit_compile=True` flag (`tf.function(jit_compile=True`). To learn more, go to [Use XLA tf.function](https://www.tensorflow.org/xla/tutorials/jit_compile).

- Minimize host Python operations between steps and reduce callbacks. Calculate metrics every few steps instead of at every step.

- Keep the device compute units busy.

- Send data to multiple devices in parallel.

- Consider [using 16-bit numerical representations](https://www.tensorflow.org/guide/mixed_precision), such as `fp16`—the half-precision floating point format specified by IEEE—or the Brain floating-point [bfloat16](https://cloud.google.com/tpu/docs/bfloat16) format.

## 追加リソース

- The [TensorFlow Profiler: Profile model performance](https://www.tensorflow.org/tensorboard/tensorboard_profiling_keras) tutorial with Keras and TensorBoard where you can apply the advice in this guide.
- The [Performance profiling in TensorFlow 2](https://www.youtube.com/watch?v=pXHAQIhhMhI) talk from the TensorFlow Dev Summit 2020.
- The [TensorFlow Profiler demo](https://www.youtube.com/watch?v=e4_4D7uNvf8) from the TensorFlow Dev Summit 2020.

## 既知の制限

### TensorFlow 2.2 と TensorFlow 2.3 におけるマルチ GPU のプロファイリング

TensorFlow 2.2 and 2.3 support multiple GPU profiling for single host systems only; multiple GPU profiling for multi-host systems is not supported. To profile multi-worker GPU configurations, each worker has to be profiled independently. From TensorFlow 2.4 multiple workers can be profiled using the `tf.profiler.experimental.client.trace` API.

CUDA® Toolkit 10.2 or later is required to profile multiple GPUs. As TensorFlow 2.2 and 2.3 support CUDA® Toolkit versions only up to 10.1, you need to create symbolic links to `libcudart.so.10.1` and `libcupti.so.10.1`:

```shell
sudo ln -s /usr/local/cuda/lib64/libcudart.so.10.2 /usr/local/cuda/lib64/libcudart.so.10.1
sudo ln -s /usr/local/cuda/extras/CUPTI/lib64/libcupti.so.10.2 /usr/local/cuda/extras/CUPTI/lib64/libcupti.so.10.1
```
