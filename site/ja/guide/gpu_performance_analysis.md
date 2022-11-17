# TensorFlow Profiler を使用した TensorFlow GPU パフォーマンスの最適化

## 概要

このガイドでは、TensorBoard で TensorFlow Profiler を使用して、GPU の洞察を得て最大のパフォーマンスを引き出し、1 つ以上の GPU が十分に活用されていない場合にデバッグする方法を示します。

Profiler を初めて使用する場合は、次を行います。

- Keras の例と [TensorBoard](https://www.tensorflow.org/tensorboard) を使って、[TensorFlow Profiler: モデルパフォーマンスをプロファイリングする](https://www.tensorflow.org/tensorboard/tensorboard_profiling_keras)ノートブックを使い始める。
- [Profiler を使用した TensorFlow のパフォーマンス最適化](https://www.tensorflow.org/guide/profiler#profiler_tools)ガイドで、ホスト（CPU）で TensorFlow のパフォーマンスを最適化するために使用できるさまざまなプロファイリングツールと方法について学びます。

計算を GPU にオフロードすることは、特に小さなモデルの場合、常にメリットがあるとは限らないことに注意してください。次の理由により、オーバーヘッドが発生する可能性があります。

- ホスト（CPU）とデバイス（GPU）間のデータ転送
- ホストが GPU カーネルを起動するときの遅延のため

### パフォーマンス最適化のワークフロー

このガイドでは、単一の GPU から始めて、複数の GPU を備えた単一のホストに移行して、パフォーマンスの問題をデバッグする方法について概説します。

次の順序でパフォーマンスの問題をデバッグすることをお勧めします。

1. 1 つの GPU でパフォーマンスを最適化してデバッグします。
    1. 入力パイプラインがボトルネックになっていないか確認します。
    2. 1 つの GPU でパフォーマンスをデバッグします。
    3. 混合精度（`fp16`（float16）を使用）を有効にし、オプションで [XLA](https://www.tensorflow.org/xla) を有効にします。
2. マルチ GPU 単一ホストでのパフォーマンスを最適化してデバッグします。

たとえば、TensorFlow [分散戦略](https://www.tensorflow.org/guide/distributed_training)を使用して、複数の GPU を備えた単一のホストでモデルをトレーニングし、最適でない GPU 使用率に気付いた場合、マルチ GPU システムをデバッグする前に、最初に 1 つの GPU のパフォーマンスを最適化してデバッグする必要があります。

GPU でパフォーマンスの高いコードを取得するためのベースラインとして、このガイドでは既に `tf.function` を使用していることを前提としています。Keras `Model.compile` および `Model.fit` API は、内部で `tf.function` を自動的に利用します。`tf.GradientTape` を使用してカスタムトレーニングループを作成する場合、`tf.function` を有効にする方法については、[tf.function によるパフォーマンスの改善](https://www.tensorflow.org/guide/function)をご覧ください。

次のセクションでは、パフォーマンスのボトルネックを特定して修正するために、上記のシナリオごとに推奨されるアプローチについて説明します。

## 1. 1 つの GPU でパフォーマンスを最適化する

理想的なケースでは、プログラムの GPU 使用率が高く、CPU（ホスト）から GPU（デバイス）への通信が最小限であり、入力パイプラインからのオーバーヘッドがない必要があります。

パフォーマンスを分析する最初のステップは、1 つの GPU で実行されているモデルのプロファイルを取得することです。

TensorBoard の Profiler [概要ページ](https://www.tensorflow.org/guide/profiler#overview_page)（プロファイル実行中にモデルがどのように実行されたかのトップレベルビューを表示）は、プログラムが理想的なシナリオからどれだけ離れているかを示すことができます。

![TensorFlow Profiler Overview Page](images/gpu_perf_analysis/overview_page.png "TensorFlow Profiler の概要ページ")

概要ページで注意すべき重要な点は次のとおりです。

1. 実際のデバイスの実行からのステップ時間の割合
2. デバイスとホストに配置された演算の割合
3. `fp16` を使用するカーネルの数

パフォーマンスの最適化を実現するということは、3 つのケースすべてでこれらの数値を最大化することを意味します。プログラムを深く理解するには、TensorBoard の Profiler [トレースビューア](https://www.tensorflow.org/guide/profiler#trace_viewer)に精通している必要があります。以下のセクションでは、パフォーマンスのボトルネックを診断するときに探す必要がある一般的なトレースビューアのパターンをいくつか示します。

以下は、1 つの GPU で実行されているモデルトレースビューの画像です。*TensorFlow Name Scope* セクションと *TensorFlow Ops* セクションから、フォワードパス、損失関数、バックワードパス/勾配計算、オプティマイザの重み値の更新など、モデルのさまざまな部分を識別できます。また、CUDA ストリームを参照する各 *Stream* の隣の GPU で演算を実行することもできます。各ストリームは特定のタスクに使用されます。このトレースでは、*Stream#118* を使用して計算カーネルとデバイス間のコピーを起動します。*Stream#119* はホストからデバイスへのコピーに使用され、*Stream#120* はデバイスからホストへのコピーに使用されます。

以下のトレースは、パフォーマンスモデルの一般的な特性を示しています。

![image](images/gpu_perf_analysis/traceview_ideal.png "An example TensorFlow Profiler trace view")

たとえば、GPU 計算タイムライン（*Stream#118*）はギャップがほとんどなく「ビジー」に見えます。ホストからデバイスへのコピー（*ストリーム #119*）およびデバイスからホストへのコピー（*ストリーム #120*）は最小限であり、ステップ間のギャップも最小限です。プログラムの Profiler を実行すると、トレースビューでこれらの理想的な特性を特定できない場合があります。このガイドの残りの部分では、一般的なシナリオとその修正方法について説明します。

### 1. 入力パイプラインをデバッグする

GPU パフォーマンスのデバッグでの最初のステップは、プログラムが入力バウンドかどうかを判断することです。これを把握する最も簡単な方法は、TensorBoard で Profiler の[入力パイプラインアナライザー](https://www.tensorflow.org/guide/profiler#input_pipeline_analyzer)を使用することです。これは、入力パイプラインで費やされた時間の概要を提供します。

![image](images/gpu_perf_analysis/input_pipeline_analyzer.png "TensorFlow Profiler Input-Analyzer")

入力パイプラインがステップ時間に大きく影響する場合、次のアクションが実行可能です。

- `tf.data` 固有の[ガイド](https://www.tensorflow.org/guide/data_performance_analysis)を使用して、入力パイプラインをデバッグする方法を学習できます。
- 入力パイプラインがボトルネックかどうかを確認するもう 1 つの簡単な方法は、前処理を必要としない、ランダムに生成された入力データを使用することです。ResNet モデルでこの手法を使用する[例を次に示します](https://github.com/tensorflow/models/blob/4a5770827edf1c3974274ba3e4169d0e5ba7478a/official/vision/image_classification/resnet/resnet_runnable.py#L50-L57)。入力パイプラインが最適であれば、実際のデータと生成されたランダム/合成データで同様のパフォーマンスが得られるはずです。合成データの場合の唯一のオーバーヘッドは、プリフェッチして最適化できる入力データのコピーによるものです。

さらに、[入力データパイプラインを最適化するためのベストプラクティス](https://www.tensorflow.org/guide/profiler#optimize_the_input_data_pipeline)もご覧ください。

### 2. 1 つの GPU のパフォーマンスをデバッグする

GPU 使用率が低くなる要因はいくつかあります。以下は、[トレースビューア](https://www.tensorflow.org/guide/profiler#trace_viewer)と考えられる解決策を確認する際によく見られるいくつかのシナリオです。

#### 1. ステップ間のギャップを分析する

プログラムが最適に実行されていない場合によく観測されるのは、トレーニングステップ間のギャップです。以下のトレースビューの画像では、ステップ 8 と 9 の間に大きなギャップがあり、その間 GPU がアイドル状態になっていることを意味します。

![image](images/gpu_perf_analysis/traceview_step_gaps.png "ステップ間のギャップを示す TensorFlow プロファイル トレース ビュー")

トレースビューアでステップ間に大きなギャップが表示される場合は、プログラムが入力バウンドであることを示している可能性があります。その場合、入力パイプラインのデバッグに関する前のセクションをまだ参照していない場合は参照する必要があります。

ただし、最適化された入力パイプラインを使用しても、CPU スレッドの競合により、あるステップの終了と別のステップの開始の間にギャップが生じる可能性があります。`tf.data` は、バックグラウンドスレッドを利用してパイプライン処理を並列化します。これらのスレッドは、データのコピーや GPU 演算のスケジューリングなど、各ステップの開始時に発生する GPU ホスト側のアクティビティに干渉する可能性があります。

GPU でこれらの演算をスケジュールするホスト側で大きなギャップに気付いた場合は、環境変数 `TF_GPU_THREAD_MODE=gpu_private` を設定できます。これにより、GPU カーネルが独自の専用スレッドから起動され、`tf.data` 作業の背後でキューに入れられないことが保証されます。

ステップ間のギャップは、指標の計算、Keras コールバック、またはホストで実行される `tf.function` の外部の演算によっても発生する可能性があります。これらの演算は、TensorFlow グラフ内の演算ほどパフォーマンスが良くありません。さらに、これらの演算の一部は CPU 上で実行され、GPU との間でテンソルをコピーします。

入力パイプラインを最適化した後も、トレースビューアのステップ間にギャップがあることに気付いた場合は、ステップ間のモデルコードを調べて、コールバック/指標を無効にすることでパフォーマンスが改善されるかどうかを確認する必要があります。これらの操作の詳細の一部は、トレースビューアでも（デバイス側とホスト側の両方に）表示されます。このシナリオで推奨されるのは、これらの演算のオーバーヘッドを、すべてのステップではなく一定数のステップの後に実行することによって償却することです。`tf.keras` API で `Model.compile` メソッドを使用する場合、`steps_per_execution` フラグを設定すると、これが自動的に行われます。カスタムトレーニングループには、`tf.while_loop` を使用します。

#### 2. より高いデバイス使用率を達成する

##### 1. 小さな GPU カーネルとホストカーネルの起動遅延

ホストはカーネルを GPU で実行するためにキューに入れますが、カーネルが実際に GPU で実行されるまでに遅延（約 20 ～ 40 μs）が伴います。理想的なケースでは、ホストがさらに多くのカーネルをエンキューするのを待つのではなく、GPU がほとんどの時間を実行に費やすように、ホストは GPU に十分な数のカーネルをエンキューします。

TensorBoard の Profiler の[概要ページ](https://www.tensorflow.org/guide/profiler#overview_page)には、ホストがカーネルを起動するのを待っていたために GPU がアイドル状態だった時間が表示されます。下の画像では、カーネルが起動されるのを待っているステップ時間の約 10% の間、GPU がアイドル状態になっています。

![image](images/gpu_perf_analysis/performance_summary.png "Summary of performance from TensorFlow Profile")

この同じプログラムの[トレースビューア](https://www.tensorflow.org/guide/profiler#trace_viewer)は、ホストが GPU でカーネルを起動するためにビジー状態であるカーネル間に小さなギャップを示しています。

![image](images/gpu_perf_analysis/traceview_kernel_gaps.png "TensorFlow Profile trace view demonstrating gaps between kernels")

GPU で多数の小さな演算（スカラー加算など）を起動すると、ホストが GPU に追いつかない可能性があります。同じプロファイルの TensorBoard の [TensorFlow Stats](https://www.tensorflow.org/guide/profiler#tensorflow_stats) ツールは、2.77 秒かかる 126,224 Mul 演算を示しています。したがって、各カーネルは約 21.9 μs であり、これは非常に小さく（起動レイテンシとほぼ同じ時間）、ホストカーネルの起動遅延が発生する可能性があります。

![image](images/gpu_perf_analysis/tensorflow_stats_page.png "TensorFlow Profile stats page")

上記の画像のように、[トレースビューア](https://www.tensorflow.org/guide/profiler#trace_viewer)が GPU 上の演算間に多くの小さなギャップを示している場合は、次のことができます。

- 小さなテンソルを連結し、ベクトル化された演算を使用するか、より大きなバッチサイズを使用して、起動された各カーネルがより多くの作業を行うようにします。これにより、GPU がビジー状態になる時間が長くなります。
- `tf.function` を使用して TensorFlow グラフを作成していることを確認して、演算を純粋な Eager Modeで実行していないことを確認してください。`Model.fit` を使用している場合（`tf.GradientTape` を使用したカスタムトレーニングループではなく）、`tf.keras.Model.compile`は自動的にこれを行います。
- `tf.function(jit_compile=True)` または自動クラスタリングで XLA を使用してカーネルを融合します。詳細については、以下の[混合精度と XLA を有効にする](#3._enable_mixed_precision_and_xla)セクションに移動して、XLA を有効にしてパフォーマンスを向上させる方法を学習してください。この特徴量により、デバイスの使用率が高くなる可能性があります。

##### 2. TensorFlow 演算の配置

Profiler の[概要ページ](https://www.tensorflow.org/guide/profiler#overview_page)には、ホストとデバイスに配置された演算のパーセンテージが表示されます（[トレースビューア](https://www.tensorflow.org/guide/profiler#trace_viewer)を参照して、特定の演算の配置を確認することもできます）。下の画像のように、デバイスに比べて、ホスト上の演算のパーセンテージが非常に小さくなるようにします。

![image](images/gpu_perf_analysis/opp_placement.png "TF Op Placement")

理想的には、計算集約型演算のほとんどを GPU に配置する必要があります。

モデルの演算とテンソルが割り当てられているデバイスを見つけるには、プログラムの最初のステートメントとして `tf.debugging.set_log_device_placement(True)` を設定します。

場合によっては、演算を特定のデバイスに配置するように指定した場合でも、その実装がこの条件をオーバーライドする可能性があることに注意してください（例: `tf.unique`）。単一の GPU トレーニングの場合でも、`tf.distribute.OneDeviceStrategy` などの分散ストラテジーを指定すると、デバイス上で演算をより確定的に配置できます。

演算の大部分を GPU に配置する理由の 1 つは、ホストとデバイス間の過剰なメモリコピーを防ぐことです（ホストとデバイス間のモデル入力/出力データのメモリコピーが予想されます）。過度のコピーの例は、GPU ストリーム *#167*、*#168*、および *#169* に関する以下のトレースビューに示されています。

![image](images/gpu_perf_analysis/traceview_excessive_copy.png "TensorFlow Profile trace view demonstrating excessive H2D/D2H copies")

これらのコピーが GPU カーネルの実行をブロックすると、パフォーマンスが低下することがあります。[トレースビューア](https://www.tensorflow.org/guide/profiler#trace_viewer)のメモリコピー演算には、これらのコピーされたテンソルのソースである演算に関する詳細情報がありますが、memCopy を 演算に関連付けるのは必ずしも容易ではない場合があります。このような場合、近くの演算を調べて、すべてのステップでメモリコピーが同じ場所で発生しているかどうかを確認すると役立ちます。

#### 3. GPU 上のより効率的なカーネル

プログラムの GPU 使用率が許容範囲内になると、次のステップとして、テンソルコアを利用するか演算を融合することによって、GPU カーネルの効率を高めることを検討します。

##### 1. テンソルコアを利用する

最新の NVIDIA® GPU には、適格なカーネルのパフォーマンスを大幅に向上させることができる特殊な [テンソルコア](https://www.nvidia.com/en-gb/data-center/tensor-cores/)があります。

TensorBoard の[GPU カーネル統計](https://www.tensorflow.org/guide/profiler#gpu_kernel_stats)を使用して、どの GPU カーネルがテンソルコアに適しているか、どのカーネルがテンソルコアを使用しているかを視覚化できます。`fp16` を有効にする（以下の「混合精度を有効にする」セクションを参照）ことは、プログラムの General Matrix Multiply（GEMM）カーネル（matmul ops）がテンソルコアを利用するようにする 1 つの方法です。精度が fp16 で、入力/出力テンソルの次元が 8 または 16 で割り切れる場合（`int8` の場合）、GPU カーネルはテンソルコアを効率的に使用します。

注意: cuDNN v7.6.3 以降では、テンソルコアを活用するために必要な場所に畳み込み次元が自動的にパディングされます。

GPU でカーネルを効率的にする方法についてのその他の詳細な推奨事項については、[NVIDIA® ディープラーニングパフォーマンス](https://docs.nvidia.com/deeplearning/performance/index.html#perf-guidelines)ガイドをご覧ください。

##### 2. 融合演算

`tf.function(jit_compile=True)` を使用して小さな演算を融合し、大きなカーネルを形成してパフォーマンスを大幅に向上させます。詳細については、[XLA](https://www.tensorflow.org/xla) ガイドをご覧ください。

### 3. 混合精度と XLA を有効にする

上記の手順を実行した後、混合精度と XLA を有効にすることは、パフォーマンスをさらに向上させるために実行できる 2 つのオプションの手順です。推奨されるアプローチは、それらを 1 つずつ有効にして、パフォーマンス上のメリットが期待どおりであることを確認することです。

#### 1. 混合精度を有効にする

TensorFlow [混合精度](https://www.tensorflow.org/guide/keras/mixed_precision)ガイドは、GPU で `fp16` 精度を有効にする方法を示しています。NVIDIA® GPU で [AMP](https://developer.nvidia.com/automatic-mixed-precision) を有効にしてテンソルコアを使用し、Volta および新しい GPU アーキテクチャで `fp32`（float32）精度のみを使用する場合と比較して、最大 3 倍の全体的なスピードアップを実現します。

行列/テンソルの次元が、テンソルコアを使用するカーネルを呼び出すための要件を満たしていることを確認してください。精度が fp16 で、入出力次元が 8 または 16（int8 の場合）で割り切れる場合、GPU カーネルはテンソルコアを効率的に使用します。

cuDNN v7.6.3 以降では、テンソルコアを活用するために必要な場所に畳み込み次元が自動的にパディングされることに注意してください。

`fp16` 精度のパフォーマンス上のメリットを最大化するには、以下のベストプラクティスに従ってください。

##### 1. 最適な fp16 カーネルを使用する

`fp16` を有効にすると、プログラムの行列乗算（GEMM）カーネルは、テンソルコアを利用する対応する `fp16` バージョンを使用する必要があります。ただし、場合によっては、プログラムが非効率的な実装にフォールバックするため、これが発生せず、`fp16` を有効にしても期待される速度向上が得られません。

![image](images/gpu_perf_analysis/gpu_kernels.png "TensorFlow Profile GPU Kernel Stats page")

[GPU カーネル](https://www.tensorflow.org/guide/profiler#gpu_kernel_stats)統計ページには、どの演算がテンソルコアに適しているか、どのカーネルが実際に効率的なテンソルコアを使用しているかが表示されます。[ディープラーニングパフォーマンスに関する NVIDIA® ガイド](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html#opt-tensor-cores)には、テンソルコアの活用方法についての追加の提案が含まれています。さらに、演算にかかる時間が半減したため、以前はメモリにバインドされていたカーネルでも `fp16` を使用することによるメリットが見られます。

##### 2.動的と静的損失スケーリングの対比

低精度によるアンダーフローを防ぐために、`fp16` を使用する場合は、損失スケーリングが必要です。損失スケーリングには動的と静的の 2 種類があり、どちらも[混合精度ガイド](https://www.tensorflow.org/guide/keras/mixed_precision)で詳しく説明されています。`mixed_float16` ポリシーを使用して、Keras オプティマイザ内で自動的に損失スケーリングを有効にすることができます。

注意: Keras 混合精度 API は、デフォルトでスタンドアロンのソフトマックス演算（Keras 損失関数の一部ではない演算）を `fp16` として評価するため、数値の問題や収束の低下につながる可能性があります。パフォーマンスの最適化には、そのような演算を `fp32` にキャストします。

パフォーマンスを最適化しようとする場合、動的損失スケーリングによって、ホストで実行される追加の条件付き演算が導入され、トレースビューアのステップ間にギャップが生じる可能性があることを覚えておくことが重要です。一方、静的損失スケーリングにはそのようなオーバーヘッドがなく、正しい静的損失スケール値を指定する必要があるため、パフォーマンスの点で優れたオプションになる可能性があります。

#### 2. tf.function(jit_compile=True) または自動クラスタリングで XLA を有効にする

単一の GPU で最高のパフォーマンスを得るための最後のステップとして、XLA を有効にして実験できます。これにより、演算が融合され、デバイスの使用率が向上し、メモリフットプリントが削減されます。プログラムで `tf.function(jit_compile=True)` または自動クラスタリングを使用して XLA を有効にする方法の詳細については、[XLA](https://www.tensorflow.org/xla) ガイドをご覧ください。

グローバル JIT レベルを `-1`（オフ）、`1`、または `2` に設定できます。レベルが高いほどアグレッシブになり、並列処理が減り、より多くのメモリを使用する可能性があります。メモリに制限がある場合は、値を `1` に設定します。 XLA コンパイラは、新しい形状に遭遇するたびにカーネルをコンパイルし続ける必要があるため、変数入力テンソル形状を持つモデルでは XLA が適切に機能しないことに注意してください。

## 2. マルチ GPU 単一ホストでパフォーマンスを最適化する

`tf.distribute.MirroredStrategy` API を使用して、単一ホスト上の 1 つの GPU から複数の GPU にモデル トレーニングをスケーリングできます。（TensorFlow を使用して分散トレーニングを行う方法の詳細については、[TensorFlow を使用した分散トレーニング](https://www.tensorflow.org/guide/distributed_training)、[GPU を使用する](https://www.tensorflow.org/guide/gpu)、[TPUを使用する](https://www.tensorflow.org/guide/tpu)ガイド、および [Keras を使用した分散トレーニング](https://www.tensorflow.org/tutorials/distribute/keras)チュートリアルをご覧ください。）

1 つの GPU から複数の GPU への移行は理想的にはそのままでスケーラブルであるべきですが、パフォーマンスの問題が発生する場合があります。

単一の GPU を使用したトレーニングから同じホスト上の複数の GPU に移行する場合、理想的には、勾配通信の追加のオーバーヘッドとホストスレッドの使用率の増加のみでパフォーマンスのスケーリングを経験するはずです。このオーバーヘッドのため、例えば GPU を 1 つから 2 つに変更した場合、正確に 2 倍のスピードアップは得られません。

以下のトレースビューは、複数の GPU でトレーニングする場合の余分な通信オーバーヘッドの例を示しています。重みの更新を行う前に、勾配を連結し、レプリカ間で伝達し、分割するためのオーバーヘッドがあります。

![image](images/gpu_perf_analysis/traceview_multi_gpu.png "TensorFlow Profile trace view for single host multi GPU scenario")

次のチェックリストは、マルチ GPU シナリオでパフォーマンスを最適化するときにパフォーマンスを向上させるのに役立ちます。

1. バッチサイズを最大化するようにしてください。これにより、デバイスの使用率が向上し、複数の GPU 間の通信コストが償却されます。[メモリプロファイラ](https://www.tensorflow.org/guide/profiler#memory_profile_summary)を使用すると、プログラムがメモリ使用率のピークにどれだけ近づいているかを把握するのに役立ちます。バッチサイズを大きくすると収束に影響を与える可能性がありますが、通常はパフォーマンス上のメリットがそれを上回ります。
2. 単一の GPU から複数の GPU に移行する場合、同じホストでより多くの入力データを処理する必要があります。そのため、（1）の後、入力パイプラインのパフォーマンスを再確認し、ボトルネックになっていないことを確認することをお勧めします。
3. プログラムのトレースビューで GPU タイムラインをチェックして、不要な AllReduce 呼び出しがないか確認してください。この呼び出しにより、すべてのデバイス間で同期が行われるためです。上記のトレースビューでは、AllReduce は [NCCL](https://developer.nvidia.com/nccl) カーネルを介して実行され、各ステップの勾配に対して各 GPU で 1 つの NCCL 呼び出しのみが行われます。
4. 最小化できる不要な D2H、H2D、および D2D コピー操作を確認します。
5. ステップ時間をチェックして、各レプリカが同じ作業を行っていることを確認します。例えば、1 つの GPU（通常は`GPU0`）がオーバーサブスクライブされることがあります。これは、ホストが誤って GPU により多くの作業を行うことになるためです。
6. 最後に、トレースビューですべての GPU のトレーニングステップをチェックして、順番に実行されている演算を確認します。これは通常、ある GPU から別の GPU への制御の依存関係がプログラムに含まれている場合に発生します。以前は、この状況でのパフォーマンスのデバッグは個別に解決されていました。プログラムでこの動作が確認された場合は、トレースビューの画像を添えて [GitHub の課題を提出](https://github.com/tensorflow/tensorflow/issues/new/choose)してください。

### 1. 勾配 AllReduce を最適化する

同期ストラテジーでトレーニングする場合、各デバイスは入力データの一部を受け取ります。

モデルのフォワードパスとバックワードパスを計算した後、各デバイスで計算された勾配を集計して削減する必要があります。この*勾配 AllReduce* は、各デバイスでの勾配計算の後、オプティマイザがモデルの重みを更新する前に発生します。

各 GPU は最初にモデルレイヤー全体で勾配を連結し、`tf.distribute.CrossDeviceOps`（`tf.distribute.NcclAllReduce` がデフォルト）を使用して GPU 間でそれらを通信し、レイヤーごとに削減した後に勾配を返します。

オプティマイザは、これらの減少した勾配を使用して、モデルの重みを更新します。理想的には、オーバーヘッドを防ぐために、このプロセスはすべての GPU で同時に発生する必要があります。

AllReduce にかかる時間は、次とほぼ同じになります。

```
(number of parameters * 4bytes)/ (communication bandwidth)
```

この計算は、分散トレーニングジョブを実行したときのパフォーマンスが期待どおりかどうか、またはさらにパフォーマンスのデバッグを行う必要があるかどうかを理解するためのクイックチェックとして役立ちます。`Model.summary` からモデル内のパラメーターの数を取得できます。

TensorFlow は勾配の伝達に `fp32`（float32）を使用するため、各モデルパラメータのサイズは 4 バイトであることに注意してください。`fp16` を有効にしても、NCCL AllReduce は `fp32` パラメータを利用します。

スケーリングのメリットを得るには、これらのオーバーヘッドに比べてステップ時間を大幅に長くする必要があります。これを実現する 1 つの方法は、バッチサイズがステップ時間に影響するため、より大きなバッチサイズを使用することですが、通信のオーバーヘッドには影響しません。

### 2. GPU ホストスレッドの競合

複数の GPU を実行している場合、CPU の仕事は、デバイス間で GPU カーネルを効率的に起動することで、すべてのデバイスをビジー状態に保つことです。

ただし、CPU が 1 つの GPU でスケジュールできる独立した演算が多数ある場合、CPU は多くのホストスレッドを使用して 1 つの GPU をビジー状態に保ち、別の GPU で非確定的な順序でカーネルを起動することを決定できます。これにより、スキューまたは負のスケーリングが発生し、パフォーマンスに悪影響を及ぼす可能性があります。

以下の[トレースビューア](https://www.tensorflow.org/guide/profiler#trace_viewer)は、`GPU1` がアイドル状態で、`GPU2` の起動後に演算の実行を開始するため、CPU が GPU カーネルを非効率的に起動する際のオーバーヘッドを示しています。

![image](images/gpu_perf_analysis/traceview_gpu_idle.png "TensorFlow Profile device trace view demonstrating inefficient kernel launch")

ホストのトレースビューは、ホストがカーネルを `GPU1` で起動する前に `GPU2` で起動していることを示しています（以下の `tf_Compute*` 演算は CPU スレッドを示すものではないことに注意してください）。

![image](images/gpu_perf_analysis/traceview_host_contention.png "TensorFlow Profile host trace view demonstrating inefficient kernel launch")

プログラムのトレースビューでこの種の GPU カーネルのずれが発生した場合、推奨されるアクションは次のとおりです。

- TensorFlow 環境変数 `TF_GPU_THREAD_MODE` を `gpu_private` に設定します。この環境変数は、GPU のスレッドを非公開にするようにホストに指示します。
- デフォルトでは、`TF_GPU_THREAD_MODE=gpu_private` はスレッド数を 2 に設定します。ほとんどの場合、これで十分です。ただし、TensorFlow 環境変数 `TF_GPU_THREAD_COUNT` を目的のスレッド数に設定することで変更できます。
