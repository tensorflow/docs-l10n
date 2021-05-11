# TensorFlow プロファイラによる `tf.data` パフォーマンスの分析

## 概要

このガイドは、TensorFlow [プロファイラ](https://www.tensorflow.org/guide/profiler)と [`tf.data`](https://www.tensorflow.org/guide/data) に精通していることを前提にしています。また、ユーザーが入力パイプラインの問題を診断して解決するのに役立つ段階的な手順と例を提供することを目的としています。

まずは TensorFlow ジョブのプロファイルを収集することから始めましょう。[CPU/GPU](https://www.tensorflow.org/guide/profiler#collect_performance_data) と [Cloud TPU](https://cloud.google.com/tpu/docs/cloud-tpu-tools#capture_profile) それぞれの手順を確認できます。

![TensorFlow Trace Viewer](images/data_performance_analysis/trace_viewer.png "The trace viewer page of the TensorFlow Profiler")

以下で詳しく説明する分析ワークフローは、プロファイラのトレースビューアツールに焦点を当てています。このツールは、TensorFlow プログラムによって実行される演算の時間を示すタイムラインを表示し、実行に最も時間がかかる演算を特定できるようにします。トレース ビューアの詳細については、TensorFlow プロファイラガイド内の[こちらのセクション](https://www.tensorflow.org/guide/profiler#trace_viewer)をご覧ください。通常、`tf.data` イベントはホストの CPU タイムラインに表示されます。

## 分析ワークフロー

*以下のワークフローに従ってください。改善に役立つ意見があれば、“comp:data” のラベルを付けて [GitHub の課題を作成](https://github.com/tensorflow/tensorflow/issues/new/choose)してください。*

### 1. `tf.data` パイプラインが十分な速度でデータを生成しているか？

入力パイプラインが TensorFlow プログラムのボトルネックになっているかどうかを確かめるところから始めます。

そのためには、トレースビューアで `IteratorGetNext::DoCompute` 演算を探します。通常はステップの最初で見つけることができます。これらのスライスは、入力パイプラインが要求されたときに要素のバッチを生成するのに要する時間を表しています。Keras を使用しているか、`tf.function` でデータセットをイテレートしている場合、これらは `tf_data_iterator_get_next` スレッドで見つかります。

[分散戦略](https://www.tensorflow.org/guide/distributed_training)を使用している場合、`IteratorGetNext::DoCompute` ではなく `IteratorGetNextAsOptional::DoCompute` イベントが表示されるかもしれません（TensorFlow 2.3 以降）。

![image](images/data_performance_analysis/get_next_fast.png "If your IteratorGetNext::DoCompute calls return quickly, `tf.data` is not your bottleneck.")

**呼び出しの応答が速い場合（&lt;= 50 us）**、データは要求された時点で使用できます。入力パイプラインはボトルネックではありません。より一般的なパフォーマンス分析のヒントについては、[プロファイラガイド](https://www.tensorflow.org/guide/profiler)を参照してください。

![image](images/data_performance_analysis/get_next_slow.png "If your IteratorGetNext::DoCompute calls return slowly, `tf.data` is not producing data quickly enough.")

**呼び出しの応答が遅い場合**、`tf.data` は消費者の要求に対応できません。次のセクションに進んでください。

### 2. データをプリフェッチしているか？

入力パイプラインのベストプラクティスは、`tf.data.Dataset.prefetch` <br>変換を `tf.data` パイプラインの最後に挿入することです。この変換は、入力パイプラインの前処理計算をモデル計算の次のステップと一部重複させ、モデルをトレーニングする際に入力パイプラインのパフォーマンスを最適化するために必要なものです。データをプリフェッチしている場合は、`IteratorGetNext::DoCompute` 演算と同じスレッドに `Iterator::Prefetch` スライスが表示されているはずです。

![image](images/data_performance_analysis/prefetch.png "If you're prefetching data, you should see a `Iterator::Prefetch` slice in the same stack as the `IteratorGetNext::DoCompute` op.")

**`prefetch` がパイプラインの最後に存在しない場合**、1 つ追加してください。`tf.data` のパフォーマンスに関する推奨事項の詳細は、[tf.data パフォーマンスガイド](https://www.tensorflow.org/guide/data_performance#prefetching)をご覧してください。

**すでにデータをプリフェッチしており**、入力パイプラインが引き続きボトルネックになっている場合は、次のセクションに進み、さらにパフォーマンス分析を行ってください。

### 3. CPU 使用率が高くなっているか？

`tf.data` は利用可能なリソースを最大限に活用し、高いスループットを達成します。一般的にはモデルを GPU や TPU などのアクセラレータ上で実行している場合でも、`tf.data` パイプラインは CPU 上で実行されています。使用率は [sar](https://linux.die.net/man/1/sar) や [htop](https://en.wikipedia.org/wiki/Htop) のようなツールで、または GCP を利用している場合は[クラウド監視コンソール](console.cloud.google.com/compute/instances)で確認できます。

**使用率が低い場合**、入力パイプラインがホストの CPU を最大限に活用していない可能性があります。[tf.data パフォーマンスガイド](https://www.tensorflow.org/guide/data_performance)でベストプラクティスを調べることをお勧めします。ベストプラクティスを適用しても使用率とスループットが低い場合は、以下の[ボトルネック分析](#4_bottleneck_analysis)に進んでください。

**使用率がリソースの許容限度に近づいている場合**、パフォーマンスをさらに向上させるには、（不要な計算を回避するなどの対策で）入力パイプラインの効率を上げるか、計算をオフロードする必要があります。

入力パイプラインの効率は、`tf.data` 内の不要な演算を回避することで改善できます。その方法の 1 つは、データがメモリに収まる場合に [`tf.data.Dataset.cache`](https://www.tensorflow.org/guide/data_performance#caching) 変換を計算負荷の高い作業の後に挿入することです。これによってメモリの使用量は増えてしまいますが、計算量は減少します。また、`tf.data` で演算内部の並列処理を無効化することで、効率を 10% 以上高められる可能性があります。入力パイプラインで次のオプションを設定すると無効化できます。

```python
dataset = ...
options = tf.data.Options()
options.experimental_threading.max_intra_op_parallelism = 1
dataset = dataset.with_options(options)
```

### 4. ボトルネック分析

次のセクションでは、トレースビューアで `tf.data` イベントを読み取り、ボトルネックの場所と考えられる緩和策を理解する方法について説明します。

#### プロファイラの `tf.data` イベントを理解する

プロファイラ内の各 `tf.data` イベントには `Iterator::<Dataset>` という名前が付いており、`<Dataset>` はデータセットのソースまたは変換の名前です。各イベントには `Iterator::<Dataset_1>::...::<Dataset_n>` という長い名前も付いており、これは `tf.data` イベントをクリックすると確認できます。長い名前の中にある `<Dataset_n>` は（短い）名前の `<Dataset>` と一致しており、それ以外のデータセットは下流の変換を表しています。

![image](images/data_performance_analysis/map_long_name.png "tf.data.Dataset.range(10).map(lambda x: x).repeat(2).batch(5)")

たとえば、上のスクリーンショットは次のコードから生成されたものです。

```python
dataset = tf.data.Dataset.range(10)
dataset = dataset.map(lambda x: x)
dataset = dataset.repeat(2)
dataset = dataset.batch(5)
```

ここでは、`Iterator::Map` イベントに `Iterator::BatchV2::FiniteRepeat::Map` とう長い名前が付いています。データセット名は Python API と若干の違いはあるかもしれませんが（Repeat の代わりに FiniteRepeat になっているなど）、問題なく直感的に解析できるはずです。

##### 同期変換と非同期変換

同期的な `tf.data` 変換（`Batch` や `Map` など）の場合、上流の変換からのイベントは同じスレッドに表示されます。上記の例では、使用されているすべての変換が同期しているため、すべてのイベントが同じスレッドに表示されています。

非同期変換（`Prefetch`、`ParallelMap`、`ParallelInterleave`、`MapAndBatch`）の場合、上流の変換からのイベントは異なるスレッドに表示されます。このような場合、「長い名前」がイベントに対応するパイプラインの変換を識別するのに役立ちます。

![image](images/data_performance_analysis/async_long_name.png "tf.data.Dataset.range(10).map(lambda x: x).repeat(2).batch(5).prefetch(1)")

たとえば、上のスクリーンショットは次のコードから生成されたものです。

```python
dataset = tf.data.Dataset.range(10)
dataset = dataset.map(lambda x: x)
dataset = dataset.repeat(2)
dataset = dataset.batch(5)
dataset = dataset.prefetch(1)
```

ここでは `Iterator::Prefetch` イベントが `tf_data_iterator_get_next` スレッドにあります。`Prefetch` は非同期であるため、その入力イベント（`BatchV2`）は別のスレッドに表示され、`Iterator::Prefetch::BatchV2` という長い名前を検索することで特定できます。この場合、イベントは `tf_data_iterator_resource` スレッドにあります。その長い名前から、`BatchV2` は `Prefetch` の上流であると推察できます。また、`BatchV2` イベントの `parent_id` は `Prefetch` イベントの ID に一致します。

#### ボトルネックの特定

一般的に入力パイプラインのボトルネックを特定するには、入力パイプラインを最も外側の変換からソースまで調査します。パイプラインの最後の変換から始めて、遅い変換が見つかるか、`TFRecord` などのソースデータセットに到達するまで、上流の変換に向かって再帰的に調査します。上記の例では、 `Prefetch` から開始し、`BatchV2`、`FiniteRepeat`、`Map` のように上流に向かい、最後に `Range` を調査します。

一般的に、遅い変換ではそのイベントの長さに対し、入力イベントが短くなっています。以下にいくつかの例を示します。

ほとんどのホスト側の入力パイプラインで最後の（最も外側の）変換が `Iterator::Model` イベントになっていることに注意してください。モデルの変換は `tf.data` ランタイムによって自動的に導入されており、入力パイプラインのインストルメント化とパフォーマンス調整に使用されています。

ジョブが[分散戦略](https://www.tensorflow.org/guide/distributed_training)を使用している場合、トレースビューアにはデバイス側の入力パイプラインに対応する追加のイベントが含まれます。デバイス側パイプラインの最も外側の変換（`IteratorGetNextOp::DoCompute` か `IteratorGetNextAsOptionalOp::DoCompute` の下でネストしている）は、上流の `Iterator::Generator` イベントを持つ `Iterator::Prefetch` イベントになります。対応するホスト側のパイプラインは `Iterator::Model` イベントを探すと見つかります。

##### 例1

![image](images/data_performance_analysis/example_1_cropped.png "Example 1")

上のスクリーンショットは次の入力パイプラインから生成されたものです。

```python
dataset = tf.data.TFRecordDataset(filename)
dataset = dataset.map(parse_record)
dataset = dataset.batch(32)
dataset = dataset.repeat()
```

このスクリーンショットでは、(1) `Iterator::Map` イベントが長いものの、(2) その入力イベント（`Iterator::FlatMap`）がすぐに応答を返していることが分かります。これは、連続する Map 変換がボトルネックであることを示唆しています。

このスクリーンショットでは、`InstantiatedCapturedFunction::Run` イベントは map 関数を実行するのに要する時間に対応しています。

##### 例2

![image](images/data_performance_analysis/example_2_cropped.png "Example 2")

上のスクリーンショットは次の入力パイプラインから生成されたものです。

```python
dataset = tf.data.TFRecordDataset(filename)
dataset = dataset.map(parse_record, num_parallel_calls=2)
dataset = dataset.batch(32)
dataset = dataset.repeat()
```

この例は上記のものと似ていますが、Map ではなく ParallelMap を使用しています。ここでは (1) `Iterator::ParallelMap` イベントが長いものの、(2) その入力イベント `Iterator::FlatMap` （ParallelMap が非同期であるため、別スレッド上に存在します）は短くなっています。これは、ParallelMap 変換がボトルネックであることを示唆しています。

#### ボトルネックへの対処

##### ソース データセット

TFRecord ファイルからの読み込みなど、データセットソースがボトルネックであることを確認したら、データ抽出を並列化することでパフォーマンスを改善できます。そのためには、データが複数のファイル間でシャーディングされ、`num_parallel_calls` パラメータを `tf.data.experimental.AUTOTUNE` に設定して `tf.data.Dataset.interleave` を使用するようにしてください。決定論がプログラムにとって重要でない場合、`tf.data.Dataset.interleave` に `deterministic=False` フラグを設定することで、さらにパフォーマンスを改善できます（TensorFlow 2.2 以降）。たとえば、TFRecord を読み込む場合は次のようにします。

```python
dataset = tf.data.Dataset.from_tensor_slices(filenames)
dataset = dataset.interleave(tf.data.TFRecordDataset,
  num_parallel_calls=tf.data.experimental.AUTOTUNE,
  deterministic=False)
```

ファイルを開く際のオーバーヘッドを埋め合わせるため、シャーディングされたファイルは適度な大きさになっていなければなりません。並列データ抽出の詳細については、`tf.data` パフォーマンスガイドの[このセクション](https://www.tensorflow.org/guide/data_performance#parallelizing_data_extraction)をご覧ください。

##### 変換データセット

中間の `tf.data` 変換がボトルネックであることを確認したら、変換を並列化するか、（データがメモリに収まり、かつ適切な場合に）[計算をキャッシュする](https://www.tensorflow.org/guide/data_performance#caching)ことで対処できます。`Map` などの一部の変換には対応する並列変換があります。<a data-md-type="raw_html" href="https://www.tensorflow.org/guide/data_performance#parallelizing_data_transformation">`tf.data` パフォーマンスガイド</a>では、これらの並列化方法が説明されています。`Filter`、`Unbatch`、`Batch` などのその他の変換は本質的に順次実行されますが、これらは「外部並列処理」を導入することで並列ができます。たとえば、入力パイプラインが初期状態で次のようになっており、`Batch` がボトルネックになっているとします。

```python
filenames = tf.data.Dataset.list_files(file_path, shuffle=is_training)
dataset = filenames_to_dataset(filenames)
dataset = dataset.batch(batch_size)
```

シャーディングされた入力に対して入力パイプラインの複数のコピーを実行し、それらの結果を組み合わせることで、「外部並列処理」を導入できます。

```python
filenames = tf.data.Dataset.list_files(file_path, shuffle=is_training)

def make_dataset(shard_index):
  filenames = filenames.shard(NUM_SHARDS, shard_index)
  dataset = filenames_to_dataset(filenames)
  Return dataset.batch(batch_size)

indices = tf.data.Dataset.range(NUM_SHARDS)
dataset = indices.interleave(make_dataset,
                             num_parallel_calls=tf.data.experimental.AUTOTUNE)
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
```

## その他のリソース

- パフォーマンスに優れた `tf.data` 入力パイプラインの記述方法に関する [tf.data のパフォーマンスガイド](https://www.tensorflow.org/guide/data_performance)
- [Inside TensorFlow video: `tf.data` best practices ](https://www.youtube.com/watch?v=ZnukSLKEw34)
- [プロファイラガイド](https://www.tensorflow.org/guide/profiler)
- [Colab でのプロファイラチュートリアル](https://www.tensorflow.org/tensorboard/tensorboard_profiling_keras)
