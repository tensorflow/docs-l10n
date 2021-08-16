# TF Profiler로 `tf.data` 성능 분석

## 개요

이 가이드는 사용자가 TensorFlow [Profiler](https://www.tensorflow.org/guide/profiler) 및 [`tf.data`](https://www.tensorflow.org/guide/data)에 익숙하다고 가정합니다. 사용자가 입력 파이프라인의 성능 문제를 진단하고 수정하는 데 도움이 되는 예제와 함께 단계별 지침을 제공하는 것이 이 가이드의 목적입니다.

시작하려면 TensorFlow 연산의 프로필을 수집하세요. 수집 방법에 대한 지침은 [CPU/GPU](https://www.tensorflow.org/guide/profiler#collect_performance_data) 및 [Cloud TPU](https://cloud.google.com/tpu/docs/cloud-tpu-tools#capture_profile)에 사용할 수 있습니다.

![TensorFlow Trace Viewer](images/data_performance_analysis/trace_viewer.png "The trace viewer page of the TensorFlow Profiler")

아래에 설명된 분석 워크플로는 Profiler의 추적 뷰어 도구에 중점을 둡니다. 이 도구는 TensorFlow 프로그램이 실행하는 연산의 기간을 보여주는 타임라인을 표시하므로 가장 오래 실행되는 연산을 식별할 수 있습니다. 추적 뷰어에 대한 자세한 정보는 TF Profiler 가이드의 [이 섹션](https://www.tensorflow.org/guide/profiler#trace_viewer)을 확인하세요. 일반적으로 `tf.data` 이벤트는 호스트 CPU 타임라인에 나타납니다.

## 분석 워크플로

*아래의 워크플로를 따르세요. 개선에 도움이 되는 의견이 있으면 "comp:data" 레이블로 [github 문제를 보고하세요](https://github.com/tensorflow/tensorflow/issues/new/choose).*

### 1. `tf.data` 파이프라인이 데이터를 충분히 빠르게 생성하고 있습니까?

먼저, TensorFlow 프로그램의 입력 파이프라인이 병목 상태인지 확인하세요.

병목 상태를 확인하려면 추적 뷰어에서 `IteratorGetNext::DoCompute` ops를 찾으세요. 일반적으로, 단계의 시작 부분에서 병목 상태를 볼 수 있습니다. 해당 조각은 요청 시 입력 파이프라인에서 요소를 일괄 생성하는 데 걸리는 시간을 나타냅니다. Keras를 사용하거나 `tf.function`에서 데이터세트를 반복하는 경우, `tf_data_iterator_get_next` 스레드에서 볼 수 있습니다.

[배포 전략](https://www.tensorflow.org/guide/distributed_training)을 사용하는 경우, `IteratorGetNext::DoCompute` 대신 `IteratorGetNextAsOptional::DoCompute` 이벤트를 볼 수도 있습니다(TF 2.3 기준).

![image](images/data_performance_analysis/get_next_fast.png "If your IteratorGetNext::DoCompute calls return quickly, `tf.data` is not your bottleneck.")

**호출이 빨리 반환되면(&lt;= 50us),** 요청 시 데이터를 사용할 수 있음을 의미합니다. 입력 파이프라인은 병목 상태가 아닙니다. 보다 일반적인 성능 분석 팁은 [Profiler 가이드](https://www.tensorflow.org/guide/profiler)를 참조하세요.

![image](images/data_performance_analysis/get_next_slow.png "If your IteratorGetNext::DoCompute calls return slowly, `tf.data` is not producing data quickly enough.")

**호출이 늦게 반환되면,** `tf.data`가 소비자의 요청 속도를 맞출 수 없습니다. 다음 섹션으로 계속 진행하세요.

### 2. 데이터를 프리페치하고 있습니까?

입력 파이프라인 성능에 대한 모범 사례는 `tf.data` 파이프라인의 끝에 `tf.data.Dataset.prefetch` 변환을 삽입하는 것입니다. 이 변환으로 입력 파이프라인의 전처리 계산과 모델 계산의 다음 단계가 겹치며, 이 변환은 모델 학습 시 최적의 입력 파이프라인 성능을 위해 필요합니다. 데이터를 프리페치하는 경우, `IteratorGetNext::DoCompute` op와 동일한 스레드에서 `Iterator::Prefetch` 조각을 볼 수 있습니다.

![image](images/data_performance_analysis/prefetch.png "If you're prefetching data, you should see a `Iterator::Prefetch` slice in the same stack as the `IteratorGetNext::DoCompute` op.")

**파이프라인의 끝에 `prefetch`가 없는 경우**, 하나 추가해야 합니다. `tf.data` 성능 권장 사항에 대한 자세한 정보는 [tf.data 성능 가이드](https://www.tensorflow.org/guide/data_performance#prefetching)를 참조하세요.

**이미 데이터를 프리페치하고 있고**, 입력 파이프라인이 여전히 병목 상태인 경우, 다음 섹션으로 계속 진행하여 성능을 자세히 분석하세요.

### 3. CPU 사용률이 높게 유지됩니까?

`tf.data`는 사용 가능한 리소스를 최대한 활용하여 높은 처리량을 달성합니다. 일반적으로, GPU 또는 TPU와 같은 가속기에서 모델을 실행할 때도 `tf.data` 파이프라인은 CPU에서 실행됩니다. [sar](https://linux.die.net/man/1/sar) 및 [htop](https://en.wikipedia.org/wiki/Htop)과 같은 도구를 사용하거나 GCP에서 실행 중인 경우는 [클라우드 모니터링 콘솔](console.cloud.google.com/compute/instances)에서 사용률을 확인할 수 있습니다.

**사용률이 낮으면**, 입력 파이프라인이 호스트 CPU를 충분히 활용하지 못하고 있음을 나타냅니다. 모범 사례는 [tf.data 성능 가이드](https://www.tensorflow.org/guide/data_performance)를 참조하세요. 모범 사례를 적용하고 활용률과 처리량이 낮게 유지되면 아래 [병목 상태 분석](#4_bottleneck_analysis)을 계속 진행하세요.

**사용률이 리소스 한도에 근접하고 있으면**, 성능을 더 향상하기 위해 입력 파이프라인의 효율성을 개선하거나(예: 불필요한 계산 방지) 오프로드 계산을 수행해야 합니다.

`tf.data`에서 불필요한 계산을 방지함으로써 입력 파이프라인의 효율성을 개선할 수 있습니다. 이를 수행하는 한 가지 방법은 데이터를 메모리에 저장하기 적합한 경우, 계산 집약적인 연산 후에 [`tf.data.Dataset.cache`](https://www.tensorflow.org/guide/data_performance#caching) 변환을 삽입하는 것입니다. 이렇게 하면 메모리 사용량이 증가하는 대신 계산이 감소합니다. 또한, `tf.data`에서 intra-op 병렬 처리를 비활성화하면 효율성이 10% 이상 증가할 수 있으며 입력 파이프라인에서 다음 옵션을 설정하여 수행할 수 있습니다.

```python
dataset = ...
options = tf.data.Options()
options.experimental_threading.max_intra_op_parallelism = 1
dataset = dataset.with_options(options)
```

### 4. 병목 상태 분석

다음 섹션에서는 추적 뷰어에서 `tf.data` 이벤트를 읽고 병목 상태의 위치와 가능한 완화 전략을 이해하는 방법을 안내합니다.

#### Profiler의 `tf.data` 이벤트 이해

Profiler에서 각 `tf.data` 이벤트의 이름은 `Iterator::<Dataset>`이며, 여기에서 `<Dataset>`는 데이터세트 소스 또는 변환의 이름입니다. 각 이벤트에는 긴 이름 `Iterator::<Dataset_1>::...::<Dataset_n>`도 있으며, `tf.data` 이벤트를 클릭하면 볼 수 있습니다. 긴 이름에서, `<Dataset_n>`이 (짧은) 이름의 `<Dataset>`와 일치하며, 긴 이름에서 기타 데이터세트는 다운스트림 변환을 나타냅니다.

![image](images/data_performance_analysis/map_long_name.png "tf.data.Dataset.range(10).map(lambda x: x).repeat(2).batch(5)")

예를 들어, 위의 스크린샷은 다음 코드에서 생성되었습니다.

```python
dataset = tf.data.Dataset.range(10)
dataset = dataset.map(lambda x: x)
dataset = dataset.repeat(2)
dataset = dataset.batch(5)
```

여기에서 `Iterator::Map` 이벤트의 긴 이름은 `Iterator::BatchV2::FiniteRepeat::Map`입니다. 데이터세트 이름은 파이썬 API에서와 약간 다를 수 있지만(예: Repeat 대신 FiniteRepeat), 구문 분석할 수 있을 정도로 직관적이어야 합니다.

##### 동기식 및 비동기 변환

동기식 `tf.data` 변환(예: `Batch` 및 `Map`)의 경우, 업스트림 변환 이벤트는 동일한 스레드에서 볼 수 있습니다. 위 예제에서 사용된 모든 변환은 동기식이므로 모든 이벤트가 동일한 스레드에 표시됩니다.

비동기 변환(예: `Prefetch`, `ParallelMap`, `ParallelInterleave` 및 `MapAndBatch`)의 경우, 업스트림 변환 이벤트는 다른 스레드에 표시됩니다. 이러한 경우, "긴 이름"은 이벤트가 해당하는 파이프라인에서 변환을 식별하는 데 도움이 됩니다.

![image](images/data_performance_analysis/async_long_name.png "tf.data.Dataset.range(10).map(lambda x: x).repeat(2).batch(5).prefetch(1)")

예를 들어, 위의 스크린샷은 다음 코드에서 생성되었습니다.

```python
dataset = tf.data.Dataset.range(10)
dataset = dataset.map(lambda x: x)
dataset = dataset.repeat(2)
dataset = dataset.batch(5)
dataset = dataset.prefetch(1)
```

`Iterator::Prefetch` 이벤트는 `tf_data_iterator_get_next` 스레드에 표시됩니다. `Prefetch`는 비동기식이므로 입력 이벤트 (`BatchV2`)는 다른 스레드에 표시되며, 긴 이름 `Iterator::Prefetch::BatchV2`를 검색하여 찾을 수 있습니다. 이 경우에는 `tf_data_iterator_resource` 스레드에 표시됩니다. 긴 이름으로부터 `BatchV2`가 `Prefetch`의 업스트림임을 추론할 수 있습니다. 또한, `BatchV2` 이벤트의 `parent_id`는 `Prefetch` 이벤트의 ID와 일치합니다.

#### 병목 상태 식별

일반적으로, 입력 파이프라인의 병목 상태를 식별하려면 입력 파이프라인의 가장 바깥 쪽 변환에서 소스까지 이동하세요. 파이프라인의 마지막 변환에서 시작하여 느린 변환을 찾거나 `TFRecord`와 같은 소스 데이터세트에 도달할 때까지 재귀적으로 업스트림 변환을 확인하세요. 위 예제에서는, `Prefetch`에서 시작하여 `BatchV2`, `FiniteRepeat`, `Map`, 마지막으로 `Range`까지 업스트림으로 이동합니다.

일반적으로, 느린 변환은 이벤트는 길지만 입력 이벤트는 짧은 변환에 해당합니다. 몇 가지 예는 다음과 같습니다.

대부분의 호스트 입력 파이프라인에서 마지막 (가장 바깥 쪽) 변환은 `Iterator::Model` 이벤트입니다. Model 변환은 ` tf.data` 런타임에 의해 자동으로 지정되며 입력 파이프라인 성능을 계측하고 자동 튜닝하는 데 사용됩니다.

연산에서 [분배 전략](https://www.tensorflow.org/guide/distributed_training)을 사용 중인 경우, 추적 뷰어에는 기기 입력 파이프라인에 해당하는 추가 이벤트가 포함됩니다. 기기 파이프라인(`IteratorGetNextOp::DoCompute` 또는 `IteratorGetNextAsOptionalOp::DoCompute` 내부에서 중첩됨)의 가장 바깥 쪽 변환은 업스트림 `Iterator::Generator` 이벤트가 있는 `Iterator::Prefetch` 이벤트입니다. `Iterator::Model` 이벤트를 검색하여 해당 호스트 파이프라인을 찾을 수 있습니다.

##### 예제 1

![image](images/data_performance_analysis/example_1_cropped.png "Example 1")

위의 스크린샷은 다음 입력 파이프라인에서 생성됩니다.

```python
dataset = tf.data.TFRecordDataset(filename)
dataset = dataset.map(parse_record)
dataset = dataset.batch(32)
dataset = dataset.repeat()
```

스크린샷에서 (1)`Iterator::Map` 이벤트는 길지만, (2) 입력 이벤트 (`Iterator::FlatMap`)는 빨리 반환됩니다. 따라서, 순차적 Map 변환이 병목 상태임을 알 수 있습니다.

스크린샷에서 `InstantiatedCapturedFunction::Run` 이벤트는 맵 함수를 실행하는 데 걸리는 시간에 해당합니다.

##### 예제 2

![image](images/data_performance_analysis/example_2_cropped.png "Example 2")

위의 스크린샷은 다음 입력 파이프라인에서 생성됩니다.

```python
dataset = tf.data.TFRecordDataset(filename)
dataset = dataset.map(parse_record, num_parallel_calls=2)
dataset = dataset.batch(32)
dataset = dataset.repeat()
```

이 예제는 위와 비슷하지만, Map 대신 ParallelMap을 사용합니다. 여기에서 (1) `Iterator::ParallelMap` 이벤트는 길지만, (2) 입력 이벤트 `Iterator::FlatMap`(ParallelMap이 비동기식이므로 다른 스레드에 표시됨)은 짧습니다. 따라서, ParallelMap 변환이 병목 상태임을 알 수 있습니다.

#### 병목 상태 해결

##### 소스 데이터세트

TFRecord 파일에서 읽을 때와 같이 데이터세트 소스를 병목 상태로 식별한 경우, 데이터 추출을 병렬화하여 성능을 향상할 수 있습니다. 이렇게 하려면 데이터가 여러 파일에 걸쳐 분할되어 있는지 확인하고 `num_parallel_calls` 매개변수가 `tf.data.experimental.AUTOTUNE`로 설정된 `tf.data.Dataset.interleave`를 사용하세요. 프로그램에 결정론이 중요하지 않은 경우, TF 2.2에서는 `tf.data.Dataset.interleave`에서 `deterministic=False ` 플래그를 설정하여 성능을 추가로 향상할 수 있습니다. 예를 들어, TFRecords에서 읽는 경우, 다음을 수행할 수 있습니다.

```python
dataset = tf.data.Dataset.from_tensor_slices(filenames)
dataset = dataset.interleave(tf.data.TFRecordDataset,
  num_parallel_calls=tf.data.experimental.AUTOTUNE,
  deterministic=False)
```

분할된 파일은 파일을 여는 오버헤드를 상쇄하기 위해 적당히 커야 합니다. 병렬 데이터 추출에 대한 자세한 내용은 `tf.data` 성능 가이드의 [이 섹션](https://www.tensorflow.org/guide/data_performance#parallelizing_data_extraction)을 참조하세요.

##### 변환 데이터세트

중간 `tf.data` 변환을 병목 상태로 식별한 경우, 데이터를 메모리에 저장하기 적합하다면 변환을 병렬화하거나 [계산을 캐싱](https://www.tensorflow.org/guide/data_performance#caching)하여 해결할 수 있습니다. `Map`과 같은 일부 변환에는 병렬 부분이 있습니다. <a data-md-type="raw_html" href="https://www.tensorflow.org/guide/data_performance#parallelizing_data_transformation">`tf.data` 성능 가이드</a>는 이를 병렬화하는 방법을 보여줍니다. `Filter`, `Unbatch` 및 `Batch`와 같은 기타 변환은 기본적으로 순차적입니다. "외부 병렬 처리"를 사용하여 병렬화할 수 있습니다. 예를 들어, 입력 파이프라인이 처음에 다음과 같으며 `Batch`를 병목 상태로 가정합니다.

```python
filenames = tf.data.Dataset.list_files(file_path, shuffle=is_training)
dataset = filenames_to_dataset(filenames)
dataset = dataset.batch(batch_size)
```

분할된 입력에 대해 여러 개의 입력 파이프라인 사본을 실행하고 결과를 결합함으로써 "외부 병렬 처리"를 사용할 수 있습니다.

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

## 추가 자료

- 성능 `tf.data` 입력 파이프라인의 작성 방법에 대한 [tf.data 성능 가이드](https://www.tensorflow.org/guide/data_performance)
- [Profiler 가이드](https://www.tensorflow.org/guide/profiler)
- [colab을 사용한 Profiler 튜토리얼](https://www.tensorflow.org/tensorboard/tensorboard_profiling_keras)
- [Colab을 사용한 프로파일 러 자습서](https://www.tensorflow.org/tensorboard/tensorboard_profiling_keras)
