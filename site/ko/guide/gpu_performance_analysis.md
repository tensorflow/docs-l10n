# TensorFlow Profiler로 TensorFlow GPU 성능 최적화

## Overview

이 가이드에서는 TensorBoard와 함께 TensorFlow Profiler를 사용하여 GPU에 대한 통찰력을 얻고 최대 성능을 얻고, 하나 이상의 GPU가 충분히 활용되지 않을 때 디버그하는 방법을 보여줍니다.

프로파일러를 처음 사용하는 경우:

- [TensorFlow Profiler:](https://www.tensorflow.org/tensorboard/tensorboard_profiling_keras) Keras 예제 및 [TensorBoard가 포함된](https://www.tensorflow.org/tensorboard) 프로필 모델 성능 노트북으로 시작하세요.
- 프로파일러 가이드를 [사용하여 TensorFlow 성능 최적화](https://www.tensorflow.org/guide/profiler#profiler_tools) 를 통해 호스트(CPU)에서 TensorFlow 성능을 최적화하는 데 사용할 수 있는 다양한 프로파일링 도구 및 방법에 대해 알아보세요.

GPU로 계산을 오프로딩하는 것이 특히 작은 모델의 경우 항상 유익한 것은 아니라는 점을 염두에 두십시오. 다음으로 인해 오버헤드가 발생할 수 있습니다.

- 호스트(CPU)와 장치(GPU) 간의 데이터 전송 그리고
- 호스트가 GPU 커널을 시작할 때 관련된 대기 시간으로 인해.

### 성능 최적화 워크플로

이 가이드에서는 단일 GPU에서 시작하여 여러 GPU가 있는 단일 호스트로 이동하는 성능 문제를 디버그하는 방법을 간략하게 설명합니다.

다음 순서로 성능 문제를 디버그하는 것이 좋습니다.

1. 하나의 GPU에서 성능 최적화 및 디버그:
    1. 입력 파이프라인이 병목 상태인지 확인합니다.
    2. 한 GPU의 성능을 디버그합니다.
    3. 혼합 정밀도( `fp16` (float16) 사용)를 활성화하고 선택적으로 [XLA를](https://www.tensorflow.org/xla) 활성화합니다.
2. 다중 GPU 단일 호스트에서 성능을 최적화하고 디버그합니다.

예를 들어 TensorFlow [배포 전략](https://www.tensorflow.org/guide/distributed_training) 을 사용하여 여러 GPU가 있는 단일 호스트에서 모델을 훈련하고 최적이 아닌 GPU 사용률을 확인한 경우 다중 GPU 시스템을 디버깅하기 전에 먼저 하나의 GPU에 대한 성능을 최적화하고 디버그해야 합니다.

GPU에서 고성능 코드를 얻기 위한 기준으로 이 가이드에서는 이미 `tf.function` 사용하고 있다고 가정합니다. `Model.compile` 및 `Model.fit` API는 후드 아래에서 자동으로 `tf.function` 와 사용자 정의 교육 루프를 작성할 때 `tf.GradientTape` , 참고하여주십시오 [tf.function와 더 나은 성능을](https://www.tensorflow.org/guide/function) 사용하는 방법에 `tf.function` 의.

다음 섹션에서는 성능 병목 현상을 식별하고 수정하는 데 도움이 되도록 위의 각 시나리오에 대해 제안된 접근 방식에 대해 설명합니다.

## 1. 하나의 GPU에서 성능 최적화

이상적인 경우 프로그램은 GPU 사용률이 높고 CPU(호스트)와 GPU(장치) 간의 통신이 최소화되어야 하며 입력 파이프라인의 오버헤드가 없어야 합니다.

성능 분석의 첫 번째 단계는 하나의 GPU로 실행되는 모델의 프로필을 얻는 것입니다.

TensorBoard의 Profiler [개요 페이지(](https://www.tensorflow.org/guide/profiler#overview_page) 프로필 실행 중에 모델이 어떻게 수행되었는지에 대한 최상위 보기를 보여줌)는 프로그램이 이상적인 시나리오에서 얼마나 멀리 떨어져 있는지에 대한 아이디어를 제공할 수 있습니다.

![TensorFlow Profiler Overview Page](images/gpu_perf_analysis/overview_page.png "TensorFlow 프로파일러의 개요 페이지")

개요 페이지에 주의해야 할 주요 번호는 다음과 같습니다.

1. 실제 장치 실행에서 얼마나 많은 단계 시간이 소요되는지
2. 장치 대 호스트에 배치된 작업의 비율
3. 얼마나 많은 커널이 `fp16`

최적의 성능을 달성한다는 것은 세 가지 경우 모두에서 이러한 수치를 최대화하는 것을 의미합니다. 프로그램을 심층적으로 이해하려면 TensorBoard의 Profiler [추적 뷰어에](https://www.tensorflow.org/guide/profiler#trace_viewer) 익숙해야 합니다. 아래 섹션에서는 성능 병목 현상을 진단할 때 찾아야 하는 몇 가지 일반적인 추적 뷰어 패턴을 보여줍니다.

아래는 하나의 GPU에서 실행되는 모델 추적 보기의 이미지입니다. *TensorFlow Name Scope* 및 *TensorFlow Ops* 섹션에서 정방향 통과, 손실 함수, 역방향 통과/기울기 계산, 최적화기 가중치 업데이트와 같은 모델의 다양한 부분을 식별할 수 있습니다. CUDA 스트림을 참조하는 *각 Stream* 옆의 GPU에서 작업을 실행할 수도 있습니다. 각 스트림은 특정 작업에 사용됩니다. 이 추적에서 *Stream#118* 은 컴퓨팅 커널 및 장치 간 복사를 시작하는 데 사용됩니다. *Stream#119* 는 호스트 대 장치 복사에 사용되며 *Stream#120* 은 장치 대 호스트 복사에 사용됩니다.

아래 추적은 성능 모델의 일반적인 특성을 보여줍니다.

![image](images/gpu_perf_analysis/traceview_ideal.png "TensorFlow Profiler 추적 보기의 예")

예를 들어 GPU 컴퓨팅 타임라인( *Stream#118* )은 간격이 거의 없는 "바쁜" 것처럼 보입니다. 호스트에서 장치로( *Stream #119* ) 및 장치에서 호스트로( *Stream #120* ) 최소한의 복사본이 있으며 단계 간의 간격도 최소화됩니다. 프로그램에 대해 프로파일러를 실행할 때 추적 보기에서 이러한 이상적인 특성을 식별하지 못할 수 있습니다. 이 가이드의 나머지 부분에서는 일반적인 시나리오와 해결 방법을 다룹니다.

### 1. 입력 파이프라인 디버그

GPU 성능 디버깅의 첫 번째 단계는 프로그램이 입력 바인딩되어 있는지 확인하는 것입니다. 이를 파악하는 가장 쉬운 방법은 입력 파이프라인에 소요된 시간에 대한 개요를 제공하는 TensorBoard [에서 Profiler의 입력 파이프라인 분석기를 사용하는 것입니다.](https://www.tensorflow.org/guide/profiler#input_pipeline_analyzer)

![image](images/gpu_perf_analysis/input_pipeline_analyzer.png "TensorFlow 프로파일러 입력 분석기")

입력 파이프라인이 단계 시간에 크게 기여하는 경우 다음과 같은 잠재적 조치를 취할 수 있습니다.

- `tf.data` 관련 [가이드](https://www.tensorflow.org/guide/data_performance_analysis) 를 사용하여 입력 파이프라인을 디버그하는 방법을 배울 수 있습니다.
- 입력 파이프라인이 병목 현상인지 확인하는 또 다른 빠른 방법은 사전 처리가 필요하지 않은 무작위로 생성된 입력 데이터를 사용하는 것입니다. [다음은](https://github.com/tensorflow/models/blob/4a5770827edf1c3974274ba3e4169d0e5ba7478a/official/vision/image_classification/resnet/resnet_runnable.py#L50-L57) ResNet 모델에 이 기술을 사용하는 예입니다. 입력 파이프라인이 최적인 경우 실제 데이터와 생성된 임의/합성 데이터에서 유사한 성능을 경험해야 합니다. 합성 데이터 경우의 유일한 오버헤드는 다시 프리페치 및 최적화할 수 있는 입력 데이터 복사로 인한 것입니다.

[또한 입력 데이터 파이프라인 최적화를 위한 모범 사례를](https://www.tensorflow.org/guide/profiler#optimize_the_input_data_pipeline) 참조하십시오.

### 2. 하나의 GPU 성능 디버그

GPU 사용률을 낮추는 데 기여할 수 있는 몇 가지 요인이 있습니다. [다음은 추적 뷰어](https://www.tensorflow.org/guide/profiler#trace_viewer) 및 잠재적 솔루션을 볼 때 일반적으로 관찰되는 몇 가지 시나리오입니다.

#### 1. 단계 간 격차 분석

프로그램이 최적으로 실행되지 않을 때 일반적으로 관찰되는 것은 훈련 단계 사이의 간격입니다. 아래의 트레이스 보기 이미지에서 8단계와 9단계 사이에 큰 간격이 있습니다. 이는 해당 시간 동안 GPU가 유휴 상태임을 의미합니다.

![image](images/gpu_perf_analysis/traceview_step_gaps.png "단계 간의 간격을 보여주는 TensorFlow 프로필 추적 보기")

추적 뷰어에서 단계 사이에 큰 간격이 표시되면 프로그램이 입력 바인딩되었음을 나타낼 수 있습니다. 이 경우 아직 수행하지 않은 경우 입력 파이프라인 디버깅에 대한 이전 섹션을 참조해야 합니다.

그러나 최적화된 입력 파이프라인을 사용하더라도 CPU 스레드 경합으로 인해 한 단계의 끝과 다른 단계의 시작 사이에 여전히 간격이 있을 수 있습니다. `tf.data` 는 백그라운드 스레드를 사용하여 파이프라인 처리를 병렬화합니다. 이러한 스레드는 데이터 복사 또는 GPU 작업 예약과 같이 각 단계 시작 시 발생하는 GPU 호스트 측 활동을 방해할 수 있습니다.

GPU에서 이러한 작업을 예약하는 호스트 측에서 큰 간격을 발견하면 환경 변수 `TF_GPU_THREAD_MODE=gpu_private` 설정할 수 있습니다. 이렇게 하면 GPU 커널이 자체 전용 스레드에서 시작되고 `tf.data` 작업 뒤에 대기하지 않습니다.

단계 사이의 간격은 메트릭 계산, Keras 콜백 또는 호스트에서 실행되는 `tf.function` 이러한 작업은 TensorFlow 그래프 내부의 작업만큼 성능이 좋지 않습니다. 또한 이러한 작업 중 일부는 CPU에서 실행되고 GPU에서 앞뒤로 텐서를 복사합니다.

입력 파이프라인을 최적화한 후에도 추적 뷰어에서 단계 사이의 간격이 계속 보이면 단계 사이의 모델 코드를 살펴보고 콜백/메트릭을 비활성화하면 성능이 향상되는지 확인해야 합니다. 이러한 작업에 대한 일부 세부 정보는 트레이스 뷰어(장치 및 호스트 측 모두)에도 있습니다. 이 시나리오의 권장 사항은 모든 단계 대신 고정된 수의 단계 후에 실행하여 이러한 작업의 오버헤드를 상각하는 것입니다. `tf.keras` `compile` 방법을 사용할 때 `experimental_steps_per_execution` 플래그를 설정하면 자동으로 수행됩니다. 사용자 지정 훈련 루프의 경우 `tf.while_loop` 사용합니다.

#### 2. 더 높은 장치 활용도 달성

##### 1. 작은 GPU 커널 및 호스트 커널 실행 지연

호스트는 GPU에서 실행할 커널을 대기열에 넣지만 커널이 실제로 GPU에서 실행되기 전에 관련된 지연 시간(약 20-40μs)이 있습니다. 이상적인 경우 호스트는 호스트가 더 많은 커널을 대기열에 넣을 때까지 기다리지 않고 GPU가 실행하는 데 대부분의 시간을 할애하도록 충분한 커널을 GPU에 대기열에 넣습니다.

TensorBoard의 Profiler [개요 페이지](https://www.tensorflow.org/guide/profiler#overview_page) 는 호스트가 커널을 시작하기를 기다리는 동안 GPU가 유휴 상태였던 시간을 보여줍니다. 아래 이미지에서 GPU는 커널 실행을 기다리는 단계 시간의 약 10% 동안 유휴 상태입니다.

![image](images/gpu_perf_analysis/performance_summary.png "TensorFlow 프로필의 성능 요약")

[이 동일한 프로그램에 대한 추적 뷰어](https://www.tensorflow.org/guide/profiler#trace_viewer) 는 호스트가 GPU에서 커널을 시작하는 데 바쁜 커널 사이의 작은 간격을 보여줍니다.

![image](images/gpu_perf_analysis/traceview_kernel_gaps.png "커널 간의 간격을 보여주는 TensorFlow 프로필 추적 보기")

GPU에서 많은 작은 작업(예: 스칼라 추가)을 시작하면 호스트가 GPU를 따라가지 못할 수 있습니다. 동일한 프로필에 대한 TensorBoard의 [TensorFlow Stats](https://www.tensorflow.org/guide/profiler#tensorflow_stats) 도구는 2.77초가 소요되는 126,224 Mul 작업을 보여줍니다. 따라서 각 커널은 약 21.9μs로 매우 작고(시작 대기 시간과 거의 같은 시간) 잠재적으로 호스트 커널 시작 지연이 발생할 수 있습니다.

![image](images/gpu_perf_analysis/tensorflow_stats_page.png "TensorFlow 프로필 통계 페이지")

[추적 뷰어에](https://www.tensorflow.org/guide/profiler#trace_viewer) 위의 이미지와 같이 GPU의 작업 간에 많은 작은 간격이 표시되는 경우 다음을 수행할 수 있습니다.

- 작은 텐서를 연결하고 벡터화된 연산을 사용하거나 더 큰 배치 크기를 사용하여 실행된 각 커널이 더 많은 작업을 수행하도록 하면 GPU가 더 오래 사용됩니다.
- 순수 열망 모드에서 작업을 실행하지 않도록 `tf.function` 을 사용하여 TensorFlow 그래프를 생성하고 있는지 확인하십시오. `Model.fit` 을 사용하는 경우( tf.GradientTape 가 있는 사용자 지정 교육 루프와 `tf.GradientTape` ), `tf.keras.Model.compile` 이 자동으로 이 작업을 수행합니다.
- `tf.function(jit_compile=True)` 또는 자동 클러스터링과 함께 사용하여 커널을 융합합니다. 자세한 내용은 [아래의 혼합 정밀도 및 XLA 활성화](#3._enable_mixed_precision_and_xla) 섹션으로 이동하여 XLA를 활성화하여 더 높은 성능을 얻는 방법을 알아보세요. 이 기능은 높은 장치 활용도로 이어질 수 있습니다.

##### 2. TensorFlow 연산 배치

프로파일러 [개요 페이지](https://www.tensorflow.org/guide/profiler#overview_page) 는 호스트에 배치된 작업 대 장치의 백분율을 보여줍니다( [추적 뷰어](https://www.tensorflow.org/guide/profiler#trace_viewer) 를 보고 특정 작업의 배치를 확인할 수도 있습니다. 아래 이미지와 같이 호스트에 있는 작업의 백분율을 원합니다. 장치에 비해 매우 작습니다.

![image](images/gpu_perf_analysis/opp_placement.png "TF 작전 배치")

이상적으로는 대부분의 컴퓨팅 집약적 작업을 GPU에 배치해야 합니다.

모델의 작업과 텐서가 할당된 장치를 찾으려면 `tf.debugging.set_log_device_placement(True)` 를 프로그램의 첫 번째 명령문으로 설정하십시오.

경우에 따라 특정 장치에 배치할 작업을 지정하더라도 해당 구현이 이 조건을 재정의할 수 있습니다(예: `tf.unique` ). `tf.distribute.OneDeviceStrategy` 와 같은 배포 전략을 지정하면 장치에 작업을 더 결정적으로 배치할 수 있습니다.

대부분의 연산을 GPU에 배치하는 한 가지 이유는 호스트와 장치 간의 과도한 메모리 복사를 방지하기 위함입니다(호스트와 장치 간의 모델 입력/출력 데이터에 대한 메모리 복사가 예상됨). *과도한 복사의 예는 GPU 스트림 #167* , *#168* 및 *#169* 에 대한 아래의 추적 보기에 나와 있습니다.

![image](images/gpu_perf_analysis/traceview_excessive_copy.png "과도한 H2D/D2H 사본을 보여주는 TensorFlow 프로필 추적 보기")

이러한 복사본은 GPU 커널의 실행을 차단하는 경우 성능을 저하시킬 수 있습니다. [추적 뷰어](https://www.tensorflow.org/guide/profiler#trace_viewer) 의 메모리 복사 작업에는 이러한 복사된 텐서의 소스인 작업에 대한 추가 정보가 있지만 memCopy를 작업과 연결하는 것이 항상 쉬운 것은 아닙니다. 이러한 경우 모든 단계에서 동일한 위치에서 메모리 복사가 발생하는지 확인하기 위해 주변의 작업을 살펴보는 것이 도움이 됩니다.

#### 3. GPU에서 보다 효율적인 커널

프로그램의 GPU 사용률이 허용 가능한 수준이면 다음 단계는 Tensor Core 또는 융합 작업을 활용하여 GPU 커널의 효율성을 높이는 것입니다.

##### 1. 텐서 코어 활용

최신 NVIDIA® GPU에는 적격 커널의 성능을 크게 향상시킬 수 [있는 특수 Tensor 코어가 있습니다.](https://www.nvidia.com/en-gb/data-center/tensor-cores/)

[TensorBoard의 GPU 커널 통계](https://www.tensorflow.org/guide/profiler#gpu_kernel_stats) 를 사용하여 어떤 GPU 커널이 Tensor Core에 적합하고 어떤 커널이 Tensor Core를 사용하는지 시각화할 수 있습니다. `fp16` 활성화(아래의 혼합 정밀도 활성화 섹션 참조)는 프로그램의 GEMM(General Matrix Multiply) 커널(matmul ops)이 텐서 코어를 활용하도록 하는 한 가지 방법입니다. GPU 커널은 정밀도가 fp16이고 입력/출력 텐서 차원이 8 또는 16( `int8` )으로 나눌 수 있는 경우 Tensor Core를 효율적으로 사용합니다.

참고: cuDNN v7.6.3 이상에서는 Tensor Core를 활용하는 데 필요한 경우 컨볼루션 차원이 자동으로 채워집니다.

GPU에서 커널을 효율적으로 만드는 방법에 대한 기타 자세한 권장 사항은 [NVIDIA® 딥 러닝 성능](https://docs.nvidia.com/deeplearning/performance/index.html#perf-guidelines) 가이드를 참조하십시오.

##### 2. 퓨즈 작업

`tf.function(jit_compile=True)` 을 사용하여 더 작은 연산을 융합하여 더 큰 커널을 형성하여 상당한 성능 향상을 가져옵니다. 자세한 내용은 [XLA](https://www.tensorflow.org/xla) 가이드를 참조하십시오.

### 3. 혼합 정밀도 및 XLA 사용

위의 단계를 수행한 후 혼합 정밀도와 XLA를 활성화하면 성능을 더욱 향상시키기 위해 취할 수 있는 두 가지 선택적 단계입니다. 제안된 접근 방식은 이를 하나씩 활성화하고 성능 이점이 예상대로인지 확인하는 것입니다.

#### 1. 혼합 정밀도 사용

TensorFlow [혼합 정밀도](https://www.tensorflow.org/guide/keras/mixed_precision) 가이드는 GPU에서 `fp16` 정밀도를 활성화하는 방법을 보여줍니다. [NVIDIA® GPU에서 AMP](https://developer.nvidia.com/automatic-mixed-precision) 를 활성화하여 Tensor 코어를 사용하고 Volta 및 최신 GPU 아키텍처에서 `fp32` (float32) 정밀도만 사용하는 것과 비교할 때 전체 속도가 최대 3배 향상됩니다.

행렬/텐서 차원이 텐서 코어를 사용하는 커널 호출에 대한 요구 사항을 충족하는지 확인하십시오. GPU 커널은 정밀도가 fp16이고 입력/출력 차원이 8 또는 16(int8의 경우)으로 나눌 수 있는 경우 Tensor Core를 효율적으로 사용합니다.

cuDNN v7.6.3 이상에서는 Tensor Core를 활용하는 데 필요한 경우 컨볼루션 차원이 자동으로 채워집니다.

`fp16` 정밀도의 성능 이점을 최대화하려면 아래 모범 사례를 따르십시오.

##### 1. 최적의 fp16 커널 사용

`fp16` 활성화되면 프로그램의 GEMM(행렬 곱셈) 커널은 Tensor Core를 활용 `fp16` 그러나 어떤 경우에는 이러한 일이 발생하지 않고 프로그램이 대신 비효율적인 구현으로 대체되기 때문에 `fp16` 을 활성화하여 예상되는 속도 향상을 경험하지 못합니다.

![image](images/gpu_perf_analysis/gpu_kernels.png "TensorFlow 프로필 GPU 커널 통계 페이지")

[GPU 커널](https://www.tensorflow.org/guide/profiler#gpu_kernel_stats) 통계 페이지는 어떤 작업이 Tensor Core에 적합하고 어떤 커널이 실제로 효율적인 Tensor Core를 사용하고 있는지 보여줍니다. [딥 러닝 성능에](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html#opt-tensor-cores) 대한 NVIDIA® 가이드에는 Tensor Core를 활용하는 방법에 대한 추가 제안 사항이 포함되어 있습니다. `fp16` 사용의 이점은 이전에 메모리 바인딩된 커널에서도 나타납니다. 이제 작업에 절반의 시간이 걸리기 때문입니다.

##### 2. 동적 대 정적 손실 스케일링

`fp16` 을 사용할 때 낮은 정밀도로 인한 언더플로를 방지하기 위해 Loss scaling이 필요합니다. 손실 스케일링에는 동적 및 정적의 두 가지 유형이 있으며, 둘 다 [혼합 정밀도 가이드 에](https://www.tensorflow.org/guide/keras/mixed_precision) 자세히 설명되어 있습니다. `mixed_float16` 정책을 사용하여 Keras 옵티마이저 내에서 손실 조정을 자동으로 활성화할 수 있습니다.

참고: Keras 혼합 정밀도 API는 기본적으로 독립형 softmax 연산( `fp16` 손실 함수의 일부가 아닌 연산)을 fp16으로 평가하여 수치 문제와 수렴 불량을 유발할 수 있습니다. 최적의 성능을 위해 이러한 작업을 `fp32`

성능을 최적화하려고 할 때 동적 손실 크기 조정은 호스트에서 실행되는 추가 조건부 작업을 도입할 수 있고 추적 뷰어의 단계 간에 표시될 간격으로 이어질 수 있음을 기억하는 것이 중요합니다. 반면, 정적 손실 스케일링은 이러한 오버헤드가 없으며 올바른 정적 손실 스케일 값을 지정해야 하는 캐치로 성능 면에서 더 나은 옵션이 될 수 있습니다.

#### 2. tf.function(jit_compile=True) 또는 자동 클러스터링으로 XLA 활성화

단일 GPU로 최고의 성능을 얻기 위한 마지막 단계로 XLA를 활성화하여 작업을 통합하고 장치 활용도를 높이고 메모리 사용 공간을 낮추는 실험을 할 수 있습니다. `tf.function(jit_compile=True)` 또는 자동 클러스터링을 사용하여 프로그램에서 XLA를 활성화하는 방법에 대한 자세한 내용 [은 XLA](https://www.tensorflow.org/xla) 가이드를 참조하십시오.

전역 JIT 수준을 `-1` (해제), `1` 또는 `2` 설정할 수 있습니다. 더 높은 수준은 더 공격적이며 병렬 처리를 줄이고 더 많은 메모리를 사용할 수 있습니다. 메모리 제한이 있는 경우 값을 `1` XLA 컴파일러는 새로운 모양을 만날 때마다 커널을 계속 컴파일해야 하므로 XLA는 가변 입력 텐서 모양이 있는 모델에 대해 잘 수행되지 않습니다.

## 2. 다중 GPU 단일 호스트에서 성능 최적화

`tf.distribute.MirroredStrategy` API는 단일 호스트에서 하나의 GPU에서 여러 GPU로 모델 훈련을 확장하는 데 사용할 수 있습니다. (TensorFlow를 사용하여 분산 교육을 수행하는 방법에 대해 자세히 알아보려면 TensorFlow를 사용한 [분산 교육](https://www.tensorflow.org/guide/distributed_training) [, GPU](https://www.tensorflow.org/guide/gpu) [사용, TPU 사용](https://www.tensorflow.org/guide/tpu) 가이드 및 [Keras를 사용한 분산 교육](https://www.tensorflow.org/tutorials/distribute/keras) 자습서를 참조하세요.)

하나의 GPU에서 여러 GPU로의 전환은 기본적으로 이상적으로 확장 가능해야 하지만 때때로 성능 문제가 발생할 수 있습니다.

단일 GPU를 사용한 훈련에서 동일한 호스트의 여러 GPU로 이동할 때 이상적으로는 그래디언트 통신의 추가 오버헤드와 호스트 스레드 활용도 증가만으로 성능 확장을 경험해야 합니다. 이 오버헤드로 인해 예를 들어 1개에서 2개의 GPU로 이동하는 경우 정확한 2배 속도 향상을 얻을 수 없습니다.

아래의 추적 보기는 여러 GPU에서 훈련할 때 추가 통신 오버헤드의 예를 보여줍니다. 그라디언트를 연결하고, 복제본 간에 통신하고, 가중치 업데이트를 수행하기 전에 분할하는 데 약간의 오버헤드가 있습니다.

![image](images/gpu_perf_analysis/traceview_multi_gpu.png "단일 호스트 다중 GPU 시나리오에 대한 TensorFlow 프로필 추적 보기")

다음 체크리스트는 다중 GPU 시나리오에서 성능을 최적화할 때 더 나은 성능을 달성하는 데 도움이 됩니다.

1. 배치 크기를 최대화하여 장치 활용도를 높이고 여러 GPU에서 통신 비용을 상각합니다. [메모리 프로파일러를](https://www.tensorflow.org/guide/profiler#memory_profile_summary) 사용하면 프로그램이 최대 메모리 사용률에 얼마나 근접했는지 알 수 있습니다. 배치 크기가 클수록 수렴에 영향을 줄 수 있지만 일반적으로 성능 이점이 더 중요합니다.
2. 단일 GPU에서 여러 GPU로 이동할 때 이제 동일한 호스트에서 훨씬 더 많은 입력 데이터를 처리해야 합니다. 따라서 (1) 이후에는 입력 파이프라인 성능을 다시 확인하여 병목 현상이 없는지 확인하는 것이 좋습니다.
3. 불필요한 AllReduce 호출이 있는지 프로그램의 추적 보기에서 GPU 타임라인을 확인하십시오. 그러면 모든 장치에서 동기화됩니다. 위에 표시된 추적 보기에서 AllReduce는 [NCCL](https://developer.nvidia.com/nccl) 커널을 통해 수행되며 각 단계의 그라디언트에 대해 각 GPU에서 하나의 NCCL 호출만 있습니다.
4. 최소화할 수 있는 불필요한 D2H, H2D 및 D2D 복사 작업을 확인합니다.
5. 각 복제본이 동일한 작업을 수행하는지 단계 시간을 확인하십시오. 예를 들어, 호스트가 실수로 더 많은 작업을 수행하기 때문에 하나의 GPU(일반적으로 `GPU0`
6. 마지막으로, 순차적으로 실행되는 작업에 대한 추적 보기의 모든 GPU에 대한 교육 단계를 확인합니다. 이것은 일반적으로 프로그램에 한 GPU에서 다른 GPU로의 제어 종속성이 포함될 때 발생합니다. 과거에는 이러한 상황에서 성능을 디버깅하는 것이 사례별로 해결되었습니다. 프로그램에서 이 동작을 관찰하면 추적 보기의 이미지와 함께 [GitHub 문제를 제출하십시오.](https://github.com/tensorflow/tensorflow/issues/new/choose)

### 1. 그라디언트 AllReduce 최적화

동기식 전략으로 훈련할 때 각 장치는 입력 데이터의 일부를 받습니다.

모델을 통해 정방향 및 역방향 통과를 계산한 후 각 장치에서 계산된 기울기를 집계하고 줄여야 합니다. 이 *그래디언트 AllReduce* 는 각 장치에서 그래디언트 계산 후, 그리고 최적화 프로그램이 모델 가중치를 업데이트하기 전에 발생합니다.

각 GPU는 먼저 모델 레이어의 그라디언트를 연결하고 `tf.distribute.CrossDeviceOps` ( `tf.distribute.NcclAllReduce` 가 기본값임)를 사용하여 GPU 간에 전달한 다음 레이어별로 축소한 후 그라디언트를 반환합니다.

옵티마이저는 이러한 감소된 그라디언트를 사용하여 모델의 가중치를 업데이트합니다. 이상적으로는 오버헤드를 방지하기 위해 이 프로세스가 모든 GPU에서 동시에 발생해야 합니다.

AllReduce에 걸리는 시간은 대략 다음과 같아야 합니다.

```
(number of parameters * 4bytes)/ (communication bandwidth)
```

이 계산은 분산 교육 작업을 실행할 때의 성능이 예상대로인지 또는 추가 성능 디버깅을 수행해야 하는지를 빠르게 확인하는 데 유용합니다. `Model.summary` 에서 모델의 매개변수 수를 가져올 수 있습니다.

`fp32` (float32)를 사용하여 그래디언트를 전달하므로 각 모델 매개변수의 크기는 4바이트입니다. 당신이 한 경우에도 `fp16` 활성화 NCCL AllReduce는 사용 `fp32` 매개 변수를.

스케일링의 이점을 얻으려면 이러한 오버헤드에 비해 단계 시간이 훨씬 높아야 합니다. 이를 달성하는 한 가지 방법은 배치 크기가 단계 시간에 영향을 주지만 통신 오버헤드에는 영향을 미치지 않으므로 더 높은 배치 크기를 사용하는 것입니다.

### 2. GPU 호스트 스레드 경합

여러 GPU를 실행할 때 CPU의 역할은 장치 전체에서 GPU 커널을 효율적으로 실행하여 모든 장치를 바쁘게 유지하는 것입니다.

그러나 CPU가 하나의 GPU에서 예약할 수 있는 많은 독립적인 작업이 있는 경우 CPU는 많은 호스트 스레드를 사용하여 하나의 GPU를 사용 중인 상태로 유지한 다음 비결정적 순서로 다른 GPU에서 커널을 시작할 수 있습니다. . 이로 인해 성능에 부정적인 영향을 줄 수 있는 왜곡 또는 음의 크기 조정이 발생할 수 있습니다.

아래의 [추적 뷰어](https://www.tensorflow.org/guide/profiler#trace_viewer) `GPU1` `GPU2` 가 시작된 후 작업 실행을 시작하기 때문에 CPU가 GPU 커널을 비효율적으로 시작할 때의 오버헤드를 보여줍니다.

![image](images/gpu_perf_analysis/traceview_gpu_idle.png "비효율적인 커널 실행을 보여주는 TensorFlow 프로필 장치 추적 보기")

호스트에 대한 추적 보기는 호스트가 `GPU1` 에서 커널을 시작하기 전에 `GPU2` 커널을 시작하고 있음을 보여줍니다(아래 `tf_Compute*` 작업은 CPU 스레드를 나타내지 않음).

![image](images/gpu_perf_analysis/traceview_host_contention.png "비효율적인 커널 실행을 보여주는 TensorFlow 프로필 호스트 추적 보기")

프로그램의 추적 보기에서 GPU 커널의 이러한 종류의 비틀림이 발생하는 경우 권장되는 조치는 다음과 같습니다.

- TensorFlow 환경 변수 `TF_GPU_THREAD_MODE` 를 `gpu_private` 설정합니다. 이 환경 변수는 GPU에 대한 스레드를 비공개로 유지하도록 호스트에 지시합니다.
- 기본적으로 `TF_GPU_THREAD_MODE=gpu_private` 는 스레드 수를 2로 설정하며 대부분의 경우 충분합니다. 그러나 이 숫자는 TensorFlow 환경 변수 `TF_GPU_THREAD_COUNT` 를 원하는 스레드 수로 설정하여 변경할 수 있습니다.
