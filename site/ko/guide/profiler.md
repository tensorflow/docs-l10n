# Profiler를 사용한 TensorFlow 성능 최적화

[TOC]

Use the tools available with the Profiler to track the performance of your TensorFlow models. See how your model performs on the host (CPU), the device (GPU), or on a combination of both the host and device(s).

프로파일링을 통해 모델에서 다양한 TensorFlow 연산(ops)이 사용하는 하드웨어 리소스(시간 및 메모리)를 이해하고 성능 병목 현상을 해결함으로써 궁극적으로 모델의 실행 속도를 높일 수 있습니다.

This guide will walk you through how to install the Profiler, the various tools available, the different modes of how the Profiler collects performance data, and some recommended best practices to optimize model performance.

Cloud TPU의 모델 성능을 프로파일링하려면 [Cloud TPU 가이드](https://cloud.google.com/tpu/docs/cloud-tpu-tools#capture_profile)를 참조하세요.

## Profiler 및 GPU 필수 구성 요소 설치

[GitHub 리포지토리](https://raw.githubusercontent.com/tensorflow/profiler/master/install_and_run.py)에서 <a><code>install_and_run.py</code></a> 스크립트를 다운로드한 후 실행하여 Profiler를 설치하세요.

GPU를 프로파일링하려면 다음을 수행해야 합니다.

1. Meet the NVIDIA® GPU drivers and CUDA® Toolkit requirements listed on [TensorFlow GPU support software requirements](https://www.tensorflow.org/install/gpu#linux_setup).

2. Ensure CUPTI exists on the path:

    ```shell
    /sbin/ldconfig -N -v $(sed 's/:/ /g' <<< $LD_LIBRARY_PATH) | \
    grep libcupti
    ```

If you don't have CUPTI on the path, prepend its installation directory to the `$LD_LIBRARY_PATH` environment variable by running:

```shell
export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
```

위의 `ldconfig` 명령을 다시 실행하여 CUPTI 라이브러리가 있는지 확인하세요.

### 권한 문제 해결하기

When you run profiling with CUDA® Toolkit in a Docker environment or on Linux, you may encounter issues related to insufficient CUPTI privileges (`CUPTI_ERROR_INSUFFICIENT_PRIVILEGES`). See the [NVIDIA Developer Docs](https://developer.nvidia.com/nvidia-development-tools-solutions-ERR_NVGPUCTRPERM-permission-issue-performance-counters){:.external} to learn more about how you can resolve these issues on Linux.

Docker 환경에서 CUPTI 권한 문제를 해결하려면 다음을 실행합니다.

```shell
docker run option '--privileged=true'
```

<a name="profiler_tools"></a>

## Profiler 도구

일부 모델 데이터를 캡처한 후에만 표시되는 TensorBoard의 **Profile** 탭에서 Profiler에 액세스합니다.

참고: Profiler에서 [Google 차트 라이브러리](https://developers.google.com/chart/interactive/docs/basic_load_libs#basic-library-loading)를 로드하려면 인터넷에 연결되어 있어야 합니다. 로컬 컴퓨터, 회사 방화벽 뒤 또는 데이터 센터에서 TensorBoard를 완전히 오프라인으로 실행하면 일부 차트와 표가 누락될 수 있습니다.

Profiler에는 성능 분석에 도움이 되는 다양한 도구가 있습니다.

- 개요 페이지
- 입력 파이프라인 분석기
- TensorFlow 통계
- 추적 뷰어
- GPU 커널 통계
- Memory profile tool
- Pod viewer

<a name="overview_page"></a>

### 개요 페이지

개요 페이지는 프로파일 실행 중에 수행된 모델에 대한 최상위 레벨 보기를 제공합니다. 이 페이지에는 호스트 및 모든 기기에 대한 종합적인 개요 페이지와 모델 훈련 성능을 향상하기 위한 몇 가지 권장 사항이 표시됩니다. 호스트 드롭다운 메뉴에서 개별 호스트를 선택할 수도 있습니다.

개요 페이지에는 다음과 같은 데이터가 표시됩니다.

![image](./images/tf_profiler/overview_page.png)

- **성능 요약 -** 모델 성능에 대한 높은 수준의 요약을 표시합니다. 성능 요약은 두 부분으로 구성됩니다.

    1. 단계 시간 분석 - 평균 단계 시간을 시간이 사용된 여러 범주로 나눕니다.

        - 컴파일 - 커널 컴파일에 소요된 시간
        - 입력 - 입력 데이터를 읽는 데 소요된 시간
        - 출력 - 출력 데이터를 읽는 데 소요된 시간
        - 커널 시작 - 호스트가 커널을 시작하는 데 소요된 시간
        - 호스트 컴퓨팅 시간
        - 기기 간 통신 시간
        - 기기 내 컴퓨팅 시간
        - Python 오버헤드를 포함한 기타

    2. 기기 컴퓨팅 정밀도 - 16 및 32bit 계산을 사용하는 기기 컴퓨팅 시간의 백분율을 보고합니다.

- **단계 시간 그래프 - ** 샘플링한 모든 단계에서 기기 단계 시간(밀리 초)의 그래프를 표시합니다. 각 단계는 시간이 사용된 여러 범주(서로 다른 색상)로 나뉩니다. 빨간색 영역은 기기가 호스트로부터 입력 데이터를 기다리는 동안 유휴 상태인 단계 시간의 일부에 해당합니다. 녹색 영역은 기기가 실제로 작동한 시간을 나타냅니다.

- **기기 내 상위 10개의 TensorFlow 연산 -** 가장 오래 실행된 기기 연산을 표시합니다.

    각 행에는 연산의 자체 시간(모든 연산에 소요된 시간의 백분율), 누적 시간, 범주 및 이름이 표시됩니다.

- **실행 환경 -** 다음을 포함하여 모델 실행 환경에 대한 높은 수준의 요약을 표시합니다.

    - 사용된 호스트 수
    - 기기 유형(GPU/TPU)
    - 기기 코어 수

- **Recommendation for next steps -** Reports when a model is input bound and recommends tools you can use to locate and resolve model performance bottlenecks

<a name="input_pipeline_analyzer"></a>

### 입력 파이프라인 분석기

TensorFlow 프로그램이 파일에서 데이터를 읽을 때 파이프라인 방식으로 TensorFlow 그래프의 맨 위에서 시작합니다. 읽기 프로세스는 직렬로 연결된 다수의 데이터 처리 단계로 나누어지며, 한 단계의 출력은 다음 단계의 입력이 됩니다. 이러한 데이터 읽기 방식을 *입력 파이프라인*이라고 합니다.

파일에서 레코드를 읽는 일반적인 파이프라인에는 다음 단계가 있습니다.

1. 파일 읽기
2. 파일 전처리(선택 사항)
3. 호스트에서 기기로 파일 전송

비효율적인 입력 파이프라인으로 인해 애플리케이션 속도가 크게 느려질 수 있습니다. 애플리케이션이 입력 파이프라인에서 상당한 시간을 소비하면 **입력 바운드**로 간주합니다. 입력 파이프라인 분석기에서 얻은 통찰력을 사용하여 입력 파이프라인이 비효율적인 위치를 파악하세요.

입력 파이프라인 분석기는 프로그램의 입력 바운드 여부를 즉시 알려주고 입력 파이프라인의 모든 단계에서 성능 병목 현상을 디버깅하기 위해 기기 및 호스트 쪽 분석을 안내합니다.

데이터 입력 파이프라인을 최적화하기 위한 권장 모범 사례는 입력 파이프라인 성능 지침을 참조하세요.

#### 입력 파이프라인 대시보드

입력 파이프라인 분석기를 열려면 **프로파일**을 선택한 다음 **도구** 드롭다운 메뉴에서 **input_pipeline_analyzer**를 선택하세요.

![image](./images/tf_profiler/overview_page.png)

대시보드에는 세 개의 섹션이 있습니다.

1. **요약 -** 애플리케이션이 입력 바운드인지 여부와 정보의 양에 따라 전체 입력 파이프라인을 요약하여 보여줍니다.
2. **기기 쪽 분석 -** 기기 단계 시간 및 각 단계에서 코어를 통해 입력 데이터를 기다리는 데 소요된 기기 시간 범위를 포함하여 자세한 기기 쪽 분석 결과를 표시합니다.
3. **호스트 쪽 분석 -** 호스트의 입력 처리 시간 분석을 포함하여 호스트 쪽 분석을 자세히 보여줍니다.

#### 입력 파이프라인 요약

요약에서는 호스트로부터 입력을 기다리는 데 소요된 기기 시간의 백분율을 보여줌으로써 프로그램이 입력 바운드인지 보고합니다. 계측된 표준 입력 파이프라인을 사용하는 경우에는 도구에서 대부분의 입력 처리 시간이 소비된 위치를 보고합니다.

#### 기기 쪽 분석

기기 쪽 분석은 기기와 호스트에 소요된 시간 및 호스트로부터 입력 데이터를 기다리는 데 소요된 기기 시간에 대한 통찰력을 제공합니다.

1. **단계 번호에 대한 단계 시간 -** 샘플링한 모든 단계에서 기기 단계 시간(밀리 초) 그래프를 표시합니다. 각 단계는 시간이 소비된 여러 범주(서로 다른 색상)로 나뉩니다. 빨간색 영역은 기기가 호스트로부터 입력 데이터를 기다리는 유휴 상태의 일부 단계 시간에 해당합니다. 녹색 영역은 기기가 실제로 작동한 시간을 나타냅니다.
2. **단계 시간 통계 -** 기기 단계 시간의 평균, 표준 편차 및 범위([최소, 최대])를 보고합니다.

#### 호스트 쪽 분석

호스트 쪽 분석은 호스트의 입력 처리 시간(`tf.data` API 연산에 소요된 시간)을 여러 범주로 분류하여 보고합니다.

- **요청 시 파일에서 데이터 읽기 -** 캐싱, 프리페치 및 인터리빙 없이 파일에서 데이터를 읽는 데 소요된 시간입니다.
- **파일에서 미리 데이터 읽기 -** 캐싱, 프리페치 및 인터리빙을 포함하여 파일을 읽는 데 소요된 시간
- **데이터 전처리 -** 이미지 압축 풀기와 같은 사전 처리 연산에 소요된 시간
- **기기로 전송될 데이터 큐에 넣기 -** 데이터를 기기로 전송하기 전에 데이터를 인피드 큐에 넣는 데 소요된 시간

**입력 Op 통계**를 확장하여 개별 입력 연산 및 해당 범주에 대한 통계를 실행 시간별로 분류하여 볼 수 있습니다.

![image](./images/tf_profiler/input_pipeline_analyzer.png)

소스 데이터 표에는 다음 정보가 포함하여 각 항목이 표시됩니다.

1. **입력 Op -** 입력 op의 TensorFlow op 이름을 표시합니다.
2. **Count -** 프로파일링 기간 동안 작업 실행의 총 인스턴스 수를 표시합니다.
3. **총 시간(밀리 초) -** 각 인스턴스에 소요된 시간의 누적 합계를 보여줍니다.
4. **총 시간 % -** 입력 처리에 소요된 총 시간의 일부로 작업에 소요된 총 시간을 표시합니다.
5. **총 자체 시간(밀리 초) -** 각 인스턴스에 소요된 자체 시간의 누적 합계를 표시합니다. 여기에서 자체 시간은 호출하는 함수에서 소비한 시간을 제외하고 함수 본문 내에서 소비된 시간을 측정합니다.
6. **총 자체 시간 %**. 총 자체 시간을 입력 처리에 소요된 총 시간의 일부로 표시합니다.
7. **범주**. 입력 op의 처리 범주를 표시합니다.

<a name="tf_stats"></a>

### TensorFlow 통계

TensorFlow 통계 도구는 프로파일링 세션 동안 호스트 또는 기기에서 실행되는 모든 TensorFlow 연산(op)의 성능을 표시합니다.

![image](./images/tf_profiler/input_op_stats.png)

이 도구는 성능 정보를 두 개의 창에서 표시합니다.

- 상단 창에는 최대 4개의 원형 차트가 표시됩니다.

    1. 호스트에서 각 op의 자체 실행 시간 분포
    2. 호스트에서 각 op 유형의 자체 실행 시간 분포
    3. 기기에서 각 op의 자체 실행 시간 분포
    4. 기기에서 각 op 유형의 자체 실행 시간 분포

- 하단 창에 표시되는 표에는 TensorFlow 연산에 대한 데이터를 보고합니다. 행에는 연산별로, 열에는 데이터 유형별로 표시됩니다(열의 제목을 클릭하여 정렬). 상단 창의 오른쪽에 있는 CSV로 내보내기 버튼을 클릭하여 테이블의 데이터를 CSV 파일로 내보낼 수 있습니다.

    참고:

    - 어떤 연산이 하위 연산을 포함하는 경우:

        - 작업의 총 "누적" 시간에는 하위 작업 내부에서 보낸 시간이 포함됩니다.
        - 작업의 총 "자체" 시간에는 하위 작업 내부에서 보낸 시간이 포함되지 않습니다.

    - 호스트에서 op가 실행되는 경우:

        - The percentage of the total self-time on device incurred by the op on will be 0
        - 이 op를 포함하여 기기의 총 자체 시간 누적 백분율은 0입니다.

    - 기기에서 op가 실행되는 경우:

        - 이 op로 발생한 호스트의 총 자체 시간 백분율은 0입니다.
        - 이 op를 포함하여 호스트의 총 자체 시간 누적 백분율은 0입니다.

파이 차트 및 테이블에서 유휴 시간을 포함하거나 제외하도록 선택할 수 있습니다.

<a name="trace_viewer"></a>

### 추적 뷰어

추적 뷰어의 타임라인을 통해 다음을 알 수 있습니다.

- TensorFlow 모델에 의해 실행된 연산의 기간
- 시스템의 어느 부분(호스트 또는 기기)에서 op를 실행했는지 알 수 있습니다. 일반적으로 호스트는 입력 연산을 실행하고 학습 데이터를 사전 처리하여 기기로 전송하는 반면, 기기는 실제 모델 학습을 실행합니다.

추적 뷰어를 사용하면 모델의 성능 문제를 식별한 다음 해결을 위한 단계를 수행할 수 있습니다. 예를 들어, 입력 또는 모델 훈련에 많은 시간이 소요되는지 높은 수준에서 확인할 수 있습니다. 드릴다운하면 실행하는 데 가장 오래 걸리는 ops를 식별할 수 있습니다. 추적 뷰어는 기기당 백만 개의 이벤트로 제한됩니다.

#### 추적 뷰어 인터페이스

추적 뷰어를 열면 가장 최근에 실행된 내용이 표시됩니다.

![image](./images/tf_profiler/tf_stats.png)

이 화면에는 다음과 같은 주요 요소가 포함되어 있습니다.

1. **Timeline pane -** Shows ops that the device and the host executed over time
2. **세부 정보 창 -** 타임라인 창에서 선택한 ops에 대해 추가 정보를 표시합니다.

타임라인 창에는 다음 요소가 포함되어 있습니다.

1. **상단 바 -** 다양한 보조 컨트롤이 포함되어 있습니다.
2. **시간 축 -** 추적 시작과 관련하여 시간을 표시합니다.
3. **섹션 및 트랙 레이블 -** 각 섹션에는 여러 트랙이 있으며 왼쪽에 있는 삼각형을 클릭하여 섹션을 확장하거나 축소할 수 있습니다. 시스템의 모든 처리 요소마다 하나의 섹션이 있습니다.
4. **도구 선택기 -** 줌, 팬, 선택 및 타이밍과 같은 추적 뷰어를 조작하기 위한 다양한 도구가 포함되어 있습니다. 타이밍 도구를 사용하여 시간 간격을 표시하세요.
5. **이벤트 -** 여기에는 op가 실행된 기간 또는 훈련 단계와 같은 메타 이벤트의 기간이 표시됩니다.

##### 섹션과 트랙

추적 뷰어에는 다음 섹션이 포함되어 있습니다.

- **기기 노드별 섹션 하나**, 기기 칩 번호와 칩 내 기기 노드로 표시됩니다(예를 들어, `/device:GPU:0 (pid 0)`). 각 기기 노드 섹션에는 다음 트랙이 포함되어 있습니다.
    - **Step -** 기기에서 실행 중인 학습 단계의 기간을 표시합니다.
    - **TensorFlow Ops -**. Shows the ops executed on the device
    - **XLA 연산 -** XLA 컴파일러가 사용된 경우에는 기기에서 실행된 [XLA](https://www.tensorflow.org/xla/) 연산(ops)을 표시합니다. (각 TensorFlow op는 하나 또는 여러 개의 XLA ops로 변환됩니다. XLA 컴파일러는 XLA ops를 기기에서 실행되는 코드로 변환합니다.)
- **호스트 머신의 CPU에서 실행되는 스레드에 대한 섹션 하나,** **"Host Threads"**로 표시됩니다. 이 섹션에는 CPU 스레드마다 하나의 트랙이 있습니다. 섹션 레이블과 함께 표시되는 정보는 무시해도 됩니다.

##### 이벤트

타임라인 내의 이벤트는 다른 색상으로 표시됩니다. 색상 자체는 특별한 의미가 없습니다.

The trace viewer can also display traces of Python function calls in your TensorFlow program. If you use the `tf.profiler.experimental.start()` API, you can enable Python tracing by using the `ProfilerOptions` namedtuple when starting profiling. Alternatively, if you use the sampling mode for profiling, you can select the level of tracing by using the dropdown options in the **Capture Profile** dialog.

![image](./images/tf_profiler/python_tracer.png)

<a name="gpu_kernel_stats"></a>

### GPU 커널 통계

이 도구는 모든 GPU 가속 커널에 대한 성능 통계 및 원래 op를 보여줍니다.

![image](./images/tf_profiler/gpu_kernel_stats.png)

이 도구는 두 개의 창에서 정보를 표시합니다.

- 상단 창에는 총 시간이 가장 높은 CUDA 커널을 보여주는 파이 차트가 표시됩니다.

- 하단 창에 표시되는 표에서는 각 고유 kernel-op 쌍에 대한 다음 데이터를 보여줍니다.

    - kernel-op 쌍별로 그룹화된 총 경과 GPU 기간의 내림차순 순위
    - 시작된 커널의 이름
    - 커널이 사용하는 GPU 레지스터 수
    - The total size of shared (static + dynamic shared) memory used in bytes
    - `blockDim.x, blockDim.y, blockDim.z`로 표현된 블록 차원
    - `gridDim.x, gridDim.y, gridDim.z`로 표현된 그리드 차원
    - op가 TensorCore를 사용할 수 있는지 여부
    - 커널에 TensorCore 명령어가 포함되어 있는지 여부
    - 이 커널을 시작한 op의 이름
    - 이 kernel-op 쌍의 발생 횟수
    - 총 경과된 GPU 시간(마이크로 초)
    - 평균 경과된 GPU 시간(마이크로 초)
    - 최소 경과된 GPU 시간(마이크로 초)
    - 최대 경과된 GPU 시간(마이크로 초)

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

![image](./images/tf_profiler/trace_viewer.png)

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

## 성능 데이터 수집

TensorFlow 프로파일러는 TensorFlow 모델의 호스트 활동 및 GPU 추적을 수집합니다. 프로그래밍 모드 또는 샘플링 모드를 통해 성능 데이터를 수집하도록 프로파일러를 구성할 수 있습니다.

### 프로파일링 API

You can use the following APIs to perform profiling.

- TensorBoard Keras 콜백(`tf.keras.callbacks.TensorBoard`)을 사용하는 프로그래밍 모드

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

- `tf.profiler` 함수 API를 사용하는 프로그래밍 모드

    ```python
    tf.profiler.experimental.start('logdir')
    # Train the model here
    tf.profiler.experimental.stop()
    ```

- 컨텍스트 관리자를 사용하는 프로그래밍 모드

    ```python
    with tf.profiler.experimental.Profile('logdir'):
        # Train the model here
        pass
    ```

참고: 프로파일러를 너무 오래 실행하면 메모리가 부족해질 수 있습니다. 한 번에 10단계를 초과하지 않는 것이 좋습니다. 초기화 오버헤드로 인한 부정확성을 피하려면 처음 몇 개의 배치를 프로파일링하지 마세요.

<a name="sampling_mode"></a>

- Sampling mode - Perform on-demand profiling by using `tf.profiler.experimental.server.start()` to start a gRPC server with your TensorFlow model run. After starting the gRPC server and running your model, you can capture a profile through the **Capture Profile** button in the TensorBoard profile plugin. Use the script in the Install profiler section above to launch a TensorBoard instance if it is not already running.

    예를 들면,

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

**Capture Profile** 대화 상자를 사용하여 다음을 지정합니다.

- A comma delimited list of profile service URLs or TPU name.
- A profiling duration.
- The level of device, host, and Python function call tracing.
- How many times you want the Profiler to retry capturing profiles if unsuccessful at first.

### 사용자 정의 훈련 루프 프로파일링

To profile custom training loops in your TensorFlow code, instrument the training loop with the `tf.profiler.experimental.Trace` API to mark the step boundaries for the Profiler. The `name` argument is used as a prefix for the step names, the `step_num` keyword argument is appended in the step names, and the `_r` keyword argument makes this trace event get processed as a step event by the Profiler.

As an example,

```python
for step in range(NUM_STEPS):
    with tf.profiler.experimental.Trace('train', step_num=step, _r=1):
        train_data = next(dataset)
        train_step(train_data)
```

This will enable the Profiler's step-based performance analysis and cause the step events to show up in the trace viewer.

입력 파이프라인의 정확한 분석을 위해 `tf.profiler.experimental.Trace` 컨텍스트 내에 데이터세트 반복기를 포함합니다.

아래 코드 조각은 안티 패턴입니다.

경고: 이로 인해 입력 파이프라인이 부정확하게 분석됩니다.

```python
for step, train_data in enumerate(dataset):
    with tf.profiler.experimental.Trace('train', step_num=step, _r=1):
        train_step(train_data)
```

### 프로파일링 사용 사례

Profiler는 4가지 축을 따라 여러 가지 사용 사례를 다룹니다. 일부 조합은 현재 지원되며 다른 조합은 향후에 추가될 예정입니다. 사용 사례 중 일부는 다음과 같습니다.

- 로컬 및 원격 프로파일링: 프로파일링 환경을 설정하는 일반적인 두 가지 방법입니다. 로컬 프로파일링에서 프로파일링 API는 모델이 실행 중인 같은 머신(예: GPU가 있는 로컬 워크스테이션)에서 호출됩니다. 원격 프로파일링에서 프로파일링 API는 모델이 실행 중인 다른 머신(예: Cloud TPU)에서 호출됩니다.
- Profiling multiple workers: You can profile multiple machines when using the distributed training capabilities of TensorFlow.
- 하드웨어 플랫폼: CPU, GPU 및 TPU를 프로파일링합니다.

The table below is a quick overview of which of the above use cases are supported by the various profiling APIs in TensorFlow:

<a name="profiling_api_table"></a>

| Profiling API                | Local     | Remote    | Multiple  | Hardware  | :                              :           :           : workers   : Platforms : | :--------------------------- | :-------- | :-------- | :-------- | :-------- | | **TensorBoard Keras          | Supported | Not       | Not       | CPU, GPU  | : Callback**                   :           : Supported : Supported :           : | **`tf.profiler.experimental` | Supported | Not       | Not       | CPU, GPU  | : start/stop [API](https://www.tensorflow.org/api_docs/python/tf/profiler/experimental#functions_2)**    :           : Supported : Supported :           : | **`tf.profiler.experimental` | Supported | Supported | Supported | CPU, GPU, | : client.trace [API](https://www.tensorflow.org/api_docs/python/tf/profiler/experimental/client/trace)**  :           :           :           : TPU       : | **Context manager API**      | Supported | Not       | Not       | CPU, GPU  | :                              :           : supported : Supported :           :

<a name="performance_best_practices"></a>

## 최적의 모델 성능을 위한 모범 사례

최적의 성능을 얻으려면 TensorFlow 모델에 적용 가능한 다음 권장 사항을 사용하세요.

일반적으로 기기에서 모든 변환을 수행하고 플랫폼에 맞는 최신 호환 버전의 라이브러리(cuDNN, Intel MKL 등)를 사용해야 합니다.

### 입력 데이터 파이프라인의 최적화

An efficient data input pipeline can drastically improve the speed of your model execution by reducing device idle time. Consider incorporating the following best practices as detailed [here](https://www.tensorflow.org/guide/data_performance) to make your data input pipeline more efficient:

- 데이터 프리페치
- 데이터 추출 병렬화
- 데이터 변환 병렬화
- 메모리에 데이터 캐시
- 사용자 정의 함수 벡터화
- 변환 적용 시 메모리 사용량 축소

Additionally, try running your model with synthetic data to check if the input pipeline is a performance bottleneck.

### 기기 성능 향상하기

- Increase training mini-batch size (number of training samples used per device in one iteration of the training loop)
- TF 통계를 사용하여 기기 연산이 얼마나 효율적으로 실행되는지 확인합니다.
- Use `tf.function` to perform computations and optionally, enable the `experimental_compile` flag
- Minimize host Python operations between steps and reduce callbacks. Calculate metrics every few steps instead of at every step
- 기기 컴퓨팅 단위의 사용률을 높게 유지합니다.
- 여러 기기에 병렬로 데이터를 전송합니다.
- 채널을 우선적으로 선호하도록 데이터 레이아웃을 최적화합니다(예: NHWC보다 NCHW). NVIDIA® V100과 같은 특정 GPU는 NHWC 데이터 레이아웃에서 성능이 더 우수합니다.
- IEEE에서 지정한 반정밀도 부동 소수점 형식인 `fp16` 또는 Brain 부동 소수점 [bfloat16](https://cloud.google.com/tpu/docs/bfloat16) 형식과 같은 16-bit 숫자 표현의 사용을 고려해보세요.
- [Keras 혼합 정밀도 API](https://www.tensorflow.org/guide/keras/mixed_precision)의 사용을 고려해보세요.
- When training on GPUs, make use of the TensorCore. GPU kernels use the TensorCore when the precision is fp16 and input/output dimensions are divisible by 8 or 16 (for int8)

## 추가 자료

- 이 가이드의 조언을 구현하려면 엔드 투 엔드 [TensorBoard 프로파일러 튜토리얼](https://www.tensorflow.org/tensorboard/tensorboard_profiling_keras)을 참조하세요.
- Watch the [Performance profiling in TF 2](https://www.youtube.com/watch?v=pXHAQIhhMhI) talk from the TensorFlow Dev Summit 2020.

## 알려진 제한 사항

### Profiling multiple GPUs on TensorFlow 2.2 and TensorFlow 2.3

TensorFlow 2.2 and 2.3 support multiple GPU profiling for single host systems only; multiple GPU profiling for multi-host systems is not supported. To profile multi-worker GPU configurations, each worker has to be profiled independently. On TensorFlow 2.4, multiple workers can be profiled using the [`tf.profiler.experimental.trace`](https://www.tensorflow.org/api_docs/python/tf/profiler/experimental/client/trace) API.

CUDA® Toolkit 10.2 or later is required to profile multiple GPUs. As TensorFlow 2.2 and 2.3 support CUDA® Toolkit versions only up to 10.1 , create symbolic links to `libcudart.so.10.1` and `libcupti.so.10.1`.

```shell
sudo ln -s /usr/local/cuda/lib64/libcudart.so.10.2 /usr/local/cuda/lib64/libcudart.so.10.1
sudo ln -s /usr/local/cuda/extras/CUPTI/lib64/libcupti.so.10.2 /usr/local/cuda/extras/CUPTI/lib64/libcupti.so.10.1
```
