# 공동 작업 최적화

<sub>Arm ML 툴링으로 유지 관리</sub>

본 문서는 여러 기술을 조합하기 위한 실험적인 API 개요를 제공하여 배포를 위한 머신러닝 모델을 최적화합니다.

## 개요

공동 작업 최적화는 배포 시 추론 속도, 모델 크기 및 정확도와 같은 대상 특성의 최적 균형을 나타내는 모델을 생성하기 위한 다양한 기술을 망라하는 대단히 중요한 프로세스입니다.

공동 작업 아이디어는 누적된 최적화 효과를 달성하기 위해 차례로 이를 적용함으로써 개별 기술에 구축하기 위한 것입니다. 다음과 같은 최적화의 다양한 조합이 가능합니다.

- [가중치 잘라내기](https://medium.com/tensorflow/tensorflow-model-optimization-toolkit-pruning-api-42cac9157a6a)

- [가중치 클러스터링](https://blog.tensorflow.org/2020/08/tensorflow-model-optimization-toolkit-weight-clustering-api.html)

- 양자화

    - 사후 훈련 양자화
    - [양자화 인식 훈련](https://blog.tensorflow.org/2020/04/quantization-aware-training-with-tensorflow-model-optimization-toolkit.html) (QAT)

이러한 기술을 함께 연결하려고 할 때 발생하는 문제는 하나를 적용하면 일반적으로 이전 기술의 결과를 파괴하여, 모든 기술을 동시에 적용하여 얻는 전반적인 이점을 망치는 것입니다. 예를 들어, 클러스터링은 잘라내기 API에 의해 도입된 희소성을 보존하지 않습니다. 이 문제를 해결하기 위해, 다음과 같은 실험적인 공동 작업 최적화 기술을 소개합니다.

- [클러스터링을 보존하는 희소성](https://www.tensorflow.org/model_optimization/guide/combine/sparse_clustering_example)
- [양자화 인식 훈련을 보존하는 희소성](https://www.tensorflow.org/model_optimization/guide/combine/pqat_example) (PQAT)
- [양자화 인식 훈련을 보존하는 클러스터](https://www.tensorflow.org/model_optimization/guide/combine/cqat_example) (CQAT)
- [양자화 인식 훈련을 보존하는 희소성 및 클러스터](https://www.tensorflow.org/model_optimization/guide/combine/pcqat_example)

이는 머신러닝 모델을 압축하고 추론 시간에 하드웨어 가속화를 이용하는 데 사용될 수 있는 여러 배포 경로를 제공합니다. 아래 도표는 리프 노드가 부분적으로 또는 완전히 양자화되고 tflite 형식임을 의미하는 배포 준비 모델인 바람직한 배포 특성을 가진 모델을 검색하여 탐색될 수 있는 여러 배포 경로를 보여줍니다. 녹색으로 채워진 부분은 재훈련/세밀한 조정이 필요한 단계를 나타내며 빨간색 테두리는 공동 작업 최적화 단계를 강조합니다. 특정 노드에서 모델을 확보하기 위해 사용된 기술은 해당하는 라벨 내에 표시됩니다.

![공동 작업 최적화](images/collaborative_optimization.png "collaborative optimization")

위의 그림에서는 직접 양자화 전용(사후 훈련 또는 QAT) 배포 경로가 생략되어 있습니다.

이 아이디어는 위의 배포 트리의 세 번째 수준에서 완전히 최적화된 모델에 도달하기 위한 것입니다. 하지만, 다른 모든 수준의 최적화는 만족스러운 것으로 입증될 수 있으며 필요한 추론 지연/정확성 절충을 달성할 수 있으며, 이 경우 추가적인 최적화가 필요하지 않습니다. 권장되는 훈련 프로세스는 대상 배포 시나리오에 적용되는 배포 트리 수준을 반복적으로 살펴보고 모델이 추론 지연 시간 요구 사항을 충족하는지 확인하고, 그렇지 않은 경우 필요하다면 해당 공동 작업 최적화 기술을 사용하여 모델을 추가로 압축하고 모델이 완전히 최적화(잘라내기, 클러스터링 및 양자화)가 될 때까지 반복하는 것입니다.

아래 그림은 공동 작업 최적화 파이프라인을 거치는 샘플 가중치 커널의 밀도 플롯을 보여줍니다.

![공동 작업 최적화 밀도 플롯](images/collaborative_optimization_dist.png "collaborative optimization density plot")

이 결과는 훈련 시 대상 희소성에 따라 고윳값의 수가 감소하고 희소 가중치의 수가 상당한 양자화된 배포 모델입니다. 상당한 모델 압축 이점 외에 특정 하드웨어 지원은 이러한 희소 클러스터 모델을 사용하여 추론 대기 시간을 상당히 줄일 수 있습니다.

## 결과

아래는 PQAT 및 CQAT 공동 작업 최적화 경로로 실험 시 확보한 몇몇 정확성 및 압축 결과입니다.

### 양자화 인식 훈련을 보존하는-희소성(PQAT)

<figure>
<table class="tableizer-table">
<tr class="tableizer-firstrow"><th>Model</th><th>Items</th><th>Baseline</th><th>Pruned Model (50% sparsity)</th><th>QAT Model</th><th>PQAT Model</th></tr>
 <tr><td>DS-CNN-L</td><td>FP32 Top1 Accuracy</td><td><b>95.23%</b></td><td>94.80%</td><td>(Fake INT8) 94.721%</td><td>(Fake INT8) 94.128%</td></tr>
 <tr><td> </td><td>INT8 full integer quantization</td><td>94.48%</td><td><b>93.80%</b></td><td>94.72%</td><td><b>94.13%</b></td></tr>
 <tr><td> </td><td>Compression</td><td>528,128 → 434,879 (17.66%)</td><td>528,128 → 334,154 (36.73%)</td><td>512,224 → 403,261 (21.27%)</td><td>512,032 → 303,997 (40.63%)</td></tr>
 <tr><td>Mobilenet_v1-224</td><td>FP32 Top 1 Accuracy</td><td><b>70.99%</b></td><td>70.11%</td><td>(Fake INT8) 70.67%</td><td>(Fake INT8) 70.29%</td></tr>
 <tr><td> </td><td>INT8 full integer quantization</td><td>69.37%</td><td><b>67.82%</b></td><td>70.67%</td><td><b>70.29%</b></td></tr>
 <tr><td> </td><td>Compression</td><td>4,665,520 → 3,880,331 (16.83%)</td><td>4,665,520 → 2,939,734 (37.00%)</td><td>4,569,416 → 3,808,781 (16.65%)</td><td>4,569,416 → 2,869,600 (37.20%)</td></tr>
</table>
</figure>

### 양자화 인식 훈련을 보존하는-클러스터(CQAT)

<figure>
<table class="tableizer-table">
<tr class="tableizer-firstrow"><th>Model</th><th>Items</th><th>Baseline</th><th>Clustered Model</th><th>QAT Model</th><th>CQAT Model</th></tr>
 <tr><td>Mobilenet_v1 on CIFAR-10</td><td>FP32 Top1 Accuracy</td><td><b>94.88%</b></td><td>94.48%</td><td>(Fake INT8) 94.80%</td><td>(Fake INT8) 94.60%</td></tr>
 <tr><td> </td><td>INT8 full integer quantization</td><td>94.65%</td><td><b>94.41%</b></td><td>94.77%</td><td><b>94.52%</b></td></tr>
 <tr><td> </td><td>Size</td><td>3.00 MB</td><td>2.00 MB</td><td>2.84 MB</td><td>1.94 MB</td></tr>
 <tr><td>Mobilenet_v1 on ImageNet</td><td>FP32 Top 1 Accuracy</td><td><b>71.07%</b></td><td>65.30%</td><td>(Fake INT8) 70.39%</td><td>(Fake INT8) 65.35%</td></tr>
 <tr><td> </td><td>INT8 full integer quantization</td><td>69.34%</td><td><b>60.60%</b></td><td>70.35%</td><td><b>65.42%</b></td></tr>
 <tr><td> </td><td>Compression</td><td>4,665,568 → 3,886,277 (16.7%)</td><td>4,665,568 → 3,035,752 (34.9%)</td><td>4,569,416 → 3,804,871 (16.7%)</td><td>4,569,472 → 2,912,655 (36.25%)</td></tr>
</table>
</figure>

### 채널마다 클러스트된 모델에 대한 CQAT 및 PCQAT 결과

아래 결과는 [채널마다 클러스팅하는](https://www.tensorflow.org/model_optimization/guide/clustering) 기술로 확보되었습니다. 모델의 컨볼루셔널 레이어가 채널마다 클러스트 된다면 모델 정확성이 더 높다는 것을 보여줍니다. 모델에 컨볼루셔널 레이어가 많다면, 채널마다 클러스트 하는 것이 좋습니다. 압축률은 동일하게 유지되지만, 모델 정확성이 높아질 것입니다. 모델 최적화 파이프라인은 실험의 '클러스터 됨 -&gt; QAT를 보존하는 클러스터 -&gt;사후 훈련 양자화, int8' 입니다.

<figure>
<table  class="tableizer-table">
<tr class="tableizer-firstrow"><th>Model</th><th>Clustered -> CQAT, int8 quantized</th><th>Clustered per channel -> CQAT, int8 quantized</th>
 <tr><td>DS-CNN-L</td><td>95.949%</td><td> 96.44%</td></tr>
 <tr><td>MobileNet-V2</td><td>71.538%</td><td>72.638%</td></tr>
 <tr><td>MobileNet-V2 (pruned)</td><td>71.45%</td><td>71.901%</td></tr>
</table>
</figure>

## 예시

여기에서 설명된 공동 작업 최적화 기술의 엔드 투 엔드 예시의 경우, [CQAT](https://www.tensorflow.org/model_optimization/guide/combine/cqat_example), [PQAT](https://www.tensorflow.org/model_optimization/guide/combine/pqat_example), [클러스터링을 보존하는-희소성](https://www.tensorflow.org/model_optimization/guide/combine/sparse_clustering_example) 및 [PCQAT](https://www.tensorflow.org/model_optimization/guide/combine/pcqat_example) 예시 노트북을 참고하시기 바랍니다.
