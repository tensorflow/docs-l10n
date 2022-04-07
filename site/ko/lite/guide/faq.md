# 자주 묻는 질문

여기에서 질문에 대한 답을 찾지 못한 경우, 해당 주제에 대한 자세한 설명서를 찾아보거나 [GitHub 문제](https://github.com/tensorflow/tensorflow/issues)를 제출하세요.

## 모델 변환

#### TensorFlow에서 TensorFlow Lite로의 변환이 지원되는 형식은 무엇입니까?

freeze_graph.py에 의해 생성된 고정 [GraphDefs](../convert/index.md#python_api): <a>TFLiteConverter.from_frozen_graph</a>

#### TensorFlow Lite에서 일부 연산이 구현되지 않은 이유는 무엇입니까?

In order to keep TFLite lightweight, only certain TF operators (listed in the [allowlist](op_select_allowlist.md)) are supported in TFLite.

#### 내 모델이 변환되지 않는 이유는 무엇입니까?

Since the number of TensorFlow Lite operations is smaller than TensorFlow's, some models may not be able to convert. Some common errors are listed [here](../convert/index.md#conversion-errors).

누락된 연산 또는 제어 흐름 연산과 관련이 없는 변환 문제의 경우, [GitHub 문제](https://github.com/tensorflow/tensorflow/issues?q=label%3Acomp%3Alite+)를 검색하거나 [새 문제](https://github.com/tensorflow/tensorflow/issues)를 제출하세요.

#### TensorFlow Lite 모델이 원래 TensorFlow 모델과 동일하게 동작하는지 어떻게 테스트합니까?

The best way to test is to compare the outputs of the TensorFlow and the TensorFlow Lite models for the same inputs (test data or random inputs) as shown [here](inference.md#load-and-run-a-model-in-python).

#### GraphDef 프로토콜 버퍼의 입력/출력을 어떻게 결정합니까?

`.pb` 파일에서 그래프를 검사하는 가장 쉬운 방법은 머신러닝 모델용 오픈 소스 뷰어인 [Netron](https://github.com/lutzroeder/netron)을 사용하는 것입니다.

Netron이 그래프를 열 수 없는 경우, [summarize_graph](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/graph_transforms/README.md#inspecting-graphs) 도구를 시도해 볼 수 있습니다.

summary_graph 도구에서 오류가 발생하면 [TensorBoard](https://www.tensorflow.org/guide/summaries_and_tensorboard)로 GraphDef를 시각화하고 그래프에서 입력과 출력을 찾을 수 있습니다. `.pb` 파일을 시각화하려면 아래와 같이 [`import_pb_to_tensorboard.py`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/import_pb_to_tensorboard.py) 스크립트를 사용합니다.

```shell
python import_pb_to_tensorboard.py --model_dir <model path> --log_dir <log dir path>
```

#### `.tflite` 파일을 어떻게 검사합니까?

[Netron](https://github.com/lutzroeder/netron)은 TensorFlow Lite 모델을 시각화하는 가장 쉬운 방법입니다.

Netron이 TensorFlow Lite 모델을 열 수 없는 경우, 리포지토리에서 [visualize.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/tools/visualize.py) 스크립트를 사용해 볼 수 있습니다.

If you're using TF 2.5 or a later version

```shell
python -m tensorflow.lite.tools.visualize model.tflite visualized_model.html
```

Otherwise, you can run this script with Bazel

- [TensorFlow 리포지토리 복제](https://www.tensorflow.org/install/source)
- bazel을 사용하여 `visualize.py` 스크립트를 실행합니다.

```shell
bazel run //tensorflow/lite/tools:visualize model.tflite visualized_model.html
```

## 최적화

#### 변환된 TensorFlow Lite 모델의 크기를 줄이려면 어떻게 해야 합니까?

모델의 크기를 줄이기 위해 TensorFlow Lite로 변환하는 동안 [훈련 후 양자화](../performance/post_training_quantization.md)를 사용할 수 있습니다. 훈련 후 양자화는 부동 소수점에서 가중치를 8bit 정밀도로 양자화하고 런타임 중에 양자화를 해제하여 부동 소수점 계산을 수행합니다. 그러나 이때 정확성에 영향을 줄 수 있다는 점에 유의하세요.

모델 재훈련이 옵션인 경우, [양자화 인식 훈련](https://github.com/tensorflow/tensorflow/tree/r1.13/tensorflow/contrib/quantize)을 고려하세요. 그러나 양자화 인식 훈련은 컨볼루셔널 신경망 아키텍처의 일부에서만 사용할 수 있습니다.

다양한 최적화 방법에 대해 더 깊이 있게 이해하려면 [모델 최적화](../performance/model_optimization.md)를 참조하세요.

#### 머신러닝 작업을 위해 TensorFlow Lite 성능을 최적화하려면 어떻게 합니까?

TensorFlow Lite 성능을 최적화하기 위한 상위 수준 프로세스는 다음과 같습니다.

- *작업에 적합한 모델이 있는지 확인합니다.* 이미지 분류의 경우, [호스팅 모델 목록](hosted_models.md)을 확인하세요.
- *스레드 수를 조정합니다.* 많은 TensorFlow Lite 연산자는 다중 스레드 커널을 지원합니다. 이를 위해 [C++ API](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/interpreter.h#L345)에서 `SetNumThreads()`를 사용할 수 있습니다. 그러나 스레드를 늘리면 환경에 따라 성능이 달라집니다.
- *하드웨어 가속기를 사용합니다.* TensorFlow Lite는 delegate(대리자)를 사용하여 특정 하드웨어에 대한 모델 가속을 지원합니다. 예를 들어, Android의 Neural Networks API를 사용하려면 인터프리터에서 [`UseNNAPI`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/interpreter.h#L343)를 호출합니다. 또는 [GPU 대리자 튜토리얼](../performance/gpu.md)을 살펴보세요.
- *(고급) 모델을 프로파일링합니다.* Tensorflow Lite [벤치마킹 도구](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark)에는 연산자별 통계를 표시할 수 있는 프로파일러가 내장되어 있습니다. 특정 플랫폼에 대해 연산자의 성능을 최적화할 수 있는 방법을 알고 있다면 [사용자 정의 연산자](ops_custom.md)를 구현할 수 있습니다.

성능을 최적화하는 방법에 대해 더 자세히 알아보려면 [모범 사례](../performance/best_practices.md)를 참조하세요.
