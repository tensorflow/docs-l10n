# TensorFlow 그래픽용 디버그 모드

Tensorflow Graphics는 L2 정규화된 텐서와 입력이 특정 범위에 있을 것으로 예상하는 삼각 함수에 크게 의존합니다. 최적화 중에 업데이트를 통해 이들 변수에서 이들 함수가 `Inf` 또는 `NaN` 값을 반환하도록 하는 값을 갖을 수 있습니다. 이러한 문제를 보다 간단하게 디버깅하기 위해 TensorFlow Graphics는 그래프에 어설션을 주입하여 반환된 값의 올바른 범위와 유효성을 확인하는 디버그 플래그를 제공합니다. 이로 인해 계산 속도가 느려질 수 있으므로 디버그 플래그는 기본적으로 `False`로 설정됩니다.

사용자는 `-tfg_debug` 플래그를 설정하여 디버그 모드에서 코드를 실행할 수 있습니다. 플래그는 먼저 다음 두 모듈을 가져와서 프로그래밍 방식으로 설정할 수도 있습니다.

```python
from absl import flags
from tensorflow_graphics.util import tfg_flags
```

그런 다음 코드에 다음 줄을 추가합니다.

```python
flags.FLAGS[tfg_flags.TFG_ADD_ASSERTS_TO_GRAPH].value = True
```
