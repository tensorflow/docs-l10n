# 알려진 문제

XLA를 사용한 컴파일은 프로그램의 성능을 크게 향상할 수 있지만 TensorFlow interop에는 알려진 껄끄러운 문제들이 있습니다.

## 다른 장치에서의 `tf.Variable`

*오류 메시지*: `INVALID_ARGUMENT: Trying to access resource <Variable> (defined @ <Loc>) located in device CPU:0 from device GPU:0`

XLA 클러스터는 정확히 하나의 장치에서 실행되며 다른 장치에 있는 `tf.Variable`을 읽거나 쓸 수 없습니다. 일반적으로 이 오류 메시지는 변수가 처음부터 올바른 장치에 배치되지 않았음을 나타냅니다. 오류 메시지는 문제가 되는 변수의 위치를 정확하게 지정해야 합니다.

참고: `int32` 유형의 `tf.Variable`은 항상 호스트에 배치되며 GPU에는 배치될 수 없습니다. 해결 방법으로 `int64`를 사용할 수 있습니다.

## TensorArray TF/XLA 상호 전환은 지원되지 않음

오류 메시지: `Support for TensorList crossing the XLA/TF boundary is not implemented`.

XLA는 `tf.TensorArray`를 지원합니다. 그러나, TF와 XLA 표현 간의 *상호 변환(interconversion)*은 아직 구현되지 않았습니다. 이 오류는 `TensorArray`가 컴파일된 블록 내에서 사용되지만 파생 요소가 외부에서 사용되는 경우 종종 발생합니다.

해결 방법: 파생 요소를 사용하는 가장 바깥쪽 범위를 컴파일하세요.

## TensorFlow while 루프는 제한되어야 함(또는 backprop를 비활성화)

오류 메시지: `XLA compilation requires a fixed tensor list size. Set the max number of elements. This could also happen if you're using a TensorArray in a while loop that does not have its maximum_iteration set, you can fix this by setting maximum_iteration to a suitable value`.

TF while created using `tf.while_loop`를 사용하여 생성한 TF while [loops](https://www.tensorflow.org/api_docs/python/tf/while_loop)는 `TensorArray`에 모든 중간 결과를 누적하는 방식으로 역전파를 지원합니다. 단, XLA는 제한된 `TensorArray`s만을 지원하니다.

해결 방법: 컴파일한 모든 while 루프는 `maximum_iterations` 매개변수를 컴파일할 때, 알려진 상수 값으로 설정하거나 `back_prop=False`를 사용하여 역전파를 비활성화해야 합니다.

## 동적 `tf.TensorArray`는 지원되지 않음

`tf.TensorArray(..., dynamic_size=True)`에 대한 쓰기는 XLA로 컴파일할 수 없습니다. 이러한 쓰기에는 배열이 원래 경계를 초과할 때 알 수 없는 횟수의 재할당이 필요하기 때문입니다.

해결 방법: 배열에 정적으로 알려진 바인딩을 제공하세요.

## 난수 생성은 TF 시드를 무시함

XLA는 현재 임의 연산에 대한 TF 시드를 무시합니다. 이는 `tf.random.normal` 또는 `tf.nn.dropout`과 같은 상태 저장 TF 임의 연산에 영향을 줍니다. XLA는 컴파일이 동일한 프로세스 내에서 실행할 때마다 새로운 고유 시드로 시드된 것처럼 동작합니다(프로세스의 첫 번째 실행은 항상 동일한 결과를 산출함).

*해결 방법*: `tf.random.stateless_uniform` 또는 `tf.random.Generator`와 같은 [권장 RNG](https://www.tensorflow.org/guide/random_numbers#stateless_rngs)를 직접 사용합니다.
