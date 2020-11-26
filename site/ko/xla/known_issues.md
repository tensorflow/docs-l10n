# 알려진 문제

XLA를 사용한 컴파일은 프로그램의 성능을 크게 향상할 수 있지만 TensorFlow interop에는 알려진 껄끄러운 문제들이 있습니다.

## TensorArray TF/XLA 상호 변환

이 문제는 `Support for TensorList crossing the XLA/TF boundary is not implemented`라는 오류 메시지로 나타납니다.

XLA는 `tf.TensorArray`를 지원합니다. 그러나, TF와 XLA 표현 간의 *상호 변환(interconversion)*은 아직 구현되지 않았습니다. 이 오류는 `TensorArray`가 컴파일된 블록 내에서 사용되지만 파생 요소가 외부에서 사용되는 경우 종종 발생합니다.

해결 방법: 파생 요소를 사용하는 가장 바깥쪽 범위를 컴파일하세요.

## 동적 `tf.TensorArray`는 지원되지 않음

`tf.TensorArray(..., dynamic_size=True)`에 대한 쓰기는 XLA로 컴파일할 수 없습니다. 이러한 쓰기에는 배열이 원래 경계를 초과할 때 알 수 없는 횟수의 재할당이 필요하기 때문입니다.

해결 방법: 배열에 정적으로 알려진 바인딩을 제공하세요.

## 난수 생성

XLA는 현재 임의 연산에 대한 TF 시드를 무시합니다. 이는 `tf.random.normal` 또는 `tf.nn.dropout`과 같은 상태 저장 TF 임의 연산에 영향을 줍니다. XLA는 컴파일이 각 실행 시 새로운 고유 시드로 시드된 것처럼 동작합니다. 이 제한은 상태 비저장 임의 ops에는 적용되지 않습니다.
