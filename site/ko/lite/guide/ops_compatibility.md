# TensorFlow Lite 및 TensorFlow 연산자 호환성

모델에서 사용하는 머신러닝(ML) 연산자는 TensorFlow 모델에서 TensorFlow Lite 형식으로 변환하는 프로세스에 영향을 미칠 수 있습니다. TensorFlow Lite 변환기는 공통 추론 모델에서 사용되는 제한된 수의 TensorFlow 연산을 지원하며, 이는 모든 모델을 직접 변환할 수 없다는 것을 의미합니다. 변환기 도구를 통해 추가 연산자를 포함할 수 있지만 이 방식으로 모델을 변환하려면 모델을 실행하는 데 사용하는 TensorFlow Lite 런타임 환경을 수정해야 하며, 이는 [Google Play 서비스](../android/play_services)와 같은 표준 런타임 개발 옵션을 사용하는 기능을 제한할 수 있습니다.

TensorFlow Lite Converter는 직접 지원되는 연산자와 호환되도록 모델 구조를 분석하고 최적화를 적용하도록 설계되었습니다. 예를 들면, 모델의 ML 연산자에 따라 변환기는 TensorFlow Lite의 모델에 매핑하기 위해 연산자를 [생략하거나 결합](../models/convert/operation_fusion)할 수 있씁니다.

지원되는 연산이라도, 성능상의 이유로 특정 활용 패턴이 가끔 예상됩니다. TensorFlow Lite와 함께 사용할 수 있는 TensorFlow 모델을 구축하는 방법을 이해하는 가장 좋은 방법은 이 프로세스에 의해 부과되는 제한과 함께 연산을 변환하고 최적화하는 방법을 신중하게 생각하는 것입니다.

## 지원되는 연산자

TensorFlow Lite 내장 연산자는 TensorFlow 코어 라이브러리의 일부분인 연산자의 하위 집합입니다. TensorFlow 모델은 또한 여러분이 정의한 합성 연산자나 새로운 연산자의 형태인 사용자 정의 연산자를 포함합니다. 아래의 다이어그램은 이러한 연산자 간의 관계를 보여줍니다.

![TensorFlow 연산자](../images/convert/tf_operators_relationships.png)

이러한 범위의 ML 모델 연산자에서 변환 프로세스에 의해 지원되는 3가지 유형의 모델이 있습니다.

1. TensorFlow Lite 내장 연산자만 있는 모델(**권장됨**)
2. 내장 연산자가 있는 모델 및 select TensorFlow 코어 연산자.
3. 내장 연산자가 있는 모델, TensorFlow 코어 연산자 및/또는 사용자 정의 연산자.

모델인 TensorFlow Lite가 네이티브로 지원하는 연산만 포함한다면 이를 변환하기 위해 추가 플래그가 필요하지 않습니다. 이러한 유형의 모델은 원활하게 변환되며 기본 TensorFlow Lite 런타임을 사용하여 최적화하고 실행하는 것이 더 간단하기 때문에 이는 권장되는 경로입니다. 여러분의 모델에 [Google Play 서비스](../android/play_services)와 같은 더 많은 개발 옵션도 사용할 수 있습니다. [TensorFlow Lite 변환기 가이드](../models/convert/convert_models)로 시작할 수 있습니다. 내장 연산자에 대한 목록은 [TensorFlow Lite Ops 페이지](https://www.tensorflow.org/mlir/tfl_ops)를 참조하세요.

코어 라이브러리에서 select TensorFlow 연산을 포함해야 한다면 변환 시 해당 연산을 지정해야 하며 런타임은 이러한 연산을 포함해야 합니다. 자세한 단계는 [Select TensorFlow 연산자](ops_select.md) 토픽을 참조하세요.

가능하면 항상 변환된 모델의 사용자 정의 연산자를 포함하는 최후의 옵션을 사용하지 마세요. [사용자 정의 연산자](https://www.tensorflow.org/guide/create_op)는 여러 기본 형식 TensorFlow 코어 연산자를 결합하거나 완전히 새로운 연산자를 정의하여 생성된 연산자 중 하나입니다. 사용자 정의 연산자가 변환되면, 내장 TensorFlow Lite 라이브러리의 외부에서 불일치를 유발하여 전체적인 모델의 크기를 증가시킬 수 있습니다. 모바일이나 기기 개발을 위해 특별히 생성된 것이 아닌 사용자 정의 연산은 기기가 제한된 리소스에 배포되는 경우 서버 환경보다 성능이 더욱 좋지 않은 결과를 초래할 수 있습니다. 마지막으로, select TensorFlow 코어 연산자를 포함하는 것과 마찬가지로, 사용자 지정 연산자를 사용하려면 [Google Play 서비스](../android/play_services)와 같은 표준 런타임 서비스를 활용하는 것을 제한하는 모델 런타임 환경을 수정해야 합니다.

## 지원되는 유형

대부분의 TensorFlow Lite 연산은 부동 소수점(`float32`) 및 양자화된(`uint8`, `int8`) 추론을 모두 대상으로 하지만, 많은 연산은 아직 `tf.float16` 및 문자열과 같은 다른 유형을 처리하지 않습니다.

서로 다른 버전의 연산을 사용하는 것 외에도 부동 소수점 모델과 양자화된 모델 사이에는 변환 방식에서도 차이가 있습니다. 양자화 변환에는 텐서에 대한 동적 범위 정보가 필요합니다. 이를 위해서는 모델 훈련 중에 "가짜 양자화", 보정 데이터 세트를 통한 범위 정보 가져오기 또는 "즉석" 범위 추정하기가 필요합니다. 더 자세한 내용은 [양자화](../performance/model_optimization.md)를 참조하세요.

## 간단한 변환, 지속적인 통합과 융합

TensorFlow Lite에 직접적으로 대응하는 연산이 없더라도 많은 TensorFlow 연산을 TensorFlow Lite에서 처리할 수 있습니다. 그래프(`tf.identity`)에서 제거되거나, 텐서(`tf.placeholder`)로 대체되거나, 더 복잡한 연산(`tf.nn.bias_add`)으로 융합될 수 있는 연산자가 이에 해당합니다. 때로는 지원되는 일부 연산이 이러한 프로세스 중 하나를 통해 제거될 수도 있습니다.

다음은 일반적으로 그래프에서 제거되는 일부 TensorFlow 연산을 나타낸 목록입니다.

- `tf.add`
- `tf.debugging.check_numerics`
- `tf.constant`
- `tf.div`
- `tf.divide`
- `tf.fake_quant_with_min_max_args`
- `tf.fake_quant_with_min_max_vars`
- `tf.identity`
- `tf.maximum`
- `tf.minimum`
- `tf.multiply`
- `tf.no_op`
- `tf.placeholder`
- `tf.placeholder_with_default`
- `tf.realdiv`
- `tf.reduce_max`
- `tf.reduce_min`
- `tf.reduce_sum`
- `tf.rsqrt`
- `tf.shape`
- `tf.sqrt`
- `tf.square`
- `tf.subtract`
- `tf.tile`
- `tf.nn.batch_norm_with_global_normalization`
- `tf.nn.bias_add`
- `tf.nn.fused_batch_norm`
- `tf.nn.relu`
- `tf.nn.relu6`

참고: 이러한 연산의 대부분은 TensorFlow Lite에서 대응하는 연산이 없으며, 이들 연산을 제거하거나 융합할 수 없는 경우 해당 모델을 변환할 수 없습니다.

## 실험적 연산

다음 TensorFlow Lite 연산도 제공되지만 사용자 정의 모델에 사용할 준비가 되지 않았습니다.

- `CALL`
- `CONCAT_EMBEDDINGS`
- `CUSTOM`
- `EMBEDDING_LOOKUP_SPARSE`
- `HASHTABLE_LOOKUP`
- `LSH_PROJECTION`
- `SKIP_GRAM`
- `SVDF`
