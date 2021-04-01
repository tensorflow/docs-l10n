# TensorFlow Lite 및 TensorFlow 연산자 호환성

TensorFlow Lite는 공통 추론 모델에 사용되는 여러 TensorFlow 연산을 지원합니다. 이들 연산은 TensorFlow Lite 최적화 변환기에서 처리되므로, 지원되는 연산이 TensorFlow Lite의 해당 연산에 매핑되기 전에 제거되거나 통합될 수 있습니다.

TensorFlow Lite 내장 연산자 라이브러리는 제한된 수의 TensorFlow 연산자만 지원하므로 모든 모델을 변환할 수 있는 것은 아닙니다. 지원되는 연산의 경우에도 성능상의 이유로 매우 특정한 사용 패턴이 예상되는 경우가 있습니다. 향후 TensorFlow Lite 릴리스에서는 지원되는 연산을 확장할 예정입니다.

TensorFlow Lite에서 동작하는 TensorFlow 모델을 빌드하는 방법을 이해하는 가장 좋은 방법은 이 프로세스에 적용되는 제한 사항과 함께 연산이 어떻게 변환되고 최적화되는지를 신중하게 고려하는 것입니다.

## 지원되는 유형

대부분의 TensorFlow Lite 연산은 부동 소수점(`float32`) 및 양자화된(`uint8`, `int8`) 추론을 모두 대상으로 하지만, 많은 연산은 아직 `tf.float16` 및 문자열과 같은 다른 유형을 처리하지 않습니다.

서로 다른 버전의 연산을 사용하는 외에도 부동 소수점 모델과 양자화된 모델 사이에는 변환 방식에서도 차이가 있습니다. 양자화 변환에는 텐서에 대한 동적 범위 정보가 필요합니다. 이를 위해서는 모델 훈련 중에 "가짜 양자화", 보정 데이터세트를 통한 범위 정보 가져오기, 또는 "즉석" 범위 추정하기가 필요합니다. [양자화](../performance/model_optimization.md)를 참조하세요.

## 지원되는 연산 및 제한 사항

TensorFlow Lite는 TensorFlow 연산의 일부를 지원하며 몇 가지 제한 사항이 있습니다. 연산 및 제한 사항의 전체 목록은 [TF Lite 연산 페이지](https://www.tensorflow.org/mlir/tfl_ops)를 참조하세요.

## 간단한 변환, 지속적인 통합과 융합

TensorFlow Lite에 직접적으로 대응하는 연산이 없더라도 많은 TensorFlow 연산을 TensorFlow Lite에서 처리할 수 있습니다. 그래프에서 제거되거나(`tf.identity`), 텐서로 대체되거나(`tf.placeholder`), 더 복잡한 연산으로 융합(`tf.nn.bias_add`)될 수 있는 연산자가 이에 해당합니다. 때로는 지원되는 일부 연산이 이러한 프로세스 중 하나를 통해 제거될 수도 있습니다.

다음은 일반적으로 그래프에서 제거되는 일부 TensorFlow 연산을 나타낸 목록입니다.

- `tf.add`
- `tf.check_numerics`
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
