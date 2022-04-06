<!--* freshness: { owner: 'maringeo' reviewed: '2021-10-10' review_interval: '6 months' } *-->

# TF1 허브 형식으로 모델 내보내기

이 형식에 대한 자세한 내용은 [TF1 허브 형식](tf1_hub_module.md)에서 읽을 수 있습니다.

## 호환성 참고

TF1 허브 형식은 TensorFlow 1에 맞춰져 있습니다. TensorFlow 2의 TF 허브에서는 부분적으로만 지원됩니다. 대신 [모델 내보내기](exporting_tf2_saved_model) 가이드에 따라 새로운 [TF2 SavedModel](tf2_saved_model.md) 형식으로 게시하는 것을 고려하세요.

TF1 허브 형식은 구문 수준(동일한 파일 이름 및 프로토콜 메시지)에서 TensorFlow 1의 SavedModel 형식과 유사하지만 모듈 재사용, 구성 및 재교육(예: 리소스 초기화 프로그램의 다른 저장소, 메타그래프에 대한 다른 태그 지정 방식)을 허용하기 위해 의미상 다릅니다. 디스크에서 이들을 구별하는 가장 쉬운 방법은 `tfhub_module.pb` 파일이 있는지 여부입니다.

## 일반적인 접근

새 모듈을 정의하려면, 게시자는 `module_fn` 함수로 `hub.create_module_spec()`을 호출합니다. 이 함수는 호출자가 제공할 입력에 대해 `tf.placeholder()`를 사용하여 모듈의 내부 구조를 나타내는 그래프를 구성합니다. 그런 다음 `hub.add_signature(name, inputs, outputs)`을 한 번 이상 호출하여 서명을 정의합니다.

예를 들면 다음과 같습니다.

```python
def module_fn():
  inputs = tf.placeholder(dtype=tf.float32, shape=[None, 50])
  layer1 = tf.layers.dense(inputs, 200)
  layer2 = tf.layers.dense(layer1, 100)
  outputs = dict(default=layer2, hidden_activations=layer1)
  # Add default signature.
  hub.add_signature(inputs=inputs, outputs=outputs)

...
spec = hub.create_module_spec(module_fn)
```

특정 TensorFlow 그래프 내에서 모듈 객체를 인스턴스화하기 위해 경로 대신 `hub.create_module_spec()`의 결과를 사용할 수 있습니다. 이 경우 체크 포인트가 없으며 모듈 인스턴스는 변수 이니셜라이저를 대신 사용합니다.

모든 모듈 인스턴스는 `export(path, session)` 메서드를 통해 디스크에 직렬화될 수 있습니다. 모듈을 내보내면 `session`에 있는 변수의 현재 상태와 함께 해당 정의가 전달된 경로로 직렬화됩니다. 모듈을 처음으로 내보낼 때뿐만 아니라 미세 조정된 모듈을 내보낼 때도 사용할 수 있습니다.

TensorFlow Estimator와의 호환성을 위해 `hub.LatestModuleExporter`는 최신 체크포인트에서, `tf.estimator.LatestExporter`가 최신의 체크포인트에서 전체 모델을 내보는 것과 같이 모델을 내보냅니다.

모듈 게시자는 가능한 경우 [공통 서명](common_signatures/index.md)을 구현하여 소비자가 쉽게 모듈을 교환하고 해당 문제에 가장 적합한 모델을 찾을 수 있도록 해야 합니다.

## 실제 예

일반적인 텍스트 임베딩 형식에서 모듈을 만드는 방법에 대한 실제 예제를 보려면 [텍스트 임베딩 모듈 내보내기](https://github.com/tensorflow/hub/blob/master/examples/text_embeddings/export.py)를 살펴보세요.

## 게시자를 위한 조언

소비자가 보다 쉽게 미세 조정할 수 있도록 다음 사항에 유의하세요.

- 미세 조정에는 정규화가 필요합니다. 모듈은 `REGULARIZATION_LOSSES` 모음과 함께 내보내집니다. 소비자가 `tf.losses.get_regularization_losses()`에서 얻는 항목에 선택한 `tf.layers.dense(..., kernel_regularizer=...)` 등을 추가합니다. L1/L2 정규화 손실을 정의하는 이 방법을 선호합니다.

- 게시자 모델에서 `tf.train.FtrlOptimizer`, `tf.train.ProximalGradientDescentOptimizer` 및 기타 근위 옵티마이저의 `l1_` 및 `l2_regularization_strength` 매개변수를 통해 L1/L2 정규화를 정의하지 마세요. 이들은 모듈과 함께 내보내지지 않으며, 전체적으로 정규화 강도를 설정하는 것은 소비자에게 적합하지 않을 수 있습니다. 넓은(즉, 희소 선형) 또는 넓고 깊은 모델의 L1 정규화를 제외하고는 개별 정규화 손실을 대신 사용할 수 있습니다.

- 드롭아웃, 배치 정규화 또는 유사한 훈련 테크닉을 사용하는 경우, 하이퍼 매개변수를 예상되는 여러 용도에서 의미 있는 값으로 설정합니다. 드롭아웃은 대상 문제의 과대적합 경향에 따라 조정되어야 할 수 있습니다. 배치 정규화에서 모멘텀(일명 감쇄 계수)은 작은 데이터세트 및/또는 큰 배치로 미세 조정이 가능할 만큼 충분히 작아야 합니다. 고급 소비자의 경우, 중요한 하이퍼 매개변수를 제어할 수 있는 서명을 추가하는 것이 좋습니다.
