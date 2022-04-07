<!--* freshness: { owner: 'maringeo' reviewed: '2021-10-10' review_interval: '6 months' } *-->

# TF1 Hub 형식

2018년 출시 당시 TensorFlow Hub는 TensorFlow 1 프로그램으로 가져올 수 있는 단일 유형의 자산인 TF1 Hub 형식을 제공했습니다.

이 페이지에서는 `hub.Module` 클래스 및 관련 API와 함께 TF1(또는 TF2의 TF1 호환 모드)에서 TF1 Hub 형식을 사용하는 방법을 설명합니다. (일반적인 용도는 TF1 Hub 형식의 하나 이상의 모델을 `tf.compat.layers` 또는 `tf.layers`와 결합하여 TF1 `Estimator` 내부에서 `tf.Graph`를 빌드하는 것입니다.)

TensorFlow 2(TF1 호환 모드의 외부)의 사용자는 [`hub.load()` 또는 `hub.KerasLayer`](tf2_saved_model.md)와 함께 새 API를 사용해야 합니다. 새 API는 새 TF2 SavedModel 자산 유형을 로드하지만, [TF1 Hub 형식을 TF2로 로드하기 위한 지원](migration_tf2.md)도 제한적입니다.

## TF1 Hub 형식의 모델 사용하기

### TF1 Hub 형식의 모델 인스턴스화하기

TF1 Hub 형식의 모델은 다음과 같이 URL 또는 파일 시스템 경로가 있는 문자열에서 `hub.Module` 객체를 생성하여 TensorFlow 프로그램으로 가져옵니다.

```python
m = hub.Module("path/to/a/module_dir")
```

그러면 현재 TensorFlow 그래프에 모듈의 변수가 추가됩니다. 이니셜라이저를 실행하면 디스크에서 사전 훈련된 값을 읽습니다. 마찬가지로, 테이블 및 기타 상태가 그래프에 추가됩니다.

### 캐싱 모듈

URL에서 모듈을 만들 때, 모듈 콘텐츠가 다운로드되고 로컬 시스템의 임시 디렉토리에 캐싱됩니다. 모듈이 캐싱되는 위치는 `TFHUB_CACHE_DIR` 환경 변수를 사용하여 재정의할 수 있습니다. 자세한 내용은 [캐싱](caching.md)을 참조하세요.

### 모듈 적용하기

인스턴스화되면, 모듈 `m`은 텐서 입력에서 텐서 출력까지 Python 함수처럼 0번 이상 호출될 수 있습니다.

```python
y = m(x)
```

각 호출에서 현재 TensorFlow 그래프에 연산을 추가하여 `x`에서 `y`를 계산합니다. 여기에 훈련된 가중치가 있는 변수가 포함되는 경우, 모든 애플리케이션 간에 공유됩니다.

모듈은 여러 가지 방식으로 적용될 수 있도록 여러 개의 *서명*을 정의할 수 있습니다(Python 객체에 *메서드*가 있는 방식과 유사). 모듈의 설명서는 사용 가능한 서명을 설명해야 합니다. 위의 호출은 `"default"`라는 서명을 적용합니다. 선택적 `signature=` 인수에 이름을 전달하여 모든 서명을 선택할 수 있습니다.

서명에 여러 입력이 있는 경우, 서명으로 정의된 키와 함께 dict로 전달되어야합니다. 마찬가지로, 서명에 여러 출력이 있는 경우, 서명으로 정의된 키에서 `as_dict=True`를 전달하여 dict로 검색할 수 있습니다(키 `"default"`는 `as_dict=False`인 경우 반환되는 단일 출력용입니다). 따라서 모듈을 적용하는 가장 일반적인 형태는 다음과 같습니다.

```python
outputs = m(dict(apples=x1, oranges=x2), signature="fruit_to_pet", as_dict=True)
y1 = outputs["cats"]
y2 = outputs["dogs"]
```

호출자는 서명에서 정의된 모든 입력을 제공해야 하지만, 모듈의 모든 출력을 사용할 필요는 없습니다. TensorFlow는 `tf.Session.run()`에서 대상의 종속성으로 끝나는 모듈의 해당 부분만 실행합니다. 실제로 모듈 게시자는 주요 출력과 함께 고급 용도(예: 중간 레이어 활성화)를 위한 다양한 출력을 제공하도록 선택할 수 있습니다. 모듈 소비자는 추가 출력을 정상적으로 처리해야 합니다.

### 대체 모듈 시도하기

동일한 작업에 대한 여러 모듈이있을 때마다 TensorFlow Hub는 모듈 핸들을 문자열 값 하이퍼 파라미터처럼 변경하는 것만 큼 쉽게 다른 것을 시도 할 수 있도록 호환되는 서명 (인터페이스)을 장착하도록 권장합니다.

이를 위해 인기 있는 작업에 권장되는 [공통 서명](common_signatures/index.md) 모음을 유지합니다.

## 새 모듈 생성하기

### 호환성 참고 사항

TF1 Hub 형식은 TensorFlow 1에 맞춰져 있습니다. TensorFlow 2의 TF Hub에서는 부분적으로만 지원됩니다. 대신 새로운 [TF2 SavedModel](tf2_saved_model.md) 형식으로 게시하는 것을 고려해 보세요.

TF1 Hub 형식은 구문 수준(같은 파일 이름 및 프로토콜 메시지)에서 TensorFlow 1의 SavedModel 형식과 유사하지만, 모듈 재사용, 구성 및 재훈련(예: 리소스 이니셜라이저의 다른 저장소, 메타그래프에 대한 다른 태그 지정 규칙)을 허용하는 측면에서는 의미상 다릅니다. 디스크에서 이들을 구별하는 가장 쉬운 방법은 `tfhub_module.pb` 파일이 있는지 여부입니다.

### 일반적인 접근 방식

새 모듈을 정의하려면, 게시자는 `module_fn` 함수로 `hub.create_module_spec()`을 호출합니다. 이 함수는 호출자가 제공할 입력에 대해 `tf.placeholder()`를 사용하여 모듈의 내부 구조를 나타내는 그래프를 구성합니다. 그런 다음 `hub.add_signature(name, inputs, outputs)`을 한 번 이상 호출하여 서명을 정의합니다.

예를 들면:

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

특정 TensorFlow 그래프 내에서 모듈 객체를 인스턴스화하기 위해 경로 대신 `hub.create_module_spec()` 의 결과를 사용할 수 있습니다. 이 경우 체크포인트가 없으며, 모듈 인스턴스는 변수 이니셜라이저를 대신 사용합니다.

모든 모듈 인스턴스는 `export(path, session)` 메서드를 통해 디스크에 직렬화될 수 있습니다. 모듈을 내보내면 `session`에 있는 변수의 현재 상태와 함께 해당 정의가 전달된 경로로 직렬화됩니다. 모듈을 처음으로 내보낼 때뿐만 아니라 미세 조정된 모듈을 내보낼 때도 사용할 수 있습니다.

TensorFlow Estimator와의 호환성을 위해 `hub.LatestModuleExporter`는 최신 체크포인트에서, `tf.estimator.LatestExporter`가 최신의 체크포인트에서 전체 모델을 내보는 것과 같이 모델을 내보냅니다.

모듈 게시자는 가능한 경우 [공통 서명](common_signatures/index.md)을 구현하여 소비자가 쉽게 모듈을 교환하고 해당 문제에 가장 적합한 모델을 찾을 수 있도록 해야 합니다.

### 실제 예제

일반적인 텍스트 임베딩 형식에서 모듈을 만드는 방법에 대한 실제 예제를 보려면 [텍스트 임베딩 모듈 내보내기](https://github.com/tensorflow/hub/blob/master/examples/text_embeddings/export.py)를 살펴보세요.

## 미세 조정

가져온 모듈의 변수를 주변 모델의 변수와 함께 훈련하는 것을 *미세 조정*이라고 합니다. 미세 조정은 품질을 향상할 수 있지만, 새로운 문제를 제기합니다. 소비자는 더 간단한 품질 조정을 알아본 후 모듈 게시자가 권장하는 경우에만 미세 조정을 고려할 것을 권장합니다.

### 소비자용

미세 조정을 사용하려면, `hub.Module(..., trainable=True)`로 모듈을 인스턴스화하여 변수를 훈련 가능하게 만들고 TensorFlow의 `REGULARIZATION_LOSSES`를 가져옵니다. 모듈에 여러 그래프 변형이 있는 경우, 훈련에 적합한 변형을 선택해야 합니다. 일반적으로 태그 `{"train"}`이 있는 것을 선택합니다.

예를 들어, 처음부터 훈련하는 것보다 낮은 학습률과 같이, 사전 훈련된 가중치를 망치지 않는 훈련 체계를 선택합니다.

### 게시자용

소비자가 보다 쉽게 미세 조정할 수 있도록 다음 사항에 유의하세요.

- 미세 조정에는 정규화가 필요합니다. 모듈은 `REGULARIZATION_LOSSES` 모음과 함께 내보내집니다. 소비자가 `tf.losses.get_regularization_losses()`에서 얻는 항목에 선택한 `tf.layers.dense(..., kernel_regularizer=...)` 등을 추가합니다. L1/L2 정규화 손실을 정의하는 이 방법을 선호합니다.

- 게시자 모델에서는 `tf.train.FtrlOptimizer`, `tf.train.ProximalGradientDescentOptimizer` 및 기타 근위 옵티마이저의 `l1_` 및 `l2_regularization_strength` 매개변수를 통해 L1/L2 정규화를 정의하지 마세요. 이들은 모듈과 함께 내보내지지 않으며, 전체적으로 정규화 강도를 설정하는 것은 소비자에게 적합하지 않을 수 있습니다. 넓은(즉, 희소 선형) 또는 넓고 깊은 모델의 L1 정규화를 제외하고는 개별 정규화 손실을 대신 사용할 수 있습니다.

- If you use dropout, batch normalization, or similar training techniques, set their hyperparameters to values that make sense across many expected uses. The dropout rate may have to be adjusted to the target problem's propensity to overfitting. In batch normalization, the momentum (a.k.a. decay coefficient) should be small enough to enable fine-tuning with small datasets and/or large batches. For advanced consumers, consider adding a signature that exposes control over critical hyperparameters.
