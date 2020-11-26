<!--* freshness: { owner: 'arnoegw' reviewed: '2020-09-11' } *-->

# 텍스트에 대한 공통 서명

이 페이지에서는 텍스트 입력을 수용하는 작업을 위해 [TF1 Hub 형식](../tf1_hub_module.md)의 모듈에서 구현해야 하는 일반적인 서명을 설명합니다. [TF2 SavedModel 형식](../tf2_saved_model.md)에 대해서는 유사한 [SavedModel API](../common_saved_model_apis/text.md)를 참조하세요.

## 텍스트 특성 벡터

**텍스트 특성 벡터** 모듈은 텍스트 특성으로부터 밀집 벡터 표현을 만듭니다. 모듈은 형상 `[batch_size]`의 문자열 배치를 받아들이고 이를 형상 `[batch_size, N]`의 `float32` 텐서에 매핑합니다. 이 작업을 종종 차원 `N`에서 **텍스트 임베딩**이라고 합니다.

### 기본 사용법

```python
  embed = hub.Module("path/to/module")
  representations = embed([
      "A long sentence.",
      "single-word",
      "http://example.com"])
```

### 특성 열 사용법

```python
    feature_columns = [
      hub.text_embedding_column("comment", "path/to/module", trainable=False),
    ]
    input_fn = tf.estimator.inputs.numpy_input_fn(features, labels, shuffle=True)
    estimator = tf.estimator.DNNClassifier(hidden_units, feature_columns)
    estimator.train(input_fn, max_steps=100)
```

## 참고

모듈은 여러 도메인 및/또는 작업에서 사전 훈련되었으므로 모든 텍스트 특성 벡터 모듈이 해당 문제에 적합하지는 않습니다. 예를 들어, 일부 모듈은 단일 언어에서 훈련되었습니다.

모듈이 문자열 처리와 훈련 가능한 변수를 동시에 인스턴스화해야 하기 때문에 이 인터페이스는 TPU에서 텍스트 표현의 미세 조정을 허용하지 않습니다.
