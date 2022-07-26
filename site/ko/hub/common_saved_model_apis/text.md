<!--* freshness: { owner: 'akhorlin' reviewed: '2021-11-22' } *-->

# 텍스트 작업을 위한 일반적인 SavedModel API

이 페이지에서는 텍스트 관련 작업용 [TF2 SavedModel](../tf2_saved_model.md)에서 [Reusable SavedModel API](../reusable_saved_models.md)를 구현하는 방법을 설명합니다. (이는 현재 지원 중단된 [TF1 Hub 형식](../common_signatures/text.md)의 [텍스트에 대한 일반적인 서명](../tf1_hub_module)을 대체합니다.)

## 개요

**텍스트 임베딩**(텍스트의 조밀한 표현 또는 텍스트 특성 벡터라고도 함)을 계산하는 여러 API가 있습니다.

- *텍스트 입력의 텍스트 임베딩*을 위한 API는 문자열 배치를 임베딩 벡터 배치에 매핑하는 SavedModel에 의해 구현됩니다. 이것은 사용하기 매우 쉽고 TF Hub의 많은 모델이 이를 구현했습니다. 그러나 이렇게 해도 TPU에서 모델을 미세 조정할 수 없습니다.

- *전처리된 입력이 있는 텍스트 임베딩*을 위한 API는 동일한 작업을 해결하지만 두 개의 별도 SavedModel에 의해 구현됩니다.

    - tf.data 입력 파이프라인 내에서 실행될 수 있고 문자열 및 기타 가변 길이 데이터를 숫자 텐서로 변환할 수 있는 *전처리기*
    - 전처리기의 결과를 받아들이고 포함 계산의 훈련 가능한 부분을 수행하는 *인코더*

    이 분할을 통해 입력이 훈련 루프에 공급되기 전에 비동기적으로 전처리될 수 있습니다. 특히 [TPU](https://www.tensorflow.org/guide/tpu)에서 실행하고 미세 조정할 수 있는 인코더를 빌드할 수 있습니다.

- *Transformer 인코더를 사용한 텍스트 임베딩*을 위한 API는 전처리된 입력에서 BERT 및 기타 Transformer 인코더의 특정 사례로 텍스트 임베딩용 API를 확장합니다.

    - *전처리기*는 하나 이상의 입력 텍스트 세그먼트에서 인코더 입력을 구축하도록 확장됩니다.
    - *Transformer 인코더*는 개별 토큰의 컨텍스트 인식 임베딩을 노출합니다.

각각의 경우 텍스트 입력은 모델 문서에서 달리 제공하지 않는 한 일반적으로 일반 텍스트의 UTF-8 인코딩 문자열입니다.

API에 관계 없이, 다양한 언어 및 도메인의 텍스트에 대해 다양한 모델이 사전 학습되었으며 다양한 작업이 고려되었습니다. 따라서 모든 텍스트 임베딩 모델이 모든 문제에 적합한 것은 아닙니다.

<a name="feature-vector"></a>
<a name="text-embeddings-from-text"></a>

## 텍스트 입력에서 텍스트 임베딩

**텍스트 입력의 텍스트 임베딩**을 위한 SavedModel은 `[batch_size]` 형상의 문자열 텐서에서 입력 배치를 받아 입력의 밀도가 높은 표현(특성 벡터)을 사용하여 형상 `[batch_size, dim]`의 float32 텐서에 매핑합니다.

### 사용법 요약

```python
obj = hub.load("path/to/model")
text_input = ["A long sentence.",
              "single-word",
              "http://example.com"]
embeddings = obj(text_input)
```

[Reusable SavedModel API](../reusable_saved_models.md)로부터, 학습 모드(예: 드롭아웃)에서 모델을 실행하려면 키워드 인수 `obj(..., training=True)`가 필요할 수 있으며, 이 `obj`는 해당 사항에 따라 `.variables`, `.trainable_variables` 및 `.regularization_losses` 속성을 제공한다는 사실을 상기하세요.

Keras에서는 이 모든 작업이 다음에 의해 처리됩니다.

```python
embeddings = hub.KerasLayer("path/to/model", trainable=...)(text_input)
```

### 분산 훈련

텍스트 임베딩이 배포 전략으로 학습되는 모델의 일부로 사용되는 경우, `hub.load("path/to/model")` 또는 `hub.KerasLayer("path/to/model", ...)`에 대한 호출은 분산 방식으로 모델의 변수를 생성하기 위해 DistributionStrategy 범위 내에서 이루어져야 합니다. 예를 들면 다음과 같습니다.

```python
  with strategy.scope():
    ...
    model = hub.load("path/to/model")
    ...
```

### 예

- Colab 튜토리얼 [영화 리뷰로 텍스트 분류](https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/tf2_text_classification.ipynb)

<a name="text-embeddings-preprocessed"></a>

## 전처리된 입력이 포함된 텍스트 임베딩

**전처리된 입력이 포함된 텍스트 임베딩**은 두 개의 개별 SavedModel에 의해 구현됩니다.

- `[batch_size]` 형상의 문자열 텐서를 숫자 텐서의 dict에 매핑하는 **전처리기**
- 전처리기가 반환한 텐서 dict를 받아들이고, 임베딩 계산의 학습 가능한 부분을 수행하며, 출력의 dict를 반환하는 **인코더**. `"default"` 키 아래의 출력은 `[batch_size, dim]` 형상의 float32 텐서입니다.

이를 통해 입력 파이프라인에서 전처리기를 실행할 수 있지만 더 큰 모델의 일부로 인코더에서 계산한 임베딩을 미세 조정할 수 있습니다. 특히 [TPU](https://www.tensorflow.org/guide/tpu)에서 실행하고 미세 조정할 수 있는 인코더를 빌드할 수 있습니다.

이것은 텐서가 전처리기의 출력에 포함되어 있고 `"default"` 외의 추가 텐서(있는 경우)가 인코더의 출력에 포함된 구현 세부 정보입니다.

인코더의 문서는 함께 사용할 전처리기를 지정해야 합니다. 일반적으로, 정확히 하나의 올바른 선택이 존재합니다.

### 사용법 요약

```python
text_input = tf.constant(["A long sentence.",
                          "single-word",
                          "http://example.com"])
preprocessor = hub.load("path/to/preprocessor")  # Must match `encoder`.
encoder_inputs = preprocessor(text_input)

encoder = hub.load("path/to/encoder")
enocder_outputs = encoder(encoder_inputs)
embeddings = enocder_outputs["default"]
```

[Reusable SavedModel API](../reusable_saved_models.md)로부터, 학습 모드(예: 드롭아웃)에서 인코더를 실행하려면 키워드 인수 `encoder(..., training=True)`가 필요할 수 있으며, 이 `encoder`는 해당 사항에 따라 `.variables`, `.trainable_variables` 및 `.regularization_losses` 속성을 제공한다는 사실을 상기하세요.

`preprocessor` 모델이 `.variables`를 가질 수는 있지만 더 이상 학습할 수는 없습니다. 전처리는 모드 의존적이지 않습니다. 즉, `preprocessor()`에 `training=...` 인수가 있으면 아무런 효과가 없습니다.

Keras에서는 이 모든 작업이 다음에 의해 처리됩니다.

```python
encoder_inputs = hub.KerasLayer("path/to/preprocessor")(text_input)
encoder_outputs = hub.KerasLayer("path/to/encoder", trainable=True)(encoder_inputs)
embeddings = encoder_outputs["default"]
```

### 분산 훈련

인코더가 배포 전략으로 학습되는 모델의 일부로 사용되는 경우 `hub.load("path/to/encoder")` 또는 `hub.KerasLayer("path/to/encoder", ...)`에 대한 호출은 다음 내부에서 이루어져야 합니다.

```python
  with strategy.scope():
    ...
```

인코더 변수를 분산 방식으로 다시 생성하기 위해서입니다.

마찬가지로, 전처리기가 훈련된 모델의 일부인 경우(위의 간단한 예에서와 같이) 배포 전략 범위에서 로드되어야 합니다. 그러나 전처리기가 입력 파이프라인에서 사용되는 경우(예: `tf.data.Dataset.map()`에 호출 가능하게 전달됨), 해당 변수(있는 경우)를 호스트 CPU에서 배치하려면 분산 전략 범위 *밖에서* 로딩이 이루어져야 합니다.

### 예제

- Colab 튜토리얼 [BERT로 텍스트 분류](https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/text/classify_text_with_bert.ipynb)

<a name="transformer-encoders"></a>

## Transformer 인코더를 사용한 텍스트 임베딩

텍스트의 Transformer 인코더는 *n*의 일부 모델별 경계 내에서 각 시퀀스가 *n* ≥ 1개의 토큰화된 텍스트 세그먼트로 구성된 입력 시퀀스 배치에서 작동합니다. BERT 및 많은 확장의 경우 해당 경계는 2이므로 단일 세그먼트 및 세그먼트 쌍을 허용합니다.

**Transformer 인코더를 사용한 텍스트 임베딩** API는 전처리된 입력이 있는 텍스트 임베딩용 API를 이러한 환경으로 확장합니다.

### 전처리기

Transformer 인코더를 사용한 텍스트 임베딩을 위한 전처리기 SavedModel은 전처리된 입력(위 참조)이 있는 텍스트 임베딩을 위한 전처리기 SavedModel API를 구현하여 단일 세그먼트 텍스트 입력을 인코더 입력에 직접 매핑하는 방법을 제공합니다.

또한 전처리기 SavedModel은 토큰화(세그먼트당 별도)를 위한 호출 가능한 하위 객체 `tokenize`와 *n* 토큰화된 세그먼트를 인코더를 위한 하나의 입력 시퀀스로 패킹하기 위한 `bert_pack_inputs`를 제공합니다. 각 하위 객체는 [Reusable SavedModel API](../reusable_saved_models.md)를 따릅니다.

#### 사용법 요약

두 텍스트 세그먼트에 대한 구체적인 예로서 전제(첫 번째 세그먼트)가 가설(두 번째 세그먼트)을 내포하는지 여부를 묻는 문장 수반 작업을 살펴보겠습니다.

```python
preprocessor = hub.load("path/to/preprocessor")

# Tokenize batches of both text inputs.
text_premises = tf.constant(["The quick brown fox jumped over the lazy dog.",
                             "Good day."])
tokenized_premises = preprocessor.tokenize(text_premises)
text_hypotheses = tf.constant(["The dog was lazy.",  # Implied.
                               "Axe handle!"])       # Not implied.
tokenized_hypotheses = preprocessor.tokenize(text_hypotheses)

# Pack input sequences for the Transformer encoder.
seq_length = 128
encoder_inputs = preprocessor.bert_pack_inputs(
    [tokenized_premises, tokenized_hypotheses],
    seq_length=seq_length)  # Optional argument.
```

Keras에서 이 계산은 다음과 같이 표현될 수 있습니다.

```python
tokenize = hub.KerasLayer(preprocessor.tokenize)
tokenized_hypotheses = tokenize(text_hypotheses)
tokenized_premises = tokenize(text_premises)

bert_pack_inputs = hub.KerasLayer(
    preprocessor.bert_pack_inputs,
    arguments=dict(seq_length=seq_length))  # Optional argument.
encoder_inputs = bert_pack_inputs([tokenized_premises, tokenized_hypotheses])
```

#### `tokenize` 세부 사항

`preprocessor.tokenize()`에 대한 호출은 `[batch_size]` 형상의 문자열 텐서를 받아 입력 문자열을 나타내는 int32 토큰 ID를 값으로 갖는 <code>[batch_size, ...]</code> 형상의 [RaggedTensor](https://www.tensorflow.org/guide/ragged_tensor)를 반환합니다. `batch_size` 뒤에 *r* ≥ 1인 비정형 차원이 있을 수 있지만 다른 균일한 차원은 없습니다.

- *r*=1이면 형상은 `[batch_size, (tokens)]`이고 각 입력은 단순 토큰 시퀀스로 토큰화됩니다.
- *r*&gt;1이면 *r*-1 추가 수준의 그룹화가 존재합니다. 예를 들어, [tensorflow_text.BertTokenizer](https://github.com/tensorflow/text/blob/v2.3.0/tensorflow_text/python/ops/bert_tokenizer.py#L138)는 *r*=2를 사용하여 단어별로 토큰을 그룹화하고 형상 `[batch_size, (words), (tokens_per_word)]`를 내놓습니다. 이러한 추가 수준이 얼마나 되는지, 그리고 이러한 수준이 어떤 그룹화를 나타내는지는 현재 모델에 달려 있습니다.

사용자는 토큰화된 입력을 수정할 수 있습니다(예: 인코더 입력 패킹에 적용되는 seq_length 제한을 수용하기 위해). 토크나이저 출력의 추가 차원이 여기에서 도움이 될 수 있지만(예: 단어 경계를 유지하는 측면에서) 다음 단계에서는 의미가 없습니다.

[Reusable SavedModel API](../reusable_saved_models.md) 측면에서, `preprocessor.tokenize` 객체는 `.variables`을 가질 수 있지만 더 이상 학습되지는 않습니다. 토큰화는 모드 의존적이지 않습니다. 즉, `preprocessor.tokenize()`에 `training=...` 인수가 있으면 효과가 없습니다.

#### `bert_pack_inputs` 세부 사항

`preprocessor.bert_pack_inputs()`에 대한 호출은 토큰화된 입력의 Python 목록(각 입력 세그먼트에 대해 개별적으로 배치됨)을 받아들이고 Transformer 인코더 모델에 대한 고정 길이 입력 시퀀스의 배치를 나타내는 텐서 dict을 반환합니다.

각 토큰화된 입력은 `[batch_size, ...]` 형상의 int32 RaggedTensor입니다. 여기서 batch_size 이후 비정형 차원의 수 *r*은 1이거나 `preprocessor.tokenize().` 출력에서와 동일합니다(후자는 편의를 위한 것이며 추가 차원은 패킹 전에 평면화됨).

패킹은 인코더가 예상하는 대로 입력 세그먼트 주변에 특수 토큰을 추가합니다. `bert_pack_inputs()` 호출은 원래 BERT 모델과 많은 확장에서 사용하는 패킹 체계를 정확히 구현합니다. 패킹된 시퀀스는 하나의 시퀀스 시작 토큰으로 시작하고 토큰화된 세그먼트가 이어지며, 각 세그먼트는 하나의 세그먼트 끝 토큰에 의해 종료됩니다. seq_length까지 남은 위치(있는 경우)는 패딩 토큰으로 채워집니다.

패킹된 시퀀스가 seq_length를 초과하는 경우 `bert_pack_inputs()`은 패킹된 시퀀스가 seq_length 내에 정확히 맞도록 세그먼트를 대략 동일한 크기의 접두사로 자릅니다.

패킹은 모드 의존적이지 않습니다. 즉, `preprocessor.bert_pack_inputs()`에 `training=...` 인수가 있으면 효과가 없습니다. 또한 `preprocessor.bert_pack_inputs`은 변수를 갖거나 미세 조정을 지원할 것으로 예상되지 않습니다.

### 인코더

인코더는 [Reusable SavedModel API](../reusable_saved_models.md)의 프로비전을 포함하여 전처리된 입력(위 참조)이 있는 텍스트 임베딩 API에서와 동일한 방식으로 `encoder_inputs`의 dict에서 호출됩니다.

#### 사용법 요약

```python
enocder = hub.load("path/to/encoder")
enocder_outputs = encoder(encoder_inputs)
```

Keras에서는 다음에 해당합니다.

```python
encoder = hub.KerasLayer("path/to/encoder", trainable=True)
encoder_outputs = encoder(encoder_inputs)
```

#### 세부 사항

`encoder_outputs`은 다음과 같은 키 텐서의 dict입니다.

<!-- TODO(b/172561269): More guidance for models trained without poolers. -->

- `"sequence_output"`: 패킹된 모든 입력 시퀀스의 각 토큰의 컨텍스트 인식 임베딩을 포함한 `[batch_size, seq_length, dim]` 형상의 float32 텐서
- `"pooled_output"`: 훈련 가능한 방식으로 sequence_output에서 파생된 각 입력 시퀀스가 전체적으로 임베딩된 `[batch_size, dim]` 형상의 float32 텐서
- 전처리된 입력이 있는 텍스트 임베딩 API에서 요구하는 `"default"`: 각 입력 시퀀스가 임베딩된 `[batch_size, dim]` 형상의 float32 텐서(이것은 pooled_output의 별칭에 불과할 수 있음)

`encoder_inputs`의 내용은 이 API 정의에서 엄격하게 요구되지 않습니다. 그러나 BERT 스타일 입력을 사용하는 인코더의 경우 인코더를 교환하고 전처리기 모델을 재사용할 때 마찰을 최소화하기 위해 다음 이름([TensorFlow Model Garden의 NLP 모델링 도구 키트](https://github.com/tensorflow/models/tree/master/official/nlp)에서 제공)을 사용하는 것이 좋습니다.

- `"input_word_ids"`: 패킹된 입력 시퀀스의 토큰 ID가 있는 `[batch_size, seq_length]` 형상의 int32 텐서(즉, 시퀀스 시작 토큰, 세그먼트 끝 토큰 및 패딩 포함)
- `"input_mask"`: 패딩 앞에 있는 모든 입력 토큰의 위치에 값 1이 있고 패딩 토큰에 대한 값이 0인 `[batch_size, seq_length]` 형상의 int32 텐서
- `"input_type_ids"`: 각 위치에서 입력 토큰을 발생시킨 입력 세그먼트의 인덱스가 있는 `[batch_size, seq_length]` 형상의 int32 텐서. 첫 번째 입력 세그먼트(인덱스 0)에는 시퀀스 시작 토큰과 세그먼트 끝 토큰이 포함됩니다. 두 번째 및 이후 세그먼트(있는 경우)에는 해당 세그먼트 끝 토큰이 포함됩니다. 패딩 토큰은 다시 인덱스 0을 얻습니다.

### 분산 훈련

배포 전략 범위 내부 또는 외부에서 전처리기 및 인코더 객체를 로드하는 경우, 전처리된 입력이 있는 텍스트 임베딩 API에서와 동일한 규칙이 적용됩니다(위 참조).

### 예제

- Colab 튜토리얼 [TPU에서 BERT를 사용하여 GLUE 작업 해결](https://colab.research.google.com/github/tensorflow/text/blob/master/docs/tutorials/bert_glue.ipynb).
