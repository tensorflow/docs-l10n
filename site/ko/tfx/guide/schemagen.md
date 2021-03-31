# SchemaGen TFX 파이프라인 구성 요소

일부 TFX 구성 요소는 *스키마*라는 입력 데이터에 대한 설명을 사용합니다. 스키마는 [schema.proto](https://github.com/tensorflow/metadata/blob/master/tensorflow_metadata/proto/v0/schema.proto)의 인스턴스입니다. 특성 값에 대한 데이터 유형, 특성이 모든 예에 있어야 하는지 여부, 허용된 값 범위 및 기타 속성을 지정할 수 있습니다. SchemaGen 파이프라인 구성 요소는 훈련 데이터에서 유형, 범주 및 범위를 추론하여 스키마를 자동으로 생성합니다.

- 입력: StatisticsGen 구성 요소의 통계
- 출력: 데이터 스키마 proto

다음은 스키마 proto에서 발췌한 것입니다.

```proto
...
feature {
  name: "age"
  value_count {
    min: 1
    max: 1
  }
  type: FLOAT
  presence {
    min_fraction: 1
    min_count: 1
  }
}
feature {
  name: "capital-gain"
  value_count {
    min: 1
    max: 1
  }
  type: FLOAT
  presence {
    min_fraction: 1
    min_count: 1
  }
}
...
```

다음 TFX 라이브러리는 스키마를 사용합니다.

- TensorFlow 데이터 검증
- TensorFlow 변환
- TensorFlow 모델 분석

일반적인 TFX 파이프라인에서 SchemaGen은 다른 파이프라인 구성 요소에서 사용되는 스키마를 생성합니다.

참고: 자동 생성된 스키마는 최상의 결과이며 데이터의 기본 속성만 추론하려고 합니다. 개발자는 필요에 따라 검토하고 수정해야 합니다.

## SchemaGen과 TensorFlow 데이터 검증

SchemaGen은 스키마 추론을 위해 [TensorFlow 데이터 검증](tfdv.md)을 광범위하게 사용합니다.

## SchemaGen 구성 요소 사용하기

SchemaGen 파이프라인 구성 요소는 일반적으로 배포가 매우 쉽고 사용자 정의가 거의 필요하지 않습니다. 일반적인 코드는 다음과 같습니다.

```python
from tfx import components

...

infer_schema = components.SchemaGen(
    statistics=compute_training_stats.outputs['statistics'])
```
