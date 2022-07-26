# ExampleValidator TFX 파이프라인 구성 요소

ExampleValidator 파이프라인 구성 요소는 훈련 및 제공 데이터에 존재하는 문제점을 식별합니다. 데이터에서 다양한 종류의 문제점을 감지할 수 있습니다. 예를 들어 다음을 수행할 수 있습니다.

1. 사용자의 기대치를 코드화하는 스키마와 데이터 통계를 비교하여 검증을 수행합니다.
2. 훈련 및 제공 데이터를 비교하여 훈련-제공의 편향을 감지합니다.
3. 일련의 데이터를 확인하여 데이터 이탈을 감지합니다.

ExampleValidator 파이프라인 구성 요소는 StatisticsGen 파이프라인 구성 요소에 의해 계산된 데이터 통계를 스키마와 비교하여 예제 데이터에 존재하는 문제점을 식별합니다. 추론된 스키마는 입력 데이터가 만족할 것으로 예상되는 속성을 코드화하고 개발자가 이를 수정할 수 있습니다.

- 입력: SchemaGen 구성 요소의 스키마 및 StatisticsGen 구성 요소의 통계
- 출력: 검증 결과

## ExampleValidator 및 TensorFlow 데이터 검증

ExampleValidator는 입력 데이터 검증을 위해 [TensorFlow Data Validation](tfdv.md)을 광범위하게 사용합니다.

## ExampleValidator 구성 요소 사용하기

ExampleValidator 파이프라인 구성 요소는 일반적으로 배포가 매우 쉽고 사용자 정의가 거의 필요하지 않습니다. 일반적인 코드는 다음과 같습니다.

```python
validate_stats = ExampleValidator(
      statistics=statistics_gen.outputs['statistics'],
      schema=schema_gen.outputs['schema']
      )
```

자세한 내용은 [ExampleValidator API 참조](https://www.tensorflow.org/tfx/api_docs/python/tfx/v1/components/ExampleValidator)에서 확인할 수 있습니다.
