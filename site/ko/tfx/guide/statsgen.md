# StatisticsGen TFX 파이프라인 구성 요소

StatisticsGen TFX 파이프라인 구성 요소는 다른 파이프라인 구성 요소에서 사용할 수 있는 훈련 및 적용 데이터 모두에 대한 특성 통계를 생성합니다. StatisticsGen은 Beam을 사용하여 대규모 데이터세트로 확장합니다.

- 입력: ExampleGen 파이프라인 구성 요소로 만들어진 데이터세트
- 출력: 데이터세트 통계

## StatisticsGen 및 TensorFlow 데이터 검증

StatisticsGen은 데이터세트에서 통계를 생성하기 위해 [TensorFlow 데이터 검증](tfdv.md)을 광범위하게 사용합니다.

## StatsGen 구성 요소 사용하기

StatisticsGen 파이프라인 구성 요소는 일반적으로 배포가 매우 쉽고 사용자 정의가 거의 필요하지 않습니다. 일반적인 코드는 다음과 같습니다.

```python
compute_eval_stats = StatisticsGen(
      examples=example_gen.outputs['examples'],
      name='compute-eval-stats'
      )
```

## 스키마와 StatsGen 구성 요소 사용하기

파이프라인의 첫 번째 실행에서 StatisticsGen의 출력은 스키마를 추론하는 데 사용됩니다. 그러나 후속 실행에서는 데이터세트에 대한 추가 정보를 포함하는 수동으로 큐레이팅된 스키마가 있을 수 있습니다. 이 스키마를 StatisticsGen에 제공함으로써 TFDV는 데이터세트의 선언된 속성을 기반으로 더 유용한 통계를 제공할 수 있습니다.

이 설정에서 다음과 같이 ImporterNode에서 가져온 큐레이팅된 스키마로 StatisticsGen을 호출합니다.

```python
user_schema_importer = Importer(
    source_uri=user_schema_dir, # directory containing only schema text proto
    artifact_type=standard_artifacts.Schema).with_id('schema_importer')

compute_eval_stats = StatisticsGen(
      examples=example_gen.outputs['examples'],
      schema=user_schema_importer.outputs['result'],
      name='compute-eval-stats'
      )
```

### 큐레이팅된 스키마 만들기

TFX의 `Schema`는 TensorFlow Metadata <a data-md-type="raw_html" href="https://github.com/tensorflow/metadata/blob/master/tensorflow_metadata/proto/v0/schema.proto">`Schema` proto</a>의 인스턴스입니다. 처음부터 [텍스트 형식](https://googleapis.dev/python/protobuf/latest/google/protobuf/text_format.html)으로 구성할 수 있습니다. 하지만 `SchemaGen`에서 생성한 추론된 스키마를 시작점으로 사용하기가 더 쉽습니다. `SchemaGen` 구성 요소가 실행되면 스키마는 다음 경로의 파이프라인 루트 아래에 있습니다.

```
<pipeline_root>/SchemaGen/schema/<artifact_id>/schema.pbtxt
```

여기서 `<artifact_id>`는 MLMD에서 이 버전의 스키마에 대한 고유 ID를 나타냅니다. 그런 다음 스키마 proto를 수정하여 안정적으로 추론할 수 없는 데이터세트에 대한 정보를 전달할 수 있습니다. 그러면 `StatisticsGen`의 출력이 더 유용해지고 [`ExampleValidator`](https://www.tensorflow.org/tfx/guide/exampleval) 구성 요소에서 수행되는 검증이 더 엄격해집니다.

자세한 내용은 [StatisticsGen API 참조](https://www.tensorflow.org/tfx/api_docs/python/tfx/v1/components/StatisticsGen)에서 확인할 수 있습니다.
