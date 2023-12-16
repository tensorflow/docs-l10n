# Google Cloud를 사용하여 ML용 데이터 전처리하기

이 튜토리얼에서는 [TensorFlow Transform](https://github.com/tensorflow/transform){: .external}(`tf.Transform` 라이브러리)를 사용하여 머신러닝(ML) 데이터 전처리를 구현하는 방법을 설명합니다. TensorFlow용 `tf.Transform` 라이브러리를 사용하면 데이터 전처리 파이프라인을 통해 인스턴스 수준 및 풀패스 데이터 변환을 모두 정의할 수 있습니다. 이러한 파이프라인은 [Apache Beam](https://beam.apache.org/){: .external}을 사용하여 효율적으로 실행되며, 모델이 제공될 때와 동일한 변환을 예측 중에 적용하기 위해 TensorFlow 그래프를 부산물로 생성합니다.

이 튜토리얼에서는 [Dataflow](https://cloud.google.com/dataflow/docs){: .external }를 Apache Beam의 실행기로 사용하는 엔드 투 엔드 예제를 제공합니다. 이때 여러분이 [BigQuery](https://cloud.google.com/bigquery/docs){: .external }, Dataflow, [Vertex AI](https://cloud.google.com/vertex-ai/docs/start/introduction-unified-platform){: .external }, 그리고 TensorFlow [Keras](https://www.tensorflow.org/guide/keras/overview) API를 잘 알고 있다고 가정합니다. 또한 [Vertex AI Workbench](https://cloud.google.com/vertex-ai/docs/workbench/introduction){: .external }와 같이 Jupyter Notebook을 사용해 본 경험이 있다고 가정합니다.

또한 이 튜토리얼은 [ML용 데이터 전처리: 옵션 및 권장 사항](../../guide/tft_bestpractices)에 설명된 대로 여러분이 Google Cloud의 전처리 유형, 과제 및 옵션에 대한 개념을 잘 알고 있다고 가정합니다.

## 목적

- `tf.Transform` 라이브러리를 사용하여 Apache Beam 파이프라인을 구현합니다.
- Dataflow에서 파이프라인을 실행합니다.
- `tf.Transform` 라이브러리를 사용하여 TensorFlow 모델을 구현합니다.
- 예측을 위해 모델을 훈련하고 사용합니다.

## 비용

이 튜토리얼은 다음과 같은 Google Cloud의 청구 가능한 구성 요소를 사용합니다.

- [Vertex AI](https://cloud.google.com/vertex-ai/pricing){: .external}
- [클라우드 스토리지](https://cloud.google.com/storage/pricing){: .external}
- [BigQuery](https://cloud.google.com/bigquery/pricing){: .external}
- [Dataflow](https://cloud.google.com/dataflow/pricing){: .external}

<!-- This doc uses plain text cost information because the pricing calculator is pre-configured -->

이 튜토리얼을 실행하는 비용을 추정하려면 하루 종일 모든 리소스를 사용한다고 가정하고 미리 구성된 [가격 계산기](/products/calculator/#id=fad4d8-dd68-45b8-954e-5a56a5d){: .external }를 사용하세요.

## 시작하기 전에

1. Google Cloud 콘솔의 프로젝트 선택 페이지에서 프로젝트를 선택하거나 [Google Cloud 프로젝트 만듭니다](https://cloud.google.com/resource-manager/docs/creating-managing-projects).

참고: 이 단계에서 만든 리소스를 유지하지 않으려면 기존 프로젝트를 선택하지 말고 새 프로젝트를 만드세요. 이 단계를 완료한 후에는 프로젝트를 삭제하여 프로젝트와 관련된 모든 리소스를 제거할 수 있습니다.

[프로젝트 선택으로 이동](https://console.cloud.google.com/projectselector2/home/dashboard){: class="button button-primary" target="console" track-type="solution" track-name="consoleLink" track-metadata-position="body" }

1. 클라우드 프로젝트에 청구가 활성화되어 있는지 확인합니다. [프로젝트의 결제 상태 확인](https://cloud.google.com/billing/docs/how-to/verify-billing-enabled) 방법을 알아보세요.

2. Dataflow, Vertex AI, Notebooks API를 활성화합니다. [API 활성화](https://console.cloud.google.com/flows/enableapi?apiid=dataflow,aiplatform.googleapis.com,notebooks.googleapis.com){: class="button button-primary" target="console" track-type="solution" track-name="consoleLink" track-metadata-position="body" }

## 이 솔루션에서 사용하는 Jupyter 노트북

다음 Jupyter 노트북은 구현 예제를 보여줍니다.

- [Notebook 1](https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/blogs/babyweight_tft/babyweight_tft_keras_.ipynb){: .external }은 데이터 전처리에 대한 내용을 다룹니다. 자세한 내용은 나중에 [Apache Beam 파이프라인 구현하기](#implement-the-apache-beam-pipeline) 섹션에서 제공됩니다.
- [Notebook 2](https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/blogs/babyweight_tft/babyweight_tft_keras_.ipynb){: .external }는 모델 훈련에 대한 내용을 다룹니다. 자세한 내용은 나중에 [TensorFlow 모델 구현하기](#implement-the-tensorflow-model) 섹션에서 제공됩니다.

다음 섹션에서는 이러한 노트북을 복제하고 노트북을 실행하여 구현 예제가 작동하는 방식을 알아봅니다.

## 사용자 관리 노트북 인스턴스의 실행하기

1. Google Cloud 콘솔에서 **Vertex AI Workbench** 페이지로 이동합니다.

    [Workbench로 이동](https://console.cloud.google.com/ai-platform/notebooks/list/instances){: class="button button-primary" target="console" track-type="solution" track-name="consoleLink" track-metadata-position="body" }

2. **사용자 관리 노트북** 탭에서 **+새 노트북**을 클릭합니다.

3. 인스턴스 유형으로 **GPU가 없는 TensorFlow Enterprise 2.8(LTS 포함)**을 선택합니다.

4. **만들기**를 클릭합니다.

노트북을 만든 후, JupyterLab이 프록시 초기화를 마칠 때까지 기다립니다. 준비가 완료되면 노트북 이름 옆에 **JupyterLab 열기**가 표시됩니다.

## 노트북 복제

1. **사용자 관리 노트북 탭**의 노트북 이름 옆의 **JupyterLab 열기**를 클릭합니다. JupyterLab 인터페이스가 새 탭에서 열립니다.

    JupyterLab에 **권장 빌드** 대화 상자가 표시되면 **취소**를 클릭하여 제안된 빌드를 거부합니다.

2. **런처** 탭에서 **터미널**을 클릭합니다.

3. 터미널 창에서 노트북을 복제합니다.

    ```sh
    git clone https://github.com/GoogleCloudPlatform/training-data-analyst
    ```

## Apache Beam 파이프라인 구현

이 섹션과 다음 섹션인 [Dataflow에서 파이프라인 실행하기](#run-the-pipeline-in-dataflow){: track-type="solution" track-name="internalLink" track-metadata-position="body" }은 노트북 1에 대한 개요와 컨텍스트를 제공합니다. 이 노트북은 `tf.Transform` 라이브러리를 사용하여 데이터를 전처리하는 방법을 설명하는 실용적인 예제를 제공합니다. 이 예제는 다양한 입력을 기반으로 아기 몸무게를 예측하는 데 사용되는 Natality 데이터세트를 사용합니다. 이 데이터는 BigQuery의 공개 [natality](https://console.cloud.google.com/bigquery?p=bigquery-public-data&d=samples&t=natality&page=table&_ga=2.267763789.2122871960.1676620306-376763843.1676620306){: target="console" track-type="solution" track-name="consoleLink" track-metadata-position="body" } 테이블에 저장됩니다.

### 노트북 1 실행하기

1. JupyterLab 인터페이스에서 **파일 &gt; 경로에서 열기**를 클릭한 후 다음 경로를 입력합니다.

    ```sh
    training-data-analyst/blogs/babyweight_tft/babyweight_tft_keras_01.ipynb
    ```

2. **편집 &gt; 모든 출력 지우기**를 클릭합니다.

3. **필수 패키지 설치** 섹션에서 첫 번째 셀을 실행하여 `pip install apache-beam` 명령어를 실행합니다.

    출력의 마지막 부분은 다음과 같습니다.

    ```none{:.devsite-disable-click-to-copy}
    Successfully installed ...
    ```

    출력에서 종속성 오류를 무시할 수 있습니다. 아직 커널을 재시작할 필요는 없습니다.

4. 두 번째 셀을 실행하여 `pip install tensorflow-transform ` 명령을 실행합니다. 출력의 마지막 부분은 다음과 같습니다.

    ```none{:.devsite-disable-click-to-copy}
    Successfully installed ...
    Note: you may need to restart the kernel to use updated packages.
    ```

    출력에서 종속성 오류를 무시할 수 있습니다.

5. **커널 &gt; 커널 재시작**을 클릭합니다.

6. **설치된 패키지 확인** 및 **setup.py를 만들어 Dataflow 컨테이너에 패키지 설치하기** 섹션의 셀을 실행합니다.

7. **전역 플래그 설정** 섹션에서 `PROJECT` 및 `BUCKET` 옆에 있는 `your-project`를 Cloud 프로젝트 ID로 교체한 후 셀을 실행합니다.

8. 노트북의 마지막 셀까지 나머지 모든 셀을 실행합니다. 각 셀에서 수행할 작업에 대한 자세한 내용은 노트북의 지침을 참조하세요.

### 파이프라인 개요

노트북 예제에서 Dataflow는 `tf.Transform` 파이프라인을 대규모로 실행하여 데이터를 준비하고 변환 아티팩트를 생성합니다. 이 문서의 뒷부분에서는 파이프라인의 각 단계를 수행하는 함수에 대해 설명합니다. 전체 파이프라인 단계는 다음과 같습니다.

1. BigQuery에서 훈련 데이터를 읽습니다.
2. `tf.Transform` 라이브러리를 사용하여 훈련 데이터를 분석하고 변환합니다.
3. 변환된 훈련 데이터를 [TFRecord](https://www.tensorflow.org/tutorials/load_data/tfrecord){: target="external" class="external" track-type="solution" track-name="externalLink" track-metadata-position="body" } 형식으로 클라우드 스토리지에 기록합니다.
4. BigQuery에서 평가 데이터를 읽습니다.
5. 2단계에서 생성한 `transform_fn` 그래프를 사용하여 평가 데이터를 변환합니다.
6. 변환된 훈련 데이터를 TFRecord 형식으로 클라우드 스토리지에 기록합니다.
7. 나중에 모델을 만들고 내보내는 데 사용할 변환 아티팩트를 클라우드 스토리지에 기록합니다.

다음 예제는 전체 파이프라인에 대한 Python 코드를 보여줍니다. 이어지는 섹션에서는 각 단계에 대한 설명과 코드 목록을 제공합니다.

```py{:.devsite-disable-click-to-copy}
def run_transformation_pipeline(args):

    pipeline_options = beam.pipeline.PipelineOptions(flags=[], **args)

    runner = args['runner']
    data_size = args['data_size']
    transformed_data_location = args['transformed_data_location']
    transform_artefact_location = args['transform_artefact_location']
    temporary_dir = args['temporary_dir']
    debug = args['debug']

    # Instantiate the pipeline
    with beam.Pipeline(runner, options=pipeline_options) as pipeline:
        with impl.Context(temporary_dir):

            # Preprocess train data
            step = 'train'
            # Read raw train data from BigQuery
            raw_train_dataset = read_from_bq(pipeline, step, data_size)
            # Analyze and transform raw_train_dataset
            transformed_train_dataset, transform_fn = analyze_and_transform(raw_train_dataset, step)
            # Write transformed train data to sink as tfrecords
            write_tfrecords(transformed_train_dataset, transformed_data_location, step)

            # Preprocess evaluation data
            step = 'eval'
            # Read raw eval data from BigQuery
            raw_eval_dataset = read_from_bq(pipeline, step, data_size)
            # Transform eval data based on produced transform_fn
            transformed_eval_dataset = transform(raw_eval_dataset, transform_fn, step)
            # Write transformed eval data to sink as tfrecords
            write_tfrecords(transformed_eval_dataset, transformed_data_location, step)

            # Write transformation artefacts
            write_transform_artefacts(transform_fn, transform_artefact_location)

            # (Optional) for debugging, write transformed data as text
            step = 'debug'
            # Write transformed train data as text if debug enabled
            if debug == True:
                write_text(transformed_train_dataset, transformed_data_location, step)
```

### BigQuery에서 원시 훈련 데이터 읽기{: id="read_raw_training_data"}

첫 번째 단계는 `read_from_bq` 함수를 사용하여 BigQuery에서 원시 훈련 데이터를 읽는 것입니다. 이 함수는 BigQuery에서 추출된 `raw_dataset` 객체를 반환합니다. `data_size` 값을 전달하고 `train` 또는 `eval`의 `step` 값을 전달합니다. BigQuery 소스 쿼리는 다음 예제와 같이 `get_source_query` 함수를 사용하여 구성됩니다.

```py{:.devsite-disable-click-to-copy}
def read_from_bq(pipeline, step, data_size):

    source_query = get_source_query(step, data_size)
    raw_data = (
        pipeline
        | '{} - Read Data from BigQuery'.format(step) >> beam.io.Read(
                           beam.io.BigQuerySource(query=source_query, use_standard_sql=True))
        | '{} - Clean up Data'.format(step) >> beam.Map(prep_bq_row)
    )

    raw_metadata = create_raw_metadata()
    raw_dataset = (raw_data, raw_metadata)
    return raw_dataset
```

`tf.Transform` 전처리를 수행하기 전에 맵, 필터, 그룹핑 및 창 처리와 같은 일반적인 Apache Beam 기반 처리를 수행해야 할 수 있습니다. 이 예제에서 코드는 `beam.Map(prep_bq_row)` 메서드를 사용하여 BigQuery에서 읽은 레코드를 정리합니다. 여기서 `prep_bq_row`는 사용자 정의 함수입니다. 이 사용자 정의 함수는 범주형 특성의 숫자 코드를 사람이 읽을 수 있는 레이블로 변환합니다.

또한 `tf.Transform` 라이브러리를 사용하여 BigQuery에서 추출한 `raw_data` 객체를 분석하고 변환하려면 `raw_dataset` 객체를 생성해야 하며, 이는 `raw_data` 및 `raw_metadata` 객체의 튜플입니다. `raw_metadata` 객체는 다음과 같이 `create_raw_metadata` 함수를 사용하여 생성됩니다.

```py{:.devsite-disable-click-to-copy}
CATEGORICAL_FEATURE_NAMES = ['is_male', 'mother_race']
NUMERIC_FEATURE_NAMES = ['mother_age', 'plurality', 'gestation_weeks']
TARGET_FEATURE_NAME = 'weight_pounds'

def create_raw_metadata():

    feature_spec = dict(
        [(name, tf.io.FixedLenFeature([], tf.string)) for name in CATEGORICAL_FEATURE_NAMES] +
        [(name, tf.io.FixedLenFeature([], tf.float32)) for name in NUMERIC_FEATURE_NAMES] +
        [(TARGET_FEATURE_NAME, tf.io.FixedLenFeature([], tf.float32))])

    raw_metadata = dataset_metadata.DatasetMetadata(
        schema_utils.schema_from_feature_spec(feature_spec))

    return raw_metadata
```

노트북에서 이 메서드를 정의하는 셀 바로 뒤에 있는 셀을 실행하면 `raw_metadata.schema` 객체의 콘텐츠가 표시됩니다. 여기에는 다음 열이 포함됩니다.

- `gestation_weeks`(유형: `FLOAT`)
- `is_male`(유형: `BYTES`)
- `mother_age`(유형: `FLOAT`)
- `mother_race`(유형: `BYTES`)
- `plurality`(유형: `FLOAT`)
- `weight_pounds`(유형: `FLOAT`)

### 원시 훈련 데이터로 변환하기

훈련 데이터의 입력 원시 특성에 일반적인 전처리 변환을 적용하여 ML용으로 준비한다고 가정해 보겠습니다. 이러한 변환에는 다음 표에 표시된 것처럼 풀패스 및 인스턴스 수준 연산이 모두 포함됩니다.

<table>
<thead>
  <tr>
    <th>입력 특성</th>
    <th>변환</th>
    <th>필요한 스탯</th>
    <th>유형</th>
    <th>출력 특성</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td><code>weight_pound</code></td>
    <td>없음</td>
    <td>없음</td>
    <td>NA</td>
    <td><code>weight_pound</code></td>
  </tr>
  <tr>
    <td><code>mother_age</code></td>
    <td>정규화</td>
    <td>mean, var</td>
    <td>풀패스</td>
    <td><code>mother_age_normalized</code></td>
  </tr>
  <tr>
    <td><code>mother_age</code></td>
    <td>동일한 크기의 버킷화</td>
    <td>quantiles</td>
    <td>풀패스</td>
    <td><code>mother_age_bucketized</code></td>
  </tr>
  <tr>
    <td><code>mother_age</code></td>
    <td>로그 계산</td>
    <td>없음</td>
    <td>인스턴스 수준</td>
    <td>
        <code>mother_age_log</code>
    </td>
  </tr>
  <tr>
    <td><code>plurality</code></td>
    <td>단일 또는 다중 아기인지 표시</td>
    <td>없음</td>
    <td>인스턴스 수준</td>
    <td><code>is_multiple</code></td>
  </tr>
  <tr>
    <td><code>is_multiple</code></td>
    <td>공칭값을 수치 인덱스로 변환</td>
    <td>vocab</td>
    <td>풀패스</td>
    <td><code>is_multiple_index</code></td>
  </tr>
  <tr>
    <td><code>gestation_weeks</code></td>
    <td>0에서 1 사이의 스케일</td>
    <td>min, max</td>
    <td>풀패스</td>
    <td><code>gestation_weeks_scaled</code></td>
  </tr>
  <tr>
    <td><code>mother_race</code></td>
    <td>공칭값을 수치 인덱스로 변환</td>
    <td>vocab</td>
    <td>풀패스</td>
    <td><code>mother_race_index</code></td>
  </tr>
  <tr>
    <td><code>is_male</code></td>
    <td>공칭값을 수치 인덱스로 변환</td>
    <td>vocab</td>
    <td>풀패스</td>
    <td><code>is_male_index</code></td>
  </tr>
</tbody>
</table>

이러한 변환은 텐서 사전(`input_features`)을 예상하고 처리된 특성 사전(`output_features`)을 반환하는 `preprocess_fn` 함수에서 구현됩니다.

다음 코드는 `preprocess_fn` 함수의 구현을 보여 주며, `tf.Transform` 풀패스 변환 API(접두사 `tft.`) 및 TensorFlow(접두사 `tf.`) 인스턴스 수준 연산을 사용합니다.

```py{:.devsite-disable-click-to-copy}
def preprocess_fn(input_features):

    output_features = {}

    # target feature
    output_features['weight_pounds'] = input_features['weight_pounds']

    # normalization
    output_features['mother_age_normalized'] = tft.scale_to_z_score(input_features['mother_age'])

    # scaling
    output_features['gestation_weeks_scaled'] =  tft.scale_to_0_1(input_features['gestation_weeks'])

    # bucketization based on quantiles
    output_features['mother_age_bucketized'] = tft.bucketize(input_features['mother_age'], num_buckets=5)

    # you can compute new features based on custom formulas
    output_features['mother_age_log'] = tf.math.log(input_features['mother_age'])

    # or create flags/indicators
    is_multiple = tf.as_string(input_features['plurality'] > tf.constant(1.0))

    # convert categorical features to indexed vocab
    output_features['mother_race_index'] = tft.compute_and_apply_vocabulary(input_features['mother_race'], vocab_filename='mother_race')
    output_features['is_male_index'] = tft.compute_and_apply_vocabulary(input_features['is_male'], vocab_filename='is_male')
    output_features['is_multiple_index'] = tft.compute_and_apply_vocabulary(is_multiple, vocab_filename='is_multiple')

    return output_features
```

`tf.Transform` [framework](https://github.com/tensorflow/transform){: .external }에는 앞의 예제 외에도 다음 표에 나열된 내용 등 몇 가지 다른 변환이 있습니다.

<table>
<thead>
  <tr>
  <th>변환</th>
  <th>적용 대상</th>
  <th>설명</th>
  </tr>
</thead>
<tbody>
    <tr>
    <td><code>scale_by_min_max</code></td>
    <td>숫자 특성</td>
    <td>숫자 열을 [<code>output_min</code>,       <code>output_max</code>] 범위로 스케일링</td>
  </tr>
  <tr>
    <td><code>scale_to_0_1</code></td>
    <td>숫자 특성</td>
    <td>입력 열이 [<code>0</code>,<code>1</code>] 범위를 갖도록 스케일링된 열을 반환</td>
  </tr>
  <tr>
    <td><code>scale_to_z_score</code></td>
    <td>숫자 특성</td>
    <td>평균 0이고 분산이 1인 표준화된 열을 반환</td>
  </tr>
  <tr>
    <td><code>tfidf</code></td>
    <td>텍스트 특성</td>
    <td> <i>x</i>의 용어를 해당 용어 빈도 * 역 문서 빈도에 매핑</td>
  </tr>
  <tr>
    <td><code>compute_and_apply_vocabulary</code></td>
    <td>범주형 특성</td>
    <td>범주형 특성에 대한 어휘를 생성하고 이 어휘를 사용하여 정수에 매핑</td>
  </tr>
  <tr>
    <td><code>ngrams</code></td>
    <td>텍스트 특성</td>
    <td>n-gram의 <code>SparseTensor</code> 생성</td>
  </tr>
  <tr>
    <td><code>hash_strings</code></td>
    <td>범주형 특성</td>
    <td>문자열을 버킷으로 해시</td>
  </tr>
  <tr>
    <td><code>pca</code></td>
    <td>숫자 특성</td>
    <td>바이어스된 공분산을 사용하여 데이터세트에서 PCA를 계산</td>
  </tr>
  <tr>
    <td><code>bucketize</code></td>
    <td>숫자 특성</td>
    <td>각 입력에 할당된 버킷 인덱스와 함께 동일한 크기(사분위수 기반)의 버킷화된 열을 반환</td>
  </tr>
</tbody>
</table>

파이프라인의 이전 단계에서 생성된 `preprocess_fn` 함수에 구현된 변환을 `raw_train_dataset` 객체에 적용하려면 `AnalyzeAndTransformDataset` 메서드를 사용해야 합니다. 이 메서드는 `raw_dataset` 객체를 입력으로 예상하고 `preprocess_fn` 함수를 적용한 후 `transformed_dataset` 객체와 `transform_fn` 그래프를 생성합니다. 다음 코드는 이 처리 과정을 보여줍니다.

```py{:.devsite-disable-click-to-copy}
def analyze_and_transform(raw_dataset, step):

    transformed_dataset, transform_fn = (
        raw_dataset
        | '{} - Analyze & Transform'.format(step) >> tft_beam.AnalyzeAndTransformDataset(
            preprocess_fn, output_record_batches=True)
    )

    return transformed_dataset, transform_fn
```

변환은 분석 단계와 변환 단계의 두 단계에서 원시 데이터에 적용됩니다. 이 문서의 뒷부분에 있는 그림 3은 `AnalyzeAndTransformDataset` 메서드가 `AnalyzeDataset` 메서드와 `TransformDataset` 메서드로 분해되는 방식을 보여줍니다.

#### 분석 단계

분석 단계에서는 원시 훈련 데이터를 풀패스 프로세스로 분석하여 변환에 필요한 통계를 계산합니다. 여기에는 평균, 분산, 최소, 최대, 분위수 및 어휘의 계산이 포함됩니다. 분석 프로세스는 원시 데이터세트(원시 데이터 및 원시 메타데이터)를 예상하고 두 가지 출력을 생성합니다.

- `transform_fn`: 분석 단계에서 계산된 스탯과 스탯을 사용하는 변환 로직을 인스턴스 수준 연산으로 포함하는 TensorFlow 그래프입니다. [그래프 저장하기](#save_the_graph){: track-type="solution" track-name="internalLink" track-metadata-position="body" }에서 논의한 것과 같이 `transform_fn` 그래프가 저장되어 모델 `serving_fn` 함수에 첨부됩니다. 이렇게 하면 온라인 예측 데이터 포인트에 동일한 변환을 적용할 수 있습니다.
- `transform_metadata`: 변환 후 데이터의 예상 스키마를 설명하는 객체입니다.

분석 단계는 다음 다이어그램, 그림 1와 같습니다.

<figure id="tf-transform-analyze-phase">
  <img src="images/data-preprocessing-for-ml-with-tf-transform-tf-transform-analyze-phase.svg"
    alt="The tf.Transform analyze phase.">
  <figcaption><b>Figure 1.</b> The <code>tf.Transform</code> analyze phase.</figcaption>
</figure>

`tf.Transform` [분석기](https://github.com/tensorflow/transform/blob/master/tensorflow_transform/beam/analyzer_impls.py){: target="github" class="external" track-type="solution" track-name="gitHubLink" track-metadata-position="body" }는 `min`, `max`, `sum`, `size`, `mean`, `var`, `covariance`, `quantiles`, `vocabulary`, `pca`를 포함합니다.

#### 변환 단계

변환 단계에서는 분석 단계에서 생성된 `transform_fn` 그래프를 사용하여 인스턴스 수준 프로세스에서 원시 훈련 데이터를 변환하고 이를 통해 변환된 훈련 데이터를 생성합니다. 변환된 훈련 데이터는 변환된 메타데이터(분석 단계에서 생성된)와 페어링되어 `transformed_train_dataset` 데이터세트를 생성합니다.

변환 단계는 다음 다이어그램, 그림 2와 같습니다.

<figure id="tf-transform-transform-phase">
  <img src="images/data-preprocessing-for-ml-with-tf-transform-tf-transform-transform-phase.svg"
    alt="The tf.Transform transform phase.">
  <figcaption><b>Figure 2.</b> The <code>tf.Transform</code> transform phase.</figcaption>
</figure>

특성을 전처리하려면 `preprocess_fn` 함수의 구현에서 필요한 `tensorflow_transform` 변환(코드에서 `tft`로 가져옴)을 호출하면 됩니다. 예를 들어 `tft.scale_to_z_score` 연산을 호출하면 `tf.Transform` 라이브러리가 이 함수 호출을 평균 및 분산 분석기로 변환하고 분석 단계에서 스탯을 계산한 다음 이 스탯을 적용하여 변환 단계에서 숫자 특성을 정규화합니다. 이 모든 작업은 `AnalyzeAndTransformDataset(preprocess_fn)` 메서드를 호출하면 자동으로 수행됩니다.

이 호출에 의해 생성된 `transformed_metadata.schema` 엔티티에는 다음 열이 포함됩니다:

- `gestation_weeks_scaled`(유형: `FLOAT`)
- `is_male_index`(유형: `INT`, is_categorical: `True`)
- `is_multiple_index`(유형: `INT`, is_categorical: `True`)
- `mother_age_bucketized`(유형: `INT`, is_categorical: `True`)
- `mother_age_log`(유형: `FLOAT`)
- `mother_age_normalized`(유형: `FLOAT`)
- `mother_race_index`(유형: `INT`, is_categorical: `True`)
- `weight_pounds`(유형: `FLOAT`)

이 시리즈의 첫 번째 부분인 [전처리 연산](data-preprocessing-for-ml-with-tf-transform-pt1#preprocessing_operations)에서 설명한 것처럼 특성 변환은 범주형 특성을 숫자 표현으로 변환합니다. 변환 후에는 범주형 특성이 정수 값으로 표시됩니다. `transformed_metadata.schema` 엔티티에서 `INT` 유형 열에 대한 `is_categorical` 플래그는 열이 범주형 특성인지 아니면 실제 숫자 특성을 나타내는지 여부를 나타냅니다.

### 변환된 훈련 데이터 기록하기{: id="step_3_write_transformed_training_data"}

분석 및 변환 단계에서 `preprocess_fn` 함수로 훈련 데이터를 전처리한 후, 데이터를 싱크에 기록하여 TensorFlow 모델 훈련에 사용할 수 있습니다. Dataflow를 사용하여 Apache Beam 파이프라인을 실행할 때의 싱크는 클라우드 스토리지입니다. 그렇지 않은 경우의 싱크는 로컬 디스크입니다. 데이터를 고정 너비 형식의 CSV 파일로 작성할 수 있지만, TensorFlow 데이터세트에 권장되는 파일 형식은 TFRecord 형식입니다. 이것은 `tf.train.Example` 프로토콜 버퍼 메시지로 구성된 간단한 레코드 지향 바이너리 형식입니다.

각 `tf.train.Example` 레코드에는 하나 이상의 특성이 포함되어 있습니다. 이러한 특성은 훈련용으로 모델에 공급될 때 텐서로 변환됩니다. 다음 코드는 변환된 데이터세트를 지정된 위치의 TFRecord 파일에 기록합니다.

```py{:.devsite-disable-click-to-copy}
def write_tfrecords(transformed_dataset, location, step):
    from tfx_bsl.coders import example_coder

    transformed_data, transformed_metadata = transformed_dataset
    (
        transformed_data
        | '{} - Encode Transformed Data'.format(step) >> beam.FlatMapTuple(
                            lambda batch, _: example_coder.RecordBatchToExamples(batch))
        | '{} - Write Transformed Data'.format(step) >> beam.io.WriteToTFRecord(
                            file_path_prefix=os.path.join(location,'{}'.format(step)),
                            file_name_suffix='.tfrecords')
    )
```

### 평가 데이터 읽기, 변환하기 및 쓰기

훈련 데이터를 변환하고 `transform_fn` 그래프를 생성하면 이를 사용하여 평가 데이터를 변환할 수 있습니다. 먼저 [BigQuery에서 원시 훈련 데이터 읽기](#read-raw-training-data-from-bigquery){: track-type="solution" track-name="internalLink" track-metadata-position="body" }에 설명된 `read_from_bq` 함수를 사용하고 `step` 매개변수에 `eval` 값을 전달하여 BigQuery에서 평가 데이터를 읽고 정정합니다. 그런 다음 다음 코드를 사용하여 원시 평가 데이터세트(`raw_dataset`)를 예상 변환 형식(`transformed_dataset`)으로 변환합니다.

```py{:.devsite-disable-click-to-copy}
def transform(raw_dataset, transform_fn, step):

    transformed_dataset = (
        (raw_dataset, transform_fn)
        | '{} - Transform'.format(step) >> tft_beam.TransformDataset(output_record_batches=True)
    )

    return transformed_dataset
```

평가 데이터를 변환할 때는 `transform_fn` 그래프의 로직과 훈련 데이터의 분석 단계에서 계산된 통계를 모두 사용하여 인스턴스 수준 연산만 적용합니다. 즉, 평가 데이터에서 숫자 특성의 z-스코어 정규화에 대한 평균 및 분산과 같은 새로운 통계를 계산하기 위해 평가 데이터를 풀패스 방식으로 분석하지 않습니다. 대신, 훈련 데이터에서 계산한 통계를 사용하여 인스턴스 수준에서 평가 데이터를 변환합니다.

따라서 훈련 데이터의 컨텍스트에서 `AnalyzeAndTransform` 메서드를 사용하여 통계를 계산하고 데이터를 변환합니다. 동시에 평가 데이터를 변환하는 컨텍스트에서 `TransformDataset` 메서드를 사용하여 훈련 데이터에서 계산된 통계를 사용하여 데이터만 변환할 수 있습니다.

그런 다음 훈련 프로세스를 진행하며 TensorFlow 모델을 평가할 수 있도록 데이터를 TFRecord 형식으로 싱크(실행기에 따라 클라우드 스토리지 또는 로컬 디스크)에 기록합니다. 이를 위해 [변환된 훈련 데이터 기록하기](#step_3_write_transformed_training_data){: track-type="solution" track-name="internalLink" track-metadata-position="body" }에서 논의된 `write_tfrecords` 함수를 사용합니다. 다음 다이어그램, 그림 3은 훈련 데이터의 분석 단계에서 생성된 `transform_fn` 그래프가 평가 데이터를 변환하는 방식을 보여줍니다.

<figure id="transform-eval-data-using-transform-fn">
  <img src="images/data-preprocessing-for-ml-with-tf-transform-transforming-eval-data-using-transform_fn.svg"
    alt="Transforming evaluation data using the transform_fn graph.">
  <figcaption><b>Figure 3.</b> Transforming evaluation data using the <code>transform_fn</code> graph.</figcaption>
</figure>

### 그래프 저장하기

`tf.Transform` 전처리 파이프라인의 마지막 단계는 훈련 데이터에 대한 분석 단계에서 생성된 `transform_fn` 그래프를 포함하는 아티팩트를 저장하는 것입니다. 아티팩트를 저장하는 코드는 다음 `write_transform_artefacts` 함수에 나와 있습니다.

```py{:.devsite-disable-click-to-copy}
def write_transform_artefacts(transform_fn, location):

    (
        transform_fn
        | 'Write Transform Artifacts' >> transform_fn_io.WriteTransformFn(location)
    )
```

이러한 아티팩트는 나중에 모델 훈련 및 제공을 위한 내보내기에 사용됩니다. 다음 섹션에 표시된 것처럼 다음 아티팩트도 생성됩니다.

- `saved_model.pb`: 원시 데이터 포인트를 변환된 형식으로 변환하기 위해 모델 제공 인터페이스에 첨부될 변환 로직(`transform_fn` 그래프)을 포함하는 TensorFlow 그래프를 나타냅니다.
- `variables`: 훈련 데이터의 분석 단계에서 계산된 통계를 포함하며 `saved_model.pb` 아티팩트의 변환 로직에 사용됩니다.
- `assets`: `compute_and_apply_vocabulary` 메서드로 처리된 각 범주형 특성에 대해 하나씩 어휘 파일을 포함하며, 입력된 원시 명목값을 숫자 인덱스로 변환하는 데 사용할 수 있습니다.
- `transformed_metadata`: 변환된 데이터의 스키마를 설명하는 `schema.json` 파일이 포함된 디렉터리입니다.

## Dataflow에서 파이프라인 실행하기{:#run_the_pipeline_in_dataflow}

`tf.Transform` 파이프라인을 정의한 후 Dataflow를 사용하여 파이프라인을 실행합니다. 다음 다이어그램, 그림 4는 예제에 설명된 `tf.Transform` 파이프라인의 Dataflow 실행 그래프를 보여줍니다.

<figure id="dataflow-execution-graph">
  <img src="images/data-preprocessing-for-ml-with-tf-transform-dataflow-execution-graph.png"
    alt="Dataflow execution graph of the tf.Transform pipeline." class="screenshot">
  <figcaption><b>Figure 4.</b> Dataflow execution graph
     of the <code>tf.Transform</code> pipeline.</figcaption>
</figure>

Dataflow  파이프라인을 실행하여 훈련 및 평가 데이터를 전처리한 후, 노트북의 마지막 셀을 실행하여 클라우드 스토리지에서 생성된 개체를 탐색할 수 있습니다. 이 섹션의 코드 스니펫은 결과를 보여 주며, 여기서 <var><code>YOUR_BUCKET_NAME</code></var>은 클라우드 스토리지 버킷의 이름입니다.

TFRecord 형식으로 변환된 훈련 및 평가 데이터는 다음 위치에 저장됩니다.

```none{:.devsite-disable-click-to-copy}
gs://YOUR_BUCKET_NAME/babyweight_tft/transformed
```

변환 아티팩트는 다음 위치에서 생성됩니다.

```none{:.devsite-disable-click-to-copy}
gs://YOUR_BUCKET_NAME/babyweight_tft/transform
```

다음 목록은 생성된 데이터 개체와 아티팩트를 보여주는 파이프라인의 출력입니다.

```none{:.devsite-disable-click-to-copy}
transformed data:
gs://YOUR_BUCKET_NAME/babyweight_tft/transformed/eval-00000-of-00001.tfrecords
gs://YOUR_BUCKET_NAME/babyweight_tft/transformed/train-00000-of-00002.tfrecords
gs://YOUR_BUCKET_NAME/babyweight_tft/transformed/train-00001-of-00002.tfrecords

transformed metadata:
gs://YOUR_BUCKET_NAME/babyweight_tft/transform/transformed_metadata/
gs://YOUR_BUCKET_NAME/babyweight_tft/transform/transformed_metadata/asset_map
gs://YOUR_BUCKET_NAME/babyweight_tft/transform/transformed_metadata/schema.pbtxt

transform artefact:
gs://YOUR_BUCKET_NAME/babyweight_tft/transform/transform_fn/
gs://YOUR_BUCKET_NAME/babyweight_tft/transform/transform_fn/saved_model.pb
gs://YOUR_BUCKET_NAME/babyweight_tft/transform/transform_fn/assets/
gs://YOUR_BUCKET_NAME/babyweight_tft/transform/transform_fn/variables/

transform assets:
gs://YOUR_BUCKET_NAME/babyweight_tft/transform/transform_fn/assets/
gs://YOUR_BUCKET_NAME/babyweight_tft/transform/transform_fn/assets/is_male
gs://YOUR_BUCKET_NAME/babyweight_tft/transform/transform_fn/assets/is_multiple
gs://YOUR_BUCKET_NAME/babyweight_tft/transform/transform_fn/assets/mother_race
```

## TensorFlow 모델 구현하기{: id="implementing_the_tensorflow_model"}

이 섹션과 다음 섹션인 [예측용 모델 훈련하고 사용하기](#train_and_use_the_model_for_predictions){: track-type="solution" track-name="internalLink" track-metadata-position="body" }는 노트북 2에 대한 개요와 컨텍스트를 제공합니다. 이 노트북은 아기 몸무게를 예측하는 ML 모델 예제를 제공합니다. 이 예제에서는 Keras API를 사용해 TensorFlow 모델을 구현합니다. 이 모델은 앞에서 설명한 `tf.Transform` 전처리 파이프라인에서 생성된 데이터와 아티팩트를 사용합니다.

### 노트북 2 실행하기

1. JupyterLab 인터페이스에서 **파일 &gt; 경로에서 열기**를 클릭한 후 다음 경로를 입력합니다.

    ```sh
    training-data-analyst/blogs/babyweight_tft/babyweight_tft_keras_02.ipynb
    ```

2. **편집 &gt; 모든 출력 지우기**를 클릭합니다.

3. **필수 패키지 설치** 섹션에서 첫 번째 셀을 실행하여 `pip install tensorflow-transform` 명령어를 실행합니다.

    출력의 마지막 부분은 다음과 같습니다.

    ```none{:.devsite-disable-click-to-copy}
    Successfully installed ...
    Note: you may need to restart the kernel to use updated packages.
    ```

    출력에서 종속성 오류를 무시할 수 있습니다.

4. **커널** 메뉴에서 **커널 재시작**을 선택합니다.

5. **설치된 패키지 확인** 및 **setup.py를 만들어 Dataflow 컨테이너에 패키지 설치하기** 섹션의 셀을 실행합니다.

6. **전역 플래그 설정** 섹션에서 `PROJECT` 및 `BUCKET` 옆에 있는 <code>your-project</code>를 Cloud 프로젝트 ID로 교체한 후 셀을 실행합니다.

7. 노트북의 마지막 셀까지 나머지 모든 셀을 실행합니다. 각 셀에서 수행할 작업에 대한 자세한 내용은 노트북의 지침을 참조하세요.

### 모델 만들기 개요

모델을 만드는 단계는 다음과 같습니다.

1. `transformed_metadata` 디렉터리에 저장된 스키마 정보를 사용하여 특성 열을 만듭니다.
2. 특성 열을 모델에 대한 입력으로 사용하여 Keras API로 와이드 및 딥 모델을 만듭니다.
3. 변환 아티팩트를 사용하여 훈련 및 평가 데이터를 읽고 구문 분석하는 `tfrecords_input_fn` 함수를 만듭니다.
4. 모델을 훈련하고 평가합니다.
5. `transform_fn` 그래프가 첨부된 `serving_fn` 함수를 정의하여 훈련한 모델을 내보냅니다.
6. 내보낸 모델을 [`saved_model_cli`](https://www.tensorflow.org/guide/saved_model) 도구를 사용하여 검사합니다.
7. 내보낸 모델을 예측에 사용합니다.

이 문서는 모델을 빌드하는 방법을 설명하지 않으므로 모델을 빌드하는 방식이나 훈련 방식에 대한 자세한 내용을 다루지 않습니다. 그러나 다음 섹션은 `tf.Transform` 프로세스로 생성하는 `transform_metadata` 디렉터리에 저장된 정보가 모델의 특성 열을 만드는 방식을 보여 줍니다. 또한 이 문서는 `tf.Transform` 프로세스로 생성하는 `transform_fn` 그래프가 모델을 제공할 수 있도록 내보내기를 수행할 때 `serving_fn` 함수에서 사용되는 방식도 보여 줍니다.

### 모델 훈련에서 생성된 변환 아티팩트 사용하기

TensorFlow 모델을 훈련할 때 이전 데이터 처리 단계에서 생성한 변환된 `train` 및 `eval` 객체를 사용합니다. 이러한 객체는 TFRecord 형식의 샤드 파일로 저장됩니다. 이전 단계에서 생성한 `transformed_metadata` 디렉터리의 스키마 정보는 훈련 및 평가를 위해 모델에 입력되는 데이터(`tf.train.Example` 객체)를 파싱할 때 유용할 수 있습니다.

#### 데이터 구문 분석하기

모델에 훈련 및 평가 데이터를 제공하기 위해 TFRecord 형식의 파일을 읽기 때문에 파일에 있는 각 `tf.train.Example` 객체를 구문 분석하여 특성 사전(텐서)을 만들어야 합니다. 이렇게 하면 모델 훈련 및 평가 인터페이스 역할을 하는 특성 열을 사용하여 특성이 모델 입력 레이어에 매핑할 수 있습니다. 데이터를 구문 분석하려면 이전 단계에서 생성된 아티팩트로 만든 `TFTransformOutput` 객체를 사용해야 합니다.

1. [그래프 저장](#save_the_graph){: track-type="solution" track-name="internalLink" track-metadata-position="body" } 섹션에 설명된 대로 이전 전처리 단계에서 생성 및 저장한 아티팩트로부터 `TFTransformOutput` 객체를 만듭니다.

    ```py
    tf_transform_output = tft.TFTransformOutput(TRANSFORM_ARTEFACTS_DIR)
    ```

2. `TFTransformOutput` 객체에서 `feature_spec` 객체를 추출합니다.

    ```py
    tf_transform_output.transformed_feature_spec()
    ```

3. `feature_spec` 객체를 사용하여 `tfrecords_input_fn` 함수에서와 같이 `tf.train.Example` 객체에 포함된 특성을 지정합니다.

    ```py
    def tfrecords_input_fn(files_name_pattern, batch_size=512):

        tf_transform_output = tft.TFTransformOutput(TRANSFORM_ARTEFACTS_DIR)
        TARGET_FEATURE_NAME = 'weight_pounds'

        batched_dataset = tf.data.experimental.make_batched_features_dataset(
            file_pattern=files_name_pattern,
            batch_size=batch_size,
            features=tf_transform_output.transformed_feature_spec(),
            reader=tf.data.TFRecordDataset,
            label_key=TARGET_FEATURE_NAME,
            shuffle=True).prefetch(tf.data.experimental.AUTOTUNE)

        return batched_dataset
    ```

#### 특성 열 만들기

이 파이프라인은 훈련 및 평가를 위해 모델에서 예상하는 변환된 데이터의 스키마를 설명하는 스키마 정보를 `transformed_metadata` 디렉터리에서 생성합니다. 스키마는 다음과 같은 특성 이름과 데이터 유형을 포함합니다.

- `gestation_weeks_scaled`(유형: `FLOAT`)
- `is_male_index`(유형: `INT`, is_categorical: `True`)
- `is_multiple_index`(유형: `INT`, is_categorical: `True`)
- `mother_age_bucketized`(유형: `INT`, is_categorical: `True`)
- `mother_age_log`(유형: `FLOAT`)
- `mother_age_normalized`(유형: `FLOAT`)
- `mother_race_index`(유형: `INT`, is_categorical: `True`)
- `weight_pounds`(유형: `FLOAT`)

이 정보를 확인하려면 다음 명령을 사용해야 합니다.

```sh
transformed_metadata = tft.TFTransformOutput(TRANSFORM_ARTEFACTS_DIR).transformed_metadata
transformed_metadata.schema
```

다음 코드는 특성 이름을 사용하여 특성 열을 만드는 방법을 보여줍니다.

```py
def create_wide_and_deep_feature_columns():

    deep_feature_columns = []
    wide_feature_columns = []
    inputs = {}
    categorical_columns = {}

    # Select features you've checked from the metadata
    # Categorical features are associated with the vocabulary size (starting from 0)
    numeric_features = ['mother_age_log', 'mother_age_normalized', 'gestation_weeks_scaled']
    categorical_features = [('is_male_index', 1), ('is_multiple_index', 1),
                            ('mother_age_bucketized', 4), ('mother_race_index', 10)]

    for feature in numeric_features:
        deep_feature_columns.append(tf.feature_column.numeric_column(feature))
        inputs[feature] = layers.Input(shape=(), name=feature, dtype='float32')

    for feature, vocab_size in categorical_features:
        categorical_columns[feature] = (
            tf.feature_column.categorical_column_with_identity(feature, num_buckets=vocab_size+1))
        wide_feature_columns.append(tf.feature_column.indicator_column(categorical_columns[feature]))
        inputs[feature] = layers.Input(shape=(), name=feature, dtype='int64')

    mother_race_X_mother_age_bucketized = tf.feature_column.crossed_column(
        [categorical_columns['mother_age_bucketized'],
         categorical_columns['mother_race_index']],  55)
    wide_feature_columns.append(tf.feature_column.indicator_column(mother_race_X_mother_age_bucketized))

    mother_race_X_mother_age_bucketized_embedded = tf.feature_column.embedding_column(
        mother_race_X_mother_age_bucketized, 5)
    deep_feature_columns.append(mother_race_X_mother_age_bucketized_embedded)

    return wide_feature_columns, deep_feature_columns, inputs
```

이 코드는 숫자 특성의 경우 `tf.feature_column.numeric_column` 열을 만들고, 범주형 특성의 경우 `tf.feature_column.categorical_column_with_identity` 열을 만듭니다.

이 시리즈의 첫 번째 부분의 [옵션 C: TensorFlow](/architecture/data-preprocessing-for-ml-with-tf-transform-pt1#option_c_tensorflow){: track-type="solution" track-name="internalLink" track-metadata-position="body" }에 설명된 대로 확장 특성 열을 만들 수도 있습니다. 이 시리즈에 사용된 예제에서는 `tf.feature_column.crossed_column`을 사용하여 `mother_race` 및 `mother_age_bucketized` 특성을 교차함으로써 새로운 특성인 `mother_race_X_mother_age_bucketized`를 만듭니다. 이 교차 특성의 저차원 밀도 표현은 `tf.feature_column.embedding_column` 특성 열을 사용하여 만듭니다.

다음 다이어그램, 그림 5는 변환된 데이터와 변환된 메타데이터가 TensorFlow 모델을 정의하고 훈련하기 위해 사용되는 방식을 보여줍니다.

<figure id="training-tf-with-transformed-data">
  <img src="images/data-preprocessing-for-ml-with-tf-transform-training-tf-model-with-transformed-data.svg"
    alt="Training the TensorFlow model with transformed data.">
  <figcaption><b>Figure 5.</b> Training the TensorFlow model with
    the transformed data.</figcaption>
</figure>

### 예측 서빙용 모델 내보내기

Keras API를 사용하여 TensorFlow 모델을 훈련한 후에는 훈련한 모델을 SavedModel 객체로 내보내어 예측용 새로운 데이터 포인트로 사용할 수 있습니다. 모델을 내보낼 때 인터페이스, 즉 서빙하는 동안 예상되는 입력 특징 스키마를 정의해야 합니다. 이 입력 특성 스키마는 다음 코드에 표시된 것처럼 `serving_fn` 함수에 정의되어 있습니다.

```py{:.devsite-disable-click-to-copy}
def export_serving_model(model, output_dir):

    tf_transform_output = tft.TFTransformOutput(TRANSFORM_ARTEFACTS_DIR)
    # The layer has to be saved to the model for Keras tracking purposes.
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serveing_fn(uid, is_male, mother_race, mother_age, plurality, gestation_weeks):
        features = {
            'is_male': is_male,
            'mother_race': mother_race,
            'mother_age': mother_age,
            'plurality': plurality,
            'gestation_weeks': gestation_weeks
        }
        transformed_features = model.tft_layer(features)
        outputs = model(transformed_features)
        # The prediction results have multiple elements in general.
        # But we need only the first element in our case.
        outputs = tf.map_fn(lambda item: item[0], outputs)

        return {'uid': uid, 'weight': outputs}

    concrete_serving_fn = serveing_fn.get_concrete_function(
        tf.TensorSpec(shape=[None], dtype=tf.string, name='uid'),
        tf.TensorSpec(shape=[None], dtype=tf.string, name='is_male'),
        tf.TensorSpec(shape=[None], dtype=tf.string, name='mother_race'),
        tf.TensorSpec(shape=[None], dtype=tf.float32, name='mother_age'),
        tf.TensorSpec(shape=[None], dtype=tf.float32, name='plurality'),
        tf.TensorSpec(shape=[None], dtype=tf.float32, name='gestation_weeks')
    )
    signatures = {'serving_default': concrete_serving_fn}

    model.save(output_dir, save_format='tf', signatures=signatures)
```

서빙하는 동안 모델은 데이터 포인트를 원시 형식(즉, 변환 전의 원시 특성)으로 사용할 수 있길 기대합니다. 따라서 `serving_fn` 함수는 원시 특성을 수신하고 `features` 객체에서 Python 사전으로 저장합니다. 그러나 앞서 설명한 것처럼 훈련된 모델은 변환된 스키마의 데이터 포인트를 기대합니다. 원시 특성을 모델 인터페이스에서 기대하는 `transformed_features` 객체로 변환하려면 다음 단계에 따라 저장된 `transform_fn` 그래프를 `features` 객체에 적용해야 합니다.

1. 이전 전처리 단계에서 생성하고 저장한 아티팩트에서 `TFTransformOutput` 객체를 만듭니다.

    ```py
    tf_transform_output = tft.TFTransformOutput(TRANSFORM_ARTEFACTS_DIR)
    ```

2. `TFTransformOutput` 객체에서 `TransformFeaturesLayer` 객체를 만듭니다.

    ```py
    model.tft_layer = tf_transform_output.transform_features_layer()
    ```

3. `TransformFeaturesLayer` 객체를 사용하여 `transform_fn` 그래프를 적용합니다.

    ```py
    transformed_features = model.tft_layer(features)
    ```

다음 다이어그램, 그림 6은 서빙을 위해 모델을 내보내는 마지막 단계를 보여줍니다.

<figure id="exporting-model-for-serving-with-transform_fn">
  <img src="images/data-preprocessing-for-ml-with-tf-transform-exporting-model-for-serving-with-transform_fn.svg"
    alt="Exporting the model for serving with the transform_fn graph attached.">
  <figcaption><b>Figure 6.</b> Exporting the model for serving with the
    <code>transform_fn</code> graph attached.</figcaption>
</figure>

## 예측을 위해 모델 훈련하고 사용하기

노트북의 셀을 실행하여 로컬에서 모델을 훈련할 수 있습니다. Vertex AI Training을 사용하여 대규모로 코드를 패키징하고 모델을 훈련하는 방법에 대한 예제는 Google Cloud [cloudml-samples](https://github.com/GoogleCloudPlatform/cloudml-samples){: .external } GitHub 리포지토리를 참조하세요.

`saved_model_cli` 도구를 사용하여 내보낸 SavedModel 객체를 검사하면 다음 예제와 같이 서명 정의 `signature_def`의 `inputs` 요소에 원시 특성이 포함된 것을 확인할 수 있습니다.

```py{:.devsite-disable-click-to-copy}
signature_def['serving_default']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['gestation_weeks'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1)
        name: serving_default_gestation_weeks:0
    inputs['is_male'] tensor_info:
        dtype: DT_STRING
        shape: (-1)
        name: serving_default_is_male:0
    inputs['mother_age'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1)
        name: serving_default_mother_age:0
    inputs['mother_race'] tensor_info:
        dtype: DT_STRING
        shape: (-1)
        name: serving_default_mother_race:0
    inputs['plurality'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1)
        name: serving_default_plurality:0
    inputs['uid'] tensor_info:
        dtype: DT_STRING
        shape: (-1)
        name: serving_default_uid:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['uid'] tensor_info:
        dtype: DT_STRING
        shape: (-1)
        name: StatefulPartitionedCall_6:0
    outputs['weight'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1)
        name: StatefulPartitionedCall_6:1
  Method name is: tensorflow/serving/predict
```

노트북의 나머지 셀은 내보낸 모델을 로컬 예측에 사용하는 방법과 Vertex AI Prediction을 사용하여 모델을 마이크로서비스로 배포하는 방법을 보여 줍니다. 두 경우 모두 입력(샘플) 데이터 포인트가 원시 스키마에 있다는 점을 강조하는 것이 중요합니다.

## 정리하기

이 튜토리얼에서 사용한 리소스에 대해 Google 클라우드 계정에 추가 요금이 발생되지 않도록 하려면 리소스가 포함된 프로젝트를 삭제해야 합니다.

### 프로젝트 삭제하기

  <aside class="caution"><strong>주의</strong>: 프로젝트를 삭제하면 다음과 같이 될 수 있습니다.    <ul>       <li>         <strong>프로젝트의 모든 항목이 삭제됩니다.</strong> 이 튜토리얼에서 기존 프로젝트를 사용한 경우, 프로젝트를 삭제하면 프로젝트에서 수행한 다른 작업도 삭제됩니다.       </li>       <li>         <strong>사용자 정의 프로젝트 ID가 손실됩니다.</strong>         이 프로젝트를 만들 때 나중에 사용하려는 사용자 정의 프로젝트 ID를 만들었을 수 있습니다. 프로젝트 ID를 사용하는 <code translate="no" dir="ltr">appspot.com</code>과 같은 URL을 유지하려면 전체 프로젝트를 삭제하는 대신 프로젝트 내부에서 선택한 리소스만 삭제하세요.       </li>     </ul>     <p>       여러 튜토리얼과 빠른 시작을 탐색하려는 경우 프로젝트를 재사용하면 프로젝트 할당량 한도를 초과하지 않을 수 있습니다.     </p></aside>


1. Google Cloud 콘솔에서 **리소스 관리** 페이지로 이동합니다.

    [리소스 관리로 이동](https://console.cloud.google.com/iam-admin/projects){: class="button button-primary" target="console" track-type="solution" track-name="consoleLink" track-metadata-position="body" }

2. 프로젝트 목록에서 삭제할 프로젝트를 선택한 다음 **삭제**를 클릭합니다.

3. 대화 상자에서 프로젝트 ID를 입력한 다음 **종료**를 클릭하여 프로젝트를 삭제합니다.

## 다음 내용

- Google Cloud에서 머신러닝을 하기 위한 데이터 전처리의 개념, 과제 및 옵션에 대해 알아보려면 이 시리즈의 첫 번째 문서인 [ML을 위한 데이터 전처리: 옵션 및 권장 사항](../guide/tft_bestpractices)을 참조하세요.
- Dataflow에서 tf.Transform 파이프라인을 구현하고 패키징하고 실행하는 방법에 대한 자세한 내용은 [인구 조사 데이터세트로 소득 예측하기](https://github.com/GoogleCloudPlatform/cloudml-samples/tree/master/census/tftransformestimator){: .external } 샘플을 참조하세요.
- [Google Cloud의 TensorFlow](https://www.coursera.org/specializations/machine-learning-tensorflow-gcp){: .external }를 사용하여 ML 진행하기에 대한 Coursera 전문 과정을 수강해보세요.
- [ML 규칙](https://developers.google.com/machine-learning/guides/rules-of-ml/){: .external }에서 ML 엔지니어링 모범 사례에 대해 알아보세요.
- 더 많은 참조 아키텍처, 다이어그램 및 모범 사례를 보려면 [클라우드 아키텍처 센터](https://cloud.google.com/architecture)를 살펴보세요.
