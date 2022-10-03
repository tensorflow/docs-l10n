# 로컬에서 TFX 파이프라인 빌드하기

TFX를 사용하면 머신러닝(ML) 워크플로를 파이프라인으로 쉽게 오케스트레이션하여 다음과 같은 결과를 얻을 수 있습니다.

- Automate your ML process, which lets you regularly retrain, evaluate, and deploy your model.
- Create ML pipelines which include deep analysis of model performance and validation of newly trained models to ensure performance and reliability.
- Monitor training data for anomalies and eliminate training-serving skew
- Increase the velocity of experimentation by running a pipeline with different sets of hyperparameters.

일반적인 파이프라인 개발 프로세스는 운영에 배포되기 전에 데이터 분석 및 구성 요소 설정과 함께 로컬 시스템에서 시작됩니다. 이 가이드에서는 로컬에서 파이프라인을 구축하는 두 가지 방법을 설명합니다.

- Customize a TFX pipeline template to fit the needs of your ML workflow. TFX pipeline templates are prebuilt workflows that demonstrate best practices using the TFX standard components.
- Build a pipeline using TFX. In this use case, you define a pipeline without starting from a template.

파이프라인을 개발하는 동안 `LocalDagRunner`을 파이프라인과 함께 실행할 수 있습니다. 파이프라인 구성 요소를 잘 정의하고 테스트한 후에는 Kubeflow 혹은 Airflow와 같은 운영 등급 오케스트레이터를 사용할 수 있습니다.

## 시작하기 전에

TFX는 Python 패키지이므로 가상 환경이나 Docker 컨테이너와 같은 Python 개발 환경을 설정해야 합니다. 그런 다음 다음을 수행합니다.

```bash
pip install tfx
```

If you are new to TFX pipelines, [learn more about the core concepts for TFX pipelines](understanding_tfx_pipelines) before continuing.

## Build a pipeline using a template

TFX 파이프라인 템플릿을 사용하면 사용 사례에 맞게 사용자 정의할 수 있는 사전 빌드된 파이프라인 정의 세트를 제공하기에 파이프라인 개발을 더 쉽게 시작할 수 있습니다.

The following sections describe how to create a copy of a template and customize it to meet your needs.

### Create a copy of the pipeline template

1. 사용 가능한 TFX 파이프라인 템플릿을 확인합니다.

    <pre class="devsite-click-to-copy devsite-terminal">    tfx template list
        </pre>

2. 목록에서 템플릿을 선택합니다.

    <pre class="devsite-click-to-copy devsite-terminal">    tfx template copy --model=&lt;var&gt;template&lt;/var&gt; --pipeline_name=&lt;var&gt;pipeline-name&lt;/var&gt; \
        --destination_path=&lt;var&gt;destination-path&lt;/var&gt;
        </pre>

    Replace the following:

    - <var>template</var>: The name of the template you want to copy.
    - <var>pipeline-name</var>: The name of the pipeline to create.
    - <var>destination-path</var>: The path to copy the template into.

    Learn more about the [`tfx template copy` command](cli#copy).

3. A copy of the pipeline template has been created at the path you specified.

참고: 이 가이드의 나머지 부분에서는 여러분이 `penguin` 템플릿을 선택했다고 가정합니다.

### Explore the pipeline template

이 섹션에서는 템플릿으로 만든 스캐폴딩을 개괄적으로 설명합니다.

1. 파이프라인의 루트 디렉터리에 복사된 디렉터리 및 파일 탐색

    - 다음을 사용하는 **파이프라인** 디렉터리

        - `pipeline.py` - 파이프라인을 정의하고 사용 중인 구성 요소를 나열
        - `configs.py` - 데이터의 출처 또는 사용 중인 오케스트레이터와 같은 구성 세부 정보 보유

    - **데이터** 디렉터리

        - 여기에는 일반적으로 `ExampleGen`의 기본 소스인 `data.csv` 파일이 포함됩니다. `configs.py`에서 데이터 소스를 변경할 수 있습니다.

    - 코드 및 모델 구현을 전처리하는 **models** 디렉터리

    - 템플릿이 로컬 환경 및 Kubeflow용 DAG 실행기를 복사

    - 머신러닝 메타데이터로 데이터와 아티팩트를 탐색할 수 있도록 일부 템플릿은 Python 노트북도 포함

2. 파이프라인 디렉터리에서 다음 명령을 실행합니다.

    <pre class="devsite-click-to-copy devsite-terminal">    tfx pipeline create --pipeline_path local_runner.py
        </pre>

    <pre class="devsite-click-to-copy devsite-terminal">    tfx run create --pipeline_name &lt;var&gt;pipeline_name&lt;/var&gt;
        </pre>

    이 명령은 `LocalDagRunner`으로 파이프라인 실행을 생성하여 파이프라인에 다음 디렉터리를 추가합니다.

    - 로컬에서 사용되는 ML 메타데이터 저장소가 포함된 **tfx_metadata** 디렉터리
    - 파이프라인의 파일 출력을 포함하는 **tfx_pipeline_output** 디렉터리

    참고: `LocalDagRunner`은 TFX에서 지원되는 여러 오케스트레이터 중 하나입니다. 이는 더 작은 데이터세트로 더 빠른 반복을 위해 로컬에서 파이프라인을 실행하는 데 특히 적합합니다.`LocalDagRunner`은 단일 머신에서 실행되기 때문에 프로덕션 용도로 적합하지 않을 수 있으며, 시스템을 사용할 수 없게 되었을 때 작업을 손실할 위험이 더 큽니다. TFX는 Apache Beam, Apache Airflow 및 Kubeflow Pipeline과 같은 오케스트레이터도 지원합니다. 다른 오케스트레이터와 함께 TFX를 사용하는 경우, 해당 오케스트레이터에 적합한 DAG 실행기를 사용하세요.

    참고: 이 내용을 작성하는 현재에 `LocalDagRunner`는 `penguin` 템플릿에서 사용되는 반면 `taxi` 템플릿은 Apache Beam을 사용합니다. `taxi` 템플릿의 구성 파일은 Beam을 사용하도록 설정되어 있으며 CLI 명령어는 동일합니다.

3. 파이프라인의 `pipeline/configs.py` 파일을 열고 콘텐츠를 검토합니다. 이 스크립트는 파이프라인 및 구성 요소 함수에서 사용하는 구성 옵션을 정의합니다. 여기에서 데이터 소스의 위치나 실행 훈련 단계 수와 같은 항목을 지정할 수 있습니다.

4. 파이프라인의 `pipeline/pipeline.py` 파일을 열고 내용을 검토합니다. 이 스크립트는 TFX 파이프라인을 생성합니다. 처음에는 파이프라인에 `ExampleGen` 구성 요소만 포함됩니다.

    - 파이프라인에 단계를 더 추가하려면 `pipeline.py`의 **TODO** 주석에 있는 지침을 따르세요

5. `local_runner.py` 파일을 열고 내용을 검토합니다. 이 스크립트는 파이프라인 실행을 생성하고 `data_path` 및 `preprocessing_fn`과 같은 실행 *매개변수를* 지정합니다.

6. 템플릿으로 생성한 스캐폴딩을 검토하고 `LocalDagRunner`을 사용하여 파이프라인 실행을 생성했습니다. 이제, 요구 사항에 맞게 템플릿을 사용자 정의합니다.

### Customize your pipeline

이 섹션에서는 템플릿 사용자 정의를 시작하는 방법을 개괄적으로 설명합니다.

1. Design your pipeline. The scaffolding that a template provides helps you implement a pipeline for tabular data using the TFX standard components. If you are moving an existing ML workflow into a pipeline, you may need to revise your code to make full use of [TFX standard components](index#tfx_standard_components). You may also need to create [custom components](understanding_custom_components) that implement features which are unique to your workflow or that are not yet supported by TFX standard components.

2. Once you have designed your pipeline, iteratively customize the pipeline using the following process. Start from the component that ingests data into your pipeline, which is usually the `ExampleGen` component.

    1. Customize the pipeline or a component to fit your use case. These customizations may include changes like:

        - Changing pipeline parameters.
        - Adding components to the pipeline or removing them.
        - Replacing the data input source. This data source can either be a file or queries into services such as BigQuery.
        - Changing a component's configuration in the pipeline.
        - Changing a component's customization function.

    2. `local_runner.py` 스크립트를 사용하거나, 다른 오케스트레이터를 사용하는 경우 다른 적절한 DAG 실행기를 사용하여 구성 요소를 로컬에서 실행합니다. 스크립트가 실패하면 오류를 디버그하고 스크립트를 다시 실행합니다.

    3. Once this customization is working, move on to the next customization.

3. Working iteratively, you can customize each step in the template workflow to meet your needs.

## Build a custom pipeline

Use the following instructions to learn more about building a custom pipeline without using a template.

1. Design your pipeline. The TFX standard components provide proven functionality to help you implement a complete ML workflow. If you are moving an existing ML workflow into a pipeline, you may need to revise your code to make full use of TFX standard components. You may also need to create [custom components](understanding_custom_components) that implement features such as data augmentation.

    - Learn more about [standard TFX components](index#tfx_standard_components).
    - Learn more about [custom components](understanding_custom_components).

2. Create a script file to define your pipeline using the following example. This guide refers to this file as `my_pipeline.py`.

    <pre class="devsite-click-to-copy prettyprint">    import os
        from typing import Optional, Text, List
        from absl import logging
        from ml_metadata.proto import metadata_store_pb2
        import tfx.v1 as tfx

        PIPELINE_NAME = 'my_pipeline'
        PIPELINE_ROOT = os.path.join('.', 'my_pipeline_output')
        METADATA_PATH = os.path.join('.', 'tfx_metadata', PIPELINE_NAME, 'metadata.db')
        ENABLE_CACHE = True

        def create_pipeline(
          pipeline_name: Text,
          pipeline_root:Text,
          enable_cache: bool,
          metadata_connection_config: Optional[
            metadata_store_pb2.ConnectionConfig] = None,
          beam_pipeline_args: Optional[List[Text]] = None
        ):
          components = []

          return tfx.dsl.Pipeline(
                pipeline_name=pipeline_name,
                pipeline_root=pipeline_root,
                components=components,
                enable_cache=enable_cache,
                metadata_connection_config=metadata_connection_config,
                beam_pipeline_args=beam_pipeline_args, &lt;!-- needed? --&gt;
            )

        def run_pipeline():
          my_pipeline = create_pipeline(
              pipeline_name=PIPELINE_NAME,
              pipeline_root=PIPELINE_ROOT,
              enable_cache=ENABLE_CACHE,
              metadata_connection_config=tfx.orchestration.metadata.sqlite_metadata_connection_config(METADATA_PATH)
              )

          tfx.orchestration.LocalDagRunner().run(my_pipeline)

        if __name__ == '__main__':
          logging.set_verbosity(logging.INFO)
          run_pipeline()
        </pre>

    다음 단계에서는 `create_pipeline`에서 파이프라인을 정의하고 로컬 실행기를 사용하여 파이프라인을 로컬로 실행합니다.

    Iteratively build your pipeline using the following process.

    1. Customize the pipeline or a component to fit your use case. These customizations may include changes like:

        - Changing pipeline parameters.
        - Adding components to the pipeline or removing them.
        - Replacing a data input file.
        - Changing a component's configuration in the pipeline.
        - Changing a component's customization function.

    2. 로컬 실행기를 사용하거나 스크립트를 직접 실행하여 구성 요소를 로컬에서 실행합니다. 스크립트가 실패하면 오류를 디버그하고 스크립트를 다시 실행합니다.

    3. Once this customization is working, move on to the next customization.

    Start from the first node in your pipeline's workflow, typically the first node ingests data into your pipeline.

3. 워크플로의 첫 번째 노드를 파이프라인에 추가합니다. 이 예에서 파이프라인은 `ExampleGen` 표준 구성 요소를 사용하여 `./data`의 디렉터리에서 CSV를 로드합니다.

    <pre class="devsite-click-to-copy prettyprint">    from tfx.components import CsvExampleGen

        DATA_PATH = os.path.join('.', 'data')

        def create_pipeline(
          pipeline_name: Text,
          pipeline_root:Text,
          data_path: Text,
          enable_cache: bool,
          metadata_connection_config: Optional[
            metadata_store_pb2.ConnectionConfig] = None,
          beam_pipeline_args: Optional[List[Text]] = None
        ):
          components = []

          example_gen = tfx.components.CsvExampleGen(input_base=data_path)
          components.append(example_gen)

          return tfx.dsl.Pipeline(
                pipeline_name=pipeline_name,
                pipeline_root=pipeline_root,
                components=components,
                enable_cache=enable_cache,
                metadata_connection_config=metadata_connection_config,
                beam_pipeline_args=beam_pipeline_args, &lt;!-- needed? --&gt;
            )

        def run_pipeline():
          my_pipeline = create_pipeline(
            pipeline_name=PIPELINE_NAME,
            pipeline_root=PIPELINE_ROOT,
            data_path=DATA_PATH,
            enable_cache=ENABLE_CACHE,
            metadata_connection_config=tfx.orchestration.metadata.sqlite_metadata_connection_config(METADATA_PATH)
            )

          tfx.orchestration.LocalDagRunner().run(my_pipeline)
        </pre>

    `CsvExampleGen`은 지정한 데이터 경로에 있는 CSV의 데이터를 사용하여 직렬화된 예시 레코드를 생성합니다. 데이터 루트로 `CsvExampleGen` 구성 요소의 `input_base` 매개변수를 설정합니다.

4. `my_pipeline.py`와 동일한 디렉터리에 `data` 디렉터리를 만듭니다. `data` 디렉터리에 작은 CSV 파일을 추가합니다.

5. 다음 명령을 사용하여 `my_pipeline.py` 스크립트를 실행합니다.

    <pre class="devsite-click-to-copy devsite-terminal">    python my_pipeline.py
        </pre>

    The result should be something like the following:

    <pre>    INFO:absl:Component CsvExampleGen depends on [].
        INFO:absl:Component CsvExampleGen is scheduled.
        INFO:absl:Component CsvExampleGen is running.
        INFO:absl:Running driver for CsvExampleGen
        INFO:absl:MetadataStore with DB connection initialized
        INFO:absl:Running executor for CsvExampleGen
        INFO:absl:Generating examples.
        INFO:absl:Using 1 process(es) for Local pipeline execution.
        INFO:absl:Processing input csv data ./data/* to TFExample.
        WARNING:root:Couldn't find python-snappy so the implementation of _TFRecordUtil._masked_crc32c is not as fast as it could be.
        INFO:absl:Examples generated.
        INFO:absl:Running publisher for CsvExampleGen
        INFO:absl:MetadataStore with DB connection initialized
        INFO:absl:Component CsvExampleGen is finished.
        </pre>

6. Continue to iteratively add components to your pipeline.
