## 스텁 실행기를 사용하여 파이프라인 테스트하기

### 시작하기

**이 튜토리얼을 진행하려면 *6단계*까지 [template.ipynb](https://github.com/tensorflow/tfx/blob/master/docs/tutorials/tfx/template.ipynb) 튜토리얼을 완료해야 합니다.**

이 문서는 골든 테스트 데이터로 가짜 아티팩트를 생성하는 `BaseStubExecuctor`를 사용하여 TensorFlow Extended(TFX) 파이프라인을 테스트하기 위한 지침을 제공합니다. 이 지침은 사용자가 실제 실행기를 실행하는 시간을 절약할 수 있도록 테스트하고 싶지 않은 실행기를 교체하려고 할 때 필요합니다. 스텁 실행기는 `tfx.experimental.pipeline_testing.base_stub_executor` 아래에 TFX Python 패키지와 함께 제공됩니다.

이 튜토리얼은 `template.ipynb` 튜토리얼의 확장 역할을 하므로 시카고시에서 공개한 [Taxi Trips 데이터세트](https://data.cityofchicago.org/Transportation/Taxi-Trips/wrvz-psew)도 사용하게 됩니다. 스텁 실행기를 사용하기 전에 구성 요소를 수정하는 것이 좋습니다.

### 1. Google Cloud Storage에 파이프라인 출력 기록

스텁 실행기가 기록된 출력에서 아티팩트를 복사할 수 있도록 먼저 파이프라인 출력을 기록해야 합니다.

이 튜토리얼에서는 `template.ipynb`를 6단계까지 완료했다고 가정하므로 성공적인 파이프라인 실행이 [MLMD](https://www.tensorflow.org/tfx/guide/mlmd)에 저장되어 있어야 합니다. MLMD의 실행 정보는 gRPC 서버를 사용하여 확인할 수 있습니다.

터미널을 열고 다음 명령을 실행합니다.

1. 적절한 자격 증명으로 kubeconfig 파일을 생성합니다. `bash gcloud container clusters get-credentials $cluster_name --zone $compute_zone --project $gcp_project_id` `$compute_zone`은 gcp 엔진에 대한 영역이고 `$gcp_project_id`는 GCP 프로젝트의 프로젝트 ID입니다.

2. MLMD 연결을 위한 포트 전달 설정: `bash nohup kubectl port-forward deployment/metadata-grpc-deployment -n $namespace $port:8080 &` `$namespace`는 클러스터 네임스페이스이고 `$port`는 포트 전달에 사용되는 미사용 포트입니다.

3. tfx GitHub 리포지토리를 복제합니다. tfx 디렉터리 내에서 다음 명령을 실행합니다.

```bash
python tfx/experimental/pipeline_testing/pipeline_recorder.py \
--output_dir=gs://<gcp_project_id>-kubeflowpipelines-default/testdata \
--host=$host \
--port=$port \
--pipeline_name=$pipeline_name
```

`$output_dir`은 파이프라인 출력이 기록될 Google Cloud Storage의 경로로 설정되어야 하므로 `<gcp_project_id>`를 GCP 프로젝트 ID로 바꿔야 합니다.

`$host` 및 `$port`는 MLMD에 연결할 메타데이터 grpc 서버의 호스트 이름 및 포트입니다. `$port`는 포트 전달에 사용한 포트 번호로 설정해야 하며 호스트 이름에 "localhost"를 설정할 수 있습니다.

`template.ipynb` 튜토리얼에서는 파이프라인 이름이 기본적으로 "my_pipeline"으로 설정되므로 `pipeline_name="my_pipeline"`을 설정합니다. 템플릿 튜토리얼을 실행할 때 파이프라인 이름을 수정한 경우 그에 따라 `--pipeline_name`을 수정해야 합니다.

### 2. Kubeflow DAG 러너에서 스텁 실행기 사용

우선, `tfx template copy` CLI 명령을 사용하여 미리 정의된 템플릿이 프로젝트 디렉터리에 복사되었는지 확인합니다. 복사된 소스 파일에서 다음 두 개의 파일을 편집해야 합니다.

1. kubeflow_dag_runner.py가 있는 디렉터리에 `stub_component_launcher.py`라는 파일을 생성하고 다음 내용을 넣습니다.

    ```python
    from tfx.experimental.pipeline_testing import base_stub_component_launcher
    from pipeline import configs

    class StubComponentLauncher(
        base_stub_component_launcher.BaseStubComponentLauncher):
      pass

    # GCS directory where KFP outputs are recorded
    test_data_dir = "gs://{}/testdata".format(configs.GCS_BUCKET_NAME)
    # TODO: customize self.test_component_ids to test components, replacing other
    # component executors with a BaseStubExecutor.
    test_component_ids = ['Trainer']
    StubComponentLauncher.initialize(
        test_data_dir=test_data_dir,
        test_component_ids=test_component_ids)
    ```

    참고: 이 스텁 구성 요소 론처는 `kubeflow_dag_runner.py` 내에서 정의할 수 없는데, 론처 클래스를 모듈 경로에서 가져오기 때문입니다.

2. 구성 요소 ID를 테스트할 구성 요소 ID 목록으로 설정합니다(즉, 다른 구성 요소의 실행기가 BaseStubExecutor로 대체됨).

3. `kubeflow_dag_runner.py`를 엽니다. 방금 추가한 `StubComponentLauncher` 클래스를 사용하려면 맨 위에 다음 import 문을 추가합니다.

    ```python
    import stub_component_launcher
    ```

4. `kubeflow_dag_runner.py`에서 `KubeflowDagRunnerConfig`의 `supported_launcher_class`에 `StubComponentLauncher` 클래스를 추가하여 스텁 실행기가 실행될 수 있도록 합니다.

    ```python
    runner_config = kubeflow_dag_runner.KubeflowDagRunnerConfig(
        supported_launcher_classes=[
            stub_component_launcher.StubComponentLauncher
        ],
    ```

### 3. 스텁 실행기로 파이프라인 업데이트 및 실행

스텁 실행기로 기존 파이프라인을 수정된 파이프라인 정의로 업데이트합니다.

```bash
tfx pipeline update --pipeline-path=kubeflow_dag_runner.py \
  --endpoint=$endpoint --engine=kubeflow
```

`$endpoint`는 KFP 클러스터 엔드포인트로 설정되어야 합니다.

다음 명령을 실행하여 업데이트된 파이프라인의 새 실행을 생성합니다.

```bash
tfx run create --pipeline-name $pipeline_name --endpoint=$endpoint \
  --engine=kubeflow
```

## 정리

`fg` 명령을 사용하여 백그라운드에서 포트 전달에 액세스한 다음 ctrl-C를 사용하여 종료시킵니다. `gsutil -m rm -R $output_dir`을 사용하여 기록된 파이프라인 출력이 있는 디렉터리를 삭제할 수 있습니다.

이 프로젝트에서 사용한 모든 Google Cloud 리소스를 정리하려면 튜토리얼에서 사용한 [Google Cloud 프로젝트를 삭제](https://cloud.google.com/resource-manager/docs/creating-managing-projects#shutting_down_projects)하면 됩니다.

또는 각 콘솔을 방문하여 개별 리소스를 정리할 수 있습니다: - [Google Cloud Storage](https://console.cloud.google.com/storage) - [Google Container Registry](https://console.cloud.google.com/gcr) - [Google Kubernetes Engine](https://console.cloud.google.com/kubernetes)
