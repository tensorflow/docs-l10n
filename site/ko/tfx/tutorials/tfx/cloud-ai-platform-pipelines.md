# Cloud AI Platform 파이프라인의 TFX

## 시작하기

This tutorial is designed to introduce [TensorFlow Extended (TFX)](https://www.tensorflow.org/tfx) and [AIPlatform Pipelines] (https://cloud.google.com/ai-platform/pipelines/docs/introduction), and help you learn to create your own machine learning pipelines on Google Cloud. It shows integration with TFX, AI Platform Pipelines, and Kubeflow, as well as interaction with TFX in Jupyter notebooks.

이 튜토리얼이 끝나면 GCP에서 호스팅되는 ML 파이프라인을 만들고 실행할 수 있게 됩니다. 각 실행의 결과를 시각화하고 생성된 아티팩트의 계보를 볼 수 있습니다.

핵심 용어: TFX 파이프라인은 Directed Acyclic Graph 또는 "DAG"입니다. 종종 파이프라인을 DAG라고 합니다.

데이터세트를 검사하는 것으로 시작하여 완전하게 작동하는 파이프라인을 만드는 것으로 완성되는 일반적인 ML 개발 프로세스를 따릅니다. 그 과정에서 파이프라인을 디버그 및 업데이트하고 성능을 측정하는 방법을 살펴봅니다.

참고: 이 튜토리얼은 완료하는 데 45-60분이 소요될 수 있습니다.

### Chicago Taxi 데이터세트

<!-- Image free for commercial use, does not require attribution:
https://pixabay.com/photos/new-york-cab-cabs-taxi-urban-city-2087998/ -->

![택시](images/airflow_workshop/taxi.jpg)![시카고 택시](images/airflow_workshop/chicago.png)

시카고 시에서 공개한 [Taxi Trips 데이터세트](https://data.cityofchicago.org/Transportation/Taxi-Trips/wrvz-psew)를 사용하고 있습니다.

참고: 이 사이트는 원 출처인 시카고 시의 공식 웹 사이트 www.cityofchicago.org를 바탕으로 수정된 데이터를 사용하는 애플리케이션을 제공합니다. 시카고 시는 이 사이트에서 제공되는 데이터의 내용, 정확성, 적시성 또는 완전성에 대해 어떠한 주장도하지 않습니다. 이 사이트에서 제공되는 데이터는 언제든지 변경될 수 있습니다. 이 사이트에서 제공하는 데이터는 자신의 책임 하에 사용되는 것으로 이해됩니다.

[Google BigQuery](https://cloud.google.com/bigquery/)에서 데이터세트에 대해 [자세히](https://cloud.google.com/bigquery/public-data/chicago-taxi) 알아볼 수 있습니다. [BigQuery UI](https://bigquery.cloud.google.com/dataset/bigquery-public-data:chicago_taxi_trips)에서 전체 데이터세트를 살펴보세요.

#### 모델 목표 - 이진 분류

고객이 20% 이상 팁을 줄까요?

## 1. Google Cloud 프로젝트 설정하기

### 1.a GCP에서 환경 설정하기

시작하려면 Google Cloud 계정이 필요합니다. 이미 있는 경우 [새 프로젝트 만들기](#create_project)로 건너뜁니다.

경고: 이 데모는 [Google Cloud의 무료 등급](https://cloud.google.com/free) 한도를 초과하지 않도록 설계되었습니다. 이미 Google 계정이 있는 경우 무료 등급 한도에 도달했거나 새 사용자에게 제공되는 무료 Google Cloud 크레딧이 모두 소진되었을 수 있습니다. **이 경우, 이 데모를 수행하면 Google Cloud 계정에 요금이 부과됩니다**.

1. [Google Cloud Console](https://console.cloud.google.com/)로 이동합니다.

2. Google Cloud 이용 약관에 동의합니다.

    <img style="width: 50%;" src="images/cloud-ai-platform-pipelines/welcome-popup.png">

3. If you would like to start with a free trial account, click on [**Try For Free**](https://console.cloud.google.com/freetrial) (or [**Get started for free**](https://console.cloud.google.com/freetrial)).

    1. 해당 국가를 선택합니다.

    2. 서비스 약관에 동의합니다.

    3. 청구 세부 정보를 입력합니다.

        You will not be charged at this point. If you have no other Google Cloud projects, you can complete this tutorial without exceeding the [Google Cloud Free Tier](https://cloud.google.com/free) limits, which includes a max of 8 cores running at the same time.

참고: 이 시점에서 무료 평가판에 의존하지 않고 유료 사용자로 전환할 수 있습니다. 이 튜토리얼은 무료 등급 한도 내에서 유지되므로 이것이 유일한 프로젝트이고 이 한도 내에서 유지한다면 요금이 청구되지 않습니다. 자세한 내용은 [Google Cloud 비용 계산기](https://cloud.google.com/products/calculator/) 및 [Google Cloud Platform 무료 등급](https://cloud.google.com/free)을 참조하세요.

### 1.b 새 프로젝트 만들기<a name="create_project"></a>

참고: 이 튜토리얼에서는 새 프로젝트에서 이 데모 작업을 한다고 가정합니다. 원하는 경우 기존 프로젝트에서 작업할 수 있습니다.

참고: 프로젝트를 생성하기 전에 검증된 신용카드가 있어야 합니다.

1. [기본 Google Cloud 대시보드](https://console.cloud.google.com/home/dashboard)**에서 Google Cloud Platform** 헤더 옆에 있는 프로젝트 드롭다운을 클릭하고 **새 프로젝트**를 선택합니다.
2. 프로젝트 이름을 지정하고 다른 프로젝트 세부 정보를 입력합니다.
3. **Once you have created a project, make sure to select it from the project drop-down.**

## 2. 새 Kubernetes 클러스터에서 AI Platform 파이프라인 설정 및 배포하기

참고: 여러 지점에서 리소스가 제공될 때까지 기다려야 하므로 이 작업에 최대 10분이 소요됩니다.

1. [AI Platform 파이프라인 클러스터](https://console.cloud.google.com/ai-platform/pipelines) 페이지로 이동합니다.

    기본 탐색 메뉴 아래: ≡&gt; AI Platform&gt; 파이프라인

2. **+ New Instance(+ 새 인스턴스)**를 클릭하여 새 클러스터를 만듭니다.

    <img style="width: 65%;" src="images/cloud-ai-platform-pipelines/new-instance.png">

3. **Kubeflow Pipelines(Kubeflow 파이프라인)** 개요 페이지에서 **Configure(구성)**을 클릭합니다.

    <img style="width: 65%;" src="images/cloud-ai-platform-pipelines/configure.png">

4. Click "Enable" to enable the Kubernetes Engine API

    <img src="images/cloud-ai-platform-pipelines/select-notebook.png" alt="선택 노트북" data-md-type="image">

    Note: You may have to wait several minutes before moving on, while the Kubernetes Engine APIs are being enabled for you.

5. **Deploy Kubeflow Pipelines(Kubeflow 파이프라인 배포)** 페이지에서 다음을 수행합니다.

    1. Select a [zone](https://cloud.google.com/compute/docs/regions-zones) (or "region") for your cluster. The network and subnetwork can be set, but for the purposes of this tutorial we will leave them as defaults.

    2. **IMPORTANT** Check the box labeled *Allow access to the following cloud APIs*. (This is required for this cluster to access the other pieces of your project. If you miss this step, fixing it later is a bit tricky.)

        <img style="width: 65%;" src="images/cloud-ai-platform-pipelines/two-cpus.png">

    3. Click **Create New Cluster**, and wait several minutes until the cluster has been created.  This will take a few minutes.  When it completes you will see a message like:

        > Cluster "cluster-1" successfully created in zone "us-central1-a".

    4. Select a namespace and instance name (using the defaults is fine). For the purposes of this tutorial do not check *executor.emissary* or *managedstorage.enabled*.

    5. Click **Deploy**, and wait several moments until the pipeline has been deployed. By deploying Kubeflow Pipelines, you accept the Terms of Service.

## 3. Cloud AI Platform Notebook 인스턴스를 설정합니다.

1. Go to the [Vertex AI Workbench](https://console.cloud.google.com/vertex-ai/workbench) page.  The first time you run Workbench you will need to enable the Notebooks API.

    Under the Main Navigation Menu: ≡ -&gt; Vertex AI -&gt; Workbench

2. 메시지가 표시되면 Compute Engine API를 사용 설정합니다.

3. Create a **New Notebook** with TensorFlow Enterprise 2.7 (or above) installed.

    <img src="images/cloud-ai-platform-pipelines/open-dashboard.png" alt="오픈 대시 보드" data-md-type="image">

    New Notebook -&gt; TensorFlow Enterprise 2.7 -&gt; Without GPU

    Select a region and zone, and give the notebook instance a name.

    To stay within the Free Tier limits, you may need to change the default settings here to reduce the number of vCPUs available to this instance from 4 to 2:

    1. Select **Advanced Options** at the bottom of the **New notebook** form.

    2. Under **Machine configuration** you may want to select a configuration with 1 or 2 vCPUs if you need to stay in the free tier.

        <img style="width: 65%;" src="images/cloud-ai-platform-pipelines/two-cpus.png">

    3. Wait for the new notebook to be created, and then click **Enable Notebooks API**

Note: You may experience slow performance in your notebook if you use 1 or 2 vCPUs instead of the default or higher. This should not seriously hinder your completion of this tutorial. If would like to use the default settings, [upgrade your account](https://cloud.google.com/free/docs/gcp-free-tier#to_upgrade_your_account) to at least 12 vCPUs. This will accrue charges. See [Google Kubernetes Engine Pricing](https://cloud.google.com/kubernetes-engine/pricing/) for more details on pricing, including a [pricing calculator](https://cloud.google.com/products/calculator) and information about the [Google Cloud Free Tier](https://cloud.google.com/free).

## 4. 시작하기 노트북 시작

1. Go to the [**AI Platform Pipelines Clusters**] (https://console.cloud.google.com/ai-platform/pipelines) page.

    기본 탐색 메뉴에서 : ≡-&gt; AI Platform-&gt; 파이프 라인

2. 이 튜토리얼에서 사용중인 클러스터 라인에서 **Open Pipelines Dashboard를** 클릭합니다.


    <img src="images/cloud-ai-platform-pipelines/open-dashboard.png">

3. On the **Getting Started** page, click **Open a Cloud AI Platform Notebook on Google Cloud**.


    <img src="images/cloud-ai-platform-pipelines/open-template.png">

4. Select the Notebook instance you are using for this tutorial and **Continue**, and then **Confirm**.

    <img style="width: 50%;" src="images/cloud-ai-platform-pipelines/check-the-box.png">

## 5. 노트북에서 계속 작업

중요 :이 자습서의 나머지 부분은 이전 단계에서 연 Jupyter Lab Notebook에서 완료해야합니다. 여기에서 지침과 설명을 참조 할 수 있습니다.

### 설치

시작하기 노트북은 Jupyter Lab이 실행되는 VM에 [TFX](https://www.tensorflow.org/tfx) 및 [Kubeflow Pipelines (KFP)](https://www.kubeflow.org/docs/pipelines/) 를 설치하는 것으로 시작됩니다.

그런 다음 설치된 TFX 버전을 확인하고 가져 오기를 수행하며 프로젝트 ID를 설정 및 인쇄합니다.

![파이썬 버전 확인 및 가져 오기](images/cloud-ai-platform-pipelines/check-version-nb-cell.png)

### Google Cloud 서비스와 연결

파이프 라인 구성에는 노트북을 통해 가져와 환경 변수로 설정할 수있는 프로젝트 ID가 필요합니다.

```python
# Read GCP project id from env.
shell_output=!gcloud config list --format 'value(core.project)' 2>/dev/null
GCP_PROJECT_ID=shell_output[0]
print("GCP project ID:" + GCP_PROJECT_ID)
```

이제 KFP 클러스터 엔드 포인트를 설정하십시오.

이 내용은 파이프라인 대시보드의 URL에서 찾을 수 있습니다. Kubeflow Pipeline 대시보드로 이동하여 URL을 확인합니다. 엔드 포인트는 URL에서 `https://`*로 시작하여* `googleusercontent.com`까지에 포함된 모든 것입니다.

```python
ENDPOINT='' # Enter YOUR ENDPOINT here.
```

그런 다음 노트북은 사용자 지정 Docker 이미지의 고유 한 이름을 설정합니다.

```python
# Docker image name for the pipeline image
CUSTOM_TFX_IMAGE='gcr.io/' + GCP_PROJECT_ID + '/tfx-pipeline'
```

## 6. 프로젝트 디렉토리에 템플릿 복사

다음 노트북 셀을 편집하여 파이프 라인의 이름을 설정하십시오. 이 자습서에서는 `my_pipeline` 을 사용합니다.

```python
PIPELINE_NAME="my_pipeline"
PROJECT_DIR=os.path.join(os.path.expanduser("~"),"deployed_notebook",PIPELINE_NAME)
```

그런 다음 노트북은 `tfx` CLI를 사용하여 파이프 라인 템플릿을 복사합니다. 이 가이드에서는 Chicago Taxi 데이터 세트를 사용하여 이진 분류를 수행하므로 템플릿은 모델을 `taxi` 설정합니다.

```python
!tfx template copy \
  --pipeline-name={PIPELINE_NAME} \
  --destination-path={PROJECT_DIR} \
  --model=taxi
```

그런 다음 노트북은 CWD 컨텍스트를 프로젝트 디렉토리로 변경합니다.

```
%cd {PROJECT_DIR}
```

### 파이프 라인 파일 찾아보기

Cloud AI Platform Notebook의 왼쪽에 파일 브라우저가 표시되어야합니다. `my_pipeline` )이있는 디렉토리가 있어야합니다. 그것을 열고 파일을 봅니다. (노트북 환경에서도 파일을 열고 편집 할 수 있습니다.)

```
# You can also list the files from the shell
! ls
```

위의 `tfx template copy` 명령은 파이프 라인을 구축하는 파일의 기본 스캐 폴드를 생성했습니다. 여기에는 Python 소스 코드, 샘플 데이터 및 Jupyter 노트북이 포함됩니다. 이것들은이 특정 예를위한 것입니다. 자체 파이프 라인의 경우 파이프 라인에 필요한 지원 파일이됩니다.

다음은 Python 파일에 대한 간략한 설명입니다.

- `pipeline` -이 디렉토리는 파이프 라인의 정의를 포함합니다.
    - `configs.py` 대한 공통 상수를 정의합니다.
    - pipeline.py — TFX 구성 요소 및 `pipeline.py`
- `models` 이 디렉토리에는 ML 모델 정의가 포함되어 있습니다.
    - `features.py` `features_test.py` — 모델의 기능을 정의합니다.
    - `preprocessing.py` / `preprocessing_test.py` — defines preprocessing jobs using `tf::Transform`
    - `estimator` 이 디렉토리는 Estimator 기반 모델을 포함합니다.
        - `constants.py` — 모델의 상수를 정의합니다.
        - `model.py` / `model_test.py` — TF 추정기를 사용하여 DNN 모델 정의
    - `keras` 이 디렉토리는 Keras 기반 모델을 포함합니다.
        - `constants.py` — 모델의 상수를 정의합니다.
        - `model.py` / `model_test.py` 사용하여 DNN 모델을 정의합니다.
- `beam_runner.py` / `kubeflow_runner.py` — 각 오케스트레이션 엔진의 러너를 정의합니다.

## 7. Kubeflow에서 첫 번째 TFX 파이프 라인 실행

`tfx run` CLI 명령을 사용하여 파이프 라인을 실행합니다.

### 저장소에 연결

[실행중인 파이프 라인은 ML-Metadata에](https://github.com/google/ml-metadata) 저장해야하는 아티팩트를 생성합니다. 아티팩트는 파일 시스템 또는 블록 스토리지에 저장되어야하는 파일 인 페이로드를 나타냅니다. 이 자습서에서는 GCS를 사용하여 설정 중에 자동으로 생성 된 버킷을 사용하여 메타 데이터 페이로드를 저장합니다. 이름은 `<your-project-id>-kubeflowpipelines-default` 입니다.

### 파이프 라인 생성

노트북은 샘플 데이터를 GCS 버킷에 업로드하므로 나중에 파이프 라인에서 사용할 수 있습니다.

```python
!gsutil cp data/data.csv gs://{GOOGLE_CLOUD_PROJECT}-kubeflowpipelines-default/tfx-template/data/taxi/data.csv
```

그런 다음 노트북은 `tfx pipeline create` 명령을 사용하여 파이프 라인을 생성합니다.

```python
!tfx pipeline create  \
--pipeline-path=kubeflow_runner.py \
--endpoint={ENDPOINT} \
--build-image
```

파이프 라인을 생성하는 동안 Docker 이미지를 빌드하기 위해 `Dockerfile` 이러한 파일을 다른 소스 파일과 함께 소스 제어 시스템 (예 : git)에 추가하는 것을 잊지 마십시오.

### 파이프 라인 실행

그런 다음 노트북은 `tfx run create` 명령을 사용하여 파이프 라인의 실행 실행을 시작합니다. Kubeflow 파이프 라인 대시 보드의 실험 아래에도이 실행이 나열됩니다.

```python
!tfx run create --pipeline-name={PIPELINE_NAME} --endpoint={ENDPOINT}
```

Kubeflow 파이프 라인 대시 보드에서 파이프 라인을 볼 수 있습니다.

참고 : 파이프 라인 실행이 실패하면 KFP 대시 보드에서 자세한 로그를 볼 수 있습니다. 실패의 주요 원인 중 하나는 권한 관련 문제입니다. KFP 클러스터에 Google Cloud API에 액세스 할 수있는 권한이 있는지 확인하세요. 이는 [GCP에서 KFP 클러스터를 만들 때](https://cloud.google.com/ai-platform/pipelines/docs/setting-up) [구성하거나 GCP의 문제 해결 문서를](https://cloud.google.com/ai-platform/pipelines/docs/troubleshooting) 참조하세요.

## 8. 데이터 유효성 검사

데이터 과학 또는 ML 프로젝트의 첫 번째 작업은 데이터를 이해하고 정리하는 것입니다.

- 각 기능의 데이터 유형 이해
- 이상과 결 측값 찾기
- 각 기능의 분포 이해

### 구성품

![데이터 구성 요소](images/airflow_workshop/examplegen1.png)![데이터 구성 요소](images/airflow_workshop/examplegen2.png)

- [ExampleGen](https://www.tensorflow.org/tfx/guide/examplegen) 은 입력 데이터 세트를 수집하고 분할합니다.
- StatisticsGen은 데이터 세트에 대한 [통계를 계산합니다.](https://www.tensorflow.org/tfx/guide/statsgen)
- [SchemaGen](https://www.tensorflow.org/tfx/guide/schemagen) SchemaGen은 통계를 검사하고 데이터 스키마를 생성합니다.
- [ExampleValidator](https://www.tensorflow.org/tfx/guide/exampleval) 는 데이터 세트에서 이상 항목과 누락 된 값을 찾습니다.

### Jupyter 랩 파일 편집기에서 :

`pipeline` / `pipeline.py` 에서 이러한 구성 요소를 파이프 라인에 추가하는 줄의 주석 처리를 제거합니다.

```python
# components.append(statistics_gen)
# components.append(schema_gen)
# components.append(example_validator)
```

(템플릿 파일을 복사 할 때 `ExampleGen`

### 파이프 라인을 업데이트하고 다시 실행하십시오.

```python
# Update the pipeline
! tfx pipeline update \
  --pipeline-path=kubeflow_runner.py \
  --endpoint={ENDPOINT}

! tfx run create --pipeline-name "{PIPELINE_NAME}"
```

### 파이프 라인 확인

Kubeflow Orchestrator의 경우 KFP 대시 보드를 방문하여 파이프 라인 실행 페이지에서 파이프 라인 출력을 찾으십시오. 왼쪽의 "실험"탭을 클릭하고 실험 페이지에서 "모든 실행"을 클릭합니다. 파이프 라인 이름으로 실행을 찾을 수 있어야합니다.

### 고급 예

여기에 제시된 예는 실제로 시작하기위한 것입니다. 고급 예제는[TensorFlow 데이터 유효성 검사 Colab을](https://www.tensorflow.org/tfx/tutorials/data_validation/chicago_taxi) 참조하세요.

TFDV를 사용하여 데이터 세트를 탐색하고 유효성을 검사하는 방법에 대한 자세한 내용은 tensorflow.org [의 예를 참조하세요](https://www.tensorflow.org/tfx/data_validation) .

## 9. 기능 엔지니어링

기능 엔지니어링을 통해 데이터의 예측 품질을 높이거나 차원을 줄일 수 있습니다.

- 특징 교차
- 어휘
- 임베딩
- PCA
- 범주 형 인코딩

TFX 사용의 이점 중 하나는 변환 코드를 한 번 작성하면 결과 변환이 학습과 제공간에 일관성이 있다는 것입니다.

### 구성품

![변환](images/airflow_workshop/transform.png)

- [Transform](https://www.tensorflow.org/tfx/guide/transform) performs feature engineering on the dataset.

### Jupyter 랩 파일 편집기에서 :

`pipeline` / `pipeline.py` [에 Transform](https://www.tensorflow.org/tfx/guide/transform) 을 추가하는 줄을 찾아 주석 처리를 제거합니다.

```python
# components.append(transform)
```

### 파이프 라인을 업데이트하고 다시 실행하십시오.

```python
# Update the pipeline
! tfx pipeline update \
  --pipeline-path=kubeflow_runner.py \
  --endpoint={ENDPOINT}

! tfx run create --pipeline-name "{PIPELINE_NAME}"
```

### 파이프 라인 출력 확인

Kubeflow Orchestrator의 경우 KFP 대시 보드를 방문하여 파이프 라인 실행 페이지에서 파이프 라인 출력을 찾으십시오. 왼쪽의 "실험"탭을 클릭하고 실험 페이지에서 "모든 실행"을 클릭합니다. 파이프 라인 이름으로 실행을 찾을 수 있어야합니다.

### 고급 예

여기에 제시된 예는 실제로 시작하기위한 것입니다. 고급 예제는 [TensorFlow Transform Colab을](https://www.tensorflow.org/tfx/tutorials/transform/census) 참조하십시오.

## 10. 훈련

멋지고 깨끗하며 변환 된 데이터로 TensorFlow 모델을 학습 시키십시오.

- 일관되게 적용되도록 이전 단계의 변환을 포함합니다.
- 프로덕션을 위해 결과를 저장된 모델로 저장
- TensorBoard를 사용하여 훈련 프로세스 시각화 및 탐색
- 모델 성능 분석을 위해 EvalSavedModel도 저장하십시오.

### 구성품

- [Trainer](https://www.tensorflow.org/tfx/guide/trainer) trains a TensorFlow model.

### Jupyter 랩 파일 편집기에서 :

`pipeline` / `pipeline.py` 에서 Trainer를 파이프 라인에 추가하는을 찾아 주석 처리를 제거합니다.

```python
# components.append(trainer)
```

### 파이프 라인을 업데이트하고 다시 실행하십시오.

```python
# Update the pipeline
! tfx pipeline update \
  --pipeline-path=kubeflow_runner.py \
  --endpoint={ENDPOINT}

! tfx run create --pipeline-name "{PIPELINE_NAME}"
```

### 파이프 라인 출력 확인

Kubeflow Orchestrator의 경우 KFP 대시 보드를 방문하여 파이프 라인 실행 페이지에서 파이프 라인 출력을 찾으십시오. 왼쪽의 "실험"탭을 클릭하고 실험 페이지에서 "모든 실행"을 클릭합니다. 파이프 라인 이름으로 실행을 찾을 수 있어야합니다.

### 고급 예

여기에 제시된 예는 실제로 시작하기위한 것입니다. 고급 예제는 [TensorBoard Tutorial을](https://www.tensorflow.org/tensorboard/r1/summaries) 참조하십시오.

## 11. 모델 성능 분석

최상위 수준의 측정 항목 이상을 이해합니다.

- 사용자는 쿼리에 대해서만 모델 성능을 경험합니다.
- 데이터 조각의 성능 저하를 최상위 메트릭으로 숨길 수 있음
- 모델 공정성이 중요합니다
- 종종 사용자 또는 데이터의 주요 하위 집합이 매우 중요하고 작을 수 있습니다.
    - 중요하지만 비정상적인 조건에서의 성능
    - 인플 루 언서와 같은 주요 청중을위한 성과
- 현재 생산중인 모델을 교체하는 경우 먼저 새 모델이 더 나은지 확인하십시오.

### 구성품

- [Evaluator](https://www.tensorflow.org/tfx/guide/evaluator) performs deep analysis of the training results.

### Jupyter 랩 파일 편집기에서 :

`pipeline` / `pipeline.py` 에 Evaluator를 추가하는 줄을 찾아 주석 처리를 제거합니다.

```python
components.append(evaluator)
```

### 파이프 라인을 업데이트하고 다시 실행하십시오.

```python
# Update the pipeline
! tfx pipeline update \
  --pipeline-path=kubeflow_runner.py \
  --endpoint={ENDPOINT}

! tfx run create --pipeline-name "{PIPELINE_NAME}"
```

### 파이프 라인 출력 확인

Kubeflow Orchestrator의 경우 KFP 대시 보드를 방문하여 파이프 라인 실행 페이지에서 파이프 라인 출력을 찾으십시오. 왼쪽의 "실험"탭을 클릭하고 실험 페이지에서 "모든 실행"을 클릭합니다. 파이프 라인 이름으로 실행을 찾을 수 있어야합니다.

## 12. 모델 제공

새 모델이 준비 되었으면 준비하십시오.

- Pusher는 저장된 모델을 잘 알려진 위치에 배포합니다.

배포 대상은 잘 알려진 위치에서 새 모델을받습니다.

- TensorFlow Serving
- TensorFlow Lite
- TensorFlow JS
- TensorFlow 허브

### 구성품

- [Pusher](https://www.tensorflow.org/tfx/guide/pusher) deploys the model to a serving infrastructure.

### Jupyter 랩 파일 편집기에서 :

`pipeline` / `pipeline.py` 에 Pusher를 추가하는 줄을 찾아 주석 처리를 제거합니다.

```python
# components.append(pusher)
```

### 파이프 라인 출력 확인

Kubeflow Orchestrator의 경우 KFP 대시 보드를 방문하여 파이프 라인 실행 페이지에서 파이프 라인 출력을 찾으십시오. 왼쪽의 "실험"탭을 클릭하고 실험 페이지에서 "모든 실행"을 클릭합니다. 파이프 라인 이름으로 실행을 찾을 수 있어야합니다.

### 사용 가능한 배포 대상

이제 모델을 교육하고 검증했으며 이제 모델을 생산할 준비가되었습니다. 이제 다음을 포함하여 모든 TensorFlow 배포 대상에 모델을 배포 할 수 있습니다.

- [TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving) , 서버 또는 서버 팜에서 모델을 제공하고 REST 및 / 또는 gRPC 추론 요청을 처리합니다.
- [TensorFlow Lite](https://www.tensorflow.org/lite) , Android 또는 iOS 네이티브 모바일 애플리케이션 또는 Raspberry Pi, IoT 또는 마이크로 컨트롤러 애플리케이션에 모델을 포함합니다.
- 웹 브라우저 또는 Node.JS 애플리케이션에서 모델을 실행하기위한 [TensorFlow.js.](https://www.tensorflow.org/js)

## 더 고급 예제

위에 제시된 예는 실제로 시작하기위한 것입니다. 다음은 다른 클라우드 서비스와의 통합 예입니다.

### Kubeflow Pipelines 리소스 고려 사항

워크로드의 요구 사항에 따라 Kubeflow Pipelines 배포의 기본 구성이 요구 사항을 충족하거나 충족하지 못할 수 있습니다. `KubeflowDagRunnerConfig` 호출에서 `pipeline_operator_funcs` 를 사용하여 리소스 구성을 사용자 지정할 수 있습니다.

`pipeline_operator_funcs` 목록이다 `OpFunc` 생성 된 모든 변형 항목 `ContainerOp` 로부터 컴파일되는 KFP 배관 사양에서 인스턴스 `KubeflowDagRunner` .

예를 들어 메모리를 구성하기 위해 [`set_memory_request`](https://github.com/kubeflow/pipelines/blob/646f2fa18f857d782117a078d626006ca7bde06d/sdk/python/kfp/dsl/_container_op.py#L249) 를 사용하여 필요한 메모리 양을 선언 할 수 있습니다. 이를 수행하는 일반적인 방법은 `set_memory_request` `OpFunc` 목록에 추가하는 것입니다.

```python
def request_more_memory():
  def _set_memory_spec(container_op):
    container_op.set_memory_request('32G')
  return _set_memory_spec

# Then use this opfunc in KubeflowDagRunner
pipeline_op_funcs = kubeflow_dag_runner.get_default_pipeline_operator_funcs()
pipeline_op_funcs.append(request_more_memory())
config = KubeflowDagRunnerConfig(
    pipeline_operator_funcs=pipeline_op_funcs,
    ...
)
kubeflow_dag_runner.KubeflowDagRunner(config=config).run(pipeline)
```

유사한 리소스 구성 기능은 다음과 같습니다.

- `set_memory_limit`
- `set_cpu_request`
- `set_cpu_limit`
- `set_gpu_limit`

### `BigQueryExampleGen` 사용해보기

[BigQuery](https://cloud.google.com/bigquery) 는 확장 성이 뛰어나며 비용 효율적인 서버리스 클라우드 데이터웨어 하우스입니다. BigQuery는 TFX에서 학습 예제의 소스로 사용할 수 있습니다. 이 단계에서는 파이프 라인 `BigQueryExampleGen`

#### Jupyter 랩 파일 편집기에서 :

**두 번 클릭하여 `pipeline.py`** 를 엽니 다. `CsvExampleGen` `BigQueryExampleGen` 인스턴스를 만드는 줄의 주석 처리를 제거합니다. `create_pipeline` `query` 인수의 주석 처리를 제거해야합니다.

BigQuery에 사용할 GCP 프로젝트를 지정해야합니다. 파이프 라인을 만들 때 `beam_pipeline_args` `--project` 를 설정하면됩니다.

**두 번 클릭하여 `configs.py`** 를 엽니 다. `BIG_QUERY_WITH_DIRECT_RUNNER_BEAM_PIPELINE_ARGS` 및 BIG_QUERY_QUERY의 정의를 주석 `BIG_QUERY_QUERY` . 이 파일의 프로젝트 ID와 지역 값을 GCP 프로젝트의 올바른 값으로 바꿔야합니다.

> **Note: You MUST set your GCP project ID and region in the `configs.py` file before proceeding.**

**디렉토리를 한 수준 위로 변경합니다.** 파일 목록 위의 디렉토리 이름을 클릭하십시오. 디렉토리 이름은 파이프 라인 이름을 변경하지 않은 경우 `my_pipeline`

**두 번 클릭하여 `kubeflow_runner.py`** 를 엽니 다. `create_pipeline` 함수에 대한 `query` 및 `beam_pipeline_args` 주석 처리를 제거하십시오.

이제 파이프 라인에서 BigQuery를 예제 소스로 사용할 준비가되었습니다. 이전과 같이 파이프 라인을 업데이트하고 5 단계와 6 단계에서했던 것처럼 새 실행 실행을 만듭니다.

#### 파이프 라인을 업데이트하고 다시 실행하십시오.

```python
# Update the pipeline
!tfx pipeline update \
  --pipeline-path=kubeflow_runner.py \
  --endpoint={ENDPOINT}

!tfx run create --pipeline-name {PIPELINE_NAME} --endpoint={ENDPOINT}
```

### Dataflow 사용해보기

여러 [TFX 구성 요소는 Apache Beam](https://www.tensorflow.org/tfx/guide/beam) [을 사용하여 데이터 병렬 파이프 라인을 구현하며, 이는 Google Cloud Dataflow를](https://cloud.google.com/dataflow/) 사용하여 데이터 처리 워크로드를 분산 할 수 있음을 의미합니다. 이 단계에서는 Apache Beam의 데이터 처리 백엔드로 Dataflow를 사용하도록 Kubeflow 오케 스트레이터를 설정합니다.

> **Note:** If the Dataflow API is not already enabled, you can enable it using the console, or from the CLI using this command (for example, in the Cloud Shell):

```bash
# Select your project:
gcloud config set project YOUR_PROJECT_ID

# Get a list of services that you can enable in your project:
gcloud services list --available | grep Dataflow

# If you don't see dataflow.googleapis.com listed, that means you haven't been
# granted access to enable the Dataflow API.  See your account adminstrator.

# Enable the Dataflow service:

gcloud services enable dataflow.googleapis.com
```

> **Note:** Execution speed may be limited by default [Google Compute Engine (GCE)](https://cloud.google.com/compute) quota. We recommend setting a sufficient quota for approximately 250 Dataflow VMs: **250 CPUs, 250 IP Addresses, and 62500 GB of Persistent Disk**. For more details, please see the [GCE Quota](https://cloud.google.com/compute/quotas) and [Dataflow Quota](https://cloud.google.com/dataflow/quotas) documentation. If you are blocked by IP Address quota, using a bigger [`worker_type`](https://cloud.google.com/dataflow/docs/guides/specifying-exec-params#setting-other-cloud-dataflow-pipeline-options) will reduce the number of needed IPs.

**`pipeline` 을 두 번 클릭하여 디렉토리를 변경하고 두 번 클릭하여 `configs.py`** 를 엽니 다. `GOOGLE_CLOUD_REGION` 및 `DATAFLOW_BEAM_PIPELINE_ARGS` 정의의 주석 처리를 제거합니다.

**디렉토리를 한 수준 위로 변경합니다.** 파일 목록 위의 디렉토리 이름을 클릭하십시오. 디렉토리 이름은 변경하지 않은 경우 `my_pipeline` 파이프 라인의 이름입니다.

**두 번 클릭하여 `kubeflow_runner.py`** 를 엽니 다. `beam_pipeline_args` 주석을 제거하십시오. (또한 7 단계에서 추가 한 `beam_pipeline_args`

#### 파이프 라인을 업데이트하고 다시 실행하십시오.

```python
# Update the pipeline
!tfx pipeline update \
  --pipeline-path=kubeflow_runner.py \
  --endpoint={ENDPOINT}

!tfx run create --pipeline-name {PIPELINE_NAME} --endpoint={ENDPOINT}
```

[Cloud Console의 Dataflow에서](http://console.cloud.google.com/dataflow) Dataflow 작업을 찾을 수 있습니다.

### KFP로 Cloud AI Platform 학습 및 예측 사용해보기

[TFX는 Cloud AI Platform for Training and Prediction과](https://cloud.google.com/ai-platform/) 같은 여러 관리 형 GCP 서비스와 상호 운용됩니다. ML 모델 학습을위한 관리 형 서비스 인 Cloud AI Platform Training을 사용하도록 `Trainer` 구성 요소를 설정할 수 있습니다. 또한 모델이 빌드되고 제공 될 준비가되면 제공을 위해 모델을 Cloud AI Platform 예측으로 *푸시 할 수 있습니다.* 이 단계에서는 Cloud AI Platform 서비스를 사용하도록 `Trainer` 및 `Pusher`

*파일을 수정하기 전에 먼저 AI Platform Training &amp; Prediction API* 를 사용 설정해야 할 수 있습니다.

**`pipeline` 을 두 번 클릭하여 디렉토리를 변경하고 두 번 클릭하여 `configs.py`** 를 엽니 다. `GOOGLE_CLOUD_REGION` , `GCP_AI_PLATFORM_TRAINING_ARGS` 및 `GCP_AI_PLATFORM_SERVING_ARGS` 의 정의를 주석 해제합니다. 커스텀 빌드 컨테이너 이미지를 사용하여 Cloud AI Platform Training에서 모델을 학습 `masterConfig.imageUri` 의 `GCP_AI_PLATFORM_TRAINING_ARGS` 를 위의 `CUSTOM_TFX_IMAGE` 와 동일한 값으로 설정해야합니다.

**디렉토리를 한 수준 위로 변경하고 두 번 클릭하여 `kubeflow_runner.py`** 를 엽니 다. `ai_platform_training_args` 및 `ai_platform_serving_args` 주석을 제거합니다.

> Note: If you receive a permissions error in the Training step, you may need to provide Storage Object Viewer permissions to the Cloud Machine Learning Engine (AI Platform Prediction &amp; Training) service account. More information is available in the [Container Registry documentation](https://cloud.google.com/container-registry/docs/access-control#grant).

#### 파이프 라인을 업데이트하고 다시 실행하십시오.

```python
# Update the pipeline
!tfx pipeline update \
  --pipeline-path=kubeflow_runner.py \
  --endpoint={ENDPOINT}

!tfx run create --pipeline-name {PIPELINE_NAME} --endpoint={ENDPOINT}
```

[Cloud AI Platform](https://console.cloud.google.com/ai-platform/jobs) 작업에서 학습 작업을 찾을 수 있습니다. 파이프 라인이 성공적으로 완료되면 [Cloud AI Platform 모델](https://console.cloud.google.com/ai-platform/models) 에서 모델을 찾을 수 있습니다.

## 14. 자신의 데이터 사용

이 자습서에서는 Chicago Taxi 데이터 세트를 사용하여 모델에 대한 파이프 라인을 만들었습니다. 이제 자신의 데이터를 파이프 라인에 넣어보십시오. Google Cloud Storage, BigQuery 또는 CSV 파일을 포함하여 파이프 라인이 액세스 할 수있는 모든 위치에 데이터를 저장할 수 있습니다.

데이터를 수용하려면 파이프 라인 정의를 수정해야합니다.

### 데이터가 파일에 저장된 경우

1. 위치를 나타내는 `kubeflow_runner.py` `DATA_PATH` 를 수정합니다.

### 데이터가 BigQuery에 저장된 경우

1. `BIG_QUERY_QUERY` 의 BIG_QUERY_QUERY를 쿼리 문으로 수정합니다.
2. `models` / `features.py` 기능을 추가하십시오.
3. `models` `preprocessing.py` 를 수정 [하여 학습을위한 입력 데이터](https://www.tensorflow.org/tfx/guide/transform) 를 변환합니다.
4. [ML 모델](https://www.tensorflow.org/tfx/guide/trainer) 을 `models` / `keras` / `model.py` 및 `models` / `keras` / `constants.py` 를 수정하십시오.

### Trainer에 대해 더 알아보기

학습 파이프 라인에 대한 자세한 내용은 [Trainer 구성 요소 가이드를 참조하세요.](https://www.tensorflow.org/tfx/guide/trainer)

## 청소

이 프로젝트에 사용 된 모든 GCP 리소스를 정리하려면 가이드에 사용한 [GCP 프로젝트를 삭제할 수 있습니다.](https://cloud.google.com/resource-manager/docs/creating-managing-projects#shutting_down_projects)

또는 각 콘솔을 방문하여 개별 리소스를 정리할 수 있습니다.- [Google Cloud Storage](https://console.cloud.google.com/storage) - [Google Container Registry](https://console.cloud.google.com/gcr) - [Google Kubernetes Engine](https://console.cloud.google.com/kubernetes)
