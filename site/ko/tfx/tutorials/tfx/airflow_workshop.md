# **TFX Airflow 튜토리얼**

## 개요

## 개요

이 튜토리얼은 TensorFlow Extended(TFX) 및 Apache Airflow를 오케스트레이터로 사용하여 자체 머신러닝 파이프라인을 생성하는 방법을 배우는 데 도움을 주도록 설계되었습니다. Vertex AI Workbench에서 실행되며 TFX 및 TensorBoard와의 통합은 물론 Jupyter Lab 환경에서 TFX와의 상호 작용도 보여줍니다.

### 수행할 작업

TFX를 사용하여 ML 파이프라인을 만드는 방법을 배웁니다.

- TFX 파이프라인은 Directed Acyclic Graph 또는 "DAG"입니다. 파이프라인을 종종 DAG라고 부릅니다.
- TFX 파이프라인은 프로덕션 ML 애플리케이션을 배포할 때 적합합니다.
- TFX 파이프라인은 데이터세트가 크거나 커질 수 있는 경우에 적합합니다.
- TFX 파이프라인은 훈련/서비스 일관성이 중요한 경우에 적합합니다.
- TFX 파이프라인은 추론을 위한 버전 관리가 중요한 경우에 적합합니다.
- Google은 프로덕션 ML에 TFX 파이프라인을 사용합니다.

자세한 내용은 TFX 사용 설명서를 참조하세요.

일반적인 ML 개발 프로세스를 따르게 됩니다.

- 데이터 수집, 이해 및 정리
- 특성 엔지니어링
- 훈련
- [모델 성능 분석](#step_6_analyzing_model_performance)
- 다듬고 정리하고 반복
- 프로덕션 준비

## **파이프라인 오케스트레이션을 위한 Apache Airflow**

TFX 오케스트레이터는 파이프라인에서 정의한 종속성을 기반으로 TFX 파이프라인의 구성 요소를 예약하는 작업을 담당합니다. TFX는 여러 환경 및 오케스트레이션 프레임워크에 이식할 수 있도록 설계되었습니다. TFX에서 지원하는 기본 오케스트레이터 중 하나는 [Apache Airflow](https://www.tensorflow.org/tfx/guide/airflow)입니다. 이 실습에서는 TFX 파이프라인 오케스트레이션에 Apache Airflow를 사용하는 방법을 보여줍니다. Apache Airflow는 프로그래밍 방식으로 워크플로를 작성, 예약 및 모니터링하기 위한 플랫폼입니다. TFX는 Airflow를 사용하여 작업의 DAG(방향성 비순환 그래프)로 워크플로를 작성합니다. 풍부한 사용자 인터페이스를 통해 프로덕션에서 실행되는 파이프라인을 쉽게 시각화하고 진행 상황을 모니터링하며 필요할 때 문제를 해결할 수 있습니다. Apache Airflow 워크플로는 코드로 정의됩니다. 이를 통해 유지 관리, 버전 관리, 테스트 및 협업이 더 쉬워집니다. Apache Airflow는 일괄 처리 파이프라인에 적합합니다. 부담이 없이 배우기 쉽습니다.

이 예제에서는 Airflow를 수동으로 설정하여 인스턴스에서 TFX 파이프라인을 실행합니다.

TFX에서 지원하는 다른 기본 오케스트레이터는 Apache Beam과 Kubeflow입니다. [Apache Beam](https://www.tensorflow.org/tfx/guide/beam_orchestrator)은 여러 데이터 처리 백엔드(Beam Runners)에서 실행할 수 있습니다. Cloud Dataflow는 TFX 파이프라인을 실행하는 데 사용할 수 있는 이러한 빔 러너 중 하나입니다. Apache Beam은 스트리밍과 일괄 처리 파이프라인 모두에 사용할 수 있습니다.<br> [Kubeflow](https://www.tensorflow.org/tfx/guide/kubeflow)는 Kubernetes에서 머신러닝(ML) 워크플로를 간단하고 이식 가능하며 확장 가능하게 배포하는 데 중점을 둔 오픈소스 ML 플랫폼입니다. Kubernetes 클러스터에 배포해야 하는 경우 Kubeflow를 TFFX 파이프라인의 오케스트레이터로 사용할 수 있습니다. 또한 고유한 [사용자 지정 오케스트레이터](https://www.tensorflow.org/tfx/guide/custom_orchestrator)를 사용하여 TFX 파이프라인을 실행할 수도 있습니다.

Airflow에 대한 자세한 내용은 [여기](https://airflow.apache.org/)를 참조하세요.

## **Chicago Taxi 데이터세트**

[](https://github.com/tensorflow/tfx)[](https://github.com/tensorflow/tfx)[](https://github.com/tensorflow/tfx)[ ](https://github.com/tensorflow/tfx)[](https://badge.fury.io/py/tfx)![PyPI](https://badge.fury.io/py/tfx.svg)[](https://badge.fury.io/py/tfx)[](https://badge.fury.io/py/tfx)[](https://github.com/tensorflow/tfx) [](https://badge.fury.io/py/tfx)<a href="https://badge.fury.io/py/tfx"></a>![PyPI](https://badge.fury.io/py/tfx.svg)[](https://badge.fury.io/py/tfx)

![특성 엔지니어링](images/airflow_workshop/step4.png)

시카고 시에서 공개한 [Taxi Trips 데이터세트](https://data.cityofchicago.org/Transportation/Taxi-Trips/wrvz-psew)를 사용합니다.

참고: 이 튜토리얼은 원 출처인 시카고 시의 공식 웹 사이트인 www.cityofchicago.org의 자료를 기초로 수정된 데이터를 사용하여 애플리케이션을 빌드합니다. 시카고 시는 이 튜토리얼에서 제공되는 데이터의 내용, 정확성, 적시성 또는 완전성에 대해 어떠한 주장도 하지 않습니다. 이 사이트에서 제공되는 데이터는 언제든지 변경될 수 있습니다. 이 튜토리얼에서 제공되는 데이터는 자신의 책임 하에 사용되는 것으로 간주합니다.

### 모델 목표 - 이진 분류

고객이 20% 이상 팁을 줄까요?

## Google Cloud 프로젝트 설정하기

**실습 시작 버튼을 클릭하기 전에** 다음 지침을 읽으세요. 실습은 시간 제한이 있으며 일시 중지할 수 없습니다. **실습 시작**을 클릭하면 시작되는 타이머는 Google Cloud 리소스를 사용할 수 있는 시간을 보여줍니다.

이 실습 랩에서는 시뮬레이션이나 데모 환경이 아닌 실제 클라우드 환경에서 랩 활동을 직접 수행할 수 있습니다. 실습 기간 동안 Google Cloud에 로그인하고 액세스하는 데 사용되는 새로운 임시 사용자 인증 정보가 제공됩니다.

**필요 사항** 이 실습을 완료하려면 다음이 필요합니다.

- 표준 인터넷 브라우저에 액세스(Chrome 브라우저 권장)
- 실습을 완료할 시간

**참고:** 개인 Google Cloud 계정이나 프로젝트가 이미 있는 경우 이 실습에 사용하지 마세요.

**참고:** Chrome OS 기기를 사용하는 경우 Incognito 창을 열어 이 실습을 실행하세요.

**실습을 시작하고 Google Cloud Console에 로그인하는 방법** 1. **실습 시작** 버튼을 클릭합니다. 실습 비용을 지불해야 하는 경우 지불 방법을 선택할 수 있는 팝업이 열립니다. 왼쪽에는 이 실습에 사용해야 하는 임시 인증 정보가 들어간 패널이 있습니다.

![택시](images/airflow_workshop/taxi.jpg) ![시카고 택시](images/airflow_workshop/chicago.png)

1. 사용자 이름을 복사한 다음 **Google 콘솔 열기**를 클릭합니다. 실습에서 리소스를 가동한 다음 **로그인** 페이지를 표시하는 다른 탭을 엽니다.

![데이터 구성 요소](images/airflow_workshop/examplegen1.png) ![데이터 구성 요소](images/airflow_workshop/examplegen2.png)

***팁:*** 별도의 창에서 탭을 나란히 여세요.

![DAG 버튼](images/airflow_workshop/airflow_dag_buttons.png)

1. **로그인** 페이지에서 왼쪽 패널에서 복사한 사용자 이름을 붙여넣습니다. 그런 다음 비밀번호를 복사하여 붙여 넣으세요.

***중요:*** - 왼쪽 패널의 인증 정보를 사용해야 합니다. Google Cloud Training 사용자 인증 정보를 사용하지 마세요. 자체 Google Cloud 계정이 있는 경우 이 실습에 사용하지 마세요(요금이 부과되지 않도록).

1. 이어지는 페이지를 클릭하면서 진행합니다.
2. 이용 약관에 동의합니다.

- 복구 옵션 또는 이중 인증을 추가하지 마세요(임시 계정이기 때문).

- 무료 평가판을 등록하지 마세요.

잠시 후 이 탭에서 Cloud Console이 열립니다.

**참고:** 왼쪽 상단의 **탐색 메뉴**를 클릭하면 Google Cloud 제품 및 서비스 목록이 포함된 메뉴를 볼 수 있습니다.

![프로덕션 준비](images/airflow_workshop/step7.png)

### Cloud Shell 활성화하기

Cloud Shell은 개발 도구가 로드되는 가상 머신으로, 영구 5GB 홈 디렉터리를 제공하고 Google Cloud에서 실행됩니다. Cloud Shell은 Google Cloud 리소스에 대한 명령줄 액세스를 제공합니다.

Cloud Console의 오른쪽 상단 도구 모음에서 **Cloud Shell 활성화** 버튼을 클릭합니다.

![그래프 새로 고침 버튼](images/airflow_workshop/graph_refresh_button.png)

**계속**을 클릭합니다.

![설정 완료](images/airflow_workshop/step2.png)

프로비저닝 후 환경에 연결하는 데 약간의 시간이 걸립니다. 연결되면 인증이 된 것이며 프로젝트는 *PROJECT_ID*로 설정됩니다. 예를 들면 다음과 같습니다.

![그래프 새로 고침 버튼](images/airflow_workshop/step5.png)

`gcloud`는 Google Cloud를 위한 명령줄 도구입니다. Cloud Shell에 사전 설치되어 제공되며 탭 완성을 지원합니다.

다음 명령으로 활성 계정 이름을 나열할 수 있습니다.

```
gcloud auth list
```

(출력)

> ACTIVE: * ACCOUNT: student-01-xxxxxxxxxxxx@qwiklabs.net To set the active account, run: $ gcloud config set account `ACCOUNT`

`gcloud config list project`(출력) 명령으로 프로젝트 ID를 나열할 수 있습니다.

> [core] project = &lt;project_ID&gt;

(출력 예)

> [core] project = qwiklabs-gcp-44776a13dea667a6

gcloud에 대한 전체 문서는 [gcloud 명령줄 도구 개요](https://cloud.google.com/sdk/gcloud)를 참조하세요.

## Google 클라우드 서비스 사용하기

1. Cloud Shell에서 gcloud를 사용하여 실습에서 사용되는 서비스를 활성화합니다. `gcloud services enable notebooks.googleapis.com`

## Vertex 노트북 인스턴스 배포하기

1. **탐색 메뉴**를 클릭하고 **Vertex AI**로 이동한 다음 **Workbench**로 이동합니다.

![데이터 고찰하기](images/airflow_workshop/step3.png)

1. 노트북 인스턴스 페이지에서 **새 노트북**을 클릭합니다.

2. 인스턴스 사용자 지정 메뉴에서 **TensorFlow Enterprise**를 선택하고 **TensorFlow Enterprise 2.x(with LTS)** &gt; **Without GPUs** 버전을 선택합니다.

![데이터 고찰하기](images/airflow_workshop/step3notebook.png)

1. **새 노트북 인스턴스** 대화 상자에서 연필 아이콘을 클릭하여 인스턴스 속성을 **편집**합니다.

2. **인스턴스 이름**에 인스턴스 이름을 입력합니다.

3. **지역**에 대해 `us-east1`을 선택하고 **구역**에 대해 선택한 지역 내의 구역을 선택합니다.

4. 머신 구성까지 아래로 스크롤하고 머신 유형으로 **e2-standard-2**를 선택합니다.

5. 나머지 필드는 기본값으로 두고 **만들기**를 클릭합니다.

몇 분 후 Vertex AI 콘솔에 해당 인스턴스 이름이 표시되고 그 다음에 **Jupyterlab 열기**가 표시됩니다.

1. **JupyterLab 열기**를 클릭합니다. JupyterLab 창이 새 탭에서 열립니다.

## 환경 설정하기

### 실습 리포지토리 복제하기

다음으로 JupyterLab 인스턴스에서 `tfx` 리포지토리를 복제합니다. 1. JupyterLab에서 **터미널** 아이콘을 클릭하여 새 터미널을 엽니다.

{ql-infobox0}<strong>참고:</strong> 메시지가 표시되면 빌드 권장 사항에 대해 <code>취소</code>를 클릭합니다.{/ql-infobox0}

1. `tfx` Github 리포지토리를 복제하려면 다음 명령을 입력하고 **Enter**를 누릅니다.

```
git clone https://github.com/tensorflow/tfx.git
```

1. 리포지토리를 복제했는지 확인하려면 `tfx` 디렉터리를 두 번 클릭하고 해당 콘텐츠를 볼 수 있는지 확인합니다.

![변환](images/airflow_workshop/transform.png)

### 실습 종속성 설치하기

1. 다음을 실행하여 `tfx/tfx/examples/airflow_workshop/taxi/setup/` 폴더로 이동한 다음 `./setup_demo.sh`를 실행하여 실습 종속성을 설치합니다.

```bash
cd ~/tfx/tfx/examples/airflow_workshop/taxi/setup/
./setup_demo.sh
```

위의 코드는 다음을 수행합니다.

- 필요한 패키지를 설치합니다.
- 홈 폴더에 `airflow` 폴더를 생성합니다.
- `dags` 폴더를 `tfx/tfx/examples/airflow_workshop/taxi/setup/` 폴더에서 `~/airflow/` 폴더로 복사합니다.
- csv 파일을 `tfx/tfx/examples/airflow_workshop/taxi/setup/data`에서 `~/airflow/data`로 복사합니다.

![모델 성능 분석](images/airflow_workshop/step6.png)

## Airflow 서버 구성하기

### 브라우저에서 Airflow 서버에 액세스하기 위한 방화벽 규칙 생성하기

1. `https://console.cloud.google.com/networking/firewalls/list`로 이동하여 프로젝트 이름이 적절하게 선택되었는지 확인합니다.
2. 상단의 `CREATE FIREWALL RULE` 옵션을 클릭합니다.

![변환](images/airflow_workshop/step5tboard.png)

**방화벽 만들기 대화 상자**에서 아래 나열된 단계를 따릅니다.

1. **이름**에 `airflow-tfx`를 입력합니다.
2. **우선 순위**로 `1`을 선택합니다.
3. **대상**으로 `All instances in the network`를 선택합니다.
4. **소스 IPv4 범위**에 대해 `0.0.0.0/0`을 선택합니다.
5. **프로토콜 및 포트**에 대해 `tcp`를 클릭하고 `tcp` 옆의 상자에 `7000`을 입력합니다.
6. `Create`를 클릭합니다.

![모델 성능 분석](images/airflow_workshop/step6notebook.png)

### 쉘에서 Airflow 서버 실행하기

Jupyter 실습 터미널 창에서 홈 디렉터리로 변경하고 `airflow users create` 명령을 실행하여 Airflow에 대한 관리 사용자를 생성합니다.

```bash
cd
airflow users  create --role Admin --username admin --email admin --firstname admin --lastname admin --password admin
```

그런 다음 `airflow webserver` 및 `airflow scheduler` 명령을 실행하여 서버를 실행합니다. 방화벽을 통과하도록 허용되는 포트 `7000`을 선택합니다.

```bash
nohup airflow webserver -p 7000 &> webserver.out &
nohup airflow scheduler &> scheduler.out &
```

### 외부 IP 가져오기

1. Cloud Shell에서 `gcloud`를 사용하여 외부 IP를 가져옵니다.

```
gcloud compute instances list
```

![모델 훈련](images/airflow_workshop/gcloud-instance-ip.png)

## DAG/파이프라인 실행

### 브라우저에서

브라우저를 열고 http://&lt;external_ip&gt;:7000으로 이동합니다.

- 로그인 페이지에서 `airflow users create` 명령을 실행할 때 선택한 사용자 이름(`admin`)과 비밀번호(`admin`)를 입력합니다.

![모델 훈련](images/airflow_workshop/airflow-login.png)

Airflow는 Python 소스 파일에서 DAG를 로드합니다. 각 파일을 가져와서 실행합니다. 그런 다음 해당 파일에서 모든 DAG 객체를 로드합니다. DAG 객체를 정의하는 모든 `.py` 파일은 Airflow 홈페이지에서 파이프라인으로 나열됩니다.

이 튜토리얼에서 Airflow는 `~/airflow/dags/` 폴더에서 DAG 객체를 검색합니다.

`~/airflow/dags/taxi_pipeline.py`를 열고 하단으로 스크롤하면 `DAG`라는 변수에 DAG 객체를 생성하여 저장하는 것을 확인할 수 있습니다. 따라서 아래와 같이 Airflow 홈페이지에 파이프라인으로 나열됩니다.

![dag-home-full.png](images/airflow_workshop/dag-home-full.png)

택시를 클릭하면 DAG의 그리드 보기로 리디렉션됩니다. 상단의 `Graph` 옵션을 클릭하여 DAG의 그래프 보기를 얻을 수 있습니다.

![airflow-dag-graph.png](images/airflow_workshop/airflow-dag-graph.png)

### 택시 파이프라인 트리거

홈페이지에서 DAG와 상호 작용하는 데 사용할 수 있는 버튼을 볼 수 있습니다.

![dag-buttons.png](images/airflow_workshop/dag-buttons.png)

**작업** 헤더 아래에서 **트리거** 버튼을 클릭하여 파이프라인을 트리거합니다.

택시 **DAG** 페이지에서 오른쪽에 있는 버튼을 사용하여 파이프라인이 실행될 때 DAG의 그래프 보기 상태를 새로 고칩니다. 또한 **자동 새로 고침**을 활성화하여 상태가 변경될 때 그래프 보기를 자동으로 새로 고치도록 Airflow에 지시할 수 있습니다.

![dag-button-refresh.png](images/airflow_workshop/dag-button-refresh.png)

터미널에서 [Airflow CLI](https://airflow.apache.org/cli.html)를 사용하여 DAG를 활성화하고 트리거할 수도 있습니다.

```bash
# enable/disable
airflow pause <your DAG name>
airflow unpause <your DAG name>

# trigger
airflow trigger_dag <your DAG name>
```

#### 파이프라인이 완료될 때까지 대기

파이프라인을 트리거한 후 DAG 보기에서 실행 중인 파이프라인의 진행 상황을 볼 수 있습니다. 각 구성 요소가 실행되면 DAG 그래프에서 구성 요소의 윤곽선 색상이 상태를 표시하도록 변경됩니다. 구성 요소의 처리가 완료되면 윤곽선이 짙은 녹색으로 바뀌어 완료되었음을 나타냅니다.

![dag-step7.png](images/airflow_workshop/dag-step7.png)

## 구성 요소 이해하기

이제 이 파이프라인의 구성 요소를 자세히 알아보고 파이프라인의 각 단계에서 생성된 출력을 개별적으로 살펴보겠습니다.

1. JupyterLab에서 `~/tfx/tfx/examples/airflow_workshop/taxi/notebooks/`로 이동합니다.

2. **notebook.ipynb.** ![notebook-ipynb.png](images/airflow_workshop/notebook-ipynb.png)를 엽니다.

3. 노트북에서 실습을 계속하고 화면 상단에서 **실행**(<img src="images/airflow_workshop/f1abc657d9d2845c.png" width="28.00" alt="run-button.png">) 아이콘을 클릭하여 각 셀을 실행합니다. 또는 **SHIFT + ENTER**를 사용하여 셀에서 코드를 실행할 수 있습니다.

내용을 읽고 각 세포에서 무슨 일이 일어나고 있는지 이해하도록 하세요.
