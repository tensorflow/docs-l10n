# TFX Airflow 튜토리얼

[](https://github.com/tensorflow/tfx)[](https://github.com/tensorflow/tfx)[](https://github.com/tensorflow/tfx)[ ](https://github.com/tensorflow/tfx)[](https://badge.fury.io/py/tfx)![PyPI](https://badge.fury.io/py/tfx.svg)[](https://badge.fury.io/py/tfx)[](https://badge.fury.io/py/tfx)[](https://github.com/tensorflow/tfx)
<a href="https://badge.fury.io/py/tfx" data-md-type="link" data-segment-id="7662006">![PyPI](https://badge.fury.io/py/tfx.svg)</a>[](https://badge.fury.io/py/tfx)

## 가중치 값만 저장합니다. 이것은 일반적으로 모델을 훈련할 때 사용됩니다.

이 튜토리얼은 TensorFlow Extended(TFX)를 소개하고 자체 머신러닝 파이프라인을 만드는 방법을 배우는 데 도움이 되도록 설계되었습니다. 로컬에서 실행되며 TFX 및 TensorBoard와의 통합과 더불어 Jupyter 노트북에서 TFX와의 상호 작용도 보여줍니다.

핵심 용어: TFX 파이프라인은 Directed Acyclic Graph 또는 "DAG"입니다. 종종 파이프라인을 DAG라고 합니다.

데이터세트를 검사하는 것으로 시작하여 완전하게 작동하는 파이프라인을 만드는 것으로 완성되는 일반적인 ML 개발 프로세스를 따릅니다. 그 과정에서 파이프라인을 디버그 및 업데이트하고 성능을 측정하는 방법을 살펴봅니다.

### 자세히 알아보기

자세한 내용은 TFX 사용 설명서를 참조하세요.

## 단계별 안내

일반적인 ML 개발 프로세스에 따라 단계별로 작업하여 점차적으로 파이프라인을 생성합니다. 단계는 다음과 같습니다.

1. [환경 설정](#step_1_setup_your_environment)
2. [초기 파이프라인 골격 가져오기](#step_2_bring_up_initial_pipeline_skeleton)
3. [데이터 고찰하기](#step_3_dive_into_your_data)
4. [특성 엔지니어링](#step_4_feature_engineering)
5. [훈련](#step_5_training)
6. [모델 성능 분석](#step_6_analyzing_model_performance)
7. [프로덕션 준비](#step_7_ready_for_production)

## 전제 조건

- Linux / MacOS
- Virtualenv
- Python 3.5+
- Git

### 필수 패키지

환경에 따라 여러 패키지를 설치해야 할 수 있습니다.

```bash
sudo apt-get install \
    build-essential libssl-dev libffi-dev \
    libxml2-dev libxslt1-dev zlib1g-dev \
    python3-pip git software-properties-common
```

Python 3.6을 실행 중인 경우 python3.6-dev를 설치해야 합니다.

```bash
sudo apt-get install python3.6-dev
```

Python 3.7을 실행 중인 경우 python3.7-dev를 설치해야 합니다.

```bash
sudo apt-get install python3.7-dev
```

또한 시스템의 GCC 버전이 7 미만이면 GCC를 업데이트해야 합니다. 그렇지 않으면 airflow webserver를 실행할 때 오류가 표시됩니다. 다음을 사용하여 현재 버전을 확인할 수 있습니다.

```bash
gcc --version
```

GCC를 업데이트해야 하는 경우 다음을 실행할 수 있습니다.

```bash
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt update
sudo apt install gcc-7
sudo apt install g++-7
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 60 --slave /usr/bin/g++ g++ /usr/bin/g++-7
```

### MacOS 환경

Python 3 및 git이 아직 설치되지 않은 경우 Homebrew 패키지 관리자를 사용하여 설치할 수 있습니다.

```bash
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
export PATH="/usr/local/bin:/usr/local/sbin:$PATH"
brew update
brew install python
brew install git
```

MacOS는 구성에 따라 Airflow를 실행할 때 스레드 분기에서 문제를 일으키는 경우가 종종 있습니다. 이러한 문제를 방지하려면 ~/.bash_profile을 편집하고 파일 끝에 다음 줄을 추가해야 합니다.

```bash
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
```

## 튜토리얼 자료

이 튜토리얼의 코드는 https://github.com/tensorflow/tfx/tree/master/tfx/examples/airflow_workshop에서 확인할 수 있습니다.

코드는 작업하는 대상 단계별로 구성되어 있으므로 각 단계마다 필요한 코드와 함께 수행할 작업에 대한 지침이 제공됩니다.

튜토리얼 파일에는 연습과 함께 문제가 발생할 경우를 대비하여 연습에 대한 솔루션이 모두 포함되어 있습니다.

#### 연습

- taxi_pipeline.py
- taxi_utils.py
- taxi DAG

#### 솔루션

- taxi_pipeline_solution.py
- taxi_utils_solution.py
- taxi_solution DAG

## 수행하는 작업

TFX를 사용하여 ML 파이프라인을 생성하는 방법을 배우게 됩니다.

- TFX 파이프라인은 프로덕션 ML 애플리케이션을 배포할 때 적합합니다.
- TFX 파이프라인은 데이터세트가 클 때 적합합니다.
- TFX 파이프라인은 훈련/서비스 일관성이 중요한 경우에 적합합니다.
- TFX 파이프라인은 추론을 위한 버전 관리가 중요한 경우에 적합합니다.
- Google은 프로덕션 ML에 TFX 파이프라인을 사용합니다.

일반적인 ML 개발 프로세스를 따릅니다.

- 데이터 수집, 이해 및 정리
- 특성 엔지니어링
- Training
- 모델 성능 분석
- 다듬고 정리하고 반복
- 프로덕션 준비

### 각 단계에 대한 코드 추가하기

이 튜토리얼은 모든 코드가 파일에 포함되도록 설계되었지만 3-7 단계의 모든 코드는 주석 처리되고 인라인 주석으로 표시됩니다. 인라인 주석은 코드 줄이 적용되는 단계를 나타냅니다. 예를 들어, 3단계의 코드는 # Step 3 주석으로 표시됩니다.

각 단계에 대해 추가할 코드는 일반적으로 세 가지 코드 영역에 속합니다.

- imports
- DAG 구성
- create_pipeline () 호출에서 반환된 목록
- taxi_utils.py의 지원 코드

튜토리얼을 진행하면서 현재 작업 중인 튜토리얼 단계에 적용되는 코드 줄의 주석 처리를 제거합니다. 그러면 해당 단계에 대한 코드가 추가되고 파이프라인이 업데이트됩니다. 이 때 주석 처리를 제거하는 코드를 검토할 것을 강력히 권장합니다.

## Chicago Taxi 데이터세트

<!-- Image free for commercial use, does not require attribution:
https://pixabay.com/photos/new-york-cab-cabs-taxi-urban-city-2087998/ -->

![택시](images/airflow_workshop/taxi.jpg) ![시카고 택시](images/airflow_workshop/chicago.png)

시카고 시에서 공개한 [Taxi Trips 데이터세트](https://data.cityofchicago.org/Transportation/Taxi-Trips/wrvz-psew)를 사용하고 있습니다.

참고: 이 사이트는 원 출처인 시카고 시의 공식 웹 사이트 www.cityofchicago.org를 바탕으로 수정된 데이터를 사용하는 애플리케이션을 제공합니다. 시카고 시는 이 사이트에서 제공되는 데이터의 내용, 정확성, 적시성 또는 완전성에 대해 어떠한 주장도하지 않습니다. 이 사이트에서 제공되는 데이터는 언제든지 변경될 수 있습니다. 이 사이트에서 제공하는 데이터는 자신의 책임 하에 사용되는 것으로 이해됩니다.

[Google BigQuery](https://cloud.google.com/bigquery/public-data/chicago-taxi)에서 데이터세트에 대해 [자세히](https://cloud.google.com/bigquery/) 알아볼 수 있습니다. [BigQuery UI](https://bigquery.cloud.google.com/dataset/bigquery-public-data:chicago_taxi_trips)에서 전체 데이터세트를 살펴보세요.

### 모델 목표 - 이진 분류

고객이 20% 이상 팁을 줄까요?

## 1단계: 환경 설정

설정 스크립트(setup_demo.sh)는 TFX 및 Airflow를 설치하고 이 튜토리얼에서 쉽게 작업할 수 있는 방식으로 Airflow를 구성합니다.

셸에서:

```bash
cd
virtualenv -p python3 tfx-env
source ~/tfx-env/bin/activate

git clone https://github.com/tensorflow/tfx.git
cd ~/tfx
# These instructions are specific to the 0.21 release
git checkout -f origin/r0.21
cd ~/tfx/tfx/examples/airflow_workshop/setup
./setup_demo.sh
```

setup_demo.sh를 검토하여 어떤 작업을 하는지 확인해야 합니다.

## 2단계: 초기 파이프라인 골격 가져오기

### Hello World

셸에서:

```bash
# Open a new terminal window, and in that window ...
source ~/tfx-env/bin/activate
airflow webserver -p 8080

# Open another new terminal window, and in that window ...
source ~/tfx-env/bin/activate
airflow scheduler

# Open yet another new terminal window, and in that window ...
# Assuming that you've cloned the TFX repo into ~/tfx
source ~/tfx-env/bin/activate
cd ~/tfx/tfx/examples/airflow_workshop/notebooks
jupyter notebook
```

이 단계에서 Jupyter 노트북을 시작했습니다. 나중에 이 폴더에서 노트북을 실행하게 됩니다.

### 브라우저에서:

- 브라우저를 열고 http://127.0.0.1:8080으로 이동합니다.

#### 문제 해결

웹 브라우저에서 Airflow 콘솔을 로드하는 데 문제가 있거나 airflow webserver를 실행할 때 오류가 발생한 경우, 포트 8080에서 다른 애플리케이션이 실행 중일 수 있습니다. 이 포트는 Airflow의 기본 포트이지만 사용되지 않는 다른 사용자 포트로 변경할 수 있습니다. 예를 들어, 포트 7070에서 Airflow를 실행하려면 다음을 실행할 수 있습니다.

```bash
airflow webserver -p 7070
```

#### DAG 보기 버튼

![DAG 버튼](images/airflow_workshop/airflow_dag_buttons.png)

- 왼쪽에 있는 버튼을 사용하여 DAG를 활성화합니다.
- 오른쪽에 있는 버튼을 사용하여 변경을 수행할 때 DAG를 새로 고침합니다.
- 오른쪽에 있는 버튼을 사용하여 DAG를 트리거합니다.
- 택시를 클릭하여 DAG의 그래프 보기로 이동합니다.

![그래프 새로 고침 버튼](images/airflow_workshop/graph_refresh_button.png)

#### Airflow CLI

Airflow CLI를 사용하여 DAG를 활성화하고 트리거할 수도 있습니다.

```bash
# enable/disable
airflow pause <your DAG name>
airflow unpause <your DAG name>

# trigger
airflow trigger_dag <your DAG name>
```

#### 파이프라인이 완료될 때까지 대기

DAG 보기에서 파이프라인을 트리거한 후, 파이프라인이 처리를 완료하는 것을 볼 수 있습니다. 각 구성 요소가 실행됨에 따라 DAG 그래프에서 구성 요소의 외곽선 색상이 변경되어 해당 상태를 표시합니다. 구성 요소 처리가 완료되면 윤곽선이 진한 녹색으로 바뀌어 완료되었음을 나타냅니다.

참고: 구성 요소가 실행될 때 업데이트된 상태를 보려면 오른쪽에 있는 그래프 새로 고침 버튼을 사용하거나 페이지를 새로 고쳐야 합니다.

지금까지는 파이프라인에 CsvExampleGen 구성 요소만 있으므로 짙은 녹색으로 변할 때까지 기다려야 합니다(~ 1분).

![설정 완료](images/airflow_workshop/step2.png)

## 3단계: 데이터 고찰하기

데이터 과학 또는 ML 프로젝트의 첫 번째 작업은 데이터를 이해하고 정리하는 것입니다.

- 각 특성의 데이터 유형 이해
- 이상 및 누락된 값 찾기
- 각 특성의 분포 이해

### Components

![데이터 구성 요소](images/airflow_workshop/examplegen1.png) ![데이터 구성 요소](images/airflow_workshop/examplegen2.png)

- ExampleGen은 입력 데이터세트를 수집하고 분할합니다.
- StatisticsGen은 데이터세트에 대한 통계를 계산합니다.
- SchemaGen SchemaGen은 통계를 검사하고 데이터 스키마를 생성합니다.
- ExampleValidator는 데이터세트에서 이상 항목과 누락된 값을 찾습니다.

### 편집기에서:

- ~/airflow/dags에서 taxi_pipeline.py의 Step 3으로 표시된 줄의 주석 처리를 제거합니다.
- 주석 처리를 제거할 코드를 잠시 검토합니다.

### 브라우저에서:

- 왼쪽 상단 모서리에 있는 "DAG" 링크를 클릭하여 Airflow의 DAG 목록 페이지로 돌아갑니다.
- 택시 DAG의 오른쪽에 있는 새로 고침 버튼을 클릭합니다.
    - "DAG [taxi] is now fresh as a daisy"라는 메시지가 표시됩니다.
- 택시를 트리거합니다.
- 파이프라인이 완료될 때까지 기다립니다.
    - 모든 진한 녹색
    - 오른쪽에서 새로 고침을 사용하거나 페이지를 새로 고칩니다.

![데이터 고찰하기](images/airflow_workshop/step3.png)

### Jupyter로 돌아가기:

이전에 jupyter notebook을 실행했을 때는 브라우저 탭에서 Jupyter 세션이 열렸습니다. 이제 브라우저에서 해당 탭으로 돌아갑니다.

- step3.ipynb를 엽니다.
- 노트북을 따릅니다.

![데이터 고찰하기](images/airflow_workshop/step3notebook.png)

### 고급 예제

여기에 제시된 예는 처음 시작을 위한 것일 뿐입니다. 고급 예제를 보려면 TensorFlow Data Validation Colab을 참조하세요.

TFDV를 사용하여 데이터세트를 탐색하고 유효성을 검사하는 방법에 대한 자세한 내용은 tensorflow.org의 예를 참조하세요.

## 4단계: 특성 엔지니어링

특성 엔지니어링을 통해 데이터의 예측 품질을 높이거나 차원을 줄일 수 있습니다.

- 특성 교차
- 어휘
- Embeddings
- PCA
- 범주형 인코딩

TFX를 사용할 때의 이점 중 하나는 변환 코드를 한 번 작성하면 결과 변환이 훈련과 서비스 사이에서 일관되다는 것입니다.

### Components

![변환](images/airflow_workshop/transform.png)

- Transform은 데이터세트에서 특성 엔지니어링을 수행합니다.

### 편집기에서:

- ~/airflow/dags에서 taxi_pipeline.py 및 taxi_utils.py의 Step 4로 표시된 줄의 주석 처리를 제거합니다.
- 주석 처리를 제거할 코드를 잠시 검토합니다.

### 브라우저에서:

- Airflow의 DAG 목록 페이지로 돌아갑니다.
- 택시 DAG의 오른쪽에 있는 새로 고침 버튼을 클릭합니다.
    - "DAG [taxi] is now fresh as a daisy"라는 메시지가 표시됩니다.
- 택시를 트리거합니다.
- 파이프라인이 완료될 때까지 기다립니다.
    - 모든 진한 녹색
    - 오른쪽에서 새로 고침을 사용하거나 페이지를 새로 고칩니다.

![특성 엔지니어링](images/airflow_workshop/step4.png)

### Jupyter로 돌아가기:

브라우저의 Jupyter 탭으로 돌아갑니다.

- step4.ipynb를 엽니다.
- 노트북을 따릅니다.

### 고급 예제

여기에 제시된 예는 처음 시작을 위한 것일 뿐입니다. 고급 예제를 보려면 TensorFlow Transform Colab을 참조하세요.

## 5단계: 훈련

멋지고 깔끔하게 변환된 데이터로 TensorFlow 모델을 훈련시킵니다.

- 일관되게 적용되도록 4단계의 변환을 포함합니다.
- 프로덕션을 위해 결과를 SavedModel로 저장합니다.
- TensorBoard를 사용하여 훈련 프로세스를 시각화하고 탐색합니다.
- 모델 성능 분석을 위해 EvalSavedModel도 저장합니다.

### Components

- Trainer는 TensorFlow Estimators를 사용하여 모델 훈련합니다.

### 편집기에서:

- ~/airflow/dags에서 taxi_pipeline.py 및 taxi_utils.py의 Step 5로 표시된 줄의 주석 처리를 제거합니다.
- 주석 처리를 제거할 코드를 잠시 검토합니다.

### 브라우저에서:

- Airflow의 DAG 목록 페이지로 돌아갑니다.
- 택시 DAG의 오른쪽에 있는 새로 고침 버튼을 클릭합니다.
    - "DAG [taxi] is now fresh as a daisy"라는 메시지가 표시됩니다.
- 택시를 트리거합니다.
- 파이프라인이 완료될 때까지 기다립니다.
    - 모든 진한 녹색
    - 오른쪽에서 새로 고침을 사용하거나 페이지를 새로 고칩니다.

![모델 훈련](images/airflow_workshop/step5.png)

### Jupyter로 돌아가기:

브라우저의 Jupyter 탭으로 돌아갑니다.

- step5.ipynb를 엽니다.
- 노트북을 따릅니다.

![모델 훈련](images/airflow_workshop/step5tboard.png)

### 고급 예제

여기에 제시된 예는 처음 시작을 위한 것일 뿐입니다. 고급 예제를 보려면 TensorBoard 튜토리얼을 참조하세요.

## 6단계: 모델 성능 분석

최상위 메트릭 그 이상을 이해합니다.

- 사용자는 쿼리에 대해서만 모델 성능을 경험합니다.
- 데이터 조각에서의 성능 저하는 최상위 메트릭에 의해 숨겨질 수 있습니다.
- 모델 공정성이 중요합니다.
- 종종 사용자 또는 데이터에서 중요한 일부분이 매우 중요하지만 작을 수 있습니다.
    - 중요하지만 비정상적인 조건에서 발휘되는 성능
    - 인플루언서와 같은 주요 대상이 경험하는 성능
- 현재 프로덕션 상태인 모델을 교체하는 경우, 먼저 새 모델이 더 나은지 확인하세요.
- Evaluator는 모델이 정상인지 여부를 Pusher 구성 요소에 알립니다.

### Components

- Evaluator는 훈련 결과에 대한 심층 분석을 수행하고 모델이 프로덕션으로 푸시하기에 "충분히 좋은지" 확인합니다.

### 편집기에서:

- ~/airflow/dags에서 taxi_pipeline.py의 Step 6로 표시된 줄의 주석 처리를 제거합니다.
- 주석 처리를 제거할 코드를 잠시 검토합니다.

### 브라우저에서:

- Airflow의 DAG 목록 페이지로 돌아갑니다.
- 택시 DAG의 오른쪽에 있는 새로 고침 버튼을 클릭합니다.
    - "DAG [taxi] is now fresh as a daisy"라는 메시지가 표시됩니다.
- 택시를 트리거합니다.
- 파이프라인이 완료될 때까지 기다립니다.
    - 모든 진한 녹색
    - 오른쪽에서 새로 고침을 사용하거나 페이지를 새로 고칩니다.

![모델 성능 분석](images/airflow_workshop/step6.png)

### Jupyter로 돌아가기:

브라우저의 Jupyter 탭으로 돌아갑니다.

- step6.ipynb를 엽니다.
- 노트북을 따릅니다.

![모델 성능 분석](images/airflow_workshop/step6notebook.png)

### 고급 예제

여기에 제시된 예는 처음 시작을 위한 것일 뿐입니다. 고급 예제를 보려면 TFMA Chicago Taxi 튜토리얼을 참조하세요.

## 7단계: 프로덕션 준비

새 모델이 준비되었으면 계속 진행합니다.

- Pusher가 SavedModel을 잘 알려진 위치에 배포합니다.

배포 대상은 잘 알려진 위치에서 새 모델을 받습니다.

- TensorFlow Serving
- TensorFlow Lite
- TensorFlow JS
- TensorFlow Hub

### Components

- Pusher는 모델을 적용 인프라에 배포합니다.

### 편집기에서:

- ~/airflow/dags에서 taxi_pipeline.py의 Step 7로 표시된 줄의 주석 처리를 제거합니다.
- 주석 처리를 제거할 코드를 잠시 검토합니다.

### 브라우저에서:

- Airflow의 DAG 목록 페이지로 돌아갑니다.
- 택시 DAG의 오른쪽에 있는 새로 고침 버튼을 클릭합니다.
    - "DAG [taxi] is now fresh as a daisy"라는 메시지가 표시됩니다.
- 택시를 트리거합니다.
- 파이프라인이 완료될 때까지 기다립니다.
    - 모든 진한 녹색
    - 오른쪽에서 새로 고침을 사용하거나 페이지를 새로 고칩니다.

![프로덕션 준비](images/airflow_workshop/step7.png)

## 다음 단계

이제 모델을 훈련하고 유효성을 검증했고 SavedModel 파일을 ~/airflow/saved_models/taxi 디렉토리 아래에 내보냈습니다. 이제 모델의 프로덕션 준비가 완료되었습니다. 이제 다음을 포함하여 모든 TensorFlow 배포 대상에 모델을 배포할 수 있습니다.

- TensorFlow Serving - 서버 또는 서버 팜에서 모델을 제공하고 REST 및/또는 gRPC 추론 요청을 처리합니다.
- TensorFlow Lite - Android 또는 iOS 네이티브 모바일 애플리케이션 또는 Raspberry Pi, IoT 또는 마이크로 컨트롤러 애플리케이션에 모델을 포함합니다.
- TensorFlow.js - 웹 브라우저 또는 Node.JS 애플리케이션에서 모델을 실행합니다.
