# Cloud AI Platform 파이프라인의 TFX

## Introduction

This tutorial is designed to introduce [TensorFlow Extended (TFX)](https://www.tensorflow.org/tfx) and [AIPlatform Pipelines] (https://cloud.google.com/ai-platform/pipelines/docs/introduction), and help you learn to create your own machine learning pipelines on Google Cloud. It shows integration with TFX, AI Platform Pipelines, and Kubeflow, as well as interaction with TFX in Jupyter notebooks.

이 튜토리얼이 끝나면 GCP에서 호스팅되는 ML 파이프라인을 만들고 실행할 수 있게 됩니다. 각 실행의 결과를 시각화하고 생성된 아티팩트의 계보를 볼 수 있습니다.

핵심 용어: TFX 파이프라인은 Directed Acyclic Graph 또는 "DAG"입니다. 종종 파이프라인을 DAG라고 합니다.

데이터세트를 검사하는 것으로 시작하여 완전하게 작동하는 파이프라인을 만드는 것으로 완성되는 일반적인 ML 개발 프로세스를 따릅니다. 그 과정에서 파이프라인을 디버그 및 업데이트하고 성능을 측정하는 방법을 살펴봅니다.

참고: 이 튜토리얼은 완료하는 데 45-60분이 소요될 수 있습니다.

### Chicago Taxi 데이터세트

<!-- Image free for commercial use, does not require attribution:
https://pixabay.com/photos/new-york-cab-cabs-taxi-urban-city-2087998/ -->

![Taxi](images/airflow_workshop/taxi.jpg) ![Chicago taxi](images/airflow_workshop/chicago.png)

시카고 시에서 공개한 [Taxi Trips 데이터세트](https://data.cityofchicago.org/Transportation/Taxi-Trips/wrvz-psew)를 사용하고 있습니다.

참고: 이 사이트는 원 출처인 시카고 시의 공식 웹 사이트 www.cityofchicago.org를 바탕으로 수정된 데이터를 사용하는 애플리케이션을 제공합니다. 시카고 시는 이 사이트에서 제공되는 데이터의 내용, 정확성, 적시성 또는 완전성에 대해 어떠한 주장도하지 않습니다. 이 사이트에서 제공되는 데이터는 언제든지 변경될 수 있습니다. 이 사이트에서 제공하는 데이터는 자신의 책임 하에 사용되는 것으로 이해됩니다.

[Google BigQuery](https://cloud.google.com/bigquery/)에서 데이터세트에 대해 [자세히](https://cloud.google.com/bigquery/public-data/chicago-taxi) 알아볼 수 있습니다. [BigQuery UI](https://bigquery.cloud.google.com/dataset/bigquery-public-data:chicago_taxi_trips)에서 전체 데이터세트를 살펴보세요.

#### 모델 목표 - 이진 분류

고객이 20% 이상 팁을 줄까요?

## 1. Google Cloud 프로젝트 설정하기

### 1.a GCP에서 환경 설정하기

시작하려면 Google Cloud 계정이 필요합니다. 이미 있는 경우 [새 프로젝트 만들기](#create_project)로 건너뜁니다.

경고: 이 데모는 [Google Cloud의 무료 등급](https://cloud.google.com/free) 한도를 초과하지 않도록 설계되었습니다. 이미 Google 계정이 있는 경우 무료 등급 한도에 도달했거나 새 사용자에게 제공되는 무료 Google Cloud 크레딧이 모두 소진되었을 수 있습니다. **이 경우, 이 데모를 수행하면 Google Cloud 계정에 요금이 부과됩니다**.

1. Go to the [Google Cloud Console](https://console.cloud.google.com/).

2. Google Cloud 이용 약관에 동의합니다.


    <img style="width: 50%;" src="images/cloud-ai-platform-pipelines/welcome-popup.png">

3. If you would like to start with a free trial account, click on [**Try For Free**](https://console.cloud.google.com/freetrial) (or [**Get started for free**](https://console.cloud.google.com/freetrial)).

    1. 해당 국가를 선택합니다.

    2. Agree to the terms of service.

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


    <img style="width: 65%;" src="images/cloud-ai-platform-pipelines/enable_api.png">

    Note: You may have to wait several minutes before moving on, while the Kubernetes Engine APIs are being enabled for you.

5. **Deploy Kubeflow Pipelines(Kubeflow 파이프라인 배포)** 페이지에서 다음을 수행합니다.

    1. Select a [zone](https://cloud.google.com/compute/docs/regions-zones) (or "region") for your cluster. The network and subnetwork can be set, but for the purposes of this tutorial we will leave them as defaults.

    2. **IMPORTANT** Check the box labeled *Allow access to the following cloud APIs*. (This is required for this cluster to access the other pieces of your project. If you miss this step, fixing it later is a bit tricky.)


        <img style="width: 50%;" src="images/cloud-ai-platform-pipelines/check-the-box.png">

    3. Click **Create New Cluster**, and wait several minutes until the cluster has been created.  This will take a few minutes.  When it completes you will see a message like:

        > Cluster "cluster-1" successfully created in zone "us-central1-a".

    4. Select a namespace and instance name (using the defaults is fine). For the purposes of this tutorial do not check *executor.emissary* or *managedstorage.enabled*.

    5. Click **Deploy**, and wait several moments until the pipeline has been deployed. By deploying Kubeflow Pipelines, you accept the Terms of Service.

## 3. Set up Cloud AI Platform Notebook instance.

1. Go to the [Vertex AI Workbench](https://console.cloud.google.com/vertex-ai/workbench) page.  The first time you run Workbench you will need to enable the Notebooks API.

    Under the Main Navigation Menu: ≡ -&gt; Vertex AI -&gt; Workbench

2. If prompted, enable the Compute Engine API.

3. Create a **New Notebook** with TensorFlow Enterprise 2.7 (or above) installed.


    <img style="width: 65%;" src="images/cloud-ai-platform-pipelines/new-notebook.png">

    New Notebook -&gt; TensorFlow Enterprise 2.7 -&gt; Without GPU

    Select a region and zone, and give the notebook instance a name.

    To stay within the Free Tier limits, you may need to change the default settings here to reduce the number of vCPUs available to this instance from 4 to 2:

    1. Select **Advanced Options** at the bottom of the **New notebook** form.

    2. Under **Machine configuration** you may want to select a configuration with 1 or 2 vCPUs if you need to stay in the free tier.


        <img style="width: 65%;" src="images/cloud-ai-platform-pipelines/two-cpus.png">

    3. Wait for the new notebook to be created, and then click **Enable Notebooks API**

Note: You may experience slow performance in your notebook if you use 1 or 2 vCPUs instead of the default or higher. This should not seriously hinder your completion of this tutorial. If would like to use the default settings, [upgrade your account](https://cloud.google.com/free/docs/gcp-free-tier#to_upgrade_your_account) to at least 12 vCPUs. This will accrue charges. See [Google Kubernetes Engine Pricing](https://cloud.google.com/kubernetes-engine/pricing/) for more details on pricing, including a [pricing calculator](https://cloud.google.com/products/calculator) and information about the [Google Cloud Free Tier](https://cloud.google.com/free).

## 4. Launch the Getting Started Notebook

1. Go to the [**AI Platform Pipelines Clusters**] (https://console.cloud.google.com/ai-platform/pipelines) page.

    Under the Main Navigation Menu: ≡ -&gt; AI Platform -&gt; Pipelines

2. On the line for the cluster you are using in this tutorial, click **Open Pipelines Dashboard**.


    <img src="images/cloud-ai-platform-pipelines/open-dashboard.png">

3. On the **Getting Started** page, click **Open a Cloud AI Platform Notebook on Google Cloud**.


    <img src="images/cloud-ai-platform-pipelines/open-template.png">

4. Select the Notebook instance you are using for this tutorial and **Continue**, and then **Confirm**.

    ![select-notebook](images/cloud-ai-platform-pipelines/select-notebook.png)

## 5. Continue working in the Notebook

Important: The rest of this tutorial should be completed in Jupyter Lab Notebook you opened in the previous step. The instructions and explanations are available here as a reference.

### Install

The Getting Started Notebook starts by installing [TFX](https://www.tensorflow.org/tfx) and [Kubeflow Pipelines (KFP)](https://www.kubeflow.org/docs/pipelines/) into the VM which Jupyter Lab is running in.

It then checks which version of TFX is installed, does an import, and sets and prints the Project ID:

![check python version and import](images/cloud-ai-platform-pipelines/check-version-nb-cell.png)

### Connect with your Google Cloud services

The pipeline configuration needs your project ID, which you can get through the notebook and set as an environmental variable.

```python
# Read GCP project id from env.
shell_output=!gcloud config list --format 'value(core.project)' 2>/dev/null
GCP_PROJECT_ID=shell_output[0]
print("GCP project ID:" + GCP_PROJECT_ID)
```

Now set your KFP cluster endpoint.

이 내용은 파이프라인 대시보드의 URL에서 찾을 수 있습니다. Kubeflow Pipeline 대시보드로 이동하여 URL을 확인합니다. 엔드 포인트는 URL에서 `https://`*로 시작하여* `googleusercontent.com`까지에 포함된 모든 것입니다.

```python
ENDPOINT='' # Enter YOUR ENDPOINT here.
```

The notebook then sets a unique name for the custom Docker image:

```python
# Docker image name for the pipeline image
CUSTOM_TFX_IMAGE='gcr.io/' + GCP_PROJECT_ID + '/tfx-pipeline'
```

## 6. Copy a template into your project directory

Edit the next notebook cell to set a name for your pipeline. In this tutorial we will use `my_pipeline`.

```python
PIPELINE_NAME="my_pipeline"
PROJECT_DIR=os.path.join(os.path.expanduser("~"),"imported",PIPELINE_NAME)
```

The notebook then uses the `tfx` CLI to copy the pipeline template. This tutorial uses the Chicago Taxi dataset to perform binary classification, so the template sets the model to `taxi`:

```python
!tfx template copy \
  --pipeline-name={PIPELINE_NAME} \
  --destination-path={PROJECT_DIR} \
  --model=taxi
```

The notebook then changes its CWD context to the project directory:

```
%cd {PROJECT_DIR}
```

### Browse the pipeline files

On the left-hand side of the Cloud AI Platform Notebook, you should see a file browser. There should be a directory with your pipeline name (`my_pipeline`). Open it and view the files. (You'll be able to open them and edit from the notebook environment as well.)

```
# You can also list the files from the shell
! ls
```

The `tfx template copy` command above created a basic scaffold of files that build a pipeline. These include Python source codes, sample data, and Jupyter notebooks. These are meant for this particular example. For your own pipelines these would be the supporting files that your pipeline requires.

Here is brief description of the Python files.

- `pipeline` - This directory contains the definition of the pipeline
    - `configs.py` — defines common constants for pipeline runners
    - `pipeline.py` — defines TFX components and a pipeline
- `models` - This directory contains ML model definitions.
    - `features.py` `features_test.py` — defines features for the model
    - `preprocessing.py` / `preprocessing_test.py` — defines preprocessing jobs using `tf::Transform`
    - `estimator` - This directory contains an Estimator based model.
        - `constants.py` — defines constants of the model
        - `model.py` / `model_test.py` — defines DNN model using TF estimator
    - `keras` - This directory contains a Keras based model.
        - `constants.py` — defines constants of the model
        - `model.py` / `model_test.py` — defines DNN model using Keras
- `beam_runner.py` / `kubeflow_runner.py` — define runners for each orchestration engine

## 7. Run your first TFX pipeline on Kubeflow

The notebook will run the pipeline using the `tfx run` CLI command.

### Connect to storage

Running pipelines create artifacts which have to be stored in [ML-Metadata](https://github.com/google/ml-metadata). Artifacts refer to payloads, which are files that must be stored in a file system or block storage. For this tutorial, we'll use GCS to store our metadata payloads, using the bucket that was created automatically during setup. Its name will be `<your-project-id>-kubeflowpipelines-default`.

### Create the pipeline

The notebook will upload our sample data to GCS bucket so that we can use it in our pipeline later.

```python
!gsutil cp data/data.csv gs://{GOOGLE_CLOUD_PROJECT}-kubeflowpipelines-default/tfx-template/data/taxi/data.csv
```

The notebook then uses the `tfx pipeline create` command to create the pipeline.

```python
!tfx pipeline create  \
--pipeline-path=kubeflow_runner.py \
--endpoint={ENDPOINT} \
--build-image
```

While creating a pipeline, `Dockerfile` will be generated to build a Docker image. Don't forget to add these files to your source control system (for example, git) along with other source files.

### Run the pipeline

The notebook then uses the `tfx run create` command to start an execution run of your pipeline. You will also see this run listed under Experiments in the Kubeflow Pipelines Dashboard.

```python
!tfx run create --pipeline-name={PIPELINE_NAME} --endpoint={ENDPOINT}
```

You can view your pipeline from the Kubeflow Pipelines Dashboard.

Note: If your pipeline run fails, you can see detailed logs in the KFP Dashboard. One of the major sources of failure is permission related problems. Make sure your KFP cluster has permissions to access Google Cloud APIs. This can be configured [when you create a KFP cluster in GCP](https://cloud.google.com/ai-platform/pipelines/docs/setting-up), or see [Troubleshooting document in GCP](https://cloud.google.com/ai-platform/pipelines/docs/troubleshooting).

## 8. Validate your data

The first task in any data science or ML project is to understand and clean the data.

- Understand the data types for each feature
- Look for anomalies and missing values
- Understand the distributions for each feature

### Components

![Data Components](images/airflow_workshop/examplegen1.png) ![Data Components](images/airflow_workshop/examplegen2.png)

- [ExampleGen](https://www.tensorflow.org/tfx/guide/examplegen) ingests and splits the input dataset.
- [StatisticsGen](https://www.tensorflow.org/tfx/guide/statsgen) calculates statistics for the dataset.
- [SchemaGen](https://www.tensorflow.org/tfx/guide/schemagen) SchemaGen examines the statistics and creates a data schema.
- [ExampleValidator](https://www.tensorflow.org/tfx/guide/exampleval) looks for anomalies and missing values in the dataset.

### In Jupyter lab file editor:

In `pipeline`/`pipeline.py`, uncomment the lines which append these components to your pipeline:

```python
# components.append(statistics_gen)
# components.append(schema_gen)
# components.append(example_validator)
```

(`ExampleGen` was already enabled when the template files were copied.)

### Update the pipeline and re-run it

```python
# Update the pipeline
! tfx pipeline update \
  --pipeline-path=kubeflow_runner.py \
  --endpoint={ENDPOINT}

! tfx run create --pipeline-name "{PIPELINE_NAME}"
```

### Check the pipeline

For Kubeflow Orchestrator, visit KFP dashboard and find pipeline outputs in the page for your pipeline run. Click "Experiments" tab on the left, and "All runs" in the Experiments page. You should be able to find the run with the name of your pipeline.

### More advanced example

The example presented here is really only meant to get you started. For a more advanced example see the [TensorFlow Data Validation Colab](https://www.tensorflow.org/tfx/tutorials/data_validation/chicago_taxi).

For more information on using TFDV to explore and validate a dataset, [see the examples on tensorflow.org](https://www.tensorflow.org/tfx/data_validation).

## 9. Feature engineering

You can increase the predictive quality of your data and/or reduce dimensionality with feature engineering.

- Feature crosses
- Vocabularies
- Embeddings
- PCA
- Categorical encoding

One of the benefits of using TFX is that you will write your transformation code once, and the resulting transforms will be consistent between training and serving.

### Components

![Transform](images/airflow_workshop/transform.png)

- [Transform](https://www.tensorflow.org/tfx/guide/transform) performs feature engineering on the dataset.

### In Jupyter lab file editor:

In `pipeline`/`pipeline.py`, find and uncomment the line which appends [Transform](https://www.tensorflow.org/tfx/guide/transform) to the pipeline.

```python
# components.append(transform)
```

### Update the pipeline and re-run it

```python
# Update the pipeline
! tfx pipeline update \
  --pipeline-path=kubeflow_runner.py \
  --endpoint={ENDPOINT}

! tfx run create --pipeline-name "{PIPELINE_NAME}"
```

### Check pipeline outputs

For Kubeflow Orchestrator, visit KFP dashboard and find pipeline outputs in the page for your pipeline run. Click "Experiments" tab on the left, and "All runs" in the Experiments page. You should be able to find the run with the name of your pipeline.

### More advanced example

The example presented here is really only meant to get you started. For a more advanced example see the [TensorFlow Transform Colab](https://www.tensorflow.org/tfx/tutorials/transform/census).

## 10. Training

Train a TensorFlow model with your nice, clean, transformed data.

- Include the transformations from the previous step so that they are applied consistently
- Save the results as a SavedModel for production
- Visualize and explore the training process using TensorBoard
- Also save an EvalSavedModel for analysis of model performance

### Components

- [Trainer](https://www.tensorflow.org/tfx/guide/trainer) trains a TensorFlow model.

### In Jupyter lab file editor:

In `pipeline`/`pipeline.py`, find and uncomment the which appends Trainer to the pipeline:

```python
# components.append(trainer)
```

### Update the pipeline and re-run it

```python
# Update the pipeline
! tfx pipeline update \
  --pipeline-path=kubeflow_runner.py \
  --endpoint={ENDPOINT}

! tfx run create --pipeline-name "{PIPELINE_NAME}"
```

### Check pipeline outputs

For Kubeflow Orchestrator, visit KFP dashboard and find pipeline outputs in the page for your pipeline run. Click "Experiments" tab on the left, and "All runs" in the Experiments page. You should be able to find the run with the name of your pipeline.

### More advanced example

The example presented here is really only meant to get you started. For a more advanced example see the [TensorBoard Tutorial](https://www.tensorflow.org/tensorboard/r1/summaries).

## 11. Analyzing model performance

Understanding more than just the top level metrics.

- Users experience model performance for their queries only
- Poor performance on slices of data can be hidden by top level metrics
- Model fairness is important
- Often key subsets of users or data are very important, and may be small
    - Performance in critical but unusual conditions
    - Performance for key audiences such as influencers
- If you’re replacing a model that is currently in production, first make sure that the new one is better

### Components

- [Evaluator](https://www.tensorflow.org/tfx/guide/evaluator) performs deep analysis of the training results.

### In Jupyter lab file editor:

In `pipeline`/`pipeline.py`, find and uncomment the line which appends Evaluator to the pipeline:

```python
components.append(evaluator)
```

### Update the pipeline and re-run it

```python
# Update the pipeline
! tfx pipeline update \
  --pipeline-path=kubeflow_runner.py \
  --endpoint={ENDPOINT}

! tfx run create --pipeline-name "{PIPELINE_NAME}"
```

### Check pipeline outputs

For Kubeflow Orchestrator, visit KFP dashboard and find pipeline outputs in the page for your pipeline run. Click "Experiments" tab on the left, and "All runs" in the Experiments page. You should be able to find the run with the name of your pipeline.

## 12. Serving the model

If the new model is ready, make it so.

- Pusher deploys SavedModels to well-known locations

Deployment targets receive new models from well-known locations

- TensorFlow Serving
- TensorFlow Lite
- TensorFlow JS
- TensorFlow Hub

### Components

- [Pusher](https://www.tensorflow.org/tfx/guide/pusher) deploys the model to a serving infrastructure.

### In Jupyter lab file editor:

In `pipeline`/`pipeline.py`, find and uncomment the line that appends Pusher to the pipeline:

```python
# components.append(pusher)
```

### Check pipeline outputs

For Kubeflow Orchestrator, visit KFP dashboard and find pipeline outputs in the page for your pipeline run. Click "Experiments" tab on the left, and "All runs" in the Experiments page. You should be able to find the run with the name of your pipeline.

### Available deployment targets

You have now trained and validated your model, and your model is now ready for production. You can now deploy your model to any of the TensorFlow deployment targets, including:

- [TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving), for serving your model on a server or server farm and processing REST and/or gRPC inference requests.
- [TensorFlow Lite](https://www.tensorflow.org/lite), for including your model in an Android or iOS native mobile application, or in a Raspberry Pi, IoT, or microcontroller application.
- [TensorFlow.js](https://www.tensorflow.org/js), for running your model in a web browser or Node.JS application.

## More advanced examples

The example presented above is really only meant to get you started. Below are some examples of integration with other Cloud services.

### Kubeflow Pipelines resource considerations

Depending on the requirements of your workload, the default configuration for your Kubeflow Pipelines deployment may or may not meet your needs.  You can customize your resource configurations using `pipeline_operator_funcs` in your call to `KubeflowDagRunnerConfig`.

`pipeline_operator_funcs` is a list of `OpFunc` items, which transforms all the generated `ContainerOp` instances in the KFP pipeline spec which is compiled from `KubeflowDagRunner`.

For example, to configure memory we can use [`set_memory_request`](https://github.com/kubeflow/pipelines/blob/646f2fa18f857d782117a078d626006ca7bde06d/sdk/python/kfp/dsl/_container_op.py#L249) to declare the amount of memory needed. A typical way to do that is to create a wrapper for `set_memory_request` and use it to add to to the list of pipeline `OpFunc`s:

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

Similar resource configuration functions include:

- `set_memory_limit`
- `set_cpu_request`
- `set_cpu_limit`
- `set_gpu_limit`

### Try `BigQueryExampleGen`

[BigQuery](https://cloud.google.com/bigquery) is a serverless, highly scalable, and cost-effective cloud data warehouse. BigQuery can be used as a source for training examples in TFX. In this step, we will add `BigQueryExampleGen` to the pipeline.

#### In Jupyter lab file editor:

**Double-click to open `pipeline.py`**. Comment out `CsvExampleGen` and uncomment the line which creates an instance of `BigQueryExampleGen`. You also need to uncomment the `query` argument of the `create_pipeline` function.

We need to specify which GCP project to use for BigQuery, and this is done by setting `--project` in `beam_pipeline_args` when creating a pipeline.

**Double-click to open `configs.py`**. Uncomment the definition of `BIG_QUERY_WITH_DIRECT_RUNNER_BEAM_PIPELINE_ARGS` and `BIG_QUERY_QUERY`. You should replace the project id and the region value in this file with the correct values for your GCP project.

> **Note: You MUST set your GCP project ID and region in the `configs.py` file before proceeding.**

**Change directory one level up.** Click the name of the directory above the file list. The name of the directory is the name of the pipeline which is `my_pipeline` if you didn't change the pipeline name.

**Double-click to open `kubeflow_runner.py`**. Uncomment two arguments, `query` and `beam_pipeline_args`, for the `create_pipeline` function.

Now the pipeline is ready to use BigQuery as an example source. Update the pipeline as before and create a new execution run as we did in step 5 and 6.

#### Update the pipeline and re-run it

```python
# Update the pipeline
!tfx pipeline update \
  --pipeline-path=kubeflow_runner.py \
  --endpoint={ENDPOINT}

!tfx run create --pipeline-name {PIPELINE_NAME} --endpoint={ENDPOINT}
```

### Try Dataflow

Several [TFX Components use Apache Beam](https://www.tensorflow.org/tfx/guide/beam) to implement data-parallel pipelines, and it means that you can distribute data processing workloads using [Google Cloud Dataflow](https://cloud.google.com/dataflow/). In this step, we will set the Kubeflow orchestrator to use Dataflow as the data processing back-end for Apache Beam.

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

**Double-click `pipeline` to change directory, and double-click to open `configs.py`**. Uncomment the definition of `GOOGLE_CLOUD_REGION`, and `DATAFLOW_BEAM_PIPELINE_ARGS`.

**Change directory one level up.** Click the name of the directory above the file list. The name of the directory is the name of the pipeline which is `my_pipeline` if you didn't change.

**Double-click to open `kubeflow_runner.py`**. Uncomment `beam_pipeline_args`. (Also make sure to comment out current `beam_pipeline_args` that you added in Step 7.)

#### Update the pipeline and re-run it

```python
# Update the pipeline
!tfx pipeline update \
  --pipeline-path=kubeflow_runner.py \
  --endpoint={ENDPOINT}

!tfx run create --pipeline-name {PIPELINE_NAME} --endpoint={ENDPOINT}
```

You can find your Dataflow jobs in [Dataflow in Cloud Console](http://console.cloud.google.com/dataflow).

### Try Cloud AI Platform Training and Prediction with KFP

TFX interoperates with several managed GCP services, such as [Cloud AI Platform for Training and Prediction](https://cloud.google.com/ai-platform/). You can set your `Trainer` component to use Cloud AI Platform Training, a managed service for training ML models. Moreover, when your model is built and ready to be served, you can *push* your model to Cloud AI Platform Prediction for serving. In this step, we will set our `Trainer` and `Pusher` component to use Cloud AI Platform services.

Before editing files, you might first have to enable *AI Platform Training &amp; Prediction API*.

**Double-click `pipeline` to change directory, and double-click to open `configs.py`**. Uncomment the definition of `GOOGLE_CLOUD_REGION`, `GCP_AI_PLATFORM_TRAINING_ARGS` and `GCP_AI_PLATFORM_SERVING_ARGS`. We will use our custom built container image to train a model in Cloud AI Platform Training, so we should set `masterConfig.imageUri` in `GCP_AI_PLATFORM_TRAINING_ARGS` to the same value as `CUSTOM_TFX_IMAGE` above.

**Change directory one level up, and double-click to open `kubeflow_runner.py`**. Uncomment `ai_platform_training_args` and `ai_platform_serving_args`.

> Note: If you receive a permissions error in the Training step, you may need to provide Storage Object Viewer permissions to the Cloud Machine Learning Engine (AI Platform Prediction &amp; Training) service account. More information is available in the [Container Registry documentation](https://cloud.google.com/container-registry/docs/access-control#grant).

#### Update the pipeline and re-run it

```python
# Update the pipeline
!tfx pipeline update \
  --pipeline-path=kubeflow_runner.py \
  --endpoint={ENDPOINT}

!tfx run create --pipeline-name {PIPELINE_NAME} --endpoint={ENDPOINT}
```

You can find your training jobs in [Cloud AI Platform Jobs](https://console.cloud.google.com/ai-platform/jobs). If your pipeline completed successfully, you can find your model in [Cloud AI Platform Models](https://console.cloud.google.com/ai-platform/models).

## 14. Use your own data

In this tutorial, you made a pipeline for a model using the Chicago Taxi dataset. Now try putting your own data into the pipeline. Your data can be stored anywhere the pipeline can access it, including Google Cloud Storage, BigQuery, or CSV files.

You need to modify the pipeline definition to accommodate your data.

### If your data is stored in files

1. Modify `DATA_PATH` in `kubeflow_runner.py`, indicating the location.

### If your data is stored in BigQuery

1. Modify `BIG_QUERY_QUERY` in configs.py to your query statement.
2. Add features in `models`/`features.py`.
3. Modify `models`/`preprocessing.py` to [transform input data for training](https://www.tensorflow.org/tfx/guide/transform).
4. Modify `models`/`keras`/`model.py` and `models`/`keras`/`constants.py` to [describe your ML model](https://www.tensorflow.org/tfx/guide/trainer).

### Learn more about Trainer

See [Trainer component guide](https://www.tensorflow.org/tfx/guide/trainer) for more details on Training pipelines.

## Cleaning up

To clean up all Google Cloud resources used in this project, you can [delete the Google Cloud project](https://cloud.google.com/resource-manager/docs/creating-managing-projects#shutting_down_projects) you used for the tutorial.

Alternatively, you can clean up individual resources by visiting each consoles: - [Google Cloud Storage](https://console.cloud.google.com/storage) - [Google Container Registry](https://console.cloud.google.com/gcr) - [Google Kubernetes Engine](https://console.cloud.google.com/kubernetes)
