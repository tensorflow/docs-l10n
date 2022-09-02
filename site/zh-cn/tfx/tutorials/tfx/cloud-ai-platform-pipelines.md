# 在 Cloud AI Platform Pipelines 上使用 TFX

## 简介

本教程旨在介绍 TensorFlow Extended (TFX) 和 [AIPlatform Pipelines] (https://cloud.google.com/ai-platform/pipelines/docs/introduction)，并帮助您学习在 Google Cloud 上创建自己的机器学习流水线。教程展示了与 TFX、AI Platform Pipelines 和 Kubeflow 的集成，以及在 Jupyter 笔记本中与 TFX 的交互。

在本教程结束时，您将完成对托管在 Google Cloud 上的 ML 流水线的创建和运行。您将能够呈现每个运行的结果，并查看创建的工件的沿袭。

关键术语：TFX 流水线是一种“有向无环图”，简称“DAG”。我们经常将流水线称为 DAG。

您将遵循典型的 ML 开发流程，从检查数据集开始，最后得到一个完整有效的流水线。在此过程中，您将探索用于调试和更新流水线以及衡量性能的方式。

注：完成本教程大约需要 45-60 分钟。

### 芝加哥出租车数据集

<!-- Image free for commercial use, does not require attribution:
https://pixabay.com/photos/new-york-cab-cabs-taxi-urban-city-2087998/ -->

![Taxi](https://github.com/tensorflow/docs-l10n/blob/master/site/zh-cn/tfx/tutorials/tfx/images/airflow_workshop/taxi.jpg?raw=true) ![芝加哥出租车](https://github.com/tensorflow/docs-l10n/blob/master/site/zh-cn/tfx/tutorials/tfx/images/airflow_workshop/chicago.png?raw=true)

您将使用芝加哥市发布的 [Taxi Trips 数据集](https://data.cityofchicago.org/Transportation/Taxi-Trips/wrvz-psew)。

注：本网站提供的应用所使用的数据来自原始源（www.cityofchicago.org，芝加哥市官方网站），但在使用时进行了修改。芝加哥市不对本网站提供的任何数据的内容、准确性、时效性或完整性承担任何责任。本网站提供的数据可能会随时更改。您了解并同意，使用本网站提供的数据须自担风险。

您可以在 [Google BigQuery](https://cloud.google.com/bigquery/public-data/chicago-taxi) 中[详细了解](https://cloud.google.com/bigquery/)此数据集，并在 [BigQuery 界面](https://bigquery.cloud.google.com/dataset/bigquery-public-data:chicago_taxi_trips)中探索完整的数据集。

#### 模型目标 - 二元分类

客户给的小费是多于还是少于 20%？

## 1. 设置 Google Cloud 项目

### 1.a 在 Google Cloud 上设置您的环境

首先，您需要一个 Google Cloud 帐号。如果您已有帐号，请跳至[创建新项目](#create_project)。

警告：此演示旨在不超过 [Google Cloud 的免费层级](https://cloud.google.com/free)限制。如果您已有 Google 帐号，您可能已经达到免费层级限制，或已用尽送给新用户的 Google Cloud 赠金。**在这种情况下，执行此演示将导致您的 Google Cloud 帐号产生费用**。

1. 转到 [Google Cloud Console](https://console.cloud.google.com/)。

2. 同意 Google Cloud 条款及条件

      <img src="https://github.com/tensorflow/docs-l10n/blob/master/site/zh-cn/tfx/tutorials/tfx/images/cloud-ai-platform-pipelines/welcome-popup.png?raw=true">

3. 如果您想从免费试用帐号开始，请点击 [**Try For Free**](https://console.cloud.google.com/freetrial)（或 [**Get started for free**](https://console.cloud.google.com/freetrial)）。

    1. 选择您的国家/地区。

    2. 同意服务条款。

    3. 输入结算信息。

        此时，您不会被收取费用。如果您没有其他 Google Cloud 项目，您可以在不超出 [Google Cloud 免费层级](https://cloud.google.com/free)限制（最多同时运行 8 个核心）的情况下完成本教程。

注：此时，您可以选择成为付费用户，而不必依赖免费试用。由于本教程处于免费层级限制之内，因此如果这是您的唯一项目且您不超出限制，则仍然不会向您收取费用。有关详情，请参阅 [Google Cloud 价格计算器](https://cloud.google.com/products/calculator/)和 [Google Cloud Platform 免费层级](https://cloud.google.com/free)。

### 1.b 创建新项目<a name="create_project"></a>

注：本教程假设您要在新项目中进行此演示。如果愿意，您也可以在现有项目中进行。

注：创建项目之前，您必须登记一张经过验证的信用卡。

1. 在 [Google Cloud 信息中心](https://console.cloud.google.com/home/dashboard)主界面上，点击 **Google Cloud Platform** 标题旁边的项目下拉菜单，然后选择 **New Project**。
2. 输入项目名称和其他项目详细信息
3. **创建后，从项目下拉菜单中选择该项目。**

## 2. 在新的 Kubernetes 集群上设置并部署 AI Platform 流水线

注：这最长可能需要 10 分钟，因为它需要在几个点上等待资源配置。

1. 转到 [AI Platform Pipelines Clusters](https://console.cloud.google.com/ai-platform/pipelines) 页面。

    在主导航菜单下：≡ &gt; AI Platform &gt; Pipelines

2. 点击 **+ New Instance** 创建一个新集群。

      <img src="https://github.com/tensorflow/docs-l10n/blob/master/site/zh-cn/tfx/tutorials/tfx/images/cloud-ai-platform-pipelines/new-instance.png?raw=true">

3. 在 **Kubeflow Pipelines** 概览页面上，点击 **Configure**。

      <img src="https://github.com/tensorflow/docs-l10n/blob/master/site/zh-cn/tfx/tutorials/tfx/images/cloud-ai-platform-pipelines/configure.png?raw=true">

4. 点击“Enable”以启用 Kubernetes Engine API

     <img src="images/cloud-ai-platform-pipelines/open-template.png">

    注：您可能需要等待几分钟才能继续，此时我们将为您启用 Kubernetes Engine API。

5. 在 **Deploy Kubeflow Pipelines** 页面上：

    1. 为您的集群选择一个[区域](https://cloud.google.com/compute/docs/regions-zones)。可以设置网络和子网络，但出于本教程目的，我们将它们保留为默认值。

    2. **重要提示**：选中标有 *Allow access to the following cloud APIs* 的复选框。（此群集需要此权限才能访问项目的其他部分。如果跳过此步骤，稍后修复会有些棘手。）

         <img style="width: 65%;" src="images/cloud-ai-platform-pipelines/two-cpus.png">

    3. 点击 **Create New Cluster** 并稍候几分钟，直到集群创建完成为止。此过程需要几分钟时间。完成后，您将看到如下消息：

        > Cluster "cluster-1" successfully created in zone "us-central1-a".

    4. 选择命名空间和实例名称（使用默认值即可）。出于本教程的目的，无需选中 *executor.emissary* 或 *managedstorage.enabled*。

    5. 点击 **Deploy**，然后等待几分钟，直到流水线部署完毕。部署 Kubeflow 流水线，即表示您接受《服务条款》。

## 3. 设置 Cloud AI Platform 笔记本实例

1. 转到 [Vertex AI Workbench](https://console.cloud.google.com/vertex-ai/workbench) 页面。首次运行 Workbench 时，您需要启用 Notebooks API。

    在主导航菜单下：≡ -&gt; Vertex AI -&gt; Workbench

2. 如果出现提示，请启用 Compute Engine API。

3. 创建已安装 TensorFlow Enterprise 2.7（或更高版本）的**新笔记本**。

     <img src="images/cloud-ai-platform-pipelines/open-dashboard.png">

    New Notebook -&gt; TensorFlow Enterprise 2.7 -&gt; Without GPU

    选择区域，并为笔记本实例命名。

    为了不超出免费层级限制，您可能需要在此更改默认设置，将此实例可用的 vCPU 数量从 4 个减少到 2 个：

    1. 在 **New notebook** 表单底部选择 **Advanced Options**。

    2. 如果您需要留在免费层级，则需要在 **Machine configuration** 下选择 1 个或 2 个 vCPU 的配置。

         <img style="width: 65%;" src="images/cloud-ai-platform-pipelines/two-cpus.png">

    3. 等待新笔记本创建完成，然后点击 **Enable Notebooks API**

注：如果使用 1 个或 2 个而非默认数量或更多的 vCPU，笔记本的性能可能会变慢。这不会严重妨碍您完成本教程。如果要使用默认设置，请[将您的帐号升级](https://cloud.google.com/free/docs/gcp-free-tier#to_upgrade_your_account)到至少 12 个 vCPU。这将产生费用。有关价格的详细信息，请参阅 [Google Kubernetes Engine 价格](https://cloud.google.com/kubernetes-engine/pricing/)，包括[价格计算器](https://cloud.google.com/products/calculator)和有关 [Google Cloud 免费层级](https://cloud.google.com/free)的信息。

## 4. 启动 Getting Started 笔记本

1. 转到 [**AI Platform Pipelines Clusters**] (https://console.cloud.google.com/ai-platform/pipelines) 页面。

    在主导航菜单下：≡ -&gt; AI Platform -&gt; Pipelines

2. 在本教程中使用的集群所在行上，点击 **Open Pipelines Dashboard**。

     <img src="https://github.com/tensorflow/docs-l10n/blob/master/site/zh-cn/tfx/tutorials/tfx/images/cloud-ai-platform-pipelines/select-notebook.png?raw=true" alt="select-notebook" data-md-type="image">

3. 在 **Getting Started** 页面上，点击 **Open a Cloud AI Platform Notebook on Google Cloud**。

     <img src="https://github.com/tensorflow/docs-l10n/blob/master/site/zh-cn/tfx/tutorials/tfx/images/cloud-ai-platform-pipelines/open-dashboard.png?raw=true" alt="open-dashboard" data-md-type="image">

4. 选择您用于本教程的笔记本实例并点击 **Continue**，然后点击 **Confirm**。

      <img src="https://github.com/tensorflow/docs-l10n/blob/master/site/zh-cn/tfx/tutorials/tfx/images/cloud-ai-platform-pipelines/check-the-box.png?raw=true">

## 5. 继续在笔记本中操作

重要提示：本教程的其余部分应在上一步中打开的 JupyterLab 笔记本中完成。可在此处查看说明和解释作为参考。

### 安装

Getting Started 笔记本首先会将 [TFX](https://www.tensorflow.org/tfx) 和 [Kubeflow Pipelines (KFP)](https://www.kubeflow.org/docs/pipelines/) 安装到运行 Jupyter Lab 的虚拟机中：

然后，它会检查已安装的 TFX 版本，进行导入，并设置和打印项目 ID：

![Install tf and kfp](https://github.com/tensorflow/docs-l10n/blob/master/site/zh-cn/tfx/tutorials/tfx/images/cloud-ai-platform-pipelines/pip-install-nb-cell.png?raw=true)

### 连接您的 Google Cloud 服务

流水线配置需要您的项目 ID，您可以通过笔记本获得此 ID 并将其设置为环境变量。

```python
# Read GCP project id from env.
shell_output=!gcloud config list --format 'value(core.project)' 2>/dev/null
GCP_PROJECT_ID=shell_output[0]
print("GCP project ID:" + GCP_PROJECT_ID)
```

现在，设置您的 KFP 集群端点。

端点可通过 Pipelines 信息中心的网址找到。转到 Kubeflow 流水线信息中心并查看网址。端点为网址中*从* `https://` *后到* `googleusercontent.com`（含）的所有内容。

```python
ENDPOINT='' # Enter YOUR ENDPOINT here.
```

之后，笔记本会为自定义 Docker 镜像设置唯一名称：

```python
# Docker image name for the pipeline image
CUSTOM_TFX_IMAGE='gcr.io/' + GCP_PROJECT_ID + '/tfx-pipeline'
```

## 6. 将模板复制到项目目录中

编辑下一个笔记本单元，为您的流水线设置名称。在本教程中，我们将使用 `my_pipeline`。

```python
PIPELINE_NAME="my_pipeline"
PROJECT_DIR=os.path.join(os.path.expanduser("~"),"imported",PIPELINE_NAME)
```

接下来，笔记本会使用 `tfx` CLI 复制流水线模板。本教程使用芝加哥出租车数据集来执行二元分类，因此模板将模型设置为 `taxi`：

```python
!tfx template copy \
  --pipeline-name={PIPELINE_NAME} \
  --destination-path={PROJECT_DIR} \
  --model=taxi
```

然后，笔记本将其 CWD 上下文更改为项目目录：

```
%cd {PROJECT_DIR}
```

### 浏览流水线文件

您应该会在 Cloud AI Platform 笔记本的左侧看到一个文件浏览器。里面应该有一个包含您的流水线名称 (`my_pipeline`) 的目录。打开目录并查看文件。（您还可以在笔记本环境中打开它们并进行编辑。）

```
# You can also list the files from the shell
! ls
```

上面的 `tfx template copy` 命令创建了用于构建流水线的文件的基本基架。其中包括 Python 源代码、样本数据和 Jupyter 笔记本。这些内容针对此特定示例。对于您自己的流水线，这些应为您的流水线所需的支持文件。

以下是 Python 文件的简要说明。

- `pipeline` - 此目录包含流水线的定义
    - `constants.py` - 定义流水线运行程序的通用常量
    - `pipeline.py` - 定义 TFX 组件和流水线
- `models` - 此目录包含 ML 模型定义
    - `features.py` `features_test.py` 定义模型的特征
    - `preprocessing.py` / `preprocessing_test.py` - 使用 `tf::Transform` 定义预处理作业
    - `estimator` - 此目录包含一个基于 Estimator 的模型
        - `constants.py` - 定义模型的常量
        - `model.py` / `model_test.py` - 使用 TF Estimator 定义 DNN 模型
    - `keras` - 此目录包含一个基于 Keras 的模型
        - `constants.py` - 定义模型的常量
        - `model.py` / `model_test.py` 使用 Keras 定义 DNN 模型
- `beam_runner.py` / `kubeflow_runner.py` - 为每个编排引擎定义运行程序

## 7. 在 Kubeflow 上运行您的第一个 TFX 流水线

此笔记本将使用 `tfx run` CLI 命令运行流水线。

### 连接到存储

运行流水线会创建必须存储在 [ML-Metadata](https://github.com/google/ml-metadata) 中的工件。工件是指载荷，即必须存储在文件系统或块存储中的文件。在本教程中，我们将使用在设置期间自动创建的存储分区来使用 GCS 存储元数据载荷。它的名称将为 `<your-project-id>-kubeflowpipelines-default`。

### 创建流水线

笔记本会将样本数据上传到 GCS 存储分区，以便可以稍后在流水线中使用。

```python
!gsutil cp data/data.csv gs://{GOOGLE_CLOUD_PROJECT}-kubeflowpipelines-default/tfx-template/data/taxi/data.csv
```

之后，笔记本会使用 `tfx pipeline create` 命令创建流水线。

```python
!tfx pipeline create  \
--pipeline-path=kubeflow_runner.py \
--endpoint={ENDPOINT} \
--build-image
```

创建流水线时，将生成 `Dockerfile` 来构建 Docker 镜像。不要忘记将这些文件和其他源文件一起添加到您的源代码控制系统（例如 Git）。

### 运行流水线

之后，笔记本会使用 `tfx run create` 命令启动流水线的执行运行。您还将在 Kubeflow Pipelines 信息中心的“Experiments”下看到此运行。

```python
!tfx run create --pipeline-name={PIPELINE_NAME} --endpoint={ENDPOINT}
```

您可以在 Kubeflow Pipelines 信息中心中查看流水线。

注：如果流水线运行失败，您可以在 KFP 信息中心中查看详细日志。失败的一个主要来源是权限相关问题。请确保您的 KFP 集群具有访问 Google Cloud API 的权限。您可以[在 GCP 中创建 KFP 集群时](https://cloud.google.com/ai-platform/pipelines/docs/setting-up)对此进行配置，或参阅 [GCP 中的问题排查](https://cloud.google.com/ai-platform/pipelines/docs/troubleshooting)文档。

## 8. 验证数据

所有数据科学或机器学习项目的第一项任务都是理解和清理数据。

- 了解每个特征的数据类型
- 查找异常和缺失值
- 了解每个特征的分布

### 组件

![Data Components](https://github.com/tensorflow/docs-l10n/blob/master/site/zh-cn/tfx/tutorials/tfx/images/airflow_workshop/examplegen1.png?raw=true) ![Data Components](https://github.com/tensorflow/docs-l10n/blob/master/site/zh-cn/tfx/tutorials/tfx/images/airflow_workshop/examplegen2.png?raw=true)

- [ExampleGen](https://www.tensorflow.org/tfx/guide/examplegen) 用于提取并拆分输入数据集。
- [StatisticsGen](https://www.tensorflow.org/tfx/guide/statsgen) 用于计算数据集的统计信息。
- [SchemaGen](https://www.tensorflow.org/tfx/guide/schemagen) 用于检查统计信息并创建数据架构。
- [ExampleValidator](https://www.tensorflow.org/tfx/guide/exampleval) 用于查找数据集中的异常和缺失值。

### 在 JupyterLab 文件编辑器中：

在 `pipeline`/`pipeline.py` 中，取消注释将这些组件追加到流水线的行：

```python
# components.append(statistics_gen)
# components.append(schema_gen)
# components.append(example_validator)
```

（复制模板文件时已启用 `ExampleGen`。）

### 更新流水线并重新运行

```python
# Update the pipeline
! tfx pipeline update \
  --pipeline-path=kubeflow_runner.py \
  --endpoint={ENDPOINT}

! tfx run create --pipeline-name "{PIPELINE_NAME}"
```

### 查看流水线

对于 Kubeflow 编排器，访问 KFP 信息中心并在流水线运行页面中找到流水线输出。点击左侧的“Experiments”标签页，然后在“Experiments”页面中点击“All runs”。您应该能够找到带有流水线名称的运行。

### 更高级的示例

本教程展示的示例仅帮助您入门。如需更高级的示例，请参阅 [TensorFlow Data Validation Colab](https://www.tensorflow.org/tfx/tutorials/data_validation/chicago_taxi)。

要详细了解如何使用 TFDV 探索和验证数据集，请参阅[ tensorflow.org 中的示例](https://www.tensorflow.org/tfx/data_validation)。

## 9. 特征工程

您可以通过特征工程提高数据的预测质量和/或降低维数。

- 特征交叉
- 词汇
- 嵌入向量
- PCA
- 分类编码

使用 TFX 的一个好处是，您只需编写一次转换代码，生成的转换将在训练和应用之间保持一致。

### 组件

![set path](https://github.com/tensorflow/docs-l10n/blob/master/site/zh-cn/tfx/tutorials/tfx/images/airflow_workshop/transform.png?raw=true)

- [Transform](https://www.tensorflow.org/tfx/guide/transform) 会对数据集执行特征工程。

### 在 JupyterLab 文件编辑器中：

在 `pipeline`/`pipeline.py`中，找到并取消注释将 [Transform](https://www.tensorflow.org/tfx/guide/transform) 追加到流水线的行。

```python
# components.append(transform)
```

### 更新流水线并重新运行

```python
# Update the pipeline
! tfx pipeline update \
  --pipeline-path=kubeflow_runner.py \
  --endpoint={ENDPOINT}

! tfx run create --pipeline-name "{PIPELINE_NAME}"
```

### 查看流水线输出

对于 Kubeflow 编排器，访问 KFP 信息中心并在流水线运行页面中找到流水线输出。点击左侧的“Experiments”选项卡，然后在“Experiments”页面中点击“All runs”。您应该能够找到带有流水线名称的运行。

### 更高级的示例

本教程展示的示例仅帮助您入门。如需更高级的示例，请参阅 [TensorFlow Transform Colab](https://www.tensorflow.org/tfx/tutorials/transform/census)。

## 10. 训练

使用干净、整洁并经过转换的数据训练 TensorFlow 模型。

- 包括来自上一步的转换，以便一致地应用它们
- 将结果保存为 SavedModel 以便投入生产环境
- 使用 TensorBoard 呈现并探索训练过程
- 还要保存 EvalSavedModel 以分析模型的性能

### 组件

- [Trainer](https://www.tensorflow.org/tfx/guide/trainer) 用于训练 TensorFlow 模型。

### 在 JupyterLab 文件编辑器中：

在 `pipeline`/`pipeline.py` 中，找到并取消注释将 Trainer 追加到流水线的行：

```python
# components.append(trainer)
```

### 更新流水线并重新运行

```python
# Update the pipeline
! tfx pipeline update \
  --pipeline-path=kubeflow_runner.py \
  --endpoint={ENDPOINT}

! tfx run create --pipeline-name "{PIPELINE_NAME}"
```

### 查看流水线输出

对于 Kubeflow 编排器，访问 KFP 信息中心并在流水线运行页面中找到流水线输出。点击左侧的“Experiments”选项卡，然后在“Experiments”页面中点击“All runs”。您应该能够找到带有流水线名称的运行。

### 更高级的示例

本教程展示的示例仅帮助您入门。如需更高级的示例，请参阅 [TensorBoard 教程](https://www.tensorflow.org/tensorboard/get_started)。

## 11. 分析模型性能

不仅仅要了解顶级指标。

- 用户只会体验到模型的查询性能
- 顶级指标可能会掩盖部分数据切片性能不佳的问题
- 模型的公平性十分重要
- 通常，用户或数据的关键子集非常重要，并且可能会很小
    - 在重要但不常见的条件下的性能
    - 针对关键受众（如意见领袖）的性能
- 如果要替换目前在生产环境中的模型，首先应确保新模型的性能更优

### 组件

- [Evaluator](https://www.tensorflow.org/tfx/guide/evaluator) 会对训练结果执行深度分析。

### 在 JupyterLab 文件编辑器中：

在 `pipeline`/`pipeline.py` 中，找到并取消注释将 Evaluator 追加到流水线的行：

```python
components.append(evaluator)
```

### 更新流水线并重新运行

```python
# Update the pipeline
! tfx pipeline update \
  --pipeline-path=kubeflow_runner.py \
  --endpoint={ENDPOINT}

! tfx run create --pipeline-name "{PIPELINE_NAME}"
```

### 查看流水线输出

对于 Kubeflow 编排器，访问 KFP 信息中心并在流水线运行页面中找到流水线输出。点击左侧的“Experiments”选项卡，然后在“Experiments”页面中点击“All runs”。您应该能够找到带有流水线名称的运行。

## 12. 应用模型

如果新模型已准备就绪，请进行相应设置。

- Pusher 会将 SavedModels 部署到已知位置

部署目标会从已知位置接收新模型

- TensorFlow Serving
- TensorFlow Lite
- TensorFlow JS
- TensorFlow Hub

### 组件

- [Pusher](https://www.tensorflow.org/tfx/guide/pusher) 会将模型部署到应用基础架构。

### 在 JupyterLab 文件编辑器中：

在 `pipeline`/`pipeline.py` 中，找到并取消注释将 Pusher 追加到流水线的行：

```python
# components.append(pusher)
```

### 查看流水线输出

对于 Kubeflow 编排器，访问 KFP 信息中心并在流水线运行页面中找到流水线输出。点击左侧的“Experiments”标签页，然后在“Experiments”页面中点击“All runs”。您应该能够找到带有流水线名称的运行。

### 可用部署目标

现在，您已经训练并验证了模型，模型现在可以投入生产环境了。您现在可以将模型部署到任何 TensorFlow 部署目标，包括：

- [TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving)，用于在服务器或服务器场中应用模型，并处理 REST 和/或 gRPC 推断请求。
- [TensorFlow Lite](https://www.tensorflow.org/lite)，用于将模型包含在 Android 或 iOS 原生移动应用中，或包含在 Raspberry Pi、IoT 或微控制器应用中。
- [TensorFlow.js](https://www.tensorflow.org/js)，用于在网络浏览器或 Node.JS 应用中运行模型。

## 更高级的示例

本教程展示的示例仅帮助您入门。下面是一些与其他 Cloud 服务集成的示例。

### Kubeflow Pipelines 资源注意事项

根据载荷的要求，Kubeflow Pipelines 部署的默认配置可能满足也可能不满足您的需求。您可以在调用 `KubeflowDagRunnerConfig` 时使用 `pipeline_operator_funcs` 自定义您的资源配置。

`pipeline_operator_funcs` 是 `OpFunc` 项的列表，它会转换在 KFP 流水线规范（从 `KubeflowDagRunner` 编译）中生成的所有 `ContainerOp` 实例。

例如，要配置内存，我们可以使用 [`set_memory_request`](https://github.com/kubeflow/pipelines/blob/646f2fa18f857d782117a078d626006ca7bde06d/sdk/python/kfp/dsl/_container_op.py#L249) 来声明所需的内存量。一种典型的实现方式是为 `set_memory_request` 创建封装容器，并使用该封装容器将其添加到流水线 `OpFunc` 的列表中：

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

类似的资源配置函数包括：

- `set_memory_limit`
- `set_cpu_request`
- `set_cpu_limit`
- `set_gpu_limit`

### 尝试 `BigQueryExampleGen`

[BigQuery](https://cloud.google.com/bigquery) 是一种无服务器、扩缩能力极强且经济实惠的云数据仓库。BigQuery 可在 TFX 中用作训练样本的来源。在此步骤中，我们将向流水线添加 `BigQueryExampleGen`。

#### 在 JupyterLab 文件编辑器中：

**双击打开 `pipeline.py`**。注释掉 `CsvExampleGen` 并取消注释创建 `BigQueryExampleGen` 实例的行。您还需要取消注释 `create_pipeline` 函数的 `query` 参数。

我们需要指定用于 BigQuery 的 GCP 项目，为此，您需要在创建流水线时在 `beam_pipeline_args` 中设置 `--project`。

**双击打开 `configs.py`**。取消注释 `BIG_QUERY_WITH_DIRECT_RUNNER_BEAM_PIPELINE_ARGS` 和 `BIG_QUERY_QUERY` 的定义。您应将此文件中的项目 ID 和区域值替换为用于您的 GCP 项目的正确值。

> **注：您必须先在 `configs.py` 文件中设置您的 GCP 项目 ID 和区域，然后才能继续。**

**Change directory one level up.** Click the name of the directory above the file list. The name of the directory is the name of the pipeline which is `my_pipeline` if you didn't change the pipeline name.

**双击打开 `kubeflow_runner.py`**。为 `create_pipeline` 函数取消注释 `query` 和 `beam_pipeline_args` 两个参数。

现在，流水线已准备好使用 BigQuery 作为样本源。像之前一样更新流水线，并像在第 5 和第 6 步中那样创建新的执行运行。

#### 更新流水线并重新运行

```python
# Update the pipeline
!tfx pipeline update \
  --pipeline-path=kubeflow_runner.py \
  --endpoint={ENDPOINT}

!tfx run create --pipeline-name {PIPELINE_NAME} --endpoint={ENDPOINT}
```

### 尝试 Dataflow

多个 [TFX 组件使用 Apache Beam](https://www.tensorflow.org/tfx/guide/beam) 来实现数据并行流水线，这意味着您可以使用 [Google Cloud Dataflow](https://cloud.google.com/dataflow/) 分布数据处理工作负载。在此步骤中，我们将设置 Kubeflow 编排器，将 Dataflow 用作 Apache Beam 的数据处理后端。

> **注**：如果尚未启用 Dataflow API，您可以使用控制台或在 CLI 中使用以下命令（例如，在 Cloud Shell 中）启用：

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

> **注**：执行速度可能受限于默认的 [Google Compute Engine (GCE)](https://cloud.google.com/compute) 配额。我们建议为大约 250 个 Dataflow 虚拟机设置足够的配额： **250 个 CPU、250 个 IP 地址，以及 62500 GB 永久性磁盘**。有关详情，请参阅 [GCE 配额](https://cloud.google.com/compute/quotas)和 [Dataflow 配额](https://cloud.google.com/dataflow/quotas)文档。如果您受到 IP 地址配额的限制，使用更大的 [`worker_type`](https://cloud.google.com/dataflow/docs/guides/specifying-exec-params#setting-other-cloud-dataflow-pipeline-options) 将减少所需的 IP 数量。

**双击 `pipeline` 更改目录，然后双击打开 `configs.py`**。取消注释 `GOOGLE_CLOUD_REGION` 和 `DATAFLOW_BEAM_PIPELINE_ARGS` 的定义。

**浏览到上一级目录**。点击文件列表上方的目录名称。该目录名称为流水线的名称，如果您未更改流水线名称，则为 `my_pipeline`。

**双击打开 `kubeflow_runner.py`**。取消注释 `beam_pipeline_args`。（还要确保注释掉您在第 7 步中添加的当前 `beam_pipeline_args`。）

#### 更新流水线并重新运行

```python
# Update the pipeline
!tfx pipeline update \
  --pipeline-path=kubeflow_runner.py \
  --endpoint={ENDPOINT}

!tfx run create --pipeline-name {PIPELINE_NAME} --endpoint={ENDPOINT}
```

您可以在 [Cloud Console 的 Dataflow](http://console.cloud.google.com/dataflow) 中找到您的 Dataflow 作业。

### 使用 KFP 尝试 Cloud AI Platform Training 和 Cloud AI Platform Prediction

TFX 可与多种代管式 GCP 服务（例如，[Cloud AI Platform Training 和 Cloud AI Platform Prediction](https://cloud.google.com/ai-platform/)）互操作。您可以设置 `Trainer` 组件以使用 Cloud AI Platform Training（一项用于训练机器学习模型的代管式服务）。此外，当您完成模型构建并准备应用时，可以将您的模型*推送*到 Cloud AI Platform Prediction 进行应用。在此步骤中，我们将设置 `Trainer` 和 `Pusher` 组件以使用 Cloud AI Platform 服务。

在编辑文件之前，您可能必须首先启用 *AI Platform Training &amp; Prediction API*。

**双击 `pipeline` 更改目录，然后双击打开 `configs.py`**。取消注释 `GOOGLE_CLOUD_REGION`、`GCP_AI_PLATFORM_TRAINING_ARGS` 和 `GCP_AI_PLATFORM_SERVING_ARGS` 的定义。我们将使用自定义构建的容器镜像在 Cloud AI Platform Training 中训练模型，因此我们应将 `GCP_AI_PLATFORM_TRAINING_ARGS` 中的 `masterConfig.imageUri` 设置为与上文中 `CUSTOM_TFX_IMAGE` 相同的值。

**浏览到上一级目录，然后双击打开 `kubeflow_runner.py`**。取消注释 `ai_platform_training_args` 和 `ai_platform_serving_args`。

> 注：如果您在训练步骤中收到权限错误，则可能需要为 Cloud Machine Learning Engine（AI Platform Prediction 和 AI Platform Training）服务帐号提供 Storage Object Viewer 权限。可在 [Container Registry 文档](https://cloud.google.com/container-registry/docs/access-control#grant)中获得更多信息。

#### 更新流水线并重新运行

```python
# Update the pipeline
!tfx pipeline update \
  --pipeline-path=kubeflow_runner.py \
  --endpoint={ENDPOINT}

!tfx run create --pipeline-name {PIPELINE_NAME} --endpoint={ENDPOINT}
```

您可以在 [Cloud AI Platform 作业](https://console.cloud.google.com/ai-platform/jobs)中找到您的训练作业。如果流水线成功完成，则可以在 [Cloud AI Platform 模型](https://console.cloud.google.com/ai-platform/models)中找到您的模型。

## 14. 使用您自己的数据

在本教程中，您为使用芝加哥出租车数据集的模型创建了流水线。现在，请尝试将您自己的数据放入流水线中。您的数据可以存储在流水线能够访问的任何位置，包括 Google Cloud Storage、BigQuery 或 CSV 文件。

您需要修改流水线定义以适应您的数据。

### 如果您的数据保存在文件中

1. 修改 `kubeflow_runner.py` 中的 `DATA_PATH` 以指示位置。

### 如果您的数据保存在 BigQuery 中

1. 将 configs.py 中的 `BIG_QUERY_QUERY` 修改为您的查询语句。
2. 在 `models`/`features.py` 中添加特征。
3. 修改 `models`/`preprocessing.py` 以[转换输入数据用于训练](https://www.tensorflow.org/tfx/guide/transform)。
4. 修改 `models`/`keras`/`model.py` 和 `models`/`keras`/`constants.py` 以[描述您的 ML 模型](https://www.tensorflow.org/tfx/guide/trainer)。

### 详细了解 Trainer

有关训练流水线的更多详细信息，请参阅 [Trainer 组件指南](https://www.tensorflow.org/tfx/guide/trainer)。

## 清理

要清理此项目中使用的所有 Google Cloud 资源，您可以[删除用于本教程的 Google Cloud 项目](https://cloud.google.com/resource-manager/docs/creating-managing-projects#shutting_down_projects)。

或者，您可以通过访问每个控制台来清理各个资源：- [Google Cloud Storage](https://console.cloud.google.com/storage) - [Google Container Registry](https://console.cloud.google.com/gcr) - [Google Kubernetes Engine](https://console.cloud.google.com/kubernetes)
