# **TFX Airflow 教程**

## 概览

## 概览

本教程旨在帮助您学习使用 TensorFlow Extended (TFX) 和 Apache Airflow 作为编排器来创建自己的机器学习流水线。它运行在 Vertex AI Workbench 上，展示了与 TFX 和 TensorBoard 的集成以及在 Jupyter Lab 环境中与 TFX 的交互。

### 您将要做什么？

您将学习如何使用 TFX 创建 ML 流水线

- TFX 流水线是一种“有向无环图”，简称“DAG”。我们经常将流水线称为 DAG。
- 当部署正式 ML 应用时，适合使用 TFX 流水线
- 当数据集很大或可能增长到很大时，适合使用 TFX 流水线
- 当训练/应用一致性很重要时，适合使用 TFX 流水线
- 当推断版本管理很重要时，适合使用 TFX 流水线
- Google 在正式 ML 中使用 TFX 流水线

请参阅 [TFX 用户指南](https://www.tensorflow.org/tfx/guide)了解详情。

您将遵循典型的 ML 开发流程：

- 注入、理解并清理数据
- 特征工程
- 训练
- [分析模型性能](#step_6_analyzing_model_performance)
- 不断优化
- 为投入生产环境做好准备

## **用于流水线编排的 Apache Airflow**

TFX 编排器负责根据流水线定义的依赖项调度 TFX 流水线的组件。TFX 可以移植到多种环境和编排框架。TFX 支持的默认编排器之一是 [Apache Airflow](https://www.tensorflow.org/tfx/guide/airflow)。此实验说明了如何使用 Apache Airflow 进行 TFX 流水线编排。Apache Airflow 是一个以编程方式创作、调度和监控工作流的平台。TFX 使用 Airflow 将工作流创作为任务的有向无环图 (DAG)。借助丰富的界面，可以轻松呈现生产中运行的流水线、监控进度并在需要时排查问题。Apache Airflow 工作流被定义为代码。这样可以提高它们的可维护性、可版本化性、测试性和协作性。Apache Airflow 适用于批处理流水线。它是一种易于学习的轻量级平台。

在此示例中，我们将通过手动设置 Airflow 以在实例上运行 TFX 流水线。

TFX 支持的其他默认编排器为 [Apache Beam](https://www.tensorflow.org/tfx/guide/beam_orchestrator) 和 Kubeflow。[Apache Beam](https://www.tensorflow.org/tfx/guide/beam_orchestrator) 可以在多个数据处理后端（Beam 运行程序）上运行。Cloud Dataflow 就是这样一种可用于运行 TFX 流水线的 Beam 运行程序。Apache Beam 可用于流式传输和批处理流水线。<br>[Kubeflow](https://www.tensorflow.org/tfx/guide/kubeflow) 是一个开源机器学习平台，致力于使机器学习 (ML) 工作流在 Kubernetes 上的部署变得简单、可移植且可扩缩。当需要在 Kubernetes 集群上部署时，Kubeflow 可以用作 TFFX 流水线的编排器。此外，也可以使用自己的[自定义编排器](https://www.tensorflow.org/tfx/guide/custom_orchestrator)来运行 TFX 流水线。

在[此处](https://airflow.apache.org/)阅读关于 Airflow 的更多信息。

## **芝加哥出租车数据集**

![Taxi](images/airflow_workshop/taxi.jpg)

![Feature Engineering](images/airflow_workshop/step4.png)

您将使用芝加哥市发布的 [Taxi Trips 数据集](https://data.cityofchicago.org/Transportation/Taxi-Trips/wrvz-psew)。

注：本教程构建的应用所使用的数据来自原始源（www.cityofchicago.org，芝加哥市官方网站），但已经过修改以供使用。芝加哥市对本教程中提供的任何数据的内容、准确性、及时性或完整性不做任何声明。此网站提供的数据随时可能更改。使用本教程提供的数据须自负风险。

### 模型目标 - 二元分类

客户给的小费是多于还是少于 20%？

## 设置 Google Cloud 项目

**在点击 Start Lab 按钮之前** 阅读这些说明。实验是计时的，您不能将它们暂停。计时器会在您点击 **Start Lab** 时启动，显示 Google Cloud 资源可供您使用的时长。

此动手实验室让您可以在真实的云环境中自行进行实验活动，而不是在模拟或演示环境中。它通过提供新的临时凭据来实现这一点，您可以在实验期间使用这些凭据登录和访问 Google Cloud。

**您需要什么** 要完成此实验，您需要：

- 访问标准互联网浏览器（推荐使用 Chrome 浏览器）。
- 完成实验的时间。

**注**：如果您已经拥有自己的个人 Google Cloud 帐号或项目，请勿将其用于此实验。

**注**：如果您使用的是 Chrome OS 设备，请打开无痕浏览窗口以运行此实验。

**如何开始您的实验并登录到 Google Cloud Console** 1. 点击 **Start Lab** 按钮。如果您需要支付实验费用，则会打开一个弹出窗口供您选择付款方式。左侧是一个面板，其中填充了您必须用于此实验的临时凭据。

![Taxi](images/airflow_workshop/taxi.jpg)

1. 复制用户名，然后点击 **Open Google Console**。实验会加载资源，然后打开另一个显示 **Sign in** 页面的标签页。

![Data Components](images/airflow_workshop/examplegen1.png)

***提示***：在单独的窗口中并排打开标签页。

![DAG buttons](images/airflow_workshop/airflow_dag_buttons.png)

1. 在 **Sign in** 页面中，粘贴您从左侧面板复制的用户名。然后，复制并粘贴密码。

***重要提示***：您必须使用左侧面板中的凭据。不要使用您的 Google Cloud 训练凭据。如果您有自己的 Google Cloud 帐号，请不要将其用于本实验（避免产生费用）。

1. 点击浏览后续页面：
2. 接受条款与条件。

- 请勿添加恢复选项或双重身份验证（因为这是一个临时帐号）。

- 请勿注册免费试用。

片刻之后，Cloud Console 将在此标签页中打开。

**注**：您可以通过点击左上角的 **Navigation 菜单**来查看包含 Google Cloud 产品和服务列表的菜单。

![Ready for production](images/airflow_workshop/step7.png)

### 激活 Cloud Shell

Cloud Shell 是一个加载了开发工具的虚拟机。它提供了一个永久的 5GB 主目录并在 Google Cloud 上运行。Cloud Shell 提供对您的 Google Cloud 资源的命令行访问。

在 Cloud Console 右上角的工具栏中，点击 **Activate Cloud Shell** 按钮。

![Graph refresh button](images/airflow_workshop/graph_refresh_button.png)

点击 **Continue**。

![Setup complete](images/airflow_workshop/step2.png)

配置和连接到环境需要一些时间。当连接建立时，说明您已通过身份验证，并且项目已设置为您的 *PROJECT_ID*。例如：

![Graph refresh button](images/airflow_workshop/step5.png)

`gcloud` 是 Google Cloud 的命令行工具。它预先安装在 Cloud Shell 上并支持制表符补全。

可以使用以下命令列出有效帐号名称：

```
gcloud auth list
```

（输出）

> ACTIVE: * ACCOUNT: student-01-xxxxxxxxxxxx@qwiklabs.net To set the active account, run: $ gcloud config set account `ACCOUNT`

可以使用以下命令列出项目 ID：`gcloud config list project`（输出）

> [core] project = &lt;project_ID&gt;

（示例输出）

> [core] project = qwiklabs-gcp-44776a13dea667a6

有关 gcloud 的完整文档，请参阅 [gcloud 命令行工具概览](https://cloud.google.com/sdk/gcloud)。

## 启用 Google Cloud 服务

1. 在 Cloud Shell 中，使用 gcloud 启用实验中使用的服务。`gcloud services enable notebooks.googleapis.com`

## 部署 Vertex 笔记本实例

1. 点击 **Navigation Menu** 菜单并导航至 **Vertex AI**，然后导航至 **Workbench**。

![Dive into data](images/airflow_workshop/step3.png)

1. 在 Notebook instances 页面上，点击 **New Notebook**。

2. 在 Customize instance 菜单中，选择 **TensorFlow Enterprise**，然后选择版本 **TensorFlow Enterprise 2.x (with LTS)** &gt; **Without GPUs**。

![Dive into data](images/airflow_workshop/step3notebook.png)

1. 在 **New notebook instance** 对话框中，点击铅笔图标以**编辑**实例属性。

2. 对于 **Instance name**，为您的实例输入一个名称。

3. 对于 **Region**，选择 `us-east1`，对于 **Zone**，选择所选区域内的一个分区。

4. 向下滚动到 Machine configuration，然后为 Machine type 选择 **e2-standard-2**。

5. 将其余字段保留为默认值，然后点击 **Create**。

几分钟后，Vertex AI 控制台将显示您的实例名称，随后是 **Open Jupyterlab**。

1. 点击 **Open JupyterLab**。将在新标签页中打开 JupyterLab 窗口。

## 设置环境

### 克隆实验仓库

接下来，您将在 JupyterLab 实例中克隆 `tfx` 仓库。1. 在 JupyterLab 中，点击 **Terminal** 图标打开一个新的终端。

{ql-infobox0}<strong>注</strong>：如果出现提示，请为推荐的构建点击 <code>Cancel</code>。{/ql-infobox0}

1. 要克隆 `tfx` Github 仓库，请输入以下命令，然后按 **Enter**。

```
git clone https://github.com/tensorflow/tfx.git
```

1. 要确认您已克隆仓库，请双击 `tfx` 目录并确认可以看到其中的内容。

![Transform](images/airflow_workshop/transform.png)

### 安装实验依赖项

1. 运行以下命令转到 `tfx/tfx/examples/airflow_workshop/taxi/setup/` 文件夹，然后运行 `./setup_demo.sh` 以安装实验依赖项：

```bash
cd ~/tfx/tfx/examples/airflow_workshop/taxi/setup/
./setup_demo.sh
```

上面的代码将完成以下操作：

- 安装所需软件包。
- 在主文件夹中创建一个 `airflow` 文件夹。
- 将 `dags` 文件夹从 `tfx/tfx/examples/airflow_workshop/taxi/setup/` 文件夹复制到 `~/airflow/` 文件夹中。
- 将 csv 文件从 `tfx/tfx/examples/airflow_workshop/taxi/setup/data` 复制到 `~/airflow/data` 中。

![Analyzing model performance](images/airflow_workshop/step6.png)

## 配置 Airflow 服务器

### 创建防火墙规则以在浏览器中访问 Airflow 服务器

1. 转到 `https://console.cloud.google.com/networking/firewalls/list` 并确保选择了正确的项目名称
2. 点击顶部的 `CREATE FIREWALL RULE` 选项

![Transform](images/airflow_workshop/step5tboard.png)

在 **Create a firewall 对话框**中，按照下面列出的步骤操作。

1. 对于 **Name**，输入 `airflow-tfx`。
2. 对于 **Priority**，选择 `1`。
3. 对于 **Targets**，选择 `All instances in the network`。
4. 对于 **Source IPv4 ranges**，选择 `0.0.0.0/0`
5. 对于 **Protocols and ports**，点击 `tcp` 并在 `tcp` 旁边的框中输入 `7000`
6. 点击 `Create`。

![Analyzing model performance](images/airflow_workshop/step6notebook.png)

### 从 shell 运行 Airflow 服务器

在 Jupyter Lab 终端窗口中，切换到主目录，运行 `airflow users create` 命令以便为 Airflow 创建一个管理员用户：

```bash
cd
airflow users  create --role Admin --username admin --email admin --firstname admin --lastname admin --password admin
```

然后运行 ​​`airflow webserver` 和 `airflow scheduler` 命令来运行服务器。选择端口 `7000`，因为它被允许通过防火墙。

```bash
nohup airflow webserver -p 7000 &> webserver.out &
nohup airflow scheduler &> scheduler.out &
```

### 获取您的外部 IP

1. 在 Cloud Shell 中，使用 `gcloud` 获取外部 IP。

```
gcloud compute instances list
```

![Training a Model](images/airflow_workshop/gcloud-instance-ip.png)

## 运行 DAG/流水线

### 在浏览器中

打开浏览器并转到 http://&lt;external_ip&gt;:7000

- 在登录页面中，输入您在运行 `airflow users create` 命令时选择的用户名 (`admin`) 和密码 (`admin`)。

![Training a Model](images/airflow_workshop/airflow-login.png)

Airflow 从 Python 源文件加载 DAG。它获取每个文件并相应执行。随后，它从该文件加载任何 DAG 对象。所有定义 DAG 对象的 `.py` 文件都将在 Airflow 首页中列为流水线。

在本教程中，Airflow 扫描 `~/airflow/dags/` 文件夹以查找 DAG 对象。

如果您打开 `~/airflow/dags/taxi_pipeline.py` 并滚动到底部，可以看到它创建了一个 DAG 对象并将其存储在一个名为 `DAG` 的变量中。因此，它将在 Airflow 首页中列为流水线，如下所示：

![dag-home-full.png](images/airflow_workshop/dag-home-full.png)

当您点击出租车时，您将被重定向到 DAG 的网格视图。可以点击顶部的 `Graph` 选项来获取 DAG 的计算图视图。

![airflow-dag-graph.png](images/airflow_workshop/airflow-dag-graph.png)

### 触发出租车流水线

在首页上，您可以看到可用于与 DAG 交互的按钮。

![dag-buttons.png](images/airflow_workshop/dag-buttons.png)

在 **actions** 标题下，点击 **trigger** 按钮以触发流水线。

在出租车 **DAG** 页面中，使用右侧的按钮可以随着流水线的运行刷新 DAG 的计算图视图状态。此外，还可以启用 **Auto Refresh** 来指示 Airflow 在状态更改时自动刷新计算图视图。

![dag-button-refresh.png](images/airflow_workshop/dag-button-refresh.png)

您还可以在终端中使用 [Airflow CLI](https://airflow.apache.org/cli.html) 来启用和触发 DAG：

```bash
# enable/disable
airflow pause <your DAG name>
airflow unpause <your DAG name>

# trigger
airflow trigger_dag <your DAG name>
```

#### 等待流水线处理完毕

在 DAG 视图中触发流水线后，您可以在流水线运行时观察其进度。当每个组件运行时，DAG 计算图中组件的轮廓颜色会更改，以显示相应状态。当组件完成处理后，其轮廓会变为深绿色，表示已处理完毕。

![dag-step7.png](images/airflow_workshop/dag-step7.png)

## 理解组件

现在我们将详细查看此流水线的组件，并分别查看流水线中每个步骤产生的输出。

1. 在 JupyterLab 中，转到 `~/tfx/tfx/examples/airflow_workshop/taxi/notebooks/`

2. 打开 **notebook.ipynb.** ![notebook-ipynb.png](images/airflow_workshop/notebook-ipynb.png)

3. 继续笔记本中的实验，并通过点击屏幕顶部的 **Run** (<img src="images/airflow_workshop/f1abc657d9d2845c.png" width="28.00" alt="run-button.png">) 图标运行每个单元格。或者，也可以使用 **SHIFT + ENTER** 执行单元格中的代码。

阅读叙述文字并确保您理解每个单元格中发生的事情。
