# TFX Airflow 教程

[![Python](https://img.shields.io/pypi/pyversions/tfx.svg?style=plastic)](https://github.com/tensorflow/tfx)[](https://github.com/tensorflow/tfx)[](https://github.com/tensorflow/tfx)
[![PyPI](https://badge.fury.io/py/tfx.svg)](https://badge.fury.io/py/tfx)[](https://badge.fury.io/py/tfx)[](https://badge.fury.io/py/tfx)

## 简介

本教程旨在介绍 TensorFlow Extended (TFX) 并帮助您学习创建自己的机器学习流水线。它在本地运行，并在 Jupyter 笔记本中显示与 TFX 和 TensorBoard 集成以及与 TFX 互动的情况。

关键术语：TFX 流水线是一种“有向无环图”，简称“DAG”。我们经常将流水线称为 DAG。

您将遵循典型的 ML 开发流程，从检查数据集开始，最后得到一个完整且有效的流水线。在此过程中，您将探索用于调试和更新流水线以及衡量性能的方式。

### 了解详情

请参阅 [TFX 用户指南](https://www.tensorflow.org/tfx/guide)了解详情

## 分步说明

您将按照典型的 ML 开发流程逐步创建流水线。具体步骤如下所示：

1. [设置环境](#step_1_setup_your_environment)
2. [创建初始流水线框架](#step_2_bring_up_initial_pipeline_skeleton)
3. [深入剖析数据](#step_3_dive_into_your_data)
4. [特征工程](#step_4_feature_engineering)
5. [训练](#step_5_training)
6. [分析模型性能](#step_6_analyzing_model_performance)
7. [为投入生产环境做好准备](#step_7_ready_for_production)

## 前提条件

- Linux / MacOS
- Virtualenv
- Python 3.5+
- Git

### 所需软件包

根据您的环境，您可能需要安装多个软件包：

```bash
sudo apt-get install \
    build-essential libssl-dev libffi-dev \
    libxml2-dev libxslt1-dev zlib1g-dev \
    python3-pip git software-properties-common
```

如果您运行的是 Python 3.6，应该安装 python3.6-dev：

```bash
sudo apt-get install python3.6-dev
```

如果您运行的是 Python 3.7，应该安装 python3.7-dev：

```bash
sudo apt-get install python3.7-dev
```

此外，如果您系统上的 GCC 版本 &lt; 7，则应更新 GCC。否则，在运行 `airflow webserver` 时会出现错误。您可以使用以下代码查看当前的版本：

```bash
gcc --version
```

如果您需要更新 GCC，可以运行以下代码：

```bash
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt update
sudo apt install gcc-7
sudo apt install g++-7
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 60 --slave /usr/bin/g++ g++ /usr/bin/g++-7
```

### MacOS 环境

如果您尚未安装 Python 3 和 Git，可以使用 [Homebrew](https://brew.sh/) 软件包管理器进行安装：

```bash
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
export PATH="/usr/local/bin:/usr/local/sbin:$PATH"
brew update
brew install python
brew install git
```

MacOS 在运行 Airflow 时有时会在派生线程时出现问题，具体取决于配置。为了避免此类问题，您应该编辑 `~/.bash_profile`，并将以下代码行添加到文件的末尾：

```bash
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
```

## 教程资料

本教程的代码可在以下位置获得：[https://github.com/tensorflow/tfx/tree/master/tfx/examples/airflow_workshop](https://github.com/tensorflow/tfx/tree/master/tfx/examples/airflow_workshop)

代码按照您要执行的步骤进行整理，因此对于每个步骤，您都会获得所需的代码以及关于使用相应代码执行哪些操作的说明。

教程文件包含练习和解决方法，可以帮助您解决遇到的困难。

#### 练习

- taxi_pipeline.py
- taxi_utils.py
- taxi DAG

#### 解决方法

- taxi_pipeline_solution.py
- taxi_utils_solution.py
- taxi_solution DAG

## 将要执行的操作

您将学习如何使用 TFX 创建 ML 流水线

- 当部署正式 ML 应用时，适合使用 TFX 流水线
- 当数据集很大时，适合使用 TFX 流水线
- 当训练/应用一致性很重要时，适合使用 TFX 流水线
- 当推断版本管理很重要时，适合使用 TFX 流水线
- Google 会在正式 ML 中使用 TFX 流水线

您将遵循典型的 ML 开发流程

- 提取、理解并清理数据
- 特征工程
- 训练
- 分析模型性能
- 不断优化
- 为投入生产环境做好准备

### 为每个步骤添加代码

本教程将所有代码都包含在文件中，但注释掉了第 3-7 步的所有代码并使用内嵌注释进行了标记。内嵌注释可以标识代码行对应的步骤。例如，第 3 步的代码标有注释 `# Step 3`。

您将为每个步骤添加的代码通常分为 4 个代码区域：

- 导入
- DAG 配置
- 从 create_pipeline() 调用返回的列表
- taxi_utils.py 中的支持代码

在学习本教程时，您需要取消注释当前所执行的教程步骤对应的代码行。这样会添加相应步骤的代码，并更新流水线。在执行此操作时，**我们强烈建议您检查要取消注释的代码**。

## 芝加哥出租车数据集

<!-- Image free for commercial use, does not require attribution:
https://pixabay.com/photos/new-york-cab-cabs-taxi-urban-city-2087998/ -->

![Taxi](images/airflow_workshop/taxi.jpg) ![Chicago taxi](images/airflow_workshop/chicago.png)

您将使用芝加哥市发布的 [Taxi Trips 数据集](https://data.cityofchicago.org/Transportation/Taxi-Trips/wrvz-psew)。

注：本网站提供的应用所使用的数据来自原始源（www.cityofchicago.org，芝加哥市官方网站），但在使用时进行了修改。芝加哥市不对本网站提供的任何数据的内容、准确性、时效性或完整性承担任何责任。本网站提供的数据可能会随时更改。您了解并同意，使用本网站提供的数据须自担风险。

您可以在 [Google BigQuery](https://cloud.google.com/bigquery/public-data/chicago-taxi) 中[详细了解](https://cloud.google.com/bigquery/)此数据集，并在 [BigQuery 界面](https://bigquery.cloud.google.com/dataset/bigquery-public-data:chicago_taxi_trips)中探索完整的数据集。

### 模型目标 - 二元分类

客户给的小费是多于还是少于 20%？

## 第 1 步：设置环境

安装脚本（`setup_demo.sh`）会安装 TFX 和 [Airflow](https://airflow.apache.org/)，并配置 Airflow 以使其易于在本教程中使用。

在 Shell 中：

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

您应该查看 `setup_demo.sh` 以了解它会执行哪些操作。

## 第 2 步：创建初始流水线框架

### Hello World

在 Shell 中：

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

您在此步骤中启动了 Jupyter 笔记本。稍后，您会在此文件夹中运行笔记本。

### 在浏览器中：

- 打开浏览器并转到 http://127.0.0.1:8080

#### 问题排查

如果您在网络浏览器中加载 Airflow 控制台时遇到任何问题，或在运行 `airflow webserver` 时出现任何错误，则可能是因为您在端口 8080 上运行了其他应用。这是 Airflow 的默认端口，但您可以将其更改为任何其他未使用的用户端口。例如，要在端口 7070 上运行 Airflow，您可以运行以下代码：

```bash
airflow webserver -p 7070
```

#### DAG 视图按钮

![DAG buttons](images/airflow_workshop/airflow_dag_buttons.png)

- 可使用左侧按钮*启用* DAG
- 进行更改后，可使用右侧按钮*刷新* DAG
- 可使用右侧按钮*触发* DAG
- 点击“taxi”可转到 DAG 的计算图视图

![Graph refresh button](https://github.com/tensorflow/docs-l10n/blob/master/site/zh-cn/tfx/tutorials/tfx/images/airflow_workshop/graph_refresh_button.png?raw=true)

#### Airflow CLI

您还可以使用 [Airflow CLI](https://airflow.apache.org/cli.html) 启用和触发 DAG：

```bash
# enable/disable
airflow pause <your DAG name>
airflow unpause <your DAG name>

# trigger
airflow trigger_dag <your DAG name>
```

#### 等待流水线处理完毕

在 DAG 视图中触发流水线后，您可以观察流水线完成处理。当每个组件运行时，DAG 计算图中组件的轮廓颜色会更改，以显示相应状态。当组件完成处理后，其轮廓会变为深绿色，表示已处理完毕。

注：运行时，您需要使用右侧的<em>计算图刷新</em>按钮或刷新页面来查看的更新状态。

到目前为止，流水线中只有 CsvExampleGen 组件，因此您需要等待其变为深绿色（约 1 分钟）。

![Setup complete](https://github.com/tensorflow/docs-l10n/blob/master/site/zh-cn/tfx/tutorials/tfx/images/airflow_workshop/step2.png?raw=true)

## 第 3 步：深入剖析数据

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

### 在编辑器中：

- 在 ~/airflow/dags 中，对 `taxi_pipeline.py` 中标有 `Step 3` 的行取消注释
- 花点时间检查取消注释的代码

### 在浏览器中：

- 点击左上角的“DAGs”链接返回 Airflow 中的 DAG 列表页面
- 点击 taxi DAG 右侧的刷新按钮
    - 此时，您应该会看到“DAG [taxi] is now fresh as a daisy”
- 触发 taxi
- 等待流水线处理完毕
    - 全部变为深绿色
    - 使用右侧的刷新按钮或刷新页面

![Dive into data](https://github.com/tensorflow/docs-l10n/blob/master/site/zh-cn/tfx/tutorials/tfx/images/airflow_workshop/step3.png?raw=true)

### 回到 Jupyter：

之前，您运行了 `jupyter notebook`，它在浏览器标签页中打开了一个 Jupyter 会话。现在，请返回浏览器中的该标签页。

- 打开 step3.ipynb
- 按照笔记本进行操作

![Dive into data](https://github.com/tensorflow/docs-l10n/blob/master/site/zh-cn/tfx/tutorials/tfx/images/airflow_workshop/step3notebook.png?raw=true)

### 更高级的示例

本教程展示的示例仅帮助您入门。如需更高级的示例，请参阅 [TensorFlow Data Validation Colab](https://www.tensorflow.org/tfx/tutorials/data_validation/chicago_taxi)。

要详细了解如何使用 TFDV 探索和验证数据集，请参阅[ tensorflow.org 中的示例](https://www.tensorflow.org/tfx/data_validation)。

## 第 4 步：特征工程

您可以通过特征工程提高数据的预测质量和/或降低维数。

- 特征交叉
- 词汇
- 嵌入向量
- PCA
- 分类编码

使用 TFX 的一个好处是，您只需编写一次转换代码，生成的转换将在训练和应用之间保持一致。

### 组件

![Transform](https://github.com/tensorflow/docs-l10n/blob/master/site/zh-cn/tfx/tutorials/tfx/images/airflow_workshop/transform.png?raw=true)

- [Transform](https://www.tensorflow.org/tfx/guide/transform) 会对数据集执行特征工程。

### 在编辑器中：

- 在 ~/airflow/dags 中，对 `taxi_pipeline.py` 和 `taxi_utils.py` 中标有 `Step 4` 的行取消注释
- 花点时间检查取消注释的代码

### 在浏览器中：

- 返回 Airflow 中的 DAG 列表页面
- 点击 taxi DAG 右侧的刷新按钮
    - 此时，您应该会看到“DAG [taxi] is now fresh as a daisy”
- 触发 taxi
- 等待流水线处理完毕
    - 全部变为深绿色
    - 使用右侧的刷新按钮或刷新页面

![Feature Engineering](https://github.com/tensorflow/docs-l10n/blob/master/site/zh-cn/tfx/tutorials/tfx/images/airflow_workshop/step4.png?raw=true)

### 回到 Jupyter：

返回浏览器中的 Jupyter 标签页。

- 打开 step4.ipynb
- 按照笔记本进行操作

### 更高级的示例

本教程展示的示例仅帮助您入门。如需更高级的示例，请参阅 [TensorFlow Transform Colab](https://www.tensorflow.org/tfx/tutorials/transform/census)。

## 第 5 步：训练

使用干净、整洁并经过转换的数据训练 TensorFlow 模型。

- 包括来自第 4 步的转换，以便一致地应用它们
- 将结果保存为 SavedModel 以便投入生产环境
- 使用 TensorBoard 呈现并探索训练过程
- 还要保存 EvalSavedModel 以分析模型的性能

### 组件

- [Trainer](https://www.tensorflow.org/tfx/guide/trainer) 会使用 TensorFlow [Estimator](https://github.com/tensorflow/docs/blob/master/site/en/r1/guide/estimators.md) 训练模型

### 在编辑器中：

- 在 ~/airflow/dags 中，对 `taxi_pipeline.py` 和 `taxi_utils.py` 中标有 `Step 5` 的行取消注释
- 花点时间检查取消注释的代码

### 在浏览器中：

- 返回 Airflow 中的 DAG 列表页面
- 点击 taxi DAG 右侧的刷新按钮
    - 此时，您应该会看到“DAG [taxi] is now fresh as a daisy”
- 触发 taxi
- 等待流水线处理完毕
    - 全部变为深绿色
    - 使用右侧的刷新按钮或刷新页面

![Training a Model](https://github.com/tensorflow/docs-l10n/blob/master/site/zh-cn/tfx/tutorials/tfx/images/airflow_workshop/step5.png?raw=true)

### 回到 Jupyter：

返回浏览器中的 Jupyter 标签页。

- 打开 step5.ipynb
- 按照笔记本进行操作

![Training a Model](https://github.com/tensorflow/docs-l10n/blob/master/site/zh-cn/tfx/tutorials/tfx/images/airflow_workshop/step5tboard.png?raw=true)

### 更高级的示例

本教程展示的示例仅帮助您入门。如需更高级的示例，请参阅 [TensorBoard 教程](https://www.tensorflow.org/tensorboard/r1/summaries)。

## 第 6 步：分析模型性能

不仅仅要了解顶级指标。

- 用户只会体验到模型的查询性能
- 顶级指标可能会掩盖部分数据切片性能不佳的问题
- 模型的公平性十分重要
- 通常，用户或数据的关键子集非常重要，并且可能会很小
    - 在重要但不常见的条件下的性能
    - 针对关键受众（如意见领袖）的性能
- 如果要替换目前在生产环境中的模型，首先应确保新模型的性能更优
- Evaluator 会告诉 Pusher 组件模型是否正常

### 组件

- [Evaluator](https://www.tensorflow.org/tfx/guide/evaluator) 会深入分析训练结果，并确保模型“足够好”，可以推送到生产环境。

### 在编辑器中：

- 在 ~/airflow/dags 中，对两个 `taxi_pipeline.py` 中标有 `Step 6` 的行取消注释
- 花点时间检查取消注释的代码

### 在浏览器中：

- 返回 Airflow 中的 DAG 列表页面
- 点击 taxi DAG 右侧的刷新按钮
    - 此时，您应该会看到“DAG [taxi] is now fresh as a daisy”
- 触发 taxi
- 等待流水线处理完毕
    - 全部变为深绿色
    - 使用右侧的刷新按钮或刷新页面

![Analyzing model performance](https://github.com/tensorflow/docs-l10n/blob/master/site/zh-cn/tfx/tutorials/tfx/images/airflow_workshop/step6.png?raw=true)

### 回到 Jupyter：

返回浏览器中的 Jupyter 标签页。

- 打开 step6.ipynb
- 按照笔记本进行操作

![Analyzing model performance](https://github.com/tensorflow/docs-l10n/blob/master/site/zh-cn/tfx/tutorials/tfx/images/airflow_workshop/step6notebook.png?raw=true)

### 更高级的示例

本教程展示的示例仅帮助您入门。如需更高级的示例，请参阅 [TFMA 芝加哥出租车教程](https://www.tensorflow.org/tfx/tutorials/model_analysis/chicago_taxi)。

## 第 7 步：为投入生产环境做好准备

如果新模型已准备就绪，请进行相应设置。

- Pusher 会将 SavedModels 部署到已知位置

部署目标会从已知位置接收新模型

- TensorFlow Serving
- TensorFlow Lite
- TensorFlow JS
- TensorFlow Hub

### 组件

- [Pusher](https://www.tensorflow.org/tfx/guide/pusher) 会将模型部署到应用基础架构。

### 在编辑器中：

- 在 ~/airflow/dags 中，对两个 `taxi_pipeline.py` 中标有 `Step 7` 的行取消注释
- 花点时间检查取消注释的代码

### 在浏览器中：

- 返回 Airflow 中的 DAG 列表页面
- 点击 taxi DAG 右侧的刷新按钮
    - 此时，您应该会看到“DAG [taxi] is now fresh as a daisy”
- 触发 taxi
- 等待流水线处理完毕
    - 全部变为深绿色
    - 使用右侧的刷新按钮或刷新页面

![Ready for production](images/airflow_workshop/step7.png)

## 后续步骤

您现在已经训练并验证了模型，并在 `~/airflow/saved_models/taxi` 目录下导出了 `SavedModel` 文件。您的模型现在可以投入生产环境了。您现在可以将模型部署到任何 TensorFlow 部署目标，包括：

- [TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving)，用于在服务器或服务器场中应用模型，并处理 REST 和/或 gRPC 推断请求。
- [TensorFlow Lite](https://www.tensorflow.org/lite)，用于将模型包含在 Android 或 iOS 原生移动应用中，或包含在 Raspberry Pi、IoT 或微控制器应用中。
- [TensorFlow.js](https://www.tensorflow.org/js)，用于在网络浏览器或 Node.JS 应用中运行模型。
