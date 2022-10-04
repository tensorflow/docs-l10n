<!--* freshness: { owner: 'wgierke' reviewed: '2022-07-27' review_interval: '6 months' } *-->

# 编写文档

要向 tfhub.dev 贡献内容，必须提供 Markdown 格式的文档。有关向 tfhub.dev 贡献模型的过程的完整概述，请参阅[贡献模型](contribute_a_model.md)指南。

## Types of Markdown documentation

There are 3 types of Markdown documentation used in tfhub.dev:

- 发布者 Markdown - 有关发布者的信息（[请参阅 markdown 语法](#publisher)）
- 模型 Markdown - 有关特定模型及其使用方法的信息（[请参阅 markdown 语法](#model)）
- 集合 Markdown - 包含有关由发布者定义的模型集合的信息（[请参阅 markdown 语法](#collection)）

## Content organization

向 [TensorFlow Hub GitHub](https://github.com/tensorflow/tfhub.dev) 仓库贡献内容时，需要使用以下内容组织：

- 每个发布者目录都在 `assets/docs` 目录中
- 每个发布者目录都包含可选的 `models` 和 `collections` 目录
- 每个模型都应该在 `assets/docs/<publisher_name>/models` 下有自己的目录
- 每个集合都应该在 `assets/docs/<publisher_name>/collections` 下有自己的目录

发布者 Markdown 不受版本控制，而模型可以具有不同的版本。每个模型版本都需要一个单独的 Markdown 文件，以其描述的版本命名（即 1.md、2.md）。集合 Markdown 有版本控制，但只支持单个版本 (1.md)。

All model versions for a given model should be located in the model directory.

Below is an illustration on how the Markdown content is organized:

```
assets/docs
├── <publisher_name_a>
│   ├── <publisher_name_a>.md  -> Documentation of the publisher.
│   └── models
│       └── <model_name>       -> Model name with slashes encoded as sub-path.
│           ├── 1.md           -> Documentation of the model version 1.
│           └── 2.md           -> Documentation of the model version 2.
├── <publisher_name_b>
│   ├── <publisher_name_b>.md  -> Documentation of the publisher.
│   ├── models
│   │   └── ...
│   └── collections
│       └── <collection_name>
│           └── 1.md           -> Documentation for the collection.
├── <publisher_name_c>
│   └── ...
└── ...
```

## 发布者 markdown 格式 {:#publisher}

用于声明发布者文档的 Markdown 文件与模型 Markdown 文件类型相同，但语法上略有不同。

TensorFlow Hub 仓库中存储发布者文件的正确位置为：[tfhub.dev/assets/docs](https://github.com/tensorflow/tfhub.dev/tree/master/assets/docs)/&lt;publisher_id&gt;/&lt;publisher_id.md&gt;

请参见以下精简的“vtab”发布者文档示例：

```markdown
# Publisher vtab
Visual Task Adaptation Benchmark

[![Icon URL]](https://storage.googleapis.com/vtab/vtab_logo_120.png)

## VTAB
The Visual Task Adaptation Benchmark (VTAB) is a diverse, realistic and
challenging benchmark to evaluate image representations.
```

上面的示例指定了发布者 ID、发布者名称、要使用的图标的路径以及较长的自由格式 markdown 文档。请注意，发布者 ID 应该仅包含小写字母、数字和连字符。

### Publisher name guideline

Your publisher name can be your GitHub username or the name of the GitHub organization you manage.

## 模型页面 markdown 格式 {:#model}

The model documentation is a Markdown file with some add-on syntax. See the example below for a minimal example or [a more realistic example Markdown file](https://github.com/tensorflow/tfhub.dev/blob/master/examples/docs/tf2_model_example.md).

### Example documentation

A high-quality model documentation contains code snippets, information how the model was trained and intended usage. You should also make use of model-specific metadata properties [explained below](#metadata) so users can find your models on tfhub.dev faster.

```markdown
# Module google/text-embedding-model/1

Simple one sentence description.

<!-- asset-path: https://path/to/text-embedding-model/model.tar.gz -->
<!-- task: text-embedding -->
<!-- fine-tunable: true -->
<!-- format: saved_model_2 -->

## Overview

Here we give more information about the model including how it was trained,
expected use cases, and code snippets demonstrating how to use the model:

```
Code snippet demonstrating use (e.g. for a TF model using the tensorflow_hub library)

import tensorflow_hub as hub

model = hub.KerasLayer(<model name>)
inputs = ...
output = model(inputs)
```
```

### Model deployments and grouping deployments together

tfhub.dev 允许发布 TensorFlow SavedModel 的 TF.js、TFLite 和 Coral 部署。

Markdown 文件的首行应指定部署格式的类型：

- `# Module publisher/model/version` 用于 SavedModel
- `# Tfjs publisher/model/version` for TF.js deployments
- `# Lite publisher/model/version` for Lite deployments
- `# Coral publisher/model/version` for Coral deployments

最好将这些同一概念模型的不同格式显示在 tfhub.dev 上的同一模型页面中。要将给定的 TF.js、TFLite 或 Coral 部署关联至 TensorFlow SavedModel 模型，请指定 parent-model 标记：

```markdown
<!-- parent-model: publisher/model/version -->
```

Sometimes you might want to publish one or more deployments without a TensorFlow SavedModel. In that case, you'll need to create a Placeholder model and specify its handle in the `parent-model` tag. The placeholder Markdown is identical to TensorFlow model Markdown, except that the first line is: `# Placeholder publisher/model/version` and it doesn't require the `asset-path` property.

### 模型 Markdown 特定的元数据属性 {:#metadata}

Markdown 文件可以包含元数据属性。这些属性用于提供筛选和标记以帮助用户查找模型。元数据属性作为 Markdown 注释包含在 Markdown 文件的简短描述之后，例如

```markdown
# Module google/universal-sentence-encoder/1
Encoder of greater-than-word length text trained on a variety of data.

<!-- task: text-embedding -->
...
```

支持以下元数据属性：

- `format`：用于 TensorFlow 模型：模型的 TensorFlow Hub 格式。有效值为 `hub` （当模型通过旧版 [TF1 hub 格式](exporting_hub_format.md)导出时）或 `saved_model_2`（当模型通过 [TF2 Saved Model](exporting_tf2_saved_model.md) 导出时）。
- `asset-path`：要上传的实际模型资产的全球可读远程路径，例如 Google Cloud Storage 存储分区路径。应该通过 robots.txt 文件允许该 URL 被获取（因此，将不支持“https://github.com/.*/releases/download/.*”，因为 https://github.com/robots.txt 禁止该网址）。有关预期的文件类型和内容的更多信息，请参见[下文](#model-specific-asset-content)。
- `parent-model`：用于 TF.js/TFLite/Coral 模型：随附的 SavedModel/Placeholder 的句柄
- `fine-tunable`：布尔值，表示用户是否可以微调模型。
- `task`：问题域，例如“text-embedding”。[task.yaml](https://github.com/tensorflow/tfhub.dev/blob/master/tags/task.yaml) 中定义了所有支持的值。
- `dataset`：用于训练模型的数据集，例如“wikipedia”。[dataset.yaml](https://github.com/tensorflow/tfhub.dev/blob/master/tags/dataset.yaml) 中定义了所有支持的值。
- `network-architecture`：模型所基于的网络架构，例如“mobilenet-v3”。[network_architecture.yaml](https://github.com/tensorflow/tfhub.dev/blob/master/tags/network_architecture.yaml) 中定义了所有支持的值。
- `language`：训练文本模型所用语言的语言代码，例如“en”。[language.yaml](https://github.com/tensorflow/tfhub.dev/blob/master/tags/language.yaml) 中定义了所有支持的值。
- `license`：适用于模型的许可证，例如“mit”。已发布模型的默认假定许可证为 [Apache 2.0 License](https://opensource.org/licenses/Apache-2.0)。[license.yaml](https://github.com/tensorflow/tfhub.dev/blob/master/tags/license.yaml) 中定义了所有支持的值。请注意，`custom` 许可证需要根据具体情况进行特殊考虑。
- `colab`：演示如何使用或训练模型（[bigbigan-resnet50](https://tfhub.dev/deepmind/bigbigan-resnet50/1) [示例](https://colab.sandbox.google.com/github/tensorflow/hub/blob/master/examples/colab/bigbigan_with_tf_hub.ipynb)）的笔记本的 HTTPS URL。必须指向 `colab.research.google.com`。请注意，托管在 GitHub 上的 Jupyter 笔记本可以通过 `https://colab.research.google.com/github/ORGANIZATION/PROJECT/ blob/master/.../my_notebook.ipynb` 访问。
- `demo`：演示如何使用 TF.js 模型（[posenet](https://tfhub.dev/tensorflow/tfjs-model/posenet/mobilenet/float/075/1/default/1) [示例](https://teachablemachine.withgoogle.com/train/pose)）的网站的 HTTPS URL。
- `interactive-visualizer`：应嵌入到模型页面上的可视化工具的名称，例如“vision”。通过显示可视化工具，用户能以交互方式探索模型的预测。[interactive_visualizer.yaml](https://github.com/tensorflow/tfhub.dev/blob/master/tags/interactive_visualizer.yaml) 中定义了所有支持的值。

The Markdown documentation types support different required and optional metadata properties:

类型 | Required | Optional
--- | --- | ---
Publisher |  |
Collection | task | dataset、language
:             :                          : network-architecture                : |  |
Placeholder | task | dataset、fine-tunable
:             :                          : interactive-visualizer、language   : |  |
:             :                          : license、network-architecture       : |  |
SavedModel | asset-path、task | colab、dataset
:             : fine-tunable、format     : interactive-visualizer、language,   : |  |
:             :                          : license、network-architecture       : |  |
Tfjs | asset-path, parent-model | colab、demo、interactive-visualizer
Lite | asset-path, parent-model | colab、interactive-visualizer
Coral | asset-path, parent-model | colab、interactive-visualizer

### 模型特定的资产内容

根据模型类型，预期有以下文件类型和内容：

- SavedModel：包含类似如下内容的 tar.gz 归档：

```
saved_model.tar.gz
├── assets/            # Optional.
├── assets.extra/      # Optional.
├── variables/
│     ├── variables.data-?????-of-?????
│     └──  variables.index
├── saved_model.pb
├── keras_metadata.pb  # Optional, only required for Keras models.
└── tfhub_module.pb    # Optional, only required for TF1 models.
```

- TF.js：包含类似如下内容的 tar.gz 归档：

```
tf_js_model.tar.gz
├── group*
├── *.json
├── *.txt
└── *.pb
```

- TFLite：.tflite 文件
- Coral：.tflite 文件

对于 tar.gz 归档：假定您的模型文件在目录 `my_model` 中（例如，`my_model/saved_model.pb` 为 SavedModel，或者 `my_model/model.json` 为 TF.js 模型），您可以通过 `cd my_model && tar -czvf ../model.tar.gz *` 使用 [tar](https://www.gnu.org/software/tar/manual/tar.html) 工具创建有效的 tar.gz 归档。

通常，所有文件和目录（无论是否经过压缩）都必须以单词字符开头，例如，点不是文件名/目录的有效前缀。

## 集合页面 markdown 格式 {:#collection}

Collections are a feature of tfhub.dev that enables publishers to bundle related models together to improve user search experience.

See the [list of all collections](https://tfhub.dev/s?subtype=model-family) on tfhub.dev.

[github.com/tensorflow/tfhub.dev](https://github.com/tensorflow/tfhub.dev) 仓库中集合文件的正确位置为 [assets/docs](https://github.com/tensorflow/tfhub.dev/tree/master/assets/docs)/<b>&lt;发布者名称&gt;</b>/collections/<b>&lt;集合名称&gt;</b>/<b>1</b>.md

以下是进入 assets/docs/<b>vtab</b>/collections/<b>benchmark</b>/<b>1</b>.md 的最小示例。请注意，第一行中集合的名称不包括 `collections/` 部分，该部分包括在文件路径中。

```markdown
# Collection vtab/benchmark/1
Collection of visual representations that have been evaluated on the VTAB
benchmark.

<!-- task: image-feature-vector -->

## Overview
This is the list of visual representations in TensorFlow Hub that have been
evaluated on VTAB. Results can be seen in
[google-research.github.io/task_adaptation/](https://google-research.github.io/task_adaptation/)

#### Models
|                   |
|-------------------|
| [vtab/sup-100/1](https://tfhub.dev/vtab/sup-100/1)   |
| [vtab/rotation/1](https://tfhub.dev/vtab/rotation/1) |
|------------------------------------------------------|
```

The example specifies the name of the collection, a short one sentence description, problem domain metadata and free-form markdown documentation.
