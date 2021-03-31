<!--* freshness: { owner: 'wgierke' reviewed: '2021-02-25' review_interval: '3 months' } *-->

# 编写模型文档

要向 tfhub.dev 贡献模型，必须提供 Markdown 格式的文档。有关将模型添加到 tfhub.dev 的过程的完整概述，请参阅[贡献模型](contribute_a_model.md)指南。

## Markdown 文档类型

tfhub.dev 中使用了以下 3 种 Markdown 文档类型：

- 发布者 Markdown - 包含有关发布者的信息（请参阅[成为发布者](publish.md)指南了解详情）。
- 模型 Markdown - 包含有关特定模型的信息。
- 集合 Markdown - 包含有关由发布者定义的模型集合的信息（请参阅[创建集合](creating_a_collection.md)指南了解详情）。

## 内容组织

向 [TensorFlow Hub GitHub](https://github.com/tensorflow/hub) 仓库贡献内容时，建议使用以下内容组织：

- 每个发布者目录均位于 `assets` 目录下
- 每个发布者目录均包含可选的 `models` 和 `collections` 目录
- 每个模型均应在 `assets/publisher_name/models` 下具有自己的目录
- 每个集合均应在 `assets/publisher_name/collections` 下具有自己的目录

发布者与集合的 Markdown 文件不受版本控制，而模型可以具有不同的版本。每个模型版本都需要一个单独的 Markdown 文件，以其描述的版本命名（即 1.md、2.md）。

给定模型的所有模型版本应位于模型目录中。

下图展示了 Markdown 内容的组织方式：

```
assets
├── publisher_name_a
│   ├── publisher_name_a.md  -> Documentation of the publisher.
│   └── models
│       └── model          -> Model name with slashes encoded as sub-path.
│           ├── 1.md       -> Documentation of the model version 1.
│           └── 2.md       -> Documentation of the model version 2.
├── publisher_name_b
│   ├── publisher_name_b.md  -> Documentation of the publisher.
│   ├── models
│   │   └── ...
│   └── collections
│       └── collection     -> Documentation for the collection feature.
│           └── 1.md
├── publisher_name_c
│   └── ...
└── ...
```

## 模型页面特定的 Markdown 格式

模型文档是带有某些附加语法的 Markdown 文件。请参见下文提供的精简示例，或参见[更真实的 Markdown 文件示例](https://github.com/tensorflow/tfhub.dev/blob/master/examples/docs/tf2_model_example.md)。

### 示例文档

高质量的模型文档中包含代码段，以及有关如何训练模型和模型预期用途的信息。您还应该使用[下述](#model-markdown-specific-metadata-properties)模型特定的元数据属性，以便用户在 tfhub.dev 上更快地找到您的模型。

```markdown
# Module google/text-embedding-model/1

Simple one sentence description.

<!-- asset-path: https://path/to/text-embedding-model/model.tar.gz -->
<!-- module-type: text-embedding -->
<!-- fine-tunable: true -->
<!-- format: saved_model_2 -->

## Overview

Here we give more information about the model including how it was trained,
expected use cases, and code snippets demonstrating how to use the model:

``
Code snippet demonstrating use (e.g. for a TF model using the tensorflow_hub library)

import tensorflow_hub as hub

model = hub.KerasLayer(<model name>)
inputs = ...
output = model(inputs)
``
```

### 模型部署以及将部署分组到一起

tfhub.dev 支持发布 TensorFlow 模型的 TF.js、TFLite 和 Coral 部署。

Markdown 文件的首行应指定部署格式的类型：

- `# Tfjs publisher/model/version` 用于 TF.js 部署
- `# Lite publisher/model/version` 用于 Lite 部署
- `# Coral publisher/model/version` 用于 Coral 部署

建议您将这些不同部署显示在 tfhub.dev 上的同一模型页面中。要将给定的 TF.js、TFLite 或 Coral 部署关联至 TensorFlow 模型，请指定 parent-model 标记：

```markdown
<!-- parent-model: publisher/model/version -->
```

有时，您可能需要发布一个或多个不含 TensorFlow SavedModel 的部署。这种情况下，您需要创建一个占位符模型，并在 `parent-model` 标记中指定其句柄。占位符 Markdown 与 TensorFlow 模型 Markdown 基本相同，区别在于其首行为 `# Placeholder publisher/model/version` 并且不需要 `asset-path` 属性。

### 模型 Markdown 特定的元数据属性

Markdown 文件可以包含元数据属性。这些属性以 Markdown 注释形式列于 Markdown 文件描述的下方。

```
# Module google/universal-sentence-encoder/1
Encoder of greater-than-word length text trained on a variety of data.

<!-- module-type: text-embedding -->
...
```

存在以下元数据属性：

- `format`：对于 TensorFlow 模型：模型的 TensorFlow Hub 格式。有效值为 `hub`（当模型通过旧版 [TF1 Hub 格式](exporting_hub_format.md)导出时）或 `saved_model_2`（当模型通过 [TF2 SavedModel](exporting_tf2_saved_model.md) 格式导出时）。
- `asset-path`：指向要上传（例如上传到 Google Cloud Storage 存储分区）的实际模型素材资源的全局可读远程路径。该网址应支持由 robots.txt 文件提取（因此将不支持 "https://github.com/.*/releases/download/.*"，因为 https://github.com/robots.txt 禁止该网址）
- `parent-model`：对于 TF.js/TFLite/Coral 模型：随附的 SavedModel/Placeholder 的句柄
- `module-type`：问题领域，例如“文本嵌入向量”或“图像分类”
- `dataset`：训练模型的数据集，例如“ImageNet-21k”或“Wikipedia”
- `network-architecture`：模型基于的网络架构，例如“BERT”或“Mobilenet V3”
- `language`：训练文本模型所用语言的语言代码，例如“en”或“fr”
- `fine-tunable`：布尔值，用户是否可以微调模型
- `license`：适用于模型的许可证。已发布模型的默认假定许可证为 [Apache 2.0 License](https://opensource.org/licenses/Apache-2.0)。[OSI 批准的许可证](https://opensource.org/licenses)中列出了其他可接受的选项。可能的（字面）值为：`Apache-2.0`、`BSD-3-Clause`、`BSD-2-Clause`、`GPL-2.0`、`GPL-3.0`、`LGPL-2.0`、`LGPL-2.1`、`LGPL-3.0`、`MIT`、`MPL-2.0`、`CDDL-1.0`、`EPL-2.0`、`custom`。请注意，自定义许可证需视具体情况特别考虑。

Markdown 文档类型支持不同的必选和可选元数据属性：

类型 | 必选 | 可选
--- | --- | ---
发布者 |  |
集合 | module-type | dataset、language、
:             :                          : network-architecture             : |  |
占位符 | module-type | dataset、fine-tunable、language、
:             :                          : license、network-architecture    : |  |
SavedModel | asset-path、module-type、 | dataset、language、license、
:             : fine-tunable、format     : network-architecture             : |  |
Tfjs | asset-path、parent-model |
Lite | asset-path、parent-model |
Coral | asset-path、parent-model | 
