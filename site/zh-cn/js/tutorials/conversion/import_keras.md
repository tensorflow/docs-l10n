# 将 Keras 模型导入 TensorFlow.js

Keras 模型（通常通过 Python API 创建）可能保存成[多种格式之一](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model)。“完整模型”格式可以转换成 TensorFlow.js Layers 格式，这种格式可以直接加载到 TensorFlow.js 中进行推断或进一步训练。

目标 TensorFlow.js Layers 格式是一个包含 <code>model.json</code> 文件和一组二进制格式的分片权重文件的目录。<code>model.json</code> 文件包含模型拓扑（又称“架构”或“计算图”：它是对层及其连接方式的描述）和权重文件的清单。

## 要求

转换过程需要 Python 环境；您可能需要使用 [pipenv](https://github.com/pypa/pipenv) 或 [virtualenv](https://virtualenv.pypa.io) 创建一个隔离环境。要安装转换器，请使用 `pip install tensorflowjs`。

将 Keras 模型导入 TensorFlow.js 需要两个步骤。首先，将现有 Keras 模型转换成 TF.js Layers 格式，然后将其加载到 TensorFlow.js 中。

## 第1 步：将现有 Keras 模型转换成 TF.js Layers 格式

Keras 模型通常通过 `model.save(filepath)` 进行保存，这样会产生一个同时包含模型拓扑和权重的 HDF5 (.h5) 文件。要将此类文件转换成 TF.js Layers 格式，可以运行以下命令，其中 <em><code data-md-type="codespan">path/to/my_model.h5</code></em> 是源 Keras .h5 文件，而 <em><code>path/to/tfjs_target_dir</code></em> 则是 TF.js 文件的目标输出目录：

```sh
# bash

tensorflowjs_converter --input_format keras \
                       path/to/my_model.h5 \
                       path/to/tfjs_target_dir
```

## 替代方式：使用 Python API 将目录导出为 TF.js Layers 格式

如果您有一个使用 Python 编写的 Keras 模型，可以使用以下方式将其直接导出为 TensorFlow.js Layers 格式：

```py
# Python

import tensorflowjs as tfjs

def train(...):
    model = keras.models.Sequential()   # for example
    ...
    model.compile(...)
    model.fit(...)
    tfjs.converters.save_keras_model(model, tfjs_target_dir)
```

## 第 2 步：将模型加载到 TensorFlow.js 中

使用网络服务器提供您在第 1 步中生成的转换模型文件。请注意，您可能需要配置服务器以[允许跨源资源共享 (CORS)](https://enable-cors.org/)，以便允许提取 JavaScript 文件。

然后，通过提供 model.json 文件的网址将模型加载到 TensorFlow.js 中：

```js
// JavaScript

import * as tf from '@tensorflow/tfjs';

const model = await tf.loadLayersModel('https://foo.bar/tfjs_artifacts/model.json');
```

现在，模型已经可以进行推断、评估或重新训练。例如，加载的模型可以直接用于进行预测：

```js
// JavaScript

const example = tf.fromPixels(webcamElement);  // for example
const prediction = model.predict(example);
```

许多 [TensorFlow.js 示例](https://github.com/tensorflow/tfjs-examples)都采用这种方式，使用在 Google Cloud Storage 上转换和托管的预训练模型。

请注意，应使用 `model.json` 文件名引用整个模型。`loadModel(...)` 提取 `model.json`，然后通过发出额外的 HTTP(S) 请求来获取 `model.json` 权重清单中引用的分片权重文件。这种方式允许浏览器（可能还有互联网上的其他缓存服务器）缓存所有这些文件，这是因为 `model.json` 和权重碎片都小于典型的缓存文件大小限制。因此，模型可能在随后的场景中更快地加载。

## 支持的功能

TensorFlow.js Layers 目前仅支持使用标准 Keras 构造的 Keras 模型。使用不受支持的运算或层的模型（例如，自定义层、Lambda 层、自定义损失或自定义指标）无法自动导入，因为它们依赖的 Python 代码无法可靠地转换成 JavaScript。
