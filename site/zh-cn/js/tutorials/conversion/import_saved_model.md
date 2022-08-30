# 将基于 TensorFlow GraphDef 的模型导入 TensorFlow.js

基于 TensorFlow GraphDef 的模型（一般通过 Python API 创建）可以保存成以下几种格式：

1. TensorFlow [SavedModel](https://www.tensorflow.org/tutorials/keras/save_and_load)
2. <a>冻结模型</a>
3. [Tensorflow Hub 模块](https://www.tensorflow.org/hub/)

以上所有格式都可以通过 [TensorFlow.js 转换器](https://github.com/tensorflow/tfjs-converter)转换成可直接加载到 TensorFlow.js 中进行推断的格式。

（注：TensorFlow 已弃用会话包格式，请将您的模型迁移至 SavedModel 格式。）

## 必要条件

转换过程需要 Python 环境；您可能需要使用 [pipenv](https://github.com/pypa/pipenv) 或 [virtualenv](https://virtualenv.pypa.io) 创建一个隔离环境。要安装转换器，请运行以下命令：

```bash
 pip install tensorflowjs
```

将 TensorFlow 模型导入 TensorFlow.js 需要两个步骤。首先，将现有模型转换成 TensorFlow.js 网络格式，然后将其加载到 TensorFlow.js 中。

## 第1 步：将现有 TensorFlow 模型转换成 TensorFlow.js 网络格式

运行 pip 软件包提供的转换器脚本：

使用方法：SavedModel 示例：

```bash
tensorflowjs_converter \
    --input_format=tf_saved_model \
    --output_node_names='MobilenetV1/Predictions/Reshape_1' \
    --saved_model_tags=serve \
    /mobilenet/saved_model \
    /mobilenet/web_model
```

冻结模型示例：

```bash
tensorflowjs_converter \
    --input_format=tf_frozen_model \
    --output_node_names='MobilenetV1/Predictions/Reshape_1' \
    /mobilenet/frozen_model.pb \
    /mobilenet/web_model
```

Tensorflow Hub 模块示例：

```bash
tensorflowjs_converter \
    --input_format=tf_hub \
    'https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/classification/1' \
    /mobilenet/web_model
```

脚本参数 | 描述
--- | ---
`input_path` | SavedModel 目录、会话包目录、冻结模型文件的完整路径，TensorFlow Hub 模块句柄或路径。
`output_path` | 所有输出工件的路径。

选项 | 描述
--- | ---
`--input_format` | 输入模型的格式。SavedModel 为 tf_saved_model，冻结模型为 tf_frozen_model，会话包为 tf_session_bundle，TensorFlow Hub 模块为 tf_hub，Keras HDF5 为 keras。
`--output_node_names` | 输出节点的名称，用逗号分隔。
`--saved_model_tags` | 仅适用于 SavedModel 转换，要加载的 MetaGraphDef 的标记，用逗号分隔。默认为 `serve`。
`--signature_name` | 仅适用于 TensorFlow Hub 模块转换，要加载的签名。默认为 `default`。请参阅 https://tensorflow.google.cn/hub/common_signatures/。

使用以下命令获取详细的帮助消息：

```bash
tensorflowjs_converter --help
```

### 转换器生成的文件

上述转换脚本会产生两种类型的文件：

- `model.json` （数据流图和权重清单）
- `group1-shard\*of\*` （二进制权重文件）

例如，以下是转换 MobileNet v2 的输出：

```html
  output_directory/model.json
  output_directory/group1-shard1of5
  ...
  output_directory/group1-shard5of5
```

## 第 2 步：在浏览器加载并运行模型

1. 安装 tfjs-convert npm 软件包

`yarn add @tensorflow/tfjs` 或 `npm install @tensorflow/tfjs`

1. 实例化 [FrozenModel 类](https://github.com/tensorflow/tfjs-converter/blob/master/src/executor/frozen_model.ts)并运行推断：

```js
import * as tf from '@tensorflow/tfjs';
import {loadGraphModel} from '@tensorflow/tfjs-converter';

const MODEL_URL = 'model_directory/model.json';

const model = await loadGraphModel(MODEL_URL);
const cat = document.getElementById('cat');
model.execute(tf.browser.fromPixels(cat));
```

请查看我们的 [MobileNet 演示](https://github.com/tensorflow/tfjs-converter/tree/master/demo/mobilenet)。

`loadGraphModel` API 接受一个附加 `LoadOptions` 参数，该参数可以用于随请求发送凭据或自定义头。请参阅 [loadGraphModel() 文档](https://js.tensorflow.org/api/1.0.0/#loadGraphModel)了解更多详细信息。

## 受支持的运算

目前，TensorFlow.js 只支持有限的 TensorFlow 运算。如果您的模型使用不受支持的运算，`tensorflowjs_converter` 脚本将失败并打印您的模型中不受支持的运算的列表。请在 GitHub 上提交[议题](https://github.com/tensorflow/tfjs/issues)告诉我们您需要支持的运算。

## 仅加载权重

如果您想仅加载权重，可以使用以下代码段：

```js
import * as tf from '@tensorflow/tfjs';

const weightManifestUrl = "https://example.org/model/weights_manifest.json";

const manifest = await fetch(weightManifestUrl);
this.weightManifest = await manifest.json();
const weightMap = await tf.io.loadWeights(
        this.weightManifest, "https://example.org/model");
```
