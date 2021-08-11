# 保存和加载模型

TensorFlow.js 提供了保存和加载模型的功能，这些模型可以使用 [`Layers`](https://js.tensorflow.org/api/0.14.2/#Models) API 创建或从现有 TensorFlow 模型转换而来。可能是您自己训练的模型，也可能是其他人训练的模型。使用 Layers API 的一个主要好处是，使用它创建的模型是可序列化模型，这就是我们将在本教程中探讨的内容。

本教程将重点介绍如何保存和加载 TensorFlow.js 模型（可通过 JSON 文件识别）。我们也可以导入 TensorFlow Python 模型。以下两个教程介绍了如何加载这些模型：

- [导入 Keras 模型](https://gitlocalize.com/repo/4592/zh-cn/site/en-snapshot/js/tutorials/conversion/import_keras.md)
- [导入 Graphdef 模型](https://gitlocalize.com/repo/4592/zh-cn/site/en-snapshot/js/tutorials/conversion/import_saved_model.md)

## 保存 tf.Model

[`tf.Model`](https://js.tensorflow.org/api/0.14.2/#class:Model) 和 [`tf.Sequential`](https://js.tensorflow.org/api/0.14.2/#class:Model) 都提供了 [`model.save`](https://js.tensorflow.org/api/0.14.2/#tf.Model.save) 函数，您可以借助该函数保存模型的*拓扑*和*权重* 。

- 拓扑：这是一个描述模型架构的文件（例如模型使用了哪些运算）。它包含对外部存储的模型权重的引用。

- 权重：这些是以有效格式存储给定模型权重的二进制文件。它们通常存储在与拓扑相同的文件夹中。

我们来看看用于保存模型的代码：

```js
const saveResult = await model.save('localstorage://my-model-1');
```

一些需要注意的地方：

- `save` 方法采用以**协议名称**开头的类网址字符串参数。它描述了我们想保存模型的地址的类型。在上例中，协议名称为 `localstorage://`。
- 协议名称之后是**路径**。在上例中，路径是 `my-model-1`。
- `save` 方法是异步的。
- `model.save` 的返回值是一个 JSON 对象，包含模型的拓扑和权重的字节大小等信息。
- 用于保存模型的环境不会影响可以加载模型的环境。在 node.js 中保存模型不会阻碍在浏览器中加载模型。

我们将在下面查看不同协议名称。

### 本地存储空间（仅限浏览器）

**协议名称：** `localstorage://`

```js
await model.save('localstorage://my-model');
```

这会在浏览器的[本地存储空间](https://developer.mozilla.org/en-US/docs/Web/API/Window/localStorage)中以名称 `my-model` 保存模型。这样能够在浏览器刷新后保持不变，而当存储空间成为问题时，用户或浏览器本身可以清除本地存储。每个浏览器还可为给定域设置本地存储空间中可以存储的数据量。

### IndexedDB（仅限浏览器）

**协议名称：** `indexeddb://`

```js
await model.save('indexeddb://my-model');
```

这会将模型保存到浏览器的 [IndexedDB](https://developer.mozilla.org/en-US/docs/Web/API/IndexedDB_API) 存储空间中。与本地存储一样，它在刷新后仍然存在，同时所存储对象大小的上限更高。

### 文件下载（仅限浏览器）

**协议名称：** `downloads://`

```js
await model.save('downloads://my-model');
```

这会让浏览器将模型文件下载至用户的机器上。将生成两个文件：

1. 一个名为 `[my-model].json` 的 JSON 文本文件，其中包含模型拓扑和对下文所述权重文件的引用。
2. 一个二进制文件，其中包含名为 `[my-model].weights.bin` 的权重值。

您可以更改 `[my-model]` 名称以获得一个名称不同的文件。

由于 `.json` 文件使用相对路径指向 `.bin`，因此两个文件应位于同一个文件夹中。

> 注：某些浏览器要求用户先授予权限，然后才能同时下载多个文件。

### HTTP(S) 请求

**协议名称**：`http://` 或 `https://`

```js
await model.save('http://model-server.domain/upload')
```

这将创建一个 Web 请求，以将模型保存到远程服务器。您应该控制该远程服务器，确保它能够处理该请求。

模型将通过 [POST](https://developer.mozilla.org/en-US/docs/Web/HTTP/Methods/POST) 请求发送至指定的 HTTP 服务器。POST 主体采用 `multipart/form-data` 格式并包含两个文件：

1. 一个名为 `model.json` 的 JSON 文本文件，其中包含模型拓扑和对下文所述权重文件的引用。
2. 一个二进制文件，其中包含名为 `model.weights.bin` 的权重值。

请注意，这两个文件的名称需要始终与上面所指定的完全相同（因为名称内置于函数中）。此 [API 文档](https://js.tensorflow.org/api/latest/#tf.io.browserHTTPRequest)包含一个 Python 代码段，演示了如何使用 [Flask](http://flask.pocoo.org/) Web 框架处理源自 `save` 的请求。

通常，您必须向 HTTP 服务器传递更多参数或请求头（例如，用于身份验证，或者如果要指定应保存模型的文件夹）。您可以通过替换 `tf.io.browserHTTPRequest` 中的网址字符串参数来获得对来自 `save` 的请求在这些方面的细粒度控制。此 API 在控制 HTTP 请求方面提供了更大的灵活性。

例如：

```js
await model.save(tf.io.browserHTTPRequest(
    'http://model-server.domain/upload',
    {method: 'PUT', headers: {'header_key_1': 'header_value_1'}}));
```

### 原生文件系统（仅限 Node.js）

**协议名称：** `file://`

```js
await model.save('file:///path/to/my-model');
```

在 Node.js 上运行时，我们还可以直接访问文件系统并保存模型。上面的命令会将两个文件保存到在 `scheme` 后指定的 `path` 中。

1. 一个名为 `[model].json` 的 JSON 文本文件，其中包含模型拓扑和对下文所述权重文件的引用。
2. 一个二进制文件，其中包含名为 `[model].weights.bin` 的权重值。

请注意，这两个文件的名称需要始终与上面所指定的完全相同（因为名称内置于函数中）。

## 加载 tf.Model

给定一个使用上述方法之一保存的模型，我们可以使用 `tf.loadLayersModel` API 加载它。

我们来看看加载模型的代码：

```js
const model = await tf.loadLayersModel('localstorage://my-model-1');
```

一些需要注意的地方：

- 与 `model.save()` 类似，`loadLayersModel` 函数也采用以<strong>协议名称</strong>开头的类网址字符串参数。它描述了我们想要从中加载模型的目标类型。
- 协议名称之后是**路径**。在上例中，路径是 `my-model-1`。
- 类网址字符串可以替换为与 IOHandler 接口匹配的对象。
- `tf.loadLayersModel()` 函数是异步的。
- `tf.loadLayersModel` 的返回值为 `tf.Model`。

我们将在下面查看不同协议名称。

### 本地存储空间（仅限浏览器）

**协议名称：** `localstorage://`

```js
const model = await tf.loadLayersModel('localstorage://my-model');
```

这将从浏览器的[本地存储空间](https://developer.mozilla.org/en-US/docs/Web/API/Window/localStorage)加载一个名为 `my-model` 的模型。

### IndexedDB（仅限浏览器）

**协议名称：** `indexeddb://`

```js
const model = await tf.loadLayersModel('indexeddb://my-model');
```

这将从浏览器的 [IndexedDB](https://developer.mozilla.org/en-US/docs/Web/API/IndexedDB_API) 存储空间加载一个模型。

### HTTP(S)

**协议名称**：`http://` 或 `https://`

```js
const model = await tf.loadLayersModel('http://model-server.domain/download/model.json');
```

这将从 HTTP 端点加载模型。加载 `json` 文件后，函数将请求 `json` 文件引用的对应 `.bin` 文件。

> 注：此实现依赖于 [`fetch`](https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API/Using_Fetch) 方法，如果您的环境没有提供原生 fetch 方法，您可以提供满足接口要求的全局方法名称 [`fetch`](https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API/Using_Fetch)，或者使用类似于 (`node-fetch`)[https://www.npmjs.com/package/node-fetch] 的库。

### 原生文件系统（仅限 Node.js）

**协议名称：** `file://`

```js
const model = await tf.loadLayersModel('file://path/to/my-model/model.json');
```

在 Node.js 上运行时，我们还可以直接访问文件系统并加载模型。请注意，在上面的函数调用中，我们引用 model.json 文件本身（在保存时，我们指定一个文件夹）。对应的 `.bin` 文件应与 `json` 文件位于同一个文件夹中。

## 使用 IOHandler 加载模型

如果上述协议名称没有满足您的需求，您可以使用 `IOHandler` 实现自定义加载行为。Tensorflow.js 提供的一个 `IOHandler` 是 [`tf.io.browserFiles`](https://js.tensorflow.org/api/latest/#io.browserFiles)，它允许浏览器用户在浏览器中上传模型文件。请参阅[文档](https://js.tensorflow.org/api/latest/#io.browserFiles)了解更多信息。

# 使用自定义 IOHandler 保存或加载模型

如果上述协议名称没有满足您的保存或加载需求，您可以通过实现 `IOHandler` 来实现自定义序列化行为。

`IOHandler` 是一个包含 `save` 和 `load` 方法的对象。

`save` 函数采用一个与 [ModelArtifacts](https://github.com/tensorflow/tfjs-core/blob/master/src/io/types.ts#L165) 接口匹配的参数，应返回一个解析为 [SaveResult](https://github.com/tensorflow/tfjs-core/blob/master/src/io/types.ts#L107) 对象的 promise。

`load` 函数不采用参数，应返回一个解析为 [ModelArtifacts](https://github.com/tensorflow/tfjs-core/blob/master/src/io/types.ts#L165) 对象的 promise。这是传递给 `save` 的同一对象。

请参阅 [BrowserHTTPRequest](https://github.com/tensorflow/tfjs-core/blob/master/src/io/browser_http.ts) 获取如何实现 IOHandler 的示例。
