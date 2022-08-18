<!--* freshness: { owner: 'maringeo' reviewed: '2022-06-13'} *-->

# 模型托管协议

本文档介绍了在 [tfhub.dev](https://tfhub.dev) 上托管所有模型类型（TFJS、TF Lite 和 TensorFlow 模型）时使用的网址惯例。此外，本文档还介绍了由 `tensorflow_hub` 库实现的基于 HTTP(S) 的协议，目的是将 [tfhub.dev](https://tfhub.dev) 中的 TensorFlow 模型和兼容服务加载到 TensorFlow 程序中。

它的关键功能是在代码中使用相同的网址来加载模型，并在浏览器中使用相同的网址来查看模型文档。

## 通用网址惯例

[tfhub.dev](https://tfhub.dev) 支持以下网址格式：

- TF Hub 发布者遵循 `https://tfhub.dev/<publisher>`
- TF Hub 集合遵循 `https://tfhub.dev/<publisher>/collection/<collection_name>`
- TF Hub 模型具有版本化网址 `https://tfhub.dev/<publisher>/<model_name>/<version>` 和可解析为最新版本模型的未版本化网址 `https://tfhub.dev/<publisher>/<model_name>`。

通过将网址参数附加到 [tfhub.dev](https://tfhub.dev) 模型网址，可以将 TF Hub 模型下载为压缩资源。但是，实现该目标所需的网址参数取决于模型类型：

- TensorFlow 模型（SavedModel 和 TF1 Hub 格式）：将 `?tf-hub-format=compressed` 附加到 TensorFlow 模型网址。
- TFJS 模型：将 `?tfjs-format=compressed` 附加到 TFJS 模型网址以下载压缩资源，或者附加 `/model.json?tfjs-format=file` 以便从远程存储空间读取。
- TF Lite 模型：将 `?lite-format=tflite` 附加到 TF Lite 模型网址。

例如：

<table style="width: 100%;">
  <tr style="text-align: center">
    <col style="width: 10%">
    <col style="width: 20%">
    <col style="width: 15%">
    <col style="width: 30%">
    <col style="width: 25%">
    <td style="text-align: center; background-color: #D0D0D0">类型</td>
    <td style="text-align: center; background-color: #D0D0D0">模型网址</td>
    <td style="text-align: center; background-color: #D0D0D0">下载类型</td>
    <td style="text-align: center; background-color: #D0D0D0">网址参数</td>
    <td style="text-align: center; background-color: #D0D0D0">下载网址</td>
  </tr>
  <tr>
    <td>TensorFlow（SavedModel，TF1 Hub 格式）</td>
    <td>https://tfhub.dev/google/spice/2</td>
    <td>.tar.gz</td>
    <td>?tf-hub-format=compressed</td>
    <td>https://tfhub.dev/google/spice/2?tf-hub-format=compressed</td>
  </tr>
  <tr>
    <td>TF Lite</td>
    <td>https://tfhub.dev/google/lite-model/spice/1</td>
    <td>.tflite</td>
    <td>?lite-format=tflite</td>
    <td>https://tfhub.dev/google/lite-model/spice/1?lite-format=tflite</td>
  </tr>
  <tr>
    <td>TF.js</td>
    <td>https://tfhub.dev/google/tfjs-model/spice/2/default/1</td>
    <td>.tar.gz</td>
    <td>?tfjs-format=compressed</td>
    <td>https://tfhub.dev/google/tfjs-model/spice/2/default/1?tfjs-format=compressed</td>
  </tr>
</table>

此外，某些模型还以可直接从远程存储空间读取而无需下载的格式托管。如果没有可用的本地存储空间，例如在浏览器中运行 TF.js 模型或在 [Colab](https://colab.research.google.com/) 上加载 SavedModel，则此功能特别有用。请注意，读取远程托管而不在本地下载的模型可能会增加延迟。

<table style="width: 100%;">
  <tr style="text-align: center">
    <col style="width: 10%">
    <col style="width: 20%">
    <col style="width: 15%">
    <col style="width: 30%">
    <col style="width: 25%">
    <td style="text-align: center; background-color: #D0D0D0">类型</td>
    <td style="text-align: center; background-color: #D0D0D0">模型网址</td>
    <td style="text-align: center; background-color: #D0D0D0">响应类型</td>
    <td style="text-align: center; background-color: #D0D0D0">网址参数</td>
    <td style="text-align: center; background-color: #D0D0D0">请求网址</td>
  </tr>
  <tr>
    <td>TensorFlow（SavedModel，TF1 Hub 格式）</td>
    <td>https://tfhub.dev/google/spice/2</td>
    <td>字符串（存储未压缩模型的 GCS 文件夹的路径）</td>
    <td>?tf-hub-format=uncompressed</td>
    <td>https://tfhub.dev/google/spice/2?tf-hub-format=uncompressed</td>
  </tr>
  <tr>
    <td>TF.js</td>
    <td>https://tfhub.dev/google/tfjs-model/spice/2/default/1</td>
    <td>.json</td>
    <td>?tfjs-format=file</td>
    <td>https://tfhub.dev/google/tfjs-model/spice/2/default/1/model.json?tfjs-format=file</td>
  </tr>
</table>

## tensorflow_hub 库协议

本部分介绍如何在 [tfhub.dev](https://tfhub.dev) 上托管模型以与 tensorflow_hub 库一起使用。如果您想托管自己的模型仓库以使用 tensorflow_hub 库，则您的 HTTP 分发服务应提供此协议的实现。

请注意，本部分不会介绍如何托管 TF Lite 和 TFJS 模型，因为它们不通过 `tensorflow_hub` 库下载。有关托管这些模型类型的详细信息，请参阅[上文](#general-url-conventions)。

### 压缩托管

模型以压缩的 tar.gz 文件形式存储在 [tfhub.dev](https://tfhub.dev) 上。默认情况下，tensorflow_hub 库会自动下载压缩模型。此外，也可以通过将 `?tf-hub-format=compressed` 附加到模型网址来手动下载它们，例如：

```shell
wget https://tfhub.dev/tensorflow/albert_en_xxlarge/1?tf-hub-format=compressed
```

归档的根是模型目录的根，并且应包含 SavedModel，如以下示例所示：

```shell
# Create a compressed model from a SavedModel directory.
$ tar -cz -f model.tar.gz --owner=0 --group=0 -C /tmp/export-model/ .

# Inspect files inside a compressed model
$ tar -tf model.tar.gz
./
./variables/
./variables/variables.data-00000-of-00001
./variables/variables.index
./assets/
./saved_model.pb
```

与旧版 [TF1 Hub 格式](https://www.tensorflow.org/hub/tf1_hub_module)一起使用的 Tarball 还会包含一个 `./tfhub_module.pb` 文件。

当调用 `tensorflow_hub` 库模型加载 API 之一（[hub.KerasLayer](https://www.tensorflow.org/hub/api_docs/python/hub/KerasLayer)、[hub.load](https://www.tensorflow.org/hub/api_docs/python/hub/load) 等）时，库会下载模型，解压缩模型并将其在本地缓存。`tensorflow_hub` 库期望模型网址进行版本化，并且给定版本的模型内容是不可变的，以便可以无限期地对其进行缓存。详细了解[缓存模型](caching.md)。

![](https://raw.githubusercontent.com/tensorflow/hub/master/docs/images/library_download_cache.png)

### 未压缩托管

当环境变量 `TFHUB_MODEL_LOAD_FORMAT` 或命令行标志 `--tfhub_model_load_format` 设置为 `UNCOMPRESSED` 时，会直接从远程存储空间 (GCS) 读取模型，而不是在本地下载和解压缩模型。启用此行为后，库会将 `?tf-hub-format=uncompressed` 附加到模型网址。该请求将返回 GCS 上包含未压缩模型文件的文件夹的路径。举例来说，<br> `https://tfhub.dev/google/spice/2?tf-hub-format=uncompressed` <br>会在 303 响应的正文中返回 <br>`gs://tfhub-modules/google/spice/2/uncompressed`。随后，库从该 GCS 目标读取模型。
