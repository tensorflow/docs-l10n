# 托管自己的模型

TensorFlow Hub 在 [thub.dev](https://tfhub.dev) 中提供了一个开放的训练模型存储库。`tensorflow_hub` 库可以从这个存储库和其他基于 HTTP 的机器学习模型存储库中加载模型。特别是，该协议允许将标识模型的网址用于模型的文档以及获取模型的端点。

如果您有意托管自己的模型存储库，而这些模型可使用 `tensorflow_hub` 库来加载，那么您的 HTTP 分发服务应遵循以下协议。

## 协议

使用诸如 `https://example.com/model` 的网址标识要加载或实例化的模型时，在追加查询参数 `?tf-hub-format=compressed` 后，模型解析器将尝试从网址下载压缩的 tarball。

查询参数将解释为客户端感兴趣的模型格式的逗号分隔列表。目前仅定义了“压缩”格式。

**压缩**格式表示客户端期望一个包含模型内容的 `tar.gz` 归档。归档的根是模型目录的根，并且应包含 SavedModel，如下面的示例中所示：

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

与 TF1 中已弃用的 `hub.Module()` API 一起使用的 tarball 还包含 `./tfhub_module.pb` 文件。TF2 SavedModel 的 `hub.load()` API 会忽略此类文件。

`tensorflow_hub` 库期望模型网址进行版本控制，并且给定版本的模型内容是不可变的，以便可以无限期地对其进行缓存。
