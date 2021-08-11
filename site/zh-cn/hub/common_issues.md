<!--* freshness: { owner: 'maringeo' reviewed: '2021-04-12' review_interval: '6 months' } *-->

# 常见问题

如果您的问题未在此处列出，请先搜索 [GitHub 议题](https://github.com/tensorflow/hub/issues)，然后再提交新问题。

## TypeError: 'AutoTrackable' object is not callable

```python
# BAD: Raises error
embed = hub.load('https://tfhub.dev/google/nnlm-en-dim128/1')
embed(['my text', 'batch'])
```

使用 TF2 中的 `hub.load()` API 加载 TF1 Hub 格式的模型时，经常会出现此错误。添加正确的签名应当可以解决此问题。有关迁移到 TF2 以及在 TF2 中使用 TF1 Hub 格式的模型的更多详细信息，请参阅 [TF2 的 TF-Hub 迁移指南](migration_tf2.md)。

```python

embed = hub.load('https://tfhub.dev/google/nnlm-en-dim128/1')
embed.signatures['default'](['my text', 'batch'])
```

## 无法下载模块

在从网址使用模块的过程中，由于网络堆栈，可能会出现许多错误。通常，这是运行代码的机器特有的问题，而不是库的问题。下面是一些常见问题的列表：

- **EOF occurred in violation of protocol** - 如果安装的 Python 版本不支持托管该模块的服务器的 TLS 要求，则可能会产生此问题。值得注意的是，已知 Python 2.7.5 无法解析来自 tfhub.dev 域的模块。**解决方法**：请更新到较新的 Python 版本。

- **cannot verify tfhub.dev's certificate** - 如果网络上的某些内容试图充当 dev gTLD，则很可能会产生此问题。在将 .dev 用作 gTLD 之前，开发者和框架有时会使用 .dev 名称来帮助测试代码。**解决方法**：确定并重新配置拦截“.dev”域中名称解析的软件。

- 未能写入缓存目录 `/tmp/tfhub_modules`（或类似目录）：要了解缓存以及如何更改其位置，请参阅[缓存](caching.md)。

如果上述错误的解决方法不起作用，用户可以尝试通过如下操作来手动下载模块：模拟将 `?tf-hub-format=compressed` 附加到网址的协议，以下载必须手动解压缩到本地文件的 tar 压缩文件。随后可以使用本地文件的路径替代网址。下面是一个简单的示例：

```bash
# Create a folder for the TF hub module.
$ mkdir /tmp/moduleA
# Download the module, and uncompress it to the destination folder. You might want to do this manually.
$ curl -L "https://tfhub.dev/google/universal-sentence-encoder/2?tf-hub-format=compressed" | tar -zxvC /tmp/moduleA
# Test to make sure it works.
$ python
> import tensorflow_hub as hub
> hub.Module("/tmp/moduleA")
```

## 在预初始化的模块上运行推理

如果您正在编写一个将模块多次应用于输入数据的 Python 程序，则可以应用以下配方。（注：对于生产服务中的服务请求，请考虑使用 [TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving) 或其他可扩展的无 Python 解决方案。）

假设您的用例模型为**初始化**和后续**请求**（例如 Django、Flask、自定义 HTTP 服务器等），则可以按以下方式设置服务：

### TF2 SavedModel

- 在初始化部分中：
    - 加载 TF2.0 模型。

```python
import tensorflow_hub as hub

embedding_fn = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
```

- 在请求部分中：
    - 使用嵌入函数运行推理。

```python
embedding_fn(["Hello world"])
```

对 tf.function 的这种调用已针对性能进行了优化，请参阅 [tf.function 指南](https://www.tensorflow.org/guide/function)。

### TF1 Hub 模块

- 在初始化部分中：
    - 使用**占位符**构建计算图 - 占位符是计算图的入口点。
    - 初始化会话。

```python
import tensorflow as tf
import tensorflow_hub as hub

# Create graph and finalize (finalizing optional but recommended).
g = tf.Graph()
with g.as_default():
  # We will be feeding 1D tensors of text into the graph.
  text_input = tf.placeholder(dtype=tf.string, shape=[None])
  embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")
  embedded_text = embed(text_input)
  init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
g.finalize()

# Create session and initialize.
session = tf.Session(graph=g)
session.run(init_op)
```

- 在请求部分中：
    - 使用会话通过占位符将数据馈入计算图。

```python
result = session.run(embedded_text, feed_dict={text_input: ["Hello world"]})
```

## 无法更改模型的数据类型（例如，将 float32 更改为 bfloat16）

TensorFlow 的 SavedModel（在 TF Hub 上或其他位置共享）包含适用于固定数据类型的运算（对于神经网络的权重和中间激活，通常使用 float32）。在加载 SavedModel 之后，将无法更改这些数据类型（但模型发布者可以选择发布具有不同数据类型的不同模型）。

## 更新模型版本

可以更新模型版本的文档元数据。但是，版本的资产（模型文件）不可变。如果要更改模型资产，可以发布较新版本的模型。最好使用描述版本之间变化的变更日志来扩展文档。
