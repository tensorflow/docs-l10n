<!--* freshness: { owner: 'wgierke' reviewed: '2021-07-28' } *-->

# 从 TF Hub 缓存下载的模型

## 概述

`tensorflow_hub` 库目前支持两种模式来下载模型。默认情况下，模型以压缩归档的形式下载并缓存到磁盘上。其次，可以将模型直接从远程存储空间读取到 TensorFlow 中。无论采用哪种方式，实际 Python 代码中对 `tensorflow_hub` 函数的调用都可以并且应当继续使用模型的规范 tfhub.dev 网址，这些网址可在系统之间移植且可用于在文档中导航。在极少数情况下，用户代码需要实际的文件系统位置（在下载和解压缩之后，或者将模型句柄解析到文件系统路径中之后），可以通过 `hub.resolve(handle)` 函数获得该位置。

### 缓存压缩的下载

从 tfhub.dev（或其他[托管站点](hosting.md)）下载模型并解压缩后，默认情况下，`tensorflow_hub` 库会在文件系统上缓存模型。推荐将这种模式用于大多数环境，除非磁盘空间很少，但网络带宽和延迟非常出色。

下载位置默认为本地临时目录，但可以通过设置环境变量 `TFHUB_CACHE_DIR`（推荐）或通过传递命令行标志 `--tfhub_cache_dir` 进行自定义。在大多数情况下，默认缓存位置 `/tmp/tfhub_modules`（或诸如 `os.path.join(tempfile.gettempdir(), "tfhub_modules")` 等经过评估的位置）应当有效。

希望在系统重新启动后永久缓存的用户可以将 `TFHUB_CACHE_DIR` 设置为其主目录中的某个位置。例如，Linux 系统上的 bash shell 用户可以在 `~/.bashrc` 中添加如下所示的行：

```bash
export TFHUB_CACHE_DIR=$HOME/.cache/tfhub_modules
```

…重启 shell，随后将使用此位置。使用永久位置时，请注意不会自动清理。

### 从远程存储空间读取

用户可以指示 `tensorflow_hub` 库直接从远程存储空间 (GCS) 读取模型，而不是通过以下代码在本地下载模型：

```shell
os.environ["TFHUB_MODEL_LOAD_FORMAT"] = "UNCOMPRESSED"
```

或者通过将命令行标志 `--tfhub_model_load_format` 设置为 `UNCOMPRESSED`。这样一来，将不需要缓存目录，这在磁盘空间很少但互联网连接很快的环境中特别有用。

### 在 Colab 笔记本中的 TPU 上运行

在 [colab.research.google.com](https://colab.research.google.com) 上，下载压缩模型将与 TPU 运行时发生冲突，因为计算工作负载会委托给另一台默认无法访问缓存位置的计算机。可通过两种方法解决此问题：

#### 1) 使用 TPU 工作进程可以访问的 GCS 存储分区

最简单的解决方案是，按照上面的说明指示 `tensorflow_hub` 库从 TF Hub 的 GCS 存储分区中读取模型。拥有自己的 GCS 存储分区的用户可以使用以下代码将存储分区中的目录指定为缓存位置：

```python
import os
os.environ["TFHUB_CACHE_DIR"] = "gs://my-bucket/tfhub-modules-cache"
```

…在调用 `tensorflow_hub` 库前完成此操作。

#### 2) 通过 Colab 主机重定向所有读取

另一种解决方法是通过 Colab 主机重定向所有读取（甚至大型变量的读取）：

```python
load_options =
tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')
reloaded_model = hub.load("https://tfhub.dev/...", options=load_options)
```
