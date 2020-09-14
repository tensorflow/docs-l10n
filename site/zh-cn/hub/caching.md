# 从 TF Hub 缓存下载的模型

## 摘要

从 tfhub.dev（或其他[托管网站](hosting.md)）下载并解压缩模型后，`tensorflow_hub` 库会将这些模型缓存到文件系统上。下载位置默认为本地临时目录，但可以通过设置环境变量 `TFHUB_CACHE_DIR`（推荐）或传递命令行标记 `--tfhub_cache_dir` 进行自定义。使用永久位置时，请注意不会自动清理。

在实际 Python 代码中调用 `tensorflow_hub` 函数可以并且应当继续使用模型的规范 tfhub.dev 网址，这些网址可以跨系统移植且适用于文档导航。

## 特定执行环境

是否需要更改默认的 `TFHUB_CACHE_DIR` 以及如何更改取决于执行环境。

### 在工作站上本地运行

对于在其工作站上运行 TensorFlow 程序的用户，它在大多数情况下都可以正常使用默认位置 `/tmp/tfhub_modules`，或者 Python 为 `os.path.join(tempfile.gettempdir(), "tfhub_modules")` 返回的位置。

希望在系统重新启动后永久缓存的用户可以将 `TFHUB_CACHE_DIR` 设置为其主目录中的某个位置。例如，Linux 系统上的 bash shell 用户可以在 `~/.bashrc` 中添加如下所示的行：

```bash
export TFHUB_CACHE_DIR=$HOME/.cache/tfhub_modules
```

…重新启动 shell，随后将使用此位置。

### 在 Colab 笔记本中的 TPU 上运行

通过 [Colab](https://colab.research.google.com/) 笔记本在 CPU 和 GPU 上运行 TensorFlow 时，使用默认的本地缓存位置即可。

在 TPU 上运行会委托给另一台无法访问默认本地缓存位置的计算机。具有自己的 Google Cloud Storage (GCS) 存储分区的用户可通过以下方式解决此问题：使用如下所示的代码将该存储分区中的目录设置为缓存位置

```python
import os
os.environ["TFHUB_CACHE_DIR"] = "gs://my-bucket/tfhub-modules-cache"
```

…在调用 `tensorflow_hub` 库前完成此操作。
