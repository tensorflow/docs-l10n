# tfds 和 Google Cloud Storage

可以将 tfds 与 Google Cloud Storage (GCS) 结合使用来实现以下目标：

- 存储预处理数据
- 访问在 GCS 上存储数据的数据集

## 通过 TFDS GCS 桶访问

一些数据集可直接在我们的 GCS 存储分区 [`gs://tfds-data/datasets/`](https://console.cloud.google.com/storage/browser/tfds-data) 中获得，无需进行任何身份验证：

- 如果 `tfds.load(..., try_gcs=False)`（默认），则将在 `download_and_prepare` 期间在 `~/tensorflow_datasets` 中本地复制数据集。
- 如果 `tfds.load(..., try_gcs=True)`，则将直接从 GCS 流式传输数据集（将跳过 `download_and_prepare`）。

您可以使用 `tfds.is_dataset_on_gcs('mnist')` 检查数据集是否托管在公共存储分区上。

## 身份验证

首先，您应该决定如何进行身份验证。共有三个选项：

- 无身份验证（又名匿名访问）
- 使用您的 Google 帐号
- 使用服务帐号（可以轻松地与团队中的其他人共享）

您可以在 [Google Cloud 文档](https://cloud.google.com/docs/authentication/getting-started)中找到详细信息

### 简化说明

如果从 Colab 运行，您可以使用您的帐号进行身份验证，但需要运行以下代码：

```python
from google.colab import auth
auth.authenticate_user()
```

如果您在本地计算机上（或 VM 中）运行，则可以通过运行以下代码来使用您的帐号进行身份验证：

```shell
gcloud login application-default
```

如果要使用服务帐号登录，请下载 JSON 文件密钥并设置

```shell
export GOOGLE_APPLICATION_CREDENTIALS=<JSON_FILE_PATH>
```

## 使用 Google Cloud Storage 存储预处理数据

通常，当您使用 TensorFlow Datasets 时，下载并准备好的数据将被缓存在本地目录中（默认路径为 `~/tensorflow_datasets`）。

在本地磁盘可能是临时磁盘（临时云服务器或 [Colab 笔记本](https://colab.research.google.com)）或数据需要被多台计算机访问的某些环境中，将 `data_dir` 设置到云存储系统（例如 Google Cloud Storage (GCS) 存储分区）非常实用。

### 如何设置？

[创建 GCS 存储分区](https://cloud.google.com/storage/docs/creating-buckets)并确保您（或您的服务帐号）拥有它的读写权限（请参阅上方身份验证说明）

使用 `tfds` 时，可以将 `data_dir` 设置为 `"gs://YOUR_BUCKET_NAME"`

```python
ds_train, ds_test = tfds.load(name="mnist", split=["train", "test"], data_dir="gs://YOUR_BUCKET_NAME")
```

### 注意事项：

- 此方法适用于仅使用 `tf.io.gfile` 进行数据访问的数据集。对于大多数数据集都是如此，但并非全部。
- 请记住，访问 GCS 是在访问远程服务器并从中流式传输数据，因此可能会产生网络费用。

## 访问存储在 GCS 上的数据集

如果数据集所有者允许匿名访问，则直接运行 tfds.load 代码即可——与常规 Internet 下载方式相同。

如果数据集需要身份验证，请使用上面的说明来确定所需的选项（使用自己的帐号还是服务帐号），并将帐号名称（又称电子邮件）传达给数据集所有者。在他们为您分配 GCS 目录访问权限之后，您即可运行 tfds 下载代码。
