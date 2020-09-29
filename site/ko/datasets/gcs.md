# tfds 및 Google 클라우드 스토리지

Google Cloud Storage (GCS) can be used with tfds for multiple reasons:

- Storing preprocessed data
- Accessing datasets that have data stored on GCS

## Authentication

Before starting, you should decide on how you want to authenticate. There are three options:

- no authentication (a.k.a anonymous access)
- using your Google account
- using a service account (can be easily shared with others in your team)

You can find detailed informations in [Google Cloud documentation](https://cloud.google.com/docs/authentication/getting-started)

### Simplified instructions

If you run from colab, you can authenticate with your account, but running:

```python
from google.colab import auth
auth.authenticate_user()
```

If you run on your local machine (or in VM), you can authenticate with your account by running:

```shell
gcloud login application-default
```

If you want to login with service account, download the JSON file key and set

```shell
export GOOGLE_APPLICATION_CREDENTIALS=<JSON_FILE_PATH>
```

## Using Google Cloud Storage to store preprocessed data

Normally when you use TensorFlow Datasets, the downloaded and prepared data will be cached in a local directory (by default `~/tensorflow_datasets`).

In some environments where local disk may be ephemeral (a temporary cloud server or a [Colab notebook](https://colab.research.google.com)) or you need the data to be accessible by multiple machines, it's useful to set `data_dir` to a cloud storage system, like a Google Cloud Storage (GCS) bucket.

### 어떻게?

[Create a GCS bucket](https://cloud.google.com/storage/docs/creating-buckets) and ensure you (or your service account) have read/write permissions on it (see authorization instructions above)

When you use `tfds`, you can set `data_dir` to `"gs://YOUR_BUCKET_NAME"`

```python
ds_train, ds_test = tfds.load(name="mnist", split=["train", "test"], data_dir="gs://YOUR_BUCKET_NAME")
```

### Caveats:

- This approach works for datasets that only use `tf.io.gfile` for data access. This is true for most datasets, but not all.
- GCS에 액세스하면 원격 서버에 액세스하고 데이터를 스트리밍하므로 네트워크 비용이 발생할 수 있습니다.

## Accessing datasets stored on GCS

If dataset owners allowed anonymous access, you can just go ahead and run the tfds.load code - and it would work like a normal internet download.

If dataset requires authentication, please use the instructions above to decide on which option you want (own account vs service account) and communicate the account name (a.k.a email) to the dataset owner. After they enable you access to the GCS directory, you should be able to run the tfds download code.
