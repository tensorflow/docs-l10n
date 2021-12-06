# tfds および Google Cloud Storage

Google Cloud Storage（GCS）は、次の理由により tfds と併用することができます。

- 事前処理されたデータの格納
- GCS に格納されたデータを持つデータセットへのアクセス

## TFDS GCS バケットを通じてアクセスする

一部のデータセットは、認証を行わずに、直接 GCS バケット [`gs://tfds-data/datasets/`](https://console.cloud.google.com/storage/browser/tfds-data) で利用することができます。

- `tfds.load(..., try_gcs=False)`（デフォルト）である場合、データセットは `download_and_prepare` 中にローカルに `~/tensorflow_datasets` にコピーされます。
- `tfds.load(..., try_gcs=True)` である場合、データセットは GCS から直接ストリーミングされます（`download_and_prepare` はスキップされます）。

データセットがパブリックバケットにホストされているかどうかは、`tfds.is_dataset_on_gcs('mnist')` を使って確認できます。

## 認証

始める前に、認証方法を決定する必要があります。これには、次の 3 つのオプションがあります。

- 認証無し（匿名アクセス）
- Google アカウントの使用
- サービスアカウントの使用（チームの他のメンバーと簡単に共有可能）

詳細は、[Google Cloud ドキュメント](https://cloud.google.com/docs/authentication/getting-started)をご覧ください。

### 簡易手順

Colab から実行する場合は、アカウントを使って認証することができますが、次のコードを実行する必要があります。

```python
from google.colab import auth
auth.authenticate_user()
```

ローカルマシン（または VM）で実行する場合は、次のコードを実行することで、アカウントを使って認証することができます。

```shell
gcloud login application-default
```

サービスアカウントを使ってログインする場合は、JSON ファイル器をダウンロードし、次のように設定する必要があります。

```shell
export GOOGLE_APPLICATION_CREDENTIALS=<JSON_FILE_PATH>
```

## Google Cloud Storage を使って事前処理データを格納する

通常、TensorFlow Dataset を使用する場合、ダウンロードして準備されたデータはローカルディレクトリ（デフォルトは `~/tensorflow_datasets`）にキャッシュされます。

ローカルディスクがエフェメラル（一時クラウドサーバーまたは [Colab ノートブック](https://colab.research.google.com)）であるか、複数のマシンでデータにアクセスする必要がある環境では、`data_dir` を Google Cloud Storage（GCS）バケットなどのクラウドストレージシステムに設定すると役立ちます。

### その方法は？

[GCS バケットを作成](https://cloud.google.com/storage/docs/creating-buckets)し、あなた（またはサービスアカウント）にそのバケットに対する読み取り/書き込み権限があることを確認します（上記の認証手順をご覧ください）。

`tfds` を使用する際に、`data_dir` を `"gs://YOUR_BUCKET_NAME"` に設定することができます。

```python
ds_train, ds_test = tfds.load(name="mnist", split=["train", "test"], data_dir="gs://YOUR_BUCKET_NAME")
```

### 警告:

- このアプローチは、`tf.io.gfile` を使ってのみデータにアクセスするデータセットで機能します。ほとんどのデータセットが該当しますが、すべてではありません。
- GCS にアクセスすると、リモートサーバーにアクセスしてそこからデータをストリーミングすることであるため、ネットワーク接続費がかかる場合があることに注意してください。

## GCS に格納されたデータセットにアクセスする

データセットの所有者が匿名アクセスを許可した場合、tfds.load コードをすぐに実行することができ、通常のインターネットダウンロードと同様に動作します。

データセットで認証が必要な場合は、上記の指示に従って使用するオプション（独自のアカウントかサービスアカウント）を選択し、アカウント名（電子メール）をデータセットの所有者に知らせる必要があります。GCS ディレクトリへのアクセスが有効化されたら、tfds ダウンロードコードを実行できるようになります。
