# Apache Beam でビッグデータセットを生成する

データセットによっては、1 台のマシンで処理するには大きすぎるものがあります。`tfds`は、[Apache Beam](https://beam.apache.org/) を使用することによって、多くのマシンにまたがったデータ生成のサポートをします。

このドキュメントには、2 つのセクションがあります。

- 既存の Beam のデータセットを生成するユーザー向け
- 新規の Beam のデータセットを作成する開発者向け

目次 :

- [Beam のデータセットを生成する](#generating-a-beam-dataset)
    - [Google Cloud Dataflow で生成する](#on-google-cloud-dataflow)
    - [ローカルで生成する](#locally)
    - [カスタムスクリプト内で生成する](#with-a-custom-script)
- [Beam のデータセットを実装する](#implementing-a-beam-dataset)
    - [必要な準備](#prerequisites)
    - [手順](#instructions)
    - [例](#example)
    - [パイプラインの実行](#run-your-pipeline)

## Beam のデータセットを生成する

クラウドとローカルで Beam のデータセットを生成する例を以下に紹介します。

**警告**: `tensorflow_datasets.scripts.download_and_prepare`スクリプトでデータセットを生成する際には、生成するデータセットの設定を必ず指定しなければなりません。例えば、[wikipedia](https://www.tensorflow.org/datasets/catalog/wikipedia) の場合は、`--dataset=wikipedia`の代わりに`--dataset=wikipedia/20200301.en`を使用します。

### Google Cloud Dataflow で生成する

[Google Cloud Dataflow](https://cloud.google.com/dataflow/) を使用してパイプラインを実行し、分散計算を活用できるようにするには、まず[クイックスタートの手順](https://cloud.google.com/dataflow/docs/quickstarts/quickstart-python)に従います。

環境の設定ができたら、<a class="" href="https://cloud.google.com/storage/">GCS</a> のデータディレクトリを使用して<code>download_and_prepare</code>スクリプトを実行し、<code>--beam_pipeline_options</code>フラグに<a class="" href="https://cloud.google.com/dataflow/docs/guides/specifying-exec-params#configuring-pipelineoptions-for-execution-on-the-cloud-dataflow-service">必要なオプション</a>を指定します。

スクリプトの起動を容易にするためには、GCP/GCS のセットアップと生成するデータセットの実際の値を使用して、以下の変数を定義すると有用です。

```sh
DATASET_NAME=<dataset-name>
DATASET_CONFIG=<dataset-config>
GCP_PROJECT=my-project-id
GCS_BUCKET=gs://my-gcs-bucket
```

次に、ワーカーに`tfds`をインストールするよう Dataflow に指示をするファイルを作成する必要があります。

```sh
echo "tensorflow_datasets[$DATASET_NAME]" > /tmp/beam_requirements.txt
```

`tfds-nightly`を使用している場合には、データセットが前回のリリースから更新されている場合に備え、`tfds-nightly`からエコーするようにします。

```sh
echo "tfds-nightly[$DATASET_NAME]" > /tmp/beam_requirements.txt
```

最後に、以下のコマンドを使用してジョブを起動します。

```sh
python -m tensorflow_datasets.scripts.download_and_prepare \
  --datasets=$DATASET_NAME/$DATASET_CONFIG \
  --data_dir=$GCS_BUCKET/tensorflow_datasets \
  --beam_pipeline_options=\
"runner=DataflowRunner,project=$GCP_PROJECT,job_name=$DATASET_NAME-gen,"\
"staging_location=$GCS_BUCKET/binaries,temp_location=$GCS_BUCKET/temp,"\
"requirements_file=/tmp/beam_requirements.txt"
```

### ローカルで生成する

デフォルトの Apache Beam ランナーを使用してローカルでスクリプトを実行する場合、コマンドは他のデータセットの場合と同じです。

```sh
python -m tensorflow_datasets.scripts.download_and_prepare \
  --datasets=my_new_dataset
```

**警告**: Beam のデータセットは**巨大**（テラバイト級）な場合があり、生成にはかなりの量のリソースを必要とします（ローカルコンピュータでは数週間かかることもあります）。データセットの生成には分散環境の使用を推奨しています。サポートされているランタイムのリストについては [Apache Beam ドキュメント](https://beam.apache.org/)を参照してください。

### カスタムスクリプト内で生成する

Beam でデータセットを生成する場合、API は他のデータセットの場合と同じですが、Beam のオプションかランナーを`DownloadConfig`に渡す必要があります。

```py
# If you are running on Dataflow, Spark,..., you may have to set-up runtime
# flags. Otherwise, you can leave flags empty [].
flags = ['--runner=DataflowRunner', '--project=<project-name>', ...]

# To use Beam, you have to set at least one of `beam_options` or `beam_runner`
dl_config = tfds.download.DownloadConfig(
    beam_options=beam.options.pipeline_options.PipelineOptions(flags=flags)
)

data_dir = 'gs://my-gcs-bucket/tensorflow_datasets'
builder = tfds.builder('wikipedia/20190301.en', data_dir=data_dir)
builder.download_and_prepare(
    download_dir=FLAGS.download_dir,
    download_config=dl_config,
)
```

## Beam のデータセットを実装する

### 必要な準備

Apache Beam のデータセットを書き込むにあたり、以下の概念を理解しておく必要があります。

- ほとんどの内容が Beam のデータセットにも適用されるため、[`tfds`データセット作成ガイド](https://github.com/tensorflow/datasets/tree/master/docs/add_dataset.md)に精通しましょう。
- [Beam プログラミングガイド](https://beam.apache.org/documentation/programming-guide/)で Apache Beam の概要を把握しましょう。
- Cloud Dataflow を使用してデータセットを生成する場合は、[Google Cloud ドキュメント](https://cloud.google.com/dataflow/docs/quickstarts/quickstart-python) と [Apache Beam 依存性ガイド](https://beam.apache.org/documentation/sdks/python-pipeline-dependencies/)をお読みください。

### 手順

[データセット作成ガイド](https://github.com/tensorflow/datasets/tree/master/docs/add_dataset.md)に精通している場合は、少し変更を加えるだけで Beam のデータセットを追加することができます。

- `DatasetBuilder`は、`tfds.core.GeneratorBasedBuilder`の代わりに`tfds.core.BeamBasedBuilder`を継承します。
- Beam のデータセットは、メソッド`_generate_examples(self, **kwargs)`の代わりに抽象メソッド`_build_pcollection(self, **kwargs)`を実装します。`_build_pcollection`は、スプリットに関連付けられた例を含む`beam.PCollection`を返します。
- Beam のデータセットの単体テストの記述は、他のデータセットの場合と同じです。

その他の考慮事項 :

- Apache Beam のインポートには、`tfds.core.lazy_imports`を使用します。遅延依存関係を使用すると、ユーザーは Beam をインストールしなくても、生成された後のデータセットを読むことができます。
- Python のクロージャには注意が必要です。パイプラインを実行すると、`beam.Map`関数と`beam.DoFn`関数は`pickle`を使用してシリアライズされ、すべてのワーカーに送信されます。これはバグを発生させる可能性があります。例えば、関数の外部で宣言された可変オブジェクトを関数内で使用している場合、`pickle`エラーや予期せぬ動作が発生する場合があります。一般的な解決策は、関数閉包を変更しないようにすることです。
- Beam のパイプラインに`DatasetBuilder`のメソッドを使用することは問題ありません。しかし、pickle 中にクラスをシリアライズする方法では、作成中に加えられた特徴の変更は無視されます。

### 例

以下は Beam のデータセットの例です。より複雑な実例については、[`Wikipedia`データセット](https://github.com/tensorflow/datasets/tree/master/tensorflow_datasets/text/wikipedia.py)をご覧ください。

```python
class DummyBeamDataset(tfds.core.BeamBasedBuilder):

  VERSION = tfds.core.Version('1.0.0')

  def _info(self):
    return tfds.core.DatasetInfo(
        builder=self,
        features=tfds.features.FeaturesDict({
            'image': tfds.features.Image(shape=(16, 16, 1)),
            'label': tfds.features.ClassLabel(names=['dog', 'cat']),
        }),
    )

  def _split_generators(self, dl_manager):
    ...
    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            gen_kwargs=dict(file_dir='path/to/train_data/'),
        ),
        splits_lib.SplitGenerator(
            name=tfds.Split.TEST,
            gen_kwargs=dict(file_dir='path/to/test_data/'),
        ),
    ]

  def _build_pcollection(self, pipeline, file_dir):
    """Generate examples as dicts."""
    beam = tfds.core.lazy_imports.apache_beam

    def _process_example(filename):
      # Use filename as key
      return filename, {
          'image': os.path.join(file_dir, filename),
          'label': filename.split('.')[1],  # Extract label: "0010102.dog.jpeg"
      }

    return (
        pipeline
        | beam.Create(tf.io.gfile.listdir(file_dir))
        | beam.Map(_process_example)
    )

```

### パイプラインの実行

パイプラインの実行には、上記のセクションをご覧ください。

**警告**: 初めてデータセットを実行してダウンロードを登録する際には、`download_and_prepare`スクリプトにレジスタチェックサム`--register_checksums`フラグの追加を忘れないようにしてください。

```sh
python -m tensorflow_datasets.scripts.download_and_prepare \
  --register_checksums \
  --datasets=my_new_dataset
```
