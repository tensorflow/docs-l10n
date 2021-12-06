# Apache Beam でビッグデータセットを生成する

データセットによっては、1 台のマシンで処理するには大きすぎるものがあります。`tfds`は、[Apache Beam](https://beam.apache.org/) を使用することによって、多くのマシンにまたがったデータ生成のサポートをします。

このドキュメントには、2 つのセクションがあります。

- 既存の Beam のデータセットを生成するユーザー向け
- 新規の Beam のデータセットを作成する開発者向け

## Beam のデータセットを生成する

クラウドまたはローカルで Beam のデータセットを生成するさまざまな例を以下に紹介します。

**警告**: [`tfds build` CLI](https://www.tensorflow.org/datasets/cli#tfds_build_download_and_prepare_a_dataset) でデータセットを生成する際には、生成するデータセットの構成を必ず指定してください。指定しない場合、すべての既存の構成が生成されてしまいます。たとえば[ウィキペディア](https://www.tensorflow.org/datasets/catalog/wikipedia)の場合は、`tfds build wikipedia` の代わりに `tfds build wikipedia/20200301.en` を使用します。

### Google Cloud Dataflow で生成する

[Google Cloud Dataflow](https://cloud.google.com/dataflow/) を使用してパイプラインを実行し、分散計算を活用するには、まず[クイックスタートの手順](https://cloud.google.com/dataflow/docs/quickstarts/quickstart-python)に従います。

環境をセットアップしたら、[`tfds build` CLI](https://www.tensorflow.org/datasets/cli#tfds_build_download_and_prepare_a_dataset) を実行できます。これには、[GCS](https://cloud.google.com/storage/) のデータディレクトリを使用し、[必要なオプション](https://cloud.google.com/dataflow/docs/guides/specifying-exec-params#configuring-pipelineoptions-for-execution-on-the-cloud-dataflow-service)を `--beam_pipeline_options` フラグに指定します。

スクリプトの起動を容易にするためには、GCP/GCS セットアップの実際の値と生成するデータセットを使用して、以下の変数を定義すると便利です。

```sh
DATASET_NAME=<dataset-name>
DATASET_CONFIG=<dataset-config>
GCP_PROJECT=my-project-id
GCS_BUCKET=gs://my-gcs-bucket
```

次に、ワーカーに `tfds` をインストールするよう Dataflow に指示をするファイルを作成する必要があります。

```sh
echo "tensorflow_datasets[$DATASET_NAME]" > /tmp/beam_requirements.txt
```

`tfds-nightly` を使用している場合には、データセットが前回のリリースから更新されている場合に備え、`tfds-nightly` からエコーするようにします。

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
tfds build my_dataset
```

**警告**: Beam のデータセットは**巨大な**（テラバイト以上）場合があり、生成には相当量のリソースを必要とします（ローカルコンピュータでは数週間かかることもあります）。データセットの生成には分散環境の使用を推奨しています。サポートされているランタイムのリストについては [Apache Beam ドキュメント](https://beam.apache.org/)を参照してください。

### カスタムスクリプト内で生成する

Beam でデータセットを生成する場合、API は他のデータセットの場合と同じですが、[`beam.Pipeline`](https://beam.apache.org/documentation/programming-guide/#creating-a-pipeline) を、`DownloadConfig` の `beam_options`（および `beam_runner`）引数を使ってカスタマイズできます。

```python
# If you are running on Dataflow, Spark,..., you may have to set-up runtime
# flags. Otherwise, you can leave flags empty [].
flags = ['--runner=DataflowRunner', '--project=<project-name>', ...]

# `beam_options` (and `beam_runner`) will be forwarded to `beam.Pipeline`
dl_config = tfds.download.DownloadConfig(
    beam_options=beam.options.pipeline_options.PipelineOptions(flags=flags)
)
data_dir = 'gs://my-gcs-bucket/tensorflow_datasets'
builder = tfds.builder('wikipedia/20190301.en', data_dir=data_dir)
builder.download_and_prepare(download_config=dl_config)
```

## Beam のデータセットを実装する

### 前提条件

Apache Beam のデータセットを書き込むにあたり、以下の概念を理解しておく必要があります。

- ほとんどの内容が Beam のデータセットにも適用されるため、[`tfds` データセット作成ガイド](https://github.com/tensorflow/datasets/tree/master/docs/add_dataset.md)に精通しましょう。
- [Beam プログラミングガイド](https://beam.apache.org/documentation/programming-guide/)で Apache Beam の概要を把握しましょう。
- Cloud Dataflow を使用してデータセットを生成する場合は、[Google Cloud ドキュメント](https://cloud.google.com/dataflow/docs/quickstarts/quickstart-python) と [Apache Beam 依存性ガイド](https://beam.apache.org/documentation/sdks/python-pipeline-dependencies/)をお読みください。

### 手順

[データセット作成ガイド](https://github.com/tensorflow/datasets/tree/master/docs/add_dataset.md)を理解しているのであれば、Beam データセットの追加には、`_generate_examples` 関数のみを変更する必要があることはお分かりでしょう。この関数はジェネレータではなく Beam 関数を返します。

Beam 以外のデータセット:

```python
def _generate_examples(self, path):
  for f in path.iterdir():
    yield _process_example(f)
```

Beam データセット:

```python
def _generate_examples(self, path):
  return (
      beam.Create(path.iterdir())
      | beam.Map(_process_example)
  )
```

その他すべては、テストも含め、まったく同じになります。

その他の考慮事項 :

- Apache Beam のインポートには、`tfds.core.lazy_imports`を使用します。遅延依存関係を使用すると、ユーザーは Beam をインストールしなくても、生成された後のデータセットを読むことができます。
- Python のクロージャには注意してください。パイプラインを実行する際、`beam.Map` と `beam.DoFn` 関数は、`pickle` を使ってシリアル化され、すべてのワーカーに送信されます。ワーカー間で状態を共有する必要がある場合は、`beam.PTransform` 内でオブジェクトをミュータブルにしないでください。
- `tfds.core.DatasetBuilder` が pickle でシリアル化される方法により、データ作成中、ワーカーでの `tfds.core.DatasetBuilder` のミュート化は無視されます（`_split_generators` で `self.info.metadata['offset'] = 123` を設定し、`beam.Map(lambda x: x + self.info.metadata['offset'])` のようにしてワーカーからそれにアクセスすることはできません）。
- Split 間で一部のパイプラインステップを共有する櫃夜ぐあある場合は、追加の `pipeline: beam.Pipeline` kwarg を `_split_generator` に追加して、生成パイプライン全体を制御することができます。`tfds.core.GeneratorBasedBuilder` の `_generate_examples` ドキュメントをご覧ください。

### 例

Beam データセットの例を以下に示します。

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

**注意**: Beam 以外のデータセットと同様に、`--register_checksums` でダウンロードチェックサムを必ず登録してください（ダウンロードを初めて登録する場合のみ）。

```sh
tfds build my_dataset --register_checksums
```

## TFDS を入力として使用するパイプライン

TFDS データセットをソースとして取る Beam パイプラインを作成する場合は、`tfds.beam.ReadFromTFDS` を使用できます。

```python
builder = tfds.builder('my_dataset')

_ = (
    pipeline
    | tfds.beam.ReadFromTFDS(builder, split='train')
    | beam.Map(tfds.as_numpy)
    | ...
)
```

データセットの各シャードを並行して処理します。

注意: これには、データベースがすでに生成されていることが必要です。Beam を使ってデータセットを生成するには、ほかのセクションをご覧ください。
