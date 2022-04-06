# Tuner TFX パイプラインコンポーネント

Tuner コンポーネントは、モデルのハイパーパラメータをチューニングします。

## Tuner コンポーネントと KerasTuner ライブラリ

Tuner コンポーネントは、ハイパーパラメータのチューニングに Python [KerasTuner](https://www.tensorflow.org/tutorials/keras/keras_tuner) API を多大に使用しています。

注意: KerasTuner ライブラリは Keras モデルだけではなく、モデリング API に関係なくハイパーパラメータのチューニングに使用できます。

## コンポーネント

Tuner は次を取り込みます。

- training と eval に使用される tf.Examples
- モデルの定義、ハイパーパラメータの検索スペース、目的など、チューニングロジックを定義するユーザー指定モジュールファイル（または module_fn）
- train args と eval args の [Protobuf](https://developers.google.com/protocol-buffers) 定義
- （オプション）tuning args の [Protobuf](https://developers.google.com/protocol-buffers) 定義
- （オプション）上流の Transform コンポーネントが生成する transform グラフ
- （オプション）SchemaGen パイプラインコンポーネントが作成し、開発者がオプションとして変更できるデータスキーマ

特定のデータ、モデル、および目的を使用して、Tuner はハイパーパラメータをチューニング氏、最善の結果を出力します。

## 手順

Tuner には次のシグネチャーによるユーザーモジュール関数 `tuner_fn` が必要です。

```python
...
from keras_tuner.engine import base_tuner

TunerFnResult = NamedTuple('TunerFnResult', [('tuner', base_tuner.BaseTuner),
                                             ('fit_kwargs', Dict[Text, Any])])

def tuner_fn(fn_args: FnArgs) -> TunerFnResult:
  """Build the tuner using the KerasTuner API.
  Args:
    fn_args: Holds args as name/value pairs.
      - working_dir: working dir for tuning.
      - train_files: List of file paths containing training tf.Example data.
      - eval_files: List of file paths containing eval tf.Example data.
      - train_steps: number of train steps.
      - eval_steps: number of eval steps.
      - schema_path: optional schema of the input data.
      - transform_graph_path: optional transform graph produced by TFT.
  Returns:
    A namedtuple contains the following:
      - tuner: A BaseTuner that will be used for tuning.
      - fit_kwargs: Args to pass to tuner's run_trial function for fitting the
                    model , e.g., the training and validation dataset. Required
                    args depend on the above tuner's implementation.
  """
  ...
```

この関数では、モデルとハイパーパラメータの両方の検索スペースを定義し、チューニングの目的とアルゴリズムを選択します。Tuner コンポーネントをこのモジュールコードを入力として取り、ハイパーパラメータをチューニングして最善の結果を出力します。

Trainer は Tuner の出力ハイパーパラメータを入力として取り、ユーザーモジュールコードに使用します。パイプライン定義は次のようになります。

```python
...
tuner = Tuner(
    module_file=module_file,  # Contains `tuner_fn`.
    examples=transform.outputs['transformed_examples'],
    transform_graph=transform.outputs['transform_graph'],
    train_args=trainer_pb2.TrainArgs(num_steps=20),
    eval_args=trainer_pb2.EvalArgs(num_steps=5))

trainer = Trainer(
    module_file=module_file,  # Contains `run_fn`.
    examples=transform.outputs['transformed_examples'],
    transform_graph=transform.outputs['transform_graph'],
    schema=schema_gen.outputs['schema'],
    # This will be passed to `run_fn`.
    hyperparameters=tuner.outputs['best_hyperparameters'],
    train_args=trainer_pb2.TrainArgs(num_steps=100),
    eval_args=trainer_pb2.EvalArgs(num_steps=5))
...
```

モデルを取得するたびにハイパーパラメータをチューニングしないようにする場合は、Tuner を使用して、良い結果を出すハイパーパラメータを特定したら、パイプラインから Tuner を削除して、`ImporterNode` を使って前回のトレーニングランの Tuner アーティファクトをインポートし、Trainer にフィードすることができます。

```python
hparams_importer = Importer(
    # This can be Tuner's output file or manually edited file. The file contains
    # text format of hyperparameters (keras_tuner.HyperParameters.get_config())
    source_uri='path/to/best_hyperparameters.txt',
    artifact_type=HyperParameters,
).with_id('import_hparams')

trainer = Trainer(
    ...
    # An alternative is directly use the tuned hyperparameters in Trainer's user
    # module code and set hyperparameters to None here.
    hyperparameters = hparams_importer.outputs['result'])
```

## Google Cloud プラットフォーム（GCP）でのチューニング

Google Cloud プラットフォーム（GCP）で実行している場合、Tuner コンポーネントは以下の 2 つのサービスを利用できます。

- [AI Platform Vizier](https://cloud.google.com/ai-platform/optimizer/docs/overview)（CloudTuner 実装を介して）
- [AI Platform Training](https://cloud.google.com/ai-platform/training/docs)（分散型チューニングのフロックマネージャーとして）

### ハイパーパラメータチューニングのバックエンドとしての AI Platform Vizier

[AI Platform Vizier](https://cloud.google.com/ai-platform/optimizer/docs/overview) は [Google Vizier](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/bcb15507f4b52991a0783013df4222240e942381.pdf) テクノロジーに基づいてブラックボックス最適化を実施するマネージドサービスです。

[CloudTuner](https://github.com/tensorflow/cloud/blob/master/src/python/tensorflow_cloud/tuner/tuner.py) は AI Platform Vizer サービスに調査バックエンドとして話しかける [KerasTuner](https://www.tensorflow.org/tutorials/keras/keras_tuner) の実装です。CloudTuner は `kerastuner.Tuner` のサブクラスであるため、`tuner_fn` モジュールにドロップインできる代替として使用し、TFX Tuner コンポーネントの一部として実行できます。

以下は、`CloudTuner` の使用方法を示すコードスニペットです。`CloudTuner` への構成には、`project_id` や `region` といった GCP 固有の項目が必要であることに注意してください。

```python
...
from tensorflow_cloud import CloudTuner

...
def tuner_fn(fn_args: FnArgs) -> TunerFnResult:
  """An implementation of tuner_fn that instantiates CloudTuner."""

  ...
  tuner = CloudTuner(
      _build_model,
      hyperparameters=...,
      ...
      project_id=...,       # GCP Project ID
      region=...,           # GCP Region where Vizier service is run.
  )

  ...
  return TuneFnResult(
      tuner=tuner,
      fit_kwargs={...}
  )

```

### Cloud AI Platformトレーニングの分散ワーカーフロックの並行チューニング

Tuner コンポーネントの基盤の実装としての KerasTuner フレームワークには、ハイパーパラメータ検索を並行して行える機能があります。ストック Tuner コンポーネントには 2 つ以上の検索ワーカーを並行して実行する機能はありませんが、[Google Cloud AI Platform 拡張機能の Tuner コンポーネント](https://github.com/tensorflow/tfx/blob/master/tfx/extensions/google_cloud_ai_platform/tuner/component.py)を使用すると、AI Platform トレーニングジョブを分散型ワーカーのフロックマネージャとして使用し、チューニングを並行して行えるようになります。[TuneArgs](https://github.com/tensorflow/tfx/blob/master/tfx/proto/tuner.proto) がこのコンポーネントに与えられた構成です。これはストック Tuner コンポーネントのドロップイン代替コンポーネントです。

```python
tuner = google_cloud_ai_platform.Tuner(
    ...   # Same kwargs as the above stock Tuner component.
    tune_args=proto.TuneArgs(num_parallel_trials=3),  # 3-worker parallel
    custom_config={
        # Configures Cloud AI Platform-specific configs . For for details, see
        # https://cloud.google.com/ai-platform/training/docs/reference/rest/v1/projects.jobs#traininginput.
        TUNING_ARGS_KEY:
            {
                'project': ...,
                'region': ...,
                # Configuration of machines for each master/worker in the flock.
                'masterConfig': ...,
                'workerConfig': ...,
                ...
            }
    })
...

```

拡張機能 Tuner コンポーネントの動作と出力は、ストック Tuner コンポーネントと変わりませんが、複数のハイパーパラメータ検索を異なるワーカーマシンで並行して行える点で異なります。このため、`num_trials` がより速く完了します。これは、検索アルゴリズムが `RandomSearch` のようにに驚異的に並列化可能である場合に特に効果的ですが、AI Platform Vizier に実装された Google Vizier アルゴリズムのように、検索アルゴリズムが前回のトライアルの結果の情報を使用する場合、過剰に並列化された検索によって検索の効率性に悪影響を及ぼしてしまいます。

注意: 並列検索の各トライアルは、ワーカーフロックの 1 つのマシンで実行されるため、各トライアルはマルチワーカーの分散型トレーニングを利用しません。マルチワーカー分散を各トライアルで実施する場合は、[`DistributingCloudTuner`](https://github.com/tensorflow/cloud/blob/b9c8752f5c53f8722dfc0b5c7e05be52e62597a8/src/python/tensorflow_cloud/tuner/tuner.py#L384-L676) を参照してください（`CloudTuner` は使用しません）。

注意: `CloudTuner` と Google Cloud AI Platform 拡張機能 Tuner コンポーネントを合わせて使用することができます。この場合、Ai Platform Vizier のハイパーパラメータ検索アルゴリズムのサポートで、分散型並列検索が可能です。ただし、これを行うにはCloud AI Platform ジョブが AI Platform Vizier サービスにアクセスする必要があります。カスタムサービスアカウントをセットアップするには、こちらの[ガイド](https://cloud.google.com/ai-platform/training/docs/custom-service-account#custom)をご覧ください。その後で、パイプラインコードにトレーニングジョブのカスタムサービスアカウントを指定する必要があります。詳細は、[GCP における E2E CloudTuner の例](https://github.com/tensorflow/tfx/blob/master/tfx/examples/penguin/penguin_pipeline_kubeflow_gcp.py)をご覧ください。

## リンク

[E2E の例](https://github.com/tensorflow/tfx/blob/master/tfx/examples/penguin/penguin_pipeline_local.py)

[E2E CloudTuner on GCP Example](https://github.com/tensorflow/tfx/blob/master/tfx/examples/penguin/penguin_pipeline_kubeflow.py)

[KerasTuner チュートリアル](https://www.tensorflow.org/tutorials/keras/keras_tuner)

[CloudTuner チュートリアル](https://github.com/GoogleCloudPlatform/ai-platform-samples/blob/master/notebooks/samples/optimizer/ai_platform_vizier_tuner.ipynb)

[提案](https://github.com/tensorflow/community/blob/master/rfcs/20200420-tfx-tuner-component.md)

より詳細な情報は、[Tuner API リファレンス](https://www.tensorflow.org/tfx/api_docs/python/tfx/v1/components/Tuner)をご覧ください。
