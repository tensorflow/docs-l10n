# Apache Beam と TFX

[Apache Beam](https://beam.apache.org/) は、さまざまな実行エンジンで実行する、バッチおよびストリーミングデータ処理のジョブを実行するためのフレームワークを提供します。複数の TFX ライブラリではタスクの実行に Beam が使用されているため、コンピュートクラスタでの高度なスケーラビリティが可能です。Beam には、多様な実行エンジンまたは「ランナー」のサポートが含まれています。こういったランナーには、単一のコンピュートノードで実行する Direct Runner があり、開発、テスト、または小型のデプロイで非常に有用です。Beam は、TFX がコードを変更することなく、サポートされているあらゆるランナーで実行できるようにする抽象レイヤーを提供しています。TFX は Beam Python API を使用するため、Python API がサポートするランナーに制限されています。

## デプロイとスケーラビリティ

ワークロードの要件が増大するにつれ、Beam は大規模なコンピュートクラスタの非常に大規模なデプロイに合わせてスケーリングできます。唯一の制限は、その基盤にあるランナーのスケーラビリティによるものです。大規模なデプロイのランナーは通常、アプリケーションの自動デプロイ、スケーリング、および管理が可能な Kubernetes または Apache Mesos などのコンテナオーケストレーションシステムにデプロイされます。

Apache Beam についての詳細は、[Apache Beam](https://beam.apache.org/) ドキュメントを参照してください。

Google Cloud ユーザーの場合、[Dataflow](https://cloud.google.com/dataflow) が推奨されるランナーです。これは、リソースの自動スケーリング、動的な作業のリバランス、他の Google Cloud サービスとの緊密な統合、組み込みのセキュリティ、モニタリングを通じて、サーバーレスの費用効果の高いプラットフォームを提供します。

## カスタム Python コードと依存関係

TFX パイプラインで Beam を使用する際には、カスタムコードの処理や追加の Python モジュールから必要とされる依存関係に注意する必要があります。

- preprocessing_fn は、ユーザー自身の Python モジュールを参照する必要があります
- Evaluator コンポーネントのカスタムエクストラクタ
- TFX コンポーネントからサブクラス化されたカスタムモジュール

TFX は、Python の依存関係を処理するために [Python パイプラインの依存関係の管理](https://beam.apache.org/documentation/sdks/python-pipeline-dependencies/)に対する Beam のサポートに依存しています。現在、これを管理する方法は2つあります。

1. Python コードと依存関係をソースパッケージとして提供する
2. [Dataflow のみ]コンテナイメージをワーカーとして使用する

これらについての説明は、以下のとおりです。

### Python コードと依存関係をソースパッケージとして提供する

これは、次のユーザーに推奨されます。

1. Python パッケージに精通しているユーザー
2. Python ソースコードのみを使用するユーザー（C モジュールや共有ライブラリは使用しない）。

これを提供するには、次の beam_pipeline_args のいずれかを使用して [Python パイプラインの依存関係の管理](https://beam.apache.org/documentation/sdks/python-pipeline-dependencies/)のいずれかのパスに従います。

- --setup_file
- --extra_package
- --requirements_file

注意：上記のいずれの場合でも、同じバージョンの`tfx`が依存関係としてリストされていることを確認してください。

### [Dataflow のみ]コンテナイメージをワーカーとして使用する

TFX 0.26.0 以降では、Dataflow ワーカーに[カスタムコンテナイメージ](https://beam.apache.org/documentation/runtime/environments/#customizing-container-images)を使用するための試験的なサポートがあります。

これを使用するには、次のことを行う必要があります。

- `tfx`とユーザーのカスタムコードおよび依存関係の両方がプリインストールされた Docker イメージを作成します。
    - （1）`tfx>=0.26`を使用し、（2）python 3.7 を使用してパイプラインを開発するユーザーの場合、これを行う最も簡単な方法は、公式の`tensorflow/tfx`の対応するバージョンを拡張することです。

```Dockerfile
# You can use a build-arg to dynamically pass in the
# version of TFX being used to your Dockerfile.

ARG TFX_VERSION
FROM tensorflow/tfx:${TFX_VERSION}
# COPY your code and dependencies in
```

- ビルドされたイメージを、Dataflow で使用されるプロジェクトからアクセスできるコンテナイメージレジストリにプッシュします。
    - Google Cloud ユーザーは、上記の手順を適切に自動化する [Cloud Build](https://cloud.google.com/cloud-build/docs/quickstart-build) を使用できます。
- 以下の`beam_pipeline_args`を提供します。

```python
beam_pipeline_args.extend([
    '--runner=DataflowRunner',
    '--project={project-id}',
    '--worker_harness_container_image={image-ref}',
    '--experiments=use_runner_v2',
])
```

**TODO(b/171733562): Dataflow のデフォルトになったら、use_runner_v2 を削除します。**

**TODO(b/179738639): https://issues.apache.org/jira/browse/BEAM-5440 の後に、カスタムコンテナをローカルでテストする方法のドキュメントを作成します。**

## Beam パイプラインの引数

いくつかの TFX コンポーネントは、分散データ処理のために Beam に依存しています。これらは`beam_pipeline_args`で構成されます。これは、パイプラインの作成中に指定されます。

```python
my_pipeline = Pipeline(
    ...,
    beam_pipeline_args=[...])
```

TFX 0.30 以降では、コンポーネントごとにパイプラインレベルのビーム引数を拡張するためのインターフェイス`with_beam_pipeline_args`が追加されています。

```python
example_gen = CsvExampleGen(input_base=data_root).with_beam_pipeline_args([...])
```
