## Stub Executor を使用したパイプラインのテスト

### はじめに

**このチュートリアルを始める前に、[template.ipynb](https://github.com/tensorflow/tfx/blob/master/docs/tutorials/tfx/template.ipynb) チュートリアルの *ステップ 6 *まで完了する必要があります。**

このドキュメントでは、ゴールデンテストデータを使用して偽のアーティファクトを生成する `BaseStubExecuctor` を使用して TensorFlow Extended (TFX) パイプラインをテストする手順を説明します。これは、テストしない Executor を置き換えることにより、実際の Executor の実行時間を短縮化することを目的としています。Stub Executor は、TFX Python パッケージの `tfx.experimental.pipeline_testing.base_stub_executor` から提供されています。

このチュートリアルは、`template.ipynb` チュートリアルを拡張するものであり、シカゴ市が公開した[タクシー乗降データセット](https://data.cityofchicago.org/Transportation/Taxi-Trips/wrvz-psew)を使用します。Stub Executor を利用する前に、コンポーネントを変更してみることを強くお勧めします。

### 1. Google Cloud Storage にパイプライン出力を記録する

まず、パイプライン出力を記録して、Stub Executor が記録された出力からアーティファクトをコピーできるようにする必要があります。

このチュートリアルは、`template.ipynb` をステップ 6 まで完了していることを前提としているので、正常なパイプラインの実行は [MLMD](https://www.tensorflow.org/tfx/guide/mlmd) に保存されているはずです。MLMD の実行情報には、gRPC サーバーを使用してアクセスできます。

ターミナルを開き、次のコマンドを実行します。

1. 適切な資格情報を使用して kubeconfig ファイルを生成します。`bash gcloud container clusters get-credentials $cluster_name --zone $compute_zone --project $gcp_project_id` `$compute_zone` は、gcp エンジンのリージョンで、`$gcp_project_id` は GCP プロジェクトのプロジェクト ID です。

2. MLMD に接続するためのポート転送を設定します。`bash nohup kubectl port-forward deployment/metadata-grpc-deployment -n $namespace $port:8080 &` `$namespace` はクラスターの名前空間で、`$port` はポート転送に使用される未使用のポートです。

3. tfx GitHub リポジトリのクローンを作成します。tfx ディレクトリ内で、次のコマンドを実行します。

```bash
python tfx/experimental/pipeline_testing/pipeline_recorder.py \
--output_dir=gs://<gcp_project_id>-kubeflowpipelines-default/testdata \
--host=$host \
--port=$port \
--pipeline_name=$pipeline_name
```

`$output_dir` は、パイプライン出力を記録する Google Cloud Storage のパスに設定する必要があるため、必ず `<gcp_project_id>` を GCP プロジェクト ID に置き換えてください。

`$host` と `$port` は、MLMD に接続するためのメタデータ grpc サーバーのホスト名とポートです。`$port` は、ポート転送に使用したポート番号、ホスト名は「localhost」に設定します。

`template.ipynb` チュートリアルでは、パイプライン名はデフォルトで「my_pipeline」に設定されているため、`pipeline_name="my_pipeline"` と設定します。テンプレートチュートリアルの実行時にパイプライン名を変更した場合は、それに応じて `--pipeline_name` を変更する必要があります。

### 2. Kubeflow DAG Runner で Stub Executor を有効にする

まず、`tfx template copy` CLI コマンドを使用して、事前定義されたテンプレートがプロジェクトディレクトリにコピーされていることを確認します。コピーしたソースファイル内の以下の 2 つのファイルを編集する必要があります。

1. kubeflow_dag_runner.py が配置されているディレクトリに `stub_component_launcher.py` というファイルを作成し、次のコンテンツをそのファイルに配置します。

    ```python
    from tfx.experimental.pipeline_testing import base_stub_component_launcher
    from pipeline import configs

    class StubComponentLauncher(
        base_stub_component_launcher.BaseStubComponentLauncher):
      pass

    # GCS directory where KFP outputs are recorded
    test_data_dir = "gs://{}/testdata".format(configs.GCS_BUCKET_NAME)
    # TODO: customize self.test_component_ids to test components, replacing other
    # component executors with a BaseStubExecutor.
    test_component_ids = ['Trainer']
    StubComponentLauncher.initialize(
        test_data_dir=test_data_dir,
        test_component_ids=test_component_ids)
    ```

    注意：ランチャークラスはモジュールパスによりインポートされるため、このスタブコンポーネントランチャーは `kubeflow_dag_runner.py` 内で定義できません。

2. コンポーネント ID を、テストするコンポーネント ID のリストに設定します (つまり、他のコンポーネントの Executor が BaseStubExecutor に置き換えられます)。

3. `kubeflow_dag_runner.py` を開きます。追加した `StubComponentLauncher` クラスを使用するには、次のインポートステートメントを上部に追加します。

    ```python
    import stub_component_launcher
    ```

4. `kubeflow_dag_runner.py` で、`StubComponentLauncher` クラスを `KubeflowDagRunnerConfig` の `supported_launcher_class` に追加して、Stub Executor の起動を有効にします。

    ```python
    runner_config = kubeflow_dag_runner.KubeflowDagRunnerConfig(
        supported_launcher_classes=[
            stub_component_launcher.StubComponentLauncher
        ],
    ```

### 3. Stub Executor を使用してパイプラインを更新および実行する

Stub Executor を使用してパイプライン定義を変更し、既存のパイプラインを更新します。

```bash
tfx pipeline update --pipeline-path=kubeflow_dag_runner.py \
  --endpoint=$endpoint --engine=kubeflow
```

`$endpoint` は、KFP クラスターエンドポイントに設定する必要があります。

次のコマンドを実行して、更新されたパイプラインの新しい実行を作成します。

```bash
tfx run create --pipeline-name $pipeline_name --endpoint=$endpoint \
  --engine=kubeflow
```

## クリーンアップ

コマンド `fg` を使用してバックグラウンドでポート転送にアクセスし、次に ctrl-C を使用して終了します。`gsutil -m rm -R $output_dir` を使用すると、パイプライン出力が記録されているディレクトリを削除できます。

このプロジェクトで使用されているすべての Google Cloud リソースをクリーンアップするには、チュートリアルで使用した [Google Cloud プロジェクトを削除](https://cloud.google.com/resource-manager/docs/creating-managing-projects#shutting_down_projects)します。

または、各コンソール ([Google Cloud Storage](https://console.cloud.google.com/storage) - [Google Container Registry](https://console.cloud.google.com/gcr) - [Google Kubernetes Engine](https://console.cloud.google.com/kubernetes)) にアクセスして、個々のリソースをクリーンアップできます。
