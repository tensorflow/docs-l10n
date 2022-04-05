# Cloud AI Platform Pipelines上の TFX

## はじめに

このチュートリアルでは、[TensorFlow Extended（TFX）](https://www.tensorflow.org/tfx)と [AI Platform Pipelines]（https://cloud.google.com/ai-platform/pipelines/docs/introduction）を紹介し、Google Cloud で独自の機械学習パイプラインを作成する方法を確認します。また、TFX、AI Platform Pipelines、Kubeflow との統合や Jupyter Notebook の TFX とのインタラクティブな動作を見ていきます。

このチュートリアルの最終ステップでは、Google Cloud でホストされる機械学習パイプラインを作成して実行します。また、それぞれの実行の結果を視覚化し、作成されたアーティファクトの系統を表示します。

重要な用語：TFX パイプラインは、有向非巡回グラフ（DAG）で DAG と呼ばれることもあります。

ここでは典型的な機械学習開発プロセスに従い、データセットを調べてから最終的に完全に機能するパイプラインを作成します。また、パイプラインをデバッグおよび更新し、パフォーマンスを測定する方法を学びます。

注: このチュートリアルの所要時間は、約45 ～ 60 分です。

### シカゴのタクシー データセット

<!-- Image free for commercial use, does not require attribution:
https://pixabay.com/photos/new-york-cab-cabs-taxi-urban-city-2087998/ -->

![Taxi](https://github.com/tensorflow/docs-l10n/blob/master/site/ja/tfx/tutorials/tfx/images/airflow_workshop/taxi.jpg?raw=true) ![Chicago taxi](https://github.com/tensorflow/docs-l10n/blob/master/site/ja/tfx/tutorials/tfx/images/airflow_workshop/chicago.png?raw=true)

シカゴ市からリリースされた[タクシートリップデータセット](https://data.cityofchicago.org/Transportation/Taxi-Trips/wrvz-psew)のデータを使用します。

注：この Web サイトは、シカゴ市の公式 Web サイト www.cityofchicago.org で公開されたデータを変更して使用するアプリケーションを提供します。シカゴ市は、この Web サイトで提供されるデータの内容、正確性、適時性、または完全性について一切の表明を行いません。この Web サイトで提供されるデータは、随時変更される可能性があり、提供されるデータはユーザーの自己責任で利用されるものとします。

データセットの詳細については、[Google BigQuery](https://cloud.google.com/bigquery/public-data/chicago-taxi) を[参照](https://cloud.google.com/bigquery/)してください。[BigQuery UI](https://bigquery.cloud.google.com/dataset/bigquery-public-data:chicago_taxi_trips) でデータセット全体をご確認ください。

#### モデル目標 - 二項分類

顧客は 20% 以上のチップを払うでしょうか？

## 1. Google Cloud プロジェクトをセットアップする

### 1.a Google Cloud で環境を設定する

このチュートリアルをはじめるには、Google Cloud アカウントが必要です。すでにアカウントがある場合は、スキップして新規プロジェクトを作成してください。

警告: このデモは、Google Cloud の無料枠の制限を超えないように設計されています。すでに Google アカウントをお持ちの場合は、無料枠の上限に達しているか、新規ユーザーに提供される無料の Google Cloud クレジットを使い果たしている可能性があります。そのような場合、このデモを行うと、Google Cloud アカウントに課金が発生します。

1. Google Cloud Console に移動します。

2. Google Cloud の利用規約に同意します。

    <img src="https://github.com/tensorflow/docs-l10n/blob/master/site/ja/tfx/tutorials/tfx/images/cloud-ai-platform-pipelines/welcome-popup.png?raw=true" class="">

3. 無料トライアルアカウントを使用する場合は、[**Try For Free**](https://console.cloud.google.com/freetrial)（または [**Get started for free**](https://console.cloud.google.com/freetrial)）をクリックします。

    1. 国を選択します。

    2. 利用規約に同意します。

    3. 請求先の詳細を入力します。

        この時点では課金されることはありません。他の Google Cloud プロジェクトがない場合は、Google Cloud の無料枠の上限を超えることなくこのチュートリアルを完了することができます。これには、同時に実行される最大 8 コアが含まれます。

注: この時点で、無料トライアルを使用する代わりに、有料ユーザーになることを選択できます。このチュートリアルは無料枠の上限を超えないので、これが唯一のプロジェクトで、この制限内で利用する場合は課金は発生しません。詳細については、Google Cloud 料金計算ツールと Google Cloud Platform の無料枠をご覧ください。

### 1.b 新規プロジェクトを作成する

注: このチュートリアルは、新しいプロジェクトでこのデモに取り組むことを前提としています。必要に応じて、既存のプロジェクトで作業することもできます。

注：プロジェクトを作成する前に、検証済みのクレジットカードを登録しておく必要があります。

1. Google Cloud メイン ダッシュボードで、[Google Cloud Platform] ヘッダーの横にあるプロジェクトのプルダウンをクリックし、[新規プロジェクト] を選択します。
2. プロジェクトに名前を付け、他のプロジェクトの詳細を入力します。
3. **プロジェクトを作成したら、プロジェクトのドロップダウンからそれを選択します。**

## 2. AI Platform Pipeline をセットアップして新しい Kubernetes クラスタにデプロイする

注: リソースがプロビジョニングされるまでに数回待機する必要があるため、これには最大 10 分かかります。

1. AI Platform Pipelines クラスタのページに移動します。

    メインナビゲーションメニューの下: ≡ &gt; AI Platform &gt; パイプライン

2. [+ 新規インスタンス] をクリックして新しいクラスタを作成します。

    <img style="width: 65%;" src="images/cloud-ai-platform-pipelines/new-instance.png">

3. Kubeflow Pipelines の概要ページで、[構成] をクリックします。

    <img style="width: 65%;" src="images/cloud-ai-platform-pipelines/configure.png">

4. [有効化] をクリックして、Kubernetes Engine API を有効にします。

    <img src="images/cloud-ai-platform-pipelines/select-notebook.png" data-md-type="image" alt="select-notebook">

    注意: 次に進む前に、Kubernetes Engine API が有効になるまで数分かかることがある場合があります。

5. [Deploy Kubeflow Pipelines] ページで:

    1. クラスタの[ゾーン](https://cloud.google.com/compute/docs/regions-zones)（または「リージョン」）を背たくします。ネットワークとサブネットワークを設定することはできますが、このチュートリアルではデフォルトのままにします。

    2. **重要** [*次のクラウド API へのアクセスを許可する*] というラベルの付いたボックスをオンにします。（これは、このクラスターがプロジェクトの他の部分にアクセスするために必要です。この手順を怠ると、後で修正するのが少し難しくなります。）

        <img style="width: 65%;" src="images/cloud-ai-platform-pipelines/two-cpus.png">

    3. [**新しいクラスターを作成する**] をクリックして、クラスターが作成されるまで数分待ちます。完了したら、次のようなメッセージが表示されます。

        > Cluster "cluster-1" successfully created in zone "us-central1-a".

    4. 名前空間とインスタンス名を選択します (デフォルトを使用しても問題ありません)。このチュートリアルでは、*executor.emissary* または *managedstorage.enabled* をオンにしないでください。

    5. [**デプロイ**] をクリックし、パイプラインがデプロイされるまでしばらく待ちます。Kubeflow Pipelines をデプロイすることにより、利用規約に同意したことになります。

## 3. Cloud AI Platform Notebook インスタンスをセットアップする

1. [Vertex AI Workbench](https://console.cloud.google.com/vertex-ai/workbench) ページに移動します。Workbench を初めて実行した場合は、Notebooks API を有効化する必要があります。

    メインナビゲーションメニューの ≡ から、Vertex AI -&gt; Workbench い移動します。

2. プロンプトが表示されたら、Compute Engine API を有効にします。

3. インストール済みの TensorFlow Enterprise 2.7（以降）を使って、**新しいノートブック**を作成します。

    <img src="https://github.com/tensorflow/docs-l10n/blob/master/site/ja/tfx/tutorials/tfx/images/airflow_workshop/taxi.jpg?raw=true" data-md-type="image" class="" alt="Taxi"> <img src="https://github.com/tensorflow/docs-l10n/blob/master/site/ja/tfx/tutorials/tfx/images/airflow_workshop/chicago.png?raw=true" data-md-type="image" class="" alt="Chicago taxi">

    新規ノートブック -&gt; TensorFlow Enterprise 2.7 -&gt; GPU なし

    リージョンとゾーンを選択し、ノートブックインスタンスに名前を付けます。

    無料利用枠の制限内に留まるには、ここに示されるデフォルト設定を変更することをお勧めします。このインスタンスで利用できる vCPU の数を 4 から 2 に減らしてください。

    1. [**新規 Notebook**] フォームの下部にある [**高度なオプション**] を選択します。

    2. 無料利用枠の制限内に留まるには、[**マシンの構成**] で、vCPU の数を 1 または 2 に設定します。

        <img style="width: 65%;" src="images/cloud-ai-platform-pipelines/two-cpus.png">

    3. 新しいノートブックが作成されるのを待ってから、[**Notebooks API を有効化**] をクリックします。

注意: デフォルト設定またはそれ以上の設定ではなく 1 つまたは 2 つの vCPU を使用する場合、Notebook のパフォーマンスが低下する可能性がありますが、このチュートリアルの完了を著しく妨げることはありません。デフォルト設定を使用する場合は、少なくとも 12 vCPU に[アカウントをアップグレード](https://cloud.google.com/free/docs/gcp-free-tier#to_upgrade_your_account)してください。これにより、課金が発生します。[料金計算ツール](https://cloud.google.com/products/calculator)や [Google Cloud 無料枠](https://cloud.google.com/free)に関する情報など、料金の詳細については、[Google Kubernetes Engine の料金](https://cloud.google.com/kubernetes-engine/pricing/)を参照してください。

## 4. Getting Started Notebook をローンチする

1. [**AI Platform Pipelines Clusters**]（(https://console.cloud.google.com/ai-platform/pipelines）ページに移動します。

    メインナビゲーションメニュー ≡ から AI Platform &gt; Pipelines

2. このチュートリアルで使用しているクラスタの行の、[パイプライン ダッシュボードを開く] をクリックします。

    <img src="images/cloud-ai-platform-pipelines/open-dashboard.png">

3. **Getting Started** ページで、[**Google Cloud で Cloud AI Platform Notebook を開く**] をクリックします。

    <img src="images/cloud-ai-platform-pipelines/open-template.png">

4. このチュートリアルで使用する Notebook インスタンスを選択し、[**続行**]、次に [**確認**] をクリックします。

    <img style="width: 50%;" src="images/cloud-ai-platform-pipelines/check-the-box.png">

## 5. Notebook で作業を続行する

重要: このチュートリアルの残りの部分は、前述のステップで開いた Jupyter Lab Notebook で完了する必要があります。ここに記載された手順と説明を参照してください。

### インストール

Getting Started Notebook をはじめるには、まず、Jupyter Lab が実行されている VM に TFX と Kubeflow Pipelines (KFP) をインストールします。

次に、インストールされている TFX のバージョンを確認し、インポートを行い、プロジェクト ID を設定して出力します。

![check python version and import](images/cloud-ai-platform-pipelines/check-version-nb-cell.png)

### Google Cloud サービスと接続する

パイプライン構成にはプロジェクト ID が必要です。これは、Notebook を介して取得し、環境変数として設定できます。

```python
# Read GCP project id from env.
shell_output=!gcloud config list --format 'value(core.project)' 2>/dev/null
GCP_PROJECT_ID=shell_output[0]
print("GCP project ID:" + GCP_PROJECT_ID)
```

次に、KFP クラスターのエンドポイントを設定します。

これは、Pipelines ダッシュボードの URL から見つけることができます。Kubeflow Pipeline ダッシュボードに移動し、URL を確認します。エンドポイントは、URL 内のhttps://からgoogleusercontent.comまでのすべてです（googleusercontent.com を含む）。

```python
ENDPOINT='' # Enter YOUR ENDPOINT here.
```

次に、Notebook はカスタム Docker イメージに一意の名前を設定します。

```python
# Docker image name for the pipeline image
CUSTOM_TFX_IMAGE='gcr.io/' + GCP_PROJECT_ID + '/tfx-pipeline'
```

## 6. テンプレートをプロジェクト ディレクトリにコピーする

次の Notebook セルを編集して、パイプラインの名前を設定します。このチュートリアルでは、my_pipelineを使用します。

```python
PIPELINE_NAME="my_pipeline"
PROJECT_DIR=os.path.join(os.path.expanduser("~"),"imported",PIPELINE_NAME)
```

次に、Notebook は tfx CLI を使用してパイプライン テンプレートをコピーします。このチュートリアルでは、Chicago Taxi データセットを使用してバイナリ分類を実行するため、テンプレートはモデルをtaxiに設定します。

```python
!tfx template copy \
  --pipeline-name={PIPELINE_NAME} \
  --destination-path={PROJECT_DIR} \
  --model=taxi
```

次に、Notebook はその CWD コンテキストをプロジェクト ディレクトリに変更します。

```
%cd {PROJECT_DIR}
```

### パイプラインファイルを閲覧する

Cloud AI Platform Notebook の左側に、ファイル ブラウザが表示されます。そのパイプライン名 (my_pipeline) のディレクトリがあるはずです。それを開いてファイルを表示します。 (Notebook 環境からも開いて編集できます。)

```
# You can also list the files from the shell
! ls
```

上記のtfx template copyコマンドは、パイプラインを構築するファイルの基本的なスキャフォールドを作成しました。これらには、Python ソース コード、サンプルデータ、Jupyter Notebook が含まれます。これらは、この特定のサンプルを対象としています。独自のパイプラインの場合、これらはパイプラインに必要なサポート ファイルになります。

Python ファイルの簡単な説明を次に示します。

- pipeline - このディレクトリには、パイプラインの定義が含まれています。
    - configs.py — パイプライン ランナーの共通定数を定義します。
    - pipeline.py — TFX コンポーネントとパイプラインを定義します。
- models - このディレクトリには、機械学習モデルの定義が含まれています。
    - features.py features_test.py — モデルの機能を定義します
    - `preprocessing.py` / `preprocessing_test.py` — `tf::Transform` を使って前処理ジョブを定義します
    - estimator - このディレクトリには、Estimator ベースのモデルが含まれています。
        - constants.py — モデルの定数を定義します。
        - model.py / model_test.py — TF estimator を使用して DNN モデルを定義します
    - keras - このディレクトリには、Keras ベースのモデルが含まれています。
        - constants.py — モデルの定数を定義します。
        - model.py / model_test.py — Keras を使用して DNN モデルを定義します。
- beam_runner.py / kubeflow_runner.py — オーケストレーション エンジンごとにランナーを定義します。

## 7. Kubeflow で最初の TFX パイプラインを実行

Notebook は、tfx run CLI コマンドを使用してパイプラインを実行します。

### ストレージに接続

パイプラインを実行するとアーティファクトが作成されます。これは、ML-Metadata に保存する必要があります。アーティファクトは、ファイル システムまたはブロック ストレージに格納する必要があるファイルであるペイロードを指します。このチュートリアルでは、セットアップ中に自動的に作成されたバケットを使用して、GCS を使用してメタデータ ペイロードを保存します。名前は&lt;your-project-id&gt;-kubeflowpipelines-defaultになります。

### パイプラインを作成する

Notebook はサンプル データを GCS バケットにアップロードして、後でパイプラインで使用できるようにします。

```python
!gsutil cp data/data.csv gs://{GOOGLE_CLOUD_PROJECT}-kubeflowpipelines-default/tfx-template/data/taxi/data.csv
```

次に、Notebook はtfx pipeline createコマンドを使用してパイプラインを作成します。

```python
!tfx pipeline create  \
--pipeline-path=kubeflow_runner.py \
--endpoint={ENDPOINT} \
--build-image
```

パイプラインの作成中に、Docker イメージをビルドするためのDockerfileが生成されます。これらのファイルを他のソースファイルと一緒にソース管理システム (たとえば、git) に追加することを忘れないでください。

### パイプラインの実行

次に、Notebook はtfx run createコマンドを使用して、パイプラインの実行を開始します。この実行は、Kubeflow Pipelines ダッシュボードの [実験] の下にも表示されます。

```python
!tfx run create --pipeline-name={PIPELINE_NAME} --endpoint={ENDPOINT}
```

Kubeflow Pipelines ダッシュボードからパイプラインを表示できます。

注: パイプラインの実行が失敗した場合、KFP ダッシュボードで詳細なログを確認できます。失敗の主な原因の 1 つは、許可関連の問題です。KFP クラスタに Google Cloud API へのアクセス権限があることを確認してください。これは、GCP で KFP クラスタを作成するときに構成します。GCP でのトラブルシューティング ドキュメントを参照してください。

## 8. データを検証する

データ サイエンスや機械学習プロジェクトの最初のタスクは、データを理解してクリーンアップすることです。

- 各特徴のデータ型を理解する
- 異常と欠損値を探す
- 各特徴の分布を理解する

### コンポーネント

![Data Components](images/airflow_workshop/examplegen1.png) ![Data Components](images/airflow_workshop/examplegen2.png)

- [ExampleGen](https://www.tensorflow.org/tfx/guide/examplegen) は入力データセットを取り込み、分割します。
- [StatisticsGen](https://www.tensorflow.org/tfx/guide/statsgen) はデータセットの統計を計算します。
- [SchemaGen](https://www.tensorflow.org/tfx/guide/schemagen) は統計を調べ、データ スキーマを作成します。
- [ExampleValidator](https://www.tensorflow.org/tfx/guide/exampleval) はデータセット内の異常と欠損値を探します。

### Jupyter ラボ ファイル エディターで以下を行います

pipeline/pipeline.pyで、これらのコンポーネントをパイプラインに追加する行のコメントを外します。

```python
# components.append(statistics_gen)
# components.append(schema_gen)
# components.append(example_validator)
```

（ExampleGenは、テンプレート ファイルがコピーされたときにすでに有効になっています）。

### パイプラインを更新して再実行する

```python
# Update the pipeline
! tfx pipeline update \
  --pipeline-path=kubeflow_runner.py \
  --endpoint={ENDPOINT}

! tfx run create --pipeline-name "{PIPELINE_NAME}"
```

### パイプラインを確認する

Kubeflow Orchestrator の場合、KFP ダッシュボードにアクセスし、パイプライン実行のページでパイプライン出力を見つけます。左側の [実験] タブをクリックし、[実験] ページの [すべての実行] をクリックします。パイプラインの名前の実行が表示されるはずです。

### より高度な例

ここに示されている例は、初心者向けのものです。より高度な例については、[TensorFlow Data Validation Colab](https://www.tensorflow.org/tfx/tutorials/data_validation/chicago_taxi) を参照してください。

TFDV を使用してデータセットを調査および検証する方法の詳細については、[tensorflow.org の例を参照してください](https://www.tensorflow.org/tfx/data_validation)。

## 9. 特徴量エンジニアリング

特徴量エンジニアリングを使用すると、データの予測品質を向上させたり、次元を減らしたりすることができます。

- 特徴量クロス
- 語彙
- 埋め込み
- PCA
- カテゴリカル変数のエンコーディング

TFX を使用する利点の 1 つは、変換コードを 1 回記述すれば、結果として得られる変換はトレーニングとサービングの間で一貫性を保てることです。

### コンポーネント

![Transform](images/airflow_workshop/transform.png)

- [Transform](https://www.tensorflow.org/tfx/guide/transform)は、データセットに対する特徴量エンジニアリングを実行します。

### Jupyter ラボ ファイル エディターで以下を行います

pipeline/pipeline.pyで、パイプラインに Transform を追加する行を見つけてコメントを外します。

```python
# components.append(transform)
```

### パイプラインを更新して再実行する

```python
# Update the pipeline
! tfx pipeline update \
  --pipeline-path=kubeflow_runner.py \
  --endpoint={ENDPOINT}

! tfx run create --pipeline-name "{PIPELINE_NAME}"
```

### パイプライン出力を確認する

Kubeflow Orchestrator の場合、KFP ダッシュボードにアクセスし、パイプライン実行のページでパイプライン出力を見つけます。左側の [実験] タブをクリックし、[実験] ページの [すべての実行] をクリックします。パイプラインの名前の実行が表示されるはずです。

### より高度な例

ここに示されている例は、初心者向けのものです。より高度な例については、[TensorFlow Transform Colab](https://www.tensorflow.org/tfx/tutorials/transform/census) を参照してください。

## 10. トレーニング

TensorFlow モデルを、クリーンアップおよび変換されたデータでトレーニングします。

- 前のステップの変換を含めて、一貫して適用されるようにします。
- 実稼働用に結果を SavedModel として保存します。
- TensorBoard を使用してトレーニング プロセスを視覚化して調査します。
- また、モデル パフォーマンスの分析のために EvalSavedModel を保存します。

### コンポーネント

- [Trainer](https://www.tensorflow.org/tfx/guide/trainer) は TensorFlow モデルをトレーニングします。

### Jupyter ラボ ファイル エディターで以下を行います

pipeline/pipeline.py で、パイプラインに Trainer を追加する行を見つけてコメントを外します。

```python
# components.append(trainer)
```

### パイプラインを更新して再実行する

```python
# Update the pipeline
! tfx pipeline update \
  --pipeline-path=kubeflow_runner.py \
  --endpoint={ENDPOINT}

! tfx run create --pipeline-name "{PIPELINE_NAME}"
```

### パイプライン出力を確認する

Kubeflow Orchestrator の場合、KFP ダッシュボードにアクセスし、パイプライン実行のページでパイプライン出力を見つけます。左側の [実験] タブをクリックし、[実験] ページの [すべての実行] をクリックします。パイプラインの名前の実行が表示されるはずです。

### より高度な例

ここに示されている例は、初心者向けのものです。より高度な例については、[TensorFlow チュートリアル](https://www.tensorflow.org/tensorboard/r1/summaries) を参照してください。

## 11. モデルのパフォーマンスの分析

トップレベルの指標以上のことを理解します。

- ユーザーは自分のクエリに対してのみモデルのパフォーマンスを体験します。
- データ スライスのパフォーマンスの低下は、トップレベルの指標には示されない可能性があります。
- モデルの公平性は重要です。
- 多くの場合、ユーザーまたはデータの主要なサブセットは非常に重要です。これは小さい場合があります。
    - 重要かつ異常な状態でのパフォーマンス
    - インフルエンサーなどの主要オーディエンスに対するパフォーマンス
- 実稼働中のモデルを置き換える場合は、まず新しいモデルの方が優れていることを確認してください。

### コンポーネント

- [Evaluator](https://www.tensorflow.org/tfx/guide/evaluator) はトレーニング結果の詳細分析を実行します。

### Jupyter ラボ ファイル エディターで以下を行います

pipeline/pipeline.py で、Evaluator をパイプラインに追加する行を見つけてコメント解除します。

```python
components.append(evaluator)
```

### パイプラインを更新して再実行する

```python
# Update the pipeline
! tfx pipeline update \
  --pipeline-path=kubeflow_runner.py \
  --endpoint={ENDPOINT}

! tfx run create --pipeline-name "{PIPELINE_NAME}"
```

### パイプライン出力を確認する

Kubeflow Orchestrator の場合、KFP ダッシュボードにアクセスし、パイプライン実行のページでパイプライン出力を見つけます。左側の [実験] タブをクリックし、[実験] ページの [すべての実行] をクリックします。パイプラインの名前の実行が表示されるはずです。

## 12. モデルのサービング

新しいモデルの準備ができている場合は、準備を完了します。

- Pusher は SavedModels を既知の場所にデプロイします

デプロイメントターゲットは、既知の場所から新しいモデルを受け取ります。

- <a>TensorFlow Serving</a>
- TensorFlow Lite
- TensorFlow JS
- TensorFlow Hub

### コンポーネント

- [Pusher](https://www.tensorflow.org/tfx/guide/pusher) はモデルをサービングインフラストラクチャにデプロイします。

### Jupyter ラボ ファイル エディターで以下を行います

pipeline/pipeline.pyで、Pusher をパイプラインに追加する行を見つけてコメントを外します。

```python
# components.append(pusher)
```

### パイプライン出力を確認する

Kubeflow Orchestrator の場合、KFP ダッシュボードにアクセスし、パイプライン実行のページでパイプライン出力を見つけます。左側の [実験] タブをクリックし、[実験] ページの [すべての実行] をクリックします。パイプラインの名前の実行が表示されるはずです。

### 利用可能なデプロイメントターゲット

これでモデルのトレーニングと検証が完了し、モデルの実稼働環境の準備が整いました。次のような TensorFlow デプロイメント ターゲットのいずれかにモデルをデプロイできるようになりました。

- [TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving) はサーバーまたはサーバー ファームでモデルをサービングし、REST および/または gRPC 推論リクエストを処理します。
- [TensorFlow Lite](https://www.tensorflow.org/lite) は、モデルを Android または iOS のネイティブ モバイル アプリケーション、または Raspberry Pi、IoT、またはマイクロコントローラー アプリケーションに含めます。
- [TensorFlow.js](https://www.tensorflow.org/js) は、モデルを Web ブラウザまたは Node.JS アプリケーションで実行します。

## より高度な例

前述の例は、初心者向けのものです。以下は、他のクラウドサービスとの統合の例です。

### Kubeflow Pipelines のリソースに関する考慮事項

ワークロードの要件に対して、Kubeflow Pipelines デプロイメントのデフォルト構成がニーズを満たしていない場合があります。KubeflowDagRunnerConfigを呼び出し、pipeline_operator_funcsを使用して、リソース構成をカスタマイズできます。

pipeline_operator_funcsはOpFuncアイテムのリストであり、KubeflowDagRunnerからコンパイルされた KFP パイプライン仕様で生成されたすべてのContainerOpインスタンスを変換します。

たとえば、メモリを構成するには、set_memory_requestを使用して必要なメモリ量を宣言できます。これを行うには、一般的にset_memory_requestのラッパーを作成し、それを使用してパイプライン AggFuncのリストに追加します。

```python
def request_more_memory():
  def _set_memory_spec(container_op):
    container_op.set_memory_request('32G')
  return _set_memory_spec

# Then use this opfunc in KubeflowDagRunner
pipeline_op_funcs = kubeflow_dag_runner.get_default_pipeline_operator_funcs()
pipeline_op_funcs.append(request_more_memory())
config = KubeflowDagRunnerConfig(
    pipeline_operator_funcs=pipeline_op_funcs,
    ...
)
kubeflow_dag_runner.KubeflowDagRunner(config=config).run(pipeline)
```

同様に以下の関数を使用してリソースを構成できます。

- `set_memory_limit`
- `set_cpu_request`
- `set_cpu_limit`
- `set_gpu_limit`

### BigQueryExampleGenを試してみる

BigQuery は、サーバーレスでスケーラビリティと費用対効果の高いクラウド データ ウェアハウスです。BigQuery は、TFX のトレーニング サンプルのソースとして使用できます。このステップでは、パイプラインにBigQueryExampleGenを追加します。

#### Jupyter ラボ ファイル エディターで以下を行います

pipeline.pyをダブルクリックして開きます。CsvExampleGenをコメントアウトし、BigQuery Example Genのインスタンスを作成する行のコメントを外します。また、create_pipeline関数のquery引数のコメントも外す必要があります。

BigQuery に使用する GCP プロジェクトを指定する必要があります。そのためには、パイプラインの作成時にbeam_pipeline_argsに--projectを設定します。

configs.pyをダブルクリックして開きます。BIG_QUERY_WITH_DIRECT_RUNNER_BEAM_PIPELINE_ARGSとBIG_QUERY_QUERYの定義のコメントを外します。このファイルのプロジェクト ID とリージョンの値を、GCP プロジェクトの正しい値に置き換えます。

> **注意: 続行する前に、GCP プロジェクト ID とリージョンを `configs.py` ファイルに設定する必要があります。**

ディレクトリを 1 レベル上に変更します。ファイル リストの上にあるディレクトリの名前をクリックします。ディレクトリ名はパイプライン名で、パイプライン名を変更しなかった場合はmy_pipelineです。

ダブルクリックしてkubeflow_runner.py を開きます。create_pipeline関数の 2 つの引数 queryとbeam_pipeline_argsのコメントを外します。

パイプラインで BigQuery をサンプル ソースとして使用する準備ができました。前と同じようにパイプラインを更新し、ステップ 5 と 6 で行ったように新しい実行を作成します。

#### パイプラインを更新して再実行する

```python
# Update the pipeline
!tfx pipeline update \
  --pipeline-path=kubeflow_runner.py \
  --endpoint={ENDPOINT}

!tfx run create --pipeline-name {PIPELINE_NAME} --endpoint={ENDPOINT}
```

### Dataflow を試してみる

いくつかの [TFX コンポーネントは Apache Beam を使用](https://www.tensorflow.org/tfx/guide/beam)してデータ並列パイプラインを実装します。そのため、[Google Cloud Dataflow](https://cloud.google.com/dataflow/) を使用してデータ処理ワークロードを分散できます。このステップでは、Apache Beam のデータ処理バックエンドとして Dataflow を使用するように Kubeflow オーケストレーターを設定します。

> **注意:** Dataflow API がまだ有効になっていない場合は、コンソールを使用するか、CLI から次のコマンドを使用して（Cloud Shell など）有効にできます。

```bash
# Select your project:
gcloud config set project YOUR_PROJECT_ID

# Get a list of services that you can enable in your project:
gcloud services list --available | grep Dataflow

# If you don't see dataflow.googleapis.com listed, that means you haven't been
# granted access to enable the Dataflow API.  See your account adminstrator.

# Enable the Dataflow service:

gcloud services enable dataflow.googleapis.com
```

> **注意:** 実行速度は、デフォルトの Google Compute Engine（GCE） 割り当てにより制限される場合があります。約 250 の Dataflow VM に十分な割り当てを設定することをお勧めします（**250 個の CPU、250 個の IP アドレス、62500 GB の永続ディスク**）。詳細については、[GCE 割り当て](https://cloud.google.com/compute/quotas)と [Dataflow 割り当て](https://cloud.google.com/dataflow/quotas)のドキュメントを参照してください。IP アドレスの割り当てによりブロックされている場合は、より大きな [`worker_type`](https://cloud.google.com/dataflow/docs/guides/specifying-exec-params#setting-other-cloud-dataflow-pipeline-options) を使用すると、必要な IP の数を減らすことができます。

pipelineをダブルクリックしてディレクトリを変更し、configs.pyをダブルクリックして開きます。GOOGLE_CLOUD_REGIONとDATAFLOW_BEAM_PIPELINE_ARGSの定義のコメントを外します。

ディレクトリを 1 レベル上に変更します。ファイル リストの上にあるディレクトリの名前をクリックします。ディレクトリ名はパイプライン名で、変更しなかった場合はmy_pipelineです。

kubeflow_runner.pyをダブルクリックして開きます。beam_pipeline_argsのコメントを外します。（ステップ 7 で追加した最新のbeam_pipeline_argsも必ずコメントアウトしてください）。

#### パイプラインを更新して再実行する

```python
# Update the pipeline
!tfx pipeline update \
  --pipeline-path=kubeflow_runner.py \
  --endpoint={ENDPOINT}

!tfx run create --pipeline-name {PIPELINE_NAME} --endpoint={ENDPOINT}
```

Cloud Console の Dataflow で Dataflow のジョブを見つけることができます。

### KFP で Cloud AI Platform のトレーニングと予測を試す

TFX は、トレーニングと予測のための Cloud AI Platform など、いくつかのマネージド GCP サービスと相互運用します。機械学習モデルをトレーニングするためのマネージド サービスである Cloud AI Platform Training を使用するように Trainer コンポーネントを設定できます。さらに、モデルが構築され、サービングする準備ができたら、サービングするためにモデルを Cloud AI Platform Prediction に push できます。このステップでは、Cloud AI Platform サービスを使用するようにTrainerコンポーネントとPusherコンポーネントを設定します。

ファイルを編集する前に、最初に AI Platform Training と Prediction API を有効にする必要がある場合があります。

[pipeline]をダブルクリックしてディレクトリを変更し、ダブルクリックしてconfigs.pyを開きます。GOOGLE_CLOUD_REGION、GCP_AI_PLATFORM_TRAINING_ARGS、GCP_AI_PLATFORM_SERVING_ARGS の定義のコメントを外します。カスタム ビルドのコンテナ イメージを使用して Cloud AI Platform Training でモデルをトレーニングするため、GCP_AI_PLATFORM_TRAINING_ARGSのmasterConfig.imageUriを上記のCUSTOM_TFX_IMAGEと同じ値に設定する必要があります。

ディレクトリを 1 レベル上に変更し、ダブルクリックしてkubeflow_runner.pyを開きます。ai_platform_training_argsとai_platform_serving_argsのコメントを外します。

> 注意: トレーニングステップで権限エラーが発生した場合は、Cloud Machine Learning Engine (AI Platform Prediction と Training) サービス アカウントに Storage オブジェクト閲覧者権限を提供する必要がある場合があります。詳細については、[Container Registry のドキュメント](https://cloud.google.com/container-registry/docs/access-control#grant)をご覧ください。

#### パイプラインを更新して再実行する

```python
# Update the pipeline
!tfx pipeline update \
  --pipeline-path=kubeflow_runner.py \
  --endpoint={ENDPOINT}

!tfx run create --pipeline-name {PIPELINE_NAME} --endpoint={ENDPOINT}
```

トレーニング ジョブは Cloud AI Platform ジョブで見つけることができます。パイプラインが正常に完了した場合、Cloud AI Platform モデルでモデルを見つけることができます。

## 14. 独自のデータを使用する

このチュートリアルでは、Chicago Taxi データセットを使用してモデルのパイプラインを作成しました。次に、独自のデータをパイプラインに入れてみます。データは、Google Cloud Storage、BigQuery、CSV ファイルなど、パイプラインがアクセスできる場所ならどこにでも保存できます。

データに対応するようにパイプライン定義を変更する必要があります。

### データがファイルに保存されている場合

1. kubeflow_runner.pyのDATA_PATHを変更して、場所を示します。

### データが BigQuery に保存されている場合

1. configs.py のBIG_QUERY_QUERYをクエリ ステートメントに変更します。
2. models/features.pyに特徴量を追加します。
3. models/preprocessing.pyを変更して、トレーニング用の入力データを変換します。
4. models/keras/model.pyとmodels/keras/constants.pyを変更して、機械学習モデルを記述します。

### Trainer についての詳細

トレーニング パイプラインの詳細については、Trainer コンポーネント ガイドを参照してください。

## クリーンアップ

このプロジェクトで使用されているすべての Google Cloud リソースをクリーンアップするには、チュートリアルで使用した Google Cloud プロジェクトを削除します。

または、各コンソール（Google Cloud Storage - Google Container Registry - Google Kubernetes Engine）にアクセスして、個々のリソースをクリーンアップできます。
