# **TFX Airflow のチュートリアル**

## 概要

## 概要

このチュートリアルは、TensorFlow Extended（TFX）とオーケストレータとしての Apache Airflow を使用して、独自の機械学習パイプラインを作成する方法を学習できるように設計されています。Vertex AI Workbench で実行し、TFX と TensorBoard との統合と、Jupyter Lab 環境での TFX の操作を説明します。

### 学習内容

TFX を使用して ML パイプラインを作成する方法を学習します。

- TFX パイプラインは、有向非巡回グラフ（DAG）です。パイプラインを通常 DAG と呼びます。
- TFX パイプラインは、実稼働環境の機械学習アプリケーションをデプロイする場合に適しています。
- TFX パイプラインは、データセットが大きい場合や大規模に成長する可能性がある場合に適しています。
- TFX パイプラインは、トレーニング/サービングの一貫性が重要な場合に適しています。
- TFX パイプラインは、推論のバージョン管理が重要な場合に適しています。
- Google は実稼働環境の機械学習に TFX パイプラインを使用しています。

詳細については、[TFX ユーザーガイド](https://www.tensorflow.org/tfx/guide)を参照してください。

典型的な ML 開発プロセスに従います。

- データの取り込み、理解、クリーニング
- 特徴量エンジニアリング
- トレーニング
- [モデルのパフォーマンスの分析](#step_6_analyzing_model_performance)
- サイクルを繰り返す
- 実稼働の準備完了

## **パイプラインオーケストレーション用の Apache Airflow**

パイプラインが定義する依存関係に基づいて TFX のコンポーネントをスケジューリングするのが TFX オーケストレータです。TFX は複数の環境とオーケストレーションフレームワークに移植できるように設計されています。TFX がサポートするデフォルトのオーケストレータの 1 つは [Apache Airflow](https://www.tensorflow.org/tfx/guide/airflow) です。このラブでは、TFX パイプラインオーケストレーションに Apache Airflow を使用する方法を説明します。Apache Airflow はワークフローの作成、スケジューリング、および監視をプログラムで行うためのプラットフォームです。TFX では、タスクの有向非巡回グラフ（DAG）としてワークフローを作成するのに Airflow を使用しています。リッチユーザーインターフェースが備わっているため、本番環境で実行するパイプラインの可視化、進捗状況の監視、必要に応じた課題のトラブルシューティングを簡単に行うことができます。Apache Airflow ワークフローはコードとして定義されるため、保守、バージョン管理、テスト、共同作業をより簡単に行えます。Apache Airflow はバッチ処理パイプラインに適しており、軽量で学びやすい特徴があります。

この例では、Airflow を手動でセットアップすることで、インスタンス上で TFX パイプラインを実行することにします。

TFX がサポートする他のデフォルトのオーケストレータには Apache Beam と Kubeflow があります。[Apache Beam](https://www.tensorflow.org/tfx/guide/beam_orchestrator) は複数のデータ処理バックエンド（Beam ランナー）で実行できます。Cloud Dataflow はこのような Beam ランナーの 1 つであり、TFX パイプラインの実行に使用できます。Apache Beam はパイプラインのストリーミングとバッチ処理のいずれにも使用可能です。<br> [Kubeflow](https://www.tensorflow.org/tfx/guide/kubeflow) は、Kubernetes での機械学習（ML）ワークフローのデプロイに単純さ、移植性、および拡張性を備えさせることに特化したオープンソースの ML プラットフォームです。Kubeflow は、TFX パイプラインを Kubernetes クラスタにデプロイする必要がある場合のオーケストレータとして使用できます。また、独自の[カスタムオーケストレータ](https://www.tensorflow.org/tfx/guide/custom_orchestrator)を使用して TFX パイプラインを実行することも可能です。

Airflow についての詳細は、[こちら](https://airflow.apache.org/)をお読みください。

## **シカゴのタクシー データセット**

![taxi.jpg](images/airflow_workshop/taxi.jpg)

![Feature Engineering](https://github.com/tensorflow/docs-l10n/blob/master/site/ja/tfx/tutorials/tfx/images/airflow_workshop/step4.png?raw=true)

シカゴ市がリリースした [Taxi Trips データセット](https://data.cityofchicago.org/Transportation/Taxi-Trips/wrvz-psew)を使用します。

注意: このチュートリアルでは、シカゴ市の公式ウェブサイト www.cityofchicago.org がオリジナルとして提供するデータを変更して使用するアプリケーションを構築します。シカゴ市は、このチュートリアルで提供されるデータの内容、正確性、適時性、または完全性について一切の表明を行いません。このサイトで提供されるデータは、随時変更される可能性があり、このチュートリアルで提供されるデータはユーザーの自己責任で使用するものとします。

### モデル目標 - 二項分類

顧客は 20% 以上のチップを払うでしょうか？

## Google Cloud プロジェクトをセットアップする

**Start Lab ボタンをクリックする前に**、これらの指示をお読みください。ラボには時間制限があり、一時停止することはできません。 このタイマーは **Start Lab** をクリックした時点で開始し、Google Cloud リソースを使用できる時間を示します。

このハンズオンラボでは、シミュレーションやデモ環境ではなく、実際のクラウド環境で作業を行うことができます。ラボの期間中、提供される新しい一時的な資格情報を使って Google Cloud　にサインインし、アクセスできます。

**必要なもの** このラボを実行するには、以下が必要です。

- 標準のインターネットブラウザへのアクセス（Chrome ブラウザ推奨）。
- ラボの所要時間。

**注意:** 個人の Google Cloud アカウントまたはプロジェクトをお持ちの場合は、このラボで使用しないようにしてください。

**注意:** Chrome OS デバイスを使用している場合は、シークレットウィンドウを開いてこのラボを実行してください。

**ラボを起動して Google Cloud Console にサインインするには** 1. **Start Lab** ボタンをクリックします。ラボの料金を支払う必要がある場合は、決済方法を選択するポップアップが開きます。左側のパネルに、このラボで使用する一時的な資格情報が表示されます。

![Taxi](https://github.com/tensorflow/docs-l10n/blob/master/site/ja/tfx/tutorials/tfx/images/airflow_workshop/taxi.jpg?raw=true)

1. ユーザー名をコピーして、**Open Google Console** をクリックします。ラボのリソースが起動し、**Sign in** ページが表示された別のタブが開きます。

![Data Components](https://github.com/tensorflow/docs-l10n/blob/master/site/ja/tfx/tutorials/tfx/images/airflow_workshop/examplegen1.png?raw=true)

***ヒント:*** タブは別のウィンドウに隣り合わせに開きます。

![DAG buttons](https://github.com/tensorflow/docs-l10n/blob/master/site/ja/tfx/tutorials/tfx/images/airflow_workshop/airflow_dag_buttons.png?raw=true)

1. **Sign in** ページで左のパネルからコピーしたユーザー名を貼り付けます。そしてパスワードをコピーして貼り付けます。

***重要:***- 左のパネルに表示される資格情報を使用する必要があります。Google Cloud Training の資格情報を使用してはいけません。自分の Google Cloud アカウントをお持ちの場合は、このラボでは使用してはいけません（料金が発生してしまいます）。

1. 以降のページをクリックして進みます。
2. 利用規約に同意します。

- リカバリオプションや二要素認証を追加してはいけません（これは一時アカウントであるため）。

- 無料トライアルに登録しないでください。

しばらくすると、このタブに Cloud Console が開きます。

**注意:** 左上の**ナビゲーションメニュー**をクリックすると、Google Cloud の製品とサービスのリストメニューが表示されます。

![Ready for production](https://github.com/tensorflow/docs-l10n/blob/master/site/ja/tfx/tutorials/tfx/images/airflow_workshop/step7.png?raw=true)

### Cloud Shell のアクティベーション

Cloud Shell は、開発ツールが読み込まれた仮想マシンです。永続的な 5GB のホームディレクトリを提供し、Google Cloud で実行します。Cloud Shell では、コマンドラインで Google Cloud リソースにアクセスできます。

Cloud Console の右上のツールバーにある **Activate Cloud Shell** ボタンをクリックします。

![Graph refresh button](https://github.com/tensorflow/docs-l10n/blob/master/site/ja/tfx/tutorials/tfx/images/airflow_workshop/graph_refresh_button.png?raw=true)

**Continue** をクリックします。

![Setup complete](https://github.com/tensorflow/docs-l10n/blob/master/site/ja/tfx/tutorials/tfx/images/airflow_workshop/step2.png?raw=true)

環境のプロビジョニングと接続にしばらくかかります。接続したら、認証が完了し、プロジェクトが *PROJECT_ID* に設定されます。以下に例を示します。

![Graph refresh button](https://github.com/tensorflow/docs-l10n/blob/master/site/ja/tfx/tutorials/tfx/images/airflow_workshop/step5.png?raw=true)

`gcloud` は Google Cloud のコマンドラインツールです。Cloud Shell に事前にインストールされており、タブ補間をサポートしています。

以下のコマンドで、アクティブなアカウント名をリスト表示できます。

```
gcloud auth list
```

（出力）

> ACTIVE: * ACCOUNT: student-01-xxxxxxxxxxxx@qwiklabs.net To set the active account, run: $ gcloud config set account `ACCOUNT`

次のコマンドを使って、プロジェクト ID をリスト表示できます: `gcloud config list project` （出力）

> [core] project = &lt;project_ID&gt;

（出力例）

> [core] project = qwiklabs-gcp-44776a13dea667a6

gcloud の完全なドキュメントについては、[gcloud コマンドラインツールの概要](https://cloud.google.com/sdk/gcloud)をご覧ください。

## Google Cloud サービスを有効にする

1. Cloud Shell で gcloud を使って、ラボで使用するサービスを有効にします: `gcloud services enable notebooks.googleapis.com`

## Vertex Notebook インスタンスをデプロイする

1. **ナビゲーションメニュー**をクリックし、まず **Vertex AI**、次に **Workbench** にアクセスします。

![Dive into data](https://github.com/tensorflow/docs-l10n/blob/master/site/ja/tfx/tutorials/tfx/images/airflow_workshop/step3.png?raw=true)

1. ノートブックのインスタンスページで、**New Notebook** をクリックします。

2. Customize instance メニューで **TensorFlow Enterprise** を選択し、バージョンに **TensorFlow Enterprise 2.x (with LTS)** &gt; **Without GPUs** を指定します。

![Dive into data](https://github.com/tensorflow/docs-l10n/blob/master/site/ja/tfx/tutorials/tfx/images/airflow_workshop/step3notebook.png?raw=true)

1. **New notebook instance** ダイアログの鉛筆アイコンをクリックし、**Edit** インスタンスプロパティをクリックします。

2. **Instance name** にインスタンスの名前を入力します。

3. **Region** に `us-east1` を選択し、**Zone** に選択した地域内のゾーンを指定します。

4. Machine configuration まで下にスクロールし、Machine type に **e2-standard-2** を選択します。

5. 残りのフィールドはデフォルトのままにし、**Create** をクリックします。

数分経つと、Vertex AI コンソールにインスタンス名と **Open Jupyterlab** が表示されます。

1. **Open JupyterLab** をクリックします。新しいタブに JupyterLab ウィンドウが開きます。

## 環境をセットアップする

### ラボのリポジトリをクローンする

次に、JupyterLab インスタンスに、`tfx` リポジトリをクローンします。1. JupyterLab で **Terminal** アイコンをクリックし、新しいターミナルを開きます。

{ql-infobox0}<strong>注意:</strong> Build Recommended が表示されたら、<code>Cancel</code> をクリックしてください。{/ql-infobox0}

1. `tfx` Github リポジトリをクローンするには、以下のコマンドを入力して **Enter** を押します。

```
git clone https://github.com/tensorflow/tfx.git
```

1. リポジトリのクローンを確認するには、`tfx` ディレクトリをダブルクリックして、その内容を確認してください。

![Transform](https://github.com/tensorflow/docs-l10n/blob/master/site/ja/tfx/tutorials/tfx/images/airflow_workshop/transform.png?raw=true)

### ラボの依存関係をインストールする

1. 以下を実行して `tfx/tfx/examples/airflow_workshop/taxi/setup/` フォルダに移動し、ラボの依存関係をインストールする `./setup_demo.sh` を実行します。

```bash
cd ~/tfx/tfx/examples/airflow_workshop/taxi/setup/
./setup_demo.sh
```

上記のコードは以下の内容を実行します。

- 必要なパッケージをインストールする
- ホームフォルダに `airflow` フォルダを作成する
- `dags` フォルダを `tfx/tfx/examples/airflow_workshop/taxi/setup/` フォルダから `~/airflow/` フォルダにコピーする
- csv ファイルを `tfx/tfx/examples/airflow_workshop/taxi/setup/data` から `~/airflow/data` にコピーする

![Analyzing model performance](https://github.com/tensorflow/docs-l10n/blob/master/site/ja/tfx/tutorials/tfx/images/airflow_workshop/step6.png?raw=true)

## Airflow サーバーの構成

### ブラウザで airflow サーバーにアクセスするためのファイアウォールルールを作成する

1. `https://console.cloud.google.com/networking/firewalls/list` に移動し、プロジェクト名が正しく選択されていることを確認します。
2. 上部の `CREATE FIREWALL RULE` オプションをクリックします。

![Transform](https://github.com/tensorflow/docs-l10n/blob/master/site/ja/tfx/tutorials/tfx/images/airflow_workshop/step5tboard.png?raw=true)

**Create a firewall ダイアログ**で、以下の指示に従います。

1. **Name** に `airflow-tfx` を指定します。
2. **Priority** に `1` を指定します。
3. **Targets** に `All instances in the network` を指定します。
4. **Source IPv4 ranges** に `0.0.0.0/0` を指定します。
5. **Protocols and ports** では、`tcp` をクリックして、`tcp` の横のボックスに `7000` を入力します。
6. `Create` をクリックします。

![Analyzing model performance](https://github.com/tensorflow/docs-l10n/blob/master/site/ja/tfx/tutorials/tfx/images/airflow_workshop/step6notebook.png?raw=true)

### シェルから airflow サーバーを実行する

Jupyter Lab Terminal ウィンドウでホームディレクトリに切り替え、`airflow users create` コマンドを実行して Airflow の管理者ユーザーを作成します。

```bash
cd
airflow users  create --role Admin --username admin --email admin --firstname admin --lastname admin --password admin
```

次に、`airflow webserver` と `airflow scheduler` コマンドを実行して、サーバーを実行します。ファイアウォールを通過できる `7000` ポートを選択します。

```bash
nohup airflow webserver -p 7000 &> webserver.out &
nohup airflow scheduler &> scheduler.out &
```

### 外部 IP を取得する

1. Cloud Shell で `gcloud` を使って、外部 IP を取得します。

```
gcloud compute instances list
```

![Training a Model](images/airflow_workshop/gcloud-instance-ip.png)

## DAG/パイプラインの実行

### ブラウザでの操作

ブラウザを開き、http://&lt;external_ip&gt;:7000 にアクセスします。

- ログインページで、`airflow users create` コマンドを実行したときに選択したユーザー名（`admin`）とパスワード（`admin`）を入力します。

![Training a Model](images/airflow_workshop/airflow-login.png)

Airflow は DAG を Python ソースリストから読み込みます。ファイルを 1 つずつ取得して実行し、そのファイルからすべての DAG オブジェクトを読み込みます。airflow ホームページに、DAG オブジェクトを定義するすべての `.py` ファイルがリスト表示されます。

このチュートリアルでは、Airflow は `~/airflow/dags/` フォルダの DAG オブジェクトをスキャンします。

`~/airflow/dags/taxi_pipeline.py` を開いて下にスクロールすると、DAG オブジェクトが作成され、`DAG` という変数に格納されるのが分かります。したがって、airflow ホームページに以下のようにパイプラインとして表示されます。

![dag-home-full.png](images/airflow_workshop/dag-home-full.png)

taxi をクリックすると、DAG のグリッドビューにリダイレクトされます。トップにある `Graph` オプションをクリックすると、DAG のグラフビューが表示されます。

![airflow-dag-graph.png](images/airflow_workshop/airflow-dag-graph.png)

### taxi パイプラインをトリガーする

ホームページには、DAG の操作に使用できるボタンが表示されています。

![dag-buttons.png](images/airflow_workshop/dag-buttons.png)

**actions** ヘッダーの **trigger** ボタンをクリックすると、パイプラインがトリガーされます。

taxi の **DAG** ページでは、右側のボタンを使って、パイプラインの実行中に、DAG のグラフビューを更新することができます。また、**Auto Refresh** を有効に擦れば、状態の変更時にグラフビューを自動更新するように Airflow に指示することができます。

![dag-button-refresh.png](images/airflow_workshop/dag-button-refresh.png)

また、ターミナルで [Airflow CLI](https://airflow.apache.org/cli.html) を使用しても、DAG を有効にしてトリガーすることができます。

```bash
# enable/disable
airflow pause <your DAG name>
airflow unpause <your DAG name>

# trigger
airflow trigger_dag <your DAG name>
```

#### パイプラインの完了を待つ

パイプラインをトリガーすると、パイプラインが実行する間、DAG ビューでパイプラインの進行状況を見ることができます。コンポーネントが実行するごとに、コンポーネントのアウトラインの色が変化し、状態を示します。コンポーネントの処理が完了すると、アウトラインが濃い緑色に変わって処理の完了を示します。

![dag-step7.png](images/airflow_workshop/dag-step7.png)

## コンポーネントを理解する

このパイプラインのコンポーネントを詳しく見てみましょう。パイプラインの各ステップで生成された出力を個別に見てみます。

1. JupyterLab で、`~/tfx/tfx/examples/airflow_workshop/taxi/notebooks/` に移動します。

2. **notebook.ipynb.** を開きます。![notebook-ipynb.png](images/airflow_workshop/notebook-ipynb.png)

3. ノートブックでラボを続け、画面の上にある **Run**（<img src="images/airflow_workshop/f1abc657d9d2845c.png" width="28.00" alt="run-button.png">）アイコンをクリックして、各セルを実行します。または、**SHIFT + ENTER** で、セル内のコードを実行することもできます。

説明を読んで、各セルで何が起きているかを理解してください。
