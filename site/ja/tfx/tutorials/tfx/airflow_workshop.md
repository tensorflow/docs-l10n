# TFX Airflow のチュートリアル

[](https://github.com/tensorflow/tfx)[](https://github.com/tensorflow/tfx)![Python](https://img.shields.io/pypi/pyversions/tfx.svg?style=plastic)[](https://github.com/tensorflow/tfx)
[![Python](https://badge.fury.io/py/tfx.svg)](https://badge.fury.io/py/tfx)[](https://badge.fury.io/py/tfx)

## はじめに

このチュートリアルでは、TensorFlow Extended（TFX）を紹介し、独自の機械学習パイプラインを作成する方法を見ていきます。本チュートリアルはローカルで実行され、TFX および TensorBoard と統合し、Jupyter Notebook の TFX とインタラクティブに動作します。

重要な用語：TFX パイプラインは、有向非巡回グラフ（DAG）で DAG と呼ばれることもあります。

典型的な機械学習開発プロセスに従い、データセットを調べてから最終的に完全に機能するパイプラインを作成します。その過程で、パイプラインをデバッグおよび更新し、パフォーマンスを測定する方法を学びます。

### 詳細情報

詳細については、[TFX ユーザーガイド](https://www.tensorflow.org/tfx/guide)を参照してください。

## ステップバイステップ

典型的な機械学習開発プロセスに従って、段階的に作業し、パイプラインを作成します。手順は次のとおりです。

1. 環境をセットアップする
2. 最初のパイプラインスケルトンを表示する
3. データを調査する
4. 特徴量エンジニアリング
5. [Training](#step_5_training)
6. モデルのパフォーマンスの分析
7. 実稼働の準備完了

## 前提条件

- Linux / MacOS
- Virtualenv
- Python 3.5+
- Git

### 必要なパッケージ

環境によっては、いくつかのパッケージをインストールする必要がある場合があります。

```bash
sudo apt-get install \
    build-essential libssl-dev libffi-dev \
    libxml2-dev libxslt1-dev zlib1g-dev \
    python3-pip git software-properties-common
```

Python 3.6 を実行している場合は、python3.6-dev をインストールする必要があります。

```bash
sudo apt-get install python3.6-dev
```

Python 3.7 を実行している場合は、python3.7-dev をインストールする必要があります。

```bash
sudo apt-get install python3.7-dev
```

さらに、システムの GCC バージョンが 7 未満の場合は、GCC を更新する必要があります。そうしないと、`airflow webserver`の実行時にエラーが発生します。現在のバージョンは次の方法で確認できます。

```bash
gcc --version
```

GCC を更新する必要がある場合は、次のコマンドを実行します。

```bash
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt update
sudo apt install gcc-7
sudo apt install g++-7
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 60 --slave /usr/bin/g++ g++ /usr/bin/g++-7
```

### MacOS 環境

Python 3 と git がまだインストールされていない場合は、[ Homebrew](https://brew.sh/) パッケージマネージャーを使用してインストールします。

```bash
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
export PATH="/usr/local/bin:/usr/local/sbin:$PATH"
brew update
brew install python
brew install git
```

構成によっては、MacOS で Airflow の実行時にスレッドのフォークに問題が発生することがあります。これらの問題を回避するには、`~/.bash_profile`を編集して、ファイルの最後に次の行を追加する必要があります。

```bash
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
```

## チュートリアル資料

このチュートリアルのコードは、次から入手できます。[https://github.com/tensorflow/tfx/tree/master/tfx/examples/airflow_workshop](https://github.com/tensorflow/tfx/tree/master/tfx/examples/airflow_workshop)

コードは作業のステップごとに編成されています。各ステップには、必要なコードとその説明があります。

チュートリアルファイルには、演習と演習の解決策（問題が発生した場合に備えて）の両方が含まれています。

#### 演習

- taxi_pipeline.py
- taxi_utils.py
- taxi DAG

#### 解決策

- taxi_pipeline_solution.py
- taxi_utils_solution.py
- taxi_solution DAG

## 学習内容

TFX を使用して機械学習パイプラインを作成する方法を学びます

- TFX パイプラインは、実稼働環境の機械学習アプリケーションをデプロイする場合に適しています。
- TFX パイプラインは、データセットが大きい場合に適しています。
- TFX パイプラインは、トレーニング/サービングの一貫性が重要な場合に適しています。
- TFX パイプラインは、推論のバージョン管理が重要な場合に適しています。
- Google は実稼働環境の機械学習に TFX パイプラインを使用しています。

典型的な機械学習の開発プロセスに従います

- データの取り込み、理解、クリーニング
- 特徴量エンジニアリング
- トレーニング
- モデルのパフォーマンスを分析する
- サイクルを繰り返す
- 実稼働の準備完了

### 各ステップのコードの追加

このチュートリアルでは、すべてのコードがファイルに含まれるように設計されていますが、ステップ 3 ～ 7 のすべてのコードはコメント アウトされ、インライン コメントでマークされています。インライン コメントは、コード行が適用されるステップを識別します。たとえば、ステップ 3 のコードは、コメント`# Step 3`でマークされています。

各ステップに追加するコードは、通常、コードの 3 つの領域に分類されます。

- インポート
- DAG 構成
- create_pipeline() 呼び出しから返されるリスト
- taxy_utils.py のサポート コード

チュートリアルを進める際に、現在取り組んでいるチュートリアル ステップのコード行のコメントを外します。そうすることにより、そのステップのコードが追加され、パイプラインが更新されます。その際、**コメントを外すコードを確認することを強くおすすめします**。

## シカゴのタクシー データセット

<!-- Image free for commercial use, does not require attribution:
https://pixabay.com/photos/new-york-cab-cabs-taxi-urban-city-2087998/ -->

![Taxi](https://github.com/tensorflow/docs-l10n/blob/master/site/ja/tfx/tutorials/tfx/images/airflow_workshop/taxi.jpg?raw=true) ![Chicago taxi](https://github.com/tensorflow/docs-l10n/blob/master/site/ja/tfx/tutorials/tfx/images/airflow_workshop/chicago.png?raw=true)

シカゴ市からリリースされた[タクシートリップデータセット](https://data.cityofchicago.org/Transportation/Taxi-Trips/wrvz-psew)のデータを使用します。

注意：このWeb サイトは、シカゴ市の公式 Web サイト www.cityofchicago.org で公開されたデータを変更して使用するアプリケーションを提供します。シカゴ市は、この Web サイトで提供されるデータの内容、正確性、適時性、または完全性について一切の表明を行いません。この Web サイトで提供されるデータは、随時変更される可能性があります。かかる Web サイトで提供されるデータはユーザーの自己責任で利用されるものとします。

データセットの詳細については、[Google BigQuery](https://cloud.google.com/bigquery/public-data/chicago-taxi) を[参照](https://cloud.google.com/bigquery/)してください。[BigQuery UI](https://bigquery.cloud.google.com/dataset/bigquery-public-data:chicago_taxi_trips) でデータセット全体をご確認ください。

### モデル目標 - 二項分類

顧客は 20% 以上のチップを払うでしょうか？

## ステップ 1: 環境をセットアップする

セットアップ スクリプト (`setup_demo.sh`) は、TFX と [Airflow](https://airflow.apache.org/) をインストールし、このチュートリアルで作業しやすいように Airflow を構成します。

シェルで次を行います。

```bash
cd
virtualenv -p python3 tfx-env
source ~/tfx-env/bin/activate

git clone https://github.com/tensorflow/tfx.git
cd ~/tfx
# These instructions are specific to the 0.21 release
git checkout -f origin/r0.21
cd ~/tfx/tfx/examples/airflow_workshop/setup
./setup_demo.sh
```

`setup_demo.sh`を確認して、何が行われているかを確認してください。

## ステップ 2: 最初のパイプラインスケルトンを表示する

### Hello World

シェルで次を行います

```bash
# Open a new terminal window, and in that window ...
source ~/tfx-env/bin/activate
airflow webserver -p 8080

# Open another new terminal window, and in that window ...
source ~/tfx-env/bin/activate
airflow scheduler

# Open yet another new terminal window, and in that window ...
# Assuming that you've cloned the TFX repo into ~/tfx
source ~/tfx-env/bin/activate
cd ~/tfx/tfx/examples/airflow_workshop/notebooks
jupyter notebook
```

このステップでは Jupyter Notebook を開始しました。後で、このフォルダで Notebook を実行します。

### ブラウザで次を行います

- ブラウザを開いて http://127.0.0.1:8080 にアクセスします。

#### トラブルシューティング

Web ブラウザでの Airflow コンソールの読み込みに問題がある場合、または `airflow webserver`の実行時にエラーが発生した場合は、ポート 8080 で別のアプリケーションが実行されている可能性があります。これは Airflow のデフォルト ポートですが、使用されていない他のユーザー ポートに変更できます。たとえば、ポート 7070 で Airflow を実行するには、次のように実行できます。

```bash
airflow webserver -p 7070
```

#### DAG ビュー ボタン

![DAG buttons](https://github.com/tensorflow/docs-l10n/blob/master/site/ja/tfx/tutorials/tfx/images/airflow_workshop/airflow_dag_buttons.png?raw=true)

- 左側のボタンを使用して DAG を*有効化*します
- 変更を加えるときは、右側のボタンを使用して DAG を*更新*します。
- 右側のボタンを使用して DAG を*トリガー*します
- タクシーをクリックすると、その日のグラフビューに移動します

![Graph refresh button](https://github.com/tensorflow/docs-l10n/blob/master/site/ja/tfx/tutorials/tfx/images/airflow_workshop/graph_refresh_button.png?raw=true)

#### Airflow CLI

[Airflow CLI](https://airflow.apache.org/cli.html) を使用して DAG を有効にしてトリガーすることもできます。

```bash
# enable/disable
airflow pause <your DAG name>
airflow unpause <your DAG name>

# trigger
airflow trigger_dag <your DAG name>
```

#### パイプラインが完了するのを待ちます

DAG ビューでパイプラインをトリガーした後、パイプラインが処理を完了するのを確認できます。各コンポーネントが実行されると、DAG グラフ内のコンポーネントのアウトラインの色が変わり、状態を示します。コンポーネントの処理が完了すると、アウトラインが濃い緑色に変わり、処理が完了したことを示します。

注意: 実行中のコンポーネントの更新された状態を表示するには、右側の*グラフ更新*ボタンを使用するか、ページを更新する必要があります。

この時点では、パイプラインには CsvExampleGen コンポーネントしかありません。アウトラインの色が濃い緑色になるまで待つ必要があります (~1 分)。

![Setup complete](https://github.com/tensorflow/docs-l10n/blob/master/site/ja/tfx/tutorials/tfx/images/airflow_workshop/step2.png?raw=true)

## ステップ 3: データを調査する

データ サイエンスや機械学習プロジェクトの最初のタスクは、データを理解してクリーンアップすることです。

- 各特徴のデータ型を理解する
- 異常と欠損値を探す
- 各特徴の分布を理解する

### コンポーネント

![Data Components](https://github.com/tensorflow/docs-l10n/blob/master/site/ja/tfx/tutorials/tfx/images/airflow_workshop/examplegen1.png?raw=true) ![Data Components](https://github.com/tensorflow/docs-l10n/blob/master/site/ja/tfx/tutorials/tfx/images/airflow_workshop/examplegen2.png?raw=true)

- [ExampleGen](https://www.tensorflow.org/tfx/guide/examplegen) は入力データセットを取り込み、分割します。
- [StatisticsGen](https://www.tensorflow.org/tfx/guide/statsgen) はデータセットの統計を計算します。
- [SchemaGen](https://www.tensorflow.org/tfx/guide/schemagen) は統計を調べ、データ スキーマを作成します。
- [ExampleValidator](https://www.tensorflow.org/tfx/guide/exampleval) はデータセット内の異常と欠損値を探します。

### エディタで次を行います

- ~/airflow/dags で、`taxi_pipeline.py`の`Step 3`とマークされた行のコメントを外します。
- 少し時間を取って、コメントを外したコードを確認してください。

### ブラウザで次を行います

- 左上隅にある [DAG] リンクをクリックして、Airflow の DAG リスト ページに戻ります。
- タクシー DAG の右側にある更新ボタンをクリックします。
    - 「DAG [taxi] is now fresh as a daisy」が表示されます。
- タクシーをトリガーします
- パイプラインが完了するのを待ちます
    - 全てダークグリーンになります
    - 右側の更新を使用するか、ページを更新します

![Dive into data](https://github.com/tensorflow/docs-l10n/blob/master/site/ja/tfx/tutorials/tfx/images/airflow_workshop/step3.png?raw=true)

### Jupyter に戻ります

前にブラウザ タブで Jupyter セッションを開くために`jupyter notebook`を実行しました。ブラウザでそのタブに戻ります。

- step3.ipynb を開きます
- notebook をフォローします

![Dive into data](https://github.com/tensorflow/docs-l10n/blob/master/site/ja/tfx/tutorials/tfx/images/airflow_workshop/step3notebook.png?raw=true)

### より高度な例

ここに示されている例は、初心者向けのものです。より高度な例については、[TensorFlow Data Validation Colab](https://www.tensorflow.org/tfx/tutorials/data_validation/chicago_taxi) を参照してください。

TFDV を使用してデータセットを調査および検証する方法の詳細については、[tensorflow.org の例を参照してください](https://www.tensorflow.org/tfx/data_validation)。

## ステップ 4: 特徴量エンジニアリング

特徴量エンジニアリングを使用すると、データの予測品質を向上させたり、次元を減らしたりすることができます。

- 特徴量クロス
- 語彙
- 埋め込み
- PCA
- カテゴリカル変数のエンコーディング

TFX を使用する利点の 1 つは、変換コードを 1 回記述すれば、結果として得られる変換はトレーニングとサービングの間で一貫性を保てることです。

### コンポーネント

![Transform](https://github.com/tensorflow/docs-l10n/blob/master/site/ja/tfx/tutorials/tfx/images/airflow_workshop/transform.png?raw=true)

- [Transform](https://www.tensorflow.org/tfx/guide/transform)は、データセットに対する特徴量エンジニアリングを実行します。

### エディタで次を行います

- ~/airflow/dags で、`taxi_pipeline.py`と`taxi_utils.py`の両方で`Step 4`とマークされた行のコメントを外します。
- 少し時間を取って、コメントを外したコードを確認してください。

### ブラウザで次を行います

- Airflow の DAG リスト ページに戻ります
- タクシー DAG の右側にある更新ボタンをクリックします。
    - 「DAG [taxi] is now fresh as a daisy」が表示されます。
- タクシーをトリガーします
- パイプラインが完了するのを待ちます
    - 全てダークグリーンになります。
    - 右側の更新を使用するか、ページを更新します。

![Feature Engineering](https://github.com/tensorflow/docs-l10n/blob/master/site/ja/tfx/tutorials/tfx/images/airflow_workshop/step4.png?raw=true)

### Jupyter に戻ります

ブラウザの Jupyter タブに戻ります。

- step4.ipynb を開きます
- notebook をフォローします

### より高度な例

ここに示されている例は、初心者向けのものです。より高度な例については、[TensorFlow Transform Colab](https://www.tensorflow.org/tfx/tutorials/transform/census) を参照してください。

## ステップ 5: トレーニング

TensorFlow モデルを、クリーンアップおよび変換されたデータでトレーニングします。

- ステップ 4 の変換が一貫して適用されるように含めます。
- 実稼働用に結果を SavedModel として保存します。
- TensorBoard を使用してトレーニング プロセスを視覚化して調査します。
- また、モデル パフォーマンスの分析のために EvalSavedModel を保存します。

### コンポーネント

- [Trainer](https://www.tensorflow.org/tfx/guide/trainer) は TensorFlow [Estimators](https://github.com/tensorflow/docs/blob/master/site/en/r1/guide/estimators.md) を使用してモデルをトレーニングします。

### エディタで:

- ~/airflow/dags で、`taxi_pipeline.py`と`taxi_utils.py`の両方で`Step 5`とマークされた行のコメントを外します。
- 少し時間を取って、コメントを外したコードを確認してください。

### ブラウザで次を行います

- Airflow の DAG リスト ページに戻ります。
- タクシー DAG の右側にある更新ボタンをクリックします。
    - 「DAG [taxi] is now fresh as a daisy」が表示されます。
- タクシーをトリガーします。
- パイプラインが完了するのを待ちます。
    - 全てダークグリーンになります。
    - 右側の更新を使用するか、ページを更新します。

![Training a Model](https://github.com/tensorflow/docs-l10n/blob/master/site/ja/tfx/tutorials/tfx/images/airflow_workshop/step5.png?raw=true)

### Jupyter に戻ります

ブラウザの Jupyter タブに戻ります。

- step5.ipynb を開きます。
- notebook をフォローします。

![Training a Model](https://github.com/tensorflow/docs-l10n/blob/master/site/ja/tfx/tutorials/tfx/images/airflow_workshop/step5tboard.png?raw=true)

### より高度な例

ここに示されている例は、初心者向けのものです。より高度な例については、[TensorFlow チュートリアル](https://www.tensorflow.org/tensorboard/r1/summaries) を参照してください。

## ステップ 6: モデルのパフォーマンスの分析

トップレベルの指標以上のことを理解します。

- ユーザーは自分のクエリに対してのみモデルのパフォーマンスを体験します。
- データ スライスのパフォーマンスの低下は、トップレベルの指標には示されない可能性があります。
- モデルの公平性は重要です。
- 多くの場合、ユーザーまたはデータの主要なサブセットは非常に重要です。これは小さい場合があります。
    - 重要かつ異常な状態でのパフォーマンス
    - インフルエンサーなどの主要オーディエンスに対するパフォーマンス
- 実稼働中のモデルを置き換える場合は、まず新しいモデルの方が優れていることを確認してください。
- Evaluator コンポーネントは、モデルが OK かどうかを Pusher コンポーネントに通知します。

### コンポーネント

- [Evaluator](https://www.tensorflow.org/tfx/guide/evaluator) はトレーニング結果の詳細な分析を実行し、モデルが実稼働環境にプッシュするのに「十分」であることを確認します。

### エディタで次を行います

- ~/airflow/dags で、`taxi_pipeline.py`の両方の`Step 6` とマークされた行のコメントを外します。
- 少し時間を取って、コメントを外したコードを確認してください。

### ブラウザで次を行います

- Airflow の DAG リスト ページに戻ります。
- タクシー DAG の右側にある更新ボタンをクリックします。
    - 「DAG [taxi] is now fresh as a daisy」が表示されます。
- タクシーをトリガーします。
- パイプラインが完了するのを待ちます。
    - 全てダークグリーンになります。
    - 右側の更新を使用するか、ページを更新します。

![Analyzing model performance](https://github.com/tensorflow/docs-l10n/blob/master/site/ja/tfx/tutorials/tfx/images/airflow_workshop/step6.png?raw=true)

### Jupyter に戻ります：

ブラウザの Jupyter タブに戻ります。

- step6.ipynb を開きます。
- notebook をフォローします。

![Analyzing model performance](https://github.com/tensorflow/docs-l10n/blob/master/site/ja/tfx/tutorials/tfx/images/airflow_workshop/step6notebook.png?raw=true)

### より高度な例

ここに示されている例は、初心者向けのものです。より高度な例については、[TFMA シカゴ タクシー チュートリアル](https://www.tensorflow.org/tfx/tutorials/model_analysis/chicago_taxi)を参照してください。

## ステップ 7: 実稼働の準備完了

新しいモデルの準備ができている場合は、準備を完了します。

- Pusher は SavedModels を既知の場所にデプロイします

デプロイメントターゲットは、既知の場所から新しいモデルを受け取ります。

- TensorFlow Serving
- TensorFlow Lite
- TensorFlow JS
- TensorFlow Hub

### コンポーネント

- [Pusher](https://www.tensorflow.org/tfx/guide/pusher) モデルをサービング インフラストラクチャにデプロイします。

### エディタで次を行います

- ~/airflow/dags で、両方の`taxi_pipeline.py`の`Step 7`とマークされた行のコメントを外します。
- 少し時間を取って、コメントを外したコードを確認してください。

### ブラウザで次を行います

- Airflow の DAG リスト ページに戻ります。
- タクシー DAG の右側にある更新ボタンをクリックします。
    - 「DAG [taxi] is now fresh as a daisy」が表示されます。
- タクシーをトリガーします。
- パイプラインが完了するのを待ちます。
    - 全てダークグリーンになります。
    - 右側の更新を使用するか、ページを更新します。

![Ready for production](https://github.com/tensorflow/docs-l10n/blob/master/site/ja/tfx/tutorials/tfx/images/airflow_workshop/step7.png?raw=true)

## 次のステップ

以上で、モデルのトレーニングと検証が完了し、`~/airflow/saved_models/taxi`ディレクトリに`SavedModel`ファイルがエクスポートされました。これで、モデルの実稼働の準備が完了ました。 次のような TensorFlow デプロイメント ターゲットのいずれかにモデルをデプロイできるようになりました。

- [TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving) はサーバーまたはサーバー ファームでモデルをサービングし、REST および/または gRPC 推論リクエストを処理します。
- [TensorFlow Lite](https://www.tensorflow.org/lite) は、モデルを Android または iOS のネイティブ モバイル アプリケーション、または Raspberry Pi、IoT、またはマイクロコントローラー アプリケーションに含めます。
- [TensorFlow.js](https://www.tensorflow.org/js) は、モデルを Web ブラウザまたは Node.JS アプリケーションで実行します。
