# What-If Tool ダッシュボードを使ったモデルの理解

![What-If Tool](./images/what_if_tool.png)

What-If Tool（WIT）は、ブラックボックス分類と回帰 ML モデルの理解を広げるための使いやすいインターフェースを提供しています。このプラグインを使用すると、大規模なサンプルセットで推論を実行し、さまざまな方法で即時に結果を視覚化することができます。また、サンプルを手動またはプログラムで編集し、モデルを再実行して変更の結果を確認することも可能です。データセットのサブセットにおけるモデルのパフォーマンスと公平性を調べるためのツールも含まれています。

このツールの目的は、ビジュアルインターフェースを通じて、トレーニング済みの ML モデルを調査するための単純で直感的かつ強力な方法を、コードを全く使用せずに提供することにあります。

このツールには TensorBoard から、または Jupyter や Colab ノートブックから直接アクセスすることができます。ノートブックモードにおける WIT の使用に関するより詳しい説明、実演、ウォークスルー、および情報については、[What-If Tool のウェブサイト](https://pair-code.github.io/what-if-tool)をご覧ください。

## 要件

TensorBoard で WIT を使用するには、次の 2 つの項目が必要です。

- 調べようとしているモデルは、classify、regress、または predict API によって、[TensorFlow Serving](https://github.com/tensorflow/serving) を使って配布される必要があります。
- モデルが推論するデータセットは、TensorBoard ウェブサーバーがアクセスできる TFRecord ファイルにある必要があります。

## 使い方

TensorBoard で What-If Tool ダッシュボードを開くと、モデルサーバーのホストとポート、配信されるモデルの名前、モデルの種類、および読み込まれる TFRecords ファイルへのパスを指定できるセットアップ画面が表示されます。情報を入力し、「Accept」をクリックすると、WIT はデータセットを読み込んでモデルで推論を実行し、結果を表示します。

WIT の各機能と、モデルの理解や公平性の調査にどのように役立つかに関する詳細は、[What-If Tool ウェブサイト](https://pair-code.github.io/what-if-tool)のウォークスルーをご覧ください。

## 実演モデルとデータセット

TensorBoard で事前トレーニング済みのモデルを使用して WIT を試す場合は、事前トレーニング済みのモデルとデータセットを https://storage.googleapis.com/what-if-tool-resources/uci-census-demo/uci-census-demo.zip からダウンロードして解凍してください。モデルは二項分類モデルで、[UCI Census](https://archive.ics.uci.edu/ml/datasets/census+income) データセットを使用して、ある人物に年間 5 万ドル超の収入があるかどうかを予測します。このデータセットと予測タスクは機械学習のモデリングと公平性調査によく使用されています。

環境変数 MODEL_PATH をマシン上の結果のモデルディレクトリの場所に設定します。

次に、[公式ドキュメント](https://www.tensorflow.org/serving/docker)に従って、docker と TensorFlow Serving をインストールします。

docker を使用して、`docker run -p 8500:8500 --mount type=bind,source=${MODEL_PATH},target=/models/uci_income -e MODEL_NAME=uci_income -t tensorflow/serving` でモデルを配信します。docker のセットアップによっては、コマンドを `sudo` で実行する必要があります。

ここで、TensorBoard を起動し、ダッシュボードのドロップダウンを使って What-If Tool に移動します。

セットアップ画面で、推論アドレスを「localhost:8500」、モデル名を「uci_income」、例へのパスをダウンロードした `adult.tfrecord` ファイルのフルパスに設定し、「Accept」をクリックします。

![Setup screen for demo](./images/what_if_tool_demo_setup.png)

この実演で、What-If Tool を使って行える項目には次の内容があります。

- 単一のデータポイントを編集して、その結果の推論への変更を確認する
- データセットの個別の特徴量とモデルの推論結果の関係を部分的な依存関係プロットを通じて調べる
- データセットをサブセットにスライスして、スライス間のパフォーマンスを比較する

ツールの機能に関する詳細は、[What-If Tool のウォークスルー](https://pair-code.github.io/what-if-tool/walkthrough.html)をご覧ください。

このモデルが予測しようとしているデータセットのグラウンドトゥルースの特徴量は「Target」と名付けられているため、「Performance & Fairness」タブを使用する際、「Target」はグラウンドトゥルース特徴量ドロップダウンに指定するものとなります。
