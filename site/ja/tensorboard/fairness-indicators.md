# フェアネスインジケータダッシュボード［ベータ］でモデルを評価する

![Fairness Indicators](./images/fairness-indicators.png)

TensorBoard のフェアネスインジケータでは、*バイナリ*と*マルチクラス*分類器の一般的に識別されるフェアネスメトリックを簡単に計算できます。プラグインを使用すると、実行の公平性評価を視覚化し、グループ間のパフォーマンスを簡単に比較できます。

特に、TensorBoard のフェアネスインジケータを使うと、モデルのパフォーマンスを定義されたユーザーのグループ間に分けて評価し視覚化することができます。複数のしきい値における信頼区間と評価により、結果に自信を持つことができます。

公平性に関する事項を評価する多くの既存のツールは、大規模なデータセットやモデルではうまく機能しません。Google では、10 億ユーザーシステムで機能できるツールの存在が重要です。フェアネスインジケータでは、ユースケースの規模にかかわらず、TensorBoard 環境または [Colab](https://github.com/tensorflow/fairness-indicators) で評価することができます。

## 要件

TensorBoard のフェアネスインジケータをインストールするには、次を実行します。

```
python3 -m virtualenv ~/tensorboard_demo
source ~/tensorboard_demo/bin/activate
pip install --upgrade pip
pip install fairness_indicators
pip install tensorboard-plugin-fairness-indicators
```

## デモ

TensorBoard でフェアネスインジケータをテストするには、サンプルの TensorFlow Model Analysis 評価結果（eval_config.json、メトリック、プロットファイル）と Google Cloud Platform の `demo.py` ユーティリティを、次のコマンドを使って[ここ](https://console.cloud.google.com/storage/browser/tensorboard_plugin_fairness_indicators/)からダウンロードできます。

```
pip install gsutil
gsutil cp -r gs://tensorboard_plugin_fairness_indicators/ .
```

ダウンロードしたファイルを含むディレクトリに移動します。

```
cd tensorboard_plugin_fairness_indicators
```

この評価データは [Civil Comments データセット](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification)に基づくもので、Tensorflow Model Analysis の [model_eval_lib](https://github.com/tensorflow/model-analysis/blob/master/tensorflow_model_analysis/api/model_eval_lib.py) ライブラリを使って計算されています。また、参照用のサンプル TensorBoard 要約データファイルも含まれます。

`demo.py` ユーティリティは TensorBoard 要約データファイルを書き込み、TensorBoard はそれを読み取ってフェアネスインジケータダッシュボードが表示されます（要約データファイルの詳細については、[TensorBoard チュートリアル](https://github.com/tensorflow/tensorboard/blob/master/README.md)をご覧ください）。

次は、`demo.py` ユーティリティで使用できるフラグです。

- `--logdir`: TensorBoard が要約を書き込むディレクトリ
- `--eval_result_output_dir`: TFMA が評価した評価結果（前のステップでダウンロードしたファイル）を含むディレクトリ

`demo.py` ユーティリティを実行して。ログディレクトリに要約の結果を書き込みます。

`python demo.py --logdir=. --eval_result_output_dir=.`

TensorBoard を実行します。

注意: このデモでは、ダウンロードしたすべてのファイルを含むディレクトリから TensorBoard を実行してください。

`tensorboard --logdir=.`

上記はローカルのインスタンスを起動するコードです。ローカルインスタンスが起動したら、ターミナルにリンクが表示されます。ブラウザでリンクを開くと、フェアネスインジケータダッシュボードが表示されます。

### デモ Colab

[Fairness_Indicators_TensorBoard_Plugin_Example_Colab.ipynb](https://github.com/tensorflow/fairness-indicators/blob/master/fairness_indicators/documentation/examples/Fairness_Indicators_TensorBoard_Plugin_Example_Colab.ipynb) には、モデルをトレーニングして評価し、TensorBoard でフェアネス評価結果を視覚化するエンドツーエンドのデモが含まれます。

## 使い方

自分のデータと評価でフェアネスインジケータを使用するには、次のように行います。

1. 新しいモデルをトレーニングし、[model_eval_lib](https://github.com/tensorflow/model-analysis/blob/master/tensorflow_model_analysis/api/model_eval_lib.py) の`tensorflow_model_analysis.run_model_analysis` または `tensorflow_model_analysis.ExtractEvaluateAndWriteResult` API を使用して評価します。これを行うためのコードスニペットは、[こちら](https://github.com/tensorflow/fairness-indicators)の Fairness Indicators Colab をご覧ください。

2. `tensorboard_plugin_fairness_indicators.summary_v2` API を使ってフェアネスインジケータ要約を書き込みます。

    ```
    writer = tf.summary.create_file_writer(<logdir>)
    with writer.as_default():
        summary_v2.FairnessIndicators(<eval_result_dir>, step=1)
    writer.close()
    ```

3. TensorBoard を実行します。

    - `tensorboard --logdir=<logdir>`
    - ダッシュボードの左側にあるドロップダウンを使って、結果を視覚化する新しい評価の実行を選択します。
