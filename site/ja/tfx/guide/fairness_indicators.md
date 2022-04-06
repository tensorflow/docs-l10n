# Fairness Indicators

Fairness Indicators は、より広範な Tensorflow ツールキットと連携して、チームが公平性に関する懸念のあるモデルを評価および改善する際の支援を提供するように設計されています。このツールは現在、多くの Google 製品で活発に使用されており、独自のユースケースでお試しいただけるように、BETA バージョンでも提供されています。

![Fairness Indicator ダッシュボード](images/fairnessIndicators.png)

## Fairness Indicators とは

Fairness Indicators は二進数とマルチクラスの分類子において一般的に特定される公平性メトリクスを簡単に計算できるようにするためのライブラリです。公平性に関する懸念を評価する多数の既存のツールは大規模なデータセットやモデルであまりうまく動作しません。Google では、十億人ユーザークラスのシステムで動作するツールを重視しており、Fairness Indicators ではあらゆる規模のユースケースを評価することができます。

具体的には、Fairness Indicators には次の機能が含まれています。

- データセットの分散を評価する
- 定義されたグループのユーザー間でスライスされたモデルのパフォーマンスを評価する
    - 複数のしきい値での信頼区間と評価により、結果に確証を得られます。
- 個別のスライスを精査して、根源と改善の機会を探る

こちらの[ケーススタディ](https://developers.google.com/machine-learning/practica/fairness-indicators)には[動画](https://www.youtube.com/watch?v=pHT-ImFXPQo)とプログラミング演習も含まれており、ユーザー独自の製品で Fairness Indicators を使用して長期間にわたる公平性の懸念を評価する方法を実演しています。

[](http://www.youtube.com/watch?v=pHT-ImFXPQo)

pip パッケージダウンロードには次の項目が含まれています。

- **[Tensorflow Data Validation（TFDV）](https://www.tensorflow.org/tfx/data_validation/get_started)**
- **[Tensorflow Model Analysis（TFMA）](https://www.tensorflow.org/tfx/model_analysis/get_started)**
    - **Fairness Indicators**
- **[What-If Tool（WIT）](https://www.tensorflow.org/tensorboard/what_if_tool)**

## TensorFlow Model で Fairness Indicators を使用する

### データ

TFMA で Fairness Indicators を使用するには、評価データセットのスライスに使用する特徴量にラベルが付いていることを確認してください。公平性の懸念に使用するスライス特徴量がない場合は、その特徴量のある評価セットを探すか、結果の格差をハイライトする可能性のある特徴量セット内のプロキシ特徴量を検討してください。その他のガイダンスについては、[こちら](https://tensorflow.org/responsible_ai/fairness_indicators/guide/guidance)をご覧ください。

### モデル

Tensorflow Estimator クラスを使用して、モデルを構築することができます。TFMA では近日 Keras モデルがサポートされる予定です。Keras モデルで TFMA を実行する場合は、以下の「モデルに依存しない TFMA」セクションをご覧ください。

Estimator のトレーニングが完了したら、評価の目的で、保存したモデルをエクスポートする必要があります。詳細については、[TFMA ガイド](/tfx/model_analysis/get_started)をご覧ください。

### スライスを構成する

次の、評価に使用するスライスを定義します。

```python
slice_spec = [
  tfma.slicer.SingleSliceSpec(columns=[‘fur color’])
]
```

交差スライスを評価する場合（毛の色と高さの両方など）、次のように設定できます。

```python
slice_spec = [
  tfma.slicer.SingleSliceSpec(columns=[‘fur_color’, ‘height’])
]`
```

### 公平性メトリクスを計算する

Fairness Indicators コールバックを `metrics_callback` リストに追加します。コールバックでは、モデルが評価される際のしきい値のリストを定義できます。

```python
from tensorflow_model_analysis.addons.fairness.post_export_metrics import fairness_indicators

# Build the fairness metrics. Besides the thresholds, you also can config the example_weight_key, labels_key here. For more details, please check the api.
metrics_callbacks = \
    [tfma.post_export_metrics.fairness_indicators(thresholds=[0.1, 0.3,
     0.5, 0.7, 0.9])]

eval_shared_model = tfma.default_eval_shared_model(
    eval_saved_model_path=tfma_export_dir,
    add_metrics_callbacks=metrics_callbacks)
```

この構成を実行する前に、信頼区間の計算を有効にするかどうかを決定してください。信頼区間は Poisson ブートストラップ法を用いて計算され、20 サンプルごとに再計算が必要です。

```python
compute_confidence_intervals = True
```

TFMA 評価パイプラインを実行します。

```python
validate_dataset = tf.data.TFRecordDataset(filenames=[validate_tf_file])

# Run the fairness evaluation.
with beam.Pipeline() as pipeline:
  _ = (
      pipeline
      | beam.Create([v.numpy() for v in validate_dataset])
      | 'ExtractEvaluateAndWriteResults' >>
       tfma.ExtractEvaluateAndWriteResults(
                 eval_shared_model=eval_shared_model,
                 slice_spec=slice_spec,
                 compute_confidence_intervals=compute_confidence_intervals,
                 output_path=tfma_eval_result_path)
  )
eval_result = tfma.load_eval_result(output_path=tfma_eval_result_path)
```

### Fairness Indicators をレンダリングする

```python
from tensorflow_model_analysis.addons.fairness.view import widget_view

widget_view.render_fairness_indicator(eval_result=eval_result)
```

![Fairness Indicators](images/fairnessIndicators.png)

Fairness Indicators を使用する際のヒント:

- 左側のチェックボックスをオンにして、**表示するメトリクスを選択**します。各メトリクスのグラフは順に、ウィジェットに表示されます。
- ドロップダウンセレクターを使用して、グラフの最初の棒で示される**ベースラインスライスを変更**します。デルタはこのベースライン値を使って計算されます。
- ドロップダウンセレクターを使用して、**しきい値を選択**します。同一のフラフ上に複数のしきい値を表示することができます。選択されたしきい値は強調して示され、そのしきい値をクリックすると選択が解除されます。
- スライスのメトリクスを確認するには、**棒にマウスポインターを合わせます**。
- 「Diff w. baseline」列を使用して、**ベースラインとの格差を識別**します。現在のスライスとベースラインとの差がパーセント率で示されます。
- **スライスのデータポイントを詳しく探る**には、[What-If Tool](https://pair-code.github.io/what-if-tool/) を使用します。[こちら](https://github.com/tensorflow/fairness-indicators/)で例をご覧ください。

#### 複数のモデルの Fairness Indicators をレンダリングする

Fairness Indicators を使ってモデルを比較することもできます。1 つの eval_result を渡す代わりに、2 つのモデル名を eval_result オブジェクトにマッピングするディクショナリである multi_eval_results オブジェクトを渡します。

```python
from tensorflow_model_analysis.addons.fairness.view import widget_view

eval_result1 = tfma.load_eval_result(...)
eval_result2 = tfma.load_eval_result(...)
multi_eval_results = {"MyFirstModel": eval_result1, "MySecondModel": eval_result2}

widget_view.render_fairness_indicator(multi_eval_results=multi_eval_results)
```

![Fairness Indicators - モデルの比較](images/fairnessIndicatorsCompare.png)

モデルの比較としきい値の比較を合わせて使用することができます。たとえば、2 つのモデルを 2 セットのしきい値で比較し、公平性メトリクスに最適な組み合わせを見つけ出すことができます。

## TensorFlow 以外のモデルで Fairness Indicators を使用する

異なるモデルとワークグローを使用するクライアントをサポートするために、評価対象のモデルに依存しない評価ライブラリを開発しました。

機械学習システムを評価したい方、特に TensorFlow 以外のテクノロジーに基づくモデルを使用している方ならだれでもこれを使用できます。Apache Beam Python SDK を使用してスタンドアロンの TFMA 評価バイナリを作成し、それを実行してモデルを分析することができます。

### データ

このステップでは、評価を実行するためのデータセットを作成します。データセットは [tf.Example](https://www.tensorflow.org/tutorials/load_data/tfrecord) proto 形式であり、ラベル、予測、およびスライスの基準となるその他の特徴量が必要です。

```python
tf.Example {
    features {
        feature {
          key: "fur_color" value { bytes_list { value: "gray" } }
        }
        feature {
          key: "height" value { bytes_list { value: "tall" } }
        }
        feature {
          key: "prediction" value { float_list { value: 0.9 } }
        }
        feature {
          key: "label" value { float_list { value: 1.0 } }
        }
    }
}
```

### モデル

モデルを指定する代わりに、TFMA がメトリクスを計算するために必要とするデータを分析して提供する、モデルに依存しない評価構成と Extractor を作成します。[ModelAgnosticConfig](https://github.com/tensorflow/model-analysis/blob/master/tensorflow_model_analysis/model_agnostic_eval/model_agnostic_predict.py) 仕様は、入力 Example から使用される特徴量、予測、およびラベルを定義します。

このためには、ラベルキーと予測キー、および特徴量のデータ型を示す値など、すべての特徴量を表現するキーを伴う特徴量マップを作成します。

```python
feature_map[label_key] = tf.FixedLenFeature([], tf.float32, default_value=[0])
```

ラベルキー、予測キー、および特徴量マップを使用して、モデルに依存しない構成を作成します。

```python
model_agnostic_config = model_agnostic_predict.ModelAgnosticConfig(
    label_keys=list(ground_truth_labels),
    prediction_keys=list(predition_labels),
    feature_spec=feature_map)
```

### モデルに依存しない Extractor をセットアップする

[Extractor](https://github.com/tensorflow/model-analysis/blob/master/tensorflow_model_analysis/model_agnostic_eval/model_agnostic_extractor.py) は、モデルに依存しない構成を使用して入力から特徴量、ラベル、および予測を抽出するために使用します。さらにデータをスライスする場合は、スライスする列に関する情報を含む[スライスキー仕様](https://github.com/tensorflow/model-analysis/tree/master/tensorflow_model_analysis/slicer)を定義する必要もあります。

```python
model_agnostic_extractors = [
    model_agnostic_extractor.ModelAgnosticExtractor(
        model_agnostic_config=model_agnostic_config, desired_batch_size=3),
    slice_key_extractor.SliceKeyExtractor([
        slicer.SingleSliceSpec(),
        slicer.SingleSliceSpec(columns=[‘height’]),
    ])
]
```

### 公平性メトリクスを計算する

[EvalSharedModel](https://www.tensorflow.org/tfx/model_analysis/api_docs/python/tfma/types/EvalSharedModel) の一環として、モデルの評価に使用するすべてのメトリクスを指定できます。メトリクスは、[post_export_metrics](https://github.com/tensorflow/model-analysis/blob/master/tensorflow_model_analysis/post_export_metrics/post_export_metrics.py) や [fairness_indicators](https://github.com/tensorflow/model-analysis/blob/master/tensorflow_model_analysis/addons/fairness/post_export_metrics/fairness_indicators.py) で定義されているコールバックのように、メトリクスのコールバックの形態で指定されます。

```python
metrics_callbacks.append(
    post_export_metrics.fairness_indicators(
        thresholds=[0.5, 0.9],
        target_prediction_keys=[prediction_key],
        labels_key=label_key))
```

また、評価の実行目的で TensorFlow グラフを作成するために使用される `construct_fn` も取り込みます。

```python
eval_shared_model = types.EvalSharedModel(
    add_metrics_callbacks=metrics_callbacks,
    construct_fn=model_agnostic_evaluate_graph.make_construct_fn(
        add_metrics_callbacks=metrics_callbacks,
        fpl_feed_config=model_agnostic_extractor
        .ModelAgnosticGetFPLFeedConfig(model_agnostic_config)))
```

すべての準備が完了したら、`ExtractEvaluate` または `ExtractEvaluateAndWriteResults` 関数のいずれかを使用し（[model_eval_lib](https://github.com/tensorflow/model-analysis/blob/master/tensorflow_model_analysis/api/model_eval_lib.py) が提供する関数）、モデルを評価します。

```python
_ = (
    examples |
    'ExtractEvaluateAndWriteResults' >>
        model_eval_lib.ExtractEvaluateAndWriteResults(
        eval_shared_model=eval_shared_model,
        output_path=output_path,
        extractors=model_agnostic_extractors))

eval_result = tensorflow_model_analysis.load_eval_result(output_path=tfma_eval_result_path)
```

最後に、上記の「Fairness Indicators をレンダリングする」セクションの指示に従って、Fairness Indicators をレンダリングします。

## その他の例

[Fairness Indicators のサンプルディレクトリ](https://github.com/tensorflow/fairness-indicators/blob/master/g3doc/tutorials/)には、複数の例が掲載されています。

- [Fairness_Indicators_Example_Colab.ipynb](https://github.com/tensorflow/fairness-indicators/blob/master/g3doc/tutorials/Fairness_Indicators_Example_Colab.ipynb) は、[TensorFlow Model Analysis](https://www.tensorflow.org/tfx/guide/tfma) における Fairness Indicators の概要を示し、実際のデータセットでの使用法を説明しています。このノートブックでは、[TensorFlow Data Validation](https://www.tensorflow.org/tfx/data_validation/get_started) と [What-If Tool](https://pair-code.github.io/what-if-tool/) についても説明しています。いずれも Fairness Indicators に同梱されている TensorFlow モデルの分析ツールです。
- [Fairness_Indicators_on_TF_Hub.ipynb](https://github.com/tensorflow/fairness-indicators/blob/master/g3doc/tutorials/Fairness_Indicators_on_TF_Hub_Text_Embeddings.ipynb) では、Fairness Indicators を使用して異なる[テキスト埋め込み](https://en.wikipedia.org/wiki/Word_embedding)でトレーニングされたモデルを比較する方法が実演されています。このノートブックでは、[TensorFlow Hub](https://www.tensorflow.org/hub) に掲載されているテキスト埋め込みが使用されています。TensorFlow Hub は、モデルコンポーネントをパブリッシュ、発掘、および再利用できる TensorFlow のライブラリです。
- [Fairness_Indicators_TensorBoard_Plugin_Example_Colab.ipynb](https://github.com/tensorflow/fairness-indicators/blob/master/g3doc/tutorials/Fairness_Indicators_TensorBoard_Plugin_Example_Colab.ipynb) では、TensorBoard で Fairness Indicators を視覚化する方法が実演されています。
