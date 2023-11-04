# Keras: TensorFlow の高レベル API

Keras は TensorFlow プラットフォームの高レベル API です。機械学習（ML）問題を解決するためのアプローチしやすく生産性の高いインターフェースを、最新のディープラーニングに焦点を当てて提供しています。Keras は、データ処理からハイパーパラメータのチューニング、デプロイまで、機械学習ワークフローの各ステップに対応しています。高速実験を可能にすることに焦点を当てて開発された言語です。

Keras を使用すると、TensorFlow の拡張性とクロスプラットフォーム機能に完全にアクセスできます。Keras はTPU Pod や大規模な GPU クラスタで実行でき、Keras モデルをブラウザやモバイルデバイスで実行するためにエクスポートすることができます。また、ウェブ API を使って Keras モデルを配信することも可能です。

Keras は、以下の目標を達成することで、認知的負荷を押さえられるように設計されています。

- 単純で一貫性のあるインターフェースを提供する。
- 一般的なユースケースに必要なアクションの数を最小限に抑える。
- 明確で対応可能なエラーメッセージを提供する。
- 複雑さの段階的な開示の原理に従う。取りかかりやすく、作業を進めながら学習することで、高度なワークフローを完成させられます。
- 明白で読み取りやすいコードを書けるようにする。

## Keras の対象者

簡単に言えば、すべての TensorFlow ユーザーにデフォルトで Keras API を使用することをお勧めします。エンジニアや研究者、ML 専門家など、役職に関係なく Keras を使い始めるべきと言えます。

低レベルの [TensorFlow Core API](https://www.tensorflow.org/guide/core) が必要となるユースケースはいくつかありますが（TensorFlow 上にツールを構築する、独自の高性能プラットフォームを開発するなど）、ユースケースが [Core API アプリケーション](https://www.tensorflow.org/guide/core#core_api_applications)に該当しない場合は、Keras を優先することをお勧めします。

## Keras API コンポーネント

Keras の基本データ構造は[レイヤー](https://keras.io/api/layers/)と[モデル](https://keras.io/api/models/)です。レイヤーは単純な入力/出力変換で、モデルはレイヤーの有向非巡回グラフ（DAG）です。

### レイヤー

`tf.keras.layers.Layer` クラスは、Keras の基本的な抽象です。`Layer` は状態（重み）といくつかの計算（code2}tf.keras.layers.Layer.call メソッド内に定義）をカプセル化します。

レイヤーが作る重みはトレーニング可能である場合とトレーニング不可能である場合があります。レイヤーは繰り返し構成可能です。レイヤーインスタンスを別のレイヤーの属性として割り当てる場合、外側のレイヤーは内側のレイヤーが作成する重みを追跡し始めます。

また、レイヤーを使用して、正規化やテキストのベクトル化といったデータ前処理タスクを処理することもできます。前処理レイヤーはトレーニング中かその後にモデルに直接含めることができるため、モデルの移植が可能です。

### モデル

モデルは、レイヤーをひとまとめにしてデータでトレーニングできるオブジェクトです。

最も単純なモデルのタイプは [`Sequential` モデル](https://www.tensorflow.org/guide/keras/sequential_model)で、レイヤーの直線的なスタックです。より複雑なアーキテクチャでは、レイヤーの任意のグラフを構築する [Keras functional API](https://www.tensorflow.org/guide/keras/functional_api) を使用するか、[サブクラス化によってモデルを新規作成](https://www.tensorflow.org/guide/keras/making_new_layers_and_models_via_subclassing)できます。

`tf.keras.Model` クラスには、トレーニングと評価メソッドが組み込まれています。

- `tf.keras.Model.fit`: 一定したエポック数でモデルをトレーニングします。
- `tf.keras.Model.predict`: 入力サンプルに対して出力予測を生成します。
- `tf.keras.Model.evaluate`: モデルの損失と指標の値を返します。`tf.keras.Model.compile` メソッドで構成されます。

これらのメソッドによって、以下の組み込みトレーニング機能にアクセスできます。

- [コールバック](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks)。組み込みのコールバックを使用して、早期停止、モデルへのチェックポイントの設定、および[TensorBoard](https://www.tensorflow.org/tensorboard) での監視を行えます。また、[カスタムコールバックの実装](https://www.tensorflow.org/guide/keras/writing_your_own_callbacks)も可能です。
- [分散トレーニング](https://www.tensorflow.org/guide/keras/distributed_training)。トレーニングを複数の GPU、TPU、またはデバイスに簡単に拡張できます。
- ステップ融合。`tf.keras.Model.compile` の `steps_per_execution` 引数を使用して、単一の `tf.function` 呼び出しで複数のバッチを処理できます。TPU でのデバイス使用率を大幅に改善します。

`fit` の詳しい使用方法については、[トレーニングと評価ガイド](https://www.tensorflow.org/guide/keras/training_with_built_in_methods)をご覧ください。組み込みトレーニングと評価ループのカスタマイズ方法については、[`fit()` の処理をカスタマイズする](https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit)をご覧ください。

### その他の API とツール

Keras には、以下を含む、ディープラーニング向けの API とツールがその他多数備わっています。

- [オプティマイザ](https://keras.io/api/optimizers/)
- [指標](https://keras.io/api/metrics/)
- [損失](https://keras.io/api/losses/)
- [データ読み込みユーティリティ](https://keras.io/api/data_loading/)

利用可能な API の完全なリストについては、[Keras API リファレンス](https://keras.io/api/)をご覧ください。他の Keras プロジェクトとイニシアチブについてさらに詳しく知るには、[Keras エコシステム](https://keras.io/getting_started/ecosystem/)をご覧ください。

## 次のステップ

TensorFlow で Keras を使い始めるには、以下のトピックをご覧ください。

- [Sequential モデル](https://www.tensorflow.org/guide/keras/sequential_model)
- [Functional API](https://www.tensorflow.org/guide/keras/functional)
- [組み込みメソッドを使用したトレーニングと評価](https://www.tensorflow.org/guide/keras/training_with_built_in_methods)
- [サブクラス化による新しいレイヤーとモデルの作成](https://www.tensorflow.org/guide/keras/custom_layers_and_models)
- [シリアル化と保存](https://www.tensorflow.org/guide/keras/save_and_serialize)
- [前処理レイヤーを使用する](https://www.tensorflow.org/guide/keras/preprocessing_layers)
- [fit() の処理をカスタマイズする](https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit)
- [トレーニングループの新規作成](https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch)
- [RNN を使用する](https://www.tensorflow.org/guide/keras/rnn)
- [マスクとパディングを理解する](https://www.tensorflow.org/guide/keras/masking_and_padding)
- [独自のコールバックの作成](https://www.tensorflow.org/guide/keras/custom_callback)
- [転移学習とファインチューニング](https://www.tensorflow.org/guide/keras/transfer_learning)
- [マルチ GPU と分散トレーニング](https://www.tensorflow.org/guide/keras/distributed_training)

Keras についてさらに詳しく知るには、[keras.io](http://keras.io) で以下のトピックをご覧ください。

- [About Keras](https://keras.io/about/)（Keras について）
- [Introduction to Keras for Engineers](https://keras.io/getting_started/intro_to_keras_for_engineers/)（エンジニア向けの Keras 入門）
- [Introduction to Keras for Researchers](https://keras.io/getting_started/intro_to_keras_for_researchers/)（研究者向けの Keras 入門）
- [Keras API reference](https://keras.io/api/)（Keras API リファレンス）
- [The Keras ecosystem](https://keras.io/getting_started/ecosystem/)（Keras エコシステム）
