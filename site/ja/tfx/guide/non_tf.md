# TFX で他の ML フレームワークを使用する

プラットフォームとしての TFX はフレームワークに依存せず、JAX や scikit-learn などの他の ML フレームワークでも使用できます。

モデル開発者にとって、これは別の ML フレームワークで実装されたモデルコードを書き直す必要がなく、TFX でトレーニングコードの大部分をそのまま再利用でき、TFX と他の TensorFlow エコシステムが提供する機能を使用できるメリットがあります。

TFX パイプライン SDK や、パイプラインオーケストレータなどの TFX のほとんどのモジュールは、TensorFlow に直接依存していませんが、データ形式など TensorFlow 指向の側面がいくつかあります。特定のモデリングフレームワークのニーズを考慮すれば、TFX パイプラインは、Scikit-learn、XGBoost、PyTorch などの他の Python ベースの ML フレームワークでのモデルのトレーニングに使用することができます。標準 TFX コンポーネントを他のフレームワークで使用するには、以下を含むいくつかの考慮事項があります。

- **ExampleGen** は TFRecord ファイルに [tf.train.Example](https://www.tensorflow.org/tutorials/load_data/tfrecord) を出力します。トレーニングデータのジェネリックな表現であり、下流のコンポーネントは [TFXIO](https://github.com/tensorflow/community/blob/master/rfcs/20191017-tfx-standardized-inputs.md) を使用して、メモリ内の Arrow/RecordBatch として読み取ります。これはさらに `tf.dataset`、`Tensors`、またはその他の形式に変換することが可能です。tf.train.Example/TFRecord 以外のペイロード/ファイル形式が検討されていますが、TFXIO ユーザーに対しては、ブラックボックスである必要があります。
- **Transform** は、トレーニングに使用されたフレームワークに関係なく変換したトレーニングサンプルを生成するために使用できますが、モデル形式が `saved_model` でない場合、ユーザーは変換グラフをモデルに埋め込めなくなってしまいます。その場合、モデル予測は未加工の特徴量の代わりに、変換済みの特徴量を使用する必要があり、ユーザーは serving 時にモデル予測を呼び出す前に前処理手順として変換を実行できます。
- **Trainer** は [GenericTraining](https://www.tensorflow.org/tfx/guide/trainer#generic_trainer) をサポートするため、ユーザーはあらゆる ML フレームワークを使用してそれらのモデルをトレーニングできます。
- **Evaluator** はデフォルトでは `saved_model` のみをサポートしますが、ユーザーはモデル評価予測を生成する UDF を提供できます。

Python ベースではないフレームワークでモデルをトレーニングするには、Docker コンテナでカスタムのトレーニングコンポーネントを、Kubernetes などのコンテナ化された環境で実行するパイプラインの一部として隔離する必要があります。

## JAX

[JAX](https://github.com/google/jax) は、Autograd と XLA で構成され、機械学習リサーチのパフォーマンスを向上します。[Flax](https://github.com/google/flax) は JAX のニューラルネットワークライブラリおよびエコシステムで、柔軟性を重視して設計されています。

[jax2tf](https://github.com/google/jax/tree/main/jax/experimental/jax2tf) を使用することで、トレーニング済みの JAX/Flax モデルを `saved_model` 形式に変換することができ、ジェネリックなトレーニングとモデル評価を使用して、TFX でシームレスに使用できます。詳細については、この[例](https://github.com/tensorflow/tfx/blob/master/tfx/examples/penguin/penguin_utils_flax_experimental.py)を確認してください。

## scikit-learn

[Scikit-learn](https://scikit-learn.org/stable/) は Python プログラミング言語の機械学習ライブラリです。e2e の[例](https://github.com/tensorflow/tfx-addons/tree/main/examples/sklearn_penguins)では、TFX のアドオンでカスタマイズされたトレーニングと評価が使用されています。
