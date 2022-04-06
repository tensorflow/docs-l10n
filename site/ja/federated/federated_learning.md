# 連合学習

## 概要

このドキュメントでは、TensorFlow に実装された既存の機械学習モデルを使用した連合トレーニングや評価などの連合学習のタスクを容易にするインターフェースを紹介します。これらのインターフェースを設計する上の主な目標は、内部機能についての知識を必要とせずに、連合学習を実験できるようにし、さまざまな既存のモデルとデータに実装された連合学習アルゴリズムを評価することでした。ぜひ、このプラットフォームに貢献してください。TFF は拡張性と構成可能性を考慮して設計されているので、皆様からの貢献を歓迎します！

このレイヤーにより提供されるインターフェースは、次の 3 つの主要部分で構成されています。

- **モデル**。TFF で使用するために既存のモデルをラップできるようにするクラスとヘルパー関数。モデルのラッピングは、１つのラッピング関数 (`tff.learning.from_keras_model`) を呼び出して簡単に実行できます。または、完全にカスタマイズできるように`tff.learning.Model`インターフェースのサブクラスを定義することもできます。

- **連合コンピュテーションビルダー**。既存のモデルを使用して、トレーニングまたは評価するための連合コンピュテーションを構築するヘルパー関数。

- **データセット**。フェデレーテッドラーニングのシナリオのシミュレーションで使用するために Python でダウンロードしてアクセスできるデータのコレクション。フェデレーテッドラーニングは、集中管理された場所で簡単にダウンロードできない分散データを使用するように設計されていますが、研究開発の段階では、ダウンロードしてローカルで操作できるデータを使用して初期の実験を行うと、特にこのアプローチに不慣れな開発者にとって便利です。

<code>tff.simulation</code>にグループ化された研究データセットおよびその他のシミュレーション関連機能を除き、これらのインターフェースは主に`tff.learning`名前空間で定義されます。このレイヤーは、ランタイム環境も提供する [Federated Core (FC)](federated_core.md) により提供される下位レベルのインターフェースを使用して実装されます。

先に進む前に、まず[画像分類](tutorials/federated_learning_for_image_classification.ipynb)と[テキスト生成](tutorials/federated_learning_for_text_generation.ipynb)に関するチュートリアルを確認することをお勧めします。これらのチュートリアルでは、具体的な例を使用して、ここで説明する概念のほとんどを紹介しています。TFF のしくみについての詳細は、[カスタムアルゴリズム](tutorials/custom_federated_algorithms_1.ipynb)チュートリアルをご覧ください。このチュートリアルでは連合コンピュテーショのロジックを表現し、`tff.learning`インターフェースの既存の実装を研究するために使用する低レベルのインターフェースについて説明します。

## モデル

### アーキテクチャの前提

#### シリアル化

TFF は、さまざまな分散学習シナリオをサポートすることを目的としています。このシナリオでは、記述した機械学習モデルのコードをさまざまな機能を持つ多数の異種クライアントで実行できます。一部のアプリケーションでは、これらのクライアントは強力なデータベースサーバーである場合がありますが、プラットフォームがサポートする重要なアプリケーションの多くは、リソースが限られたモバイルデバイスや組み込みデバイスです。これらのデバイスが Python ランタイムをホストできることは想定できません。この時点で想定できるのは、ローカルの TensorFlow ランタイムをホストできることだけです。したがって、TFF で行う基本的なアーキテクチャの前提は、モデルコードが TensorFlow グラフとしてシリアル化可能でなければならないということです。

eager モードの使用など、最新のベストプラクティスに従って TF コードを開発することができますが、最終的なコードはシリアル化可能である必要があります (eager-modeコードの場合は`tf.function`としてラップできます)。これにより、実行時に必要な Python の状態または制御フローを ([Autograph ](https://www.tensorflow.org/guide/autograph)などを使用して) シリアル化できるようになります 。

現在、TensorFlow は、eager モードの TensorFlow のシリアラル化と逆シリアル化を完全にはサポートしていません。TFF でのシリアル化は現在、TF 1.0 パターンに従い、すべてのコードは TFF が制御する`tf.Graph`内に構築する必要があります。つまり、現在 TFF は既に構築されたモデルを使用できません。モデル定義ロジックは、`tff.learning.Model`を返す引数なしの関数にパッケージ化され、この関数が TFF によって呼び出され、モデルのすべてのコンポーネントが確実にシリアル化されます。さらに、強く型付けされた環境であるため、TFF にはモデルの入力タイプの仕様など追加の*メタデータ*が少し必要になります。

#### 集計

ほとんどの場合、Keras を使用してモデルを構築することを強くお勧めします。以下の [Keras コンバータ](#converters-for-keras)セクションを参照してください。これらのラッパーは、モデルの更新の集計とモデルに定義されたメトリックを自動的に処理します。 ただし、一般的な`tff.learning.Model`の集計がどのように処理されるかを理解することは有用です。

フェデレーテッドラーニングには常に少なくともローカルオンデバイス集計とクロスデバイス (または連合) 集計の 2 つの集計レイヤーがあります。

- **ローカル集計**。このレベルの集計は、個々のクライアントが所有するサンプルの複数のバッチにわたる集計を指します。これは、モデルがローカルでトレーニングされるにつれて順次進化し続ける両方のモデルパラメーター(変数)、および、計算された統計 (平均損失、精度、その他のメトリックなど) に適用されます。これらの統計は、個々のクライアントのローカルデータストリームを反復処理するときに、モデルは再びローカルで更新されます。

    このレベルでの集計の実行はモデルコードが処理し、標準の TensorFlow 構造を使用して実行されます。

    処理の一般的な構造は次のとおりです。

    - モデルはまず、`tf.Variable`を作成してバッチ数や処理されたサンプル数、バッチごとまたはサンプルごとの損失の合計などの集計を保持します。

    - TFF は、`Model`で`forward_pass`メソッドを複数回呼び出し、クライアントデータの後続のバッチで順次実行するため、副次的効果としてさまざまな集計を保持する変数を更新できます。

    - Finally, TFF invokes the `report_local_unfinalized_metrics` method on your Model to allow your model to compile all the summary statistics it collected into a compact set of metrics to be exported by the client. This is where your model code may, for example, divide the sum of losses by the number of examples processed to export the average loss, etc.

- **連合集計**。このレベルの集計は、システム内の複数のクライアント (デバイス) にわたる集計を指します。これはクライアント全体で平均化されるモデルパラメータ (変数) とローカル集計の結果としてモデルがエクスポートしたメトリックに適用されます。

    このレベルで集計を実行するのは TFF の責任です。ただし、モデル作成者はこのプロセスを制御できます (詳細は以下を参照してください)。

    処理の一般的な構造は次のとおりです。

    - 初期モデルとトレーニングに必要なすべてのパラメーターは、サーバーにより一連のトレーニングまたは評価に参加するクライアントのサブセットに配布されます。

    - 各クライアントでは、独立かつ並行してモデルコードがローカルデータバッチのストリームで繰り返し呼び出され、上記のように新しいモデルパラメーターのセット (トレーニング時) と新しいローカルメトリックのセット (ローカル集計) が生成されます。

    - TFF runs a distributed aggregation protocol to accumulate and aggregate the model parameters and locally exported metrics across the system. This logic is expressed in a declarative manner using TFF's own *federated computation* language (not in TensorFlow). See the [custom algorithms](tutorials/custom_federated_algorithms_1.ipynb) tutorial for more on the aggregation API.

### 抽象インターフェース

この基本的な *constructor* と *metadata* インターフェースは、次のようにインターフェース`tff.learning.Model `で表されます。

- The constructor, `forward_pass`, and `report_local_unfinalized_metrics` methods should construct model variables, forward pass, and statistics you wish to report, correspondingly. The TensorFlow constructed by those methods must be serializable, as discussed above.

- `input_spec`プロパティと、トレーニング可能な変数、トレーニング不可能な変数、およびローカル変数のサブセットを返す 3 つのプロパティは、メタデータを表します。TFF はこの情報を使用して、モデルの部分を連合最適化アルゴリズムに接続する方法を決定し、構築されたシステムの正確性を検証するのに役立つ内部型シグネチャを定義します (モデルが使用するように設計されているものと一致しないデータに対してモデルをインスタンス化しないようにするため)。

In addition, the abstract interface `tff.learning.Model` exposes a property `metric_finalizers` that takes in a metric's unfinalized values (returned by `report_local_unfinalized_metrics()`) and returns the finalized metric values. The `metric_finalizers` and `report_local_unfinalized_metrics()` method will be used together to build a cross-client metrics aggregator when defining the federated training processes or evaluation computations. For example, a simple `tff.learning.metrics.sum_then_finalize` aggregator will first sum the unfinalized metric values from clients, and then call the finalizer functions at the server.

独自のカスタム`tf.learning.Model`を定義する方法の例は、[画像分類](tutorials/federated_learning_for_image_classification.ipynb)チュートリアルの後半と、[`model_examples.py`](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/learning/model_examples.py)のテストで使用するサンプルモデルにあります。

### Keras 用コンバータ

TFF に必要なほぼすべての情報は、`tf.keras`インターフェースを呼び出すことで取得できます。Keras モデルがある場合は`tff.learning.from_keras_model`を使用して`tff.learning.Model`を構築できます。

TFF は、コンストラクター（次のような引数のない*モデル関数*）を提供することを求めていることに注意してください。

```python
def model_fn():
  keras_model = ...
  return tff.learning.from_keras_model(keras_model, sample_batch, loss=...)
```

モデル自体に加えて、TFF がモデルの入力のタイプと形状を決定するために使用するデータのサンプルバッチを提供します。これにより、TFF がクライアントデバイスに実際に存在するデータのモデルを適切にインスタンス化できるようになります (このデータは、シリアル化する TensorFlow を構築しているときには一般に利用できないと想定されているため)。

Keras ラッパーの使用法は、[画像分類](tutorials/federated_learning_for_image_classification.ipynb)と[テキスト生成](tutorials/federated_learning_for_text_generation.ipynb)のチュートリアルで説明されています。

## 連合コンピュテーションビルダー

`tff.learning` パッケージは、学習関連のタスクを実行する`tff.Computation`のいくつかのビルダーを提供します。そのような計算のセットは、今後拡張されることが予想されています。

### アーキテクチャの前提

#### 実行

連合コンピュテーションの実行には 2 つの異なるフェーズがあります。

- **コンパイル**: TFF は最初にフェデレーテッドラーニングアルゴリズムを分散計算全体の抽象シリアル化表現に*コンパイル*します。これは TensorFlow のシリアル化が行われるときに実行されますが、より効率的な実行をサポートするために他の変換が行われる場合があります。コンパイラーにより生成されたシリアル化された表現を、*連合コンピュテーション*と呼びます。

- **実行**: TFF は、これらの計算を*実行*する方法を提供します。現時点では、実行はローカルシミュレーションでのみサポートされています (例: シミュレートされた分散データを使用するノートブックでの実行)。

[連合モデルの平均化](https://arxiv.org/abs/1602.05629)を使用するトレーニングアルゴリズムなど TFF の Federated Learning API によって生成された連合コンピュテーション、または、主に以下のようないくつか要素を含む連合評価:

- モデルコードのシリアル化された形式、および、モデルのトレーニング/評価ループを実行するためにフェデレーテッドラーニングフレームワークにより構築された追加の TensorFlow コード (オプティマイザの構築、モデルの更新の適用、`tf.data.Dataset`の反復、メトリックの計算、サーバーにおける集計された更新の適用など)。

- *クライアント*と*サーバー*の間の通信の宣言仕様 (通常、クライアントデバイス全体のさまざまな形式の*集計*、およびサーバーからすべてのクライアントへの*ブロードキャスト*) そして、この分散通信が TensorFlow コードのクライアントローカルまたはサーバーローカルの実行とどのようにインターリーブされるか。

このシリアル化された形式で表される*連合コンピュテーション*は、Python とは異なるプラットフォームに依存しない内部言語で表現されますが、Federated Learning API を使用するため、この表現の詳細についての知識は必要はありません。計算は、Python コードのタイプ`tff.Computation`のオブジェクトとして表され、ほとんどの場合、不透明な Python `callable`として扱うことができます。

チュートリアルでは、これらの連合コンピュテーションを通常の Python 関数のように呼び出し、ローカルで実行します。ただし、TFF は実行環境のほとんどの側面にとらわれない方法で連合コンピュテーションを表現するように設計されているので、`Android`を実行しているデバイスのグループや、データセンターのクラスターにデプロイできる場合があります。繰り返しますが、これの主な結果は、[シリアル化](#serialization)に関する強力な仮定です。特に、以下で説明されている`build_...`メソッドの 1 つを呼び出すと、計算は完全にシリアル化されます。

#### 状態のモデリング

TFF は関数型プログラミング環境ですが、連合学習に関連する多くのプロセスはステートフルです。たとえば、連合モデルの平均化を複数回行うトレーニングループは、*ステートフルプロセス*として分類できます。このプロセスでは、ラウンドごとにと進化する状態には、トレーニングされているモデルパラメータのセットとオプティマイザに関連する追加の状態 (運動量ベクトルなど) が含まれます。

TFF は関数的であるため、ステートフルプロセスは、その時点の状態を入力として受け入れ、更新された状態を出力として提供する計算として TFF でモデル化されます。ステートフルプロセスを完全に定義するには、初期状態がどこから来るかを指定する必要があります (そうでないと、プロセスをブートストラップできません)。これは、ヘルパークラス`tff.templates.IterativeProcess`の定義でキャプチャされ、2 つのプロパティ`initialize`と`next`は、それぞれ、初期化と反復に対応します。

### 利用可能なビルダー

現在、TFF は、フェデレーテッドトレーニングと評価のための連合コンピュテーションを生成する 2 つのビルダー関数を提供しています。

- `tff.learning.build_federated_averaging_process`*はモデル関数*と*クライアントオプティマイザ*を受け取り、ステートフルな`tff.templates.IterativeProcess`を返します。

- 評価はステートフルではないため、`tff.learning.build_federated_evaluation`は*モデル関数*を取り、モデルの連合評価のための一つの連合コンピュテーションを返します。

## データセット

### アーキテクチャの前提

#### クライアントの選択

典型的なフェデレーテッドラーニングのシナリオでは、潜在的に何億ものクライアントデバイスの大きな*母集団*があり、その内アクティブでいつでもトレーニングに利用できるのは一部のみです。 (たとえば、従量制のネットワーク上になく、アイドル状態で電源に接続されているクライアントに限定される場合があります)。一般に、トレーニングまたは評価に参加できるクライアントのセットは、開発者の管理外です。さらに、数百万のクライアントを調整することは非現実的であるため、通常のトレーニングまたは評価のラウンドには利用可能なクライアントの一部のみが含まれます ([ランダムにサンプリング](https://arxiv.org/pdf/1902.01046.pdf)されたクライアントなど)。

これの主な結果として、連合コンピュテーションは、設計段階からクライアントの正確なセットに関連なく表現されるようになっています。すべての処理は、匿名の*クライアント*の抽象的なグループに対する集計操作として表現され、そのグループはトレーニングのラウンドごとに異なる場合があります。具体的なクライアントへの計算の実際のバインディング、および、それらが計算に供給する具体的なデータは計算外でモデル化されます。

フェデレーテッドラーニングコードの現実的なデプロイメントをシミュレートするには、通常、次のようなトレーニングループを記述します。

```python
trainer = tff.learning.build_federated_averaging_process(...)
state = trainer.initialize()
federated_training_data = ...

def sample(federate_data):
  return ...

while True:
  data_for_this_round = sample(federated_training_data)
  state, metrics = trainer.next(state, data_for_this_round)
```

これを容易にするために、シミュレーションで TFF を使用する場合、連合データは Python`list`として受け入れられ、参加しているクライアントデバイスごとに 1 つの要素を使用して、そのデバイスのローカル`tf.data.Dataset`を表します。

### 抽象インターフェース

シミュレートされた連合データセットの処理を標準化するために、TFF には抽象的なインターフェース `tff.simulation.datasets.ClientData` が提供されています。これにより、クライアントのセットを列挙し、特定のクライアントのデータを含む `tf.data.Dataset` を構築できます。これらの `tf.data.Dataset` は、eager モードで生成された連合計算への入力として直接供給することができます。

クライアント ID にアクセスする機能は、シミュレーションで使用するためのみにデータセットにより提供される機能であり、クライアントの特定のサブセットからのデータをトレーニングする機能が必要になる場合があることに注意してください (たとえば、さまざまなタイプのクライアントの日中の可用性をシミュレートする場合など)。コンパイルされた計算と基になるランタイムは、クライアント ID の概念を一切含みません。たとえば`tff.templates.IterativeProcess.next`の呼び出しなどで、クライアントの特定のサブセットからのデータが入力として選択されると、クライアント ID はその中に表示されなくなります。

### 利用可能なデータセット

We have dedicated the namespace `tff.simulation.datasets` for datasets that implement the `tff.simulation.datasets.ClientData` interface for use in simulations, and seeded it with datasets to support the [image classification](tutorials/federated_learning_for_image_classification.ipynb) and [text generation](tutorials/federated_learning_for_text_generation.ipynb) tutorials. We'd like to encourage you to contribute your own datasets to the platform.
