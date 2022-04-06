# TFFを使用した連合学習の研究

<!-- Note that some section headings are used as deep links into the document.
     If you update those section headings, please make sure you also update
     any links to the section. -->

## 概要

TFFは、現実的なプロキシデータセットで連合計算をシミュレートすることにより、連合学習（FL）研究を行うための拡張可能な強力なフレームワークです。 このページでは、研究シミュレーションに関連する主な概念とコンポーネント、およびTFFでさまざまな種類の研究を実施するための詳細なガイダンスについて説明します。

## TFF の研究コードの典型的な構造

TFFに実装された研究用の連合学習のシミュレーションは、通常、3つの主要なタイプのロジックで構成されます。

1. Individual pieces of TensorFlow code, typically `tf.function`s, that encapsulate logic that runs in a single location (e.g., on clients or on a server). This code is typically written and tested without any `tff.*` references, and can be re-used outside of TFF. For example, the [client training loop in Federated Averaging](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/simple_fedavg/simple_fedavg_tf.py#L184-L222) is implemented at this level.

2. TensorFlow Federated orchestration logic, which binds together the individual `tf.function`s from 1. by wrapping them as `tff.tf_computation`s and then orchestrating them using abstractions like `tff.federated_broadcast` and `tff.federated_mean` inside a `tff.federated_computation`. See, for example, this [orchestration for Federated Averaging](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/simple_fedavg/simple_fedavg_tff.py#L112-L140).

3. 本番環境の連合学習システムの制御ロジックをシミュレートする外部ドライバースクリプトは、データセットからシミュレートされたクライアントを選択し、それらのクライアントで2.で定義された連合計算を実行します。（例：[a Federated EMNIST experiment driver](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/examples/simple_fedavg/emnist_fedavg_main.py)）

## 連合学習データセット

TensorFlow の連合学習データセットは、連合学習で解決できる実際の問題の特徴を表す[複数のデータセットをホスト](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets)します。

注意: これらのデータセットは、[ClientData API ](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/ClientData)に記載されているように、Numpy 配列として Python ベースの機械学習フレームワークでも使用できます。

データセットには以下が含まれています。

- [**StackOverflow**。](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/stackoverflow/load_data)言語モデリングや教師あり学習タスク用の現実的なテキストデータセット。342,477人のユニークユーザーがトレーニングセットで135,818,730例（センテンス）を使用します。

- [**Federated EMNIST**。](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/emnist/load_data)各クライアントが異なるライターに対応する、EMNIST文字と数字のデータセットの連合前処理。完全なトレインセットには、62のラベルからの671,585の例を持つ3400人のユーザーが含まれています。

- [**Shakespeare**。](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/shakespeare/load_data)ウィリアムシェイクスピアの全作品に基づいた、文字レベルの小さなテキストデータセット。 データセットは715人のユーザー（シェイクスピア劇のキャラクター）で構成されます。各例は、特定の劇のキャラクターが話す連続した一連の行に対応しています。

- [**CIFAR-100**。](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/cifar100/load_data)500のトレーニングクライアントと100のテストクライアントにわたるCIFAR-100データセットの連合パーティション。各クライアントには100のユニークな例があります。 パーティションは、クライアント間でより現実的な異質性を作成する方法で行われます。 詳細については、[API](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/cifar100/load_data)をご覧ください。

- [**Google Landmark v2 データセット。**](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/gldv2/load_data) データセットにはさまざまな世界的名所の写真で構成されており、データの連合パーティションを得られるよう、画像は写真家ごとにグループ化されています。データベースには、233 件のクライアントと 23080 枚の画像が含まれる小さいデータセットと、1262 件のクライアントと 164172 枚の画像が含まれる大きなデータセットの 2 種類があります。

- [**CelebA**。](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/celeba/load_data) 有名人の顔のサンプル（画像と顔の特徴）を集めたデータセットです。連合データセットには各有名人のサンプルがクライアントを形成するようにグループ化されています。クライアントは 9343 件あり、それぞれに少なくとも 5 個のサンプルがあります。データセットは、クライアント別またはサンプル別に、トレーニンググループとテストグループに分割できます。

- [**iNaturalist**。](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/inaturalist/load_data) さまざまな種の写真で構成されるデータセット。データセットには、1,203 種の 120,300 枚の画像が含まれます。データセットには 7 つの種類があります。1 つは写真家別にグループ化されており、9257 件のクライアントが含まれます。残りのデータセットは、写真が撮影された場所の位置情報でグループ化されています。これらの 6 つのデータセットは、11～3,606 件のクライアントで構成されています。

## 高性能シミュレーション

FL *シミュレーション*の実時間は、アルゴリズムを評価するための適切な指標ではありませんが（シミュレーションハードウェアは実際のフェデレーテッドラーニングデプロイメント環境を表していないため）、フェデレーテッドラーニングシミュレーションをすばやく実行できることは、研究の生産性にとって重要です。そのため、TFF は単一および複数のマシンで高性能なランタイムを提供するために多額の投資を行ってきました。現在、ドキュメントは開発中ですが、[「TFF を使用した高性能シミュレーション」](https://www.tensorflow.org/federated/tutorials/simulations)チュートリアルと[ GCP で TFF を使用したシミュレーションを設定する](https://www.tensorflow.org/federated/tutorials/simulations_with_accelerators)手順をご覧ください。高性能 TFF ランタイムはデフォルトで有効になっています。

## さまざまな研究分野の TFF

### 連合最適化アルゴリズム

TFF を利用すると指定するカスタマイズレベルに応じて連合最適化アルゴリズムの研究をさまざまな方法で行うことができます。

[Federated Averaging](https://arxiv.org/abs/1602.05629) アルゴリズムの最小限のスタンドアロン実装は、[こちら](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/examples/simple_fedavg)に提供されています。コードには、例として、ローカル計算用の[TF関数](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/examples/simple_fedavg/simple_fedavg_tf.py)、オーケストレーション用の[TFF計算](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/examples/simple_fedavg/simple_fedavg_tff.py)、およびEMNISTデータセットの[ドライバースクリプト](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/examples/simple_fedavg/emnist_fedavg_main.py)が含まれています。これらのファイルは、[ README ](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/examples/simple_fedavg/README.md)の詳細な指示に従って、カスタマイズされたアプリケーションやアルゴリズムの変更に簡単に適合させることができます。

A more general implementation of Federated Averaging can be found [here](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/learning/algorithms/fed_avg.py). This implementation allows for more sophisticated optimization techniques, including the use of different optimizers on both the server and client. Other federated learning algorithms, including federated k-means clustering, can be found [here](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/learning/algorithms/).

### モデルと圧縮の更新

TFF は[ tensor_encoding ](https://github.com/tensorflow/model-optimization/tree/master/tensorflow_model_optimization/python/core/internal/tensor_encoding) API を使用して非可逆圧縮アルゴリズムを有効にし、サーバーとクライアント間の通信コストを削減します。サーバーからクライアントおよびクライアントからサーバーへの [Federated Averaging](https://arxiv.org/abs/1812.07210) アルゴリズムを使用した圧縮のトレーニング例については、[この実験](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/compression/run_experiment.py)をご覧ください。

カスタム圧縮アルゴリズムを実装してトレーニングループに適用するには、次の手順に従います。

1. 新しい圧縮アルゴリズムを[ `EncodingStageInterface ` ](https://github.com/tensorflow/model-optimization/blob/master/tensorflow_model_optimization/python/core/internal/tensor_encoding/core/encoding_stage.py#L75)のサブクラスとして実装します。または、[この例](https://github.com/tensorflow/model-optimization/blob/master/tensorflow_model_optimization/python/core/internal/tensor_encoding/core/encoding_stage.py#L274)のようにより一般的なバリアント<a><code>AdaptiveEncodingStageInterface</code></a>として実装します。
2. 新しい[ `Encoder` ](https://github.com/tensorflow/model-optimization/blob/master/tensorflow_model_optimization/python/core/internal/tensor_encoding/core/core_encoder.py#L38)を作成し、[モデルブロードキャスト](https://github.com/google-research/federated/blob/master/compression/run_experiment.py#L118)または[モデル更新の平均化](https://github.com/google-research/federated/blob/master/compression/run_experiment.py#L144)に特化します。
3. これらのオブジェクトを使用して、[トレーニング計算全体](https://github.com/google-research/federated/blob/master/compression/run_experiment.py#L247)を構築します。

### 差別的なプライバシー

TFF は [TensorFlow Privacy ](https://github.com/tensorflow/privacy)ライブラリと相互運用可能であり、差別化されたプライバシーを持つモデルの連合トレーニングの新しいアルゴリズムの研究を可能にします。[基本的な DP-FedAvg アルゴリズム](https://arxiv.org/abs/1710.06963)と[拡張機能](https://arxiv.org/abs/1812.06210)を使用して DP でトレーニングする例については、[この実験ドライバー](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/differential_privacy/stackoverflow/run_federated.py)をご覧ください。

カスタム DP アルゴリズムを実装し、それを Federated Averaging の集計更新に適用する場合、新しい DP 平均アルゴリズムを [`tensorflow_privacy.DPQuery`](https://github.com/tensorflow/privacy/blob/master/tensorflow_privacy/privacy/dp_query/dp_query.py#L54) のサブクラスとして実装し、クエリのインスタンスで `tff.aggregators.DifferentiallyPrivateFactory` を作成することができます。[DP-FTRL アルゴリズム](https://arxiv.org/abs/2103.00039)の実装例は[こちら](https://github.com/google-research/federated/blob/master/dp_ftrl/dp_fedavg.py)にあります。

[以下](#generative_adversarial_networks)で説明されている Federated GAN は、ユーザーレベルの差分プライバシーを実装する TFF プロジェクトの別の例です（例として[このコード](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/gans/tff_gans.py#L293)を参照してください）。

### 堅牢性と攻撃

TFF は、*[「連合学習にバックドアを組み込むことは可能か？」](https://arxiv.org/abs/1911.07963)*で取り上げられているように、連合学習システムへの標的型攻撃とプライバシーに基づく差別化された防御のシミュレーションにも使用できます。これは、潜在的に悪意のあるクライアントとのイテレーションプロセスを構築することにより実行されます。(`build_federated_averaging_process_attacked`を参照してください。) 詳細は [targeted_attack](https://github.com/tensorflow/federated/tree/6477a3dba6e7d852191bfd733f651fad84b82eab/tensorflow_federated/python/research/targeted_attack) ディレクトリをご覧ください。

- 新しい攻撃アルゴリズムは、Tensorflow関数であるクライアント更新関数を記述することにより、実装できます。
- 新しい防御策は、クライアントの出力を集約してグローバル更新を取得する[](https://github.com/tensorflow/federated/blob/6477a3dba6e7d852191bfd733f651fad84b82eab/tensorflow_federated/python/core/utils/computation_utils.py#L103)

シミュレーションのサンプルスクリプトについては、[`emnist_with_targeted_attack.py`](https://github.com/tensorflow/federated/blob/6477a3dba6e7d852191bfd733f651fad84b82eab/tensorflow_federated/python/research/targeted_attack/emnist_with_targeted_attack.py)をご覧ください。

### 生成的敵対的ネットワーク

GAN は、標準の Federated Averaging とは少し異なる興味深い[連合オーケストレーションパターン](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/gans/tff_gans.py#L266-L316)を作り出します。これには、それぞれ独自の最適化ステップでトレーニングされた 2 つの異なるネットワーク（ジェネレーターとディスクリミネーター）が含まれます。

TFF は、GAN の連合学習の研究に使用できます。たとえば、[最近の研究](https://arxiv.org/abs/1911.06679)で示されている DP-FedAvg-GAN アルゴリズムは[ TFFで実装されています](https://github.com/tensorflow/federated/tree/main/federated_research/gans)。この研究は、連合学習、生成モデル、[差分プライバシー](#differential_privacy)を組み合わせることの有効性を示しています。

### パーソナライゼーション

連合学習の設定におけるパーソナライゼーションは、活発な研究分野です。パーソナライゼーションの目標は、さまざまな推論モデルをさまざまなユーザーに提供することです。この問題に対しては様々なアプローチがあります。

One approach is to let each client fine-tune a single global model (trained using federated learning) with their local data. This approach has connections to meta-learning, see, e.g., [this paper](https://arxiv.org/abs/1909.12488). An example of this approach is given in [`emnist_p13n_main.py`](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/personalization/emnist_p13n_main.py). To explore and compare different personalization strategies, you can:

- 初期モデルから開始し、各クライアントのローカルデータセットを使用してパーソナライズモデルをトレーニングおよび評価する`tf.function `を定義します。 例は、[`build_personalize_fn`](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/examples/personalization/p13n_utils.py)をご覧ください。

- 戦略名を対応するパーソナライズ戦略にマップする`OrderedDict`を定義し、それを [`tff.learning.build_personalization_eval`](https://www.tensorflow.org/federated/api_docs/python/tff/learning/build_personalization_eval)の`personalize_fn_dict` 引数として使用します。

Another approach is to avoid training a fully global model by training part of a model entirely locally. An instantiation of this approach is described in [this blog post](https://ai.googleblog.com/2021/12/a-scalable-approach-for-partially-local.html). This approach is also connected to meta learning, see [this paper](https://arxiv.org/abs/2102.03448). To explore partially local federated learning, you can:

- Check out the [tutorial](https://www.tensorflow.org/federated/tutorials/federated_reconstruction_for_matrix_factorization) for a complete code example applying Federated Reconstruction and [follow-up exercises](https://www.tensorflow.org/federated/tutorials/federated_reconstruction_for_matrix_factorization#further_explorations).

- Create a partially local training process using [`tff.learning.reconstruction.build_training_process`](https://www.tensorflow.org/federated/api_docs/python/tff/learning/reconstruction/build_training_process), modifying `dataset_split_fn` to customize process behavior.
