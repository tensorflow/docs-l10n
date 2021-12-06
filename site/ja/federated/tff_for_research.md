# TFFを使用した連合学習の研究

<!-- Note that some section headings are used as deep links into the document.
     If you update those section headings, please make sure you also update
     any links to the section. -->

## 概要

TFFは、現実的なプロキシデータセットで連合計算をシミュレートすることにより、連合学習（FL）研究を行うための拡張可能な強力なフレームワークです。 このページでは、研究シミュレーションに関連する主な概念とコンポーネント、およびTFFでさまざまな種類の研究を実施するための詳細なガイダンスについて説明します。

## TFF の研究コードの典型的な構造

TFFに実装された研究用の連合学習のシミュレーションは、通常、3つの主要なタイプのロジックで構成されます。

1. Individual pieces of TensorFlow code, typically `tf.function`s, that encapsulate logic that runs in a single location (e.g., on clients or on a server). This code is typically written and tested without any `tff.*` references, and can be re-used outside of TFF. For example, the [client training loop in Federated Averaging](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/examples/simple_fedavg/simple_fedavg_tf.py#L184-L222) is implemented at this level.

2. TensorFlow Federated orchestration logic, which binds together the individual `tf.function`s from 1. by wrapping them as `tff.tf_computation`s and then orchestrating them using abstractions like `tff.federated_broadcast` and `tff.federated_mean` inside a `tff.federated_computation`. See, for example, this [orchestration for Federated Averaging](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/examples/simple_fedavg/simple_fedavg_tff.py#L112-L140).

3. An outer driver script that simulates the control logic of a production FL system, selecting simulated clients from a dataset and then executing federated computations defined in 2. on those clients. For example, [a Federated EMNIST experiment driver](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/examples/simple_fedavg/emnist_fedavg_main.py).

## 連合学習データセット

TensorFlow の連合学習データセットは、連合学習で解決できる実際の問題の特徴を表す[複数のデータセットをホスト](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets)します。

注意: これらのデータセットは、[ClientData API ](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/ClientData)に記載されているように、Numpy 配列として Python ベースの機械学習フレームワークでも使用できます。

データセットには以下が含まれています。

- [**StackOverflow**。](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/stackoverflow/load_data)言語モデリングや教師あり学習タスク用の現実的なテキストデータセット。342,477人のユニークユーザーがトレーニングセットで135,818,730例（センテンス）を使用します。

- [**Federated EMNIST**。](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/emnist/load_data)各クライアントが異なるライターに対応する、EMNIST文字と数字のデータセットの連合前処理。完全なトレインセットには、62のラベルからの671,585の例を持つ3400人のユーザーが含まれています。

- [**Shakespeare**。](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/shakespeare/load_data)ウィリアムシェイクスピアの全作品に基づいた、文字レベルの小さなテキストデータセット。 データセットは715人のユーザー（シェイクスピア劇のキャラクター）で構成されます。各例は、特定の劇のキャラクターが話す連続した一連の行に対応しています。

- [**CIFAR-100**。](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/cifar100/load_data)500のトレーニングクライアントと100のテストクライアントにわたるCIFAR-100データセットの連合パーティション。各クライアントには100のユニークな例があります。 パーティションは、クライアント間でより現実的な異質性を作成する方法で行われます。 詳細については、[API](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/cifar100/load_data)をご覧ください。

- [**Google Landmark v2 dataset**](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/gldv2/load_data) The dataset consists of photos of various world landmarks, with images grouped by photographer to achieve a federated partitioning of the data. Two flavors of dataset are available: a smaller dataset with 233 clients and 23080 images, and a larger dataset with 1262 clients and 164172 images.

- [**CelebA**](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/celeba/load_data) A dataset of examples (image and facial attributes) of celebrity faces. The federated dataset has each celebrity's examples grouped together to form a client. There are 9343 clients, each with at least 5 examples. The dataset can be split into train and test groups either by clients or by examples.

- [**iNaturalist**](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/inaturalist/load_data) A dataset consists of photos of various species. The dataset contains 120,300 images for 1,203 species. Seven flavors of the dataset are available. One of them is grouped by the photographer and it consists of 9257 clients. The rest of the datasets are grouped by the geo location where the photo was taken. These six flavors of the dataset consists of 11 - 3,606 clients.

## 高性能シミュレーション

While the wall-clock time of an FL *simulation* is not a relevant metric for evaluating algorithms (as simulation hardware isn't representative of real FL deployment environments), being able to run FL simulations quickly is critical for research productivity. Hence, TFF has invested heavily in providing high-performance single and multi-machine runtimes. Documentation is under development, but for now see the [High-performance simulations with TFF](https://www.tensorflow.org/federated/tutorials/simulations) tutorial, instructions on [TFF simulations with accelerators](https://www.tensorflow.org/federated/tutorials/simulations_with_accelerators), and instructions on [setting up simulations with TFF on GCP](https://www.tensorflow.org/federated/gcp_setup). The high-performance TFF runtime is enabled by default.

## さまざまな研究分野の TFF

### 連合最適化アルゴリズム

TFF を利用すると指定するカスタマイズレベルに応じて連合最適化アルゴリズムの研究をさまざまな方法で行うことができます。

A minimal stand-alone implementation of the [Federated Averaging](https://arxiv.org/abs/1602.05629) algorithm is provided [here](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/examples/simple_fedavg). The code includes [TF functions](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/examples/simple_fedavg/simple_fedavg_tf.py) for local computation, [TFF computations](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/examples/simple_fedavg/simple_fedavg_tff.py) for orchestration, and a [driver script](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/examples/simple_fedavg/emnist_fedavg_main.py) on the EMNIST dataset as an example. These files can easily be adapted for customized applciations and algorithmic changes following detailed instructions in the [README](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/examples/simple_fedavg/README.md).

A more general implementation of Federated Averaging can be found [here](https://github.com/google-research/federated/blob/master/optimization/fed_avg_schedule.py). This implementation allows for more sophisticated optimization techniques, including learning rate scheduling and the use of different optimizers on both the server and client. Code that applies this generalized Federated Averaging to various tasks and federated datasets can be found [here](https://github.com/google-research/federated/blob/master/optimization).

### モデルと圧縮の更新

TFF は[ tensor_encoding ](https://github.com/tensorflow/model-optimization/tree/master/tensorflow_model_optimization/python/core/internal/tensor_encoding) API を使用して非可逆圧縮アルゴリズムを有効にし、サーバーとクライアント間の通信コストを削減します。サーバーからクライアントおよびクライアントからサーバーへの [Federated Averaging](https://arxiv.org/abs/1812.07210) アルゴリズムを使用した圧縮のトレーニング例については、[この実験](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/compression/run_experiment.py)をご覧ください。

カスタム圧縮アルゴリズムを実装してトレーニングループに適用するには、次の手順に従います。

1. 新しい圧縮アルゴリズムを[ `EncodingStageInterface ` ](https://github.com/tensorflow/model-optimization/blob/master/tensorflow_model_optimization/python/core/internal/tensor_encoding/core/encoding_stage.py#L75)のサブクラスとして実装します。または、[この例](https://github.com/tensorflow/model-optimization/blob/master/tensorflow_model_optimization/python/core/internal/tensor_encoding/core/encoding_stage.py#L274)のようにより一般的なバリアント<a><code>AdaptiveEncodingStageInterface</code></a>として実装します。
2. 新しい[ `Encoder` ](https://github.com/tensorflow/model-optimization/blob/master/tensorflow_model_optimization/python/core/internal/tensor_encoding/core/core_encoder.py#L38)を作成し、[モデルブロードキャスト](https://github.com/google-research/federated/blob/master/compression/run_experiment.py#L118)または[モデル更新の平均化](https://github.com/google-research/federated/blob/master/compression/run_experiment.py#L144)に特化します。
3. これらのオブジェクトを使用して、[トレーニング計算全体](https://github.com/google-research/federated/blob/master/compression/run_experiment.py#L247)を構築します。

### 差別的なプライバシー

TFF is interoperable with the [TensorFlow Privacy](https://github.com/tensorflow/privacy) library to enable research in new algorithms for federated training of models with differential privacy. For an example of training with DP using [the basic DP-FedAvg algorithm](https://arxiv.org/abs/1710.06963) and [extensions](https://arxiv.org/abs/1812.06210), see [this experiment driver](https://github.com/google-research/federated/blob/master/differential_privacy/stackoverflow/run_federated.py).

If you want to implement a custom DP algorithm and apply it to the aggregate updates of federated averaging, you can implement a new DP mean algorithm as a subclass of [`tensorflow_privacy.DPQuery`](https://github.com/tensorflow/privacy/blob/master/tensorflow_privacy/privacy/dp_query/dp_query.py#L54) and create a `tff.aggregators.DifferentiallyPrivateFactory` with an instance of your query. An example of implementing the [DP-FTRL algorithm](https://arxiv.org/abs/2103.00039) can be found [here](https://github.com/google-research/federated/blob/master/dp_ftrl/dp_fedavg.py)

[以下](#generative_adversarial_networks)で説明されている Federated GAN は、ユーザーレベルの差分プライバシーを実装する TFF プロジェクトの別の例です（例として[このコード](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/gans/tff_gans.py#L293)を参照してください）。

### 堅牢性と攻撃

TFF は、*[「連合学習にバックドアを組み込むことは可能か？」](https://arxiv.org/abs/1911.07963)*で取り上げられているように、連合学習システムへの標的型攻撃とプライバシーに基づく差別化された防御のシミュレーションにも使用できます。これは、潜在的に悪意のあるクライアントとのイテレーションプロセスを構築することにより実行されます。(`build_federated_averaging_process_attacked`を参照してください。) 詳細は [targeted_attack](https://github.com/tensorflow/federated/tree/6477a3dba6e7d852191bfd733f651fad84b82eab/tensorflow_federated/python/research/targeted_attack) ディレクトリをご覧ください。

- 新しい攻撃アルゴリズムは、Tensorflow関数であるクライアント更新関数を記述することにより、実装できます。
- 新しい防御策は、クライアントの出力を集約してグローバル更新を取得する[](https://github.com/tensorflow/federated/blob/6477a3dba6e7d852191bfd733f651fad84b82eab/tensorflow_federated/python/core/utils/computation_utils.py#L103)

シミュレーションのサンプルスクリプトについては、[`emnist_with_targeted_attack.py`](https://github.com/tensorflow/federated/blob/6477a3dba6e7d852191bfd733f651fad84b82eab/tensorflow_federated/python/research/targeted_attack/emnist_with_targeted_attack.py)をご覧ください。

### 生成的敵対的ネットワーク

GANs make for an interesting [federated orchestration pattern](https://github.com/google-research/federated/blob/master/gans/tff_gans.py#L266-L316) that looks a little different than standard Federated Averaging. They involve two distinct networks (the generator and the discriminator) each trained with their own optimization step.

TFF can be used for research on federated training of GANs. For example, the DP-FedAvg-GAN algorithm presented in [recent work](https://arxiv.org/abs/1911.06679) is [implemented in TFF](https://github.com/tensorflow/federated/tree/main/federated_research/gans). This work demonstrates the effectiveness of combining federated learning, generative models, and [differential privacy](#differential_privacy).

### パーソナライゼーション

連合学習の設定におけるパーソナライゼーションは、活発な研究分野です。パーソナライゼーションの目標は、さまざまな推論モデルをさまざまなユーザーに提供することです。この問題に対しては様々なアプローチがあります。

One approach is to let each client fine-tune a single global model (trained using federated learning) with their local data. This approach has connections to meta-learning, see, e.g., [this paper](https://arxiv.org/abs/1909.12488). An example of this approach is given in [`emnist_p13n_main.py`](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/examples/personalization/emnist_p13n_main.py). To explore and compare different personalization strategies, you can:

- Define a personalization strategy by implementing a `tf.function` that starts from an initial model, trains and evaluates a personalized model using each client's local datasets. An example is given by [`build_personalize_fn`](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/examples/personalization/p13n_utils.py).

- 戦略名を対応するパーソナライズ戦略にマップする`OrderedDict`を定義し、それを [`tff.learning.build_personalization_eval`](https://www.tensorflow.org/federated/api_docs/python/tff/learning/build_personalization_eval)の`personalize_fn_dict` 引数として使用します。
