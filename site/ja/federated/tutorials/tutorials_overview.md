# TensorFlow Federated チュートリアル

これらの [colab ベースの](https://colab.research.google.com/)チュートリアルでは、実際の例を使用して、TFF の主な概念と API について説明します。詳細については[TFF ガイド](../get_started.md)を参照してください。

注意: TFF では現在 Python 3.9 以降が必要ですが、[Google Colaboratory](https://research.google.com/colaboratory/)のホストランタイムは現在  Python 3.7 を使用しているため、これらのノートブックを実行するには、[カスタムローカルランタイム](https://research.google.com/colaboratory/local-runtimes.html)を使用する必要があります。

**連合学習を始める**

- [画像分類の連合学習](federated_learning_for_image_classification.ipynb): 連合学習（FL）API の主要部分を紹介し、TFF を使用して、MNIST のような連合データで連合学習をシミュレーションする方法を実演します。
- [テキスト生成の連合学習](federated_learning_for_text_generation.ipynb): TFF の FL API を使用して、言語モデリングタスク用にシリアル化されたトレーニング済みのモデルを洗練する方法を実演します。
- [学習に推奨される集計の調整](tuning_recommended_aggregators.ipynb): `tff.learning` の基本的な FL 計算を、堅牢性、差分プライバシー、圧縮などを提供する特殊な集計ルーチンと組み合わせる方法を実演します。
- [行列因数分解のための連合再構成](federated_reconstruction_for_matrix_factorization.ipynb): 一部のクライアントパラメータがサーバーで集約されない、部分的にローカルな連合学習を紹介します。このチュートリアルでは、連合学習 API を使用して、部分的にローカルな行列因数分解モデルをトレーニングする方法を実演します。

**連合分析を始める**

- [プライベートヘビーヒッター](private_heavy_hitters.ipynb): `tff.analytics.heavy_hitters` を使用して、連合分析計算を構築し、プライベートヘビーヒッターを検出する方法を実演します。

**カスタム連合計算の記述**

- [独自の連合学習アルゴリズムの構築](building_your_own_federated_learning_algorithm.ipynb): 例として、連合学習アルゴリズムを使用して、TFF Core API を使用して連合学習アルゴリズムを実装する方法を実演します。
- [学習アルゴリズムの作成](composing_learning_algorithms.ipynb): TFF Learning API を使用して、新しい連合学習アルゴリズム、特に連合平均のバリアントを簡単に実装する方法を実演します。
- [TFF オプティマイザーを使用したカスタム連合アルゴリズム](custom_federated_algorithm_with_tff_optimizers.ipynb): `tff.learning.optimizers` を使用して、連合平均化のカスタム反復プロセスを構築する方法を実演します。
- [カスタム連合アルゴリズム、パート 1: 連合コアの概要](custom_federated_algorithms_1.ipynb) および [パート 2: 連合平均化の実装](custom_federated_algorithms_2.ipynb): Federated Core API（FC API）が提供する主要な概念とインターフェースを紹介します。
- [カスタム集計の実装](custom_aggregators.ipynb): <code>tff.aggregators</code> モジュールのデザイン原理とクライアントからサーバーへの値のカスタム集約を実装するためのベストプラクティスについて説明します。

**シミュレーションのベストプラクティス**

- [Kubernetes を使用した高性能シミュレーション](high_performance_simulation_with_kubernetes.ipynb): Kubernetes で実行される高性能 TFF ランタイムをセットアップおよび構成する方法について説明します。

- [アクセラレータを使用した TFF シミュレーション（GPU）](simulations_with_accelerators.ipynb): TFF の高性能ランタイムを GPU で使用する方法を実演します。

- [ClientData の使用](working_with_client_data.ipynb):  TFF の [ClientData](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/ClientData) ベースのシミュレーションデータセットを TFF 計算に統合するためのベストプラクティスを示します。

**中級および上級チュートリアル**

- [ランダムノイズの生成](random_noise_generation.ipynb): 分散型計算でランダム性を使用する際のいくつかの詳細な点を指摘し、ベストプラクティスと推薦されるパターンを提案します。

- [tff.federated_select を使用して特定のクライアントに異なるデータを送信する](federated_select.ipynb): `tff.federated_select` 演算子を紹介し、さまざまなデータをさまざまなクライアントに送信するカスタム連合アルゴリズムの簡単な例を実演します。

- [federated_select とスパースアグリゲーションによるクライアント効率の高い大規模モデル連合学習](sparse_federated_learning.ipynb): TFF を使用して非常に大規模なモデルをトレーニングする方法を示します。各クライアントデバイスは、`tff.federated_select` とスパースアグリゲーションを使用して、モデルの一部のみをダウンロードおよび更新します。

- [連合学習研究のための TFF: モデルと更新の圧縮](tff_for_federated_learning_research_compression.ipynb): [tensor_encoding API](https://github.com/tensorflow/model-optimization/tree/master/tensorflow_model_optimization/python/core/internal/tensor_encoding) に基づいて構築されたカスタム集計を TFF で使用する方法を示します。

- [TFF での差分プライバシーによる連合学習](federated_learning_with_differential_privacy.ipynb): TFF を使用して、ユーザーレベルの差分プライバシーを使用してモデルをトレーニングする方法を示します。

- [TFF を使用したリモートデータの読み込み](loading_remote_data.ipynb): TFF ランタイムにカスタムロジックを埋め込み、リモートマシンにデータを読み込む方法について説明します。

- [TFF での JAX のサポート](../tutorials/jax_support.ipynb): TFF で[ JAX ](https://github.com/google/jax)計算を使用する方法を示し、TFF が他のフロントエンドおよびバックエンドの  ML フレームワークと相互運用できるようにどのように設計されているかを示します。
