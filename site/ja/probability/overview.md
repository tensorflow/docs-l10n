# TensorFlow Probability

TensorFlow Probability は、TensorFlow の確率論的推論と統計分析のためのライブラリです。TensorFlow エコシステムの一部として、TensorFlow Probability は、確率論的手法とディープネットワークの統合、自動微分を使用した勾配ベースの推論、ハードウェアアクセラレーション（GPU）と分散計算を備えた大規模なデータセットとモデルへのスケーラビリティを提供します。

TensorFlow Probability をはじめるには、[インストールガイド](./install)を参照し、[Python ノートブックチュートリアル](https://github.com/tensorflow/probability/blob/main/tensorflow_probability/examples/jupyter_notebooks/){:.external} を参照してください。

## コンポーネント

確率的機械学習ツールは次のように構成されています。

### Layer 0: TensorFlow

*数値演算*、特に `LinearOperator` クラスは、効率的な計算のために特定の構造（対角、低ランクなど）を活用できる行列のない実装を可能にします。TensorFlow Probability チームによって構築および保守されており、コア TensorFlow の [`tf.linalg`](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/python/ops/linalg) の一部です。

### Layer 1: 確率的なブロックの構築

- *Distributions* ([`tfp.distributions`](https://github.com/tensorflow/probability/tree/main/tensorflow_probability/python/distributions)): バッチおよび[ブロードキャスト](https://docs.scipy.org/doc/numpy-1.14.0/user/basics.broadcasting.html){:.external} セマンティクスを使用した確率分布および関連する統計の大規模なコレクション。
- *Bijectors* ([`tfp.bijectors`](https://github.com/tensorflow/probability/tree/main/tensorflow_probability/python/bijectors)): 確率変数の可逆的で構成可能な変換。Bijectors は、[対数正規分布](https://en.wikipedia.org/wiki/Log-normal_distribution){:.external} のような古典的な例から、[マスクされた自己回帰フロー](https://arxiv.org/abs/1705.07057){:.external} のような高度な深層学習モデルまで、変換された分布の豊富なクラスを提供します。

### Layer 2: モデル構築

- 同時分布 ([`tfp.distributions.JointDistributionSequential`](https://github.com/tensorflow/probability/tree/main/tensorflow_probability/python/distributions/joint_distribution_sequential.py)): 1 つ以上の相互依存の可能性のある分布にわたる同時分布。TFP の `JointDistribution` を使用したモデリングの概要については、[このコラボ](https://github.com/tensorflow/probability/blob/main/tensorflow_probability/examples/jupyter_notebooks/Modeling_with_JointDistribution.ipynb)を参照してください。
- *確率的レイヤー* ([`tfp.layers`](https://github.com/tensorflow/probability/tree/main/tensorflow_probability/python/layers)): 不確実性を考慮した関数をもつニューラルネットワークレイヤ。TensorFlow レイヤーを拡張します。

### Layer 3: 確率的推論

- *マルコフ連鎖モンテカルロ* ([`tfp.mcmc`](https://github.com/tensorflow/probability/tree/main/tensorflow_probability/python/mcmc)): サンプリングを介して積分を近似するためのアルゴリズム。[ハミルトニアンモンテカルロ](https://en.wikipedia.org/wiki/Hamiltonian_Monte_Carlo){:. external}、ランダムウォークメトロポリス-ヘイスティングス、およびカスタム遷移カーネルを構築する機能が含まれています。
- *変分推論* ([`tfp.vi`](https://github.com/tensorflow/probability/tree/main/tensorflow_probability/python/vi)): 最適化により積分を近似するためのアルゴリズム。
- *オプティマイザ* ([`tfp.optimizer`](https://github.com/tensorflow/probability/tree/main/tensorflow_probability/python/optimizer)): TensorFlow オプティマイザを拡張する確率的最適化手法。[確率的勾配ランゲビンダイナミクス](http://www.icml-2011.org/papers/398_icmlpaper.pdf){:.external} が含まれています。
- *モンテカルロ* ([`tfp.monte_carlo`](https://github.com/tensorflow/probability/blob/main/tensorflow_probability/python/monte_carlo)): モンテカルロ法で期待値を計算するためのツール。

TensorFlow Probability は開発中であり、インターフェースが変更される可能性があります。

## 使用例

ナビゲーションにリストされている [Python ノートブックチュートリアル](https://github.com/tensorflow/probability/blob/main/tensorflow_probability/examples/jupyter_notebooks/){:.external} の他、使用可能なスクリプトの例がいくつかあります。

- [変分オートエンコーダー](https://github.com/tensorflow/probability/tree/main/tensorflow_probability/examples/vae.py) — 潜在コードと変分推論による表現学習。
- [ベクトル量子化オートエンコーダ](https://github.com/tensorflow/probability/tree/main/tensorflow_probability/examples/vq_vae.py) — ベクトル量子化による離散表現学習。
- [ベイジアンニューラルネットワーク](https://github.com/tensorflow/probability/tree/main/tensorflow_probability/examples/bayesian_neural_network.py) — 重みに不確実性があるニューラルネットワーク。
- [ベイズロジスティック回帰](https://github.com/tensorflow/probability/tree/main/tensorflow_probability/examples/logistic_regression.py) — 二項分類のためのベイズ推定。

## 問題の報告

バグの報告や機能リクエストには、 [TensorFlow Probability 課題トラッカー](https://github.com/tensorflow/probability/issues)を使用してください。
