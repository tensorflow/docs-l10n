<div align="center">   <img src="https://tensorflow.org/images/SIGAddons.png" class=""><br><br> </div>

---

# TensorFlow Addons

**TensorFlow Addons（アドオン）** は、貢献リポジトリであり、確立された API パターンに準拠しながらコアの TensorFlow では提供されていない新しい機能を実装します。TensorFlow では、多数の演算子、レイヤー、メトリクス、損失、オプティマイザなどがネイティブにサポートされています。しかし、機械学習のような変化の急激な分野では、（広く適用できるかまだ明らかでなかったり、大部分がコミュニティのごく一部でしか使用されなかったりするため）興味深いけれどもコアの TensorFlow に統合できない、新しい開発事項が多数あります。

## インストール

#### 安定したビルド

最新バージョンをインストールするには、次のように実行します。

```
pip install tensorflow-addons
```

アドオンを使用するには以下のようにします。

```python
import tensorflow as tf
import tensorflow_addons as tfa
```

#### ナイトリービルド

pip パッケージ`tfa-nightly`には、 TensorFlow の最新の安定バージョンに対して構築された、TensorFlow Addons のナイトリービルドもあります。ナイトリービルドには新しい機能が含まれていますが、バージョン管理されているリリース版よりも安定性が低い場合があります。

```
pip install tfa-nightly
```

#### ソースからインストールする

ソースからインストールすることも可能です。これには [Bazel](https://bazel.build/) 構築システムが必要です。

```
git clone https://github.com/tensorflow/addons.git
cd addons

# If building GPU Ops (Requires CUDA 10.0 and CuDNN 7)
export TF_NEED_CUDA=1
export CUDA_HOME="/path/to/cuda10" (default: /usr/local/cuda)
export CUDNN_INSTALL_PATH="/path/to/cudnn" (default: /usr/lib/x86_64-linux-gnu)

# This script links project with TensorFlow dependency
python3 ./configure.py

bazel build build_pip_pkg
bazel-bin/build_pip_pkg artifacts

pip install artifacts/tensorflow_addons-*.whl
```

## コアコンセプト

#### サブパッケージ内で標準化された API

ユーザーエクスペリエンスとプロジェクトの保守性は、TensorFlow Addons の核（コア）となるコンセプトです。これを実現するために、追加事項はコアの TensorFlow で確立されている API パターンに準拠しなければなりません。

#### GPU/CPU カスタム演算

TensorFlow Addons の大きな利点は、プリコンパイル済みの演算があることです。CUDA 10 のインストールが見つからない場合、演算は自動的に CPU 実装にフォールバックします。

#### プロキシ メンテナシップ

アドオンはサブパッケージとサブモジュールをコンパートメント化し、そのコンポーネントの専門知識と強い関心を持ったユーザーがメンテナンスできるように設計されています。

サブパッケージのメンテナシップは、書き込み権限を持つユーザー数を制限するために、かなりの貢献がなされた後にのみ付与されます。貢献の形は、問題解決、バグ修正、ドキュメンテーション、新しいコード、既存コードの最適化などです。サブモジュールのメンテナシップにはレポへの書き込み権限が含まれていないため、 低めのハードルで参加が認められる場合もあります。

このトピックのさらに詳しい情報については [RFC](https://github.com/tensorflow/community/blob/master/rfcs/20190308-addons-proxy-maintainership.md) をご覧ください。

#### サブパッケージの定期的評価

このリポジトリの性質上、サブパッケージやサブモジュールは時間が経つにつれ、コミュニティに有用でなくなっていく可能性があります。 リポジトリを持続可能な状態に保つため、半年ごとにコードのレビューを行い、すべてがまだリポジトリに属していることを確認しています。 このレビューに寄与する要素は次のとおりです。

1. アクティブなメンテナの数
2. OSS の利用量
3. コードに起因する問題やバグの量
4. もっと良いソリューションが利用可能かどうか

TensorFlow Addons 内の機能は、3 つのグループに分類することができます。

- **推奨**: よく整備されている API です。使用が推奨されます。
- **推奨しない**: もっと良い代替案が利用可能です。API が歴史的な理由で保持されている、API がメンテナンスを必要としている、非推奨になるまでの待機期間、などの理由です。
- **非推奨**: 自己責任でご使用ください。これは削除対象です。

この 3 つのグループ間のステータス変化は次の通りです。推奨 <-> 推奨しない -> 非推奨

API が非推奨とマークされてから削除されるまでの期間は 90 日間です。その理由は以下の通りです。

1. TensorFlow Addon が毎月リリースされる場合、当該 API が削除されるまでに 2 ～ 3 回のリリースがあることになります。リリースノートでユーザーに十分な警告を与えることができます。

2. 90 日間で、コード修正に充分な時間をメンテナに与えることができます。

## 貢献する

TensorFlow Addon は、コミュニティ主導のオープンソースプロジェクトです。そのため、このプロジェクトは一般の方による貢献、バグ修正、ドキュメンテーションに依存しています。貢献の仕方については[貢献ガイドライン](https://github.com/tensorflow/addons/blob/master/CONTRIBUTING.md)をご覧ください。また、このプロジェクトは [TensorFlow の行動規範](https://github.com/tensorflow/addons/blob/master/CODE_OF_CONDUCT.md)に従っています。参加することにより、この行動規範の遵守が期待されます。

## コミュニティ

- [一般メーリングリスト](https://groups.google.com/a/tensorflow.org/forum/#!forum/addons)
- [SIG 月例会議録](https://docs.google.com/document/d/1kxg5xIHWLY7EMdOJCdSGgaPu27a9YKpupUz2VTXqTJg)
    - メーリングリストに参加すると、会議へのカレンダー招待状を受け取ることができます

## ライセンス

[Apache License 2.0](LICENSE)
