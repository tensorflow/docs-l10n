# インストール

## 安定版

TensorFlow Probability の最新バージョンをインストールします。

<pre class="devsite-terminal devsite-click-to-copy prettyprint lang-shell"> pip install --upgrade tensorflow-probability</pre>

TensorFlow Probability は、[TensorFlow](https://www.tensorflow.org/install) の最近の安定版 (pip パッケージ`tensorflow`) に依存します。TensorFlow と TensorFlow Probability の間の依存関係の詳細については、[TFP リリースノート](https://github.com/tensorflow/probability/releases)をご覧ください。

注意：TensorFlow は、TensorFlow Probability パッケージの依存ファイルとして（<code>setup.py</code> に）<em>含まれていない</em>ため、TensorFlow パッケージ（`tensorflow` または `tensorflow-gpu`）を明示的にインストールする必要があります。そうすると、CPU と GPU 対応の TensorFlow パッケージを個別にではなく、まとめて管理できるようになります。

Python 3 固有のインストールを強制するには、上記のコマンドで`pip`を`pip3`に置き換えます 追加のインストールヘルプ、インストールの前提条件のガイダンス、および（オプションの）仮想環境のセットアップについては、[TensorFlow インストールガイド](https://www.tensorflow.org/install)を参照してください。

## ナイトリービルド

また、pip パッケージ`tfp-nightly`には TensorFlow Probability に対して構築されたナイトリービルドがあります。これは、`tf-nightly`または`tf-nightly-gpu`のいずれかに依存します。ナイトリービルドには新しい機能が含まれていますが、バージョン管理されたリリースよりも安定性が低い場合があります。

## ソースからインストールする

また、ソースからインストールすることもできます。これには、[Bazel](https://bazel.build/){:.external} ビルドシステムが必要です。ソースから TensorFlow Probability をビルドする前に、TensorFlow のナイトリービルド (`tf-nightly`) をインストールすることを強くお勧めします。

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">sudo apt-get install bazel git python-pip</code>
  <code class="devsite-terminal">python -m pip install --upgrade --user tf-nightly</code>
  <code class="devsite-terminal">git clone https://github.com/tensorflow/probability.git</code>
  <code class="devsite-terminal">cd probability</code>
  <code class="devsite-terminal">bazel build --copt=-O3 --copt=-march=native :pip_pkg</code>
  <code class="devsite-terminal">PKGDIR=$(mktemp -d)</code>
  <code class="devsite-terminal">./bazel-bin/pip_pkg $PKGDIR</code>
  <code class="devsite-terminal">python -m pip install --upgrade --user $PKGDIR/*.whl</code>
</pre>

<!-- common_typos_enable -->
