# TensorFlow Quantum のインストール

TensorFlow Quantum (TFQ) を使用するために環境をセットアップするには、以下のような方法があります。

- TFQ を学習および使用する最も簡単な方法は、[Google Colab](./tutorials/hello_many_worlds.ipynb) を使用することです。ブラウザで直接 [TensorFlow Quantum チュートリアル](https://colab.research.google.com/github/tensorflow/quantum/blob/master/docs/tutorials/hello_many_worlds.ipynb)を実行します。インストールする必要はありません。
- ローカルマシンで TensorFlowQuantum を使用するには、Python の pip パッケージマネージャを使用して TFQ パッケージをインストールします。
- または、ソースから TensorFlow Quantum を構築します。

TensorFlow Quantum は Python  3.7、3.8 および 3.9 でサポートされています。また、[Circq](https://github.com/quantumlib/Cirq) に直接依存しています。

## Pip パッケージ

### 要件

- pip 19.0 以降 (`manylinux2010`サポートが必要)
- [TensorFlow == 2.7.0](https://www.tensorflow.org/install/pip)

Python 開発環境と（オプションの）仮想環境をセットアップするには、[TensorFlow インストールガイド](https://www.tensorflow.org/install/pip)を参照してください。

`pip`をアップグレードして TensorFlow をインストールします。

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">pip3 install --upgrade pip</code>
  <code class="devsite-terminal">pip3 install tensorflow==2.7.0</code>
</pre>

<!-- common_typos_enable -->

### パッケージをインストールする

TensorFlow Quantum の最新のステーブル版をインストールします。

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">pip3 install -U tensorflow-quantum</code>
</pre>

<!-- common_typos_enable -->

成功: TensorFlow Quantum がインストールされました。

より新しいバージョンの TensorFlow に依存している可能性のある Nightly ビルドは、次のようにしてインストールできます。

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">pip3 install -U tfq-nightly</code>
</pre>

<!-- common_typos_enable -->

## ソースから構築する

次の手順は、Ubuntu のようなシステムでテストされています。

### 1. Python 3 開発環境をセットアップする

まず、Python 3.8 開発ツールが必要です。

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">sudo apt update</code>
  <code class="devsite-terminal">sudo apt-get install pkg-config zip g++ zlib1g-dev unzip python3.8</code>
  <code class="devsite-terminal">sudo apt install python3.8 python3.8-dev python3.8-venv python3-pip</code>
  <code class="devsite-terminal">python3.8 -m pip install --upgrade pip</code>
</pre>

<!-- common_typos_enable -->

### 2. 仮想環境を作成する

ワークスペースディレクトリに移動し、TFQ 開発用の仮想環境を作成します。

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">python3.8 -m venv quantum_env</code>
  <code class="devsite-terminal">source quantum_env/bin/activate</code>
</pre>

<!-- common_typos_enable -->

### 3. Bazel をインストールする

TensorFlow の[ソースから構築する](https://www.tensorflow.org/install/source#install_bazel)ガイドに記載されているように、<a href="https://bazel.build/" class="external">Bazel </a>ビルドシステムが必要になります。

最新のソースビルドには、TensorFlow 2.7.0 が使用されています。互換性を確保するために、`bazel` バージョン 3.7.2 を使用しています。Bazel の既存のバージョンを削除するには、以下を実行します。

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">sudo apt-get remove bazel</code>
</pre>

<!-- common_typos_enable -->

`bazel` バージョン 3.7.2 をダウンロードしてインストールします。

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">wget https://github.com/bazelbuild/bazel/releases/download/3.7.2/bazel_3.7.2-linux-x86_64.deb
</code>
  <code class="devsite-terminal">sudo dpkg -i bazel_3.7.2-linux-x86_64.deb</code>
</pre>

<!-- common_typos_enable -->

`bazel`が互換性のないバージョンに自動更新されないようにするには、次のコマンドを実行します。

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">sudo apt-mark hold bazel</code>
</pre>

<!-- common_typos_enable -->

最後に、正しい`bazel`バージョンのインストールを確認します。

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">bazel --version</code>
</pre>

<!-- common_typos_enable -->

### 4. ソースから TensorFlow Quantum を構築する

ここでは、TensorFlow [ソースから構築する](https://www.tensorflow.org/install/source)ガイドからの説明を使用しました。詳細はリンクをご覧ください。TensorFlow Quantum は、TensorFlow バージョン 2.7.0 と互換性があります。

<a href="https://github.com/tensorflow/tensorflow" class="external">TensorFlow ソースコード</a>をダウンロードします。

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">git clone https://github.com/tensorflow/tensorflow.git</code>
  <code class="devsite-terminal">cd tensorflow</code>
  <code class="devsite-terminal">git checkout v2.7.0</code>
</pre>

ステップ 2 で作成した仮想環境がアクティブになっていることを確認します。次に、TensorFlow の依存関係をインストールします。

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">pip install -U pip six numpy wheel setuptools mock 'future&gt;=0.17.1'</code>
  <code class="devsite-terminal">pip install -U keras_applications --no-deps</code>
  <code class="devsite-terminal">pip install -U keras_preprocessing --no-deps</code>
  <code class="devsite-terminal">pip install numpy==1.19.5</code>
</pre>

<!-- common_typos_enable -->

TensorFlow ビルドを構成します。Python インタープリタとライブラリの場所は、必ず仮想環境フォルダ内の場所に指定します。残りのオプションはデフォルト値のままにしておくことができます。

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">./configure</code>
</pre>

<!-- common_typos_enable -->

TensorFlow パッケージを構築します。

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">bazel build -c opt --cxxopt="-O3" --cxxopt="-march=native" --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" //tensorflow/tools/pip_package:build_pip_package</code>
</pre>

<!-- common_typos_enable -->

注：パッケージの構築には 1 時間以上かかる場合があります。

ビルドが完了したら、パッケージをインストールし、TensorFlow ディレクトリを離れます。

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg</code>
  <code class="devsite-terminal">pip install /tmp/tensorflow_pkg/&lt;var&gt;name_of_generated_wheel&lt;/var&gt;.whl</code>
  <code class="devsite-terminal">cd ..</code>
</pre>

<!-- common_typos_enable -->

### 5. TensorFlow Quantum をダウンロードする

貢献には、標準の[フォークとプルリクエストのワークフロー](https://guides.github.com/activities/forking/)を使用します。[TensorFlow Quantum](https://github.com/tensorflow/quantum) GitHub ページからフォークした後、フォークのソースをダウンロードして、要件をインストールします。

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">git clone https://github.com/&lt;var&gt;username&lt;/var&gt;/quantum.git</code>
  <code class="devsite-terminal">cd quantum</code>
  <code class="devsite-terminal">pip install -r requirements.txt</code>
</pre>

<!-- common_typos_enable -->

### 6. TensorFlow Quantum pip パッケージを構築する

TensorFlow Quantum pip パッケージを構築し、以下をインストールします。

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">./configure.sh</code>
  <code class="devsite-terminal">bazel build -c opt --cxxopt="-O3" --cxxopt="-march=native" --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" release:build_pip_package</code>
  <code class="devsite-terminal">bazel-bin/release/build_pip_package /tmp/tfquantum/</code>
  <code class="devsite-terminal">python3 -m pip install /tmp/tfquantum/&lt;var&gt;name_of_generated_wheel&lt;/var&gt;.whl</code>
</pre>

<!-- common_typos_enable -->

TensorFlow Quantum が正常にインストールされたことを確認するには、次のテストを実行します。

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">./scripts/test_all.sh</code>
</pre>

<!-- common_typos_enable -->

成功: TensorFlow Quantum がインストールされました。
