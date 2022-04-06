# TensorFlow Federated をインストールする

TensorFlow Federated（TFF）を使用するための環境は、いくつかの方法でセットアップできます。

- TFF を最も簡単に学習して使用するにはインストールの必要はありません。[Google Colaboratory](https://colab.research.google.com/notebooks/welcome.ipynb) を使用してブラウザで直接 TensorFlow Federated チュートリアルを実行することができます。
- ローカルマシンで TensorFlow Federated を使用するには、Python の`pip` パッケージマネージャを使って [TFF をインストール](#install-tensorflow-federated-using-pip)します。
- 固有のマシン構成を使用する場合は、[ソースから TFF パッケージを構築](#build-the-tensorflow-federated-python-package-from-source)します。

## `pip` を使用して TensorFlow Federated をインストールする

### 1. Python 開発環境をインストールします。

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">sudo apt update</code>
<code class="devsite-terminal">sudo apt install python3-dev python3-pip  # Python 3</code>
</pre>

### 2. 仮想環境を作成します。

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">python3 -m venv "venv"</code>
<code class="devsite-terminal">source "venv/bin/activate"</code>
<code class="devsite-terminal tfo-terminal-venv">pip install --upgrade pip</code>
</pre>

注意: 仮想環境を終了するには、`deactivate` を実行してください。

### 3. リリースされている TensorFlow Federated Python パッケージをインストールします。

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal tfo-terminal-venv">pip install --upgrade tensorflow-federated</code>
</pre>

### 4. Tensorflow Federated をテストします。

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal tfo-terminal-venv">python -c "import tensorflow_federated as tff; print(tff.federated_computation(lambda: 'Hello World')())"</code>
</pre>

成功: 最新の TensorFlow Federated Python パッケージがインストールされました。

## ソースから TensorFlow Federated Python パッケージを構築する

ソースから TensorFlow Federated Python パッケージを構築すると、次の項目を行う場合に役立ちます。

- TensorFlow Federated に変更を適用し、それらの変更が送信またはリリースされる前に、TensorFlow Federated を使用するコンポーネントでテストする。
- TensorFlow Federated に送信されたがリリースされていない変更を使用する。

### 1. Install the Python development environment.

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">sudo apt update</code>
<code class="devsite-terminal">sudo apt install python3-dev python3-pip  # Python 3</code>
</pre>

### 2. Bazel をインストールします。

[Bazel をインストール](https://docs.bazel.build/versions/master/install.html)します。これは、Tensorflow Federated をコンパイルするために使用するビルドツールです。

### 3. TensorFlow Federated リポジトリを複製します。

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">git clone https://github.com/tensorflow/federated.git</code>
<code class="devsite-terminal">cd "federated"</code>
</pre>

### 4. TensorFlow Federated Python パッケージを構築します。

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">mkdir "/tmp/tensorflow_federated"</code>
<code class="devsite-terminal">bazel run //tensorflow_federated/tools/python_package:build_python_package -- \
    --output_dir="/tmp/tensorflow_federated"</code>
</pre>

### 5. 新規プロジェクトを作成します。

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">mkdir "/tmp/project"</code>
<code class="devsite-terminal">cd "/tmp/project"</code>
</pre>

### 6. 仮想環境を作成します。

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">python3 -m venv "venv"</code>
<code class="devsite-terminal">source "venv/bin/activate"</code>
<code class="devsite-terminal tfo-terminal-venv">pip install --upgrade pip</code>
</pre>

注意: 仮想環境を終了するには、`deactivate` を実行してください。

### 7. TensorFlow Federated Python パッケージをインストールします。

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal tfo-terminal-venv">pip install --upgrade "/tmp/tensorflow_federated/"*".whl"</code>
</pre>

### 8. Tensorflow Federated をテストします。

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal tfo-terminal-venv">python -c "import tensorflow_federated as tff; print(tff.federated_computation(lambda: 'Hello World')())"</code>
</pre>

成功: TensorFlow Federated Python パッケージがソースから構築されてインストールされました。
