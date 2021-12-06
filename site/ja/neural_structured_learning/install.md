# Neural Structured Learning をインストールする

TensorFlow で Neural Structured Learning（NSL）を使用するには、複数の環境設定方法があります。

- 最も簡単な方法で NSL を学習および使用する場合、インストールの必要はありません。[Google Colab](https://colab.research.google.com/notebooks/welcome.ipynb) を使用して、ブラウザで NSL のチュートリアルを実行します。
- NSL をローカルマシンで使用する場合は、Python の`pip`パッケージと共に [NSL パッケージ](#install-neural-structured-learning-using-pip)をインストールします。
- 固有のマシン構成を使用する場合は、ソースから [ NSL を構築](#build-the-neural-structured-learning-pip-package)します。

注意: NSL を使用するには TensorFlow のバージョン 1.15 以上が必要です。NSL は TensorFlow 2.x もサポートしていますが、v2.1 には NSL と互換性のないバグが含まれています。

## pip を使用して Neural Structured Learning をインストールする

#### 1. Python 開発環境をインストールします。

Ubuntu の場合:

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">sudo apt update</code>
<code class="devsite-terminal">sudo apt install python3-dev python3-pip  # Python 3</code>
<code class="devsite-terminal">sudo pip3 install --upgrade virtualenv  # system-wide install</code>
</pre>

macOS の場合:

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"</code>
<code class="devsite-terminal">export PATH="/usr/local/bin:/usr/local/sbin:$PATH"</code>
<code class="devsite-terminal">brew update</code>
<code class="devsite-terminal">brew install python  # Python 3</code>
<code class="devsite-terminal">sudo pip3 install --upgrade virtualenv  # system-wide install</code>
</pre>

#### 2. 仮想環境を作成します。

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">virtualenv --python python3 "./venv"</code>
<code class="devsite-terminal">source "./venv/bin/activate"</code>
<code class="devsite-terminal tfo-terminal-venv">pip install --upgrade pip</code>
</pre>

注意: 仮想環境を終了するには、`deactivate`を実行してください。

#### 3. TensorFlow をインストールします。

CPU サポート:

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal tfo-terminal-venv">pip install 'tensorflow&gt;=1.15.0'</code>
</pre>

GPU サポート:

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal tfo-terminal-venv">pip install 'tensorflow-gpu&gt;=1.15.0'</code>
</pre>

#### 4. Neural Structured Learning `pip` パッケージをインストールします。

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal tfo-terminal-venv">pip install --upgrade neural_structured_learning</code>
</pre>

#### 5.（オプション）Neural Structured Learning をテストします。

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal tfo-terminal-venv">python -c "import neural_structured_learning as nsl"</code>
</pre>

成功: Neural Structured Learning がインストールされました。

## Neural Structured Learning pip パッケージを構築する

#### 1. Python 開発環境をインストールします。

Ubuntu の場合:

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">sudo apt update</code>
<code class="devsite-terminal">sudo apt install python3-dev python3-pip  # Python 3</code>
<code class="devsite-terminal">sudo pip3 install --upgrade virtualenv  # system-wide install</code>
</pre>

macOS の場合:

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"</code>
<code class="devsite-terminal">export PATH="/usr/local/bin:/usr/local/sbin:$PATH"</code>
<code class="devsite-terminal">brew update</code>
<code class="devsite-terminal">brew install python  # Python 3</code>
<code class="devsite-terminal">sudo pip3 install --upgrade virtualenv  # system-wide install</code>
</pre>

#### 2. Bazel をインストールします。

Neural Structured Learning のコンパイルに使用した構築ツール [Bazel をインストール](https://docs.bazel.build/versions/master/install.html)します。

#### 3. Neural Structured Learning のリポジトリをクローンします。

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">git clone https://github.com/tensorflow/neural-structured-learning.git</code>
</pre>

#### 4. 仮想環境を作成します。

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">virtualenv --python python3 "./venv"</code>
<code class="devsite-terminal">source "./venv/bin/activate"</code>
<code class="devsite-terminal tfo-terminal-venv">pip install --upgrade pip</code>
</pre>

注意: 仮想環境を終了するには、`deactivate`を実行してください。

#### 5. TensorFlow をインストールします。

NSL を使用するには、TensorFlow のバージョン1.15 以上が必要なことに注意してください。NSL は TensorFlow 2.0 もサポートしています。

CPU サポート:

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal tfo-terminal-venv">pip install 'tensorflow&gt;=1.15.0'</code>
</pre>

GPU サポート:

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal tfo-terminal-venv">pip install 'tensorflow-gpu&gt;=1.15.0'</code>
</pre>

#### 6. Neural Structured Learning の依存性をインストールします。

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">cd neural-structured-learning</code>
<code class="devsite-terminal tfo-terminal-venv">pip install --requirement neural_structured_learning/requirements.txt</code>
</pre>

#### 7.（オプション）Neural Structured Learning の単体テストをします。

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal tfo-terminal-venv">bazel test //neural_structured_learning/...</code>
</pre>

#### 8. pip パッケージを構築します。

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal tfo-terminal-venv">python setup.py bdist_wheel --universal --dist-dir="./wheel"</code>
</pre>

#### 9. pip パッケージをインストールします。

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal tfo-terminal-venv">pip install --upgrade ./wheel/neural_structured_learning*.whl</code>
</pre>

#### 10. Neural Structured Learning をテストします。

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal tfo-terminal-venv">python -c "import neural_structured_learning as nsl"</code>
</pre>

成功: Neural Structured Learning パッケージが構築されました。
