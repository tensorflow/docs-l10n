# 安装 TensorFlow Federated

您可以通过多种方式设置环境来使用 TensorFlow Federated (TFF)：

- 学习和使用 TFF 的最简单方式无需安装——您可以使用 [Google Colaboratory](https://colab.research.google.com/notebooks/welcome.ipynb) 直接在浏览器中运行 TensorFlow Federated 教程。
- 要在本地计算机上使用 TensorFlow Federated，请使用 Python 的 <code>pip</code> 软件包管理器安装 <a>TFF 软件包</a>。
- 如果您的计算机配置比较独特，则可以从源代码[构建 TFF 软件包](#build-the-tensorflow-federated-pip-package)。

## 使用 `pip` 安装 TensorFlow Federated

### 1. 安装 Python 开发环境。

在 Ubuntu 上：

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">sudo apt update</code>
<code class="devsite-terminal">sudo apt install python3-dev python3-pip  # Python 3</code>
<code class="devsite-terminal">sudo pip3 install --user --upgrade virtualenv</code>
</pre>

在 macOS 上：

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"</code>
<code class="devsite-terminal">export PATH="/usr/local/bin:/usr/local/sbin:$PATH"</code>
<code class="devsite-terminal">brew update</code>
<code class="devsite-terminal">brew install python  # Python 3</code>
<code class="devsite-terminal">sudo pip3 install --user --upgrade virtualenv</code>
</pre>

### 2. 创建虚拟环境。

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">virtualenv --python python3 "venv"</code>
<code class="devsite-terminal">source "venv/bin/activate"</code>
<code class="devsite-terminal tfo-terminal-venv">pip install --upgrade pip</code>
</pre>

注：要退出虚拟环境，请运行 `deactivate`。

### 3. 安装 TensorFlow Federated Python 软件包。

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal tfo-terminal-venv">pip install --upgrade tensorflow_federated</code>
</pre>

### 4. 测试 Tensorflow Federated。

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal tfo-terminal-venv">python -c "import tensorflow_federated as tff; print(tff.federated_computation(lambda: 'Hello World')())"</code>
</pre>

成功：现已安装最新的 TensorFlow Federated Python 软件包。

## 从源代码构建 TensorFlow Federated Python 软件包

当您想要执行以下操作时，从源代码构建 TensorFlow Federated Python 软件包会很有帮助：

- 更改 TensorFlow Federated，并在提交或发布更改之前，先在使用 TensorFlow Federated 的组件中测试这些更改。
- 使用已提交到 TensorFlow Federated 但未发布的更改。

### 1. 安装 Python 开发环境。

在 Ubuntu 上：

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">sudo apt update</code>
<code class="devsite-terminal">sudo apt install python3-dev python3-pip  # Python 3</code>
<code class="devsite-terminal">sudo pip3 install --user --upgrade virtualenv</code>
</pre>

在 macOS 上：

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"</code>
<code class="devsite-terminal">export PATH="/usr/local/bin:/usr/local/sbin:$PATH"</code>
<code class="devsite-terminal">brew update</code>
<code class="devsite-terminal">brew install python  # Python 3</code>
<code class="devsite-terminal">sudo pip3 install --user --upgrade virtualenv</code>
</pre>

### 2. 安装 Bazel。

[安装 Bazel](https://docs.bazel.build/versions/master/install.html)，即用于编译 Tensorflow Federated 的构建工具。

### 3. 克隆 Tensorflow Federated 仓库。

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">git clone https://github.com/tensorflow/federated.git</code>
<code class="devsite-terminal">cd "federated"</code>
</pre>

### 4. 构建 TensorFlow Federated Python 软件包。

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">mkdir "/tmp/tensorflow_federated"</code>
<code class="devsite-terminal">bazel run //tensorflow_federated/tools/development:build_pip_package -- \
    --nightly \
    --output_dir "/tmp/tensorflow_federated"</code>
</pre>

### 5. 创建新项目。

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">mkdir "/tmp/project"</code>
<code class="devsite-terminal">cd "/tmp/project"</code>
</pre>

### 6. 创建虚拟环境。

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">virtualenv --python python3 "venv"</code>
<code class="devsite-terminal">source "venv/bin/activate"</code>
<code class="devsite-terminal tfo-terminal-venv">pip install --upgrade pip</code>
</pre>

注：要退出虚拟环境，请运行 `deactivate`。

### 7. 安装 TensorFlow Federated Python 软件包。

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal tfo-terminal-venv">pip install --upgrade "/tmp/tensorflow_federated/"*".whl"</code>
</pre>

### 8. 测试 Tensorflow Federated。

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal tfo-terminal-venv">python -c "import tensorflow_federated as tff; print(tff.federated_computation(lambda: 'Hello World')())"</code>
</pre>

成功：现已从源代码构建和安装 TensorFlow Federated Python 软件包。
