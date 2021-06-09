# 安装 Neural Structured Learning

要在 TensorFlow 中使用神经结构学习 (NSL)，您可以通过以下几种方式设置环境：

- 学习和使用 NSL 的最简单方式无需安装：使用 [Google Colaboratory](https://colab.research.google.com/notebooks/welcome.ipynb) 直接在浏览器中运行 NSL 教程。
- 要在本地计算机上使用 NSL，请使用 Python 的 `pip` 软件包管理器安装 [NSL 软件包](#install-neural-structured-learning-using-pip)。
- 如果您的计算机配置比较独特，请从源代码[构建 NSL](#build-the-neural-structured-learning-pip-package)。

注：NSL 要求使用 TensorFlow 1.15 或更高版本。NSL 还支持除 v2.1（其中包含与 NSL 不兼容的错误）之外的 TensorFlow 2.x 版本。

## 使用 pip 安装 Neural Structured Learning

#### 1. 安装 Python 开发环境。

在 Ubuntu 上：

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">sudo apt update</code>
<code class="devsite-terminal">sudo apt install python3-dev python3-pip  # Python 3</code>
<code class="devsite-terminal">sudo pip3 install --upgrade virtualenv  # system-wide install</code>
</pre>

在 macOS 上：

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"</code>
<code class="devsite-terminal">export PATH="/usr/local/bin:/usr/local/sbin:$PATH"</code>
<code class="devsite-terminal">brew update</code>
<code class="devsite-terminal">brew install python  # Python 3</code>
<code class="devsite-terminal">sudo pip3 install --upgrade virtualenv  # system-wide install</code>
</pre>

#### 2. 创建虚拟环境。

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">virtualenv --python python3 "./venv"</code>
<code class="devsite-terminal">source "./venv/bin/activate"</code>
<code class="devsite-terminal tfo-terminal-venv">pip install --upgrade pip</code>
</pre>

注：要退出虚拟环境，请运行 `deactivate`。

#### 3. 安装 TensorFlow

CPU 支持：

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal tfo-terminal-venv">pip install 'tensorflow&gt;=1.15.0'</code>
</pre>

GPU 支持：

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal tfo-terminal-venv">pip install 'tensorflow-gpu&gt;=1.15.0'</code>
</pre>

#### 4. 安装 Neural Structured Learning `pip` 软件包。

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal tfo-terminal-venv">pip install --upgrade neural_structured_learning</code>
</pre>

#### 5.（可选）测试 Neural Structured Learning。

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal tfo-terminal-venv">python -c "import neural_structured_learning as nsl"</code>
</pre>

成功：Neural Structured Learning 现已完成安装。

## 构建 Neural Structured Learning pip 软件包

#### 1. 安装 Python 开发环境。

在 Ubuntu 上：

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">sudo apt update</code>
<code class="devsite-terminal">sudo apt install python3-dev python3-pip  # Python 3</code>
<code class="devsite-terminal">sudo pip3 install --upgrade virtualenv  # system-wide install</code>
</pre>

在 macOS 上：

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"</code>
<code class="devsite-terminal">export PATH="/usr/local/bin:/usr/local/sbin:$PATH"</code>
<code class="devsite-terminal">brew update</code>
<code class="devsite-terminal">brew install python  # Python 3</code>
<code class="devsite-terminal">sudo pip3 install --upgrade virtualenv  # system-wide install</code>
</pre>

#### 2. 安装 Bazel。

[安装 Bazel](https://docs.bazel.build/versions/master/install.html)，即用于编译 Neural Structured Learning 的构建工具。

#### 3. 克隆 Neural Structured Learning 仓库。

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">git clone https://github.com/tensorflow/neural-structured-learning.git</code>
</pre>

#### 4. 创建虚拟环境。

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">virtualenv --python python3 "./venv"</code>
<code class="devsite-terminal">source "./venv/bin/activate"</code>
<code class="devsite-terminal tfo-terminal-venv">pip install --upgrade pip</code>
</pre>

注：要退出虚拟环境，请运行 `deactivate`。

#### 5. 安装 Tensorflow

注意，NSL 要求使用 TensorFlow 1.15 或更高版本。NSL 还支持 TensorFlow 2.0。

CPU 支持：

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal tfo-terminal-venv">pip install 'tensorflow&gt;=1.15.0'</code>
</pre>

GPU 支持：

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal tfo-terminal-venv">pip install 'tensorflow-gpu&gt;=1.15.0'</code>
</pre>

#### 6. 安装 Neural Structured Learning 依赖项。

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">cd neural-structured-learning</code>
<code class="devsite-terminal tfo-terminal-venv">pip install --requirement neural_structured_learning/requirements.txt</code>
</pre>

#### 7.（可选）对 Neural Structured Learning 进行单元测试。

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal tfo-terminal-venv">bazel test //neural_structured_learning/...</code>
</pre>

#### 8. 构建 pip 软件包。

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal tfo-terminal-venv">python setup.py bdist_wheel --universal --dist-dir="./wheel"</code>
</pre>

#### 9. 安装 pip 软件包。

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal tfo-terminal-venv">pip install --upgrade ./wheel/neural_structured_learning*.whl</code>
</pre>

#### 10. 测试 Neural Structured Learning。

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal tfo-terminal-venv">python -c "import neural_structured_learning as nsl"</code>
</pre>

成功：Neural Structured Learning 软件包已完成构建。
