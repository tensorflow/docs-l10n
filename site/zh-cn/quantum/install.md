# 安装 TensorFlow Quantum

要使用 TensorFlow Quantum (TFQ)，您可通过以下几种方式设置环境：

- 学习和使用 TFQ 的最简单方式无需安装：使用 [Google Colab](./tutorials/hello_many_worlds.ipynb) 直接在浏览器中运行 [TensorFlow Quantum 教程](https://colab.research.google.com/github/tensorflow/quantum/blob/master/docs/tutorials/hello_many_worlds.ipynb)。
- 要在本地计算机上使用 TensorFlow Quantum，请使用 Python 的 pip 软件包管理器安装 TFQ 软件包。
- 或者从源代码构建 TensorFlow Quantum。

TensorFlow Quantum 在 Python 3.6、3.7 和 3.8 上受支持，并且直接依赖于 [Cirq](https://github.com/quantumlib/Cirq)。

## Pip 软件包

### 要求

- pip 19.0 或更高版本（需要 `manylinux2010` 支持）
- [TensorFlow == 2.5.1](https://www.tensorflow.org/install/pip)

要设置您的 Python 开发环境和一个（可选的）虚拟环境，请参阅 [TensorFlow 安装指南](https://www.tensorflow.org/install/pip)。

升级 `pip` 并安装 TensorFlow

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">pip3 install --upgrade pip</code>
  <code class="devsite-terminal">pip3 install tensorflow==2.5.1</code>
</pre>

<!-- common_typos_enable -->

### 安装软件包

安装最新稳定版本的 TensorFlow Quantum：

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">pip3 install -U tensorflow-quantum</code>
</pre>

<!-- common_typos_enable -->

成功：TensorFlow Quantum 现已完成安装。

安装最新 Nightly 版本的 TensorFlow Quantum：

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">pip3 install -U tfq-nightly</code>
</pre>

<!-- common_typos_enable -->

## 从源代码构建

以下步骤已针对与 Ubuntu 类似的系统进行测试。

### 1. 设置 Python 3 开发环境

首先，我们需要 Python 3.8 开发工具。

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">sudo apt update</code>
  <code class="devsite-terminal">sudo apt-get install pkg-config zip g++ zlib1g-dev unzip python3.8</code>
  <code class="devsite-terminal">sudo apt install python3.8 python3.8-dev python3.8-venv python3-pip</code>
  <code class="devsite-terminal">python3.8 -m pip install --upgrade pip</code>
</pre>

<!-- common_typos_enable -->

### 2. 创建虚拟环境

转到您的工作区目录并为 TFQ 开发创建一个虚拟环境。

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">python3.8 -m venv quantum_env</code>
  <code class="devsite-terminal">source quantum_env/bin/activate</code>
</pre>

<!-- common_typos_enable -->

### 3. 安装 Bazel

如 TensorFlow [从源代码构建](https://www.tensorflow.org/install/source#install_bazel)指南中所述，将需要 <a href="https://bazel.build/" class="external">Bazel</a> 构建系统。

我们最新的源代码构建使用 TensorFlow 2.5.1。为确保兼容性，我们使用 `bazel` 版本 3.7.2。要移除任何现有版本的 Bazel，请运行以下命令：

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">sudo apt-get remove bazel</code>
</pre>

<!-- common_typos_enable -->

下载并安装 `bazel` 版本 3.7.2：

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">wget https://github.com/bazelbuild/bazel/releases/download/3.7.2/bazel_3.7.2-linux-x86_64.deb
</code>
  <code class="devsite-terminal">sudo dpkg -i bazel_3.7.2-linux-x86_64.deb</code>
</pre>

<!-- common_typos_enable -->

为防止 `bazel` 自动更新到不兼容的版本，请运行以下命令：

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">sudo apt-mark hold bazel</code>
</pre>

<!-- common_typos_enable -->

最后，确认已安装正确的 `bazel` 版本：

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">bazel --version</code>
</pre>

<!-- common_typos_enable -->

### 4. 从源代码构建 TensorFlow

在这里，我们改编了 TensorFlow [从源代码构建](https://www.tensorflow.org/install/source)指南中的说明，请点击链接以了解详情。TensorFlow Quantum 与 TensorFlow 版本 2.5 兼容。

下载 <a href="https://github.com/tensorflow/tensorflow" class="external">TensorFlow 源代码</a>：

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">git clone https://github.com/tensorflow/tensorflow.git</code>
  <code class="devsite-terminal">cd tensorflow</code>
  <code class="devsite-terminal">git checkout v2.5.1</code>
</pre>

确保您在第 2 步中创建的虚拟环境已激活。随后，安装 TensorFlow 依赖项：

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">pip install -U pip six numpy wheel setuptools mock 'future&gt;=0.17.1'</code>
  <code class="devsite-terminal">pip install -U keras_applications --no-deps</code>
  <code class="devsite-terminal">pip install -U keras_preprocessing --no-deps</code>
  <code class="devsite-terminal">pip install numpy==1.19.5</code>
</pre>

<!-- common_typos_enable -->

配置 TensorFlow 构建。当询问 Python 解释器和库位置时，请务必指定虚拟环境文件夹内的位置。其余选项可以保留为默认值。

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">./configure</code>
</pre>

<!-- common_typos_enable -->

构建 TensorFlow 软件包：

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">bazel build -c opt --cxxopt="-O3" --cxxopt="-march=native" --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" //tensorflow/tools/pip_package:build_pip_package</code>
</pre>

<!-- common_typos_enable -->

注：构建软件包可能需要一个多小时。

构建完成后，安装软件包，离开 TensorFlow 目录：

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg</code>
  <code class="devsite-terminal">pip install /tmp/tensorflow_pkg/&lt;var&gt;name_of_generated_wheel&lt;/var&gt;.whl</code>
  <code class="devsite-terminal">cd ..</code>
</pre>

<!-- common_typos_enable -->

### 5. 下载 TensorFlow Quantum

我们使用标准的[复刻和拉取请求工作流](https://guides.github.com/activities/forking/)进行贡献。从 [TensorFlow Quantum](https://github.com/tensorflow/quantum) GitHub 页面复刻后，下载您的复刻的源代码并安装要求：

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">git clone https://github.com/&lt;var&gt;username&lt;/var&gt;/quantum.git</code>
  <code class="devsite-terminal">cd quantum</code>
  <code class="devsite-terminal">pip install -r requirements.txt</code>
</pre>

<!-- common_typos_enable -->

### 6. 构建 TensorFlow Quantum pip 软件包

构建 TensorFlow Quantum pip 软件包并安装：

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">./configure.sh</code>
  <code class="devsite-terminal">bazel build -c opt --cxxopt="-O3" --cxxopt="-march=native" --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" release:build_pip_package</code>
  <code class="devsite-terminal">bazel-bin/release/build_pip_package /tmp/tfquantum/</code>
  <code class="devsite-terminal">python3 -m pip install /tmp/tfquantum/&lt;var&gt;name_of_generated_wheel&lt;/var&gt;.whl</code>
</pre>

<!-- common_typos_enable -->

要确认 TensorFlow Quantum 已成功安装，可以运行测试：

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">./scripts/test_all.sh</code>
</pre>

<!-- common_typos_enable -->

成功：TensorFlow Quantum 现已完成安装。
