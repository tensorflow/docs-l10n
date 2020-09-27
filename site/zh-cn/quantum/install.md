# 安装 TensorFlow Quantum

要使用 TensorFlow Quantum (TFQ)，您可通过以下几种方式设置环境：

- 学习和使用 TFQ 的最简单方式无需安装：使用 [Google Colab](./tutorials/hello_many_worlds.ipynb) 直接在浏览器中运行 [TensorFlow Quantum 教程](./tutorials/hello_many_worlds.ipynb)。
- 要在本地计算机上使用 TensorFlow Quantum，请使用 Python 的 pip 软件包管理器安装 TFQ 软件包。
- 或者从源代码构建 TensorFlow Quantum。

TensorFlow Quantum 在 Python 3.6 和 3.7 上受支持，并且直接依赖于 [Cirq](https://github.com/quantumlib/Cirq)。

## Pip 软件包

### 要求

- pip 19.0 或更高版本（需要 `manylinux2010` 支持）
- [TensorFlow == 2.1](https://www.tensorflow.org/install/pip)

要设置您的 Python 开发环境和一个（可选的）虚拟环境，请参阅 [TensorFlow 安装指南](https://www.tensorflow.org/install/pip)。

升级 `pip` 并安装 TensorFlow

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">pip3 install --upgrade pip</code>
  <code class="devsite-terminal">pip3 install tensorflow==2.1.0</code>
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

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">sudo apt update</code>
  <code class="devsite-terminal">sudo apt-get install pkg-config zip g++ zlib1g-dev unzip python3</code>
  <code class="devsite-terminal">sudo apt install python3 python3-dev python3-venv python3-pip</code>
  <code class="devsite-terminal">python3 -m pip install --upgrade pip</code>
</pre>

<!-- common_typos_enable -->

### 2. 创建虚拟环境

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">python3 -m venv tfq_env</code>
  <code class="devsite-terminal">source tfq_env/bin/activate</code>
</pre>

<!-- common_typos_enable -->

### 3. 安装 Bazel

要安装 <a href="https://bazel.build/" class="external">Bazel</a> 构建系统，请参阅 TensorFlow [从源代码构建](https://www.tensorflow.org/install/source#install_bazel)指南。

为了确保与 TensorFlow 的兼容性，需要 0.26.1 或更低版本的 `bazel`。移除任何现有版本的 Bazel：

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">sudo apt-get remove bazel</code>
</pre>

<!-- common_typos_enable -->

然后安装 Bazel 版本 0.26.0：

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">wget https://github.com/bazelbuild/bazel/releases/download/0.26.0/bazel_0.26.0-linux-x86_64.deb</code>
  <code class="devsite-terminal">sudo dpkg -i bazel_0.26.0-linux-x86_64.deb</code>
</pre>

<!-- common_typos_enable -->

### 4. 从源代码构建 TensorFlow

有关详细信息，请阅读 TensorFlow [从源代码构建](https://www.tensorflow.org/install/source)指南。TensorFlow Quantum 与 TensorFlow 版本 2.1 兼容。

下载 <a href="https://github.com/tensorflow/tensorflow" class="external">TensorFlow 源代码</a>：

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">git clone https://github.com/tensorflow/tensorflow.git</code>
  <code class="devsite-terminal">cd tensorflow</code>
  <code class="devsite-terminal">git checkout v2.1.0</code>
</pre>

安装 TensorFlow 依赖项：

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">python3 -m pip install -U pip six numpy wheel setuptools mock 'future>=0.17.1'</code>
  <code class="devsite-terminal">python3 -m pip install -U keras_applications --no-deps</code>
  <code class="devsite-terminal">python3 -m pip install -U keras_preprocessing --no-deps</code>
</pre>

<!-- common_typos_enable -->

配置 TensorFlow 构建。默认的 Python 位置和 Python 库路径应当指向虚拟环境内部。建议使用默认选项：

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">./configure</code>
</pre>

<!-- common_typos_enable -->

验证您的 Bazel 版本是否正确：

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">bazel version</code>
</pre>

<!-- common_typos_enable -->

构建 TensorFlow 软件包：

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">bazel build -c opt --cxxopt="-O3" --cxxopt="-march=native" --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" //tensorflow/tools/pip_package:build_pip_package</code>
</pre>

<!-- common_typos_enable -->

注：构建软件包可能需要一个多小时。

构建完成后，安装软件包：

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg</code>
  <code class="devsite-terminal">pip install /tmp/tensorflow_pkg/name_of_generated_wheel.whl</code>
</pre>

<!-- common_typos_enable -->

### 5. 下载 TensorFlow Quantum

下载 TensorFlow Quantum 源代码并安装要求：

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">cd ..</code>
  <code class="devsite-terminal">git clone https://github.com/tensorflow/quantum.git</code>
  <code class="devsite-terminal">cd quantum</code>
  <code class="devsite-terminal">python3 -m pip install -r requirements.txt</code>
</pre>

<!-- common_typos_enable -->

验证您的 Bazel 版本（因为它可以自动更新）：

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">bazel version</code>
</pre>

<!-- common_typos_enable -->

### 6. 构建 TensorFlow Quantum pip 软件包

构建 TensorFlow Quantum pip 软件包并安装：

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">./configure.sh</code>
  <code class="devsite-terminal">bazel build -c opt --cxxopt="-O3" --cxxopt="-march=native" --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" release:build_pip_package</code>
  <code class="devsite-terminal">bazel-bin/release/build_pip_package /tmp/tfquantum/</code>
  <code class="devsite-terminal">python3 -m pip install /tmp/tfquantum/name_of_generated_wheel.whl</code>
</pre>

<!-- common_typos_enable -->

成功：TensorFlow Quantum 现已完成安装。
