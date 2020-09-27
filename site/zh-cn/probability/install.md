# 安装

## 稳定版本

安装最新版本的 TensorFlow Probability：

<pre class="devsite-terminal devsite-click-to-copy prettyprint lang-shell"> pip install --upgrade tensorflow-probability</pre>

TensorFlow Probability 依赖于最新稳定版本的 [TensorFlow](https://www.tensorflow.org/install)（pip 软件包 `tensorflow`）。有关 TensorFlow 和 TensorFlow Probability 之间依赖关系的详细信息，请参阅 [TFP 版本说明](https://github.com/tensorflow/probability/releases)。

注：由于 TensorFlow *不*作为 TensorFlow Probability 软件包（在 `setup.py` 中）的依赖项包含在内，因此您必须显式安装 TensorFlow 软件包（`tensorflow` 或 `tensorflow-gpu`）。这样，我们便可为支持 CPU 和 GPU 的 TensorFlow 维护一个软件包，而不用维护单独的软件包。

要强制执行特定于 Python 3 的安装，请将上述命令中的 `pip` 替换为 `pip3`。有关其他安装帮助、安装前提条件指导以及（可选）设置虚拟环境的信息，请参阅 [TensorFlow 安装指南](https://www.tensorflow.org/install)。

## Nightly 版本

此外，pip 软件包 `tfp-nightly` 下还有 Nightly 版本的 TensorFlow Probability，此版本依赖于 `tf-nightly` 和 `tf-nightly-gpu` 之一。Nightly 版本包含较新的功能，但可能不如带版本号的版本稳定。

## 从源代码安装

您也可以从源代码安装。这需要 [Bazel](https://bazel.build/){:.external} 构建系统。强烈建议您在尝试从源代码构建 TensorFlow Probability 之前安装 Nightly 版本的 TensorFlow (`tf-nightly`)。

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
