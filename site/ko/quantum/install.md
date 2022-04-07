# TensorFlow Quantum 설치하기

TensorFlow Quantum(TFQ)을 사용하도록 환경을 설정하는 몇 가지 방법이 있습니다.

- The easiest way to learn and use TFQ requires no installation—run the [TensorFlow Quantum tutorials](./tutorials/hello_many_worlds.ipynb) directly in your browser using [Google Colab](https://colab.research.google.com/github/tensorflow/quantum/blob/master/docs/tutorials/hello_many_worlds.ipynb).
- To use TensorFlow Quantum on a local machine, install the TFQ package using Python's pip package manager.
- 또는 소스에서 TensorFlow Quantum을 빌드합니다.

TensorFlow Quantum is supported on Python 3.7, 3.8, and 3.9 and depends directly on [Cirq](https://github.com/quantumlib/Cirq).

## Pip 패키지

### 요구 사항

- pip 19.0 이상(`manylinux2010` 지원 필요)
- [TensorFlow == 2.7.0](https://www.tensorflow.org/install/pip)

Python 개발 환경 및 가상 환경(선택 사항)을 설정하려면 [TensorFlow 설치 가이드](https://www.tensorflow.org/install/pip)를 참조하세요.

`pip`를 업그레이드하고 TensorFlow를 설치합니다.

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">pip3 install --upgrade pip</code>
  <code class="devsite-terminal">pip3 install tensorflow==2.7.0</code>
</pre>

<!-- common_typos_enable -->

### 패키지 설치하기

TensorFlow Quantum의 최신 안정 릴리스를 설치합니다.

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">pip3 install -U tensorflow-quantum</code>
</pre>

<!-- common_typos_enable -->

성공: 이제 TensorFlow Quantum이 설치되었습니다.

Nightly builds which might depend on newer version of TensorFlow can be installed with:

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">pip3 install -U tfq-nightly</code>
</pre>

<!-- common_typos_enable -->

## Build from source

다음 단계는 Ubuntu와 유사한 시스템에 대해 테스트되었습니다.

### 1. Python 3 개발 환경을 설정합니다.

First we need the Python 3.8 development tools.

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">sudo apt update</code>
  <code class="devsite-terminal">sudo apt-get install pkg-config zip g++ zlib1g-dev unzip python3.8</code>
  <code class="devsite-terminal">sudo apt install python3.8 python3.8-dev python3.8-venv python3-pip</code>
  <code class="devsite-terminal">python3.8 -m pip install --upgrade pip</code>
</pre>

<!-- common_typos_enable -->

### 2. 가상 환경을 만듭니다.

Go to your workspace directory and make a virtual environment for TFQ development.

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">python3.8 -m venv quantum_env</code>
  <code class="devsite-terminal">source quantum_env/bin/activate</code>
</pre>

<!-- common_typos_enable -->

### 3. Bazel을 설치합니다.

As noted in the TensorFlow [build from source](https://www.tensorflow.org/install/source#install_bazel) guide, the <a href="https://bazel.build/" class="external">Bazel</a> build system will be required.

Our latest source builds use TensorFlow 2.7.0. To ensure compatibility we use `bazel` version 3.7.2. To remove any existing version of Bazel:

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">sudo apt-get remove bazel</code>
</pre>

<!-- common_typos_enable -->

Download and install `bazel` version 3.7.2:

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">wget https://github.com/bazelbuild/bazel/releases/download/3.7.2/bazel_3.7.2-linux-x86_64.deb
</code>
  <code class="devsite-terminal">sudo dpkg -i bazel_3.7.2-linux-x86_64.deb</code>
</pre>

<!-- common_typos_enable -->

To prevent automatic updating of `bazel` to an incompatible version, run the following:

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">sudo apt-mark hold bazel</code>
</pre>

<!-- common_typos_enable -->

Finally, confirm installation of the correct `bazel` version:

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">bazel --version</code>
</pre>

<!-- common_typos_enable -->

### 4. 소스에서 TensorFlow를 빌드합니다.

Here we adapt instructions from the TensorFlow [build from source](https://www.tensorflow.org/install/source) guide, see the link for further details. TensorFlow Quantum is compatible with TensorFlow version 2.7.0.

<a href="https://github.com/tensorflow/tensorflow" class="external">TensorFlow 소스 코드</a>를 다운로드합니다.

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">git clone https://github.com/tensorflow/tensorflow.git</code>
  <code class="devsite-terminal">cd tensorflow</code>
  <code class="devsite-terminal">git checkout v2.7.0</code>
</pre>

Be sure the virtual environment you created in step 2 is activated. Then, install the TensorFlow dependencies:

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">pip install -U pip six numpy wheel setuptools mock 'future&gt;=0.17.1'</code>
  <code class="devsite-terminal">pip install -U keras_applications --no-deps</code>
  <code class="devsite-terminal">pip install -U keras_preprocessing --no-deps</code>
  <code class="devsite-terminal">pip install numpy==1.19.5</code>
</pre>

<!-- common_typos_enable -->

Configure the TensorFlow build. When asked for the Python interpreter and library locations, be sure to specify locations inside your virtual environment folder.  The remaining options can be left at default values.

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">./configure</code>
</pre>

<!-- common_typos_enable -->

Build the TensorFlow package:

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">bazel build -c opt --cxxopt="-O3" --cxxopt="-march=native" --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" //tensorflow/tools/pip_package:build_pip_package</code>
</pre>

<!-- common_typos_enable -->

참고: 패키지를 빌드하는 데 1시간 이상 걸릴 수 있습니다.

After the build is complete, install the package and leave the TensorFlow directory:

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg</code>
  <code class="devsite-terminal">pip install /tmp/tensorflow_pkg/&lt;var&gt;name_of_generated_wheel&lt;/var&gt;.whl</code>
  <code class="devsite-terminal">cd ..</code>
</pre>

<!-- common_typos_enable -->

### 5. TensorFlow Quantum을 다운로드합니다.

We use the standard [fork and pull request workflow](https://guides.github.com/activities/forking/) for contributions.  After forking from the [TensorFlow Quantum](https://github.com/tensorflow/quantum) GitHub page, download the source of your fork and install the requirements:

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">git clone https://github.com/&lt;var&gt;username&lt;/var&gt;/quantum.git</code>
  <code class="devsite-terminal">cd quantum</code>
  <code class="devsite-terminal">pip install -r requirements.txt</code>
</pre>

<!-- common_typos_enable -->

### 6. TensorFlow Quantum pip 패키지를 빌드합니다.

TensorFlow Quantum pip 패키지를 빌드하고 설치합니다.

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">./configure.sh</code>
  <code class="devsite-terminal">bazel build -c opt --cxxopt="-O3" --cxxopt="-march=native" --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" release:build_pip_package</code>
  <code class="devsite-terminal">bazel-bin/release/build_pip_package /tmp/tfquantum/</code>
  <code class="devsite-terminal">python3 -m pip install /tmp/tfquantum/&lt;var&gt;name_of_generated_wheel&lt;/var&gt;.whl</code>
</pre>

<!-- common_typos_enable -->

To confirm that TensorFlow Quantum has successfully been installed, you can run the tests:

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">./scripts/test_all.sh</code>
</pre>

<!-- common_typos_enable -->

성공: 이제 TensorFlow Quantum이 설치되었습니다.
