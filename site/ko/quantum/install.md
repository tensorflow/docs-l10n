# TensorFlow Quantum 설치하기

TensorFlow Quantum(TFQ)을 사용하려면 몇 가지 방법으로 환경을 설정할 수 있습니다.

- TFQ를 배우고 사용하는 가장 쉬운 방법은 설치가 필요하지 않습니다. [Google Colab](./tutorials/hello_many_worlds.ipynb)을 사용하여 브라우저에서 직접 [TensorFlow Quantum 튜토리얼](https://colab.research.google.com/github/tensorflow/quantum/blob/master/docs/tutorials/hello_many_worlds.ipynb)을 실행하세요.
- 로컬 컴퓨터에서 TensorFlow Quantum을 사용하려면 Python의 pip 패키지 관리자를 사용하여 TFQ 패키지를 설치합니다.
- 또는 소스에서 TensorFlow Quantum을 빌드합니다.

TensorFlow Quantum은 Python 3.6 및 3.7에서 지원되며 [Cirq](https://github.com/quantumlib/Cirq)에 직접 의존합니다.

## Pip 패키지

### 요구 사항

- pip 19.0 이상(`manylinux2010` 지원 필요)
- [TensorFlow == 2.1](https://www.tensorflow.org/install/pip)

Python 개발 환경 및 가상 환경(선택 사항)을 설정하려면 [TensorFlow 설치 가이드](https://www.tensorflow.org/install/pip)를 참조하세요.

`pip`를 업그레이드하고 TensorFlow를 설치합니다.

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">pip3 install --upgrade pip</code>
  <code class="devsite-terminal">pip3 install tensorflow==2.1.0</code>
</pre>

<!-- common_typos_enable -->

### 패키지를 설치합니다.

TensorFlow Quantum의 안정적인 최신 릴리스를 설치합니다.

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">pip3 install -U tensorflow-quantum</code>
</pre>

<!-- common_typos_enable -->

성공: 이제 TensorFlow Quantum이 설치되었습니다.

TensorFlow Quantum의 최신 야간 버전을 설치합니다.

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">pip3 install -U tfq-nightly</code>
</pre>

<!-- common_typos_enable -->

## 소스에서 빌드하기

다음 단계는 Ubuntu와 유사한 시스템에서 테스트되었습니다.

### 1. Python 3 개발 환경을 설정합니다.

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">sudo apt update</code>
  <code class="devsite-terminal">sudo apt-get install pkg-config zip g++ zlib1g-dev unzip python3</code>
  <code class="devsite-terminal">sudo apt install python3 python3-dev python3-venv python3-pip</code>
  <code class="devsite-terminal">python3 -m pip install --upgrade pip</code>
</pre>

<!-- common_typos_enable -->

### 2. 가상 환경을 만듭니다.

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">python3 -m venv tfq_env</code>
  <code class="devsite-terminal">source tfq_env/bin/activate</code>
</pre>

<!-- common_typos_enable -->

### 3. Bazel을 설치합니다.

Bazel 빌드 시스템을 설치하려면 TensorFlow [소스에서 빌드하기](https://www.tensorflow.org/install/source#install_bazel) 가이드를 참조하세요.

TensorFlow와의 호환성을 보장하려면 `bazel` 버전 0.26.1 이하가 필요합니다. 기존 버전의 Bazel을 제거하려면 다음을 실행합니다.

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">sudo apt-get remove bazel</code>
</pre>

<!-- common_typos_enable -->

그런 다음 Bazel 버전 0.26.0을 설치합니다.

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">wget https://github.com/bazelbuild/bazel/releases/download/0.26.0/bazel_0.26.0-linux-x86_64.deb</code>
  <code class="devsite-terminal">sudo dpkg -i bazel_0.26.0-linux-x86_64.deb</code>
</pre>

<!-- common_typos_enable -->

### 4. 소스에서 TensorFlow를 빌드합니다.

자세한 내용은 TensorFlow [소스에서 빌드하기](https://www.tensorflow.org/install/source) 가이드를 참조하세요. TensorFlow Quantum은 TensorFlow 버전 2.1과 호환됩니다.

<a href="https://github.com/tensorflow/tensorflow" class="external">TensorFlow 소스 코드</a>를 다운로드합니다.

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">git clone https://github.com/tensorflow/tensorflow.git</code>
  <code class="devsite-terminal">cd tensorflow</code>
  <code class="devsite-terminal">git checkout v2.1.0</code>
</pre>

TensorFlow 종속성을 설치합니다.

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">python3 -m pip install -U pip six numpy wheel setuptools mock 'future>=0.17.1'</code>
  <code class="devsite-terminal">python3 -m pip install -U keras_applications --no-deps</code>
  <code class="devsite-terminal">python3 -m pip install -U keras_preprocessing --no-deps</code>
</pre>

<!-- common_typos_enable -->

TensorFlow 빌드를 구성합니다. 기본 Python 위치 및 Python 라이브러리 경로는 가상 환경 내부를 가리켜야 합니다. 다음과 같은 기본 옵션을 사용하는 것이 좋습니다.

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">./configure</code>
</pre>

<!-- common_typos_enable -->

Bazel 버전이 올바른지 확인합니다.

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">bazel version</code>
</pre>

<!-- common_typos_enable -->

TensorFlow 패키지를 빌드합니다.

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">bazel build -c opt --cxxopt="-O3" --cxxopt="-march=native" --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" //tensorflow/tools/pip_package:build_pip_package</code>
</pre>

<!-- common_typos_enable -->

참고: 패키지를 빌드하는 데 1시간 이상 걸릴 수 있습니다.

빌드가 완료되면 패키지를 설치합니다.

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg</code>
  <code class="devsite-terminal">pip install /tmp/tensorflow_pkg/name_of_generated_wheel.whl</code>
</pre>

<!-- common_typos_enable -->

### 5. TensorFlow Quantum을 다운로드합니다.

TensorFlow Quantum 소스 코드를 다운로드하고 요구 사항을 설치합니다.

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">cd ..</code>
  <code class="devsite-terminal">git clone https://github.com/tensorflow/quantum.git</code>
  <code class="devsite-terminal">cd quantum</code>
  <code class="devsite-terminal">python3 -m pip install -r requirements.txt</code>
</pre>

<!-- common_typos_enable -->

자동 업데이트가 가능하므로 Bazel 버전을 확인합니다.

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">bazel version</code>
</pre>

<!-- common_typos_enable -->

### 6. TensorFlow Quantum pip 패키지를 빌드합니다.

TensorFlow Quantum pip 패키지를 빌드하고 설치합니다.

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">./configure.sh</code>
  <code class="devsite-terminal">bazel build -c opt --cxxopt="-O3" --cxxopt="-march=native" --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" release:build_pip_package</code>
  <code class="devsite-terminal">bazel-bin/release/build_pip_package /tmp/tfquantum/</code>
  <code class="devsite-terminal">python3 -m pip install /tmp/tfquantum/name_of_generated_wheel.whl</code>
</pre>

<!-- common_typos_enable -->

성공: 이제 TensorFlow Quantum이 설치되었습니다.
