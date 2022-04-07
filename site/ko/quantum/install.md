# TensorFlow Quantum 설치하기

TensorFlow Quantum(TFQ)을 사용하도록 환경을 설정하는 몇 가지 방법이 있습니다.

- 설치할 필요 없이 TFQ를 배우고 사용하는 가장 쉬운 방법은 [Google Colab](./tutorials/hello_many_worlds.ipynb)을 사용하여 브라우저에서 직접 [TensorFlow Quantum 튜토리얼](https://colab.research.google.com/github/tensorflow/quantum/blob/master/docs/tutorials/hello_many_worlds.ipynb)을 실행하는 것입니다.
- 로컬 컴퓨터에서 TensorFlow Quantum을 사용하려면 Python의 pip 패키지 관리자를 사용하여 TFQ 패키지를 설치해야 합니다.
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

## 소스에서 빌드

다음 단계는 Ubuntu와 유사한 시스템에 대해 테스트되었습니다.

### 1. Python 3 개발 환경을 설정합니다.

먼저 Python 3.8 개발 도구가 필요합니다.

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">sudo apt update</code>
  <code class="devsite-terminal">sudo apt-get install pkg-config zip g++ zlib1g-dev unzip python3.8</code>
  <code class="devsite-terminal">sudo apt install python3.8 python3.8-dev python3.8-venv python3-pip</code>
  <code class="devsite-terminal">python3.8 -m pip install --upgrade pip</code>
</pre>

<!-- common_typos_enable -->

### 2. 가상 환경을 만듭니다.

작업 공간 디렉토리로 이동하여 TFQ 개발을 위한 가상 환경을 만듭니다.

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">python3.8 -m venv quantum_env</code>
  <code class="devsite-terminal">source quantum_env/bin/activate</code>
</pre>

<!-- common_typos_enable -->

### 3. Bazel을 설치합니다.

[TensorFlow build from source](https://www.tensorflow.org/install/source#install_bazel) 가이드에 언급했듯이 <a href="https://bazel.build/" class="external">Bazel</a> 빌드 시스템이 필요합니다.

Our latest source builds use TensorFlow 2.7.0. To ensure compatibility we use `bazel` version 3.7.2. To remove any existing version of Bazel:

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">sudo apt-get remove bazel</code>
</pre>

<!-- common_typos_enable -->

`bazel` 버전 3.7.2를 다운로드하고 설치합니다.

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">wget https://github.com/bazelbuild/bazel/releases/download/3.7.2/bazel_3.7.2-linux-x86_64.deb
</code>
  <code class="devsite-terminal">sudo dpkg -i bazel_3.7.2-linux-x86_64.deb</code>
</pre>

<!-- common_typos_enable -->

`bazel`이 호환되지 않는 버전으로 자동 업데이트되는 것을 방지하려면 다음을 실행하세요.

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">sudo apt-mark hold bazel</code>
</pre>

<!-- common_typos_enable -->

마지막으로, `bazel` 버전 설치를 확인합니다.

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

2단계에서 생성한 가상 환경이 활성화되어 있는지 확인합니다. 그런 다음 TensorFlow 종속 요소를 설치합니다.

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">pip install -U pip six numpy wheel setuptools mock 'future&gt;=0.17.1'</code>
  <code class="devsite-terminal">pip install -U keras_applications --no-deps</code>
  <code class="devsite-terminal">pip install -U keras_preprocessing --no-deps</code>
  <code class="devsite-terminal">pip install numpy==1.19.5</code>
</pre>

<!-- common_typos_enable -->

TensorFlow 빌드를 구성합니다. Python 인터프리터 및 라이브러리 위치를 묻는 메시지가 표시되면 가상 환경 폴더 내에서 위치를 지정해야 합니다. 나머지 옵션은 기본값으로 남겨 둘 수 있습니다.

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">./configure</code>
</pre>

<!-- common_typos_enable -->

TensorFlow 패키지 빌드:

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">bazel build -c opt --cxxopt="-O3" --cxxopt="-march=native" --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" //tensorflow/tools/pip_package:build_pip_package</code>
</pre>

<!-- common_typos_enable -->

참고: 패키지를 빌드하는 데 1시간 이상 걸릴 수 있습니다.

빌드가 완료되면 패키지를 설치하고 TensorFlow 디렉토리를 그대로 남겨 둡니다.

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg</code>
  <code class="devsite-terminal">pip install /tmp/tensorflow_pkg/&lt;var&gt;name_of_generated_wheel&lt;/var&gt;.whl</code>
  <code class="devsite-terminal">cd ..</code>
</pre>

<!-- common_typos_enable -->

### 5. TensorFlow Quantum을 다운로드합니다.

우리는 기여를 위해 표준 [포크 및 풀 요청 워크플로](https://guides.github.com/activities/forking/)를 사용합니다. [TensorFlow Quantum](https://github.com/tensorflow/quantum) GitHub 페이지에서 포크한 후 포크 소스를 다운로드하고 요구 사항을 설치합니다.

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

TensorFlow Quantum이 성공적으로 설치되었는지 확인하기 위해 다음과 같이 테스트를 실행할 수 있습니다.

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">./scripts/test_all.sh</code>
</pre>

<!-- common_typos_enable -->

성공: 이제 TensorFlow Quantum이 설치되었습니다.
