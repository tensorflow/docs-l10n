# TensorFlow Federated 설치

TensorFlow Federated(TFF)를 사용하도록 환경을 설정하는 몇 가지 방법이 있습니다.

- TFF를 배우고 사용하는 가장 쉬운 방법에는 설치가 필요하지 않습니다. [Google Colaboratory](https://colab.research.google.com/notebooks/welcome.ipynb)를 사용하여 브라우저에서 직접 TensorFlow Federated 가이드를 실행합니다.
- 로컬 머신에서 TensorFlow Federated를 사용하려면 Python의 `pip` 패키지 관리자로 [TFF 패키지](#install-tensorflow-federated-using-pip)를 설치합니다.
- 고유한 머신 구성이 있는 경우, 소스에서 [TFF 패키지를 빌드](#build-the-tensorflow-federated-pip-package)합니다.

## `pip`를 사용하여 TensorFlow Federated 설치하기

### 1. Python 개발 환경을 설치합니다.

Ubuntu:

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">sudo apt update</code>
<code class="devsite-terminal">sudo apt install python3-dev python3-pip  # Python 3</code>
<code class="devsite-terminal">sudo pip3 install --user --upgrade virtualenv</code>
</pre>

macOS:

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"</code>
<code class="devsite-terminal">export PATH="/usr/local/bin:/usr/local/sbin:$PATH"</code>
<code class="devsite-terminal">brew update</code>
<code class="devsite-terminal">brew install python  # Python 3</code>
<code class="devsite-terminal">sudo pip3 install --user --upgrade virtualenv</code>
</pre>

### 2. 가상 환경을 만듭니다.

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">virtualenv --python python3 "venv"</code>
<code class="devsite-terminal">source "venv/bin/activate"</code>
<code class="devsite-terminal tfo-terminal-venv">pip install --upgrade pip</code>
</pre>

참고: 가상 환경을 종료하려면, `deactivate`를 실행하세요.

### 3. TensorFlow Federated Python 패키지를 설치합니다.

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal tfo-terminal-venv">pip install --upgrade tensorflow_federated</code>
</pre>

### 4. Tensorflow Federated를 테스트합니다.

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal tfo-terminal-venv">python -c "import tensorflow_federated as tff; print(tff.federated_computation(lambda: 'Hello World')())"</code>
</pre>

성공: 이제 최신 TensorFlow Federated Python 패키지가 설치되었습니다.

## 소스에서 TensorFlow Federated Python 패키지 빌드하기

소스에서 TensorFlow Federated Python 패키지를 빌드하면, 다음과 같은 경우에 유용합니다.

- TensorFlow Federated를 변경하고 변경 사항을 제출하거나 릴리스하기 전에 TensorFlow Federated를 사용하는 구성 요소에서 해당 변경 사항을 테스트합니다.
- TensorFlow Federated에 제출되었지만, 아직 릴리즈되지 않은 변경 사항을 사용합니다.

### 1. Python 개발 환경을 설치합니다.

Ubuntu:

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">sudo apt update</code>
<code class="devsite-terminal">sudo apt install python3-dev python3-pip  # Python 3</code>
<code class="devsite-terminal">sudo pip3 install --user --upgrade virtualenv</code>
</pre>

macOS:

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"</code>
<code class="devsite-terminal">export PATH="/usr/local/bin:/usr/local/sbin:$PATH"</code>
<code class="devsite-terminal">brew update</code>
<code class="devsite-terminal">brew install python  # Python 3</code>
<code class="devsite-terminal">sudo pip3 install --user --upgrade virtualenv</code>
</pre>

### 2. Bazel을 설치합니다.

Tensorflow Federated를 컴파일하는 데 사용되는 빌드 도구인 [Bazel을 설치](https://docs.bazel.build/versions/master/install.html)합니다.

### 3. Tensorflow Federated 리포지토리를 복제합니다.

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">git clone https://github.com/tensorflow/federated.git</code>
<code class="devsite-terminal">cd "federated"</code>
</pre>

### 4. TensorFlow Federated Python 패키지를 빌드합니다.

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">mkdir "/tmp/tensorflow_federated"</code>
<code class="devsite-terminal">bazel run //tensorflow_federated/tools/development:build_pip_package -- \
    --nightly \
    --output_dir "/tmp/tensorflow_federated"</code>
</pre>

### 5. 새 프로젝트를 만듭니다.

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">mkdir "/tmp/project"</code>
<code class="devsite-terminal">cd "/tmp/project"</code>
</pre>

### 6. 가상 환경을 만듭니다.

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">virtualenv --python python3 "venv"</code>
<code class="devsite-terminal">source "venv/bin/activate"</code>
<code class="devsite-terminal tfo-terminal-venv">pip install --upgrade pip</code>
</pre>

참고: 가상 환경을 종료하려면, `deactivate`를 실행하세요.

### 7. TensorFlow Federated Python 패키지를 설치합니다.

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal tfo-terminal-venv">pip install --upgrade "/tmp/tensorflow_federated/"*".whl"</code>
</pre>

### 8. Tensorflow Federated를 테스트합니다.

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal tfo-terminal-venv">python -c "import tensorflow_federated as tff; print(tff.federated_computation(lambda: 'Hello World')())"</code>
</pre>

성공: 이제 TensorFlow Federated Python 패키지가 소스에서 빌드되고 설치되었습니다.
