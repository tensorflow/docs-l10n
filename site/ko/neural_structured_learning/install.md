# Neural Structured Learning 설치

TensorFlow에서 Neural Structured Learning(NSL)을 사용하도록 환경을 설정하는 방법에는 여러 가지가 있습니다.

- NSL을 배우고 사용하는 가장 쉬운 방법에는 설치가 필요 없습니다. [Google Colaboratory](https://colab.research.google.com/notebooks/welcome.ipynb)를 사용하여 브라우저에서 직접 NSL 가이드를 실행하세요.
- 로컬 머신에서 NSL을 사용하려면, Python의 `pip` 패키지 관리자로 [NSL 패키지](#install-neural-structured-learning-using-pip)를 설치합니다.
- 고유한 머신 구성이 있는 경우, 소스에서 [NSL을 빌드합니다](#build-the-neural-structured-learning-pip-package).

참고: NSL에는 1.15 이상의 TensorFlow 버전이 필요합니다. NSL은 또한 NSL과 호환되지 않는 버그가 포함된 v2.1을 제외하고 TensorFlow 2.x를 지원합니다.

## pip를 사용하여 Neural Structured Learning 설치하기

#### 1. Install the Python development environment.

Ubuntu:

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">sudo apt update</code>
<code class="devsite-terminal">sudo apt install python3-dev python3-pip  # Python 3</code>
<code class="devsite-terminal">sudo pip3 install --upgrade virtualenv  # system-wide install</code>
</pre>

macOS:

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"</code>
<code class="devsite-terminal">export PATH="/usr/local/bin:/usr/local/sbin:$PATH"</code>
<code class="devsite-terminal">brew update</code>
<code class="devsite-terminal">brew install python  # Python 3</code>
<code class="devsite-terminal">sudo pip3 install --upgrade virtualenv  # system-wide install</code>
</pre>

#### 2. Create a virtual environment.

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">virtualenv --python python3 "./venv"</code>
<code class="devsite-terminal">source "./venv/bin/activate"</code>
<code class="devsite-terminal tfo-terminal-venv">pip install --upgrade pip</code>
</pre>

참고: 가상 환경을 종료하려면 `deactivate`를 실행하세요.

#### 3. TensorFlow를 설치합니다.

CPU 지원:

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal tfo-terminal-venv">pip install 'tensorflow&gt;=1.15.0'</code>
</pre>

GPU 지원:

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal tfo-terminal-venv">pip install 'tensorflow-gpu&gt;=1.15.0'</code>
</pre>

#### 4. Neural Structured Learning `pip` 패키지를 설치합니다.

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal tfo-terminal-venv">pip install --upgrade neural_structured_learning</code>
</pre>

#### 5. (선택 사항) Neural Structured Learning을 테스트합니다.

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal tfo-terminal-venv">python -c "import neural_structured_learning as nsl"</code>
</pre>

성공: 이제 Neural Structured Learning이 설치되었습니다.

## Neural Structured Learning pip 패키지 빌드하기

#### 1. Install the Python development environment.

Ubuntu:

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">sudo apt update</code>
<code class="devsite-terminal">sudo apt install python3-dev python3-pip  # Python 3</code>
<code class="devsite-terminal">sudo pip3 install --upgrade virtualenv  # system-wide install</code>
</pre>

macOS:

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"</code>
<code class="devsite-terminal">export PATH="/usr/local/bin:/usr/local/sbin:$PATH"</code>
<code class="devsite-terminal">brew update</code>
<code class="devsite-terminal">brew install python  # Python 3</code>
<code class="devsite-terminal">sudo pip3 install --upgrade virtualenv  # system-wide install</code>
</pre>

#### 2. Install Bazel.

Neural Structured Learning을 컴파일하는 데 사용되는 빌드 도구인 [Bazel을 설치합니다](https://docs.bazel.build/versions/master/install.html).

#### 3. Neural Structured Learning 리포지토리를 복제합니다.

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">git clone https://github.com/tensorflow/neural-structured-learning.git</code>
</pre>

#### 4. Create a virtual environment.

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">virtualenv --python python3 "./venv"</code>
<code class="devsite-terminal">source "./venv/bin/activate"</code>
<code class="devsite-terminal tfo-terminal-venv">pip install --upgrade pip</code>
</pre>

참고: 가상 환경을 종료하려면 `deactivate`를 실행하세요.

#### 5. Tensorflow를 설치합니다.

Note that NSL requires a TensorFlow version of 1.15 or higher. NSL also supports TensorFlow 2.0.

CPU 지원:

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal tfo-terminal-venv">pip install 'tensorflow&gt;=1.15.0'</code>
</pre>

GPU 지원:

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal tfo-terminal-venv">pip install 'tensorflow-gpu&gt;=1.15.0'</code>
</pre>

#### 6. Neural Structured Learning 종속성을 설치합니다.

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">cd neural-structured-learning</code>
<code class="devsite-terminal tfo-terminal-venv">pip install --requirement neural_structured_learning/requirements.txt</code>
</pre>

#### 7. (선택 사항) Neural Structured Learning을 단위 테스트합니다.

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal tfo-terminal-venv">bazel test //neural_structured_learning/...</code>
</pre>

#### 8. Build the pip package.

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal tfo-terminal-venv">python setup.py bdist_wheel --universal --dist-dir="./wheel"</code>
</pre>

#### 9. Install the pip package.

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal tfo-terminal-venv">pip install --upgrade ./wheel/neural_structured_learning*.whl</code>
</pre>

#### 10. Neural Structured Learning을 테스트합니다.

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal tfo-terminal-venv">python -c "import neural_structured_learning as nsl"</code>
</pre>

성공: Neural Structured Learning 패키지가 구축되었습니다.
