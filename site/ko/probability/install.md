# 설치

## 안정적 빌드

최신 버전의 TensorFlow Probability를 설치합니다.

<pre class="devsite-terminal devsite-click-to-copy prettyprint lang-shell"> pip install --upgrade tensorflow-probability </pre>

TensorFlow Probability는 최신 버전의 안정적 [TensorFlow](https://www.tensorflow.org/install)입니다(PIP 패키지 `tensorflow`). TensorFlow와 TensorFlow Probability 간의 종속성에 대한 자세한 내용은 [TFP 릴리스 노트](https://github.com/tensorflow/probability/releases)를 참조하세요.

참고: TensorFlow는 TensorFlow Probability 패키지(`setup.py`의 일부)의 종속성으로 포함되지 *않으므로* TensorFlow 패키지(`tensorflow` 또는 `tensorflow-gpu`)를 명시적으로 설치해야 합니다. 이를 통해 CPU 및 GPU 지원 TensorFlow를 위한 별도의 패키지 대신 하나의 패키지를 유지할 수 있습니다.

Python 3에 특정한 설치를 수행하려면 위 명령에서 `pip`를 <code>pip3</code>로 바꿉니다. 추가적인 설치 지원, 설치 전제 조건에 대한 안내 및 가상 환경 설정(선택 사항)에 대한 내용은 [TensorFlow 설치 가이드](https://www.tensorflow.org/install)를 참조하세요.

## 야간 빌드

`tf-nightly` 및 `tf-nightly-gpu` 중 하나로 결정되는 pip 패키지 `tfp-nightly`에 TensorFlow Probability의 야간 빌드도 들어 있습니다. 야간 빌드에는 새로운 특성이 포함되어 있지만 버전 관리되는 릴리스보다 안정성이 떨어질 수 있습니다.

## 소스에서 설치하기

소스에서 설치할 수도 있습니다. 이를 위해서는 [Bazel](https://bazel.build/){:.external} 빌드 시스템이 필요합니다. 소스에서 TensorFlow Probability를 빌드하기 전에 TensorFlow의 야간 빌드(`tf-nightly`)를 설치하는 것이 좋습니다.

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
