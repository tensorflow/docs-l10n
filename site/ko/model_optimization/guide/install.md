# TensorFlow 모델 최적화 설치하기

설치를 진행하기 전에 Python 가상 환경을 생성하는 것이 좋습니다. 자세한 내용은 TensorFlow 설치 [가이드](https://www.tensorflow.org/install/pip#2.-create-a-virtual-environment-recommended)를 참조하세요.

### 안정적인 빌드

최신 버전을 설치하려면 다음을 실행합니다.

```shell
# Installing with the `--upgrade` flag ensures you'll get the latest version. pip install --user --upgrade tensorflow-model-optimization
```

릴리스에 대한 자세한 내용은 [릴리스 노트](https://github.com/tensorflow/model-optimization/releases)를 참조하세요.

TensorFlow의 필수 버전 및 기타 호환성 정보는 사용하려는 기술에 대한 개요 페이지에서 API 호환성 행렬 섹션을 참조하세요. 예를 들어, 잘라내기(pruning)의 경우, [여기](https://www.tensorflow.org/model_optimization/guide/pruning)에서 개요 페이지를 참조하세요.

TensorFlow는 TensorFlow Model Optimization 패키지(<code>setup.py</code>)의 종속성으로 포함되어 <em>있지 않으므로</em>, TensorFlow 패키지(`tf-nightly` 또는 `tf-nightly-gpu`)를 명시적으로 설치해야 합니다. 이를 통해 CPU 및 GPU 지원 TensorFlow를 위한 별도의 패키지 대신 하나의 패키지를 유지할 수 있습니다.

### 소스에서 설치하기

소스에서 설치할 수도 있습니다. 이를 위해서는 [Bazel](https://bazel.build/) 빌드 시스템이 필요합니다.

```shell
# To install dependencies on Ubuntu:
# sudo apt-get install bazel git python-pip
# For other platforms, see Bazel docs above.
git clone https://github.com/tensorflow/model-optimization.git
cd model-optimization
bazel build --copt=-O3 --copt=-march=native :pip_pkg
PKGDIR=$(mktemp -d)
./bazel-bin/pip_pkg $PKGDIR
pip install --user --upgrade $PKGDIR/*.whl
```
