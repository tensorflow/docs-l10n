# TensorFlow 그래픽 설치하기

## Stable builds

TensorFlow Graphics depends on [TensorFlow](https://www.tensorflow.org/install) 1.13.1 or above. Nightly builds of TensorFlow (tf-nightly) are also supported.

[PyPI](https://pypi.org/project/tensorflow-graphics/)에서 최신 CPU 버전을 설치하려면 다음을 실행합니다.

```shell
# Installing with the `--upgrade` flag ensures you'll get the latest version.
pip install --upgrade tensorflow-graphics
```

최신 GPU 버전을 설치하려면 다음을 실행합니다.

```shell
# Installing with the `--upgrade` flag ensures you'll get the latest version.
pip install --upgrade tensorflow-graphics-gpu
```

추가 설치 도움말, 설치 전제 조건 안내, 가상 환경 설정(선택 사항)은 [TensorFlow 설치 가이드](https://www.tensorflow.org/install)를 참조하세요.

## 소스에서 설치하기 - macOS/Linux

You can also install from source by executing the following commands:

```shell
git clone https://github.com/tensorflow/graphics.git
sh build_pip_pkg.sh
pip install --upgrade dist/*.whl
```

## 선택적 패키지 설치하기 - Linux

TensorFlow Graphics EXR 데이터 로더를 사용하려면 OpenEXR을 설치해야합니다. 다음 명령을 실행하여 수행할 수 있습니다.

```
sudo apt-get install libopenexr-dev
pip install --upgrade OpenEXR
```
