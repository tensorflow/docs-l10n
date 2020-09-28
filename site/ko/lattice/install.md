# TensorFlow Lattice 설치하기

TensorFlow Lattice(TFL)를 사용하는 환경을 설정하는 방법에는 여러 가지가 있습니다.

- TFL을 배우고 사용하는 가장 쉬운 방법은 설치가 필요하지 않습니다. 튜토리얼 중 하나를 실행하세요(예: [준비된 estimator 튜토리얼](tutorials/canned_estimators.ipynb)).
- 로컬 컴퓨터에서 TFL을 사용하려면 `tensorflow-lattice` pip 패키지를 설치하세요.
- 고유한 시스템 구성이 있는 경우 소스에서 패키지를 빌드할 수 있습니다.

## pip를 사용하여 TensorFlow Lattice 설치하기

pip를 사용하여 설치합니다.

```shell
pip install --upgrade tensorflow-lattice
```

## 소스에서 빌드하기

github 리포지토리를 복제합니다.

```shell
git clone https://github.com/tensorflow/lattice.git
```

소스에서 pip 패키지를 빌드합니다.

```shell
python setup.py sdist bdist_wheel --universal --release
```

패키지를 설치합니다.

```shell
pip install --user --upgrade /path/to/pkg.whl
```
