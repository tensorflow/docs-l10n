<div align="center">
<img width="60%" src="https://tensorflow.org/images/SIGAddons.png"><br><br>
</div>

---

# TensorFlow Addons

**TensorFlow Addons**은 잘 확립된 API 패턴을 준수하지만, 핵심 TensorFlow에서 사용할 수 없는 새로운 기능을 구현하는 노력의 리포지토리입니다. TensorFlow는 기본적으로 많은 수의 연산자, 레이어, 메트릭, 손실 및 옵티마이저를 지원합니다. 그러나 ML과 같이 빠르게 발전하는 분야에는 핵심 TensorFlow에 통합할 수 없는 흥미로운 새로운 개발이 많이 있습니다(광범위한 적용 가능성이 아직 명확하지 않거나 커뮤니티의 작은 단위에서 주로 사용되기 때문입니다).

## 설치

#### 안정적인 빌드

최신 버전을 설치하려면, 다음을 실행합니다.

```
pip install tensorflow-addons
```

애드온을 사용하려면,

```python
import tensorflow as tf
import tensorflow_addons as tfa
```

#### 야간 빌드

TensorFlow의 안정적인 최신 버전에서 빌드된 pip 패키지 `tfa-nightly` 아래에 TensorFlow Addons의 야간 빌드가 있습니다. 야간 빌드에는 새로운 기능이 포함되어 있지만, 버전이 지정된 릴리스보다 안정성이 떨어질 수 있습니다.

```
pip install tfa-nightly
```

#### 소스에서 설치하기

소스에서 설치할 수도 있습니다. 이를 위해서는 [Bazel](https://bazel.build/) 빌드 시스템이 필요합니다.

```
git clone https://github.com/tensorflow/addons.git
cd addons

# If building GPU Ops (Requires CUDA 10.0 and CuDNN 7)
export TF_NEED_CUDA=1
export CUDA_HOME="/path/to/cuda10" (default: /usr/local/cuda)
export CUDNN_INSTALL_PATH="/path/to/cudnn" (default: /usr/lib/x86_64-linux-gnu)

# This script links project with TensorFlow dependency
python3 ./configure.py

bazel build build_pip_pkg
bazel-bin/build_pip_pkg artifacts

pip install artifacts/tensorflow_addons-*.whl
```

## 핵심 개념

#### 서브 패키지 내 표준화된 API

사용자 경험과 프로젝트 유지 관리는 TF-Addons의 핵심 개념입니다. 핵심 TensorFlow에서 볼 수 있는 확립된 API 패턴을 준수해야 합니다.

#### GPU/CPU 사용자 정의 연산

TensorFlow Addons의 주요 이점은 사전 컴파일된 ops가 있다는 것입니다. CUDA 10 설치를 찾을 수 없는 경우, 해당 op는 자동으로 CPU 구현으로 대체됩니다.

#### 프록시 유지 관리

애드온은 하위 패키지와 하위 모듈을 구분하여 해당 구성 요소에 대한 전문 지식과 관심이 있는 사용자가 유지 관리할 수 있도록 설계되었습니다.

서브 패키지의 유지 관리 권한은 쓰기 권한이 있는 사용자의 수를 제한하기 위해 기여한 바가 큰 사용자에게 부여될 수 있습니다. 기여는 이슈 종결, 버그 수정, 문서화, 새 코드 또는 기존 코드의 최적화와 같은 여러 형태를 포함합니다. 하위 모듈의 유지 관리 권한은 리포지토리에 대한 쓰기 권한을 포함하지 않으므로 진입 장벽을 낮추어 부여할 수 있습니다.

자세한 내용은 이 주제에 대한 [RFC](https://github.com/tensorflow/community/blob/master/rfcs/20190308-addons-proxy-maintainership.md)를 참조하세요.

#### 하위 패키지의 주기적 평가

이 리포지토리의 특성을 고려할 때 하위 패키지와 하위 모듈은 시간이 지남에 따라 커뮤니티에서 그 유용성이 떨어질 수 있습니다. 리포지토리가 지속될 수 있도록 유지하기 위해 2년에 한 번씩 코드를 검토하여 모든 것이 리포지토리 내에 있는지 확인할 것입니다. 검토에 영향을 주는 요소는 다음과 같습니다.

1. 활성 관리자의 수
2. OSS 사용량
3. 코드로 인한 문제 또는 버그의 양
4. 더 나은 솔루션을 사용할 수 있는 경우

TensorFlow Addons 내의 기능은 3가지 그룹으로 분류할 수 있습니다.

- **Suggested**: 잘 관리된 API, 사용이 권장됩니다.
- **Discouraged**: 더 나은 대안을 사용할 수 있습니다. API는 역사적인 이유로 보관됩니다. 또는 API에 유지 보수가 필요하며 더 이상 사용되지 않을 때까지 기다리는 기간입니다.
- **Deprecated**: 사용자의 책임 하에 사용하세요. 삭제될 수 있습니다.

이 3가지 그룹 간의 상태 변경은 다음과 같습니다. Suggested <-> Discouraged -> Deprecated

API가 'deprecated'으로 표시된 후 삭제되기까지의 기간은 90일입니다. 근거는 다음과 같습니다.

1. TensorFlow Addons이 매월 릴리스되는 경우, API가 삭제되기 전에 2~3번 릴리즈될 수 있습니다. 릴리스 노트에서 사용자에게 충분한 경고를 제공할 것입니다.

2. 90일은 관리자가 코드를 수정할 수 있는 충분한 시간을 제공합니다.

## 기여하기

TF-Addons은 커뮤니티 주도의 오픈 소스 프로젝트입니다. 따라서 프로젝트는 공개 기여, 버그 수정 및 문서화에 의존합니다. 기여 방법은 [기여 가이드라인](https://github.com/tensorflow/addons/blob/master/CONTRIBUTING.md)을 참조하세요. 이 프로젝트는 [TensorFlow의 행동 강령](https://github.com/tensorflow/addons/blob/master/CODE_OF_CONDUCT.md)을 따릅니다. 참여를 통해 이 코드를 많이 지지해 주시기 바랍니다.

## 커뮤니티

- [공개 메일 그룹](https://groups.google.com/a/tensorflow.org/forum/#!forum/addons)
- [SIG 월간 회의 노트](https://docs.google.com/document/d/1kxg5xIHWLY7EMdOJCdSGgaPu27a9YKpupUz2VTXqTJg)
    - 메일 그룹에 가입하고 회의에 대한 캘린더 초대 받기

## 라이선스

[Apache 라이선스 2.0](LICENSE)
