<!--* freshness: { owner: 'wgierke' reviewed: '2022-01-05' } *-->

# 설치

## tensorflow_hub 설치하기

`tensorflow_hub` 라이브러리는 TensorFlow 1 및 TensorFlow 2와 함께 설치할 수 있습니다. 신규 사용자는 TensorFlow 2로 바로 시작하고 현재 사용자는 업그레이드하는 것이 좋습니다.

### TensorFlow 2와 함께 사용하기

평소처럼 [pip](https://pip.pypa.io/)를 사용하여 [TensorFlow 2를 설치](https://www.tensorflow.org/install)합니다. (GPU 지원에 대한 추가 지침은 여기를 참조하세요.) 그런 다음 옆에 있는 [ `tensorflow-hub`](https://pypi.org/project/tensorflow-hub/)의 최신 버전을 설치합니다(0.5.0 이상이어야 함).

```bash
$ pip install "tensorflow>=2.0.0"
$ pip install --upgrade tensorflow-hub
```

TensorFlow Hub의 TF1 스타일 API는 TensorFlow 2의 v1 호환성 모드에서 동작합니다.

### TensorFlow 1과 함께 레거시 사용하기

TensorFlow 1.15는 `tensorflow_hub` 라이브러리에서 여전히 지원하는 TensorFlow 1.x의 유일한 버전입니다(릴리스 0.11.0 기준). TensorFlow 1.15는 기본적으로 TF1 호환 동작을 사용하지만 TensorFlow 허브의 TF2 스타일 API를 일부 사용할 수 있도록 내부에 많은 TF2 기능이 포함되어 있습니다.

```bash
$ pip install "tensorflow>=1.15,<2.0"
$ pip install --upgrade tensorflow-hub
```

### 시험판 버전 사용하기

pip 패키지 `tf-nightly` 및 `tf-hub-nightly`는 릴리스 테스트 없이 github의 소스 코드에서 자동으로 빌드됩니다. 이를 통해 개발자는 [소스에서 빌드](build_from_source.md)하지 않고도 최신 코드를 시도해 볼 수 있습니다.

```bash
$ pip install tf-nightly
$ pip install --upgrade tf-hub-nightly
```

## 다음 단계

- [라이브러리 개요](lib_overview.md)
- 튜토리얼:
    - [텍스트 분류](https://github.com/tensorflow/hub/blob/master/examples/colab/tf2_text_classification.ipynb)
    - [이미지 분류](https://github.com/tensorflow/hub/blob/master/examples/colab/tf2_image_retraining.ipynb)
    - [GitHub의](https://github.com/tensorflow/hub/blob/master/examples/README.md) 추가 예제
- [tfhub.dev](https://tfhub.dev)에서 모델 찾기
