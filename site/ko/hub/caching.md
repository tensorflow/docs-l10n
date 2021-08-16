<!--* freshness: { owner: 'wgierke' reviewed: '2021-07-28' } *-->

# TF Hub에서 모델 다운로드 캐싱하기

## 개요

`tensorflow_hub` 라이브러리는 현재 모델 다운로드를 위한 두 가지 모드를 지원합니다. 기본적으로, 모델은 압축된 아카이브로 다운로드되고 디스크에 캐시됩니다. 둘째, 원격 저장소에서 TensorFlow로 직접 모델을 읽을 수 있습니다. 어느 쪽이든 실제 Python 코드에서 `tensorflow_hub` 함수에 대한 호출은 시스템 간에 이식 가능하고 문서화를 위해 탐색할 수 있는 모델의 표준 tfhub.dev URL을 계속 사용할 수 있어야 하고, 또 그래야만 합니다. 드물게 사용자 코드에 실제 파일 시스템 위치가 필요한 경우(다운로드 및 압축 해제 후 또는 모델 핸들을 파일 시스템 경로로 확인한 후) `hub.resolve(handle)` 함수로 가져올 수 있습니다.

### 압축된 다운로드 캐싱하기

`tensorflow_hub` 라이브러리는 기본적으로 tfhub.dev(또는 다른 [호스팅 사이트](hosting.md))에서 다운로드하고 압축을 풀 때 파일 시스템에 모델을 캐시합니다. 이 모드는 디스크 공간이 부족하지만 네트워크 대역폭과 대기 시간이 뛰어난 경우를 제외하고 대부분의 환경에 권장됩니다.

다운로드 위치는 기본적으로 로컬 임시 디렉터리이지만 환경 변수 `TFHUB_CACHE_DIR`을 설정하거나(권장) 명령줄 플래그 `--tfhub_cache_dir`을 전달하여 사용자 정의할 수 있습니다. 대부분의 경우 기본 캐시 위치 `/tmp/tfhub_modules`(또는 `os.path.join(tempfile.gettempdir(), "tfhub_modules")`가 평가되는 모든 위치)가 작동합니다.

시스템 재부팅 시 영구 캐싱을 선호하는 사용자는 대신 `TFHUB_CACHE_DIR`을 홈 디렉터리의 위치로 설정할 수 있습니다. 예를 들어, Linux 시스템에서 bash 셸 사용자는 다음과 같은 줄을 `~/.bashrc`에 추가할 수 있습니다.

```bash
export TFHUB_CACHE_DIR=$HOME/.cache/tfhub_modules
```

셸을 다시 시작하면 이 위치가 사용됩니다. 영구 위치를 사용하는 경우 자동 정리가 없다는 점에 유의하세요.

### 원격 저장소에서 읽기

`tensorflow_hub` 라이브러리가 다음을 사용하여 모델을 로컬로 다운로드하는 대신 원격 저장소(GCS)에서 직접 모델을 읽도록 지시할 수 있습니다.

```shell
os.environ["TFHUB_MODEL_LOAD_FORMAT"] = "UNCOMPRESSED"
```

또는 명령줄 플래그 `--tfhub_model_load_format`을 `UNCOMPRESSED`로 설정합니다. 이렇게 하면 캐싱 디렉터리가 필요하지 않습니다. 이는 특히 디스크 공간이 적지만 인터넷 연결이 빠른 환경에서 유용합니다.

### Colab 노트북의 TPU에서 실행하기

[colab.research.google.com](https://colab.research.google.com)에서는 계산 작업 부하가 기본적으로 캐시 위치에 액세스할 수 없는 다른 머신에 위임되기 때문에 압축된 모델을 다운로드하면 TPU 런타임과 충돌합니다. 이 상황에 대한 두 가지 해결 방법이 있습니다.

#### 1) TPU 작업자가 액세스할 수 있는 GCS 버킷 사용

가장 쉬운 해결책은 위의 설명과 같이 TF Hub의 GCS 버킷에서 모델을 읽도록 `tensorflow_hub` 라이브러리에 지시하는 것입니다. 자체 GCS 버킷이 있는 사용자는 대신 다음과 같은 코드를 사용하여 버킷의 디렉터리를 캐시 위치로 지정할 수 있습니다.

```python
import os
os.environ["TFHUB_CACHE_DIR"] = "gs://my-bucket/tfhub-modules-cache"
```

`tensorflow_hub` 라이브러리를 호출하기 전에 이렇게 해야 합니다.

#### 2) Colab 호스트를 통해 모든 읽기 리디렉션

또 다른 해결 방법은 Colab 호스트를 통해 모든 읽기(큰 변수 포함)를 리디렉션하는 것입니다.

```python
load_options =
tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')
reloaded_model = hub.load("https://tfhub.dev/...", options=load_options)
```
