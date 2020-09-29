<!--* freshness: { owner: 'arnoegw' reviewed: '2020-07-06' } *-->

# TF Hub에서 모델 다운로드 캐싱하기

## Summary

`tensorflow_hub` 라이브러리는 모델이 tfhub.dev(또는 다른 [호스팅 사이트](hosting.md) )에서 다운로드되고 압축이 풀릴 때 파일 시스템의 모델을 캐싱합니다. 다운로드 위치는 기본적으로 로컬 임시 디렉토리이지만, 환경 변수 `TFHUB_CACHE_DIR`(권장)을 설정하거나 명령 줄 플래그 `--tfhub_cache_dir`를 전달하여 사용자 정의할 수 있습니다. 영구 위치를 사용하는 경우, 자동 정리가 없다는 점에 유의하세요.

실제 Python 코드에서 `tensorflow_hub` 함수에 대한 호출은 모델의 정식 tfhub.dev URL을 계속 사용할 수 있으며 계속 사용해야 합니다. 따라서 시스템 간에 이식 가능하고 설명서를 탐색할 수 있습니다.

## Specific execution environments

기본 `TFHUB_CACHE_DIR`을 변경해야 하는 경우와 방법은 실행 환경에 따라 다릅니다.

### 워크스테이션에서 로컬로 실행하기

워크스테이션에서 TensorFlow 프로그램을 실행하는 사용자의 경우, 대부분의 경우 기본 위치 `/tmp/tfhub_modules` 또는 Python이 `os.path.join(tempfile.gettempdir(), "tfhub_modules")`에 대해 반환하는 위치를 계속 사용하는 것이 좋습니다.

시스템 재부팅 시 영구 캐싱을 선호하는 사용자는 대신 `TFHUB_CACHE_DIR`을 홈 디렉터리의 위치로 설정할 수 있습니다. 예를 들어, Linux 시스템에서 bash 셸 사용자는 다음과 같은 줄을 `~/.bashrc`에 추가할 수 있습니다.

```bash
export TFHUB_CACHE_DIR=$HOME/.cache/tfhub_modules
```

...셸을 다시 시작하면, 이 위치가 사용됩니다.

### Colab 노트북의 TPU에서 실행하기

[Colab](https://colab.research.google.com/) 노트북 내 CPU 및 GPU에서 TensorFlow를 실행하려면, 기본 로컬 캐시 위치를 사용하면 됩니다.

TPU에서 실행하면 기본 로컬 캐시 위치에 대한 액세스 권한이 없는 다른 머신에 위임됩니다. 자체 Google Cloud Storage(GCS) 버킷이 있는 사용자는 다음과 같은 코드를 사용하여 해당 버킷의 디렉토리를 캐시 위치로 설정하여 이 문제를 해결할 수 있습니다.

```python
import os
os.environ["TFHUB_CACHE_DIR"] = "gs://my-bucket/tfhub-modules-cache"
```

...`tensorflow_hub` 라이브러리를 호출하기 전에.
