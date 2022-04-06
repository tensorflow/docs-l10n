<!--* freshness: { owner: 'maringeo' reviewed: '2021-10-10' review_interval: '6 months' } *-->

# 일반적인 문제

문제를 여기에서 찾을 수 없는 경우, 새 문제를 등록하기 전에 [github 문제](https://github.com/tensorflow/hub/issues)를 검색하세요.

## TypeError: 'AutoTrackable' 객체를 호출할 수 없습니다.

```python
# BAD: Raises error
embed = hub.load('https://tfhub.dev/google/nnlm-en-dim128/1')
embed(['my text', 'batch'])
```

이 오류는 TF2에서 `hub.load()` API를 사용하여 TF1 Hub 형식으로 모델을 로드할 때 자주 발생합니다. 올바른 서명을 추가하면 이 문제가 해결됩니다. TF2로의 전환 및 TF2에서 TF1 Hub 형식의 모델을 사용하는 방법에 대한 자세한 내용은 [TF2용 TF-Hub 마이그레이션 가이드](migration_tf2.md)를 참조하세요.

```python

embed = hub.load('https://tfhub.dev/google/nnlm-en-dim128/1')
embed.signatures['default'](['my text', 'batch'])
```

## 모듈을 다운로드할 수 없습니다.

URL에서 모듈을 사용하는 과정에서 네트워크 스택으로 인해 나타날 수 있는 많은 오류가 있습니다. 종종 이것은 라이브러리의 문제가 아니라 코드를 실행하는 컴퓨터와 관련된 문제입니다. 다음은 일반적인 오류입니다.

- **"프로토콜을 위반하여 EOF가 발생했습니다."** - 설치된 Python 버전이 모듈을 호스팅하는 서버의 TLS 요구 사항을 지원하지 않는 경우, 이 문제가 발생할 수 있습니다. 특히, python 2.7.5는 tfhub.dev 도메인에서 모듈을 확인하지 못하는 것으로 알려져 있습니다. **해결 방법**: 최신 Python 버전으로 업데이트하세요.

- **"tfhub.dev의 인증서를 확인할 수 없습니다."** - 이 문제는 네트워크의 무언가가 dev gTLD로 작동하려고 할 때 발생할 수 있습니다. .dev가 gTLD로 사용되기 전에는 개발자와 프레임워크가 코드 테스트를 위해 때때로 .dev 이름을 사용했습니다. **해결 방법**: ".dev" 도메인에서 이름 확인을 가로채는 소프트웨어를 식별 및 재구성합니다.

- 캐시 디렉토리 `/tmp/tfhub_modules`(또는 유사)에 쓰기 실패: 캐싱 대상 및 위치를 변경하는 방법은 [캐싱](caching.md)을 참조하세요.

위의 오류 및 해결 방법이 효과가 없으면, `?tf-hub-format=compressed`를 URL에 연결하는 프로토콜을 시뮬레이션하여 수동으로 모듈을 다운로드하여 로컬 파일로 수동으로 압축 해제해야 하는 tar 압축 파일을 다운로드할 수 있습니다. 그런 다음 URL 대신 로컬 파일의 경로를 사용할 수 있습니다. 다음은 간단한 예입니다.

```bash
# Create a folder for the TF hub module.
$ mkdir /tmp/moduleA
# Download the module, and uncompress it to the destination folder. You might want to do this manually.
$ curl -L "https://tfhub.dev/google/universal-sentence-encoder/2?tf-hub-format=compressed" | tar -zxvC /tmp/moduleA
# Test to make sure it works.
$ python
> import tensorflow_hub as hub
> hub.Module("/tmp/moduleA")
```

## 사전 초기화된 모듈에서 추론 실행하기

입력 데이터에 모듈을 여러 번 적용하는 Python 프로그램을 작성하는 경우 다음 레시피를 적용할 수 있습니다. (참고: 프로덕션 서비스에서 요청을 처리하려면 [TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving) 또는 기타 확장 가능한 Python이 없는 솔루션을 고려하세요.)

사용 사례 모델이 **초기화** 및 후속 **요청**(예: Django, Flask, 사용자 정의 HTTP 서버 등)이라고 가정하면, 다음과 같이 서비스를 설정할 수 있습니다.

### TF2 SavedModels

- 초기화 부분에서:
    - TF2.0 모델을 로드합니다.

```python
import tensorflow_hub as hub

embedding_fn = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
```

- 요청 부분에서:
    - embedding 함수를 사용하여 추론을 실행합니다.

```python
embedding_fn(["Hello world"])
```

tf.function의 호출은 성능에 최적화되어 있습니다. [tf.function 가이드](https://www.tensorflow.org/guide/function)를 참조하세요.

### TF1 Hub 모듈

- 초기화 부분에서:
    - **자리 표시자**(그래프의 진입점)를 사용하여 그래프를 작성합니다.
    - 세션을 초기화합니다.

```python
import tensorflow as tf
import tensorflow_hub as hub

# Create graph and finalize (finalizing optional but recommended).
g = tf.Graph()
with g.as_default():
  # We will be feeding 1D tensors of text into the graph.
  text_input = tf.placeholder(dtype=tf.string, shape=[None])
  embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")
  embedded_text = embed(text_input)
  init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
g.finalize()

# Create session and initialize.
session = tf.Session(graph=g)
session.run(init_op)
```

- 요청 부분에서:
    - 세션을 사용하여 자리 표시자를 통해 그래프에 데이터를 공급합니다.

```python
result = session.run(embedded_text, feed_dict={text_input: ["Hello world"]})
```

## 모델의 dtype을 변경할 수 없습니다(예: float32에서 bfloat16으로).

TensorFlow의 SavedModels(TF Hub 또는 기타에서 공유됨)에는 고정 데이터 유형(신경망의 가중치 및 중간 활성화를 위한 float32)에서 동작하는 연산이 포함됩니다. 고정 데이터 유형은 SavedMode을 로드한 후에 변경할 수 없습니다. (그러나 모델 게시자는 데이터 유형이 다른 여러 모델을 게시하도록 선택할 수 있습니다.)

## 모델 버전 업데이트하기

모델 버전의 문서 메타데이터를 업데이트할 수 있습니다. 그러나 버전의 자산(모델 파일)은 변경할 수 없습니다. 모델 자산을 변경하려면 새 버전의 모델을 게시할 수 있습니다. 버전 간에 변경된 사항을 설명하는 변경 로그로 문서를 확장하는 것이 좋습니다.
