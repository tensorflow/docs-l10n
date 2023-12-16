# 일반적인 구현 문제

본 페이지에서는 새로운 데이터세트를 구현할 때 일반적인 구현 문제를 설명합니다.

## 레거시 `SplitGenerator`는 피해야 합니다

오래된 `tfds.core.SplitGenerator` API는 사용되지 않습니다.

```python
def _split_generator(...):
  return [
      tfds.core.SplitGenerator(name='train', gen_kwargs={'path': train_path}),
      tfds.core.SplitGenerator(name='test', gen_kwargs={'path': test_path}),
  ]
```

다음으로 대체해야 합니다.

```python
def _split_generator(...):
  return {
      'train': self._generate_examples(path=train_path),
      'test': self._generate_examples(path=test_path),
  }
```

**이유**: 새로운 API는 장황하지 않고 더욱 명확합니다. 오래된 API는 추후 버전에서 삭제될 것입니다.

## 새로운 데이터세트는 폴더에 자체 포함되어야 합니다

`tensorflow_datasets/` 리포지토리 내에 데이터세트를 추가할 때, 폴더로서의 데이터세트 구조(모든 체크섬, 더미 데이터, 폴더에 자체 포함된 구현 코드)를 따르도록 하세요.

- 오래된 데이터세트 (나쁨): `<category>/<ds_name>.py`
- 새로운 데이터세트 (좋음): `<category>/<ds_name>/<ds_name>.py`

[TFDS CLI](https://www.tensorflow.org/datasets/cli#tfds_new_implementing_a_new_dataset)(`tfds new` 또는 구글러용 `gtfds new`)를 사용하여 템플릿을 생성합니다.

**이유**: 오래된 구조는 체크섬, 가짜 데이터의 절대적 경로가 필요하고 여러 위치에 데이터세트 파일을 분산시켰습니다. 이로 인해 TFDS 리포지토리 외부에 데이터세트를 구현하기 어려웠습니다. 이제 일관성을 위해 모든 위치에서 새로운 구조를 사용해야 합니다.

## 설명 목록은 마크다운 형식이어야 합니다

`DatasetInfo.description` `str`는 마크다운 형식입니다. 마크다운 목록은 첫 항목 앞에 공백 라인이 필요합니다.

```python
_DESCRIPTION = """
Some text.
                      # << Empty line here !!!
1. Item 1
2. Item 1
3. Item 1
                      # << Empty line here !!!
Some other text.
"""
```

**이유**: 형식이 잘못된 설명으로 인해 카탈로그 설명서에 시각적 결함이 생깁니다. 공백 라인이 없다면, 위의 텍스트는 다음과 같이 렌더링될 것입니다.

Some text. 1. Item 1 2. Item 1 3. Item 1 Some other text

## ClassLabel 이름을 잊었습니다

`tfds.features.ClassLabel`을 사용하는 경우, `names=` 또는 `names_file=`(`num_classes=10` 대신)을 포함한 사람이 읽을 수 있는 라벨 `str`을 제공하세요.

```python
features = {
    'label': tfds.features.ClassLabel(names=['dog', 'cat', ...]),
}
```

**이유**: 사람이 읽을 수 있는 라벨이 많은 곳에 사용됩니다.

- `str`를 `_generate_examples`에서 직접 산출 허용: `yield {'label': 'dog'}`
- `info.features['label'].names` (변환 메서드 `.str2int('dog')`,... 또한 가능)와 같은 사용자에게 노출됨
- [시각화 유틸](https://www.tensorflow.org/datasets/overview#tfdsas_dataframe) `tfds.show_examples`, `tfds.as_dataframe`에서 사용됨

## 이미지 형상을 잊었습니다

`tfds.features.Image`, `tfds.features.Video` 사용 시, 이미지가 정적 형상이라면, 다음과 같이 명시적으로 지정되어야 합니다.

```python
features = {
    'image': tfds.features.Image(shape=(256, 256, 3)),
}
```

**이유**: 이는 배치 처리에 필요한 정적 형상 추론(예: `ds.element_spec['image'].shape`)을 허용합니다(형상을 알 수 없는 이미지의 배치 처리는 우선 크기 조절이 필요함).

## `tfds.features.Tensor` 대신 보다 구체적인 유형 선호

가능하다면, 일반적인 `tfds.features.Tensor` 대신 보다 구체적인 유형인 `tfds.features.ClassLabel`, `tfds.features.BBoxFeatures`가 바람직합니다.

**이유**: 의미론적으로 더 정확할 뿐만 아니라 특정 기능은 사용자에게 추가 메타데이터를 제공하며 도구로 탐지할 수 있습니다.

## 전역 공간에서 Lazy import

전역 공간에서 Lazy import를 호출해서는 안됩니다. 예를 들어 다음 예시는 잘못되었습니다.

```python
tfds.lazy_imports.apache_beam # << Error: Import beam in the global scope

def f() -> beam.Map:
  ...
```

**이유**: 전역 범위에서 Lazy import를 사용하면 모든 tfds 사용자에 대한 모듈을 가져올 수 있으므로 Lazy import의 목적을 달성할 수 없습니다.

## 훈련/테스트 분할을 동적으로 계산

데이터세트가 공식적인 분할을 제공하지 않는 경우 TFDS도 제공하지 않아야 합니다. 다음과 같은 사항은 피해야 합니다.

```python
_TRAIN_TEST_RATIO = 0.7

def _split_generator():
  ids = list(range(num_examples))
  np.random.RandomState(seed).shuffle(ids)

  # Split train/test
  train_ids = ids[_TRAIN_TEST_RATIO * num_examples:]
  test_ids = ids[:_TRAIN_TEST_RATIO * num_examples]
  return {
      'train': self._generate_examples(train_ids),
      'test': self._generate_examples(test_ids),
  }
```

**이유**: TFDS는 원본 데이터에 가까운 데이터세트를 제공하려고 합니다. [하위 분할 API](https://www.tensorflow.org/datasets/splits)를 대신 사용하여 사용자가 원하는 하위 분할을 동적으로 만들 수 있습니다.

```python
ds_train, ds_test = tfds.load(..., split=['train[:80%]', 'train[80%:]'])
```

## Python 스타일 가이드

### pathlib API 사용 선호

`tf.io.gfile` API 대신, [pathlib API](https://docs.python.org/3/library/pathlib.html)를 사용하는 것이 좋습니다. 모든 `dl_manager` 메서드는 GCS, S3 등과 호환되는 pathlib과 같은 객체를 반환합니다.

```python
path = dl_manager.download_and_extract('http://some-website/my_data.zip')

json_path = path / 'data/file.json'

json.loads(json_path.read_text())
```

**이유**: pathlib API는 표준 양식을 없애는 현대 객체 지향 파일 API입니다. 또한 `.read_text()` / `.read_bytes()`를 사용해도 파일이 올바르게 닫힙니다.

### 메서드가 `self`를 사용하지 않는다면 함수여야 합니다

Class 메서드가 `self`를 사용하지 않는다면 단순한 함수(Class 외부에서 정의됨)여야 합니다.

**이유**: 함수가 사이드 이펙트나 숨겨진 입력/출력이 없다는 것을 분명히 합니다.

```python
x = f(y)  # Clear inputs/outputs

x = self.f(y)  # Does f depend on additional hidden variables ? Is it stateful ?
```

## Python에서 Lazy imports

우리는 TensorFlow와 같은 큰 모듈을 느리게(lazy) 가져옵니다. Lazy imports는 모듈의 실제 가져오기를 모듈의 첫 번째 사용으로 연기합니다. 따라서 이 큰 모듈이 필요하지 않은 사용자는 이러한 모듈을 가져오게 되는 일이 없게 됩니다.

```python
from tensorflow_datasets.core.utils.lazy_imports_utils import tensorflow as tf
# After this statement, TensorFlow is not imported yet

...

features = tfds.features.Image(dtype=tf.uint8)
# After using it (`tf.uint8`), TensorFlow is now imported
```

내부적으로 [`LazyModule` 클래스](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/core/utils/lazy_imports_utils.py)는 속성이 액세스될 때(`__getattr__`)만 모듈을 실제로 가져오는 팩토리 역할을 합니다.

여러분은 이를 컨텍스트 관리자와 함께 편리하게 사용할 수도 있습니다.

```python
from tensorflow_datasets.core.utils.lazy_imports_utils import lazy_imports

with lazy_imports(error_callback=..., success_callback=...):
  import some_big_module
```
