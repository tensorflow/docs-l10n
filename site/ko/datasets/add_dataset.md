# 사용자 정의 데이터세트 작성하기

이 가이드에 따라 새 데이터세트를 생성하세요(TFDS 또는 자체 저장소 이용).

원하는 데이터세트가 이미 있는지 확인하려면 [데이터세트 목록](catalog/overview.md)을 확인하세요.

## TL;DR

새 데이터세트를 작성하는 가장 쉬운 방법은 [TFDS CLI](https://www.tensorflow.org/datasets/cli)를 사용하는 것입니다.

```sh
cd path/to/my/project/datasets/
tfds new my_dataset  # Create `my_dataset/my_dataset.py` template files
# [...] Manually modify `my_dataset/my_dataset.py` to implement your dataset.
cd my_dataset/
tfds build  # Download and prepare the dataset to `~/tensorflow_datasets/`
```

`tfds.load('my_dataset')`와 함께 새 데이터세트를 사용하려면 다음을 수행합니다.

- `tfds.load`가 `~/tensorflow_datasets/my_dataset/`에서 생성된(예: `tfds build`를 통해) 데이터세트를 자동으로 감지하고 로드합니다.
- 또는, `my.project.datasets.my_dataset`를 명시적으로 가져와 데이터세트를 등록할 수 있습니다.

```python
import my.project.datasets.my_dataset  # Register `my_dataset`

ds = tfds.load('my_dataset')  # `my_dataset` registered
```

## 개요

데이터세트는 모든 종류의 형식으로 모든 장소에 배포되며, 항상 머신러닝 파이프라인에 공급할 수 있는 형식으로 저장되는 것은 아닙니다. TFDS를 입력하세요.

TFDS는 이러한 데이터세트를 표준 형식(외부 데이터 -&gt; 직렬화된 파일)으로 처리한 다음 머신 러닝 파이프라인(직렬화된 파일 -&gt; `tf.data.Dataset`)으로 로드할 수 있습니다. 직렬화는 한 번만 수행됩니다. 이후 액세스 때는 이러한 사전 처리된 파일에서 직접 읽습니다.

대부분의 전처리는 자동으로 수행됩니다. 각 데이터세트는 다음을 지정하는 `tfds.core.DatasetBuilder`의 서브 클래스를 구현합니다.

- 데이터의 출처(예: URL)
- 데이터세트의 모습(즉, 특성)
- 데이터 분할 방법(예: `TRAIN` 및 `TEST` )
- 데이터세트의 개별 예

## 데이터세트 작성하기

### 기본 템플릿: `tfds new`

[TFDS CLI](https://www.tensorflow.org/datasets/cli)를 사용하여 필요한 템플릿 Python 파일을 생성합니다.

```sh
cd path/to/project/datasets/  # Or use `--dir=path/to/project/datasets/` below
tfds new my_dataset
```

이 명령으로 다음 구조를 가진 새로운 `my_dataset/`가 생성됩니다.

```sh
my_dataset/
    __init__.py
    my_dataset.py # Dataset definition
    my_dataset_test.py # (optional) Test
    dummy_data/ # (optional) Fake data (used for testing)
    checksum.tsv # (optional) URL checksums (see `checksums` section).
```

여기에서 `TODO(my_dataset)`를 검색하고 그에 따라 수정합니다.

### 데이터세트 예제

모든 데이터세트는 대부분의 상용구를 담당하는 `tfds.core.DatasetBuilder`의 서브 클래스인 `tfds.core.GeneratorBasedBuilder`로 구현됩니다.

- 단일 머신에서 생성할 수 있는 작은/중간 크기의 데이터세트(이 튜토리얼)
- 분산 생성이 필요한 매우 큰 데이터세트([Apache Beam](https://beam.apache.org/)을 사용하며, [방대한 데이터세트 가이드](https://www.tensorflow.org/datasets/beam_datasets#implementing_a_beam_dataset) 참조)

다음은 데이터세트 클래스의 최소 예입니다.

```python
class MyDataset(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for my_dataset dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Dataset metadata (homepage, citation,...)."""
    return tfds.core.DatasetInfo(
        builder=self,
        features=tfds.features.FeaturesDict({
            'image': tfds.features.Image(shape=(256, 256, 3)),
            'label': tfds.features.ClassLabel(
                names=['no', 'yes'],
                doc='Whether this is a picture of a cat'),
        }),
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Download the data and define splits."""
    extracted_path = dl_manager.download_and_extract('http://data.org/data.zip')
    # dl_manager returns pathlib-like objects with `path.read_text()`,
    # `path.iterdir()`,...
    return {
        'train': self._generate_examples(path=extracted_path / 'train_images'),
        'test': self._generate_examples(path=extracted_path / 'test_images'),
    }

  def _generate_examples(self, path) -> Iterator[Tuple[Key, Example]]:
    """Generator of examples for each split."""
    for img_path in path.glob('*.jpeg'):
      # Yields (key, example)
      yield img_path.name, {
          'image': img_path,
          'label': 'yes' if img_path.name.startswith('yes_') else 'no',
      }
```

덮어쓰기를 위한 3가지 추상 메서드에 대해 자세히 알아보겠습니다.

### `_info`: 데이터세트 메타데이터

`_info`는 [데이터세트 메타데이터](https://www.tensorflow.org/datasets/overview#access_the_dataset_metadata)를 포함하는 `tfds.core.DatasetInfo`를 반환합니다.

```python
def _info(self):
  return tfds.core.DatasetInfo(
      builder=self,
      # Description and homepage used for documentation
      description="""
      Markdown description of the dataset. The text will be automatically
      stripped and dedent.
      """,
      homepage='https://dataset-homepage.org',
      features=tfds.features.FeaturesDict({
          'image_description': tfds.features.Text(),
          'image': tfds.features.Image(),
          # Here, 'label' can be 0-4.
          'label': tfds.features.ClassLabel(num_classes=5),
      }),
      # If there's a common `(input, target)` tuple from the features,
      # specify them here. They'll be used if as_supervised=True in
      # builder.as_dataset.
      supervised_keys=('image', 'label'),
      # Specify whether to disable shuffling on the examples. Set to False by default.
      disable_shuffling=False,
      # Bibtex citation for the dataset
      citation=r"""
      @article{my-awesome-dataset-2020,
               author = {Smith, John},}
      """,
  )
```

대부분의 필드는 자체적으로 명확해야 합니다. 일부 정밀도:

- `features`: 데이터세트 구조, 형상 등을 지정합니다. 복잡한 데이터 형식(오디오, 비디오, 중첩 시퀀스 등)을 지원합니다. 자세한 내용은 [사용 가능한 기능](https://www.tensorflow.org/datasets/api_docs/python/tfds/features#classes) 또는 [기능 커넥터 가이드](https://www.tensorflow.org/datasets/features)를 참조하세요.
- `disable_shuffling`: [데이터세트 순서 유지](#maintain-dataset-order) 섹션을 참조하세요.
- `citation`: `BibText` 인용을 찾으려면:
    - 데이터세트 웹사이트에서 인용 지침을 검색합니다(BibTex 형식으로 사용).
    - [arXiv](https://arxiv.org/) 논문의 경우: 논문을 찾아 오른쪽에 있는 `BibText` 링크를 클릭합니다.
    - [Google Scholar](https://scholar.google.com)에서 논문을 찾아 제목 아래에 있는 큰따옴표를 클릭하고 팝업에서 `BibTeX`를 클릭합니다.
    - 관련 논문이 없으면(예를 들어, 웹 사이트만 있음), [BibTeX 온라인 편집기](https://truben.no/latex/bibtex/)를 사용하여 사용자 정의 BibTeX 항목을 작성할 수 있습니다(드롭다운 메뉴에 `Online` 항목 유형이 있음).

#### 데이터세트 순서 유지

동일한 클래스에 속하는 레코드가 인접하는 경우가 많기 때문에 데이터세트 전체에서 클래스 분포를 더 균일하게 만들기 위해 저장할 때 기본적으로 데이터세트의 레코드가 섞입니다. `_generate_examples`에서 제공하는 생성된 키로 데이터세트가 분류되도록 지정하려면 `disable_shuffling` 필드를 `True`로 설정해야 합니다. 기본적으로, `False`로 설정됩니다.

```python
def _info(self):
  return tfds.core.DatasetInfo(
    # [...]
    disable_shuffling=True,
    # [...]
  )
```

셔플을 비활성화하면 샤드를 더 이상 병렬로 읽을 수 없으므로 성능에 영향을 미칩니다.

### `_split_generators`: 데이터 다운로드 및 분할

#### 소스 데이터 다운로드 및 추출하기

대부분의 데이터세트는 웹에서 데이터를 다운로드해야 합니다. 이 작업은 `_split_generators`의 `tfds.download.DownloadManager` 입력 인수를 사용하여 수행됩니다. `dl_manager`는 다음 메서드를 가지고 있습니다.

- `download`: `http(s)://`, `ftp(s)://`를 지원합니다.
- `extract`: 현재 `.zip`, `.gz` 및 `.tar` 파일을 지원합니다.
- `download_and_extract`: `dl_manager.extract(dl_manager.download(urls))`와 동일합니다.

이러한 모든 메서드는 [pathlib.Path-like](https://docs.python.org/3/library/pathlib.html) 개체인 `tfds.core.Path`([`epath.Path`](https://github.com/google/etils)의 별칭)를 반환합니다.

이러한 메서드는 다음과 같은 임의의 중첩 구조(`list`, `dict`)를 지원합니다.

```python
extracted_paths = dl_manager.download_and_extract({
    'foo': 'https://example.com/foo.zip',
    'bar': 'https://example.com/bar.zip',
})
# This returns:
assert extracted_paths == {
    'foo': Path('/path/to/extracted_foo/'),
    'bar': Path('/path/extracted_bar/'),
}
```

#### 수동 다운로드 및 추출

일부 데이터는 자동으로 다운로드할 수 없습니다(예: 로그인 필요). 이 경우 사용자는 수동으로 소스 데이터를 다운로드하여 `manual_dir/`(기본적으로 `~/tensorflow_datasets/downloads/manual/`)에 놓을 수 있습니다.

그러면 `dl_manager.manual_dir`를 통해 파일에 액세스할 수 있습니다.

```python
class MyDataset(tfds.core.GeneratorBasedBuilder):

  MANUAL_DOWNLOAD_INSTRUCTIONS = """
  Register into https://example.org/login to get the data. Place the `data.zip`
  file in the `manual_dir/`.
  """

  def _split_generators(self, dl_manager):
    # data_path is a pathlib-like `Path('<manual_dir>/data.zip')`
    archive_path = dl_manager.manual_dir / 'data.zip'
    # Extract the manually downloaded `data.zip`
    extracted_path = dl_manager.extract(archive_path)
    ...
```

`manual_dir` 위치는 `tfds build --manual_dir=` 또는 `tfds.download.DownloadConfig`를 사용하여 사용자 정의할 수 있습니다.

#### 아카이브 직접 읽기

`dl_manager.iter_archive`는 압축을 풀지 않고 순차적으로 아카이브를 읽습니다. 이것은 저장 공간을 절약하고 일부 파일 시스템의 성능을 향상시킬 수 있습니다.

```python
for filename, fobj in dl_manager.iter_archive('path/to/archive.zip'):
  ...
```

`fobj`에는 `with open('rb') as fobj:`와 동일한 메서드가 있습니다(예: `fobj.read()`).

#### 데이터세트 분할 지정하기

데이터세트에 사전 정의된 분할이 함께 제공되는 경우(예: `MNIST`에 `train` 및 `test` 분할이 있음) 이를 유지합니다. 그렇지 않으면 단일 `tfds.Split.TRAIN` 분할만 지정합니다. 사용자는 [subsplit API](https://www.tensorflow.org/datasets/splits)(예: `split='train[80%:]'`)를 사용하여 고유한 하위 분할을 동적으로 생성할 수 있습니다.

```python
def _split_generators(self, dl_manager):
  # Download source data
  extracted_path = dl_manager.download_and_extract(...)

  # Specify the splits
  return {
      'train': self._generate_examples(
          images_path=extracted_path / 'train_imgs',
          label_path=extracted_path / 'train_labels.csv',
      ),
      'test': self._generate_examples(
          images_path=extracted_path / 'test_imgs',
          label_path=extracted_path / 'test_labels.csv',
      ),
  }
```

### `_generate_examples`: 예제 생성기

`_generate_examples`는 소스 데이터에서 각 분할에 대한 예제를 생성합니다.

이 메서드는 일반적으로 소스 데이터세트 아티팩트(예: CSV 파일)를 판독하고 `(key, feature_dict)` 튜플을 산출합니다.

- `key`: 예시 식별자. `hash(key)` 사용하여 예제를 결정성 있게 셔플하거나 셔플이 비활성화된 경우 키별로 정렬하는 데 사용됩니다([데이터세트 순서 유지](#maintain-dataset-order) 섹션 참조). 다음가 같아야 합니다.
    - **고유함**: 두 예제가 동일한 키를 사용하면 예외가 발생합니다.
    - **결정성이 있음**: `download_dir`, `os.path.listdir` 순서에 의존하지 않아야 합니다. 데이터를 두 번 생성하면 동일한 키가 산출됩니다.
    - **비교 가능**: 셔플링이 비활성화된 경우 키가 데이터세트를 정렬하는 데 사용됩니다.
- `feature_dict`: 예제 값을 포함한 `dict`
    - 구조는 `tfds.core.DatasetInfo`에 정의된 `features=` 구조와 일치해야 합니다.
    - 복잡한 데이터 형식(이미지, 비디오, 오디오 등)은 자동으로 인코딩됩니다.
    - 각 기능은 종종 여러 입력 유형을 허용합니다(예: 비디오는 `/path/to/vid.mp4`, `np.array(shape=(l, h, w, c))`, `List[paths]`, `List[np.array(shape=(h, w, c)]`, `List[img_bytes]` 등을 수용함).
    - 자세한 내용은 [기능 커넥터 가이드](https://www.tensorflow.org/datasets/features)를 참조하세요.

```python
def _generate_examples(self, images_path, label_path):
  # Read the input data out of the source files
  with label_path.open() as f:
    for row in csv.DictReader(f):
      image_id = row['image_id']
      # And yield (key, feature_dict)
      yield image_id, {
          'image_description': row['description'],
          'image': images_path / f'{image_id}.jpeg',
          'label': row['label'],
      }
```

#### 파일 액세스 및 `tf.io.gfile`

클라우드 스토리지 시스템을 지원하려면 Python 내장 I/O ops를 사용하지 마세요.

대신 `dl_manager`는 Google Cloud Storage와 직접 호환되는 [pathlib 유사](https://docs.python.org/3/library/pathlib.html) 객체를 반환합니다.

```python
path = dl_manager.download_and_extract('http://some-website/my_data.zip')

json_path = path / 'data/file.json'

json.loads(json_path.read_text())
```

또는, 파일 연산에 내장된 API 대신 `tf.io.gfile` API를 사용합니다.

- `open` -&gt; `tf.io.gfile.GFile`
- `os.rename` -&gt; `tf.io.gfile.rename`
- ...

`tf.io.gfile`보다는 Pathlib이 선호됩니다([이론적 근거 ](https://www.tensorflow.org/datasets/common_gotchas#prefer_to_use_pathlib_api) 참조).

#### 추가 종속성

일부 데이터세트에는 생성 중에만 추가 Python 종속성이 필요합니다. 예를 들어 SVHN 데이터세트는 `scipy`를 사용하여 일부 데이터를 로드합니다.

TFDS 저장소에 데이터세트를 추가하는 경우, `tfds.core.lazy_imports`를 사용하여 `tensorflow-datasets` 패키지를 작게 유지하세요. 사용자는 필요한 경우에만 추가 종속성을 설치합니다.

`lazy_imports`를 사용하려면:

- [`setup.py`](https://github.com/tensorflow/datasets/tree/master/setup.py)에서 데이터세트의 항목을 `DATASET_EXTRAS`에 추가합니다. 이를 통해 사용자는 예를 들어 `pip install 'tensorflow-datasets[svhn]'`을 실행하여 추가 종속성을 설치할 수 있습니다.
- 가져오기의 항목을 [`LazyImporter`](https://github.com/tensorflow/datasets/tree/master/tensorflow_datasets/core/lazy_imports_lib.py)와 [`LazyImportsTest`](https://github.com/tensorflow/datasets/tree/master/tensorflow_datasets/core/lazy_imports_lib_test.py)에 추가합니다.
- `tfds.core.lazy_imports`를 사용하여`DatasetBuilder`에서 종속성(예를 들어, `tfds.core.lazy_imports.scipy`)에 액세스합니다.

#### 손상된 데이터

일부 데이터세트는 완벽하게 정리되지 않았으며, 일부 손상된 데이터(예: 이미지는 JPEG 파일이지만, 일부는 유효하지 않은 JPEG일 때)를 포함합니다. 이들 예제는 건너뛰어야 하지만, 데이터세트 설명에 몇 개의 예제가 삭제되었으며 그 이유는 무엇인지 메모를 남겨 주세요.

### 데이터세트 구성/변형(tfds.core.BuilderConfig)

일부 데이터세트에는 데이터가 사전 처리되고 디스크에 기록되는 방식에 대한 여러 가지 변형 또는 옵션이 있을 수 있습니다. 예를 들어, [cycle_gan](https://www.tensorflow.org/datasets/catalog/cycle_gan)에는 객체 쌍별로 하나의 구성이 있습니다(`cycle_gan/horse2zebra`, `cycle_gan/monet2photo`,...)

이는 `tfds.core.BuilderConfig`를 통해 수행됩니다.

1. 자신의 구성 객체를 `tfds.core.BuilderConfig`의 서브 클래스로 정의합니다. 예를 들면, `MyDatasetConfig`와 같습니다.

    ```python
    @dataclasses.dataclass
    class MyDatasetConfig(tfds.core.BuilderConfig):
      img_size: Tuple[int, int] = (0, 0)
    ```

    참고: https://bugs.python.org/issue33129로 인해 기본값이 필요합니다.

2. 데이터세트가 노출하는 `MyDatasetConfig`를 나열하는 `MyDataset`에서 `BUILDER_CONFIGS = []` 클래스 구성원을 정의합니다.

    ```python
    class MyDataset(tfds.core.GeneratorBasedBuilder):
      VERSION = tfds.core.Version('1.0.0')
      # pytype: disable=wrong-keyword-args
      BUILDER_CONFIGS = [
          # `name` (and optionally `description`) are required for each config
          MyDatasetConfig(name='small', description='Small ...', img_size=(8, 8)),
          MyDatasetConfig(name='big', description='Big ...', img_size=(32, 32)),
      ]
      # pytype: enable=wrong-keyword-args
    ```

    참고: 데이터 클래스 상속을 갖는 [Pytype 버그](https://github.com/google/pytype/issues/628)로 인해 `# pytype: disable=wrong-keyword-args`가 필요합니다.

3. `MyDataset`의 `self.builder_config`를 사용하여 데이터 생성을 구성합니다(예:`shape=self.builder_config.img_size`). 여기에는 `_info()`에서 여러 값을 설정하거나 다운로드 데이터 액세스를 변경하는 것이 포함될 수 있습니다.

참고:

- 각 구성에는 고유한 이름이 있습니다. 구성의 정규화된 이름은 `dataset_name/config_name`입니다(예: `coco/2017`).
- 지정되지 않은 경우 `BUILDER_CONFIGS`의 첫 번째 구성이 사용됩니다(예: `tfds.load('c4')`의 기본값은 `c4/en`).

`BuilderConfig`를 사용하는 데이터세트의 예는 [`anli`](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/text/anli.py#L69)를 참조하세요.

### 버전

버전은 두 가지 다른 의미를 나타낼 수 있습니다.

- "외부" 원본 데이터 버전: 예: COCO v2019, v2017,...
- "내부" TFDS 코드 버전: 예를 들어 `tfds.features.FeaturesDict`에서 이름 바꾸기, `_generate_examples`에서 버그 수정

데이터세트를 업데이트하려면:

- "외부" 데이터 업데이트의 경우: 여러 사용자가 특정 연도/버전에 동시에 액세스하기를 원할 수 있습니다. 이것은 버전당 하나의 `tfds.core.BuilderConfig`(예: `coco/2017`, `coco/2019`) 또는 버전당 하나의 클래스(예: `Voc2007`, `Voc2012`)를 사용하여 수행됩니다.
- "내부" 코드 업데이트의 경우: 사용자는 최신 버전만 다운로드합니다. 코드를 업데이트하면 [의미 체계 버전 관리](https://www.tensorflow.org/datasets/datasets_versioning#semantic)에 따라 `VERSION` 클래스 속성이 증가합니다(예: `1.0.0`에서 `VERSION = tfds.core.Version('2.0.0')`로 증가).

### 등록을 위해 가져오기 추가

데이터세트 모듈을 프로젝트 `__init__`로 가져와 `tfds.load`, `tfds.builder`에 자동으로 등록되도록 하는 것을 잊지 마세요.

```python
import my_project.datasets.my_dataset  # Register MyDataset

ds = tfds.load('my_dataset')  # MyDataset available
```

예를 들어, `tensorflow/datasets`에 제공하는 경우, 해당 하위 디렉토리의 `__init__.py`에 모듈 가져오기를 추가합니다(예: [`image/__init__.py`](https://github.com/tensorflow/datasets/tree/master/tensorflow_datasets/image/__init__.py).

### 일반적인 구현 문제 점검하기

[일반적인 구현 문제](https://www.tensorflow.org/datasets/common_gotchas)가 있는지 확인하세요.

## 데이터세트 테스트하기

### 다운로드 및 준비: `tfds build`

데이터세트를 생성하려면 `my_dataset/` 디렉토리에서 `tfds build`를 실행합니다.

```sh
cd path/to/datasets/my_dataset/
tfds build --register_checksums
```

개발을 위한 몇 가지 유용한 플래그:

- `--pdb`: 예외가 발생하면 디버깅 모드로 들어갑니다.
- `--overwrite`: 데이터세트가 이미 생성된 경우 기존 파일을 삭제합니다.
- `--max_examples_per_split`: 전체 데이터세트가 아닌 처음 X개 예제(기본값은 1)만 생성합니다.
- `--register_checksums`: 다운로드한 URL의 체크섬을 기록합니다. 개발 중에만 사용해야 합니다.

플래그의 전체 목록은 [CLI 설명서](https://www.tensorflow.org/datasets/cli#tfds_build_download_and_prepare_a_dataset)를 참조하세요.

### 체크섬

결정성을 보장하고 문서화를 돕기 위해 데이터세트의 체크섬을 기록하는 것이 좋습니다. 이를 위해 `--register_checksums`로 데이터세트를 생성합니다(이전 섹션 참조).

PyPI를 통해 데이터세트를 릴리스하는 경우 `checksums.tsv` 파일을 내보내는 것을 잊지 마세요(예: `setup.py`의 `package_data`에).

### 데이터세트 단위 테스트

`tfds.testing.DatasetBuilderTestCase`는 데이터세트를 완전히 실행해보기 위한 기본 `TestCase`입니다. 이 때 "더미 예제"를 소스 데이터세트의 구조를 모방한 테스트 데이터로 사용합니다.

- 테스트 데이터는 `my_dataset/dummy_data/` 디렉토리에 넣어야 하며 다운로드 및 추출된 소스 데이터세트 아티팩트를 모방해야 합니다. 스크립트([예제 스크립트](https://github.com/tensorflow/datasets/tree/master/tensorflow_datasets/image/bccd/dummy_data_generation.py))를 사용하여 이 데이터를 수동 또는 자동으로 생성할 수 있습니다.
- 데이터세트가 겹치면 테스트가 실패하므로 테스트 데이터 분할에 서로 다른 데이터를 사용해야 합니다.
- **테스트 데이터에는 저작권이 있는 자료가 포함되어서는 안 됩니다**. 의심스러운 경우, 원래 데이터세트의 자료를 사용하여 데이터를 생성하지 마세요.

```python
import tensorflow_datasets as tfds
from . import my_dataset


class MyDatasetTest(tfds.testing.DatasetBuilderTestCase):
  """Tests for my_dataset dataset."""
  DATASET_CLASS = my_dataset.MyDataset
  SPLITS = {
      'train': 3,  # Number of fake train example
      'test': 1,  # Number of fake test example
  }

  # If you are calling `download/download_and_extract` with a dict, like:
  #   dl_manager.download({'some_key': 'http://a.org/out.txt', ...})
  # then the tests needs to provide the fake output paths relative to the
  # fake data directory
  DL_EXTRACT_RESULT = {
      'name1': 'path/to/file1',  # Relative to dummy_data/my_dataset dir.
      'name2': 'file2',
  }


if __name__ == '__main__':
  tfds.testing.test_main()
```

다음 명령어를 실행하여 데이터세트를 테스트합니다.

```sh
python my_dataset_test.py
```

## 피드백 보내기

데이터세트 생성 워크플로를 개선하기 위해 지속적으로 노력하고 있지만 문제점을 알아야만 그렇게 할 수 있습니다. 데이터세트를 생성하는 동안 어떤 문제, 오류를 경험하셨나요? 혼란스럽거나 상용구이거나 아예 작동하지 않는 부분이 있었나요? [github에서 피드백](https://github.com/tensorflow/datasets/issues)을 공유해 주세요.
