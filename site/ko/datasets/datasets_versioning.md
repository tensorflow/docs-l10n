# 데이터세트의 버전 관리

## 정의

버전 관리는 다른 의미를 나타낼 수 있습니다.

- TFDS API 버전(pip 버전): `tfds.__version__`
- TFDS와 독립적인 공개 데이터세트 버전(예: [Voc2007](https://pjreddie.com/projects/pascal-voc-dataset-mirror/), Voc2012). TFDS에서 각 공개 데이터세트 버전은 독립 데이터세트로 구현되어야 합니다.
    - [빌더 구성](https://www.tensorflow.org/datasets/add_dataset#dataset_configurationvariants_tfdscorebuilderconfig)을 통해: 예를 들어 `voc/2007`, `voc/2012`
    - 두 개의 독립적인 데이터세트: 예를 들어 `wmt13_translate`, `wmt14_translate`
- TFDS의 데이터세트 생성 코드 버전(`my_dataset:1.0.0`): 예를 들어, `voc/2007`의 TFDS 구현에서 버그가 발견되면 `voc.py` 생성 코드가 업데이트됩니다(`voc/2007:1.0.0` -&gt;`voc/2007:2.0.0`).

이 가이드의 나머지 부분은 마지막 정의(TFDS 저장소의 데이터세트 코드 버전)에만 초점을 맞춥니다.

## 지원되는 버전

일반적으로:

- 마지막 최신 버전만 생성할 수 있습니다.
- 이전에 생성된 모든 데이터세트를 읽을 수 있습니다(참고: TFDS 4+로 생성된 데이터세트가 필요함).

```python
builder = tfds.builder('my_dataset')
builder.info.version  # Current version is: '2.0.0'

# download and load the last available version (2.0.0)
ds = tfds.load('my_dataset')

# Explicitly load a previous version (only works if
# `~/tensorflow_datasets/my_dataset/1.0.0/` already exists)
ds = tfds.load('my_dataset:1.0.0')
```

## 의미 체계

TFDS에 정의된 모든 `DatasetBuilder`는 다음과 같은 버전으로 제공됩니다.

```python
class MNIST(tfds.core.GeneratorBasedBuilder):
  VERSION = tfds.core.Version('2.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release',
      '2.0.0': 'Update dead download url',
  }
```

버전은 [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html):`MAJOR.MINOR.PATCH`를 따릅니다 . 버전의 목적은 재현성을 보장할 수 있도록 하는 것입니다. 일정 버전에서 지정된 데이터세트를 로드하면 같은 데이터가 생성되어야 합니다. 구체적으로 다음과 같습니다.

- `PATCH` 버전이 증가하면 디스크에서 데이터가 다르게 직렬화되거나 메타 데이터가 변경되었을 수 있지만, 클라이언트가 읽은 데이터는 같습니다. 주어진 슬라이스에 대해 슬라이싱 API는 같은 레코드 세트를 반환합니다.
- `MINOR` 버전이 증가하면 클라이언트가 읽은 기존 데이터는 같지만, 추가 데이터(각 레코드의 특성)가 있습니다. 주어진 슬라이스에 대해 슬라이싱 API는 같은 레코드 세트를 반환합니다.
- `MAJOR` 버전이 증가하면 기존 데이터가 변경되었거나 슬라이싱 API가 주어진 슬라이스에 대해 반드시 같은 레코드 세트를 반환하지는 않습니다.

TFDS 라이브러리에서 코드를 변경하고 해당 코드 변경이 클라이언트가 데이터세트를 직렬화 및/또는 읽는 방식에 영향을 주는 경우, 해당 빌더 버전이 위의 가이드라인에 따라 증가합니다.

위의 의미 체계는 최선의 노력이며, 버전이 증가하지 않은 동안 데이터세트에 영향을 미치는 눈에 띄지 않는 버그가 있을 수 있습니다. 이들 버그는 결국 수정되지만, 버전 관리에 크게 의존하는 경우, (`HEAD`와는 달리) 릴리스 버전의 TFDS를 사용하는 것이 좋습니다.

또한, 일부 데이터세트에는 TFDS 버전과 독립적인 다른 버전 관리 체계가 있습니다. 예를 들어, Open Images 데이터세트에는 여러 버전이 있으며, TFDS에서 해당 빌더는 `open_images_v4` , `open_images_v5`,...입니다.

## 특정 버전 로드하기

데이터세트 또는 `DatasetBuilder`를 로드할 때 사용할 버전을 지정할 수 있습니다. 예를 들면, 다음과 같습니다.

```python
tfds.load('imagenet2012:2.0.1')
tfds.builder('imagenet2012:2.0.1')

tfds.load('imagenet2012:2.0.0')  # Error: unsupported version.

# Resolves to 3.0.0 for now, but would resolve to 3.1.1 if when added.
tfds.load('imagenet2012:3.*.*')
```

출판물에 TFDS를 사용하는 경우, 다음을 권장합니다.

- **버전의 `MAJOR` 구성 요소만 수정합니다.**
- **결과에 사용된 데이터세트의 버전을 알려줍니다.**

이렇게 하면 미래의 자신과 독자 및 검토자가 결과를 쉽게 재현할 수 있습니다.

## BUILDER_CONFIGS 및 버전

일부 데이터세트는 여러 개의 `BUILDER_CONFIGS`를 정의합니다. 이 경우, `version` 및 `supported_versions`는 구성 객체 자체에 정의됩니다. 그 외에는 의미 체계와 사용법이 동일합니다. 예를 들면, 다음과 같습니다.

```python
class OpenImagesV4(tfds.core.GeneratorBasedBuilder):

  BUILDER_CONFIGS = [
      OpenImagesV4Config(
          name='original',
          version=tfds.core.Version('0.2.0'),
          supported_versions=[
            tfds.core.Version('1.0.0', "Major change in data"),
          ],
          description='Images at their original resolution and quality.'),
      ...
  ]

tfds.load('open_images_v4/original:1.*.*')
```

## 실험 버전

참고: 다음은 잘못된 방식이며 오류가 발생하기 쉬우므로 권장하지 않습니다.

두 개의 버전을 동시에 생성할 수 있습니다. 하나의 기본 버전과 하나의 실험 버전이며 예를 들어 다음과 같습니다.

```python
class MNIST(tfds.core.GeneratorBasedBuilder):
  VERSION = tfds.core.Version("1.0.0")  # Default version
  SUPPORTED_VERSIONS = [
      tfds.core.Version("2.0.0"),  # Experimental version
  ]


# Download and load default version 1.0.0
builder = tfds.builder('mnist')

#  Download and load experimental version 2.0.0
builder = tfds.builder('mnist', version='experimental_latest')
```

코드에서 두 가지 버전을 지원하도록 해야 합니다.

```python
class MNIST(tfds.core.GeneratorBasedBuilder):

  ...

  def _generate_examples(self, path):
    if self.info.version >= '2.0.0':
      ...
    else:
      ...
```
