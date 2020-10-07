# 데이터세트의 버전 관리

- [의미 체계](#semantic)
- [Supported versions](#supported-versions)
- [특정 버전 로드하기](#loading-a-specific-version)
- [Experiments](#experiments)
- [BUILDER_CONFIGS and versions](#builder-configs-and-versions)

## 의미 체계

TFDS에 정의된 모든 `DatasetBuilder`는 다음과 같은 버전으로 제공됩니다.

```py
class MNIST(tfds.core.GeneratorBasedBuilder):
  VERSION = tfds.core.Version("1.0.0")
```

버전은 [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html):`MAJOR.MINOR.PATCH`를 따릅니다 . 버전의 목적은 재현성을 보장할 수 있도록 하는 것입니다. 일정 버전에서 지정된 데이터세트를 로드하면 같은 데이터가 생성되어야 합니다. 구체적으로 다음과 같습니다.

- `PATCH` 버전이 증가하면 디스크에서 데이터가 다르게 직렬화되거나 메타 데이터가 변경되었을 수 있지만, 클라이언트가 읽은 데이터는 같습니다. 주어진 슬라이스에 대해 슬라이싱 API는 같은 레코드 세트를 반환합니다.
- `MINOR` 버전이 증가하면 클라이언트가 읽은 기존 데이터는 같지만, 추가 데이터(각 레코드의 특성)가 있습니다. 주어진 슬라이스에 대해 슬라이싱 API는 같은 레코드 세트를 반환합니다.
- `MAJOR` 버전이 증가하면 기존 데이터가 변경되었거나 슬라이싱 API가 주어진 슬라이스에 대해 반드시 같은 레코드 세트를 반환하지는 않습니다.

TFDS 라이브러리에서 코드를 변경하고 해당 코드 변경이 클라이언트가 데이터세트를 직렬화 및/또는 읽는 방식에 영향을 주는 경우, 해당 빌더 버전이 위의 가이드라인에 따라 증가합니다.

위의 의미 체계는 최선의 노력이며, 버전이 증가하지 않은 동안 데이터세트에 영향을 미치는 눈에 띄지 않는 버그가 있을 수 있습니다. 이들 버그는 결국 수정되지만, 버전 관리에 크게 의존하는 경우, (`HEAD`와는 달리) 릴리스 버전의 TFDS를 사용하는 것이 좋습니다.

또한, 일부 데이터세트에는 TFDS 버전과 독립적인 다른 버전 관리 체계가 있습니다. 예를 들어, Open Images 데이터세트에는 여러 버전이 있으며, TFDS에서 해당 빌더는 `open_images_v4` , `open_images_v5`,...입니다.

## Supported versions

`DatasetBuilder`는 정식 버전보다 높거나 낮은 여러 버전을 지원할 수 있습니다. 예를 들면, 다음과 같습니다.

```py
class Imagenet2012(tfds.core.GeneratorBasedBuilder):
  VERSION = tfds.core.Version('2.0.1', 'Encoding fix. No changes from user POV')
  SUPPORTED_VERSIONS = [
      tfds.core.Version('3.0.0', 'S3: tensorflow.org/datasets/splits'),
      tfds.core.Version('1.0.0'),
      tfds.core.Version('0.0.9', tfds_version_to_prepare="v1.0.0"),
  ]
```

이전 버전을 계속 지원할지에 대한 선택은 주로 데이터세트 및 버전의 인기에 따라 사항별로 이루어집니다. 데이터세트당 제한된 수의 버전(이상적으로는 하나)만 지원하는 것이 목표입니다. 위의 예에서 독자 관점에서`2.0.1`과 동일한 버전인 `2.0.0`이 더 이상 지원되지 않음을 알 수 있습니다.

정식 버전 번호보다 높은 지원 버전은 실험적인 것으로 간주되어 손상되어 있을 수 있습니다. 그러나 결국 정식 버전이 될 것입니다.

버전은 `tfds_version_to_prepare`를 지정할 수 있습니다. 즉, 이 데이터세트 버전은 이전 버전의 코드로 이미 준비되었지만 현재 준비할 수 없는 경우 현재 버전의 TFDS 코드에만 사용할 수 있습니다. `tfds_version_to_prepare` 값은 이 버전에서 데이터세트를 다운로드하고 준비하는 데 사용할 수 있는 TFDS의 마지막 알려진 버전을 지정합니다.

## 특정 버전 로드하기

데이터세트 또는 `DatasetBuilder`를 로드할 때 사용할 버전을 지정할 수 있습니다. 예를 들면, 다음과 같습니다.

```py
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

## Experiments

많은 데이터세트 빌더에 영향을 주는 TFDS의 변경 사항을 점진적으로 롤아웃하기 위해 실험의 개념을 도입했습니다. 실험이 처음 도입될 때는 기본적으로 사용 중지되어 있지만, 특정 데이터세트 버전에서는 사용하도록 결정할 수 있습니다. 실험은 일반적으로 처음에는 "미래" 버전(아직 정식 버전은 아님)에서 수행됩니다. 예를 들면, 다음과 같습니다.

```py
class MNIST(tfds.core.GeneratorBasedBuilder):
  VERSION = tfds.core.Version("1.0.0")
  SUPPORTED_VERSIONS = [
      tfds.core.Version("2.0.0", "EXP1: Opt-in for experiment 1",
                        experiments={tfds.core.Experiment.EXP1: True}),
  ]
```

실험이 예상대로 동작하는 것으로 확인되면, 모든 또는 대부분의 데이터세트로 확장하여 기본적으로 사용하도록 설정할 수 있으며, 위의 정의는 다음과 같이 변경됩니다.

```py
class MNIST(tfds.core.GeneratorBasedBuilder):
  VERSION = tfds.core.Version("1.0.0",
                              experiments={tfds.core.Experiment.EXP1: False})
  SUPPORTED_VERSIONS = [
      tfds.core.Version("2.0.0", "EXP1: Opt-in for experiment 1"),
  ]
```

모든 데이터세트 버전에서 실험이 사용되면(`{experiment: False}`를 지정하는 데이터세트 버전이 없음), 해당 실험은 삭제할 수 있습니다.

실험과 그에 대한 설명은 `core/utils/version.py`에 정의되어 있습니다.

## BUILDER_CONFIGS and versions

일부 데이터세트는 여러 개의 `BUILDER_CONFIGS`를 정의합니다. 이 경우, `version` 및 `supported_versions`는 구성 객체 자체에 정의됩니다. 그 외에는 의미 체계와 사용법이 동일합니다. 예를 들면, 다음과 같습니다.

```py
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
