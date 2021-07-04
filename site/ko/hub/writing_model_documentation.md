<!--* freshness: { owner: 'wgierke' reviewed: '2021-02-25' review_interval: '3 months' } *-->

# 모델 설명서 작성하기

tfhub.dev에 모델을 제공하려면 마크다운의 설명서를 제공해야 합니다. tfhub.dev에 모델을 추가하는 과정에 대한 전체 개요는 [모델 기여](contribute_a_model.md) 가이드를 참조하세요.

## 마크다운 설명서의 유형

tfhub.dev에서 사용되는 마크다운 설명서에는 3가지 유형이 있습니다.

- 게시자 마크다운 - 게시자에 대한 정보를 포함합니다([게시자되기](publish.md) 가이드에서 자세히 알아보기).
- 모델 마크다운 - 특정 모델에 대한 정보를 포함합니다.
- 컬렉션 마크다운 - 게시자가 정의한 모델 컬렉션에 대한 정보를 포함합니다([컬렉션 만들기](creating_a_collection.md) 가이드에서 자세히 알아보기).

## 콘텐츠 구성

[TensorFlow 허브 GitHub](https://github.com/tensorflow/hub) 리포지토리에 기여할 때 다음 콘텐츠 구성이 권장됩니다.

- 각 게시자 디렉터리는 `assets` 디렉토리에 있습니다.
- 각 게시자 디렉터리에는 선택적 `models` 및 `collections` 디렉터리가 있습니다.
- 각 모델에는 `assets/publisher_name/models` 아래에 고유한 디렉터리가 있어야 합니다.
- 각 컬렉션에는 `assets/publisher_name/collections` 아래에 자체 디렉터리가 있어야 합니다.

게시자 및 컬렉션 마크다운은 버전이 지정되지 않는 반면, 모델은 여러 버전을 가질 수 있습니다. 각 모델 버전에는 설명된 버전(예: 1.md, 2.md)에 따라 이름이 지정된 별도의 마크다운 파일이 필요합니다.

특정 모델의 모든 모델 버전은 모델 디렉터리에 있어야 합니다.

다음은 마크다운 콘텐츠가 구성되는 방식을 보여주는 그림입니다.

```
assets
├── publisher_name_a
│   ├── publisher_name_a.md  -> Documentation of the publisher.
│   └── models
│       └── model          -> Model name with slashes encoded as sub-path.
│           ├── 1.md       -> Documentation of the model version 1.
│           └── 2.md       -> Documentation of the model version 2.
├── publisher_name_b
│   ├── publisher_name_b.md  -> Documentation of the publisher.
│   ├── models
│   │   └── ...
│   └── collections
│       └── collection     -> Documentation for the collection feature.
│           └── 1.md
├── publisher_name_c
│   └── ...
└── ...
```

## 모델 페이지별 마크다운 형식

모델 설명서는 일부 애드온 구문이 있는 마크다운 파일입니다. 최소 예제 또는 [보다 현실적인 예제 마크다운 파일](https://github.com/tensorflow/tfhub.dev/blob/master/examples/docs/tf2_model_example.md)은 아래 예제를 참조하세요.

### 예시 설명서

고품질 모델 설명서에는 코드 조각, 모델의 훈련 방식 및 의도된 사용 방법이 포함되어 있습니다. 또한 사용자가 tfhub.dev에서 모델을 보다 빠르게 찾을 수 있도록 [아래 설명된](#model-markdown-specific-metadata-properties) 모델별 메타데이터 속성을 사용해야 합니다.

```markdown
# Module google/text-embedding-model/1

Simple one sentence description.

<!-- asset-path: https://path/to/text-embedding-model/model.tar.gz -->
<!-- module-type: text-embedding -->
<!-- fine-tunable: true -->
<!-- format: saved_model_2 -->

## Overview

Here we give more information about the model including how it was trained,
expected use cases, and code snippets demonstrating how to use the model:

``
Code snippet demonstrating use (e.g. for a TF model using the tensorflow_hub library)

import tensorflow_hub as hub

model = hub.KerasLayer(<model name>)
inputs = ...
output = model(inputs)
``
```

### 배포를 모델링하고 배포를 그룹화

tfhub.dev를 사용하면 TensorFlow 모델의 TF.js, TFLite 및 Coral 배포를 게시할 수 있습니다.

마크다운 파일의 첫 번째 줄에서 배포 형식의 유형을 지정해야 합니다.

- TF.js 배포용: `# Tfjs publisher/model/version`
- Lite 배포용: `# Lite publisher/model/version`
- Coral 배포용: `# Coral publisher/model/version`

다양한 배포 형식이 tfhub.dev에서 같은 모델 페이지에 표시되는 것이 좋습니다. 주어진 TF.js, TFLite 또는 Coral 배포를 TensorFlow 모델에 연결하려면 parent-model 태그를 지정하세요.

```markdown
<!-- parent-model: publisher/model/version -->
```

때로는 TensorFlow SavedModel 없이 하나 이상의 배포를 게시해야 할 수 있습니다. 이 경우 Placeholder 모델을 만들고 `parent-model` 태그에 핸들을 지정해야 합니다. 자리 표시자 마크다운은 첫 번째 줄이 `# Placeholder publisher/model/version`이고 `asset-path` 속성이 필요하지 않다는 점을 제외하면 TensorFlow 모델 마크다운과 동일합니다.

### 모델 마크다운별 메타데이터 속성

마크다운 파일에는 메타데이터 속성이 포함될 수 있습니다. 이는 마크다운 파일의 설명 뒤에 마크다운 주석으로 표시됩니다. 예를 들면 다음과 같습니다.

```
# Module google/universal-sentence-encoder/1
Encoder of greater-than-word length text trained on a variety of data.

<!-- module-type: text-embedding -->
...
```

다음과 같은 메타데이터 속성이 있습니다.

- `format`: TensorFlow 모델의 경우, 모델의 TensorFlow 허브 형식입니다. 모델을 레거시 [TF1 허브 형식](exporting_hub_format.md)을 통해 내보낸 경우 유효한 값은 `hub`이고, 모델을 [TF2 Saved Model](exporting_tf2_saved_model.md)을 통해 내보낸 경우`saved_model_2`입니다.
- `asset-path`: Google Cloud Storage 버킷과 같이 업로드할 실제 모델 자산에 대한 세계에서 읽을 수 있는 원격 경로입니다. URL은 robots.txt 파일에서 가져올 수 있어야 합니다(이러한 이유로 "https://github.com/.*/releases/download/.*"는 https://github.com/robots.txt에서 금지되므로 지원되지 않음).
- `parent-model`: TF.js/TFLite/Coral 모델의 경우, 함께 제공되는 SavedModel/Placeholder의 핸들입니다.
- `module-type`: 문제 도메인입니다(예: "text-embedding" 또는 "image-classification").
- `dataset`: 모델이 학습된 데이터세트입니다(예: "ImageNet-21k" 또는 "Wikipedia").
- `network-architecture`: 모델의 기반이 되는 네트워크 아키텍처입니다(예: "BERT" 또는 "Mobilenet V3").
- `language`: 텍스트 모델이 훈련된 언어의 언어 코드입니다(예: "en" 또는 "fr").
- `fine-tunable`: 사용자가 모델을 미세 조정할 수 있는지 여부를 나타내는 부울입니다.
- `license`: 모델에 적용되는 라이선스입니다. 게시된 모델의 기본 가정 라이선스는 [Apache 2.0 라이선스](https://opensource.org/licenses/Apache-2.0)입니다. 허용되는 다른 옵션은 [OSI 승인된 라이선스](https://opensource.org/licenses)에 나열되어 있습니다. 가능한 (리터럴) 값: `Apache-2.0`, `BSD-3-Clause`, `BSD-2-Clause`, `GPL-2.0`, `GPL-3.0`, `LGPL-2.0`, `LGPL-2.1`, `LGPL-3.0`, `MIT`, `MPL-2.0`, `CDDL-1.0`, `EPL-2.0`, `custom`. 사용자 정의 라이선스에는 개별적으로 특별한 고려 사항이 필요합니다.

마크다운 설명서 유형은 다양한 필수 및 선택적 메타데이터 속성을 지원합니다.

유형 | 필수 | 옵션
--- | --- | ---
게시자 |  |
수집 | 모듈 유형 | 데이터세트, 언어,
:             :                          : 네트워크 아키텍처             : |  |
자리 표시자 | 모듈 유형 | 데이터세트, 미세 조정 가능, 언어,
:             :                          : 라이선스, 네트워크 아키텍처    : |  |
SavedModel | 자산 경로, 모듈 유형, | 데이터세트, 언어, 라이선스,
:             : 미세 조정 가능, 형식     : 네트워크 아키텍처             : |  |
Tfjs | 자산 경로, 상위 모델 |
Lite | 자산 경로, 상위 모델 |
Coral | 자산 경로, 상위 모델 | 
