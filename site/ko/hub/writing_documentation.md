<!--* freshness: { owner: 'wgierke' reviewed: '2022-07-27' review_interval: '6 months' } *-->

# 문서 작성

tfhub.dev에 모델을 제공하려면 마크다운 형식의 설명서를 제공해야 합니다. tfhub.dev에 모델을 추가하는 과정에 대한 전체 개요는 [모델 기여](contribute_a_model.md) 가이드를 참조하세요.

## 마크다운 설명서의 유형

tfhub.dev에서 사용되는 마크다운 설명서에는 3가지 유형이 있습니다.

- 게시자 마크다운 - 게시자에 대한 정보([마크다운 구문 참조](#publisher))
- 모델 마크다운 - 특정 모델 및 사용 방법에 대한 정보([마크다운 구문 참조](#model))
- 컬렉션 마크다운 - 게시자가 정의한 모델 컬렉션에 대한 정보 포함([마크다운 구문 참조](#collection))

## 콘텐츠 구성

[TensorFlow Hub GitHub](https://github.com/tensorflow/tfhub.dev) 리포지토리에 기여할 때 다음 콘텐츠 구성이 필요합니다.

- 각 게시자 디렉터리는 `assets/docs` 디렉토리에 있습니다.
- 각 게시자 디렉터리에는 선택적 `models` 및 `collections` 디렉터리가 있습니다.
- 각 모델에는 `assets/docs/<publisher_name>/models` 아래에 고유한 디렉터리가 있어야 합니다.
- 각 컬렉션에는 `assets/docs/<publisher_name>/collections` 아래에 고유한 디렉터리가 있어야 합니다.

게시자 마크다운은 버전이 지정되지 않는 반면, 모델은 여러 버전을 가질 수 있습니다. 각 모델 버전에는 설명된 버전(예: 1.md, 2.md)에 따라 이름이 지정된 별도의 마크다운 파일이 필요합니다. 컬렉션은 버전이 지정되지만 단일 버전(1.md)만 지원됩니다.

특정 모델의 모든 모델 버전은 모델 디렉터리에 있어야 합니다.

다음은 마크다운 콘텐츠가 구성되는 방식을 보여주는 그림입니다.

```
assets/docs
├── <publisher_name_a>
│   ├── <publisher_name_a>.md  -> Documentation of the publisher.
│   └── models
│       └── <model_name>       -> Model name with slashes encoded as sub-path.
│           ├── 1.md           -> Documentation of the model version 1.
│           └── 2.md           -> Documentation of the model version 2.
├── <publisher_name_b>
│   ├── <publisher_name_b>.md  -> Documentation of the publisher.
│   ├── models
│   │   └── ...
│   └── collections
│       └── <collection_name>
│           └── 1.md           -> Documentation for the collection.
├── <publisher_name_c>
│   └── ...
└── ...
```

## 게시자 마크다운 형식 {:#publisher}

게시자 설명서는 모델과 동일한 종류의 마크다운 파일로 선언되지만 구문상 약간의 차이가 있습니다.

TensorFlow Hub 리포지토리에서 게시자 파일의 올바른 위치는  [tfhub.dev/assets/docs](https://github.com/tensorflow/tfhub.dev/tree/master/assets/docs)/&lt;publisher_id&gt;/&lt;publisher_id.md&gt;입니다.

"vtab" 게시자의 경우 최소 게시자 설명서의 예를 참조하세요.

```markdown
# Publisher vtab
Visual Task Adaptation Benchmark

[![Icon URL]](https://storage.googleapis.com/vtab/vtab_logo_120.png)

## VTAB
The Visual Task Adaptation Benchmark (VTAB) is a diverse, realistic and
challenging benchmark to evaluate image representations.
```

위의 예는 게시자 ID, 게시자 이름, 사용할 아이콘 경로 및 더 긴 자유 형식 마크다운 설명서를 지정합니다. 게시자 ID에는 소문자, 숫자 및 하이픈만 포함되어야 합니다.

### 게시자 이름 가이드라인

게시자 이름은 GitHub 사용자 이름 또는 관리하는 GitHub 조직의 이름이 될 수 있습니다.

## 모델 페이지 마크다운 형식 {:#model}

모델 설명서는 일부 애드온 구문이 있는 마크다운 파일입니다. 최소 예제 또는 [보다 현실적인 예제 마크다운 파일](https://github.com/tensorflow/tfhub.dev/blob/master/examples/docs/tf2_model_example.md)은 아래 예제를 참조하세요.

### 예시 설명서

고품질 모델 설명서에는 코드 조각, 모델의 훈련 방식 및 의도된 사용 방법이 포함되어 있습니다. 또한 사용자가 tfhub.dev에서 모델을 보다 빠르게 찾을 수 있도록 [아래 설명된](#metadata) 모델별 메타데이터 속성을 사용해야 합니다.

```markdown
# Module google/text-embedding-model/1

Simple one sentence description.

<!-- asset-path: https://path/to/text-embedding-model/model.tar.gz -->
<!-- task: text-embedding -->
<!-- fine-tunable: true -->
<!-- format: saved_model_2 -->

## Overview

Here we give more information about the model including how it was trained,
expected use cases, and code snippets demonstrating how to use the model:

```
Code snippet demonstrating use (e.g. for a TF model using the tensorflow_hub library)

import tensorflow_hub as hub

model = hub.KerasLayer(<model name>)
inputs = ...
output = model(inputs)
```
```

### 배포를 모델링하고 배포를 그룹화

tfhub.dev를 사용하면 TensorFlow SavedModel의 TF.js, TFLite 및 Coral 배포를 게시할 수 있습니다.

마크다운 파일의 첫 번째 줄에서 형식의 유형을 지정해야 합니다.

- SavedModels용: `# Module publisher/model/version`
- TF.js 배포용: `# Tfjs publisher/model/version`
- Lite 배포용: `# Lite publisher/model/version`
- Coral 배포용: `# Coral publisher/model/version`

동일한 개념 모델의 이러한 다양한 형식이 tfhub.dev에서 같은 모델 페이지에 표시되는 것이 좋습니다. 주어진 TF.js, TFLite 또는 Coral 배포를 TensorFlow SavedModel 모델에 연결하려면 parent-model 태그를 지정하세요.

```markdown
<!-- parent-model: publisher/model/version -->
```

때로는 TensorFlow SavedModel 없이 하나 이상의 배포를 게시해야 할 수 있습니다. 이 경우 Placeholder 모델을 만들고 `parent-model` 태그에 핸들을 지정해야 합니다. 자리 표시자 마크다운은 첫 번째 줄이 `# Placeholder publisher/model/version`이고 `asset-path` 속성이 필요하지 않다는 점을 제외하면 TensorFlow 모델 마크다운과 동일합니다.

### 모델 마크다운 특정 메타데이터 속성 {:#metadata}

마크다운 파일에는 메타데이터 속성이 포함될 수 있습니다. 이는 사용자가 모델을 찾는 데 도움이 되는 필터 및 태그를 제공하는 데 사용됩니다. 메타데이터 속성은 마크다운 파일에 대한 짧은 설명 뒤에 마크다운 주석으로 포함됩니다. 예:

```markdown
# Module google/universal-sentence-encoder/1
Encoder of greater-than-word length text trained on a variety of data.

<!-- task: text-embedding -->
...
```

다음과 같은 메타데이터 속성이 지원됩니다.

- `format`: TensorFlow 모델의 경우, 모델의 TensorFlow Hub 형식입니다. 모델을 레거시 [TF1 허브 형식](exporting_hub_format.md)을 통해 내보낸 경우 유효한 값은 `hub`이고, 모델을 [TF2 Saved Model](exporting_tf2_saved_model.md)을 통해 내보낸 경우`saved_model_2`입니다.
- `asset-path` : Google Cloud Storage 버킷과 같이 업로드할 실제 모델 자산에 대한 전 세계에서 읽을 수 있는 원격 경로입니다. URL은 robots.txt 파일에서 가져올 수 있어야 합니다(이러한 이유로 "https://github.com/. */releases/download/.*"는 https://github.com/robots.txt에서 금지될 때 지원되지 않습니다.). 예상되는 파일 형식 및 콘텐츠에 대한 자세한 내용은 [아래](#model-specific-asset-content)를 참조하세요.
- `parent-model`: TF.js/TFLite/Coral 모델의 경우, 함께 제공되는 SavedModel/Placeholder의 핸들입니다.
- `fine-tunable`: 사용자가 모델을 미세 조정할 수 있는지 여부를 나타내는 부울입니다.
- `task`: 문제 도메인(예: "text-embedding"). 지원되는 모든 값은 [task.yaml](https://github.com/tensorflow/tfhub.dev/blob/master/tags/task.yaml)에 정의되어 있습니다.
- `dataset`: 모델이 훈련된 데이터세트(예: "wikipedia"). 지원되는 모든 값은 [dataset.yaml](https://github.com/tensorflow/tfhub.dev/blob/master/tags/dataset.yaml)에 정의되어 있습니다.
- `network-architecture`: 모델이 기반으로 하는 네트워크 아키텍처(예: "mobilenet-v3"). 지원되는 모든 값은 [network_architecture.yaml](https://github.com/tensorflow/tfhub.dev/blob/master/tags/network_architecture.yaml)에 정의되어 있습니다.
- `language`: 텍스트 모델이 훈련된 언어의 언어 코드(예: "en"). 지원되는 모든 값은 [language.yaml](https://github.com/tensorflow/tfhub.dev/blob/master/tags/language.yaml)에 정의되어 있습니다.
- `license`: 모델에 적용되는 라이선스(예: "mit"). 게시된 모델의 기본 가정 라이선스는 [Apache 2.0 라이선스](https://opensource.org/licenses/Apache-2.0) 입니다. 지원되는 모든 값은 [license.yaml](https://github.com/tensorflow/tfhub.dev/blob/master/tags/license.yaml)에 정의되어 있습니다. `custom` 라이선스는 경우에 따라 특별히 고려할 필요가 있습니다.
- `colab`: 모델을 사용하거나 훈련하는 방법을 보여주는 노트북의 HTTPS URL([예](https://colab.sandbox.google.com/github/tensorflow/hub/blob/master/examples/colab/bigbigan_with_tf_hub.ipynb): [bigbigan-resnet50](https://tfhub.dev/deepmind/bigbigan-resnet50/1)). `colab.research.google.com`으로 연결되어야 합니다. GitHub에서 호스팅되는 Jupyter 노트북은 `https://colab.research.google.com/github/ORGANIZATION/PROJECT/ blob/master/.../my_notebook.ipynb`를 통해 액세스할 수 있습니다.
- `demo`: TF.js 모델을 사용할 수 있는 방법을 보여주는 웹사이트의 HTTPS URL([예](https://teachablemachine.withgoogle.com/train/pose): [posenet](https://tfhub.dev/tensorflow/tfjs-model/posenet/mobilenet/float/075/1/default/1)).
- `interactive-visualizer`: 모델 페이지에 포함되어야 하는 비주얼라이저의 이름(예: "vision"). 비주얼라이저를 표시하면 사용자가 모델의 예측을 대화식으로 탐색할 수 있습니다. 지원되는 모든 값은 [interactive_visualizer.yaml](https://github.com/tensorflow/tfhub.dev/blob/master/tags/interactive_visualizer.yaml)에 정의되어 있습니다.

마크다운 설명서 유형은 다양한 필수 및 선택적 메타데이터 속성을 지원합니다.

유형 | 필수 | 옵션
--- | --- | ---
게시자 |  |
컬렉션 | 작업 | 데이터세트, 언어,
:             :                          : 네트워크 아키텍처                : |  |
자리 표시자 | 작업 | 데이터세트, 미세 조정 가능
:             :                          : 대화형 비주얼라이저, 언어,   : |  |
: : : 라이선스, 네트워크 아키텍처 : |  |
SavedModel | 자산 경로, 작업, | colab, 데이터세트,
: : 미세 조정 가능, 형식 : 대화형 비주얼라이저, 언어, : |  |
: : : 라이선스, 네트워크 아키텍처 : |  |
Tfjs | 자산 경로, 상위 모델 | colab, 데모, 대화형 비주얼라이저
Lite | 자산 경로, 상위 모델 | colab, 대화형 비주얼라이저
Coral | 자산 경로, 상위 모델 | colab, 대화형 비주얼라이저

### 모델별 자산 콘텐츠

모델 유형에 따라 다음과 같은 파일 유형과 콘텐츠가 예상됩니다.

- SavedModel: 다음과 같은 콘텐츠를 포함하는 tar.gz 아카이브:

```
saved_model.tar.gz
├── assets/            # Optional.
├── assets.extra/      # Optional.
├── variables/
│     ├── variables.data-?????-of-?????
│     └──  variables.index
├── saved_model.pb
├── keras_metadata.pb  # Optional, only required for Keras models.
└── tfhub_module.pb    # Optional, only required for TF1 models.
```

- TF.js: 다음과 같은 콘텐츠를 포함하는 tar.gz 아카이브:

```
tf_js_model.tar.gz
├── group*
├── *.json
├── *.txt
└── *.pb
```

- TFLite: .tflite 파일
- Coral: .tflite 파일

tar.gz 아카이브의 경우: 모델 파일이 `my_model` 디렉터리(예: SavedModels의 경우 `my_model/saved_model.pb` 또는 TF.js 모델의 경우 `my_model/model.json`)에 있는 것으로 가정하면 `cd my_model && tar -czvf ../model.tar.gz *`를 통해 [tar](https://www.gnu.org/software/tar/manual/tar.html) 도구로 유효한 tar.gz 아카이브를 생성할 수 있습니다.

일반적으로 모든 파일과 디렉터리(압축 또는 비압축)는 단어 문자로 시작해야 합니다. 예를 들어 점은 파일 이름/디렉토리의 유효한 접두사가 아닙니다.

## 컬렉션 페이지 마크다운 형식 {:#collection}

컬렉션은 게시자가 관련 모델을 함께 묶어 사용자의 검색 경험을 개선하는 tfhub.dev의 기능입니다.

tfhub.dev의 [모든 컬렉션 목록](https://tfhub.dev/s?subtype=model-family)을 참조하세요.

[github.com/tensorflow/tfhub.dev](https://github.com/tensorflow/tfhub.dev) 리포지토리에서 컬렉션 파일의 올바른 위치는 [assets/docs](https://github.com/tensorflow/tfhub.dev/tree/master/assets/docs)/<b>publisher_name&gt;</b>/collections/<b>&lt;collection_name&gt;</b>/<b>1</b>.md입니다.

다음은 assets/docs/<b>vtab</b>/collections/<b>benchmark</b>/<b>1</b>.md.에 들어가는 최소한의 예입니다. 첫 번째 줄의 컬렉션 이름은 파일 경로에 포함된 `collections/` 부분을 포함하지 않습니다.

```markdown
# Collection vtab/benchmark/1
Collection of visual representations that have been evaluated on the VTAB
benchmark.

<!-- task: image-feature-vector -->

## Overview
This is the list of visual representations in TensorFlow Hub that have been
evaluated on VTAB. Results can be seen in
[google-research.github.io/task_adaptation/](https://google-research.github.io/task_adaptation/)

#### Models
|                   |
|-------------------|
| [vtab/sup-100/1](https://tfhub.dev/vtab/sup-100/1)   |
| [vtab/rotation/1](https://tfhub.dev/vtab/rotation/1) |
|------------------------------------------------------|
```

이 예에서는 컬렉션 이름, 짧은 한 문장 설명, 문제 도메인 메타데이터 및 자유 형식의 마크다운 설명서를 지정합니다.
