<!--* freshness: { owner: 'maringeo' reviewed: '2020-09-14' review_interval: '3 months' } *-->

# 컬렉션 만들기

컬렉션은 게시자가 관련 모델을 함께 묶어 사용자의 검색 경험을 개선하는 tfhub.dev의 기능입니다.

tfhub.dev의 [모든 컬렉션 목록](https://tfhub.dev/s?subtype=model-family)을 참조하세요.

TensorFlow Hub 리포지토리에서 컬렉션 파일의 올바른 위치는 다음과 같습니다. [hub/tfhub_dev/assets/](https://github.com/tensorflow/hub/tree/master/tfhub_dev/assets)/<publisher_name>/<collection_name>/<collection_name.md>

최소 게시자 설명서의 예를 참조하세요.

```markdown
# Collection vtab/benchmark/1
Collection of visual representations that have been evaluated on the VTAB
benchmark.



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
