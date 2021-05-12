<!--* freshness: { owner: 'maringeo' reviewed: '2021-04-12' review_interval: '6 months' } *-->

# 컬렉션 만들기

컬렉션은 게시자가 관련 모델을 함께 묶어 사용자의 검색 경험을 개선하는 tfhub.dev의 기능입니다.

tfhub.dev의 [모든 컬렉션 목록](https://tfhub.dev/s?subtype=model-family)을 참조하세요.

리포지토리 [github.com/tensorflow/tfhub.dev](https://github.com/tensorflow/tfhub.dev)에서 컬렉션 파일의 올바른 위치는 [assets/docs](https://github.com/tensorflow/tfhub.dev/tree/master/assets/docs)/<b>&lt;게시자_이름&gt;</b>/collections/<b>&lt;컬렉션_이름&gt;</b>/<b>1</b>.md입니다.

다음은 assets/docs/<b>vtab</b>/collections/<b>benchmark</b>/<b>1</b>.md에 들어가는 최소한의 예입니다. 첫 번째 줄의 컬렉션 이름이 파일 이름보다 짧다는 점에 주목하세요.

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
