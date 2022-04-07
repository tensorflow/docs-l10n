<!--* freshness: { owner: 'akhorlin' reviewed: '2020-09-14' } *-->

# TensorFlow 허브

TensorFlow Hub는 재사용 가능한 머신러닝을 위한 개방형 리포지토리 및 라이브러리입니다. [tfhub.dev](https://tfhub.dev) 리포지토리는 텍스트 임베딩, 이미지 분류 모델, TF.js/TFLite 모델 등 많은 사전 훈련된 모델을 제공합니다. 리포지토리는 [커뮤니티 기여자](https://tfhub.dev/s?subtype=publisher)에게 공개됩니다.

[`tensorflow_hub`](https://github.com/tensorflow/hub) 라이브러리를 다운로드하여 최소한의 코드로 TensorFlow 프로그램에서 재사용할 수 있습니다.

```python
import tensorflow_hub as hub

model = hub.KerasLayer("https://tfhub.dev/google/nnlm-en-dim128/2")
embeddings = model(["The rain in Spain.", "falls",
                    "mainly", "In the plain!"])

print(embeddings.shape)  #(4,128)
```

## 다음 단계

- [Find models on tfhub.dev](https://tfhub.dev)
- [tfhub.dev에 모델 게시](publish.md)
- TensorFlow Hub 라이브러리
    - [TensorFlow Hub 설치](installation.md)
    - [Library overview](lib_overview.md)
- [튜토리얼 따르기](tutorials)
