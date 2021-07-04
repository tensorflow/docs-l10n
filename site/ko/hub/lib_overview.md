<!--* freshness: { owner: 'kempy' } *-->

# TensorFlow 서브 라이브러리 개요

[`tensorflow_hub`](https://github.com/tensorflow/hub) 라이브러리를 사용하면 최소한의 코드만 사용하여 TensorFlow 프로그램에서 학습된 모델을 다운로드하고 재사용할 수 있습니다. 학습된 모델을 로드하는 주된 방법은 `hub.KerasLayer` API를 이용하는 것입니다.

```python
import tensorflow_hub as hub

embed = hub.KerasLayer("https://tfhub.dev/google/nnlm-en-dim128/2")
embeddings = embed(["A long sentence.", "single-word", "http://example.com"])
print(embeddings.shape, embeddings.dtype)
```

## 다운로드를 위한 캐시 위치 설정하기

기본적으로 `tensorflow_hub`는 시스템 전체의 임시 디렉터리를 사용하여 다운로드 및 압축되지 않은 모델을 캐시합니다. 더 영구적인 다른 위치를 사용하는 옵션은 [캐싱](caching.md)을 참조하세요.

## API 안정성

파격적인 변경 사항은 없길 바라지만, 이 프로젝트는 아직 개발 중이며 안정적인 API 또는 모델 형식을 보장하지 않습니다.

## 공평

모든 머신러닝에서와 마찬가지로, [공정성](http://ml-fairness.com)은 [중요한](https://research.googleblog.com/2016/10/equality-of-opportunity-in-machine.html) 고려 사항입니다. 많은 사전 훈련된 모델이 대규모 데이터세트에서 훈련됩니다. 모델을 재사용할 때 모델이 학습한 데이터(및 기존 바이어스가 있는지 여부)와 데이터세트가 모델 사용에 어떤 영향을 미칠 수 있는지를 염두에 두는 것이 중요합니다.

## 보안

모델은 임의의 TensorFlow 그래프를 포함하기 때문에 프로그램으로 생각할 수 있습니다. [TensorFlow를 안전하게 사용하기](https://github.com/tensorflow/tensorflow/blob/master/SECURITY.md)에서는 신뢰할 수 없는 소스에서 모델을 참조할 때 보안에 미치는 영향을 설명합니다.

## 다음 단계

- [라이브러리 사용하기](tf2_saved_model.md)
- [재사용 가능한 SavedModels](reusable_saved_models.md)
