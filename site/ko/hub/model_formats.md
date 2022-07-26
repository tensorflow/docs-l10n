<!--* freshness: { owner: 'maringeo' reviewed: '2021-12-13' review_interval: '6 months'} *-->

# 모델 형식

[tfhub.dev](https://tfhub.dev)는 SavedModel, TF1 서브 형식, TF.js 및 TFLite와 같은 모델 형식을 호스팅합니다. 이 페이지는 각 모델 형식에 대한 개요를 제공합니다.

## TensorFlow 형식

[tfhub.dev](https://tfhub.dev)는 SavedModel 형식 및 TF1 허브 형식으로 TensorFlow 모델을 호스팅합니다. 가능하면 더 이상 사용되지 않는 TF1 허브 형식 대신 표준화된 SavedModel 형식의 모델을 사용하는 것이 좋습니다.

### SavedModel

SavedModel은 TensorFlow 모델 공유에 권장되는 형식입니다. [TensorFlow 저장된 모델](https://www.tensorflow.org/guide/saved_model) 가이드에서 SavedModel 형식에 대해 자세히 알아볼 수 있습니다.

[tfhub.dev 찾아보기 페이지](https://tfhub.dev/s?subtype=module,placeholder)의 TF2 버전 필터를 사용하거나 [이 링크](https://tfhub.dev/s?subtype=module,placeholder&tf-version=tf2)를 따라 tfhub.dev에서 SavedModel을 찾아볼 수 있습니다.

이 형식은 핵심 TensorFlow의 일부이므로 `tensorflow_hub` 라이브러리에 의존하지 않고 tfhub.dev에서 SavedModel을 사용할 수 있습니다.

TF 허브의 SavedModel에 대해 자세히 알아보세요.

- [TF2 SavedModels 사용하기](tf2_saved_model.md)
- [TF2 SavedModel 내보내기](exporting_tf2_saved_model.md)
- [TF2 SavedModels의 TF1/TF2 호환성](model_compatibility.md)

### TF1 허브 형식

TF1 허브 형식은 TF 허브 라이브러리에서 사용하는 사용자 정의 직렬화 형식입니다. TF1 허브 형식은 구문 수준(동일한 파일 이름 및 프로토콜 메시지)에서 TensorFlow 1의 SavedModel 형식과 유사하지만 모듈 재사용, 구성 및 재교육(예: 리소스 초기화 프로그램의 다른 저장소, 메타그래프에 대한 다른 태그 지정 규칙)이 가능하도록 의미적으로 다릅니다. 디스크에서 이를 구별하는 가장 쉬운 방법은 `tfhub_module.pb` 파일이 있는지 여부입니다.

[tfhub.dev 찾아보기 페이지](https://tfhub.dev/s?subtype=module,placeholder)에서 TF1 버전 필터를 사용하거나 [이 링크](https://tfhub.dev/s?subtype=module,placeholder&tf-version=tf1)에 따라 tfhub.dev에서 TF1 허브 형식의 모델을 찾아볼 수 있습니다.

TF 허브의 TF1 허브 형식 모델에 대해 자세히 알아보세요.

- [TF1 Hub 형식 모델 사용하기](tf1_hub_module.md)
- [TF1 Hub 형식으로 모델 내보내기](exporting_hub_format.md)
- [TF1 Hub 형식의 TF1/TF2 호환성](model_compatibility.md)

## TFLite 형식

TFLite 형식은 온디바이스 추론에 사용됩니다. [TFLite 설명서](https://www.tensorflow.org/lite)에서 자세한 내용을 확인할 수 있습니다.

[tfhub.dev 찾아보기 페이지](https://tfhub.dev/s?subtype=module,placeholder)의 TF Lite 모델 형식 필터를 사용하거나 [이 링크](https://tfhub.dev/lite)에 따라 tfhub.dev에서 TF Lite 모델을 찾아볼 수 있습니다.

## TFJS 형식

TF.js 형식은 브라우저 내 ML에 사용됩니다. [TF.js 설명서](https://www.tensorflow.org/js)에서 자세히 알아볼 수 있습니다.

[tfhub.dev 찾아보기 페이지](https://tfhub.dev/s?subtype=module,placeholder)의 TF.js 모델 형식 필터를 사용하거나 [이 링크](https://tfhub.dev/js)에 따라 tfhub.dev에서 TF Lite 모델을 찾아볼 수 있습니다.
