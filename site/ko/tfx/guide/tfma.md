# TensorFlow 모델 분석으로 모델 품질 향상하기

## 소개

개발 중에 모델을 조정할 때 변경 사항으로 모델이 개선되는지 확인해야 합니다. 정확성을 확인하는 것만으로는 충분하지 않을 수 있습니다. 예를 들어, 인스턴스의 95%가 긍정인 문제에 대한 분류자가 있으면 항상 긍정을 예측하여 정확성을 향상할 수 있지만, 이것이 매우 강력한 분류자인 것은 아닙니다.

## 개요

TensorFlow 모델 분석의 목표는 TFX에서 모델 평가를 위한 메커니즘을 제공하는 것입니다. TensorFlow 모델 분석을 사용하면 TFX 파이프라인에서 모델 평가를 수행하고 Jupyter 노트북에서 결과 메트릭과 플롯을 볼 수 있습니다. 특히 다음을 제공할 수 있습니다.

- 전체 훈련 및 홀드아웃 데이터세트와 익일 평가에서 계산된 [메트릭](../model_analysis/metrics)
- 시간 경과에 따른 메트릭 추적
- 다양한 특성 조각에 대한 모델 품질 성능
- 모델의 일관된 성능 유지를 위한 [모델 검증](../model_analysis/model_validations)

## 다음 단계

[TFMA 튜토리얼](../tutorials/model_analysis/tfma_basic)을 사용하세요.

지원되는 [메트릭과 플롯](https://github.com/tensorflow/model-analysis) 및 관련 노트북 [시각화](../model_analysis/metrics)에 대한 자세한 내용은 [github](../model_analysis/visualizations) 페이지를 확인하세요.

독립형 파이프라인에서 [설정](../model_analysis/install)하는 방법에 대한 정보와 예제는 [설치](../model_analysis/get_started) 및 [get_started](../model_analysis/setup) 가이드를 참조하세요. TFMA는 TFX의 [Evaluator](evaluator.md) 구성 요소 내에서도 사용되므로 이러한 리소스는 TFX를 시작하는 데에도 유용할 것입니다.
