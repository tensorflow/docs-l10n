# TFX 클라우드 솔루션

TFX를 적용하여 요구 사항을 충족하는 솔루션을 구축하는 방법에 대한 인사이트를 찾고 계신가요? 자세한 내용을 다룬 이 문서와 가이드가 도움이 될 것입니다!

참고: 이 문서에서는 TFX가 핵심이지만 유일한 구성 요소는 아닌 완전한 솔루션에 대해 설명합니다. 실제 배포에서도 거의 항상 그렇습니다. 따라서 이러한 솔루션을 직접 구현할 경우 TFX 이상의 것이 필요합니다. 주요 목표는 여러분의 요구 사항과 유사한 요구 사항을 충족할 수 있는 솔루션을 다른 사람들이 구현한 방법과 그에 따른 인사이트를 제공하는 것이며, TFX의 승인된 애플리케이션 목록이나 요리책 역할은 하지 않습니다.

## 실시간에 가까운 아이템 매칭을 위한 머신러닝 시스템 아키텍처

이 문서를 사용하여 아이템 임베딩을 훈련하고 제공하는 머신러닝(ML) 솔루션의 아키텍처에 대해 알아볼 수 있습니다. 임베딩은 고객이 유사하다고 생각하는 아이템을 이해하기 위한 도움을 제공하며 이를 통해 애플리케이션에서 실시간으로 "유사한 아이템" 제안 기능을 제공할 수 있습니다. 이 솔루션은 데이터 세트에서 유사한 노래를 식별한 다음 이 정보를 사용하여 노래를 추천하는 방법을 보여줍니다. <a href="https://cloud.google.com/solutions/real-time-item-matching" class="external" target="_blank">자세히 알아보기</a>

## 머신러닝을 위한 데이터 전처리: 옵션 및 권장 사항

두 부분으로 구성된 이 문서에서는 머신러닝(ML)을 위한 데이터 엔지니어링과 기능 엔지니어링을 살펴봅니다. 첫 번째 파트에서는 Google Cloud의 머신러닝 파이프라인에서 데이터를 전처리하는 권장 사항에 대해 설명합니다. 이 문서에서는 TensorFlow 및 오픈 소스 TensorFlow Transform(tf.Transform) 라이브러리를 사용하여 데이터를 준비하고, 모델을 훈련하고, 예측용 모델을 제공하는 데 중점을 둡니다. 이 부분에서는 머신러닝을 위해 데이터를 전처리할 때 경험할 수 있는 어려운 점을 설명하고 Google Cloud에서 데이터 변환을 효과적으로 수행하기 위한 옵션과 시나리오를 보여줍니다. <a href="https://cloud.google.com/solutions/machine-learning/data-preprocessing-for-ml-with-tf-transform-pt1" class="external" target="_blank">파트 1</a> <a href="https://cloud.google.com/solutions/machine-learning/data-preprocessing-for-ml-with-tf-transform-pt2" class="external" target="_blank">파트 2</a>

## TFX, Kubeflow Pipelines, Cloud Build를 사용하는 MLOps용 아키텍처

이 문서는 TensorFlow Extended(TFX) 라이브러리를 사용하는 머신러닝(ML) 시스템의 전체 아키텍처를 설명합니다. 또한 Cloud Build와 Kubeflow Pipelines를 사용하여 ML 시스템에 대한 지속적인 통합(CI), 지속적인 제공(CD), 지속적인 학습(CT)을 설정하는 방법을 논의합니다. <a href="https://cloud.google.com/solutions/machine-learning/architecture-for-mlops-using-tfx-kubeflow-pipelines-and-cloud-build" class="external" target="_blank">자세히 알아보기</a>

## MLOps: 머신러닝의 지속적 제공 및 자동화 파이프라인

이 문서에서는 머신러닝(ML) 시스템용 CI(지속적인 통합), CD(지속적인 제공) 및 CT(지속적인 학습)를 구현하고 자동화하는 기술에 대해 설명합니다. 데이터 과학과 ML은 복잡한 현실 세계의 문제를 해결하고 산업을 변화시키며 모든 영역에서 가치를 제공하기 위한 핵심 기능이 되고 있습니다. <a href="https://cloud.google.com/solutions/machine-learning/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning" class="external" target="_blank">자세히 알아보기</a>

## Google Cloud에서 MLOps 환경 설정하기

이 참조 가이드에서는 Google Cloud의 머신러닝 작업(MLOps) 환경 아키텍처를 간략하게 설명합니다. 본 가이드는 여기에 설명된 환경을 프로비저닝하고 구성하는 과정을 안내하는 GitHub의 **핸즈온 랩을 함께 제공**합니다. 거의 모든 산업에서 빠른 속도로 머신러닝(ML)을 채택하고 있습니다. ML을 통해 가치를 얻으려면 ML 시스템을 효과적으로 배포하고 운영하는 방법을 만드는 과제를 해결해야 합니다. 이 가이드는 머신러닝(ML) 및 DevOps 엔지니어를 대상으로 합니다. <a href="https://cloud.google.com/solutions/machine-learning/setting-up-an-mlops-environment" class="external" target="_blank">자세히 알아보기</a>

## MLOps foundation의 핵심 요구 사항

AI 기반 조직은 데이터와 머신러닝을 사용하여 가장 어려운 문제를 해결하고 보상받고 있습니다.

McKinsey Global Institute는 *"2025년까지 가치를 창출하는 워크플로에 AI를 완전히 도입하는 기업은 +120%의 현금 흐름 성장으로 2030년 세계 경제를 지배할 것이다"*라고 했습니다.

하지만 당장 머신러닝을 도입하는 것은 쉽지 않습니다. 머신러닝(ML) 시스템은 잘 관리하지 않으면 기술적 부채를 생성하는 특별한 능력을 갖고 있습니다. <a href="https://cloud.google.com/blog/products/ai-machine-learning/key-requirements-for-an-mlops-foundation" class="external" target="_blank">자세히 알아보기</a>

## Scikit-Learn으로 클라우드에서 모델 카드를 생성하고 배포하는 방법

머신러닝 모델은 현재 다수의 어려운 작업을 수행하는 데 사용되고 있습니다. 큰 잠재력을 지닌 ML 모델은 사용법, 구성 및 제한 사항에 대한 질문도 제기합니다. 이러한 질문에 대한 답변을 문서화하면 명확성과 공유된 이해를 얻는 데 도움이 됩니다. 이러한 목표를 달성하는 데 도움이 되도록 Google은 모델 카드를 도입했습니다. <a href="https://cloud.google.com/blog/products/ai-machine-learning/create-a-model-card-with-scikit-learn" class="external" target="_blank">자세히 알아보기</a>

## TensorFlow Data Validation을 사용한 머신러닝 대규모 데이터 분석하기 및 유효성 검사하기

이 문서에서는 실험하는 동안 데이터 탐색 및 묘사 분석을 위해 TFDV(TensorFlow Data Validation) 라이브러리를 사용하는 방법을 논의합니다. 데이터 과학자 및 머신러닝(ML) 엔지니어는 프로덕션 ML 시스템에서 TFDV를 사용하여 지속적인 학습(CT) 파이프라인에 사용되는 데이터의 유효성을 검사하고 예측 제공을 위해 수신한 데이터의 불균형과 이상값을 감지할 수 있습니다. 여기에는 **핸즈온 랩**이 포함됩니다. <a href="https://cloud.google.com/solutions/machine-learning/analyzing-and-validating-data-at-scale-for-ml-using-tfx" class="external" target="_blank">자세히 알아보기</a>
