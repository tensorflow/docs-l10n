# 텍스트 쿡북

이 페이지에는 TensorFlow Hub를 사용하여 텍스트 도메인의 문제를 해결하는 알려진 가이드 및 도구 세트가 나열되어 있습니다. 처음부터 시작하는 대신 미리 훈련된 ML 구성 요소를 사용하여 일반적인 ML 문제를 해결하려는 모든 분들에게 좋은 출발점입니다.

## 분류

**감정**, **독성** , **기사 범주** 또는 기타 특성과 같은 주어진 예에 대한 클래스를 예측하려는 경우를 예로 들 수 있습니다.

![텍스트 분류 그래픽](https://www.gstatic.com/aihub/tfhub/universal-sentence-encoder/example-classification.png)

아래 튜토리얼은 서로 다른 관점에서 서로 다른 도구를 사용하여 동일한 작업을 해결합니다.

### Keras

[Keras를 사용한 텍스트 분류](https://www.tensorflow.org/tutorials/keras/text_classification_with_hub) - Keras 및 TensorFlow 데이터세트를 사용하여 IMDB 감정 분류자를 빌드하는 예입니다.

### Estimator

[텍스트 분류](https://github.com/tensorflow/hub/blob/master/docs/tutorials/text_classification_with_tf_hub.ipynb) - Estimator를 사용하여 IMDB 감정 분류자를 빌드하는 예입니다. 개선을 위한 여러 팁과 모듈 비교 섹션이 포함되어 있습니다.

### BERT

[TF Hub에서 BERT를 이용해 영화 리뷰 감정 예측](https://github.com/google-research/bert/blob/master/predicting_movie_reviews_with_bert_on_tf_hub.ipynb) - 분류에 BERT 모듈을 사용하는 방법을 보여줍니다. 토큰화와 전처리에 `bert` 라이브러리를 사용하는 내용을 포함합니다.

### Kaggle

[Kaggle의 IMDB 분류](https://github.com/tensorflow/hub/blob/master/examples/colab/text_classification_with_tf_hub_on_kaggle.ipynb) - 데이터 다운로드 및 결과 제출을 포함하여 Colab의 Kaggle 경쟁과 쉽게 상호 작용하는 방법을 보여줍니다.

 | Estimator | Keras | TF2 | TF Datasets | BERT | Kaggle API
--- | --- | --- | --- | --- | --- | ---
[Text classification](https://www.tensorflow.org/hub/tutorials/text_classification_with_tf_hub) | ![끝난](https://www.gstatic.com/images/icons/material/system_gm/1x/bigtop_done_googblue_18dp.png) |  |  |  |  |
[Keras를 이용한 텍스트 분류](https://www.tensorflow.org/tutorials/keras/text_classification_with_hub) |  | ![끝난](https://www.gstatic.com/images/icons/material/system_gm/1x/bigtop_done_googblue_18dp.png) | ![끝난](https://www.gstatic.com/images/icons/material/system_gm/1x/bigtop_done_googblue_18dp.png) | ![끝난](https://www.gstatic.com/images/icons/material/system_gm/1x/bigtop_done_googblue_18dp.png) |  |
[TF Hub에서 BERT를 이용한 영화 리뷰 감성 예측](https://github.com/google-research/bert/blob/master/predicting_movie_reviews_with_bert_on_tf_hub.ipynb) | ![끝난](https://www.gstatic.com/images/icons/material/system_gm/1x/bigtop_done_googblue_18dp.png) |  |  |  | ![끝난](https://www.gstatic.com/images/icons/material/system_gm/1x/bigtop_done_googblue_18dp.png) |
[Kaggle에서 IMDB 분류](https://github.com/tensorflow/hub/blob/master/examples/colab/text_classification_with_tf_hub_on_kaggle.ipynb) | ![끝난](https://www.gstatic.com/images/icons/material/system_gm/1x/bigtop_done_googblue_18dp.png) |  |  |  |  | ![끝난](https://www.gstatic.com/images/icons/material/system_gm/1x/bigtop_done_googblue_18dp.png)

### FastText 임베딩을 사용한 Bangla 작업

TensorFlow Hub는 현재 모든 언어로 모듈을 제공하지 않습니다. 다음 튜토리얼은 빠른 실험과 모듈식 ML 개발을 위해 TensorFlow Hub를 활용하는 방법을 보여줍니다.

[Bangla 기사 분류자](https://github.com/tensorflow/hub/blob/master/examples/colab/bangla_article_classifier.ipynb) - 재사용 가능한 TensorFlow Hub 텍스트 임베딩을 생성하고 이를 사용하여 [BARD Bangla 기사 데이터세트](https://github.com/tanvirfahim15/BARD-Bangla-Article-Classifier)에 대한 Keras 분류자를 훈련하는 방법을 보여줍니다.

## 의미론적 유사성

제로-샷 설정에서 서로 상관 관계가 있는 문장을 찾고 싶을 경우(훈련 예제 없음).

![시맨틱 유사성 그래픽](https://www.gstatic.com/aihub/tfhub/universal-sentence-encoder/example-similarity.png)

### 기본

[의미론적 유사성](https://github.com/tensorflow/hub/blob/master/examples/colab/semantic_similarity_with_tf_hub_universal_encoder.ipynb) - 문장 유사성을 계산하기 위해 문장 인코더 모듈을 사용하는 방법을 보여줍니다.

### 교차 언어

[교차 언어 의미론적 유사성](https://github.com/tensorflow/hub/blob/master/examples/colab/cross_lingual_similarity_with_tf_hub_multilingual_universal_encoder.ipynb) - 교차 언어 문장 인코더 중 하나를 사용하여 언어 간 문장 유사성을 계산하는 방법을 보여줍니다.

### 의미론적 검색

[의미론적 검색](https://github.com/tensorflow/hub/blob/master/examples/colab/retrieval_with_tf_hub_universal_encoder_qa.ipynb) - Q/A 문장 인코더를 사용하여 의미론적 유사성을 기반으로 검색할 문서 모음을 인덱싱하는 방법을 보여줍니다.

### SentencePiece 입력

[범용 인코더 라이트를 이용한 의미론적 유사성](https://github.com/tensorflow/hub/blob/master/examples/colab/semantic_similarity_with_tf_hub_universal_encoder_lite.ipynb) - 텍스트 대신 입력시 [SentencePiece](https://github.com/google/sentencepiece) ID를 허용하는 문장 인코더 모듈을 사용하는 방법을 보여줍니다.

## 모듈 생성

[tfhub.dev](https://tfhub.dev)의 모듈만 사용하는 대신 자체 모듈을 만드는 방법들이 있습니다. ML 코드베이스 모듈화와 공유를 개선하는 데 유용한 도구가 될 수 있습니다.

### 기존의 사전 훈련된 임베딩 래핑하기

[텍스트 임베딩 모듈 exporter](https://github.com/tensorflow/hub/blob/master/examples/text_embeddings/export.py) - 기존의 사전 훈련된 임베딩을 모듈로 래핑하는 도구입니다. 모듈에 텍스트 전처리 연산을 포함하는 방법을 보여줍니다. 이를 통해 토큰 임베딩에서 문장 임베딩 모듈을 만들 수 있습니다.

[텍스트 임베딩 모듈 exporter v2](https://github.com/tensorflow/hub/blob/master/examples/text_embeddings_v2/export_v2.py) - 위와 동일하지만 TensorFlow 2 및 즉시 실행과 호환됩니다.
