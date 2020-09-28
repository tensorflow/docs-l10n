# 전이 학습 오디오 인식기

이 튜토리얼에서는 TensorFlow.js를 사용하여 브라우저에서 훈련할 사용자 정의 오디오 분류자를 빌드하는 방법을 알아봅니다. 소리를 내어 브라우저의 슬라이더를 제어하는 데 사용합니다.

전이 학습을 사용하여 상대적으로 적은 양의 훈련 데이터로 짧은 소리를 분류하는 모델을 만듭니다. [음성 명령 인식을](https://github.com/tensorflow/tfjs-models/tree/master/speech-commands) 위해 사전에 훈련된 모델을 사용하게 됩니다. 사용자 정의 사운드 클래스를 인식하도록 해당 모델 위에 새 모델을 훈련합니다.

이 튜토리얼은 codelab으로 제공됩니다. [링크를 따라 codelab을 엽니다.](https://codelabs.developers.google.com/codelabs/tensorflowjs-audio-codelab/index.html)
