# 광학 문자 인식(OCR)

광학 문자 인식(OCR)은 컴퓨터 비전과 머신 러닝 기술을 사용하여 이미지에서 문자를 인식하는 프로세스입니다. 이 참조 앱은 TensorFlow Lite를 사용하여 OCR을 수행하는 방법을 보여줍니다. [텍스트 감지 모델](https://tfhub.dev/sayakpaul/lite-model/east-text-detector/fp16/1)과 [텍스트 인식 모델](https://tfhub.dev/tulasiram58827/lite-model/keras-ocr/float16/2)의 조합을 텍스트 문자를 인식하기 위한 OCR 파이프라인으로 사용합니다.

## 시작하기

<img src="images/screenshot.gif" class="attempt-right" style="max-width: 300px">

If you are new to TensorFlow Lite and are working with Android, we recommend exploring the following example application that can help you get started.

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/optical_character_recognition/android">Android 예제</a>

Android 이외의 플랫폼을 사용 중이거나 이미 [TensorFlow Lite API](https://www.tensorflow.org/api_docs/python/tf/lite)에 익숙하다면 [TF Hub](https://tfhub.dev/)에서 모델을 다운로드할 수 있습니다.

## How it works

OCR 작업은 2단계로 나뉘는 경우가 많습니다. 먼저 텍스트 감지 모델을 사용하여 가능한 텍스트 주변의 경계 상자를 감지합니다. 둘째, 처리된 경계 상자를 텍스트 인식 모델에 입력하여 경계 상자 내부의 특정 문자를 결정합니다(텍스트 인식 전에 Non-Maximal Suppression, 원근 변환 등도 수행해야 함). 여기서의 경우 두 모델 모두 TensorFlow Hub에서 가져오며 FP16 양자화 모델입니다.

## Performance benchmarks

Performance benchmark numbers are generated with the tool described [here](https://www.tensorflow.org/lite/performance/benchmarks).

<table>
  <thead>
    <tr>
      <th>Model Name</th>
      <th>Model size </th>
      <th>장치</th>
      <th>CPU</th>
      <th>GPU</th>
    </tr>
  </thead>
  <tr>
    <td><a href="https://tfhub.dev/sayakpaul/lite-model/east-text-detector/fp16/1">텍스트 감지</a></td>
    <td>45.9Mb</td>
     <td>Pixel 4 (Android 10)</td>
     <td>181.93ms*</td>
     <td>89.77ms*</td>
  </tr>
  <tr>
    <td><a href="https://tfhub.dev/tulasiram58827/lite-model/keras-ocr/float16/2">텍스트 인식</a></td>
    <td>16.8Mb</td>
     <td>Pixel 4 (Android 10)</td>
     <td>338.33ms*</td>
     <td>해당 없음**</td>
  </tr>
</table>

* 4 threads used.

** 이 모델은 실행을 위해 TensorFlow 작업이 필요하기 때문에 GPU 대리자를 사용할 수 없습니다.

## 입력

텍스트 감지 모델은 (1, 320, 320, 3)의 4차원 `float32` 텐서를 입력으로 받습니다.

텍스트 인식 모델은 (1, 31, 200, 1)의 4차원 `float32` 텐서를 입력으로 받습니다.

## 출력

텍스트 감지 모델은 (1, 80, 80, 5) 형상의 4차원 `float32` 텐서를 경계 상자로, 그리고 (1,80, 80, 5) 형상의 `float32` 텐서를 감지 점수로 반환합니다.

텍스트 인식 모델은 알파벳 목록 '0123456789abcdefghijklmnopqrstuvwxyz'에 대한 매핑 인덱스로 형상 (1, 48)의 2차원 `float32` 텐서를 반환합니다.

## Limitations

- 현재의 [텍스트 인식 모델](https://tfhub.dev/tulasiram58827/lite-model/keras-ocr/float16/2)은 영문자와 숫자가 포함된 합성 데이터를 사용하여 훈련되므로 영어만 지원됩니다.

- 최적이 아닌 조건에 촬영된 OCR에 충분할 정도로 모델이 일반적이지는 않습니다(예: 낮은 조명 조건에서 스마트폰 카메라로 찍은 임의의 이미지).

따라서 TensorFlow Lite로 OCR을 수행하는 방법만을 보여주기 위해 3개의 Google 제품 로고를 선택했습니다. 바로 사용할 수 있는 프로덕션급 OCR 제품을 찾고 있다면 [Google ML Kit](https://developers.google.com/ml-kit/vision/text-recognition) 사용을 고려해야 합니다. 아래에 TFLite를 사용하는 ML Kit는 대부분의 OCR 사용 사례에 충분하지만 TFLite로 고유한 OCR 솔루션을 구축해야 하는 경우가 있을 수 있습니다. 몇 가지 예를 들면 다음과 같습니다.

- 사용하려는 고유한 텍스트 감지/인식 TFLite 모델이 있습니다.
- 특별한 비즈니스 요구 사항(예: 거꾸로 된 텍스트 인식)이 있고 OCR 파이프라인을 사용자 지정해야 합니다.
- ML Kit에서 지원되지 않는 언어를 지원하려고 합니다.
- 대상 사용자 장치에 Google Play 서비스가 설치되어 있지 않을 수도 있습니다.

## References

- OpenCV 텍스트 감지/인식 예: https://github.com/opencv/opencv/blob/master/samples/dnn/text_detection.cpp
- 커뮤니티 기여자의 OCR TFLite 커뮤니티 프로젝트: https://github.com/tulasiram58827/ocr_tflite
- OpenCV 텍스트 감지: https://www.pyimagesearch.com/2018/08/20/opencv-text-detection-east-text-detector/
- OpenCV를 사용한 딥 러닝 기반 텍스트 감지: https://learnopencv.com/deep-learning-based-text-detection-using-opencv-c-python/
