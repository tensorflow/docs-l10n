# 스마트 답장

<img src="../images/smart_reply.png" class="attempt-right">

## 시작하기

스마트 답장 모델은 채팅 메시지를 기반으로 답장 제안을 생성합니다. 사용자가 수신 메시지에 쉽게 응답할 수 있도록 상황에 맞는 원터치 응답을 제공하는 제안입니다.

<a class="button button-primary" href="https://tfhub.dev/tensorflow/lite-model/smartreply/1/default/1?lite-format=tflite">스타터 모델 다운로드하기</a>

### 샘플 애플리케이션

Android에서 스마트 답장 모델을 보여주는 TensorFlow Lite 샘플 애플리케이션이 있습니다.

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/smart_reply/android">Android 예제 보기</a>

앱 동작 방식을 알아보려면 [GitHub 페이지](https://github.com/tensorflow/examples/tree/master/lite/examples/smart_reply/android/)를 읽어보세요. 이 프로젝트에서는 사용자 정의 C++ 작업을 사용하여 앱을 빌드하는 방법도 알아봅니다.

## 동작 원리

이 모델은 대화형 채팅 메시지에 대한 답장 제안을 생성합니다.

온디바이스 모델의 여러 가지 이점은 다음과 같습니다.

<ul>
  <li>빠름: 모델이 기기에 있으며 인터넷 연결이 필요하지 않습니다. 따라서 추론은 매우 빠르고 평균 지연 시간이 몇 밀리초에 불과합니다.</li>
  <li>리소스 효율성: 모델이 기기에서 차지하는 메모리 공간이 적습니다.</li>
  <li>개인 정보 보호: 사용자 데이터는 절대로 기기를 벗어나지 않습니다.</li>
</ul>

## 예제 출력

<img alt="스마트 답장을 보여주는 애니메이션" src="images/smart_reply.gif" style="max-width: 300px">

## 자세히 알아보기

<ul>
  <li><p data-md-type="paragraph"><a href="https://arxiv.org/pdf/1708.00630.pdf">연구 논문</a></p></li>
  <li><p data-md-type="paragraph"><a href="https://github.com/tensorflow/examples/tree/master/lite/examples/smart_reply/android">소스 코드</a></p></li>
</ul>

## 사용자

<ul>
  <li><p data-md-type="paragraph"><a href="https://www.blog.google/products/gmail/save-time-with-smart-reply-in-gmail/">Gmail</a></p></li>
  <li><p data-md-type="paragraph"><a href="https://www.blog.google/products/gmail/computer-respond-to-this-email/">받은 편지함</a></p></li>
  <li><p data-md-type="paragraph"><a href="https://blog.google/products/allo/google-allo-smarter-messaging-app/">Allo</a></p></li>
  <li><p data-md-type="paragraph"><a href="https://research.googleblog.com/2017/02/on-device-machine-intelligence.html">Android Wear의 스마트 답장</a></p></li>
</ul>
