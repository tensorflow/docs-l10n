# TensorFlow Lite NNAPI 대리자

[Android Neural Networks API(NNAPI)](https://developer.android.com/ndk/guides/neuralnetworks)는 Android 8.1(API 레벨 27) 이상을 실행하는 모든 Android 기기에서 사용할 수 있습니다. 다음과 같은 하드웨어 가속기를 지원하는 Android 기기의 TensorFlow Lite 모델을 속도를 향상합니다.

- 그래픽 처리 장치(GPU)
- 디지털 신호 프로세서(DSP)
- 신경 처리 장치(NPU)

성능은 기기에서 사용 가능한 특정 하드웨어에 따라 다릅니다.

이 페이지에서는 Java 및 Kotlin에서 TensorFlow Lite 인터프리터로 NNAPI 대리자를 사용하는 방법을 설명합니다. Android C API의 경우 [Android Native Developer Kit 설명서](https://developer.android.com/ndk/guides/neuralnetworks)를 참조하세요.

## 자체 모델에서 NNAPI 대리자 시도하기

### Gradle 가져오기

NNAPI 대리자는 TensorFlow Lite Android 인터프리터, 릴리스 1.14.0 이상의 일부입니다. 모듈 gradle 파일에 다음을 추가하여 프로젝트로 가져올 수 있습니다.

```groovy
dependencies {
   implementation 'org.tensorflow:tensorflow-lite:2.0.0'
}
```

### NNAPI 대리자 초기화하기

TensorFlow Lite 인터프리터를 초기화하기 전에 NNAPI 대리자를 초기화하는 코드를 추가하세요.

참고: NNAPI는 API 레벨 27(Android Oreo MR1)에서 지원되지만 API 레벨 28(Android Pie) 이후에는 연산 지원이 크게 향상되었습니다. 따라서 개발자는 대부분의 시나리오에서 Android Pie 이상의 NNAPI 대리자를 사용하는 것이 좋습니다.

```java
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.nnapi.NnApiDelegate;

Interpreter.Options options = (new Interpreter.Options());
NnApiDelegate nnApiDelegate = null;
// Initialize interpreter with NNAPI delegate for Android Pie or above
if(Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
    nnApiDelegate = new NnApiDelegate();
    options.addDelegate(nnApiDelegate);
}

// Initialize TFLite interpreter
try {
    tfLite = new Interpreter(loadModelFile(assetManager, modelFilename), options);
} catch (Exception e) {
    throw new RuntimeException(e);
}

// Run inference
// ...

// Unload delegate
tfLite.close();
if(null != nnApiDelegate) {
    nnApiDelegate.close();
}
```

## 모범 사례

### 배포 전에 성능 테스트하기

런타임 성능은 모델 아키텍처, 크기, 운영, 하드웨어 가용성 및 런타임 하드웨어 사용률에 따라 크게 달라질 수 있습니다. 예를 들어 앱에서 렌더링에 GPU를 많이 사용하는 경우 NNAPI 가속은 리소스 경합으로 인해 성능을 향상시키지 못할 수 있습니다. 추론 시간을 측정하려면 디버그 로거를 사용하여 간단한 성능 테스트를 실행하는 것이 좋습니다. 운영 환경에서 NNAPI를 활성화하기 전에 사용자 기반을 대표하는 다른 칩셋(제조업체 또는 같은 제조업체의 모델)을 사용하는 여러 전화기에서 테스트를 실행하세요.

고급 개발자를 위해 TensorFlow Lite는 [Android용 모델 벤치마크 도구](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark)도 제공합니다.

### 기기 제외 목록 만들기

프로덕션에서 NNAPI가 예상대로 수행되지 않는 경우가 있을 수 있습니다. 개발자는 특정 모델과 함께 NNAPI 가속을 사용하지 않아야 하는 기기 목록을 유지하는 것이 좋습니다. 다음 코드 조각을 사용하여 검색할 수 있는 `"ro.board.platform"` 값을 기반으로 이 목록을 만들 수 있습니다.

```java
String boardPlatform = "";

try {
    Process sysProcess =
        new ProcessBuilder("/system/bin/getprop", "ro.board.platform").
        redirectErrorStream(true).start();

    BufferedReader reader = new BufferedReader
        (new InputStreamReader(sysProcess.getInputStream()));
    String currentLine = null;

    while ((currentLine=reader.readLine()) != null){
        boardPlatform = line;
    }
    sysProcess.destroy();
} catch (IOException e) {}

Log.d("Board Platform", boardPlatform);
```

고급 개발자의 경우 리모트 구성 시스템을 통해 목록을 유지하는 것이 좋습니다. TensorFlow 팀은 최적의 NNAPI 구성 검색 및 적용을 단순화하고 자동화하는 방법을 적극적으로 연구하고 있습니다.

### 양자화

양자화는 계산을 위해 32bit 부동 소수점 대신 8bit 정수 또는 16bit 부동을 사용하여 모델 크기를 줄입니다. 8bit 정수 모델 크기는 32bit 부동 버전의 1/4이고 16bit 부동 소수점은 크기의 절반입니다. 양자화는 프로세스에서 일부 모델 정확성에 상충 관계를 만들 수도 있지만 성능을 크게 향상시킬 수 있습니다.

여러 유형의 사전 훈련 양자화 기술을 사용할 수 있지만 현재 하드웨어에서 최대한의 지원과 가속화를 위해 [전체 정수 양자화](post_training_quantization#full_integer_quantization_of_weights_and_activations)를 권장합니다. 이 접근 방식은 가중치와 연산을 모두 정수로 변환합니다. 이 양자화 프로세스에는 대표적인 데이터세트가 동작해야 합니다.

### 지원되는 모델 및 연산 사용하기

NNAPI 대리자가 모델의 일부 연산 또는 매개변수 조합을 지원하지 않는 경우 프레임워크는 가속기에서 지원되는 그래프 부분만 실행합니다. 나머지는 CPU에서 실행되므로 분할 실행이 됩니다. CPU/가속기 동기화 비용이 많으므로 CPU에서만 전체 네트워크를 실행하는 것보다 성능이 느려질 수 있습니다.

NNAPI는 모델이 [지원되는 연산](https://developer.android.com/ndk/guides/neuralnetworks#model)만 사용할 때 수행 능력이 가장 좋습니다. 다음 모델은 NNAPI와 호환되는 것으로 알려져 있습니다.

- [MobileNet v1 (224x224) 이미지 분류(부동 모델 다운로드)](https://ai.googleblog.com/2017/06/mobilenets-open-source-models-for.html) [(양자화된 모델 다운로드)](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224_quant.tgz) <br> *(모바일 및 임베디드 기반 비전 애플리케이션을 위해 설계된 이미지 분류 모델)*
- [MobileNet v2 SSD object detection](https://ai.googleblog.com/2018/07/accelerated-training-and-inference-with.html)[(download)](https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/mobile_ssd_v2_float_coco.tflite)
    *(image classification model that detects multiple objects with bounding
    boxes)*
- [MobileNet v1(300x300) SSD(Single Shot Detector) 객체 감지](https://ai.googleblog.com/2018/07/accelerated-training-and-inference-with.html) [(다운로드)] (https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip)
- [PoseNet for pose estimation](https://github.com/tensorflow/tfjs-models/tree/master/posenet)[(download)](https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/multi_person_mobilenet_v1_075_float.tflite)
    *(vision model that estimates the poses of a person(s) in image or video)*

모델에 동적 크기의 출력이 포함된 경우 NNAPI 가속도 지원되지 않습니다. 이 경우 다음과 같은 경고가 표시됩니다.

```none
ERROR: Attempting to use a delegate that only supports static-sized tensors \
with a graph that has dynamic-sized tensors.
```

### NNAPI CPU 구현 사용하기

가속기로 완전히 처리할 수 없는 그래프는 NNAPI CPU 구현으로 대체될 수 있습니다. 그러나 이것은 일반적으로 TensorFlow 인터프리터보다 성능이 떨어지기 때문에 이 옵션은 Android 10(API 레벨 29) 이상에 사용되는 NNAPI 대리자에서 기본적으로 비활성화되어 있습니다. 이 동작을 재정의하려면 `NnApiDelegate.Options` 객체에서 `setUseNnapiCpu`를 `true`로 설정합니다.
