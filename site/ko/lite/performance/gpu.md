# TensorFlow Lite용 GPU 대리자

머신러닝(ML) 모델을 실행하기 위해 그래픽 처리 장치(GPU)를 사용하는 것은 모델의 성능과 ML 지원 애플리케이션에 대한 사용자 경험을 극적으로 향상할 수 있습니다. TensorFlow Lite를 통해 [*대리자*](./delegates)라고 하는 하드웨어 드라이버를 통해 GPU와 기타 전문적인 프로세서를 사용할 수 있습니다. TensorFlow Lite ML 애플리케이션으로 GPU를 사용할 수 있도록 하면 다음과 같은 이점을 제공 받을 수 있습니다.

- **속도** - GPU는 대규모 병렬 워크로드를 많이 처리할 수 있도록 구축되었습니다. 이 설계로 인해 GPU는 병렬로 처리할 수 있는 입력 텐서에서 각각 작동하는 수많은 연산자로 이루어진 심층 신경망에 아주 적합하며, 이는 일반적으로 지연 시간을 단축합니다. 최상의 시나리오에서, GPU에서의 모델 실행은 이전에는 불가능했던 실시간 애플리케이션을 가능하게 할 만큼 충분히 빠르게 실행될 수 있습니다.
- **전력 효율** - GPU는 아주 효율적이고 최적화된 방식으로 ML 계산을 수행하며, CPU에서 동일한 작업을 실행할 때보다 일반적으로 전력 소모가 적고 발열이 적습니다.

이 문서는 TensorFlow Lite에서의 GPU 지원, GPU 프로세서를 위한 일부 고급 용도에 대한 개요를 제공합니다. 특정 플랫폼에서 GPU 지원을 구현하는 데 대한 더 구체적인 정보는 다음 가이드를 참조하세요.

- [Android용 GPU 지원](../android/delegates/gpu)
- [iOS용 GPU 지원](../ios/delegates/gpu)

## GPU ML 연산 지원{:#supported_ops}

TensorFlow Lite GPU 대리자가 가속할 수 있는 TensorFlow ML 연산(또는 *ops*)에는 몇 가지 제한이 있습니다. 대리자는 16비트와 32비트 부동 정밀도로 다음 연산을 지원합니다.

- `ADD`
- `AVERAGE_POOL_2D`
- `CONCATENATION`
- `CONV_2D`
- `DEPTHWISE_CONV_2D v1-2`
- `EXP`
- `FULLY_CONNECTED`
- `LOGICAL_AND`
- `LOGISTIC`
- `LSTM v2 (Basic LSTM only)`
- `MAX_POOL_2D`
- `MAXIMUM`
- `MINIMUM`
- `MUL`
- `PAD`
- `PRELU`
- `RELU`
- `RELU6`
- `RESHAPE`
- `RESIZE_BILINEAR v1-3`
- `SOFTMAX`
- `STRIDED_SLICE`
- `SUB`
- `TRANSPOSE_CONV`

기본적으로, 모든 연산은 버전 1에서만 지원됩니다. [양자화 지원](#quantized-models)을 활성화하면 적절한 버전을 활성화합니다(예: ADD v2.).

### GPU 지원 문제 해결

일부 연산을 GPU 대리자에서 지원하지 않는 경우, 프레임워크는 GPU에서 그래프의 일부만 실행하고 나머지는 CPU에서 실행합니다. CPU/GPU 동기화 비용이 높기 때문에 이러한 분할 실행 모드는 전체 네트워크가 CPU에서만 실행될 때보다 성능이 종종 저하됩니다. 이러한 경우, 애플리케이션은 다음과 같은 경고를 생성합니다.

```none
WARNING: op code #42 cannot be handled by this delegate.
```

실제 런타임 오류가 아니기 때문에 이러한 유형에 대한 실패의 콜백은 없습니다. GPU 대리자로 모델 실행을 테스트하는 경우, 이러한 경고에 주의해야 합니다. 이러한 경고가 많은 것은 모델이 GPU 가속화에 적합하지 않다는 것을 나타낼 수 있으며 모델을 다시 리팩토링해야 할 수도 있습니다.

## 예제 모델

다음 예제 모델은 TensorFlow Lite를 통해 GPU 가속화를 활용하기 위해 설계되었으며 참조와 테스트를 위해 제공됩니다.

- [MobileNet v1 (224x224) 이미지 분류](https://ai.googleblog.com/2017/06/mobilenets-open-source-models-for.html) - 모바일 및 임베디드 기반 비전 애플리케이션을 위해 설계된 이미지 분류 모델([model](https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/classification/5)).
- [DeepLab 세분화 (257x257)](https://ai.googleblog.com/2018/03/semantic-image-segmentation-with.html) - 개, 고양이, 자동차 등과 같은 시맨틱 레이블을 입력 이미지의 모든 픽셀에 할당하는 이미지 세분화([model](https://tfhub.dev/tensorflow/lite-model/deeplabv3/1/default/1)).
- [MobileNet SSD 객체 감지](https://ai.googleblog.com/2018/07/accelerated-training-and-inference-with.html) - 바운딩 박스로 여러 객체를 감지하는 이미지 분류 모델([model](https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/mobile_ssd_v2_float_coco.tflite)).
- [포즈 예측을 위한 PoseNet](https://github.com/tensorflow/tfjs-models/tree/master/pose-detection) - 이미지나 비디오에 있는 사람의 이미지를 예측하는 비전 모델([model](https://tfhub.dev/tensorflow/lite-model/posenet/mobilenet/float/075/1/default/1)).

## GPU 최적화

다음 기술은 TensorFlow Lite GPU 대리자를 사용하는 GPU 하드웨어에서 모델을 실행할 때 더 나은 성능을 얻을 수 있도록 도울 수 있습니다.

- **Reshape 연산** - CPU에서 빠른 몇몇 연산은 모바일 기기의 GPU에서는 비용이 많이 들 수 있습니다. `BATCH_TO_SPACE`, `SPACE_TO_BATCH`, `SPACE_TO_DEPTH` 등을 포함한 Reshape 연산은 특히 실행하는 데 비용이 많이 듭니다. Reshape 연산 사용을 면밀히 검토해야 하며 데이터 탐색 또는 모델의 초기 반복에만 적용되었을 수 있다는 점을 고려해야 합니다. Reshape 연산을 없애면 성능이 상당히 향상될 수 있습니다.

- **이미지 데이터 채널** - GPU에서, 텐서 데이터는 4채널로 분할되며 형상이 `[B,H,W,5]`인 텐서에서의 계산은 형상이 `[B,H,W,8]`인 텐서에서와 대체로 유사하게 수행되지만 `[B,H,W,4]`보다 상당히 좋지 않습니다. 사용하는 카메라 하드웨어가 RGBA로 이미지 프레임을 지원한다면 3채널 RGB에서 4채널 RGBX로의 메모리 복사를 피할 수 있으므로 해당 4채널 입력을 공급하는 것이 훨씬 더 빠릅니다.

- **모바일 최적화 모델** - 최상의 성능을 위해 모바일에 최적화된 네트워크 아키텍처를 사용하여 분류자를 다시 훈련하는 것을 고려해야 합니다. 장치 내 추론을 최적화하면 모바일 하드웨어 기능을 활용하여 대기 시간과 전력 소비를 크게 줄일 수 있습니다. 온디바이스 추론을 최적화하면 모바일 하드웨어 기능을 활용하여 지연 시간과 전력 소모를 극적으로 단축할 수 있습니다.

## 고급 GPU 지원

모델의 성능을 더 향상하기 위해 양자화 및 직렬화를 포함하여 GPU 처리를 통해 추가적인, 고급 기술을 사용할 수 있습니다. 다음 섹션은 이러한 기술을 더욱 자세하게 설명합니다.

### 양자화된 모델 사용 {:#quantized-models}

이 섹션에서는 GPU 대리자가 다음을 포함한 8비트 양자화된 모델을 가속하는 방법을 설명합니다.

- [양자화 인식 훈련](https://www.tensorflow.org/model_optimization/guide/quantization/training)으로 훈련된 모델
- 훈련 후 [동적 범위 양자화](https://www.tensorflow.org/lite/performance/post_training_quant)
- 훈련 후 [전체 정수 양자화](https://www.tensorflow.org/lite/performance/post_training_integer_quant)

성능을 최적화하려면, 부동 소수점 입력 및 출력 텐서가 모두 있는 모델을 사용하세요.

#### 어떻게 동작합니까?

GPU 백엔드는 부동 소수점 실행만 지원하므로 원래 모델의 '부동 소수점 보기'를 제공하여 양자화된 모델을 실행합니다. 하이 레벨에서 모델을 실행하려면 다음 단계가 수반됩니다.

- *상수 텐서*(예: 가중치/바이어스)는 GPU 메모리로 한 번 역양자화됩니다. 이 연산은 대리자를 TensorFlow Lite에서 사용할 수 있을 때 발생합니다.

- 8비트 양자화된 경우 GPU 프로그램에 대한 *입력 및 출력*은 각 추론에 대해 (각각) 역양자화 및 양자화됩니다. 이 연산은 TensorFlow Lite의 최적화된 커널을 사용하는 CPU에서 이루어집니다.

- *양자화 시뮬레이터*는 양자화된 동작을 모방하기 위해 연산 사이에 삽입됩니다. 이 접근 방법은 연산자가 양자화 동안 학습한 경계를 따르기를 기대하는 모델에 필요합니다.

GPU 대리자로 이 기능을 활성화하는 것에 관한 정보는 다음을 참조하세요.

- [Android에서 GPU를 통해 양자화된 모델](../android/delegates/gpu#quantized-models) 사용
- [iOS에서 GPU를 통해 양자화된 모델](../ios/delegates/gpu#quantized-models) 사용

### 직렬화로 초기화 시간 단축하기 {:#delegate_serialization}

GPU 대리자 기능을 통해 사전 컴파일된 커널 코드와 이전 실행에서 직렬화되고 디스크에 저장된 모델 데이터를 로드할 수 있습니다. 이 접근 방법은 재컴파일을 방지하고 시작 시간을 최대 90%까지 단축할 수 있습니다. 이러한 개선은 시간 절약을 위해 디스크 공간을 교환하여 달성됩니다. 다음 예제에서 나타난 대로 이 기능을 몇 가지 구성 옵션을 통해 사용할 수 있습니다.

<div>
  <devsite-selector>
    <section>
      <h3>C++</h3>
      <p></p>
<pre class="prettyprint lang-cpp">    TfLiteGpuDelegateOptionsV2 options = TfLiteGpuDelegateOptionsV2Default();
    options.experimental_flags |= TFLITE_GPU_EXPERIMENTAL_FLAGS_ENABLE_SERIALIZATION;
    options.serialization_dir = kTmpDir;
    options.model_token = kModelToken;

    auto* delegate = TfLiteGpuDelegateV2Create(options);
    if (interpreter-&gt;ModifyGraphWithDelegate(delegate) != kTfLiteOk) return false;
      </pre>
    </section>
    <section>
      <h3>Java</h3>
      <p></p>
<pre class="prettyprint lang-java">    GpuDelegate delegate = new GpuDelegate(
      new GpuDelegate.Options().setSerializationParams(
        /* serializationDir= */ serializationDir,
        /* modelToken= */ modelToken));

    Interpreter.Options options = (new Interpreter.Options()).addDelegate(delegate);
      </pre>
    </section>
  </devsite-selector>
</div>

직렬화 기능을 사용할 때 코드가 다음 구현 규칙을 준수하도록 하세요.

- 다른 앱에서 액세스할 수 없는 디렉터리에 직렬화 데이터를 저장합니다. Android 장치에서는 현재 애플리케이션에 대해 비공개인 위치를 가리키는 [`getCodeCacheDir()`](https://developer.android.com/reference/android/content/Context#getCacheDir())를 사용합니다.
- 이 모델 토큰은 특정 모델의 기기에 대해 고유해야 합니다. [`farmhash::Fingerprint64`](https://github.com/google/farmhash)와 같은 라이브러리를 사용하는 모델 데이터에서 지문을 생성하여 모델 토큰을 계산할 수 있습니다.

참고: 이 직렬화 기능을 사용하려면 [OpenCL SDK](https://github.com/KhronosGroup/OpenCL-SDK)가 필요합니다.
