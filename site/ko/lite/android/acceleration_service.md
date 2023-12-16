# Android용 가속 서비스(베타)

베타: Android용 가속 서비스는 현재 베타 버전입니다. 자세한 내용은 이 페이지의 [경고 사항](#caveats) 및 [사용 약관 및 개인 정보 취급 방침](#terms_privacy) 섹션을 참조하세요.

하드웨어 가속을 위해 GPU나 NPU 또는 DSP와 같은 특수 프로세서를 사용하면 추론 성능(경우에 따라 최대 10배 빠른 추론)과 ML 지원 Android 애플리케이션의 사용자 경험을 크게 향상시킬 수 있습니다. 다만 사용자가 보유한 하드웨어와 드라이버의 종류가 다양하기 때문에 각 사용자의 기기에 맞는 최적의 하드웨어 가속 구성을 선택하는 것은 쉽지 않을 수 있습니다. 또한 기기에서 잘못된 구성을 활성화하면 지연 시간이 길어질 수 있으며, 드물지만 하드웨어 비호환성으로 인해 런타임 오류가 발생하거나 정확성 문제로 인해 사용자 경험이 악화될 수 있습니다.

Android용 가속 서비스는 런타임 오류 또는 정확성 문제의 위험을 최소화하는 한편 해당 상황에서 사용자 기기 및 `.tflite` 모델에 맞는 최적의 하드웨어 가속 구성을 선택할 수 있도록 도와주는 API입니다.

가속 서비스는 TensorFlow Lite 모델로 내부 추론 벤치마크를 실행하여 사용자 기기의 다양한 가속 구성을 평가합니다. 이러한 테스트는 일반적으로 모델에 따라 몇 초 내에 완료됩니다. 추론 시간 전에 모든 사용자 기기에서 벤치마크를 한 번 실행하고 결과를 캐시한 뒤 추론 중에 사용할 수 있습니다. 이러한 벤치마크 작업은 프로세스와는 별개로 실행되므로 앱이 충돌할 위험이 줄어듭니다.

모델, 데이터 샘플, 예상 출력("golden" 입력 및 출력)을 제공하면 가속 서비스가 내부 TFLite 추론 벤치마크를 실행하여 하드웨어 권장 사항을 알려줍니다.

![image](../images/acceleration/acceleration_service.png)

가속 서비스는 Android의 사용자 정의 ML 스택의 일부이며 [Google Play 서비스의 TensorFlow Lite](https://www.tensorflow.org/lite/android/play_services)로 작동합니다.

## 프로젝트에 종속성 추가하기

애플리케이션의 build.gradle 파일에 다음 종속성을 추가합니다.

```
implementation  "com.google.android.gms:play-services-tflite-
acceleration-service:16.0.0-beta01"
```

가속 서비스 API는 [Google Play 서비스에 있는 TensorFlow Lite](https://www.tensorflow.org/lite/android/play_services)와 함께 작동합니다. 아직 Play 서비스를 통해 제공되는 TensorFlow Lite 런타임을 사용하고 있지 않다면 [종속성](https://www.tensorflow.org/lite/android/play_services#1_add_project_dependencies_2)을 업데이트해야 합니다.

## 가속 서비스 API 사용 방법

가속 서비스를 사용하려면 먼저 모델 평가에 사용할 가속 구성을 생성합니다(예: OpenGL을 사용하는 GPU). 그런 다음 모델, 일부 샘플 데이터 및 예상 모델 출력으로 검증 구성을 생성합니다. 마지막으로 `validateConfig()`를 호출하여 가속 구성과 검증 구성을 모두 전달합니다.

![image](../images/acceleration/acceleration_service_steps.png)

### 가속 구성 생성하기

가속 구성은 실행 시간 동안 대리자로 변환되는 하드웨어 구성의 표현입니다. 가속 서비스는 내부적으로 이러한 구성을 사용하여 테스트 추론을 수행합니다.

현재 가속 서비스를 사용하면 [GpuAccelerationConfig](https://developers.google.com/android/reference/com/google/android/gms/tflite/acceleration/GpuAccelerationConfig)를 사용하여 GPU 구성(실행 시간 동안 GPU 대리자로 변환됨)과 CPU 추론([CpuAccelerationConfig](https://developers.google.com/android/reference/com/google/android/gms/tflite/acceleration/CpuAccelerationConfig) 사용)을 평가할 수 있습니다. 저희는 현재 다른 하드웨어에 액세스할 수 있는 더 많은 대리자를 지원하고자 노력하고 있습니다.

#### GPU 가속 구성

다음과 같이 GPU 가속 구성을 생성합니다.

```
AccelerationConfig accelerationConfig = new GpuAccelerationConfig.Builder()
  .setEnableQuantizedInference(false)
  .build();
```

모델이 양자화를 사용하는지 여부는 [`setEnableQuantizedInference()`](https://developers.google.com/android/reference/com/google/android/gms/tflite/acceleration/GpuAccelerationConfig.Builder#public-gpuaccelerationconfig.builder-setenablequantizedinference-boolean-value)로 지정해야 합니다.

#### CPU 가속 구성

다음과 같이 CPU 가속을 생성합니다.

```
AccelerationConfig accelerationConfig = new CpuAccelerationConfig.Builder()
  .setNumThreads(2)
  .build();
```

[`setNumThreads()`](https://developers.google.com/android/reference/com/google/android/gms/tflite/acceleration/CpuAccelerationConfig.Builder#setNumThreads(int)) 메서드를 사용하여 CPU 추론을 평가하는 데 사용할 스레드 수를 정의할 수 있습니다.

### 검증 구성 생성하기

검증 구성을 사용하여 가속 서비스에서 추론을 평가하는 방법을 정의할 수 있습니다. 정의한 추론 평가 방법은 합격 여부 결정에 사용하게 됩니다.

- 입력 샘플
- 예상 출력
- 정확성 검증 로직

모델의 성능이 좋을 것으로 예상되는 입력 샘플("golden" 샘플이라고도 함)을 제공해야 합니다.

다음과 같이 [`CustomValidationConfig.Builder`](https://developers.google.com/android/reference/com/google/android/gms/tflite/acceleration/CustomValidationConfig.Builder)를 사용해 [`ValidationConfig`](https://developers.google.com/android/reference/com/google/android/gms/tflite/acceleration/ValidationConfig)를 생성합니다.

```
ValidationConfig validationConfig = new CustomValidationConfig.Builder()
   .setBatchSize(5)
   .setGoldenInputs(inputs)
   .setGoldenOutputs(outputBuffer)
   .setAccuracyValidator(new MyCustomAccuracyValidator())
   .build();
```

[`setBatchSize()`](https://developers.google.com/android/reference/com/google/android/gms/tflite/acceleration/CustomValidationConfig.Builder#setBatchSize(int))를 사용하여 golden 샘플의 개수를 지정합니다. [`setGoldenInputs()`](https://developers.google.com/android/reference/com/google/android/gms/tflite/acceleration/CustomValidationConfig.Builder#public-customvalidationconfig.builder-setgoldeninputs-object...-value)를 사용하여 golden 샘플의 입력을 전달합니다. [`setGoldenOutputs()`](https://developers.google.com/android/reference/com/google/android/gms/tflite/acceleration/CustomValidationConfig.Builder#public-customvalidationconfig.builder-setgoldenoutputs-bytebuffer...-value)를 사용하여 전달된 입력에 대한 예상 출력을 제공합니다.

최대 추론 시간을 [`setInferenceTimeoutMillis()`](https://developers.google.com/android/reference/com/google/android/gms/tflite/acceleration/CustomValidationConfig.Builder#public-customvalidationconfig.builder-setinferencetimeoutmillis-long-value)(기본값 5000ms)로 정의할 수 있습니다. 정의한 시간보다 추론 시간이 길어지면 구성이 거부됩니다.

선택적으로 다음과 같이 사용자 정의 [`AccuracyValidator`](https://developers.google.com/android/reference/com/google/android/gms/tflite/acceleration/CustomValidationConfig.AccuracyValidator)를 생성할 수도 있습니다.

```
class MyCustomAccuracyValidator implements AccuracyValidator {
   boolean validate(
      BenchmarkResult benchmarkResult,
      ByteBuffer[] goldenOutput) {
        for (int i = 0; i < benchmarkResult.actualOutput().size(); i++) {
            if (!goldenOutputs[i]
               .equals(benchmarkResult.actualOutput().get(i).getValue())) {
               return false;
            }
         }
         return true;

   }
}
```

자신의 사용 사례에 적합한 유효성 검사 로직을 정의해야 합니다.

검증 데이터가 이미 모델에 임베드되어 있는 경우 [`EmbeddedValidationConfig`](https://developers.google.com/android/reference/com/google/android/gms/tflite/acceleration/EmbeddedValidationConfig)를 사용할 수 있습니다.

##### 검증 출력 생성하기

Golden 출력은 선택 사항이며, golden 입력을 제공하기만 하면 가속 서비스에서 내부적으로 golden 출력을 생성할 수 있습니다. 또한 [`setGoldenConfig()`](https://developers.google.com/android/reference/com/google/android/gms/tflite/acceleration/CustomValidationConfig.Builder#setGoldenConfig(com.google.android.gms.tflite.acceleration.AccelerationConfig))를 호출하여 이러한 golden 출력을 생성하는 데 사용하는 가속 구성을 정의할 수도 있습니다.

```
ValidationConfig validationConfig = new CustomValidationConfig.Builder()
   .setBatchSize(5)
   .setGoldenInputs(inputs)
   .setGoldenConfig(customCpuAccelerationConfig)
   [...]
   .build();
```

### 가속 구성 검증하기

가속 구성과 검증 구성을 생성한 뒤 모델을 사용하여 이를 평가할 수 있습니다.

Play 서비스 런타임이 올바르게 초기화되었는지, 그리고 기기에서 GPU 대리자를 실행하여 사용할 수 있는지 확인해야 합니다.

```
TfLiteGpu.isGpuDelegateAvailable(context)
   .onSuccessTask(gpuAvailable -> TfLite.initialize(context,
      TfLiteInitializationOptions.builder()
        .setEnableGpuDelegateSupport(gpuAvailable)
        .build()
      )
   );
```

[`AccelerationService.create()`](https://developers.google.com/android/reference/com/google/android/gms/tflite/acceleration/AccelerationService#create(android.content.Context))를 호출하여 [`AccelerationService`](https://developers.google.com/android/reference/com/google/android/gms/tflite/acceleration/AccelerationService)를 인스턴스화합니다.

그런 다음 [`validateConfig()`](https://developers.google.com/android/reference/com/google/android/gms/tflite/acceleration/AccelerationService#validateConfig(com.google.android.gms.tflite.acceleration.Model,%20com.google.android.gms.tflite.acceleration.AccelerationConfig,%20com.google.android.gms.tflite.acceleration.ValidationConfig))를 호출하여 모델에 대한 가속 구성을 검증할 수 있습니다.

```
InterpreterApi interpreter;
InterpreterOptions interpreterOptions = InterpreterApi.Options();
AccelerationService.create(context)
   .validateConfig(model, accelerationConfig, validationConfig)
   .addOnSuccessListener(validatedConfig -> {
      if (validatedConfig.isValid() && validatedConfig.benchmarkResult().hasPassedAccuracyTest()) {
         interpreterOptions.setAccelerationConfig(validatedConfig);
         interpreter = InterpreterApi.create(model, interpreterOptions);
});
```

또한, [`validateConfigs()`](https://developers.google.com/android/reference/com/google/android/gms/tflite/acceleration/AccelerationService#validateConfigs(com.google.android.gms.tflite.acceleration.Model,%20java.lang.Iterable%3Ccom.google.android.gms.tflite.acceleration.AccelerationConfig%3E,%20com.google.android.gms.tflite.acceleration.ValidationConfig))를 호출하고 `Iterable<AccelerationConfig>` 객체를 매개변수로 전달하여 여러 구성을 검증할 수도 있습니다.

`validateConfig()`는 비동기식 작업을 활성화하는 Google Play 서비스에서 `Task<`[`ValidatedAccelerationConfigResult`](https://developers.google.com/android/reference/com/google/android/gms/tflite/acceleration/ValidatedAccelerationConfigResult)`>`를 반환합니다. <br> 검증 호출에서 결과를 얻으려면 [`addOnSuccessListener()`](https://developers.google.com/android/reference/com/google/android/gms/tasks/OnSuccessListener) 콜백을 추가하세요.

#### 인터프리터에서 검증된 구성 사용하기

콜백에서 반환된 `ValidatedAccelerationConfigResult`가 유효한지 확인한 후 `interpreterOptions.setAccelerationConfig()`를 호출하여 검증된 구성을 인터프리터에 대한 가속 구성으로 설정할 수 있습니다.

#### 구성 캐싱

모델에 대한 최적의 가속 구성은 기기에서 변경될 가능성이 거의 없습니다. 따라서 만족스러운 가속 구성을 받으면 다른 검증을 실행하는 대신 받은 가속 구성을 기기에 저장하고 애플리케이션이 이를 검색하여 다음 세션 동안 `InterpreterOptions`를 생성하는 데 사용하도록 해야 합니다. `ValidatedAccelerationConfigResult`의 `serialize()` 및 `deserialize()` 메서드를 사용하면 저장 및 검색 프로세스가 더 쉬워집니다.

### 샘플 애플리케이션

가속 서비스가 제대로 통합되었는지 검토하려면 [샘플 앱](https://github.com/tensorflow/examples/tree/master/lite/examples/acceleration_service/android_play_services)을 살펴보세요.

## 제한 사항

가속 서비스에는 현재 다음과 같은 제한 사항이 있습니다.

- 현재는 CPU 및 GPU 가속 구성만 지원됩니다.
- Google Play 서비스에서 TensorFlow Lite만 지원하며 번들 버전의 TensorFlow Lite를 사용 중인 경우 사용할 수 없습니다.
- `ValidatedAccelerationConfigResult` 객체를 사용하여 [`BaseOptions`](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/task/core/BaseOptions.Builder)를 직접 초기화할 수 없으므로 TensorFlow Lite [작업 라이브러리](https://www.tensorflow.org/lite/inference_with_metadata/task_library/overview)를 지원하지 않습니다.
- 가속 서비스 SDK는 API 레벨 22 이상만 지원합니다.

## 경고 사항{:#caveats}

특히 운영에서 이 SDK를 사용할 계획이 있는 경우 다음 경고 사항을 주의 깊게 검토해 주세요.

- 베타 버전을 종료하고 가속 서비스 API의 안정 버전을 출시하기 전에 현재 베타 버전과 약간의 차이가 있을 수 있는 새 SDK를 퍼블리싱할 예정입니다. 가속 서비스를 계속 사용하려면 이 새 SDK로 마이그레이션하고 제 때에 앱으로 업데이트를 푸시해야 합니다. 그렇게 하지 않을 경우 일정 시간이 지나면 베타 SDK가 더 이상 Google Play 서비스와 호환되지 않을 수 있으며 문제가 발생할 수 있습니다.

- 가속 서비스 API 내의 특정 기능이나 API 전체가 정식 버전으로 제공된다는 보장은 없습니다. 해당 기능이나 API는 무기한 베타 버전으로 유지되거나, 종료되거나, 다른 기능과 결합되어 특정 개발자 대상의 패키지로 제공될 수 있습니다. 가속 서비스 API의 일부 기능 또는 전체 API 자체는 언젠가 정식 버전으로 제공될 수 있지만, 이에 대한 일정은 정해져 있지 않습니다.

## 사용 약관 및 개인 정보 취급 방침{:#terms_privacy}

#### 서비스 약관

가속 서비스 API의 사용 시 [Google API 서비스 약관](https://developers.google.com/terms/)의 적용을 받습니다.<br> 또한 가속 서비스 API는 현재 베타 버전이므로 이를 사용함으로써 위의 경고 사항 섹션에 설명된 잠재적 문제를 인정하고 가속 서비스가 항상 지정된 대로 작동하지 않을 수 있음을 인정하게 됩니다.

#### 개인 정보 취급 방침

가속 서비스 API를 사용하면 입력 데이터(예: 이미지, 동영상, 텍스트)의 처리는 전부 온디바이스로 이루어지며, **가속 서비스는 해당 데이터를 Google 서버로 전송하지 않습니다**. 결과적으로 기기를 떠나서는 안 되는 입력 데이터를 처리할 때 이 API를 사용할 수 있습니다.<br> 가속 서비스 API는 버그 수정, 업데이트된 모델 및 하드웨어 가속기 호환성 정보 등을 수신하기 위해 때때로 Google 서버에 연결할 수 있습니다. 또한 가속 서비스 API는 앱의 API 성능 및 활용에 대한 메트릭을 Google에 전송합니다. Google은 [개인 정보 취급 방침](https://policies.google.com/privacy)에 자세히 설명된 대로 이 메트릭 데이터를 사용하여 성능을 측정하고, API를 디버그, 유지 및 개선하며, 오용 또는 남용을 감지합니다.<br> **여러분은 관련 법률에서 요구하는 바에 따라 Google의 가속 서비스 메트릭 데이터 처리에 대해 앱 사용자에게 알릴 책임이 있습니다.**<br> 당사가 수집하는 데이터에는 다음이 포함됩니다.

- 기기 정보(예: 제조업체, 모델, OS 버전 및 빌드) 및 사용 가능한 ML 하드웨어 가속기(GPU 및 DSP). 진단 및 사용 분석에 사용됩니다.
- 앱 정보(패키지 이름/번들 ID, 앱 버전). 진단 및 사용 분석에 사용됩니다.
- API 구성(예: 이미지 형식, 해상도 등). 진단 및 사용 분석에 사용됩니다.
- 이벤트 유형(예: 초기화, 모델 다운로드, 업데이트, 실행, 탐지 등). 진단 및 사용 분석에 사용됩니다.
- 오류 코드. 진단에 사용됩니다.
- 성능 메트릭. 진단에 사용됩니다.
- 사용자 또는 물리적 기기를 고유하게 식별하지 않는 설치별 식별자. 원격 구성 및 사용 분석의 운영에 사용됩니다.
- 네트워크 요청 발신자 IP 주소. 원격 구성 진단에 사용됩니다. 수집된 IP 주소는 임시 보관됩니다.

## 지원 및 피드백

TensorFlow Issue Tracker를 통해 피드백을 제공하고 지원받을 수 있습니다. Google Play 서비스에서 [이슈 템플릿](https://github.com/tensorflow/tensorflow/issues/new?title=TensorFlow+Lite+in+Play+Services+issue&template=tflite-in-play-services.md)을 사용하여 문제 및 지원 요청을 보고해주세요.
