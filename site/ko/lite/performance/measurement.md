# 성능 측정

## 벤치마크 도구

TensorFlow Lite 벤치마크 도구는 현재 다음과 같은 중요한 성능 메트릭에 대한 통계를 측정하고 계산합니다.

- 초기화 시간
- 워밍업 상태 추론 시간
- 정상 상태 추론 시간
- 초기화 시간 동안 메모리 사용량
- 전체 메모리 사용량

벤치마크 도구는 Android 및 iOS용 벤치마크 앱과 기본 명령줄 바이너리로 사용할 수 있으며, 모두 동일한 핵심 성능 측정 로직을 공유합니다. 사용 가능한 옵션 및 출력 형식은 런타임 환경의 차이로 인해 약간 다릅니다.

### Android 벤치마크 앱

Android에서 벤치마크 도구를 사용하는 두 가지 옵션이 있습니다. 하나는 [기본 벤치마크 바이너리](#native-benchmark-binary)이고 다른 하나는 Android 벤치마크 앱으로, 앱에서 모델이 어떻게 작동하는지 더 잘 보여줍니다. 어느 쪽이든 벤치마크 도구의 수치는 실제 앱에서 모델로 추론을 실행할 때와 약간 다를 것입니다.

이 Android 벤치마크 앱에는 UI가 없습니다. `adb` 명령을 사용하여 설치 및 실행하고 `adb logcat` 명령을 사용하여 결과를 가져옵니다.

#### 앱 다운로드 또는 빌드

아래 링크를 사용하여 사전 빌드된 야간 Android 벤치마크 앱을 다운로드합니다.

- [android_aarch64](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_aarch64_benchmark_model.apk)
- [android_arm](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_arm_benchmark_model.apk)

[Flex delegate](https://www.tensorflow.org/lite/guide/ops_select)를 통해 [TF ops](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/delegates/flex)를 지원하는 Android 벤치마크 앱의 경우 아래 링크를 사용합니다.

- [android_aarch64](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_aarch64_benchmark_model_plus_flex.apk)
- [android_arm](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_arm_benchmark_model_plus_flex.apk)

다음 [안내](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark/android)에 따라 소스에서 앱을 빌드할 수도 있습니다.

참고: x86 CPU 또는 Hexagon 대리자에서 Android 벤치마크 apk를 실행하려는 경우 또는 모델에 [특정 TF 연산자](../guide/ops_select) 또는 [사용자 정의 연산자](../guide/ops_custom)가 포함된 경우, 소스에서 앱을 빌드해야 합니다.

#### 벤치마크 준비

벤치마크 앱을 실행하기 전에 앱을 설치하고 다음과 같이 모델 파일을 기기에 푸시합니다.

```shell
adb install -r -d -g android_aarch64_benchmark_model.apk
adb push your_model.tflite /data/local/tmp
```

#### 벤치마크 실행

```shell
adb shell am start -S \
  -n org.tensorflow.lite.benchmark/.BenchmarkModelActivity \
  --es args '"--graph=/data/local/tmp/your_model.tflite \
              --num_threads=4"'
```

`graph`는 필수 매개변수입니다.

- `graph`: `string`<br> TFLite 모델 파일의 경로입니다.

벤치마크 실행을 위해 더 많은 선택적 매개변수를 지정할 수 있습니다.

- `num_threads`: `int`(기본값=1)<br> TFLite 인터프리터를 실행하는 데 사용할 스레드 수입니다.
- `use_gpu`: `bool`(기본값=false)<br> [GPU 대리자](gpu)를 사용합니다.
- `use_nnapi`: `bool`(기본값=false)<br> [NNAPI 대리자](nnapi)를 사용합니다.
- `use_xnnpack`: `bool`(기본값=`false`)<br> [XNNPACK 대리자](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/delegates/xnnpack)를 사용합니다.
- `use_hexagon`: `bool`(기본값=`false`)<br> [Hexagon 대리자](hexagon_delegate)를 사용합니다.

사용 중인 기기에 따라 이러한 옵션 중 일부를 사용할 수 없거나 효과가 없을 수 있습니다. 벤치마크 앱으로 실행할 수 있는 추가 성능 매개변수는 [매개변수](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark#parameters)를 참조하세요.

`logcat` 명령을 사용하여 결과를 봅니다.

```shell
adb logcat | grep "Average inference"
```

벤치마크 결과는 다음과 같이 보고됩니다.

```
... tflite  : Average inference timings in us: Warmup: 91471, Init: 4108, Inference: 80660.1
```

### 네이티브 벤치마크 바이너리

벤치마크 도구는 네이티브 바이너리 `benchmark_model`로도 제공됩니다. Linux, Mac, 임베디드 장치 및 Android 기기의 셸 명령줄에서 이 도구를 실행할 수 있습니다.

#### 바이너리 다운로드 또는 빌드

아래 링크를 따라 사전 빌드된 네이티브 명령줄 바이너리 야간 버전을 다운로드합니다.

- [linux_x86-64](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/linux_x86-64_benchmark_model)
- [linux_aarch64](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/linux_aarch64_benchmark_model)
- [linux_arm](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/linux_arm_benchmark_model)
- [android_aarch64](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_aarch64_benchmark_model)
- [android_arm](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_arm_benchmark_model)

[Flex 대리자](https://www.tensorflow.org/lite/guide/ops_select)를 통해 [TF ops](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/delegates/flex)를 지원하는 사전 빌드된 바이너리 야간 버전이 필요한 경우, 아래 링크를 사용하세요.

- [linux_x86-64](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/linux_x86-64_benchmark_model_plus_flex)
- [linux_aarch64](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/linux_aarch64_benchmark_model_plus_flex)
- [linux_arm](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/linux_arm_benchmark_model_plus_flex)
- [android_aarch64](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_aarch64_benchmark_model_plus_flex)
- [android_arm](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_arm_benchmark_model_plus_flex)

컴퓨터의 [소스](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark)에서 네이티브 벤치마크 바이너리를 빌드할 수도 있습니다.

```shell
bazel build -c opt //tensorflow/lite/tools/benchmark:benchmark_model
```

Android NDK 도구 모음으로 빌드하려면 먼저 이 [가이드](../guide/build_android#set_up_build_environment_without_docker)를 따라 우선 빌드 환경을 구성하거나 [이 가이드](../guide/build_android#set_up_build_environment_using_docker)에 설명된 대로 docker 이미지를 사용해야 합니다.

```shell
bazel build -c opt --config=android_arm64 \
  //tensorflow/lite/tools/benchmark:benchmark_model
```

참고: 벤치마킹을 위해 Android 기기에서 바이너리를 직접 푸시하고 실행하는 것은 유효한 접근 방식이지만 실제 Android 앱 내에서 실행하는 것과는 미묘한(그러나 관찰 가능한) 성능 차이가 발생할 수 있습니다. 특히 Android의 스케줄러는 `adb shell ...`을 통해 실행되는 포그라운드 활동/애플리케이션과 일반 백그라운드 바이너리 간에 서로 차이가 있는 스레드 및 프로세스 우선 순위에 따라 동작을 조정합니다. 이 맞춤형 동작은 TensorFlow Lite로 다중 스레드 CPU 실행을 활성화할 때 가장 분명합니다. 따라서 성능 측정에는 Android 벤치마크 앱이 선호됩니다.

#### 벤치마크 실행

컴퓨터에서 벤치마크를 실행하려면 셸에서 바이너리를 실행합니다.

```shell
path/to/downloaded_or_built/benchmark_model \
  --graph=your_model.tflite \
  --num_threads=4
```

네이티브 명령줄 바이너리에 대해 위에서 언급한 것과 동일한 [매개변수](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark#parameters) 집합을 사용할 수 있습니다.

#### 모델 ops 프로파일링

벤치마크 모델 바이너리를 사용하면 모델 ops를 프로파일링하고 각 연산자의 실행 시간을 가져올 수도 있습니다. 이렇게 하려면 호출 중에 `--enable_op_profiling=true` 플래그를 `benchmark_model`로 전달합니다. 자세한 내용은 [여기](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark#profiling-model-operators)에 설명되어 있습니다.

### 한 번의 실행으로 여러 성능 옵션을 제공하는 네이티브 벤치마크 바이너리

한 번의 실행으로 [여러 성능 옵션을 벤치마크](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark#benchmark-multiple-performance-options-in-a-single-run)하기 위해 편리하고 간단한 C++ 바이너리도 제공됩니다. 이 바이너리는 한 번에 하나의 성능 옵션만 벤치마크할 수 있는 앞서 언급한 벤치마크 도구를 기반으로 빌드되었습니다. 동일한 빌드/설치/실행 프로세스를 공유하지만 이 바이너리의 BUILD 대상 이름은 `benchmark_model_performance_options`이며 몇 가지 추가 매개변수가 필요합니다. 이 바이너리의 중요한 매개변수는 다음과 같습니다.

`perf_options_list`: `string`(기본값='all')<br> 벤치마크할 TFLite 성능 옵션의 쉼표로 구분된 목록입니다.

다음과 같이 이 도구에 대해 사전 빌드된 야간 바이너리 버전을 얻을 수 있습니다.

- [linux_x86-64](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/linux_x86-64_benchmark_model_performance_options)
- [linux_aarch64](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/linux_aarch64_benchmark_model_performance_options)
- [linux_arm](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/linux_arm_benchmark_model_performance_options)
- [android_aarch64](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_arm_benchmark_model_performance_options)
- [android_arm](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_arm_benchmark_model_performance_options)

### iOS 벤치마크 앱

iOS 기기에서 벤치마크를 실행하려면 [소스](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark/ios)에서 앱을 빌드해야 합니다. TensorFlow Lite 모델 파일을 소스 트리의 [benchmark_data](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark/ios/TFLiteBenchmark/TFLiteBenchmark/benchmark_data) 디렉터리에 넣고 `benchmark_params.json` 파일을 수정합니다. 이러한 파일은 앱으로 패키징되고 앱은 디렉터리에서 데이터를 읽습니다. 자세한 지침은 [iOS 벤치마크 앱](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark/ios)을 참조하세요.

## 잘 알려진 모델의 성능 벤치마크

이 섹션에는 일부 Android 및 iOS 기기에서 잘 알려진 모델을 실행할 때 TensorFlow Lite 성능 벤치마크가 나열되어 있습니다.

### Android 성능 벤치마크

이러한 성능 벤치마크 수치는 [네이티브 벤치마크 바이너리](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark)로 생성되었습니다.

Android 벤치마크의 경우, CPU 선호도는 분산을 줄이기 위해 기기의 큰 코어를 사용하도록 설정됩니다([자세한 내용](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark#reducing-variance-between-runs-on-android) 참조).

`/data/local/tmp/tflite_models` 디렉터리에 모델을 다운로드하고 압축을 풀었다고 가정합니다. 벤치마크 바이너리는 [이러한 지침](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark#on-android)에 따라 빌드되며 `/data/local/tmp` 디렉터리에 있다고 가정합니다.

벤치마크를 실행하려면:

```sh
adb shell /data/local/tmp/benchmark_model \
  --num_threads=4 \
  --graph=/data/local/tmp/tflite_models/${GRAPH} \
  --warmup_runs=1 \
  --num_runs=50
```

nnapi 대리자로 실행하려면 `--use_nnapi=true`를 설정합니다. GPU 대리자로 실행하려면 `--use_gpu=true`를 설정합니다.

아래 성능 값은 Android 10에서 측정되었습니다.

<table>
  <thead>
    <tr>
      <th>모델명</th>
      <th>기기</th>
      <th>CPU, 4 스레드</th>
      <th>GPU</th>
      <th>NNAPI</th>
    </tr>
  </thead>
  <tr>
    <td rowspan="2">       <a href="https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224.tgz" class="_active_edit_href">Mobilenet_1.0_224(float)</a>     </td>
    <td>Pixel 3</td>
    <td>23.9 ms</td>
    <td>6.45 ms</td>
    <td>13.8 ms</td>
  </tr>
   <tr>
     <td>Pixel 4</td>
    <td>14.0 ms</td>
    <td>9.0 ms</td>
    <td>14.8 ms</td>
  </tr>
  <tr>
    <td rowspan="2">       <a href="https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224_quant.tgz">Mobilenet_1.0_224 (quant)</a>
</td>
    <td>Pixel 3</td>
    <td>13.4 ms</td>
    <td>---</td>
    <td>6.0 ms</td>
  </tr>
   <tr>
     <td>Pixel 4</td>
    <td>5.0 ms</td>
    <td>---</td>
    <td>3.2 ms</td>
  </tr>
  <tr>
    <td rowspan="2">       <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/nasnet_mobile_2018_04_27.tgz">NASNet 모바일</a>
</td>
    <td>Pixel 3</td>
    <td>56 ms</td>
    <td>---</td>
    <td>102 ms</td>
  </tr>
   <tr>
     <td>Pixel 4</td>
    <td>34.5 ms</td>
    <td>---</td>
    <td>99.0 ms</td>
  </tr>
  <tr>
    <td rowspan="2">       <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/squeezenet_2018_04_27.tgz">SqueezeNet</a>
</td>
    <td>Pixel 3</td>
    <td>35.8 ms</td>
    <td>9.5 ms</td>
    <td>18.5 ms</td>
  </tr>
   <tr>
     <td>Pixel 4</td>
    <td>23.9 ms</td>
    <td>11.1 ms</td>
    <td>19.0 ms</td>
  </tr>
  <tr>
    <td rowspan="2">       <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_resnet_v2_2018_04_27.tgz">Inception_ResNet_V2</a>
</td>
    <td>Pixel 3</td>
    <td>422 ms</td>
    <td>99.8 ms</td>
    <td>201 ms</td>
  </tr>
   <tr>
     <td>Pixel 4</td>
    <td>272.6 ms</td>
    <td>87.2 ms</td>
    <td>171.1 ms</td>
  </tr>
  <tr>
    <td rowspan="2">       <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_v4_2018_04_27.tgz">Inception_V4</a>
</td>
    <td>Pixel 3</td>
    <td>486 ms</td>
    <td>93 ms</td>
    <td>292 ms</td>
  </tr>
   <tr>
     <td>Pixel 4</td>
    <td>324.1 ms</td>
    <td>97.6 ms</td>
    <td>186.9 ms</td>
  </tr>
 </table>

### iOS 성능 벤치마크

이러한 성능 벤치마크 수치는 [iOS 벤치마크 앱](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark/ios)에서 생성되었습니다.

iOS 벤치마크를 실행하기 위해 적절한 모델을 포함하도록 벤치마크 앱이 수정되었으며 `benchmark_params.json`이 `num_threads`를 2로 설정하도록 수정되었습니다. GPU 대리자를 사용하기 위해 `"use_gpu" : "1"` 및 `"gpu_wait_type" : "aggressive"` 옵션이 `benchmark_params.json`에도 추가되었습니다.

<table>
  <thead>
    <tr>
      <th>모델명</th>
      <th>기기</th>
      <th>CPU, 2 스레드</th>
      <th>GPU</th>
    </tr>
  </thead>
  <tr>
    <td>       <a href="https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224.tgz">Mobilenet_1.0_224(부동)</a>
</td>
    <td>iPhone XS</td>
    <td>14.8 ms</td>
    <td>3.4 ms</td>
  </tr>
  <tr>
    <td>       <a href="https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224_quant.tgz)">Mobilenet_1.0_224 (quant)</a>
</td>
    <td>iPhone XS</td>
    <td>11 ms</td>
    <td>---</td>
  </tr>
  <tr>
    <td>       <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/nasnet_mobile_2018_04_27.tgz">NASNet 모바일</a>
</td>
    <td>iPhone XS</td>
    <td>30.4 ms</td>
    <td>---</td>
  </tr>
  <tr>
    <td>       <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/squeezenet_2018_04_27.tgz">SqueezeNet</a>
</td>
    <td>iPhone XS</td>
    <td>21.1 ms</td>
    <td>15.5 ms</td>
  </tr>
  <tr>
    <td>       <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_resnet_v2_2018_04_27.tgz">Inception_ResNet_V2</a>
</td>
    <td>iPhone XS</td>
    <td>261.1 ms</td>
    <td>45.7 ms</td>
  </tr>
  <tr>
    <td>       <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_v4_2018_04_27.tgz">Inception_V4</a>
</td>
    <td>iPhone XS</td>
    <td>309 ms</td>
    <td>54.4 ms</td>
  </tr>
 </table>

## Android에서 TensorFlow Lite 내부 추적하기

참고: 이 기능은 실험적이며 야간 버전의 Tensorflow Lite 라이브러리로 Android 앱을 빌드한 경우에만 사용할 수 있습니다. v2.3까지의 안정적인 라이브러리는 이를 지원하지 않습니다.

Android 앱의 TensorFlow Lite 인터프리터의 내부 이벤트는 [Android 추적 도구](https://developer.android.com/topic/performance/tracing)로 캡처할 수 있습니다. 이는 Android [Trace](https://developer.android.com/reference/android/os/Trace) API와 동일한 이벤트이므로 Java/Kotlin 코드에서 캡처된 이벤트가 TensorFlow Lite 내부 이벤트와 함께 표시됩니다.

다음은 이벤트의 몇 가지 예입니다.

- 연산자 호출
- 대리자에 의한 그래프 수정
- 텐서 할당

추적 캡처를 위한 다양한 옵션 중에서 이 가이드는 Android Studio CPU Profiler 및 System Tracing 앱을 다룹니다. 다른 옵션은 [Perfetto 명령줄 도구](https://developer.android.com/studio/command-line/perfetto) 또는 [Systrace 명령줄 도구](https://developer.android.com/topic/performance/tracing/command-line)를 참조하세요.

### Java 코드에 추적 이벤트 추가하기

이것은 [이미지 분류](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android) 예제 앱의 코드 조각입니다. TensorFlow Lite 인터프리터는 `recognizeImage/runInference` 섹션에서 실행됩니다. 이 단계는 선택 사항이지만 추론 호출이 발생한 위치를 파악하는 데 유용합니다.

```java
  Trace.beginSection("recognizeImage");
  ...
  // Runs the inference call.
  Trace.beginSection("runInference");
  tflite.run(inputImageBuffer.getBuffer(), outputProbabilityBuffer.getBuffer().rewind());
  Trace.endSection();
  ...
  Trace.endSection();

```

### TensorFlow Lite 추적 활성화하기

TensorFlow Lite 추적을 활성화하려면 Android 앱을 시작하기 전에 Android 시스템 속성 `debug.tflite.tracing`을 1로 설정합니다.

```shell
adb shell setprop debug.tflite.trace 1
```

TensorFlow Lite 인터프리터가 초기화될 때 이 속성이 설정되면 인터프리터의 주요 이벤트(예: 연산자 호출)가 추적됩니다.

모든 추적을 캡처한 후 속성 값을 0으로 설정하여 추적을 비활성화합니다.

```shell
adb shell setprop debug.tflite.trace 0
```

### Android Studio CPU Profiler

아래 단계에 따라 [Android Studio CPU Profiler](https://developer.android.com/studio/profile/cpu-profiler)로 추적을 캡처합니다.

1. 상단 메뉴에서 **Run(실행) &gt; Profile 'app'(프로파일 '앱')**을 선택합니다.

2. 프로파일러 창이 나타나면 CPU 타임라인의 아무 곳이나 클릭합니다.

3. CPU 프로파일링 모드 중 'Trace System Calls(시스템 호출 추적)'을 선택합니다.

    ![Select 'Trace System Calls'](https://github.com/tensorflow/docs-l10n/blob/master/site/ko/lite/performance/images/as_select_profiling_mode.png?raw=true)

4. 'Record(기록)' 버튼을 누릅니다.

5. 'Stop(중지)' 버튼을 누릅니다.

6. 추적 결과를 살펴봅니다.

    ![Android Studio trace](https://github.com/tensorflow/docs-l10n/blob/master/site/ko/lite/performance/images/as_traces.png?raw=true)

이 예제에서는 스레드의 이벤트 계층 구조와 각 연산자 시간에 대한 통계를 볼 수 있으며 스레드 간 전체 앱의 데이터 흐름도 볼 수 있습니다.

### 시스템 추적 앱

[시스템 추적 앱](https://developer.android.com/topic/performance/tracing/on-device)에 설명된 단계에 따라 Android Studio 없이 추적을 캡처합니다.

이 예에서는 동일한 TFLite 이벤트가 캡처되어 Android 기기 버전에 따라 Perfetto 또는 Systrace 형식으로 저장되었습니다. 캡처된 추적 파일은 [Perfetto UI](https://ui.perfetto.dev/#!/)에서 열 수 있습니다.

![Perfetto trace](https://github.com/tensorflow/docs-l10n/blob/master/site/ko/lite/performance/images/perfetto_traces.png?raw=true)

### 추적 데이터 사용하기

추적 데이터를 사용하면 성능 병목현상을 식별할 수 있습니다.

다음은 성능 향상을 위해 프로파일러 및 잠재적 솔루션에서 얻을 수 있는 몇 가지 통찰력의 예입니다.

- 사용 가능한 CPU 코어 수가 추론 스레드 수보다 적으면 CPU 스케줄링 오버헤드로 인해 성능이 저하될 수 있습니다. 모델 추론과 겹치지 않도록 애플리케이션에서 다른 CPU 집약적 작업을 다시 예약하거나 인터프리터 스레드 수를 조정할 수 있습니다.
- 연산자가 완전히 대리되지 않으면 모델 그래프의 일부가 예상되는 하드웨어 가속기가 아닌 CPU에서 실행됩니다. 지원되지 않는 연산자를 지원되는 유사한 연산자로 대체할 수 있습니다.
