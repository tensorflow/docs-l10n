# TensorFlow Lite Support Library로 입력 및 출력 데이터 처리하기

참고: TensorFlow Lite Support Library는 현재 Android만 지원합니다.

모바일 애플리케이션 개발자는 일반적으로 비트맵과 같은 형식화된 객체 또는 정수와 같은 기본 형식과 상호 작용합니다. 하지만, 기기 내 머신러닝 모델을 실행하는 TensorFlow Lite 인터프리터는 디버깅 및 조작이 어려울 수 있는 ByteBuffer 형식의 텐서를 사용합니다. [TensorFlow Lite Support Library](https://github.com/tensorflow/tflite-support/tree/master/tensorflow_lite_support/java)는 TensorFlow Lite 모델의 입력 및 출력을 처리하고 TensorFlow Lite 인터프리터를 더 쉽게 사용할 수 있도록 설계되었습니다.

## 시작하기

### Gradle 종속성 및 기타 설정 가져오기

`.tflite` 모델 파일을 모델이 실행될 Android 모듈의 assets 디렉토리에 복사합니다. 파일을 압축하지 않도록 지정하고 TensorFlow Lite 라이브러리를 모듈의 `build.gradle` 파일에 추가합니다.

```java
android {
    // Other settings

    // Specify tflite file should not be compressed for the app apk
    aaptOptions {
        noCompress "tflite"
    }

}

dependencies {
    // Other dependencies

    // Import tflite dependencies
    implementation 'org.tensorflow:tensorflow-lite:0.0.0-nightly-SNAPSHOT'
    // The GPU delegate library is optional. Depend on it as needed.
    implementation 'org.tensorflow:tensorflow-lite-gpu:0.0.0-nightly-SNAPSHOT'
    implementation 'org.tensorflow:tensorflow-lite-support:0.0.0-nightly-SNAPSHOT'
}
```

[JCenter에서 호스팅되는 TensorFlow Lite Support Library AAR](https://bintray.com/google/tensorflow/tensorflow-lite-support)에서 다양한 버전의 지원 라이브러리를 살펴보세요.

### 기본 이미지 조작 및 변환

TensorFlow Lite Support Library에는 자르기 및 크기 조정과 같은 기본적인 이미지 조작 메서드 모음이 있습니다. 이러한 메서드를 사용하려면 `ImagePreprocessor`를 만들고 필요한 연산을 추가합니다. 이미지를 TensorFlow Lite 인터프리터에 필요한 텐서 형식으로 변환하려면 입력으로 사용할 `TensorImage`를 만듭니다.

```java
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;

// Initialization code
// Create an ImageProcessor with all ops required. For more ops, please
// refer to the ImageProcessor Architecture section in this README.
ImageProcessor imageProcessor =
    new ImageProcessor.Builder()
        .add(new ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR))
        .build();

// Create a TensorImage object. This creates the tensor of the corresponding
// tensor type (uint8 in this case) that the TensorFlow Lite interpreter needs.
TensorImage tImage = new TensorImage(DataType.UINT8);

// Analysis code for every frame
// Preprocess the image
tImage.load(bitmap);
tImage = imageProcessor.process(tImage);
```

텐서의 `DataType`은 [메타데이터 추출기 라이브러리](../convert/metadata.md#read-the-metadata-from-models) 및 기타 모델 정보를 통해 읽을 수 있습니다.

### 출력 객체 생성 및 모델 실행하기

모델을 실행하기 전에 결과를 저장할 컨테이너 객체를 만들어야 합니다.

```java
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

// Create a container for the result and specify that this is a quantized model.
// Hence, the 'DataType' is defined as UINT8 (8-bit unsigned integer)
TensorBuffer probabilityBuffer =
    TensorBuffer.createFixedSize(new int[]{1, 1001}, DataType.UINT8);
```

모델 로드 및 추론 실행하기:

```java
import org.tensorflow.lite.support.model.Model;

// Initialise the model
try{
    MappedByteBuffer tfliteModel
        = FileUtil.loadMappedFile(activity,
            "mobilenet_v1_1.0_224_quant.tflite");
    Interpreter tflite = new Interpreter(tfliteModel)
} catch (IOException e){
    Log.e("tfliteSupport", "Error reading model", e);
}

// Running inference
if(null != tflite) {
    tflite.run(tImage.getBuffer(), probabilityBuffer.getBuffer());
}
```

### 결과에 액세스하기

개발자는 `probabilityBuffer.getFloatArray()`를 통해 직접 출력에 액세스할 수 있습니다. 모델이 양자화된 출력을 생성하는 경우, 결과를 변환해야 합니다. MobileNet 양자화 모델의 경우, 개발자는 각 범주에 대해 0(최소 가능성)에서 1(최대 가능성)까지의 확률 범위를 얻기 위해 각 출력 값을 255로 나누어야 합니다.

### 선택 사항: 결과를 레이블에 매핑하기

개발자는 필요에 따라 결과를 레이블에 매핑할 수도 있습니다. 먼저, 레이블이 포함된 텍스트 파일을 모듈의 assets 디렉토리에 복사합니다. 다음으로, 아래 코드를 사용하여 레이블 파일을 로드합니다.

```java
import org.tensorflow.lite.support.common.FileUtil;

final String ASSOCIATED_AXIS_LABELS = "labels.txt";
List associatedAxisLabels = null;

try {
    associatedAxisLabels = FileUtil.loadLabels(this, ASSOCIATED_AXIS_LABELS);
} catch (IOException e) {
    Log.e("tfliteSupport", "Error reading label file", e);
}
```

다음 조각은 확률을 범주 레이블과 연결하는 방법을 보여줍니다.

```java
import org.tensorflow.lite.support.common.TensorProcessor;
import org.tensorflow.lite.support.label.TensorLabel;

// Post-processor which dequantize the result
TensorProcessor probabilityProcessor =
    new TensorProcessor.Builder().add(new NormalizeOp(0, 255)).build();

if (null != associatedAxisLabels) {
    // Map of labels and their corresponding probability
    TensorLabel labels = new TensorLabel(associatedAxisLabels,
        probabilityProcessor.process(probabilityBuffer));

    // Create a map to access the result based on label
    Map floatMap = labels.getMapWithFloatValue();
}
```

## 현재 사용 사례 범위

현재 버전의 TensorFlow Lite Support Library는 다음을 포함합니다.

- tflite 모델의 입력 및 출력으로 이용되는 일반적인 데이터 형식(부동 소수점, uint8, 이미지 및 이러한 객체의 배열)
- 기본 이미지 작업(이미지 자르기, 크기 조정 및 회전)
- 정규화 및 양자화
- 파일 유틸리티

향후 버전에서는 텍스트 관련 애플리케이션에 대한 지원을 개선할 예정입니다.

## ImageProcessor 아키텍처

`ImageProcessor`는 이미지 조작 연산을 미리 정의하고 빌드 프로세스 중에 최적화할 수 있게 설계되었습니다. `ImageProcessor`는 현재 세 가지 기본 전처리 연산을 지원합니다.

```java
int width = bitmap.getWidth();
int height = bitmap.getHeight();

int size = height > width ? width : height;

ImageProcessor imageProcessor =
    new ImageProcessor.Builder()
        // Center crop the image to the largest square possible
        .add(new ResizeWithCropOrPadOp(size, size))
        // Resize using Bilinear or Nearest neighbour
        .add(new ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR));
        // Rotation counter-clockwise in 90 degree increments
        .add(new Rot90Op(rotateDegrees / 90))
        .add(new NormalizeOp(127.5, 127.5))
        .add(new QuantizeOp(128.0, 1/128.0))
        .build();
```

[여기](../convert/metadata.md#normalization-and-quantization-parameters)에서 정규화 및 양자화에 대한 자세한 내용을 참조하세요.

지원 라이브러리의 최종 목표는 모든 [`tf.image`](https://www.tensorflow.org/api_docs/python/tf/image) 변환을 지원하는 것입니다. 이는 변환이 TensorFlow와 동일하고 구현이 운영 체제와 독립적임을 의미합니다.

개발자는 또한 사용자 정의 프로세서를 만들 수도 있습니다. 이러한 경우, 훈련 프로세스와 일치시키는 것이 중요합니다. 즉, 재현성을 높이기 위해 훈련과 추론 모두에 동일한 전처리를 적용해야 합니다.

## 양자화

`TensorImage` 또는 `TensorBuffer`와 같은 입력 또는 출력 객체를 시작할 때, 해당 유형을 `DataType.UINT8` 또는 `DataType.FLOAT32`로 지정해야 합니다.

```java
TensorImage tImage = new TensorImage(DataType.UINT8);
TensorBuffer probabilityBuffer =
    TensorBuffer.createFixedSize(new int[]{1, 1001}, DataType.UINT8);
```

`TensorProcessor`는 입력 텐서를 양자화하거나 출력 텐서를 역양자화하는 데 사용할 수 있습니다. 예를 들어, 양자화된 출력 `TensorBuffer`를 처리할 때 개발자는 `DequantizeOp`를 사용하여 결과를 0과 1 사이의 부동 소수점 확률로 역양자화할 수 있습니다.

```java
import org.tensorflow.lite.support.common.TensorProcessor;

// Post-processor which dequantize the result
TensorProcessor probabilityProcessor =
    new TensorProcessor.Builder().add(new DequantizeOp(0, 1/255.0)).build();
TensorBuffer dequantizedBuffer = probabilityProcessor.process(probabilityBuffer);
```

텐서의 양자화 매개변수는 [메타데이터 추출기 라이브러리](../convert/metadata.md#read-the-metadata-from-models)를 통해 읽을 수 있습니다.
