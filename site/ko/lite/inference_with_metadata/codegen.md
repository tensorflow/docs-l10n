# Generate model interfaces using metadata

Using [TensorFlow Lite Metadata](../convert/metadata), developers can generate wrapper code to enable integration on Android. For most developers, the graphical interface of [Android Studio ML Model Binding](#mlbinding) is the easiest to use. If you require more customisation or are using command line tooling, the [TensorFlow Lite Codegen](#codegen) is also available.

## Use Android Studio ML Model Binding {:#mlbinding}

[메타데이터](../convert/metadata.md)로 강화된 TensorFlow Lite 모델의 경우 개발자는 Android Studio ML Model Binding을 사용하여 프로젝트를 위한 설정을 자동으로 구성하고 모델 메타데이터에 기초한 래퍼 클래스를 생성할 수 있습니다. 래퍼 코드를 사용하면 `ByteBuffer`와 직접 상호 작용할 필요가 없습니다. 대신, 개발자는 `Bitmap` 및 `Rect`와 같은 형식화된 객체를 통해 TensorFlow Lite 모델과 상호 작용할 수 있습니다.

Note: Required [Android Studio 4.1](https://developer.android.com/studio) or above

### Import a TensorFlow Lite model in Android Studio

1. Right-click on the module you would like to use the TFLite model or click on `File`, then `New` &gt; `Other` &gt; `TensorFlow Lite Model` ![Right-click menus to access the TensorFlow Lite import functionality](../images/android/right_click_menu.png)

2. Select the location of your TFLite file. Note that the tooling will configure the module's dependency on your behalf with ML Model binding and all dependencies automatically inserted into your Android module's `build.gradle` file.

    Optional: Select the second checkbox for importing TensorFlow GPU if you want to use GPU acceleration. ![Import dialog for TFLite model](../images/android/import_dialog.png)

3. Click `Finish`.

4. The following screen will appear after the import is successful. To start using the model, select Kotlin or Java, copy and paste the code under the `Sample Code` section. You can get back to this screen by double clicking the TFLite model under the `ml` directory in Android Studio. ![Model details page in Android Studio](../images/android/model_details.png)

### Accelerating model inference {:#acceleration}

ML Model Binding은 개발자가 대리자 및 스레드 수를 사용하여 코드를 가속할 수 있는 방법을 제공합니다.

참고: TensorFlow Lite 인터프리터는 실행될 때와 동일한 스레드에서 생성되어야 합니다. 그렇지 않으면, "TfLiteGpuDelegate Invoke: GpuDelegate must run on the same thread where it was initialized."가 발생할 수 있습니다.

Step 1. Check the module `build.gradle` file that it contains the following dependency:

```java
    dependencies {
        ...
        // TFLite GPU delegate 2.3.0 or above is required.
        implementation 'org.tensorflow:tensorflow-lite-gpu:2.3.0'
    }
```

Step 2. Detect if GPU running on the device is compatible with TensorFlow GPU delegate, if not run the model using multiple CPU threads:

<div>
    <devsite-selector>
    <section>
      <h3>Kotlin</h3>
      <p></p>
<pre class="prettyprint lang-kotlin">    import org.tensorflow.lite.gpu.CompatibilityList
    import org.tensorflow.lite.gpu.GpuDelegate

    val compatList = CompatibilityList()

    val options = if(compatList.isDelegateSupportedOnThisDevice) {
        // if the device has a supported GPU, add the GPU delegate
        Model.Options.Builder().setDevice(Model.Device.GPU).build()
    } else {
        // if the GPU is not supported, run on 4 threads
        Model.Options.Builder().setNumThreads(4).build()
    }

    // Initialize the model as usual feeding in the options object
    val myModel = MyModel.newInstance(context, options)

    // Run inference per sample code
      </pre>
    </section>
    <section>
      <h3>Java</h3>
      <p></p>
<pre class="prettyprint lang-java">    import org.tensorflow.lite.support.model.Model
    import org.tensorflow.lite.gpu.CompatibilityList;
    import org.tensorflow.lite.gpu.GpuDelegate;

    // Initialize interpreter with GPU delegate
    Model.Options options;
    CompatibilityList compatList = CompatibilityList();

    if(compatList.isDelegateSupportedOnThisDevice()){
        // if the device has a supported GPU, add the GPU delegate
        options = Model.Options.Builder().setDevice(Model.Device.GPU).build();
    } else {
        // if the GPU is not supported, run on 4 threads
        options = Model.Options.Builder().setNumThreads(4).build();
    }

    MyModel myModel = new MyModel.newInstance(context, options);

    // Run inference per sample code
      </pre>
    </section>
    </devsite-selector>
</div>

## Generate model interfaces with TensorFlow Lite code generator {:#codegen}

참고: TensorFlow Lite 래퍼 코드 생성기는 현재 Android만 지원합니다.

For TensorFlow Lite model enhanced with [metadata](../convert/metadata.md), developers can use the TensorFlow Lite Android wrapper code generator to create platform specific wrapper code. The wrapper code removes the need to interact directly with `ByteBuffer`. Instead, developers can interact with the TensorFlow Lite model with typed objects such as `Bitmap` and `Rect`.

The usefulness of the code generator depend on the completeness of the TensorFlow Lite model's metadata entry. Refer to the `<Codegen usage>` section under relevant fields in [metadata_schema.fbs](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/metadata/metadata_schema.fbs), to see how the codegen tool parses each field.

### 래퍼 코드 생성하기

단말기에 다음 도구를 설치해야 합니다.

```sh
pip install tflite-support
```

Once completed, the code generator can be used using the following syntax:

```sh
tflite_codegen --model=./model_with_metadata/mobilenet_v1_0.75_160_quantized.tflite \
    --package_name=org.tensorflow.lite.classify \
    --model_class_name=MyClassifierModel \
    --destination=./classify_wrapper
```

The resulting code will be located in the destination directory. If you are using [Google Colab](https://colab.research.google.com/) or other remote environment, it maybe easier to zip up the result in a zip archive and download it to your Android Studio project:

```python
# Zip up the generated code
!zip -r classify_wrapper.zip classify_wrapper/

# Download the archive
from google.colab import files
files.download('classify_wrapper.zip')
```

### 생성된 코드 사용하기

#### 1단계: 생성된 코드 가져오기

필요한 경우 생성된 코드를 디렉토리 구조에 압축 해제합니다. 생성된 코드의 루트는 `SRC_ROOT`로 간주합니다.

TensorFlow lite 모델을 사용하려는 Android Studio 프로젝트를 열고 생성된 모듈을 가져옵니다(File-&gt; New-&gt; Import Module-&gt; `SRC_ROOT` 선택).

위의 예에서 가져온 디렉토리와 모듈은 `classify_wrapper`라고 명명됩니다.

#### 2단계: 앱의 `build.gradle` 파일 업데이트하기

생성된 라이브러리 모듈을 사용할 앱 모듈에서 다음을 수행합니다.

Under the android section, add the following:

```build
aaptOptions {
   noCompress "tflite"
}
```

참고: Android Gradle 플러그인 버전 4.1부터는 .tflite가 기본적으로 noCompress 목록에 추가되며 위의 aaptOptions는 더 이상 필요하지 않습니다.

종속성 섹션에 다음을 추가합니다.

```build
implementation project(":classify_wrapper")
```

#### 3단계: 모델 사용하기

```java
// 1. Initialize the model
MyClassifierModel myImageClassifier = null;

try {
    myImageClassifier = new MyClassifierModel(this);
} catch (IOException io){
    // Error reading the model
}

if(null != myImageClassifier) {

    // 2. Set the input with a Bitmap called inputBitmap
    MyClassifierModel.Inputs inputs = myImageClassifier.createInputs();
    inputs.loadImage(inputBitmap));

    // 3. Run the model
    MyClassifierModel.Outputs outputs = myImageClassifier.run(inputs);

    // 4. Retrieve the result
    Map<String, Float> labeledProbability = outputs.getProbability();
}
```

### 모델 추론 가속하기

The generated code provides a way for developers to accelerate their code through the use of [delegates](../performance/delegates.md) and the number of threads. These can be set when initiatizing the model object as it takes three parameters:

- **`Context`**: Android 활동 또는 서비스의 컨텍스트
- (선택 사항) **`Device`**: GPUDelegate 또는 NNAPIDelegate와 같은 TFLite 가속 대리자
- (Optional) **`numThreads`**: Number of threads used to run the model - default is one.

예를 들어, NNAPI 대리자와 최대 3개의 스레드를 사용하려면 다음과 같이 모델을 초기화할 수 있습니다.

```java
try {
    myImageClassifier = new MyClassifierModel(this, Model.Device.NNAPI, 3);
} catch (IOException io){
    // Error reading the model
}
```

### Troubleshooting

 'java.io.FileNotFoundException: This file can not be opened as a file descriptor; it is probably compressed(이 파일을 파일 설명자로 열 수 없습니다. 압축되었을 수 있습니다.)' 오류가 발생하면 라이브러리 모듈을 사용할 앱 모듈의 android 섹션 아래에 다음 줄을 삽입합니다.

```build
aaptOptions {
   noCompress "tflite"
}
```

참고: Android Gradle 플러그인 버전 4.1부터는 .tflite가 기본적으로 noCompress 목록에 추가되며 위의 aaptOptions는 더 이상 필요하지 않습니다.
