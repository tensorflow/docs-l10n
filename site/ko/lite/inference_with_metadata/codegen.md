# Generate model interfaces using metadata

Using [TensorFlow Lite Metadata](../convert/metadata), developers can generate wrapper code to enable integration on Android. For most developers, the graphical interface of [Android Studio ML Model Binding](#mlbinding) is the easiest to use. If you require more customisation or are using command line tooling, the [TensorFlow Lite Codegen](#codegen) is also available.

## Use Android Studio ML Model Binding {:#mlbinding}

For TensorFlow Lite models enhanced with [metadata](../convert/metadata.md), developers can use Android Studio ML Model Binding to automatically configure settings for the project and generate wrapper classes based on the model metadata. The wrapper code removes the need to interact directly with `ByteBuffer`. Instead, developers can interact with the TensorFlow Lite model with typed objects such as `Bitmap` and `Rect`.

Note: Required [Android Studio 4.1](https://developer.android.com/studio) or above

### Import a TensorFlow Lite model in Android Studio

1. Right-click on the module you would like to use the TFLite model or click on `File`, then `New` &gt; `Other` &gt; `TensorFlow Lite Model` ![Right-click menus to access the TensorFlow Lite import functionality](../images/android/right_click_menu.png)

2. Select the location of your TFLite file. Note that the tooling will configure the module's dependency on your behalf with ML Model binding and all dependencies automatically inserted into your Android module's `build.gradle` file.

    Optional: Select the second checkbox for importing TensorFlow GPU if you want to use GPU acceleration. ![Import dialog for TFLite model](../images/android/import_dialog.png)

3. Click `Finish`.

4. The following screen will appear after the import is successful. To start using the model, select Kotlin or Java, copy and paste the code under the `Sample Code` section. You can get back to this screen by double clicking the TFLite model under the `ml` directory in Android Studio. ![Model details page in Android Studio](../images/android/model_details.png)

### Accelerating model inference {:#acceleration}

ML Model Binding provides a way for developers to accelerate their code through the use of delegates and the number of threads.

Note: The TensorFlow Lite Interpreter must be created on the same thread as when is is run. Otherwise, TfLiteGpuDelegate Invoke: GpuDelegate must run on the same thread where it was initialized. may occur.

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
&lt;/div&gt;
&lt;pre data-md-type="block_code" data-md-language=""&gt;&lt;code&gt;GL_CODE_13&lt;/code&gt;</pre>
<div data-md-type="block_html"></div>
</section></devsite-selector>
</div>
<h2 data-md-type="header" data-md-header-level="2">Generate model interfaces with TensorFlow Lite code generator {:#codegen}</h2>
<p data-md-type="paragraph">참고: TensorFlow Lite 래퍼 코드 생성기는 현재 Android만 지원합니다.</p>
<p data-md-type="paragraph"><a href="../convert/metadata.md" data-md-type="link">메타데이터</a>로 강화된 TensorFlow Lite 모델의 경우, 개발자는 TensorFlow Lite Android 래퍼 코드 생성기를 사용하여 플랫폼별 래퍼 코드를 만들 수 있습니다. 래퍼 코드는 <code data-md-type="codespan">ByteBuffer</code>와 직접 상호 작용할 필요성을 없애줍니다. 대신, 개발자는 <code data-md-type="codespan">Bitmap</code> 및 <code data-md-type="codespan">Rect</code>와 같은 형식화된 객체를 사용하여 TensorFlow Lite 모델과 상호 작용할 수 있습니다.</p>
<p data-md-type="paragraph">코드 생성기의 유용성은 TensorFlow Lite 모델의 메타데이터 항목이 얼마나 완전한지에 달려 있습니다. <code data-md-type="codespan">&lt;Codegen usage&gt;</code> 섹션에서 <a href="https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/metadata/metadata_schema.fbs" data-md-type="link">metadata_schema.fbs</a>의 관련 필드를 참조하여 codegen 도구가 이러한 필드를 어떻게 구문 분석하는지 확인하세요.</p>
<h3 data-md-type="header" data-md-header-level="3">래퍼 코드 생성하기</h3>
<p data-md-type="paragraph">단말기에 다음 도구를 설치해야 합니다.</p>
<pre data-md-type="block_code" data-md-language="sh"><code class="language-sh">pip install tflite-support
</code></pre>
<p data-md-type="paragraph">완료되면 다음 구문을 사용하여 코드 생성기를 사용할 수 있습니다.</p>
<pre data-md-type="block_code" data-md-language="sh"><code class="language-sh">tflite_codegen --model=./model_with_metadata/mobilenet_v1_0.75_160_quantized.tflite \
    --package_name=org.tensorflow.lite.classify \
    --model_class_name=MyClassifierModel \
    --destination=./classify_wrapper</code></pre>
<p data-md-type="paragraph">결과 코드는 대상 디렉토리에 있습니다. <a href="https://colab.research.google.com/" data-md-type="link">Google Colab</a> 또는 기타 원격 환경을 사용하는 경우, 결과를 zip 아카이브로 압축하여 Android Studio 프로젝트에 다운로드하는 것이 더 쉬울 수 있습니다.</p>
<pre data-md-type="block_code" data-md-language="python"><code class="language-python"># Zip up the generated code
!zip -r classify_wrapper.zip classify_wrapper/

# Download the archive
from google.colab import files
files.download('classify_wrapper.zip')</code></pre>
<h3 data-md-type="header" data-md-header-level="3">생성된 코드 사용하기</h3>
<h4 data-md-type="header" data-md-header-level="4">1단계: 생성된 코드 가져오기</h4>
<p data-md-type="paragraph">필요한 경우 생성된 코드를 디렉토리 구조에 압축 해제합니다. 생성된 코드의 루트는 <code data-md-type="codespan">SRC_ROOT</code>로 간주합니다.</p>
<p data-md-type="paragraph">TensorFlow lite 모델을 사용하려는 Android Studio 프로젝트를 열고 생성된 모듈을 가져옵니다(File-&gt; New-&gt; Import Module-&gt; <code data-md-type="codespan">SRC_ROOT</code> 선택).</p>
<p data-md-type="paragraph">위의 예에서 가져온 디렉토리와 모듈은 <code data-md-type="codespan">classify_wrapper</code>라고 명명됩니다.</p>
<h4 data-md-type="header" data-md-header-level="4">2단계: 앱의 <code data-md-type="codespan">build.gradle</code> 파일 업데이트하기</h4>
<p data-md-type="paragraph">생성된 라이브러리 모듈을 사용할 앱 모듈에서,</p>
<p data-md-type="paragraph">android 섹션 아래에 다음을 추가합니다.</p>
<pre data-md-type="block_code" data-md-language="build"><code class="language-build">aaptOptions {
   noCompress "tflite"
}</code></pre>
<p data-md-type="paragraph">종속성 섹션에 다음을 추가합니다.</p>
<pre data-md-type="block_code" data-md-language="build"><code class="language-build">implementation project(":classify_wrapper")</code></pre>
<h4 data-md-type="header" data-md-header-level="4">3단계: 모델 사용하기</h4>
<pre data-md-type="block_code" data-md-language="java"><code class="language-java">// 1. Initialize the model
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
    Map labeledProbability = outputs.getProbability();
}</code></pre>
<h3 data-md-type="header" data-md-header-level="3">모델 추론 가속하기</h3>
<p data-md-type="paragraph">생성된 코드는 개발자가 <a href="../performance/delegates.md" data-md-type="link">대리자</a> 및 스레드 수를 사용하여 코드를 가속할 수 있는 방법을 제공합니다. 모델 객체를 초기화할 때 세 가지 매개변수를 사용하여 설정할 수 있습니다.</p>
<ul data-md-type="list" data-md-list-type="unordered" data-md-list-tight="true">
<li data-md-type="list_item" data-md-list-type="unordered">
<strong data-md-type="double_emphasis"><code data-md-type="codespan">Context</code></strong>: Android 활동 또는 서비스의 컨텍스트</li>
<li data-md-type="list_item" data-md-list-type="unordered">(선택 사항) <strong data-md-type="double_emphasis"><code data-md-type="codespan">Device</code></strong>: GPUDelegate 또는 NNAPIDelegate와 같은 TFLite 가속 대리자</li>
<li data-md-type="list_item" data-md-list-type="unordered">(선택 사항) <strong data-md-type="double_emphasis"><code data-md-type="codespan">numThreads</code></strong>: 모델을 실행하는 데 사용되는 스레드의 수 - 기본값 1</li>
</ul>
<p data-md-type="paragraph">예를 들어, NNAPI 대리자와 최대 3개의 스레드를 사용하려면 다음과 같이 모델을 초기화할 수 있습니다.</p>
<pre data-md-type="block_code" data-md-language="java"><code class="language-java">try {
    myImageClassifier = new MyClassifierModel(this, Model.Device.NNAPI, 3);
} catch (IOException io){
    // Error reading the model
}</code></pre>
<h3 data-md-type="header" data-md-header-level="3">문제 해결</h3>
<p data-md-type="paragraph"> 'java.io.FileNotFoundException: This file can not be opened as a file descriptor; it is probably compressed(이 파일을 파일 설명자로 열 수 없습니다. 압축되었을 수 있습니다.)' 오류가 발생하면 라이브러리 모듈을 사용할 앱 모듈의 android 섹션 아래에 다음 줄을 삽입합니다.</p>
<pre data-md-type="block_code" data-md-language="build"><code class="language-build">aaptOptions {
   noCompress "tflite"
}</code></pre>
