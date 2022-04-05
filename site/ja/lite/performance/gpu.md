# TensorFlow Lite GPU デリゲート

[TensorFlow Lite](https://www.tensorflow.org/lite) では、複数のハードウェアアクセラレータがサポートされています。このドキュメントでは、Android および iOS で TensorFlow Lite デリゲート API を使用して GPU バックエンドを使用する方法について説明します。

GPU は、大規模に実行する並列化可能なワークロードで高い処理能力を実現するように設計されています。そのため、これは多数の演算で構成されるディープニューラルネットに適しています。各演算は、より小さなワークロードに簡単に分割でき、並列に実行する入力テンソルで機能するため、通常レイテンシが低くなります。現在、最良のシナリオでは、GPU での推論は以前は利用できなかったリアルタイムアプリケーションで十分に速く実行できます。

CPU とは異なり、GPU は 16 ビットまたは 32 ビットの浮動小数点数で計算し、最適なパフォーマンスを得るために量子化を必要としません。デリゲートは 8 ビットの量子化モデルを受け入れますが、計算は浮動小数点数で実行されます。詳細については、[高度なドキュメント](gpu_advanced.md)をご覧ください。

GPU の推論のもう 1 つの利点は、電力効率です。GPU は非常に効率的かつ最適化された方法で計算を実行するため、同じタスクを CPU で実行する場合よりも消費電力と発熱が少なくなります。

## デモアプリのチュートリアル

以下のチュートリアルに従って GPU をサポートする分類デモアプリケーションを構築し、GPU デリゲートを簡単に試すことができます。現時点では、GPU コードはバイナリのみですが、近日中にオープンソースになります。デモが機能することを確認したら、独自のカスタムモデルでお試しください。

### Android（Android Studio を使用）

ステップバイステップのチュートリアルは、[Android 向け GPU デリゲート](https://youtu.be/Xkhgre8r5G0)の動画をご覧ください。

注意: これには、OpenCL または OpenGL ES （3.1 以降）が必要です。

#### ステップ 1. TensorFlow ソースコードを複製して Android Studio で開く

```sh
git clone https://github.com/tensorflow/tensorflow
```

#### ステップ 2. `app/build.gradle`を編集して、ナイトリーの GPU AAR を使用する

Note: You can now target **Android S+** with `targetSdkVersion="S"` in your manifest, or `targetSdkVersion "S"` in your Gradle `defaultConfig` (API level TBD). In this case, you should merge the contents of [`AndroidManifestGpu.xml`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/java/AndroidManifestGpu.xml) into your Android application's manifest. Without this change, the GPU delegate cannot access OpenCL libraries for acceleration. *AGP 4.2.0 or above is required for this to work.*

Add the `tensorflow-lite-gpu` package alongside the existing `tensorflow-lite` package in the existing `dependencies` block.

```
dependencies {
    ...
    implementation 'org.tensorflow:tensorflow-lite:2.3.0'
    implementation 'org.tensorflow:tensorflow-lite-gpu:2.3.0'
}
```

#### ステップ 3. 構築して実行する

Run → Run ‘app’ (実行 → 「アプリ」を実行)。アプリを実行すると、GPU を有効にするためのボタンが表示されます。量子化モデルから浮動小数点数モデルに変更し、GPU をクリックして GPU で実行します。

![Android GPU デモを実行し、GPU に切り替えます](images/android_gpu_demo.gif)

### iOS（XCode を使用）

For a step-by-step tutorial, watch the [GPU Delegate for iOS](https://youtu.be/a5H4Zwjp49c) video.

注意: これには XCode v10.1 以降が必要です。

#### ステップ 1. デモのソースコードを入手して、コンパイルできることを確認する。

Follow our iOS Demo App [tutorial](https://www.tensorflow.org/lite/guide/ios). This will get you to a point where the unmodified iOS camera demo is working on your phone.

#### ステップ 2. TensorFlow Lite GPU CocoaPod を使用するように Podfile を変更する

2.3.0 リリース以降では、バイナリサイズを減らすために、デフォルトで GPU デリゲートがポッドから除外されていますが、サブスペックを指定してそれらを含めることができます。`TensorFlowLiteSwift`ポッドの場合は、次のようになります。

```ruby
pod 'TensorFlowLiteSwift/Metal', '~> 0.0.1-nightly',
```

または

```ruby
pod 'TensorFlowLiteSwift', '~> 0.0.1-nightly', :subspecs => ['Metal']
```

Objective-C（2.4.0 リリース以降）または C API を使用する場合は、`TensorFlowLiteObjC`または`TensorFlowLitC`に対して同様に行うことができます。

<div>
  <devsite-expandable>
    <h4 class="showalways">2.3.0 リリースより前のバージョン</h4>
    <h4>TensorFlow Lite 2.0.0 以前</h4>
    <p>GPU デリゲートを含むバイナリ CocoaPod を構築しました。これを使用するようにプロジェクトを切り替えるには、tensorflow/tensorflow/lite/examples/ios/camera/Podfile ファイルを変更して、TensorFlowLite の代わりに TensorFlowLiteGpuExperimental ポッドを使用します。</p>
    <pre class="prettyprint lang-ruby notranslate" translate="no"><code>
    target 'YourProjectName'
      # pod 'TensorFlowLite', '1.12.0'
      pod 'TensorFlowLiteGpuExperimental'
    </code></pre>
    <h4>TensorFlow Lite 2.2.0 以前</h4>
    <p>TensorFlow Lite 2.1.0 から 2.2.0 まででは、GPU デリゲートはTensorFlowLiteC ポッドに含まれています。言語に応じて、TensorFlowLiteC または TensorFlowLiteSwift を選択できます。</p>
  </devsite-expandable>
</div>

#### ステップ 3. GPU デリゲートを有効化する

GPU デリゲートを使用するコードを有効にするには、`CameraExampleViewController.h`で`TFLITE_USE_GPU_DELEGATE`を 0 から 1 に変更する必要があります。

```c
#define TFLITE_USE_GPU_DELEGATE 1
```

#### ステップ 4. デモアプリを構築して実行する

前述の手順に従うと、アプリを実行できるようになります。

#### ステップ 5. リリースモード

While in Step 4 you ran in debug mode, to get better performance, you should change to a release build with the appropriate optimal Metal settings. In particular, To edit these settings go to the `Product > Scheme > Edit Scheme...`. Select `Run`. On the `Info` tab, change `Build Configuration`, from `Debug` to `Release`, uncheck `Debug executable`.

![メタルオプションの設定](https://github.com/tensorflow/docs-l10n/blob/master/site/ja/lite/performance/images/iosmetal.png?raw=true)

次に、`Options`タブをクリックし、`GPU Frame Capture`を`Disabled`に、`Metal API Validation`をc`Disabled`に変更します。

![リリースの設定](images/iosdebug.png)

最後に、必ず 64 ビットアーキテクチャでリリースのみのビルドを選択してください。`Project navigator -> tflite_camera_example -> PROJECT -> tflite_camera_example -> Build Settings`で、`Build Active Architecture Only> Release`を「Yes」に設定します。

![リリースオプションの設定](images/iosrelease.png)

## 独自のモデルで GPU デリゲートを試す

### Android

注意: TensorFlow Lite インタープリタは、実行時と同じスレッドで作成する必要があります。そうでないと、「`TfLiteGpuDelegate Invoke: GpuDelegate must run on the same thread where it was initialized.`」が表示される可能性があります。

モデルアクセラレーションを呼び出す方法は 2 つありますが、[Android Studio ML Model Binding](../inference_with_metadata/codegen#acceleration) または TensorFlow Lite インタープリタを使用しているかによって、方法は異なります。

#### TensorFlow Lite インタープリタ

デモを見て、デリゲートを追加する方法を確認してください。アプリで、上記のように AAR を追加し、`org.tensorflow.lite.gpu.GpuDelegate`モジュールをインポートし、`addDelegate`関数を使用して GPU デリゲートをインタープリタに登録します。

<div>
  <devsite-selector>
    <section>
      <h3>Kotlin</h3>
      <p></p>
<pre class="prettyprint lang-kotlin">    import org.tensorflow.lite.Interpreter
    import org.tensorflow.lite.gpu.CompatibilityList
    import org.tensorflow.lite.gpu.GpuDelegate

    val compatList = CompatibilityList()

    val options = Interpreter.Options().apply{
        if(compatList.isDelegateSupportedOnThisDevice){
            // if the device has a supported GPU, add the GPU delegate
            val delegateOptions = compatList.bestOptionsForThisDevice
            this.addDelegate(GpuDelegate(delegateOptions))
        } else {
            // if the GPU is not supported, run on 4 threads
            this.setNumThreads(4)
        }
    }

    val interpreter = Interpreter(model, options)

    // Run inference
    writeToInput(input)
    interpreter.run(input, output)
    readFromOutput(output)
      </pre>
    </section>
    <section>
      <h3>Java</h3>
      <p></p>
<pre class="prettyprint lang-java">    import org.tensorflow.lite.Interpreter;
    import org.tensorflow.lite.gpu.CompatibilityList;
    import org.tensorflow.lite.gpu.GpuDelegate;

    // Initialize interpreter with GPU delegate
    Interpreter.Options options = new Interpreter.Options();
    CompatibilityList compatList = CompatibilityList();

    if(compatList.isDelegateSupportedOnThisDevice()){
        // if the device has a supported GPU, add the GPU delegate
        GpuDelegate.Options delegateOptions = compatList.getBestOptionsForThisDevice();
        GpuDelegate gpuDelegate = new GpuDelegate(delegateOptions);
        options.addDelegate(gpuDelegate);
    } else {
        // if the GPU is not supported, run on 4 threads
        options.setNumThreads(4);
    }

    Interpreter interpreter = new Interpreter(model, options);

    // Run inference
    writeToInput(input);
    interpreter.run(input, output);
    readFromOutput(output);
      </pre>
    </section>
  </devsite-selector>
</div>

### iOS

注意：GPU デリゲートは、Objective-C コードで CAPI を使用することもできます。TensorFlow Lite 2.4.0 リリース以前は、これが唯一のオプションでした。

<div>
  <devsite-selector>
    <section>
      <h3>Swift</h3>
      <p></p>
<pre class="prettyprint lang-swift">    import TensorFlowLite

    // Load model ...

    // Initialize TensorFlow Lite interpreter with the GPU delegate.
    let delegate = MetalDelegate()
    if let interpreter = try Interpreter(modelPath: modelPath,
                                         delegates: [delegate]) {
      // Run inference ...
    }
      </pre>
    </section>
    <section>
      <h3>Objective-C</h3>
      <p></p>
<pre class="prettyprint lang-objc">    // Import module when using CocoaPods with module support
    @import TFLTensorFlowLite;

    // Or import following headers manually
    #import "tensorflow/lite/objc/apis/TFLMetalDelegate.h"
    #import "tensorflow/lite/objc/apis/TFLTensorFlowLite.h"

    // Initialize GPU delegate
    TFLMetalDelegate* metalDelegate = [[TFLMetalDelegate alloc] init];

    // Initialize interpreter with model path and GPU delegate
    TFLInterpreterOptions* options = [[TFLInterpreterOptions alloc] init];
    NSError* error = nil;
    TFLInterpreter* interpreter = [[TFLInterpreter alloc]
                                    initWithModelPath:modelPath
                                              options:options
                                            delegates:@[ metalDelegate ]
                                                error:&amp;error];
    if (error != nil) { /* Error handling... */ }

    if (![interpreter allocateTensorsWithError:&amp;error]) { /* Error handling... */ }
    if (error != nil) { /* Error handling... */ }

    // Run inference ...
    ```
      </pre>
    </section>
    <section>
      <h3>C (Until 2.3.0)</h3>
      <p></p>
<pre class="prettyprint lang-c">    #include "tensorflow/lite/c/c_api.h"
    #include "tensorflow/lite/delegates/gpu/metal_delegate.h"

    // Initialize model
    TfLiteModel* model = TfLiteModelCreateFromFile(model_path);

    // Initialize interpreter with GPU delegate
    TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
    TfLiteDelegate* delegate = TFLGPUDelegateCreate(nil);  // default config
    TfLiteInterpreterOptionsAddDelegate(options, metal_delegate);
    TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, options);
    TfLiteInterpreterOptionsDelete(options);

    TfLiteInterpreterAllocateTensors(interpreter);

    NSMutableData *input_data = [NSMutableData dataWithLength:input_size * sizeof(float)];
    NSMutableData *output_data = [NSMutableData dataWithLength:output_size * sizeof(float)];
    TfLiteTensor* input = TfLiteInterpreterGetInputTensor(interpreter, 0);
    const TfLiteTensor* output = TfLiteInterpreterGetOutputTensor(interpreter, 0);

    // Run inference
    TfLiteTensorCopyFromBuffer(input, inputData.bytes, inputData.length);
    TfLiteInterpreterInvoke(interpreter);
    TfLiteTensorCopyToBuffer(output, outputData.mutableBytes, outputData.length);

    // Clean up
    TfLiteInterpreterDelete(interpreter);
    TFLGpuDelegateDelete(metal_delegate);
    TfLiteModelDelete(model);
      </pre>
    </section>
  </devsite-selector>
</div>

## サポートされているモデルと演算

GPU デリゲートのリリースには、バックエンドで実行できるいくつかのモデルが含まれています。

- [MobileNet v1 (224x224) image classification](https://ai.googleblog.com/2017/06/mobilenets-open-source-models-for.html) [[download]](https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/mobilenet_v1_1.0_224.tflite)<br><i>(image classification model designed for mobile and embedded based vision applications)</i>
- [DeepLab segmentation (257x257)](https://ai.googleblog.com/2018/03/semantic-image-segmentation-with.html) [[download]](https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/deeplabv3_257_mv_gpu.tflite)<br><i>(image segmentation model that assigns semantic labels (e.g., dog, cat, car) to every pixel in the input image)</i>
- [MobileNet SSD object detection](https://ai.googleblog.com/2018/07/accelerated-training-and-inference-with.html) [[download]](https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/mobile_ssd_v2_float_coco.tflite)<br><i>(image classification model that detects multiple objects with bounding boxes)</i>
- [PoseNet for pose estimation](https://github.com/tensorflow/tfjs-models/tree/master/posenet) [[download]](https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/multi_person_mobilenet_v1_075_float.tflite)<br><i>(vision model that estimates the poses of a person(s) in image or video)</i>

サポートされている演算の完全なリストは、[アドバンストドキュメント](gpu_advanced.md)を参照してください。

## サポートされていないモデルと演算

一部の演算が GPU デリゲートでサポートされていない場合、フレームワークは GPU でグラフの一部のみを実行し、残りの部分を CPU で実行します。CPU と GPU 同期のコストは高いため、このような分割実行モードでは、ネットワーク全体が CPU のみで実行されている場合よりもパフォーマンスが遅くなることがよくあります。この場合、ユーザーには次のような警告が表示されます。

```none
WARNING: op code #42 cannot be handled by this delegate.
```

これは実行時エラーではないため、このエラーに対するコールバックは提供されませんが、開発者はデリゲートでネットワークを実行する際に確認できます。

## 最適化のヒント

### Optimizing for mobile devices

Some operations that are trivial on the CPU may have a high cost for the GPU on mobile devices. Reshape operations are particularly expensive to run, including `BATCH_TO_SPACE`, `SPACE_TO_BATCH`, `SPACE_TO_DEPTH`, and so forth. You should closely examine use of reshape operations, and consider that may have been applied only for exploring data or for early iterations of your model. Removing them can significantly improve performance.

On GPU, tensor data is sliced into 4-channels. Thus, a computation on a tensor of shape `[B,H,W,5]` will perform about the same on a tensor of shape `[B,H,W,8]` but significantly worse than `[B,H,W,4]`. In that sense, if the camera hardware supports image frames in RGBA, feeding that 4-channel input is significantly faster as a memory copy (from 3-channel RGB to 4-channel RGBX) can be avoided.

For best performance, you should consider retraining the classifier with a mobile-optimized network architecture. Optimization for on-device inferencing can dramatically reduce latency and power consumption by taking advantage of mobile hardware features.

### Reducing initialization time with serialization

The GPU delegate feature allows you to load from pre-compiled kernel code and model data serialized and saved on disk from previous runs. This approach avoids re-compilation and reduces startup time by up to 90%. For instructions on how to apply serialization to your project, see [GPU Delegate Serialization](gpu_advanced.md#gpu_delegate_serialization).
