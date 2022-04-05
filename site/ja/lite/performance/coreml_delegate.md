# Tensorflow Lite Core ML デリゲート

TensorFlow Lite Core ML デリゲートは、[Core ML フレームワーク](https://developer.apple.com/documentation/coreml)上での TensorFlow Lite モデルの実行を可能にしました。その結果、iOS デバイス上におけるモデル推論の高速化します。

注意: このデリゲートは試験（ベータ）段階です。TensorFlow Lite 2.4.0 および最新のナイトリーリリースから利用できます。

注意: Core ML デリゲートは、Core ML のバージョン 2 以降をサポートしています。

**サポートする iOS のバージョンとデバイス:**

- iOS 12 and later. In the older iOS versions, Core ML delegate will automatically fallback to CPU.
- デフォルトでは、Core ML デリゲートは A12 SoC 以降のデバイス（iPhone Xs 以降）でのみ有効で、Neural Engine（ニューラルエンジン）を推論の高速化に使用します。古いデバイスで Core ML デリゲートを使用する場合は、[ベストプラクティス](#best-practices)を参照してください。

**サポートするモデル**

現在、Core ML のデリゲートは浮動小数点数（FP32 と FP16）モデルをサポートしています。

## 独自のモデルで Core ML デリゲートを試す

Core ML デリゲートは、TensorFlow lite CocoaPods のナイトリーリリースにすでに含まれています。Core ML デリゲートを使用するには、TensorFlow lite ポッドを変更して、`Podfile`にサブスペック`CoreML`を含めます。

注意：Objective-C API の代わりに C API を使用する場合は`TensorFlowLiteC/CoreML`ポッドを含めることができます。

```
target 'YourProjectName'
  pod 'TensorFlowLiteSwift/CoreML', '~> 2.4.0'  # Or TensorFlowLiteObjC/CoreML
```

または

```
# Particularily useful when you also want to include 'Metal' subspec.
target 'YourProjectName'
  pod 'TensorFlowLiteSwift', '~> 2.4.0', :subspecs => ['CoreML']
```

注意：Core ML デリゲートは、Objective-C コードで C API を使用することもできます。TensorFlow Lite 2.4.0 リリース以前は、これが唯一のオプションでした。

<div>
  <devsite-selector>
    <section>
      <h3>Swift</h3>
      <p></p>
<pre class="prettyprint lang-swift">    let coreMLDelegate = CoreMLDelegate()
    var interpreter: Interpreter

    // Core ML delegate will only be created for devices with Neural Engine
    if coreMLDelegate != nil {
      interpreter = try Interpreter(modelPath: modelPath,
                                    delegates: [coreMLDelegate!])
    } else {
      interpreter = try Interpreter(modelPath: modelPath)
    }
      </pre>
    </section>
    <section>
      <h3>Objective-C</h3>
      <p></p>
<pre class="prettyprint lang-objc">
    // Import module when using CocoaPods with module support
    @import TFLTensorFlowLite;

    // Or import following headers manually
    # import "tensorflow/lite/objc/apis/TFLCoreMLDelegate.h"
    # import "tensorflow/lite/objc/apis/TFLTensorFlowLite.h"

    // Initialize Core ML delegate
    TFLCoreMLDelegate* coreMLDelegate = [[TFLCoreMLDelegate alloc] init];

    // Initialize interpreter with model path and Core ML delegate
    TFLInterpreterOptions* options = [[TFLInterpreterOptions alloc] init];
    NSError* error = nil;
    TFLInterpreter* interpreter = [[TFLInterpreter alloc]
                                    initWithModelPath:modelPath
                                              options:options
                                            delegates:@[ coreMLDelegate ]
                                                error:&amp;error];
    if (error != nil) { /* Error handling... */ }

    if (![interpreter allocateTensorsWithError:&amp;error]) { /* Error handling... */ }
    if (error != nil) { /* Error handling... */ }

    // Run inference ...
      </pre>
    </section>
    <section>
      <h3>C (Until 2.3.0)</h3>
      <p></p>
<pre class="prettyprint lang-c">    #include "tensorflow/lite/delegates/coreml/coreml_delegate.h"

    // Initialize interpreter with model
    TfLiteModel* model = TfLiteModelCreateFromFile(model_path);

    // Initialize interpreter with Core ML delegate
    TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
    TfLiteDelegate* delegate = TfLiteCoreMlDelegateCreate(NULL);  // default config
    TfLiteInterpreterOptionsAddDelegate(options, delegate);
    TfLiteInterpreterOptionsDelete(options);

    TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, options);

    TfLiteInterpreterAllocateTensors(interpreter);

    // Run inference ...

    /* ... */

    // Dispose resources when it is no longer used.
    // Add following code to the section where you dispose of the delegate
    // (e.g. `dealloc` of class).

    TfLiteInterpreterDelete(interpreter);
    TfLiteCoreMlDelegateDelete(delegate);
    TfLiteModelDelete(model);
      </pre>
    </section>
  </devsite-selector>
</div>

## ベストプラクティス

### Neural Engine を搭載しないデバイスで Core ML デリゲートを使用する

デフォルトでは、デバイスが Neural Engine を搭載している場合にのみ Core ML のデリゲートを作成し、デリゲートが作成されない場合は`null`を返します。他の環境（例えばシミュレータなど）で Core ML のデリゲートを実行する場合は、Swift でデリゲートを作成する際に`.all`をオプションとして渡します。C++（および Objective-C）では、`TfLiteCoreMlDelegateAllDevices`を渡すことができます。以下の例は、その方法を示しています。

<div>
  <devsite-selector>
    <section>
      <h3>Swift</h3>
      <p></p>
<pre class="prettyprint lang-swift">var options = CoreMLDelegate.Options()
options.enabledDevices = .all
let coreMLDelegate = CoreMLDelegate(options: options)!
let interpreter = try Interpreter(modelPath: modelPath,
                                  delegates: [coreMLDelegate])
</pre>
    </section>
    <section>
      <h3>Objective-C</h3>
      <p></p>
<pre class="prettyprint lang-objc">    TFLCoreMLDelegateOptions* coreMLOptions = [[TFLCoreMLDelegateOptions alloc] init];
    coreMLOptions.enabledDevices = TFLCoreMLDelegateEnabledDevicesAll;
    TFLCoreMLDelegate* coreMLDelegate = [[TFLCoreMLDelegate alloc]
                                          initWithOptions:coreMLOptions];

    // Initialize interpreter with delegate
      </pre>
    </section>
    <section>
      <h3>C</h3>
      <p></p>
<pre class="prettyprint lang-c">    TfLiteCoreMlDelegateOptions options;
    options.enabled_devices = TfLiteCoreMlDelegateAllDevices;
    TfLiteDelegate* delegate = TfLiteCoreMlDelegateCreate(&amp;options);
    // Initialize interpreter with delegate
      </pre>
    </section>
  </devsite-selector>
</div>

### Metal (GPU) デリゲートをフォールバックとして使用する。

Core ML のデリゲートが作成されない場合でも、[Metal デリゲート](https://www.tensorflow.org/lite/performance/gpu#ios) を使用してパフォーマンスの向上を図ることができます。以下の例は、その方法を示しています。

<div>
  <devsite-selector>
    <section>
      <h3>Swift</h3>
      <p></p>
<pre class="prettyprint lang-swift">    var delegate = CoreMLDelegate()
    if delegate == nil {
      delegate = MetalDelegate()  // Add Metal delegate options if necessary.
    }

    let interpreter = try Interpreter(modelPath: modelPath,
                                      delegates: [delegate!])
      </pre>
    </section>
    <section>
      <h3>Objective-C</h3>
      <p></p>
<pre class="prettyprint lang-objc">    TFLDelegate* delegate = [[TFLCoreMLDelegate alloc] init];
    if (!delegate) {
      // Add Metal delegate options if necessary
      delegate = [[TFLMetalDelegate alloc] init];
    }
    // Initialize interpreter with delegate
      </pre>
    </section>
    <section>
      <h3>C</h3>
      <p></p>
<pre class="prettyprint lang-c">    TfLiteCoreMlDelegateOptions options = {};
    delegate = TfLiteCoreMlDelegateCreate(&amp;options);
    if (delegate == NULL) {
      // Add Metal delegate options if necessary
      delegate = TFLGpuDelegateCreate(NULL);
    }
    // Initialize interpreter with delegate
      </pre>
    </section>
  </devsite-selector>
</div>

デリゲート作成のロジックがデバイスのマシン ID（iPhone11,1 など）を読み取って Neural Engine が利用できるかを判断します。詳細は[コード](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/experimental/delegates/coreml/coreml_delegate.mm)をご覧ください。または、[DeviceKit](https://github.com/devicekit/DeviceKit) などの他のライブラリを使用して、独自の拒否リストデバイスのセットを実装することもできます。

### 古い Core ML バージョンを使用する

iOS 13 は Core ML 3 をサポートしていますが、Core ML 2 のモデル仕様に変換すると動作が良くなる場合があります。デフォルトでは変換対象のバージョンは最新バージョンに設定されていますが、デリゲートオプションで`coreMLVersion`（Swift の場合は C API の`coreml_version`）を古いバージョンに設定することによって変更が可能です。

## サポートする演算子

Core ML デリゲートがサポートする演算子は以下の通りです。

- Add
    - Only certain shapes are broadcastable. In Core ML tensor layout, following tensor shapes are broadcastable. `[B, C, H, W]`, `[B, C, 1, 1]`, `[B, 1, H, W]`, `[B, 1, 1, 1]`.
- AveragePool2D
- Concat
    - 連結はチャンネル軸に沿って行う必要があります。
- Conv2D
    - 重みやバイアスは定数である必要があります。
- DepthwiseConv2D
    - 重みやバイアスは定数である必要があります。
- FullyConnected（別名 Dense または InnerProduct）
    - 重みやバイアスは（存在する場合）定数である必要があります。
    - Only supports single-batch case. Input dimensions should be 1, except the last dimension.
- Hardswish
- Logistic（別名 Sigmoid）
- MaxPool2D
- MirrorPad
    - `REFLECT`モードの 4 次元入力のみをサポートします。パディングは定数である必要があり、H 次元と W 次元にのみ許可されます。
- Mul
    - Only certain shapes are broadcastable. In Core ML tensor layout, following tensor shapes are broadcastable. `[B, C, H, W]`, `[B, C, 1, 1]`, `[B, 1, H, W]`, `[B, 1, 1, 1]`.
- Pad および PadV2
    - 4 次元入力のみをサポートします。パディングは定数である必要があり、H 次元と W 次元にのみ許可されます。
- Relu
- ReluN1To1
- Relu6
- Reshape
    - 対象の Core ML バージョンが 2 の場合にのみサポートされ、Core ML 3 の場合はサポートされません。
- ResizeBilinear
- SoftMax
- Tanh
- TransposeConv
    - 重みは定数である必要があります。

## フィードバック

For issues, please create a [GitHub](https://github.com/tensorflow/tensorflow/issues/new?template=50-other-issues.md) issue with all the necessary details to reproduce.

## よくある質問

- Does CoreML delegate support fallback to CPU if a graph contains unsupported ops?
    - はい
- CoreML デリゲートは iOS Simulator で動作しますか？
    - Yes. The library includes x86 and x86_64 targets so it can run on a simulator, but you will not see performance boost over CPU.
- TensorFlow Lite と CoreML デリゲートは MacOS をサポートしていますか？
    - TensorFlow Lite は iOS のみでテストを行っており、MacOS ではテストしていません。
- カスタムの TensorFlow Lite 演算子はサポートされますか？
    - No, CoreML delegate does not support custom ops and they will fallback to CPU.

## API

- [Core ML delegate Swift API](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/swift/Sources/CoreMLDelegate.swift)
- [Core ML delegate C API](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/coreml/coreml_delegate.h)
    - This can be used for Objective-C codes. ~~~
