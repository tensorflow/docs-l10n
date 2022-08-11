# Tensorflow Lite Core ML 委托

利用 TensorFlow Lite Core ML 委托，您可以在 [Core ML 框架](https://developer.apple.com/documentation/coreml)上运行 TensorFlow Lite 模型，从而加快 iOS 设备上的模型推断速度。

注：此委托仍处于实验（测试）阶段。它可以从 TensorFlow Lite 2.4.0 和最新的 Nightly 版本中获得。

注：Core ML 委托支持 Core ML 版本 2 及更高版本。

**支持的 iOS 版本和设备：**

- iOS 12 及更高版本。在旧 iOS 版本中，Core ML 委托会自动回退到 CPU。
- 默认情况下，仅在使用 A12 SoC 及更高版本（iPhone XS 及更新的型号）的设备上启用 Core ML 委托，从而使用 Neural Engine 加快推断速度。如果要在旧设备上也使用 Core ML 委托，请参阅[最佳做法](#best-practices)

**支持的模型**

目前，Core ML 委托支持浮点（FP32 和 FP16）模型。

## 在自己的模型上尝试 Core ML 委托

Core ML 委托已经包含在 TensorFlow lite CocoaPods 的 Nightly 版本中。要使用 Core ML 委托，请更改您的 TensorFlow lite Pod，在您的 `Podfile` 中包含子规范 `CoreML`。

注：如果您使用的是 C API 而不是 Objective-C API，则可以包含 `TensorFlowLiteC/CoreML` Pod。

```
target 'YourProjectName'
  pod 'TensorFlowLiteSwift/CoreML', '~> 2.4.0'  # Or TensorFlowLiteObjC/CoreML
```

或

```
# Particularily useful when you also want to include 'Metal' subspec.
target 'YourProjectName'
  pod 'TensorFlowLiteSwift', '~> 2.4.0', :subspecs => ['CoreML']
```

注：Core ML 委托也可以将 C API 用于 Objective-C 代码。在 TensorFlow Lite 2.4.0 版本之前，这是唯一的选择。

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

## 最佳做法

### 在没有 Neural Engine 的设备上使用 Core ML 委托

默认情况下，仅在具有 Neural Engine 的设备上创建 Core ML 委托，如果没有创建委托，将返回 `null`。如果要在其他环境（如模拟器）中运行 Core ML 委托，则在 Swift 中创建委托时，将 `.all` 作为一个选项传递。在 C++（和 Objective-C）中，您可以传递 `TfLiteCoreMlDelegateAllDevices`。下面的示例介绍了如何执行此操作：

<div>
  <devsite-selector>
    <section>
      <h3>Swift</h3>
      <p></p>
<pre class="prettyprint lang-swift">    var options = CoreMLDelegate.Options()
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

### 使用 Metal(GPU) 委托作为回退

如果未创建 Core ML 委托，您也可以使用 [Metal 委托](https://www.tensorflow.org/lite/performance/gpu#ios)获得性能优势。下面的示例介绍了如何执行此操作：

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

委托创建逻辑读取设备的机器 ID（如 iPhone11,1）以确定其 Neural Engine 可用性。有关更多详细信息，请参见[代码](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/experimental/delegates/coreml/coreml_delegate.mm)。或者，您也可以使用其他库（如 [DeviceKit](https://github.com/devicekit/DeviceKit)）实现自己的拒绝列表设备集。有关更多详细信息，请参见[代码](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/experimental/delegates/coreml/coreml_delegate.mm)。或者，您也可以使用其他库（如 [DeviceKit](https://github.com/devicekit/DeviceKit)）实现自己的拒绝名单设备集。

### 使用旧 Core ML 版本

虽然 iOS 13 支持 Core ML 3，但使用 Core ML 2 模型规范进行转换后，该模型的效果可能更好。目标转换版本默认设置为最新的版本，但您可以通过在委托选项中设置 `coreMLVersion`（在 Swift 中；在 C API 中则为 `coreml_version`），将其更改为旧版本。

## 支持的运算

Core ML 委托支持以下运算。

- Add
    - Only certain shapes are broadcastable. In Core ML tensor layout, following tensor shapes are broadcastable. `[B, C, H, W]`, `[B, C, 1, 1]`, `[B, 1, H, W]`, `[B, 1, 1, 1]`.
- AveragePool2D
- Concat
    - 串联应沿通道轴执行。
- Conv2D
    - 权重和偏差应为常量。
- DepthwiseConv2D
    - 权重和偏差应为常量。
- FullyConnected（又称 Dense 或 InnerProduct）
    - 权重和偏差（如果存在）应为常量。
    - 仅支持单批次情况。除最后一个维度外，输入维度应为 1。
- Hardswish
- Logistic（又称 Sigmoid）
- MaxPool2D
- MirrorPad
    - 仅支持使用 `REFLECT` 模式的四维输入。填充应为常量，并且只能用于 H 和 W 维度。
- Mul
    - 只能广播几种形状。在 Core ML 张量布局中，可以广播以下张量形状。`[B, C, H, W]`、`[B, C, 1, 1]`、`[B, 1, H, W]`、`[B, 1, 1, 1]`。
- Pad 和 PadV2
    - 仅支持四维输入。填充应为常量，并且只能用于 H 和 W 维度。
- Relu
- ReluN1To1
- Relu6
- Reshape
    - 仅当 Core ML 版本为 2 时才受支持，当目标版本为 Core ML 3 时不受支持。
- ResizeBilinear
- SoftMax
- Tanh
- TransposeConv
    - 权重应为常量。

## 反馈

如有问题，请创建 [GitHub](https://github.com/tensorflow/tensorflow/issues/new?template=50-other-issues.md) 问题，并提供重现问题所需的所有必要详情。

## 常见问题解答

- Does CoreML delegate support fallback to CPU if a graph contains unsupported ops?
    - 会
- CoreML 委托可以在 iOS 模拟器上工作吗？
    - 可以。该库包括 x86 和 x86_64 目标，因此，它可以在模拟器上运行，但是性能不会高于 CPU。
- TensorFlow Lite 和 CoreML 委托支持 MacOS 吗？
    - TensorFlow Lite 仅在 iOS 上进行过测试，未在 MacOS 上进行测试。
- 支持自定义 TF Lite 运算吗？
    - 不支持，CoreML 委托不支持自定义运算，它们将回退到 CPU。

## API

- [Core ML delegate Swift API](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/swift/Sources/CoreMLDelegate.swift)
- [Core ML delegate C API](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/coreml/coreml_delegate.h)
    - 可用于 Objective-C 代码。~~~
