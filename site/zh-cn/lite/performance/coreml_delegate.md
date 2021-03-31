# Tensorflow Lite Core ML 委托

利用 TensorFlow Lite Core ML 委托，您可以在 [Core ML 框架](https://developer.apple.com/documentation/coreml)上运行 TensorFlow Lite 模型，从而加快 iOS 设备上的模型推断速度。

Note: This delegate is in experimental (beta) phase. It is available from TensorFlow Lite 2.4.0 and latest nightly releases.

注：Core ML 委托支持 Core ML 版本 2 及更高版本。

**支持的 iOS 版本和设备：**

- iOS 12 and later. In the older iOS versions, Core ML delegate will automatically fallback to CPU.
- By default, Core ML delegate will only be enabled on devices with A12 SoC and later (iPhone Xs and later) to use Neural Engine for faster inference. If you want to use Core ML delegate also on the older devices, please see [best practices](#best-practices)

**支持的模型**

目前，Core ML 委托支持浮点（FP32 和 FP16）模型。

## 在自己的模型上尝试 Core ML 委托

The Core ML delegate is already included in nightly release of TensorFlow lite CocoaPods. To use Core ML delegate, change your TensorFlow lite pod to include subspec `CoreML` in your `Podfile`.

Note: If you want to use C API instead of Objective-C API, you can include `TensorFlowLiteC/CoreML` pod to do so.

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

Note: Core ML delegate can also use C API for Objective-C code. Prior to TensorFlow Lite 2.4.0 release, this was the only option.

<div>
  <devsite-selector>
    <section>
      <h3>Swift</h3>
      <p></p>
<pre class="prettyprint lang-swift">    let coreMLDelegate = CoreMLDelegate()
    var interpreter: Interpreter
&lt;/div&gt;
&lt;pre data-md-type="block_code" data-md-language=""&gt;&lt;code&gt;GL_CODE_5&lt;/code&gt;</pre>
<div data-md-type="block_html">
</div>
</section></devsite-selector>
</div>
<h2 data-md-type="header" data-md-header-level="2">最佳做法</h2>
<h3 data-md-type="header" data-md-header-level="3">在没有 Neural Engine 的设备上使用 Core ML 委托</h3>
<p data-md-type="paragraph">默认情况下，仅在具有 Neural Engine 的设备上创建 Core ML 委托，如果没有创建委托，将返回 <code data-md-type="codespan">null</code>。如果要在其他环境（如模拟器）中运行 Core ML 委托，则在 Swift 中创建委托时，将 <code data-md-type="codespan">.all</code> 作为一个选项传递。在 C++（和 Objective-C）中，您可以传递 <code data-md-type="codespan">TfLiteCoreMlDelegateAllDevices</code>。下面的示例介绍了如何执行此操作：</p>
<div data-md-type="block_html">
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
&lt;/div&gt;
&lt;pre data-md-type="block_code" data-md-language=""&gt;&lt;code&gt;GL_CODE_9&lt;/code&gt;</pre>
<div data-md-type="block_html">
</div>
</section></devsite-selector>
</div>
<h3 data-md-type="header" data-md-header-level="3">使用 Metal(GPU) 委托作为回退</h3>
<p data-md-type="paragraph">如果未创建 Core ML 委托，您也可以使用 <a href="https://www.tensorflow.org/lite/performance/gpu#ios" data-md-type="link">Metal 委托</a>获得性能优势。下面的示例介绍了如何执行此操作：</p>
<div data-md-type="block_html">
<div>
  <devsite-selector>
    <section>
      <h3>Swift</h3>
      <p></p>
<pre class="prettyprint lang-swift">    var delegate = CoreMLDelegate()
    if delegate == nil {
      delegate = MetalDelegate()  // Add Metal delegate options if necessary.
    }
&lt;/div&gt;
&lt;pre data-md-type="block_code" data-md-language=""&gt;&lt;code&gt;GL_CODE_10&lt;/code&gt;</pre>
<div data-md-type="block_html">
</div>
</section></devsite-selector>
</div>
<p data-md-type="paragraph">委托创建逻辑读取设备的机器 ID（如 iPhone11,1）以确定其 Neural Engine 可用性。有关更多详细信息，请参见<a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/experimental/delegates/coreml/coreml_delegate.mm" data-md-type="link">代码</a>。或者，您也可以使用其他库（如 <a href="https://github.com/devicekit/DeviceKit" data-md-type="link">DeviceKit</a>）实现自己的拒绝列表设备集。有关更多详细信息，请参见<a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/experimental/delegates/coreml/coreml_delegate.mm" data-md-type="link">代码</a>。或者，您也可以使用其他库（如 <a href="https://github.com/devicekit/DeviceKit" data-md-type="link">DeviceKit</a>）实现自己的拒绝名单设备集。</p>
<h3 data-md-type="header" data-md-header-level="3">使用旧 Core ML 版本</h3>
<p data-md-type="paragraph">虽然 iOS 13 支持 Core ML 3，但使用 Core ML 2 模型规范进行转换后，该模型的效果可能更好。目标转换版本默认设置为最新的版本，但您可以通过在委托选项中设置 <code data-md-type="codespan">coreMLVersion</code>（在 Swift 中；在 C API 中则为 <code data-md-type="codespan">coreml_version</code>），将其更改为旧版本。</p>
<h2 data-md-type="header" data-md-header-level="2">支持的运算</h2>
<p data-md-type="paragraph">Core ML 委托支持以下运算。</p>
<ul data-md-type="list" data-md-list-type="unordered" data-md-list-tight="true">
<li data-md-type="list_item" data-md-list-type="unordered">
<p data-md-type="paragraph">Add</p>
<ul data-md-type="list" data-md-list-type="unordered" data-md-list-tight="true">
<li data-md-type="list_item" data-md-list-type="unordered">Only certain shapes are broadcastable. In Core ML tensor layout, following tensor shapes are broadcastable. <code data-md-type="codespan">[B, C, H, W]</code>, <code data-md-type="codespan">[B, C, 1, 1]</code>, <code data-md-type="codespan">[B, 1, H, W]</code>, <code data-md-type="codespan">[B, 1, 1, 1]</code>.</li>
</ul>
</li>
<li data-md-type="list_item" data-md-list-type="unordered">AveragePool2D</li>
<li data-md-type="list_item" data-md-list-type="unordered">
<p data-md-type="paragraph">Concat</p>
<ul data-md-type="list" data-md-list-type="unordered" data-md-list-tight="true">
<li data-md-type="list_item" data-md-list-type="unordered">串联应沿通道轴执行。</li>
</ul>
</li>
<li data-md-type="list_item" data-md-list-type="unordered">
<p data-md-type="paragraph">Conv2D</p>
<ul data-md-type="list" data-md-list-type="unordered" data-md-list-tight="true">
<li data-md-type="list_item" data-md-list-type="unordered">权重和偏差应为常量。</li>
</ul>
</li>
<li data-md-type="list_item" data-md-list-type="unordered">
<p data-md-type="paragraph">DepthwiseConv2D</p>
<ul data-md-type="list" data-md-list-type="unordered" data-md-list-tight="true">
<li data-md-type="list_item" data-md-list-type="unordered">权重和偏差应为常量。</li>
</ul>
</li>
<li data-md-type="list_item" data-md-list-type="unordered">
<p data-md-type="paragraph">FullyConnected（又称 Dense 或 InnerProduct）</p>
<ul data-md-type="list" data-md-list-type="unordered" data-md-list-tight="true">
<li data-md-type="list_item" data-md-list-type="unordered">权重和偏差（如果存在）应为常量。</li>
<li data-md-type="list_item" data-md-list-type="unordered">Only supports single-batch case. Input dimensions should be 1, except the last dimension.</li>
</ul>
</li>
<li data-md-type="list_item" data-md-list-type="unordered">Hardswish</li>
<li data-md-type="list_item" data-md-list-type="unordered">Logistic（又称 Sigmoid）</li>
<li data-md-type="list_item" data-md-list-type="unordered">MaxPool2D</li>
<li data-md-type="list_item" data-md-list-type="unordered">
<p data-md-type="paragraph">MirrorPad</p>
<ul data-md-type="list" data-md-list-type="unordered" data-md-list-tight="true">
<li data-md-type="list_item" data-md-list-type="unordered">仅支持使用 <code data-md-type="codespan">REFLECT</code> 模式的四维输入。填充应为常量，并且只能用于 H 和 W 维度。</li>
</ul>
</li>
<li data-md-type="list_item" data-md-list-type="unordered">
<p data-md-type="paragraph">Mul</p>
<ul data-md-type="list" data-md-list-type="unordered" data-md-list-tight="true">
<li data-md-type="list_item" data-md-list-type="unordered">只能广播几种形状。在 Core ML 张量布局中，可以广播以下张量形状。<code data-md-type="codespan">[B, C, H, W]</code>、<code data-md-type="codespan">[B, C, 1, 1]</code>、<code data-md-type="codespan">[B, 1, H, W]</code>、<code data-md-type="codespan">[B, 1, 1, 1]</code>。</li>
</ul>
</li>
<li data-md-type="list_item" data-md-list-type="unordered">
<p data-md-type="paragraph">Pad 和 PadV2</p>
<ul data-md-type="list" data-md-list-type="unordered" data-md-list-tight="true">
<li data-md-type="list_item" data-md-list-type="unordered">仅支持四维输入。填充应为常量，并且只能用于 H 和 W 维度。</li>
</ul>
</li>
<li data-md-type="list_item" data-md-list-type="unordered">Relu</li>
<li data-md-type="list_item" data-md-list-type="unordered">ReluN1To1</li>
<li data-md-type="list_item" data-md-list-type="unordered">Relu6</li>
<li data-md-type="list_item" data-md-list-type="unordered">
<p data-md-type="paragraph">Reshape</p>
<ul data-md-type="list" data-md-list-type="unordered" data-md-list-tight="true">
<li data-md-type="list_item" data-md-list-type="unordered">仅当 Core ML 版本为 2 时才受支持，当目标版本为 Core ML 3 时不受支持。</li>
</ul>
</li>
<li data-md-type="list_item" data-md-list-type="unordered">ResizeBilinear</li>
<li data-md-type="list_item" data-md-list-type="unordered">SoftMax</li>
<li data-md-type="list_item" data-md-list-type="unordered">Tanh</li>
<li data-md-type="list_item" data-md-list-type="unordered">
<p data-md-type="paragraph">TransposeConv</p>
<ul data-md-type="list" data-md-list-type="unordered" data-md-list-tight="true">
<li data-md-type="list_item" data-md-list-type="unordered">权重应为常量。</li>
</ul>
</li>
</ul>
<h2 data-md-type="header" data-md-header-level="2">反馈</h2>
<p data-md-type="paragraph">如有问题，请创建 <a href="https://github.com/tensorflow/tensorflow/issues/new?template=50-other-issues.md" data-md-type="link">GitHub</a> 问题，并提供重现问题所需的所有必要详情。</p>
<h2 data-md-type="header" data-md-header-level="2">常见问题解答</h2>
<ul data-md-type="list" data-md-list-type="unordered" data-md-list-tight="true">
<li data-md-type="list_item" data-md-list-type="unordered">
<p data-md-type="paragraph">如果计算图包含不受支持的运算，CoreML 委托支持会回退到 CPU 吗？</p>
<ul data-md-type="list" data-md-list-type="unordered" data-md-list-tight="true">
<li data-md-type="list_item" data-md-list-type="unordered">会</li>
</ul>
</li>
<li data-md-type="list_item" data-md-list-type="unordered">
<p data-md-type="paragraph">CoreML 委托可以在 iOS 模拟器上工作吗？</p>
<ul data-md-type="list" data-md-list-type="unordered" data-md-list-tight="true">
<li data-md-type="list_item" data-md-list-type="unordered">Yes. The library includes x86 and x86_64 targets so it can run on a simulator, but you will not see performance boost over CPU.</li>
</ul>
</li>
<li data-md-type="list_item" data-md-list-type="unordered">
<p data-md-type="paragraph">TensorFlow Lite 和 CoreML 委托支持 MacOS 吗？</p>
<ul data-md-type="list" data-md-list-type="unordered" data-md-list-tight="true">
<li data-md-type="list_item" data-md-list-type="unordered">TensorFlow Lite 仅在 iOS 上进行过测试，未在 MacOS 上进行测试。</li>
</ul>
</li>
<li data-md-type="list_item" data-md-list-type="unordered">
<p data-md-type="paragraph">支持自定义 TF Lite 运算吗？</p>
<ul data-md-type="list" data-md-list-type="unordered" data-md-list-tight="true">
<li data-md-type="list_item" data-md-list-type="unordered">No, CoreML delegate does not support custom ops and they will fallback to CPU.</li>
</ul>
</li>
</ul>
<h2 data-md-type="header" data-md-header-level="2">API</h2>
<ul data-md-type="list" data-md-list-type="unordered" data-md-list-tight="true">
<li data-md-type="list_item" data-md-list-type="unordered"><a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/swift/Sources/CoreMLDelegate.swift" data-md-type="link">Core ML delegate Swift API</a></li>
<li data-md-type="list_item" data-md-list-type="unordered">
<p data-md-type="paragraph"><a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/coreml/coreml_delegate.h" data-md-type="link">Core ML delegate C API</a></p>
<ul data-md-type="list" data-md-list-type="unordered" data-md-list-tight="true">
<li data-md-type="list_item" data-md-list-type="unordered">This can be used for Objective-C codes. ~~~</li>
</ul>
</li>
</ul>
</div>
</div>
