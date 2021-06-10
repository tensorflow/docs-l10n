# Tensorflow Lite Core ML デリゲート

TensorFlow Lite Core ML デリゲートは、[Core ML フレームワーク](https://developer.apple.com/documentation/coreml)上での TensorFlow Lite モデルの実行を可能にしました。その結果、iOS デバイス上におけるモデル推論の高速化します。

注意: このデリゲートは試験（ベータ）段階です。TensorFlow Lite 2.4.0 および最新のナイトリーリリースから利用できます。

注意: Core ML デリゲートは、Core ML のバージョン 2 以降をサポートしています。

**サポートする iOS のバージョンとデバイス:**

- iOS 12 以降。古い iOS バージョンの場合、Core ML デリゲートは自動的に CPU にフォールバックします。
- デフォルトでは、Core ML デリゲートは A12 SoC 以降のデバイス（iPhone Xs 以降）でのみ有効で、Neural Engine（ニューラルエンジン）を推論の高速化に使用します。古いデバイスで Core ML デリゲートを使用する場合は、[ベストプラクティス](#best-practices)をご覧ください。

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
&lt;/div&gt;
&lt;pre data-md-type="block_code" data-md-language=""&gt;&lt;code&gt;GL_CODE_5&lt;/code&gt;</pre>
<div data-md-type="block_html">
</div>
</section></devsite-selector>
</div>
<h2 data-md-type="header" data-md-header-level="2">ベストプラクティス</h2>
<h3 data-md-type="header" data-md-header-level="3">Neural Engine を搭載しないデバイスで Core ML デリゲートを使用する</h3>
<p data-md-type="paragraph">デフォルトでは、デバイスが Neural Engine を搭載している場合にのみ Core ML のデリゲートを作成し、デリゲートが作成されない場合は<code data-md-type="codespan">null</code>を返します。他の環境（例えばシミュレータなど）で Core ML のデリゲートを実行する場合は、Swift でデリゲートを作成する際に<code data-md-type="codespan">.all</code>をオプションとして渡します。C++（および Objective-C）では、<code data-md-type="codespan">TfLiteCoreMlDelegateAllDevices</code>を渡すことができます。以下の例は、その方法を示しています。</p>
<div data-md-type="block_html">
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
&lt;/div&gt;
&lt;pre data-md-type="block_code" data-md-language=""&gt;&lt;code&gt;GL_CODE_9&lt;/code&gt;</pre>
<div data-md-type="block_html">
</div>
</section></devsite-selector>
</div>
<h3 data-md-type="header" data-md-header-level="3">Metal (GPU) デリゲートをフォールバックとして使用する。</h3>
<p data-md-type="paragraph">Core ML のデリゲートが作成されない場合でも、<a href="https://www.tensorflow.org/lite/performance/gpu#ios" data-md-type="link">Metal デリゲート</a> を使用してパフォーマンスの向上を図ることができます。以下の例は、その方法を示しています。</p>
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
<p data-md-type="paragraph">デリゲート作成のロジックがデバイスのマシン ID（iPhone11,1 など）を読み取って Neural Engine が利用できるかを判断します。詳細は<a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/experimental/delegates/coreml/coreml_delegate.mm" data-md-type="link">コード</a>をご覧ください。または、<a href="https://github.com/devicekit/DeviceKit" data-md-type="link">DeviceKit</a> などの他のライブラリを使用して、独自の拒否リストデバイスのセットを実装することもできます。</p>
<h3 data-md-type="header" data-md-header-level="3">古い Core ML バージョンを使用する</h3>
<p data-md-type="paragraph">iOS 13 は Core ML 3 をサポートしていますが、Core ML 2 のモデル仕様に変換すると動作が良くなる場合があります。デフォルトでは変換対象のバージョンは最新バージョンに設定されていますが、デリゲートオプションで<code data-md-type="codespan">coreMLVersion</code>（Swift の場合は C API の<code data-md-type="codespan">coreml_version</code>）を古いバージョンに設定することによって変更が可能です。</p>
<h2 data-md-type="header" data-md-header-level="2">サポートする演算子</h2>
<p data-md-type="paragraph">Core ML デリゲートがサポートする演算子は以下の通りです。</p>
<ul data-md-type="list" data-md-list-type="unordered" data-md-list-tight="true">
<li data-md-type="list_item" data-md-list-type="unordered">
<p data-md-type="paragraph">Add</p>
<ul data-md-type="list" data-md-list-type="unordered" data-md-list-tight="true">
<li data-md-type="list_item" data-md-list-type="unordered">特定の形状に限りブロードキャストが可能です。Core ML のテンソルレイアウトでは、次のテンソル形状をブロードキャストできます。<code data-md-type="codespan">[B, C, H, W]</code>、<code data-md-type="codespan">[B, C, 1, 1]</code>、<code data-md-type="codespan">[B, 1, H, W]</code>、 <code data-md-type="codespan">[B, 1, 1, 1]</code>。</li>
</ul>
</li>
<li data-md-type="list_item" data-md-list-type="unordered">AveragePool2D</li>
<li data-md-type="list_item" data-md-list-type="unordered">
<p data-md-type="paragraph">Concat</p>
<ul data-md-type="list" data-md-list-type="unordered" data-md-list-tight="true">
<li data-md-type="list_item" data-md-list-type="unordered">連結はチャンネル軸に沿って行う必要があります。</li>
</ul>
</li>
<li data-md-type="list_item" data-md-list-type="unordered">
<p data-md-type="paragraph">Conv2D</p>
<ul data-md-type="list" data-md-list-type="unordered" data-md-list-tight="true">
<li data-md-type="list_item" data-md-list-type="unordered">重みやバイアスは定数である必要があります。</li>
</ul>
</li>
<li data-md-type="list_item" data-md-list-type="unordered">
<p data-md-type="paragraph">DepthwiseConv2D</p>
<ul data-md-type="list" data-md-list-type="unordered" data-md-list-tight="true">
<li data-md-type="list_item" data-md-list-type="unordered">重みやバイアスは定数である必要があります。</li>
</ul>
</li>
<li data-md-type="list_item" data-md-list-type="unordered">
<p data-md-type="paragraph">FullyConnected（別名 Dense または InnerProduct）</p>
<ul data-md-type="list" data-md-list-type="unordered" data-md-list-tight="true">
<li data-md-type="list_item" data-md-list-type="unordered">重みやバイアスは（存在する場合）定数である必要があります。</li>
<li data-md-type="list_item" data-md-list-type="unordered">Only supports single-batch case. Input dimensions should be 1, except the last dimension.</li>
</ul>
</li>
<li data-md-type="list_item" data-md-list-type="unordered">Hardswish</li>
<li data-md-type="list_item" data-md-list-type="unordered">Logistic（別名 Sigmoid）</li>
<li data-md-type="list_item" data-md-list-type="unordered">MaxPool2D</li>
<li data-md-type="list_item" data-md-list-type="unordered">
<p data-md-type="paragraph">MirrorPad</p>
<ul data-md-type="list" data-md-list-type="unordered" data-md-list-tight="true">
<li data-md-type="list_item" data-md-list-type="unordered">
<code data-md-type="codespan">REFLECT</code>モードの 4 次元入力のみをサポートします。パディングは定数である必要があり、H 次元と W 次元にのみ許可されます。</li>
</ul>
</li>
<li data-md-type="list_item" data-md-list-type="unordered">
<p data-md-type="paragraph">Mul</p>
<ul data-md-type="list" data-md-list-type="unordered" data-md-list-tight="true">
<li data-md-type="list_item" data-md-list-type="unordered">特定の形状に限りブロードキャストが可能です。Core ML のテンソルレイアウトでは、次のテンソル形状をブロードキャストできます。<code data-md-type="codespan">[B, C, H, W]</code>、<code data-md-type="codespan">[B, C, 1, 1]</code>、<code data-md-type="codespan">[B, 1, H, W]</code>、 <code data-md-type="codespan">[B, 1, 1, 1]</code>。</li>
</ul>
</li>
<li data-md-type="list_item" data-md-list-type="unordered">
<p data-md-type="paragraph">Pad および PadV2</p>
<ul data-md-type="list" data-md-list-type="unordered" data-md-list-tight="true">
<li data-md-type="list_item" data-md-list-type="unordered">4 次元入力のみをサポートします。パディングは定数である必要があり、H 次元と W 次元にのみ許可されます。</li>
</ul>
</li>
<li data-md-type="list_item" data-md-list-type="unordered">Relu</li>
<li data-md-type="list_item" data-md-list-type="unordered">ReluN1To1</li>
<li data-md-type="list_item" data-md-list-type="unordered">Relu6</li>
<li data-md-type="list_item" data-md-list-type="unordered">
<p data-md-type="paragraph">Reshape</p>
<ul data-md-type="list" data-md-list-type="unordered" data-md-list-tight="true">
<li data-md-type="list_item" data-md-list-type="unordered">対象の Core ML バージョンが 2 の場合にのみサポートされ、Core ML 3 の場合はサポートされません。</li>
</ul>
</li>
<li data-md-type="list_item" data-md-list-type="unordered">ResizeBilinear</li>
<li data-md-type="list_item" data-md-list-type="unordered">SoftMax</li>
<li data-md-type="list_item" data-md-list-type="unordered">Tanh</li>
<li data-md-type="list_item" data-md-list-type="unordered">
<p data-md-type="paragraph">TransposeConv</p>
<ul data-md-type="list" data-md-list-type="unordered" data-md-list-tight="true">
<li data-md-type="list_item" data-md-list-type="unordered">重みは定数である必要があります。</li>
</ul>
</li>
</ul>
<h2 data-md-type="header" data-md-header-level="2">フィードバック</h2>
<p data-md-type="paragraph">問題などが生じた場合は、<a href="https://github.com/tensorflow/tensorflow/issues/new?template=50-other-issues.md" data-md-type="link">GitHub</a> の課題を提出し、再現に必要なすべての詳細を記載してください。</p>
<h2 data-md-type="header" data-md-header-level="2">よくある質問</h2>
<ul data-md-type="list" data-md-list-type="unordered" data-md-list-tight="true">
<li data-md-type="list_item" data-md-list-type="unordered">
<p data-md-type="paragraph">サポートされていない演算子がグラフに含まれている場合、CoreML デリゲートは CPU へのフォールバックをサポートしますか？</p>
<ul data-md-type="list" data-md-list-type="unordered" data-md-list-tight="true">
<li data-md-type="list_item" data-md-list-type="unordered">はい</li>
</ul>
</li>
<li data-md-type="list_item" data-md-list-type="unordered">
<p data-md-type="paragraph">CoreML デリゲートは iOS Simulator で動作しますか？</p>
<ul data-md-type="list" data-md-list-type="unordered" data-md-list-tight="true">
<li data-md-type="list_item" data-md-list-type="unordered">はい。ライブラリには x86 と x86_64 ターゲットが含まれているのでシミュレータ上で実行できますが、パフォーマンスが CPU より向上することはありません。</li>
</ul>
</li>
<li data-md-type="list_item" data-md-list-type="unordered">
<p data-md-type="paragraph">TensorFlow Lite と CoreML デリゲートは MacOS をサポートしていますか？</p>
<ul data-md-type="list" data-md-list-type="unordered" data-md-list-tight="true">
<li data-md-type="list_item" data-md-list-type="unordered">TensorFlow Lite は iOS のみでテストを行っており、MacOS ではテストしていません。</li>
</ul>
</li>
<li data-md-type="list_item" data-md-list-type="unordered">
<p data-md-type="paragraph">カスタムの TensorFlow Lite 演算子はサポートされますか？</p>
<ul data-md-type="list" data-md-list-type="unordered" data-md-list-tight="true">
<li data-md-type="list_item" data-md-list-type="unordered">いいえ、CoreML デリゲートはカスタム演算子をサポートしていないため、CPU にフォールバックします。</li>
</ul>
</li>
</ul>
<h2 data-md-type="header" data-md-header-level="2">API</h2>
<ul data-md-type="list" data-md-list-type="unordered" data-md-list-tight="true">
<li data-md-type="list_item" data-md-list-type="unordered"><a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/swift/Sources/CoreMLDelegate.swift" data-md-type="link">Core ML delegate Swift API</a></li>
<li data-md-type="list_item" data-md-list-type="unordered">
<p data-md-type="paragraph"><a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/coreml/coreml_delegate.h" data-md-type="link">Core ML delegate C API</a></p>
<ul data-md-type="list" data-md-list-type="unordered" data-md-list-tight="true">
<li data-md-type="list_item" data-md-list-type="unordered">これは Objective-C コードに使用可能です。</li>
</ul>
</li>
</ul>
</div>
</div>
