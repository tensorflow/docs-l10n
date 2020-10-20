# Tensorflow Lite Core ML デリゲート

TensorFlow Lite Core ML デリゲートは、[Core ML フレームワーク](https://developer.apple.com/documentation/coreml)上で TensorFlow Lite モデルの実行を可能にし、その結果、iOS デバイス上におけるモデル推論の高速化を実現しました。

注意: このデリゲートは実験（ベータ）段階です。

注意: Core ML デリゲートは、Core ML のバージョン 2 以降をサポートしています。

**サポートする iOS のバージョンとデバイス:**

- iOS 12 以降。古い iOS バージョンの場合、Core ML デリゲートは自動的に CPU にフォールバックします。
- デフォルトでは、Core ML デリゲートは A12 SoC 以降のデバイス（iPhone Xs 以降）でのみ有効で、Neural Engine（ニューラルエンジン）を推論の高速化に使用します。古いデバイスで Core ML デリゲートを使用する場合は、[ベストプラクティス](#best-practices)をご覧ください。

**サポートするモデル**

現在、Core ML のデリゲートは浮動小数点数（FP32 と FP16）モデルをサポートしています。

## 独自のモデルで Core ML デリゲートを試す

TensorFlow Lite CocoaPods のナイトリーリリースには、既に Core ML デリゲートが含まれています。Core ML デリゲートを使用する場合は、TensorFlow Lite ポッド（C API用の`TensorflowLiteC`と Swift 用の`TensorFlowLiteSwift`）のバージョンを`Podfile`で`0.0.1-nightly`に変更し、サブスペック`CoreML`をインクルードします。

```
target 'YourProjectName'
  # pod 'TensorFlowLiteSwift'
  pod 'TensorFlowLiteSwift/CoreML', '~> 0.0.1-nightly'
```

または

```
target 'YourProjectName'
  # pod 'TensorFlowLiteSwift'
  pod 'TensorFlowLiteSwift', '~> 0.0.1-nightly', :subspecs => ['CoreML']
```

注意: `Podfile`の更新後、`pod update`を実行して変更を反映させる必要があります。最新の`CoreMLDelegate.swift`ファイルが表示されない場合は、`pod cache clean TensorFlowLiteSwift`を実行します。

### Swift

Core ML デリゲートを使用して TensorFlow Lite インタープリタを初期化します。

```swift
let coreMLDelegate = CoreMLDelegate()
var interpreter: Interpreter

// Core ML delegate will only be created for devices with Neural Engine
if coreMLDelegate != nil {
  interpreter = try Interpreter(modelPath: modelPath,
                                delegates: [coreMLDelegate!])
} else {
  interpreter = try Interpreter(modelPath: modelPath)
}
```

### Objective-C

Core ML のデリゲートは Objective-C コード用の C API を使用しています。

#### ステップ 1. `coreml_delegate.h`をインクルードする。

```c
#include "tensorflow/lite/experimental/delegates/coreml/coreml_delegate.h"
```

#### ステップ 2. デリゲートを作成して TensorFlow Lite Interpreter を初期化する。

インタプリタのオプションを初期化してから、初期化された Core ML のデリゲートで`TfLiteInterpreterOptionsAddDelegate`を呼び出し、デリゲートを適用します。その後、作成したオプションでインタプリタを初期化します。

```c
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
```

#### ステップ 3. 使用されなくなった時点でリソースを破棄する。

デリゲートを破棄するセクション（例えばクラスの`dealloc`など）にこのコードを追加します。

```c
TfLiteInterpreterDelete(interpreter);
TfLiteCoreMlDelegateDelete(delegate);
TfLiteModelDelete(model);
```

## ベストプラクティス

### Neural Engine を搭載しないデバイスで Core ML デリゲートを使用する

デフォルトでは、デバイスが Neural Engine を搭載している場合にのみ Core ML のデリゲートを作成し、デリゲートが作成されない場合は`null`を返します。他の環境（例えばシミュレータなど）で Core ML のデリゲートを実行する場合は、Swift でデリゲートを作成する際に`.all`をオプションとして渡します。C++（および Objective-C）では、`TfLiteCoreMlDelegateAllDevices`を渡すことができます。以下の例は、その方法を示しています。

#### Swift

```swift
var options = CoreMLDelegate.Options()
options.enabledDevices = .all
let coreMLDelegate = CoreMLDelegate(options: options)!
let interpreter = try Interpreter(modelPath: modelPath,
                                  delegates: [coreMLDelegate])
```

#### Objective-C

```c
TfLiteCoreMlDelegateOptions options;
options.enabled_devices = TfLiteCoreMlDelegateAllDevices;
TfLiteDelegate* delegate = TfLiteCoreMlDelegateCreate(&options);
// Initialize interpreter with delegate
```

### Metal (GPU) デリゲートをフォールバックとして使用する。

Core ML のデリゲートが作成されない場合でも、[Metal デリゲート](https://www.tensorflow.org/lite/performance/gpu#ios) を使用してパフォーマンスの向上を図ることができます。以下の例は、その方法を示しています。

#### Swift

```swift
var delegate = CoreMLDelegate()
if delegate == nil {
  delegate = MetalDelegate()  // Add Metal delegate options if necessary.
}

let interpreter = try Interpreter(modelPath: modelPath,
                                  delegates: [delegate!])
```

#### Objective-C

```c
TfLiteCoreMlDelegateOptions options = {};
delegate = TfLiteCoreMlDelegateCreate(&options);
if (delegate == NULL) {
  // Add Metal delegate options if necessary
  delegate = TFLGpuDelegateCreate(NULL);
}
// Initialize interpreter with delegate
```

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
    - 単一バッチケースのみをサポートします。入力次元は、最後の次元以外は 1 である必要があります。
- Hardswish
- Logistic（別名 Sigmoid）
- MaxPool2D
- MirrorPad
    - `REFLECT`モードの 4 次元入力のみをサポートします。パディングは定数である必要があり、H 次元と W 次元にのみ許可されます。
- Mul
    - 特定の形状に限りブロードキャストが可能です。Core ML のテンソルレイアウトでは、次のテンソル形状をブロードキャストできます。`[B, C, H, W]`、`[B, C, 1, 1]`、`[B, 1, H, W]`、 `[B, 1, 1, 1]`。
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

問題などが生じた場合は、[GitHub](https://github.com/tensorflow/tensorflow/issues/new?template=50-other-issues.md)の Issue を作成し、再現に必要なすべての詳細を記載してください。

## よくある質問

- サポートされていない演算子がグラフに含まれている場合、CoreML デリゲートは CPU へのフォールバックをサポートしますか？
    - はい
- CoreML デリゲートは iOS Simulator で動作しますか？
    - はい。ライブラリには x86 と x86_64 ターゲットが含まれているのでシミュレータ上で実行できますが、パフォーマンスが CPU より向上することはありません。
- TensorFlow Lite と CoreML デリゲートは MacOS をサポートしていますか？
    - TensorFlow Lite は iOS のみでテストを行っており、MacOS ではテストしていません。
- カスタムの TensorFlow Lite 演算子はサポートされますか？
    - いいえ、CoreML デリゲートはカスタム演算子をサポートしていないため、CPU にフォールバックします。

## API

- [Core ML デリゲート Swift API](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/experimental/swift/Sources/CoreMLDelegate.swift)
- [Core ML デリゲート C API](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/experimental/delegates/coreml/coreml_delegate.h)
    - これは Objective-C コードに使用可能です。
