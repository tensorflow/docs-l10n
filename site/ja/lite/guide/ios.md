# iOS クイックスタート

iOS で TensorFlow Lite を使い始めるには、次の例をご覧ください。

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/ios">iOS image classification example</a>

ソースコードの説明については、[TensorFlow Lite iOS 画像分類](https://github.com/tensorflow/examples/blob/master/lite/examples/image_classification/ios/EXPLORE_THE_CODE.md)もあわせてお読みください。

This example app uses [image classification](https://www.tensorflow.org/lite/models/image_classification/overview) to continuously classify whatever it sees from the device's rear-facing camera, displaying the top most probable classifications. It allows the user to choose between a floating point or [quantized](https://www.tensorflow.org/lite/performance/post_training_quantization) model and select the number of threads to perform inference on.

注意: さまざまなユースケースで TensorFlow Lite を実演するその他の iOS アプリは、[例](https://www.tensorflow.org/lite/examples)をご覧ください。

## TensorFlow Lite を Swift または Objective-C プロジェクトに追加する

TensorFlow Lite は、[Swift](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/experimental/swift) と [Objective-C](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/experimental/objc) で記述されたネイティブの iOS ライブラリを提供しています。出発点として、[Swift 画像分類の例](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/ios)を使用して、独自の iOS コードを記述してみましょう。

次のセクションでは、TensorFlow Lite Swift または Objective-C をプロジェクトに追加する方法を実演しています。

### CocoaPods 開発者

`Podfile` で、TensorFlow Lite ポッドを追加し、`pod install` を実行します。

#### Swift

```ruby
use_frameworks!
pod 'TensorFlowLiteSwift'
```

#### Objective-C

```ruby
pod 'TensorFlowLiteObjC'
```

#### バージョンを指定する

安定リリースと、`TensorFlowLiteSwift` および `TensorFlowLiteObjC` ポッド用のナイトリーリリースがあります。上記の例のようにバージョン制約を指定しない場合、CocoaPods はデフォルトで最新の安定リリースをプルします。

また、バージョン制約を指定することもできます。たとえば、バージョン 2.0.0 に依存する場合は、依存関係を次のように記述できます。

```ruby
pod 'TensorFlowLiteSwift', '~> 2.0.0'
```

This will ensure the latest available 2.x.y version of the `TensorFlowLiteSwift` pod is used in your app. Alternatively, if you want to depend on the nightly builds, you can write:

```ruby
pod 'TensorFlowLiteSwift', '~> 0.0.1-nightly'
```

From 2.4.0 version and latest nightly releases, by default [GPU](https://www.tensorflow.org/lite/performance/gpu) and [Core ML delegates](https://www.tensorflow.org/lite/performance/coreml_delegate) are excluded from the pod to reduce the binary size. You can include them by specifying subspec:

```ruby
pod 'TensorFlowLiteSwift', '~> 0.0.1-nightly', :subspecs => ['CoreML', 'Metal']
```

このようにすると、TensorFlow Lite に追加される最新の機能を使用できるようになります。`pod install` コマンドを初めて実行したときに `Podfile.lock` ファイルが作成されると、ナイトリーライブラリバージョンは現在の日付のバージョンにロックされていまうことに注意してください。ナイトリーライブラリを最新のものに更新する場合は、`pod update` コマンドを実行する必要があります。

バージョン制約のさまざまな指定方法については、[ポッドバージョンを指定する](https://guides.cocoapods.org/using/the-podfile.html#specifying-pod-versions)をご覧ください。

### Bazel 開発者

`BUILD` ファイルで、ターゲットに `TensorFlowLite` 依存関係を追加します。

#### Swift

```python
swift_library(
  deps = [
      "//tensorflow/lite/swift:TensorFlowLite",
  ],
)
```

#### Objective-C

```python
objc_library(
  deps = [
      "//tensorflow/lite/objc:TensorFlowLite",
  ],
)
```

#### C/C++ API

そのほか、[C API](https://www.tensorflow.org/code/tensorflow/lite/c/c_api.h) または [C++ API](https://tensorflow.org/lite/api_docs/cc) を使用できます。

```python
# Using C API directly
objc_library(
  deps = [
      "//tensorflow/lite/c:c_api",
  ],
)

# Using C++ API directly
objc_library(
  deps = [
      "//third_party/tensorflow/lite:framework",
  ],
)
```

### ライブラリをインポートする

Swift ファイルでは、次のように TensorFlow Lite モジュールをインポートします。

```swift
import TensorFlowLite
```

Objective-C ファイルでは、次のようにアンブレラヘッダーをインポートします。

```objectivec
#import "TFLTensorFlowLite.h"
```

または、Xcode プロジェクトに `CLANG_ENABLE_MODULES = YES` を設定している場合は、次のようにモジュールをインポートします。

```objectivec
@import TFLTensorFlowLite;
```

注意: CocoaPods 開発者が、Objective-C TensorFlow Lite モジュールのインポートを希望する場合は、`Podfile` に `use_frameworks!` も含める必要があります。
