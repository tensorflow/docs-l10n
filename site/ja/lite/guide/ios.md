# iOS クイックスタート

To get started with TensorFlow Lite on iOS, we recommend exploring the following example:

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/ios">iOS image classification example</a>

ソースコードの説明については、[TensorFlow Lite iOS 画像分類](https://github.com/tensorflow/examples/blob/master/lite/examples/image_classification/ios/EXPLORE_THE_CODE.md)もあわせてお読みください。

このサンプルアプリは、[画像分類](https://www.tensorflow.org/lite/models/image_classification/overview)を使用して、デバイスの背面カメラに取り込まれるものを継続的に分類し、最も確率の高い分類を表示します。ユーザーは、浮動小数点または[量子化](https://www.tensorflow.org/lite/performance/post_training_quantization)モデルの選択と推論を実施するスレッド数の選択を行なえます。

Note: Additional iOS applications demonstrating TensorFlow Lite in a variety of use cases are available in [Examples](https://www.tensorflow.org/lite/examples).

## TensorFlow Lite を Swift または Objective-C プロジェクトに追加する

TensorFlow Lite offers native iOS libraries written in [Swift](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/experimental/swift) and [Objective-C](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/experimental/objc). Start writing your own iOS code using the [Swift image classification example](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/ios) as a starting point.

The sections below demonstrate how to add TensorFlow Lite Swift or Objective-C to your project:

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

You can also specify a version constraint. For example, if you wish to depend on version 2.0.0, you can write the dependency as:

```ruby
pod 'TensorFlowLiteSwift', '~> 2.0.0'
```

This will ensure the latest available 2.x.y version of the `TensorFlowLiteSwift` pod is used in your app. Alternatively, if you want to depend on the nightly builds, you can write:

```ruby
pod 'TensorFlowLiteSwift', '~> 0.0.1-nightly'
```

ナイトリーバージョンに関して言えば、バイナリサイズを縮小するために、デフォルトで、[GPU](https://www.tensorflow.org/lite/performance/gpu) と [Core ML デリゲート](https://www.tensorflow.org/lite/performance/coreml_delegate)がポッドから除外されますが、次のように subspec を指定することでこれらを含めることができます。

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
      "//tensorflow/lite/experimental/swift:TensorFlowLite",
  ],
)
```

#### Objective-C

```python
objc_library(
  deps = [
      "//tensorflow/lite/experimental/objc:TensorFlowLite",
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

Note: For CocoaPods developers who want to import the Objective-C TensorFlow Lite module, you must also include `use_frameworks!` in your `Podfile`.
