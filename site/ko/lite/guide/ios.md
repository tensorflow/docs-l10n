# iOS 빠른 시작

iOS에서 TensorFlow Lite를 시작하려면 다음 예제를 살펴볼 것을 권장합니다.

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/ios">iOS image classification example</a>

소스 코드에 대한 설명은 [TensorFlow Lite iOS 이미지 분류](https://github.com/tensorflow/examples/blob/master/lite/examples/image_classification/ios/EXPLORE_THE_CODE.md)를 읽어보아야 합니다.

This example app uses [image classification](https://www.tensorflow.org/lite/models/image_classification/overview) to continuously classify whatever it sees from the device's rear-facing camera, displaying the top most probable classifications. It allows the user to choose between a floating point or [quantized](https://www.tensorflow.org/lite/performance/post_training_quantization) model and select the number of threads to perform inference on.

Note: Additional iOS applications demonstrating TensorFlow Lite in a variety of use cases are available in [Examples](https://www.tensorflow.org/lite/examples).

## Add TensorFlow Lite to your Swift or Objective-C project

TensorFlow Lite offers native iOS libraries written in [Swift](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/swift) and [Objective-C](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/objc). Start writing your own iOS code using the [Swift image classification example](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/ios) as a starting point.

아래 섹션에서 TensorFlow Lite Swift 또는 Objective-C를 프로젝트에 추가하는 방법을 보여줍니다.

### CocoaPods 개발자

In your `Podfile`, add the TensorFlow Lite pod. Then, run `pod install`.

#### Swift

```ruby
use_frameworks!
pod 'TensorFlowLiteSwift'
```

#### Objective-C

```ruby
pod 'TensorFlowLiteObjC'
```

#### Specifying versions

`TensorFlowLiteSwift` 및 `TensorFlowLiteObjC` 포드 모두에 안정적인 릴리스와 야간 릴리스가 제공됩니다. 위의 예에서와 같이 버전 제약 조건을 지정하지 않으면 CocoaPods는 기본적으로 안정된 최신 릴리스를 가져옵니다.

You can also specify a version constraint. For example, if you wish to depend on version 2.0.0, you can write the dependency as:

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

This will allow you to use the latest features added to TensorFlow Lite. Note that once the `Podfile.lock` file is created when you run `pod install` command for the first time, the nightly library version will be locked at the current date's version. If you wish to update the nightly library to the newer one, you should run `pod update` command.

For more information on different ways of specifying version constraints, see [Specifying pod versions](https://guides.cocoapods.org/using/the-podfile.html#specifying-pod-versions).

### Bazel 개발자

`BUILD` 파일에서 `TensorFlowLite` 종속성을 대상에 추가합니다.

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

Alternatively, you can use [C API](https://www.tensorflow.org/code/tensorflow/lite/c/c_api.h) or [C++ API](https://tensorflow.org/lite/api_docs/cc)

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

### Import the library

For Swift files, import the TensorFlow Lite module:

```swift
import TensorFlowLite
```

For Objective-C files, import the umbrella header:

```objectivec
#import "TFLTensorFlowLite.h"
```

Or, the module if you set `CLANG_ENABLE_MODULES = YES` in your Xcode project:

```objectivec
@import TFLTensorFlowLite;
```

참고: Objective-C TensorFlow Lite 모듈을 가져오려는 CocoaPods 개발자의 경우, `Podfile`에 `use_frameworks!`도 포함해야 합니다.
