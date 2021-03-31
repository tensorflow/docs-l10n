# iOS 快速入门

To get started with TensorFlow Lite on iOS, we recommend exploring the following example:

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/ios">iOS 图像分类示例</a>

For an explanation of the source code, you should also read [TensorFlow Lite iOS image classification](https://github.com/tensorflow/examples/blob/master/lite/examples/image_classification/ios/EXPLORE_THE_CODE.md).

此示例应用使用[图像分类](https://www.tensorflow.org/lite/models/image_classification/overview)持续对从设备的后置摄像头看到的内容进行分类，并显示最可能的分类。它允许用户在浮点或[量化](https://www.tensorflow.org/lite/performance/post_training_quantization)模型之间进行选择，并选择执行推断的线程数。

注：在各种用例中演示 TensorFlow Lite 的其他 iOS 应用可在[示例](https://www.tensorflow.org/lite/examples)中获得。

## 将 TensorFlow Lite 添加到您的 Swift 或 Objective-C 项目

TensorFlow Lite 提供用 [Swift](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/swift) 和 [Objective-C](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/objc) 编写的原生 iOS 库。将使用 [Swift 图像分类示例](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/ios)作为起点，开始编写自己的 iOS 代码。

The sections below demonstrate how to add TensorFlow Lite Swift or Objective-C to your project:

### CocoaPods 开发者

在 `Podfile` 中添加 TensorFlow Lite Pod。然后运行 `pod install`。

#### Swift

```ruby
use_frameworks!
pod 'TensorFlowLiteSwift'
```

#### Objective-C

```ruby
pod 'TensorFlowLiteObjC'
```

#### 指定版本

There are stable releases, and nightly releases available for both `TensorFlowLiteSwift` and `TensorFlowLiteObjC` pods. If you do not specify a version constraint as in the above examples, CocoaPods will pull the latest stable release by default.

You can also specify a version constraint. For example, if you wish to depend on version 2.0.0, you can write the dependency as:

```ruby
pod 'TensorFlowLiteSwift', '~> 2.0.0'
```

这将确保您的应用中使用的是 `TensorFlowLiteSwift` Pod 最新可用的 2.x.y 版本。或者，如果您想依赖每日构建版本，可以编写如下代码：

```ruby
pod 'TensorFlowLiteSwift', '~> 0.0.1-nightly'
```

从 2.4.0 版本和最新的 Nightly 版本开始，默认情况下，[GPU](https://www.tensorflow.org/lite/performance/gpu) 和 [Core ML 委托](https://www.tensorflow.org/lite/performance/coreml_delegate)将从 Pod 中排除，以减少二进制文件的大小。您可以通过指定 subspec 来包含它们：

```ruby
pod 'TensorFlowLiteSwift', '~> 0.0.1-nightly', :subspecs => ['CoreML', 'Metal']
```

这样，您将能够使用添加到 TensorFlow Lite 的最新功能。请注意，当您首次运行 `pod install` 命令创建 `Podfile.lock` 文件后，Nightly 库版本将被锁定为当前日期的版本。如果您希望将 Nightly 库版本更新为最新版本，应运行 `pod update` 命令。

有关使用不同方式来指定版本约束的更多信息，请参阅[指定 Pod 版本](https://guides.cocoapods.org/using/the-podfile.html#specifying-pod-versions)。

### Bazel 开发者

在 `BUILD` 文件中，将 `TensorFlowLite` 依赖项添加到目标。

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

或者，您可以使用 [C API](https://www.tensorflow.org/code/tensorflow/lite/c/c_api.h) 或 [C++ API](https://tensorflow.org/lite/api_docs/cc)

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

### 导入库

对于 Swift 文件，导入 TensorFlow Lite 模块：

```swift
import TensorFlowLite
```

对于 Objective-C 文件，导入包罗头：

```objectivec
#import "TFLTensorFlowLite.h"
```

或者，如果您在 Xcode 项目中设置了 `CLANG_ENABLE_MODULES = YES`，则导入模块：

```objectivec
@import TFLTensorFlowLite;
```

Note: For CocoaPods developers who want to import the Objective-C TensorFlow Lite module, you must also include `use_frameworks!` in your `Podfile`.
