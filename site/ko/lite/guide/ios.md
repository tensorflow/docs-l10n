# iOS 빠른 시작

iOS에서 TensorFlow Lite를 시작하려면 다음 예제를 살펴볼 것을 권장합니다.

<a href="https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/ios">iOS 이미지 분류 예제</a>

소스 코드에 대한 설명은 TensorFlow Lite iOS 이미지 분류를 읽어보아야 합니다.

이 예제 앱은 이미지 분류를 사용하여 기기의 후면 카메라에서 보이는 내용을 지속적으로 분류하여 가장 가능성이 높은 분류를 표시합니다. 이를 통해 사용자는 부동 소수점 또는 양자화 모델 중에서 선택하고 추론을 수행할 스레드 수를 선택할 수 있습니다.

참고: 다양한 사용 사례에서 TensorFlow Lite의 사용을 시연하는 추가 iOS 애플리케이션을 예제에서 확인할 수 있습니다.

## Swift 또는 Objective-C 프로젝트에 TensorFlow Lite 추가하기

TensorFlow Lite는 Swift 및 Objective-C로 작성된 네이티브 iOS 라이브러리를 제공합니다. Swift 이미지 분류 예를 출발점으로 하여 고유한 iOS 코드 작성을 시작하세요.

아래 섹션에서 TensorFlow Lite Swift 또는 Objective-C를 프로젝트에 추가하는 방법을 보여줍니다.

### CocoaPods 개발자

Podfile에서 TensorFlow Lite 포드를 추가합니다. 그런 다음 pod install을 실행합니다.

#### Swift

```ruby
use_frameworks!
pod 'TensorFlowLiteSwift'
```

#### Objective-C

```ruby
pod 'TensorFlowLiteObjC'
```

#### 버전 지정하기

TensorFlowLiteSwift 및 TensorFlowLiteObjC 포드 모두에 안정적인 릴리스와 야간 릴리스가 제공됩니다. 위의 예에서와 같이 버전 제약 조건을 지정하지 않으면 CocoaPods는 기본적으로 안정된 최신 릴리스를 가져옵니다.

버전 제약 조건을 지정할 수도 있습니다. 예를 들어, 버전 2.0.0을 이용하려는 경우 종속성을 다음과 같이 작성할 수 있습니다.

```ruby
pod 'TensorFlowLiteSwift', '~> 2.0.0'
```

그러면 TensorFlowLiteSwift 포드의 최신 2.xy 버전이 앱에서 사용됩니다. 또는 야간 빌드를 사용하려는 경우, 다음과 같이 작성할 수 있습니다.

```ruby
pod 'TensorFlowLiteSwift', '~> 0.0.1-nightly'
```

2.4.0 버전과 최근 야간 버전부터 바이너리 크기를 줄이기 위해 기본적으로 GPU 및 Core ML delegate가 포드에서 제외되지만 하위 사양을 지정하여 포함할 수 있습니다.

```ruby
pod 'TensorFlowLiteSwift', '~> 0.0.1-nightly', :subspecs => ['CoreML', 'Metal']
```

이를 통해 TensorFlow Lite에 추가된 최신 특성을 사용할 수 있습니다. pod install 명령을 처음 실행할 때 Podfile.lock 파일이 생성되면 야간 라이브러리 버전이 현재 날짜 버전에서 잠긴다는 점에 유의해야 합니다. 야간 라이브러리를 최신 라이브러리로 업데이트하려면 pod update 명령을 실행해야 합니다.

버전 제약 조건을 지정하는 다양한 방법에 대한 자세한 설명은 포드 버전 지정하기를 참조하세요.

### Bazel 개발자

BUILD 파일에서 TensorFlowLite 종속성을 대상에 추가합니다.

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

또는, C API 또는 C++ API를 사용할 수 있습니다.

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

### 라이브러리 가져오기

Swift 파일의 경우, TensorFlow Lite 모듈을 가져옵니다.

```swift
import TensorFlowLite
```

Objective-C 파일의 경우, 마스터 헤더를 가져옵니다.

```objectivec
#import "TFLTensorFlowLite.h"
```

또는 Xcode 프로젝트에서 CLANG_ENABLE_MODULES = YES를 설정한 모듈의 경우는 다음과 같습니다.

```objectivec
@import TFLTensorFlowLite;
```

참고: Objective-C TensorFlow Lite 모듈을 가져오려는 CocoaPods 개발자의 경우, Podfile에 use_frameworks!도 포함해야 합니다.
