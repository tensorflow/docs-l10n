# iOS용 TensorFlow Lite 빌드하기

이 문서에서는 TensorFlow Lite iOS 라이브러리를 직접 빌드하는 방법을 설명합니다. 일반적으로, TensorFlow Lite iOS 라이브러리를 로컬로 빌드할 필요는 없지만, 그래야 한다면 TensorFlow Lite CocoaPod의 사전 빌드된 안정적 릴리스 또는 야간 릴리스를 사용하는 것이 가장 쉬운 방법입니다. iOS 프로젝트에서 사용하는 방법에 대한 자세한 내용은 [iOS 빠른 시작](ios.md)을 참조하세요.

## 로컬로 구축하기

경우에 따라 TensorFlow Lite의 로컬 빌드를 사용해야 할 때가 있습니다(예: TensorFlow Lite를 로컬로 변경하고 iOS 앱에서 해당 변경을 테스트하거나 제공된 동적 프레임워크 대신 정적 프레임워크를 사용하려는 경우). TensorFlow Lite용 범용 iOS 프레임워크를 로컬로 생성하려면 macOS 시스템에서 Bazel을 사용하여 빌드해야 합니다.

### Xcode 설치하기

아직 설치하지 않았다면 `xcode-select`를 사용하여 Xcode 8 이상과 도구를 설치해야 합니다.

```sh
xcode-select --install
```

새로 설치하는 경우, 다음 명령을 사용하여 모든 사용자에 대한 라이선스 계약에 동의해야 합니다.

```sh
sudo xcodebuild -license accept
```

### Bazel 설치하기

Bazel은 TensorFlow의 기본 빌드 시스템입니다. [Bazel 웹 사이트의 지침](https://docs.bazel.build/versions/master/install-os-x.html)에 따라 Bazel을 설치합니다. `tensorflow` 리포지토리 루트에 있는 [`configure.py` 파일](https://github.com/tensorflow/tensorflow/blob/master/configure.py)에서 `_TF_MIN_BAZEL_VERSION` 및 `_TF_MAX_BAZEL_VERSION` 사이의 버전을 선택해야 합니다.

### WORKSPACE 및 .bazelrc 구성하기

루트 TensorFlow 체크아웃 디렉토리에서 `./configure` 스크립트를 실행하고, 스크립트에서 iOS 지원을 통해 TensorFlow를 빌드할 것인지 묻는 메시지가 표시되면 "Yes"로 답합니다.

### TensorFlowLiteC 동적 프레임워크 빌드하기(권장)

참고: (1) 앱에 Bazel을 사용 중이거나 (2) Swift 또는 Objective-C API에 대한 로컬 변경 사항만 테스트하려는 경우에는 이 단계가 필요하지 않습니다. 이러한 경우, 아래 [고유 애플리케이션에서 사용하기](#use_in_your_own_application) 섹션으로 건너뛰세요.

Bazel이 iOS 지원으로 올바르게 구성되면 다음 명령을 사용하여 `TensorFlowLiteC` 프레임워크를 빌드할 수 있습니다.

```sh
bazel build --config=ios_fat -c opt \
  //tensorflow/lite/ios:TensorFlowLiteC_framework
```

이 명령으로 TensorFlow 루트 디렉터리 아래의 `bazel-bin/tensorflow/lite/experimental/ios/` 디렉터리에 `TensorFlowLiteC_framework.zip` 파일이 생성됩니다. 기본적으로, 생성된 프레임워크에는 armv7, arm64 및 x86_64(i386 제외)가 들어 있는 "뚱뚱한" 바이너리가 포함됩니다. `--config=ios_fat`를 지정할 때 사용되는 빌드 플래그의 전체 목록을 보려면 [`.bazelrc` 파일](https://github.com/tensorflow/tensorflow/blob/master/.bazelrc)의 iOS 구성 섹션을 참조하세요.

### TensorFlowLiteC 정적 프레임워크 빌드하기

기본적으로 Cocoapods를 통해서만 동적 프레임워크를 배포합니다. 대신 정적 프레임워크를 사용하려는 경우 다음 명령으로 `TensorFlowLiteC` 정적 프레임워크를 빌드할 수 있습니다.

```
bazel build --config=ios_fat -c opt \
  //tensorflow/lite/ios:TensorFlowLiteC_static_framework
```

이 명령은 TensorFlow 루트 디렉터리 아래의 `bazel-bin/tensorflow/lite/ios/` 디렉터리에 `TensorFlowLiteC_static_framework.zip`이라는 파일을 생성합니다. 이 정적 프레임워크는 동적 프레임워크와 똑같은 방식으로 사용할 수 있습니다.

### Selectively build TFLite frameworks

You can build smaller frameworks targeting only a set of models using selective build, which will skip unused operations in your model set and only include the op kernels required to run the given set of models. The command is as following:

```sh
bash tensorflow/lite/ios/build_frameworks.sh \
  --input_models=model1.tflite,model2.tflite \
  --target_archs=x86_64,armv7,arm64
```

The above command will generate the static framework `bazel-bin/tensorflow/lite/ios/tmp/TensorFlowLiteC_framework.zip` for TensorFlow Lite built-in and custom ops; and optionally, generates the static framework `bazel-bin/tensorflow/lite/ios/tmp/TensorFlowLiteSelectTfOps_framework.zip` if your models contain Select TensorFlow ops. Note that the `--target_archs` flag can be used to specify your deployment architectures.

## 고유 애플리케이션에서 사용하기

### CocoaPod 개발자

TensorFlow Lite용 CocoaPod는 3가지가 있습니다.

- `TensorFlowLiteSwift`: TensorFlow Lite용 Swift API를 제공합니다.
- `TensorFlowLiteObjC`: TensorFlow Lite용 Objective-C API를 제공합니다.
- `TensorFlowLiteC`: TensorFlow Lite 코어 런타임을 포함하고 위의 두 포드에서 사용하는 기본 C API를 노출하는 공통 기본 포드입니다. 사용자가 직접 사용할 수 없습니다.

개발자는 앱 작성에 이용된 언어에 따라 `TensorFlowLiteSwift` 또는 `TensorFlowLiteObjC` 포드를 선택해야 합니다. TensorFlow Lite의 로컬 빌드를 사용하기 위한 정확한 단계는 빌드하려는 특정 부분에 따라 다릅니다.

#### 로컬 Swift 또는 Objective-C API 사용하기

CocoaPod를 사용 중이고 TensorFlow Lite의 [Swift API](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/experimental/swift) 또는 [Objective-C API](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/experimental/objc)에 대한 일부 로컬 변경만 테스트하려면 여기에 있는 단계를 따르세요.

1. `tensorflow` 체크아웃에서 Swift 또는 Objective-C API를 변경합니다.

2. `TensorFlowLite(Swift|ObjC).podspec` 파일을 열고 다음 줄을 업데이트합니다. <br> `s.dependency 'TensorFlowLiteC', "#{s.version}"`.<br>그러면 다음과 같이 됩니다.<br> `s.dependency 'TensorFlowLiteC', "~> 0.0.1-nightly"`<br> 이는 로컬 `tensorflow` 체크아웃에 비해 오래되었을 수 있는 안정적 버전이 아닌 `TensorFlowLiteC` API의 사용 가능한 최신 야간 버전(태평양 표준시로 매일 밤 1~4시 사이에 빌드됨)을 사용하여 Swift 또는 Objective-C API를 빌드하기 위한 것입니다. 또는, `TensorFlowLiteC`의 고유 버전을 게시하고 해당 버전을 사용하도록 선택할 수 있습니다(아래의 [로컬 TensorFlow Lite 코어 사용하기](#using_local_tensorflow_lite_core) 섹션 참조).

3. iOS 프로젝트의 `Podfile`에서 다음과 같이 종속성을 변경하여 `tensorflow` 루트 디렉토리의 로컬 경로를 가리킵니다. <br> Swift의 경우: <br> `pod 'TensorFlowLiteSwift', :path => '<your_tensorflow_root_dir>'` <br> Objective-C의 경우: <br> `pod 'TensorFlowLiteObjC', :path => '<your_tensorflow_root_dir>'`

4. iOS 프로젝트 루트 디렉토리에서 포드 설치를 업데이트합니다. <br> `$ pod update`

5. 생성된 workspace를 다시 열고(`<project>.xcworkspace`) Xcode 내에서 앱을 다시 빌드합니다.

#### 로컬 TensorFlow Lite 코어 사용하기

비공개 CocoaPod 사양 리포지토리를 설정하고 사용자 정의 `TensorFlowLiteC` 프레임워크를 비공개 리포지토리에 게시할 수 있습니다. 이 [podspec 파일](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/experimental/ios/TensorFlowLiteC.podspec)을 복사하고 몇 가지 값을 수정할 수 있습니다.

```ruby
  ...
  s.version      = <your_desired_version_tag>
  ...
  # Note the `///`, two from the `file://` and one from the `/path`.
  s.source       = { :http => "file:///path/to/TensorFlowLiteC_framework.zip" }
  ...
  s.vendored_frameworks = 'TensorFlowLiteC.framework'
  ...
```

고유한 `TensorFlowLiteC.podspec` 파일을 만든 후, [비공개 CocoaPod 사용 지침](https://guides.cocoapods.org/making/private-cocoapods.html)에 따라 고유 프로젝트에서 이 파일을 사용할 수 있습니다. `TensorFlowLite(Swift|ObjC).podspec`을 수정하여 사용자 정의 `TensorFlowLiteC` 포드를 가리키고 앱 프로젝트에서 Swift 또는 Objective-C 포드를 사용할 수도 있습니다.

### Bazel developers

Bazel을 기본 빌드 도구로 사용하는 경우, `BUILD` 파일의 대상에 `TensorFlowLite` 종속성을 추가하기만 하면 됩니다.

Swift의 경우:

```python
swift_library(
  deps = [
      "//tensorflow/lite/swift:TensorFlowLite",
  ],
)
```

Objective-C의 경우:

```python
objc_library(
  deps = [
      "//tensorflow/lite/objc:TensorFlowLite",
  ],
)
```

앱 프로젝트를 빌드할 때 TensorFlow Lite 라이브러리의 모든 변경 사항이 선택되어 앱으로 빌드됩니다.

### Xcode 프로젝트 설정을 직접 수정하기

TensorFlow Lite 종속성을 프로젝트에 추가하려면 CocoaPod 또는 Bazel을 사용하는 것이 좋습니다. 그래도 `TensorFlowLiteC` 프레임워크를 수동으로 추가하려면 `TensorFlowLiteC` 프레임워크를 애플리케이션 프로젝트에 포함된 프레임워크로 추가해야 합니다. 위 빌드에서 생성된 `TensorFlowLiteC_framework.zip`의 압축을 풀어 `TensorFlowLiteC.framework` 디렉토리를 가져옵니다. 이 디렉토리는 Xcode가 인식할 수 있는 실제 프레임워크입니다.

`TensorFlowLiteC.framework`를 준비했으면 먼저 포함된 바이너리로 이를 앱 대상에 추가해야 합니다. 이를 위한 정확한 프로젝트 설정 섹션은 Xcode 버전에 따라 다를 수 있습니다.

- Xcode 11: 앱 대상에 대한 프로젝트 편집기의 'General' 탭으로 이동하고 'Frameworks, Libraries, and Embedded Content' 섹션 아래에 `TensorFlowLiteC.framework`를 추가합니다.
- Xcode 10 이하: 앱 대상에 대한 프로젝트 편집기의 'General' 탭으로 이동하고 'Embedded Binaries' 아래에 `TensorFlowLiteC.framework`를 추가합니다. 프레임워크는 'Linked Frameworks and Libraries' 섹션에도 자동으로 추가됩니다.

프레임워크를 포함된 바이너리로 추가하면 Xcode는 프레임워크의 상위 디렉토리를 포함하도록 'Build Settings' 탭 아래의 'Framework Search Paths' 항목도 업데이트합니다. 이 업데이트가 자동으로 이루어지지 않을 경우에는 `TensorFlowLiteC.framework` 디렉토리의 상위 디렉토리를 수동으로 추가해야 합니다.

이 두 가지 설정이 완료되면 `TensorFlowLiteC.framework/Headers` 디렉토리 아래의 헤더 파일에 정의된 TensorFlow Lite의 C API를 가져와 호출할 수 있습니다.
