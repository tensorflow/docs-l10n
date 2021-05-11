# iOS 用の TensorFlow Lite を構築する

このドキュメントでは、TensorFlow Lite iOS ライブラリを独自に構築する方法について説明します。通常、TensorFlow Lite iOS ライブラリをローカルで構築する必要はありません。使用するだけであれば、TensorFlow Lite CocoaPods の構築済みの安定リリースまたはナイトリーリリースを使用することが最も簡単な方法といえます。iOS プロジェクトでこれらを使用する方法の詳細については、[iOS クイックスタート](ios.md)を参照してください。

## Building locally

In some cases, you might wish to use a local build of TensorFlow Lite, for example when you want to make local changes to TensorFlow Lite and test those changes in your iOS app or you prefer using static framework to our provided dynamic one. To create a universal iOS framework for TensorFlow Lite locally, you need to build it using Bazel on a macOS machine.

### Xcode をインストールする

If you have not already, you will need to install Xcode 8 or later and the tools using `xcode-select`:

```sh
xcode-select --install
```

これが新規インストールの場合は、次のコマンドを使用して、すべてのユーザーの使用許諾契約に同意する必要があります。

```sh
sudo xcodebuild -license accept
```

### Bazel をインストールする

Bazel is the primary build system for TensorFlow. Install Bazel as per the [instructions on the Bazel website](https://docs.bazel.build/versions/master/install-os-x.html). Make sure to choose a version between `_TF_MIN_BAZEL_VERSION` and `_TF_MAX_BAZEL_VERSION` in [`configure.py` file](https://github.com/tensorflow/tensorflow/blob/master/configure.py) at the root of `tensorflow` repository.

### WORKSPACE と .bazelrc の構成

Run the `./configure` script in the root TensorFlow checkout directory, and answer "Yes" when the script asks if you wish to build TensorFlow with iOS support.

### Build TensorFlowLiteC dynamic framework (recommended)

Note: This step is not necessary if (1) you are using Bazel for your app, or (2) you only want to test local changes to the Swift or Objective-C APIs. In these cases, skip to the [Use in your own application](#use_in_your_own_application) section below.

Once Bazel is properly configured with iOS support, you can build the `TensorFlowLiteC` framework with the following command.

```sh
bazel build --config=ios_fat -c opt \
  //tensorflow/lite/ios:TensorFlowLiteC_framework
```

This command will generate the `TensorFlowLiteC_framework.zip` file under `bazel-bin/tensorflow/lite/ios/` directory under your TensorFlow root directory. By default, the generated framework contains a "fat" binary, containing armv7, arm64, and x86_64 (but no i386). To see the full list of build flags used when you specify `--config=ios_fat`, please refer to the iOS configs section in the [`.bazelrc` file](https://github.com/tensorflow/tensorflow/blob/master/.bazelrc).

### Build TensorFlowLiteC static framework

By default, we only distribute the dynamic framework via Cocoapods. If you want to use the static framework instead, you can build the `TensorFlowLiteC` static framework with the following command:

```
bazel build --config=ios_fat -c opt \
  //tensorflow/lite/ios:TensorFlowLiteC_static_framework
```

The command will generate a file named `TensorFlowLiteC_static_framework.zip` under `bazel-bin/tensorflow/lite/ios/` directory under your TensorFlow root directory. This static framework can be used in the exact same way as the dynamic one.

## 独自のアプリで使用する

### CocoaPods 開発者

TensorFlow Lite には 3 つの CocoaPods があります。

- `TensorFlowLiteSwift`: TensorFlow Lite の Swift API を提供します。
- `TensorFlowLiteObjC`: TensorFlow Lite の Objective-C API を提供します。
- `TensorFlowLiteC`: TensorFlow Lite コアランタイムを組み込み、上記の 2 つのポッドで使用されるベース C API を公開する共通ベースポッド。ユーザーが直接使用するためのものではありません。

As a developer, you should choose either `TensorFlowLiteSwift` or `TensorFlowLiteObjC` pod based on the language in which your app is written, but not both. The exact steps for using local builds of TensorFlow Lite differ, depending on which exact part you would like to build.

#### ローカル Swift または Objective-C API の使用

CocoaPods を使用していて、TensorFlow Lite の [Swift API](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/experimental/swift) または [Objective-C API](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/experimental/objc) に対するローカルの変更のみをテストする場合は、以下の手順に従います。

1. `tensorflow`チェックアウトの Swift または Objective-C API に変更を加えます。

2. Open the `TensorFlowLite(Swift|ObjC).podspec` file, and update this line:
     `s.dependency 'TensorFlowLiteC', "#{s.version}"`
     to be:
     `s.dependency 'TensorFlowLiteC', "~> 0.0.1-nightly"`
     This is to ensure that you are building your Swift or Objective-C APIs against the latest available nightly version of `TensorFlowLiteC` APIs (built every night between 1-4AM Pacific Time) rather than the stable version, which may be outdated compared to your local `tensorflow` checkout. Alternatively, you could choose to publish your own version of `TensorFlowLiteC` and use that version (see [Using local TensorFlow Lite core](#using_local_tensorflow_lite_core) section below).

3. In the `Podfile` of your iOS project, change the dependency as follows to point to the local path to your `tensorflow` root directory.
     For Swift:
     `pod 'TensorFlowLiteSwift', :path => '<your_tensorflow_root_dir>'`
     For Objective-C:
     `pod 'TensorFlowLiteObjC', :path => '<your_tensorflow_root_dir>'`

4. iOS プロジェクトのルートディレクトリからポッドインストールを更新します。<br> `$ pod update`

5. Reopen the generated workspace (`<project>.xcworkspace`) and rebuild your app within Xcode.

#### ローカル TensorFlow Lite コアの使用

You can set up a private CocoaPods specs repository, and publish your custom `TensorFlowLiteC` framework to your private repo. You can copy this [podspec file](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/ios/TensorFlowLiteC.podspec) and modify a few values:

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

After creating your own `TensorFlowLiteC.podspec` file, you can follow the [instructions on using private CocoaPods](https://guides.cocoapods.org/making/private-cocoapods.html) to use it in your own project. You can also modify the `TensorFlowLite(Swift|ObjC).podspec` to point to your custom `TensorFlowLiteC` pod and use either Swift or Objective-C pod in your app project.

### Bazel 開発者

If you are using Bazel as the main build tool, you can simply add `TensorFlowLite` dependency to your target in your `BUILD` file.

Swift の場合:

```python
swift_library(
  deps = [
      "//tensorflow/lite/swift:TensorFlowLite",
  ],
)
```

Objective-C の場合:

```python
objc_library(
  deps = [
      "//tensorflow/lite/objc:TensorFlowLite",
  ],
)
```

アプリプロジェクトを構築すると、TensorFlow Lite ライブラリへの変更がすべてピックアップされ、アプリに構築されます。

### Xcode プロジェクト設定を直接変更する

It is highly recommended to use CocoaPods or Bazel for adding TensorFlow Lite dependency into your project. If you still wish to add `TensorFlowLiteC` framework manually, you'll need to add the `TensorFlowLiteC` framework as an embedded framework to your application project. Unzip the `TensorFlowLiteC_framework.zip` generated from the above build to get the `TensorFlowLiteC.framework` directory. This directory is the actual framework which Xcode can understand.

Once you've prepared the `TensorFlowLiteC.framework`, first you need to add it as an embedded binary to your app target. The exact project settings section for this may differ depending on your Xcode version.

- Xcode 11: Go to the 'General' tab of the project editor for your app target, and add the `TensorFlowLiteC.framework` under 'Frameworks, Libraries, and Embedded Content' section.
- Xcode 10 and below: Go to the 'General' tab of the project editor for your app target, and add the `TensorFlowLiteC.framework` under 'Embedded Binaries'. The framework should also be added automatically under 'Linked Frameworks and Libraries' section.

When you add the framework as an embedded binary, Xcode would also update the 'Framework Search Paths' entry under 'Build Settings' tab to include the parent directory of your framework. In case this does not happen automatically, you should manually add the parent directory of the `TensorFlowLiteC.framework` directory.

Once these two settings are done, you should be able to import and call the TensorFlow Lite's C API, defined by the header files under `TensorFlowLiteC.framework/Headers` directory.
