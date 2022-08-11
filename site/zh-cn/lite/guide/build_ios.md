# 为 iOS 构建 TensorFlow Lite 库

本文介绍如何自行构建 TensorFlow Lite iOS 库。通常，您不需要在本地构建 TensorFlow Lite iOS 库。如果您只是希望使用此库，最简单的方式是使用预构建的稳定版或每日构建版 TensorFlow Lite CocoaPods。有关在 iOS 项目中如何使用这些库的更多详细信息，请参阅 [iOS 快速入门](ios.md)。

## 构建

在某些情况下，您可能希望使用 TensorFlow Lite 的本地构建，例如，当您想对 TensorFlow Lite 进行本地更改并在 iOS 应用中测试这些更改，或者您希望使用静态框架代替我们提供的动态框架时。要在本地为 TensorFlow Lite 创建通用 iOS 框架，您需要在 macOS 计算机上使用 Bazel 进行构建。

### 安装 Xcode

如果尚未安装，则需要使用 `xcode-select` 安装 Xcode 8 或更高版本和工具：

```sh
xcode-select --install
```

如果是首次安装，您还需要使用以下命令接受面向所有用户的许可协议：

```sh
sudo xcodebuild -license accept
```

### 安装 Bazel

Bazel 是 TensorFlow 的主要构建系统。按照 [Bazel 网站上的说明](https://docs.bazel.build/versions/master/install-os-x.html)安装 Bazel。确保在 `tensorflow` 仓库根下的 [`configure.py` 文件](https://github.com/tensorflow/tensorflow/blob/master/configure.py)中选择一个介于 `_TF_MIN_BAZEL_VERSION` 到 `_TF_MAX_BAZEL_VERSION` 之间的版本。

### 配置工作区和 .bazelrc

运行 TensorFlow 根签出目录下的 `./configure` 脚本，在脚本询问您是否希望构建支持 iOS 的 TensorFlow 时，请选择“Yes”。

### 构建 TensorFlowLiteC 动态框架（推荐）

如果 (1) 您为您的应用使用 Bazel，或者 (2) 您仅希望测试对 Swift 或 Objective-C API 的本地变更，则可以不执行此步骤。在这些情况下，请跳到下面的[在您自己的应用中使用](#use_in_your_own_application)部分。

正确配置支持 iOS 的 Bazel 后，您可以使用以下命令构建 `TensorFlowLiteC` 框架。

```sh
bazel build --config=ios_fat -c opt --cxxopt=--std=c++17 \
  //tensorflow/lite/ios:TensorFlowLiteC_framework
```

此命令将在 TensorFlow 根目录的 `bazel-bin/tensorflow/lite/ios/` 目录下生成 `TensorFlowLiteC_framework.zip` 文件。默认情况下，生成的框架包含一个“胖”二进制文件，其中包含 armv7、arm64 和 x86_64（但不包含 i386）。要查看在指定 `--config=ios_fat` 时使用的构建标志的完整列表，请参阅 [`.bazelrc` 文件](https://github.com/tensorflow/tensorflow/blob/master/.bazelrc)中的 iOS 配置部分。

### 构建 TensorFlowLiteC 静态框架

默认情况下，我们仅通过 Cocoapods 分发动态框架。如果要改用静态框架，则可以使用以下命令构建 `TensorFlowLiteC` 静态框架：

```
bazel build --config=ios_fat -c opt --cxxopt=--std=c++17 \
  //tensorflow/lite/ios:TensorFlowLiteC_static_framework
```

此命令将在 TensorFlow 根目录的 `bazel-bin/tensorflow/lite/ios/` 目录下生成一个名为 `TensorFlowLiteC_static_framework.zip` 的文件。此静态框架的使用方式与动态框架完全相同。

### 有选择地构建 TFLite 框架

您可以使用选择性构建来构建仅针对一组模型的较小框架，这将跳过您的模型集中未使用的运算，并且只包括运行给定的一组模型所需的运算内核。命令如下：

```sh
bash tensorflow/lite/ios/build_frameworks.sh \
  --input_models=model1.tflite,model2.tflite \
  --target_archs=x86_64,armv7,arm64
```

以上命令将为 TensorFlow Lite 内置运算和自定义运算生成 静态框架 `bazel-bin/tensorflow/lite/ios/tmp/TensorFlowLiteC_framework.zip`；如果您的模型包含 Select TensorFlow 运算，还可以选择生成静态框架 `bazel-bin/tensorflow/lite/ios/tmp/TensorFlowLiteSelectTfOps_framework.zip`。请注意，`--target_archs` 标志可用于指定您的部署架构。

## 在您自己的应用中使用

### CocoaPods 开发者

有三个适用于 TensorFlow Lite 的 CocoaPods：

- `TensorFlowLiteSwift`：为 TensorFlow Lite 提供 Swift API。
- `TensorFlowLiteObjC`：为 TensorFlow Lite 提供 Objective-C API。
- `TensorFlowLiteC`：通用基础 Pod，它嵌入了 TensorFlow Lite 核心运行时，并公开了上面两个 Pod 使用的基础 C API。不适合由用户直接使用。

作为开发者，您应基于应用的编写语言选择 `TensorFlowLiteSwift` 或 `TensorFlowLiteObjC` Pod，而不应同时选择两者。使用 TensorFlow Lite 本地构建的确切步骤有所不同，具体取决于您想要构建的部分。

#### 使用本地 Swift 或 Objective-C API

如果您使用的是 CocoaPods，并且仅希望测试对 TensorFlow Lite 的 [Swift API](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/swift) 或 [Objective-C API](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/objc) 的某些本地变更，请按以下步骤操作。

1. 在 `tensorflow` 检出中对 Swift 或 Objective-C API 进行更改。

2. 打开 `TensorFlowLite(Swift|ObjC).podspec` 文件，并将以下行：<br> `s.dependency 'TensorFlowLiteC', "#{s.version}"` <br> 更新为：<br> `s.dependency 'TensorFlowLiteC', "~> 0.0.1-nightly"` <br> 这样做的目的是，确保您是根据 `TensorFlowLiteC` API 的最新可用 Nightly 版本（太平洋时间每天凌晨 1-4 点之间构建）而不是稳定版本来构建 Swift 或 Objective-C API，与本地 `tensorflow` 检出相比，后者可能已经过时。或者，您也可以选择发布自己的 `TensorFlowLiteC` 版本并使用该版本（请参阅下面的[使用本地 TensorFlow Lite 核心](#using_local_tensorflow_lite_core)部分）。

3. 在您的 iOS 项目的 `Podfile` 中，按如下所示更改依赖项，以指向 `tensorflow` 根目录的本地路径。<br>对于 Swift：<br> `pod 'TensorFlowLiteSwift', :path => '<your_tensorflow_root_dir>'` <br>对于 Objective-C：<br> `pod 'TensorFlowLiteObjC', :path => '<your_tensorflow_root_dir>'`

4. 从 iOS 项目根目录更新 pod 安装。<br> `$ pod update`

5. 重新打开生成的工作区 (`<project>.xcworkspace`)，然后在 Xcode 中重新构建您的应用。

#### 使用本地 TensorFlow Lite 核心

您可以设置一个专用的 CocoaPods 规范仓库，并将您的自定义 `TensorFlowLiteC` 框架发布到您的专用仓库中。您可以复制此 [podspec 文件](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/ios/TensorFlowLiteC.podspec)并修改一些值：

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

创建自己的 `TensorFlowLiteC.podspec` 文件后，您可以按照[使用私有 CocoaPods 的说明](https://guides.cocoapods.org/making/private-cocoapods.html)在您自己的项目中加以使用。此外，您还可以修改 `TensorFlowLite(Swift|ObjC).podspec` 以指向您的自定义 `TensorFlowLiteC` Pod，并在您的应用项目中使用 Swift 或 Objective-C Pod。

### Bazel 开发者

如果您将 Bazel 用作主要构建工具，则只需将 `TensorFlowLite` 依赖项添加到 `BUILD` 文件中的目标。

对于 Swift：

```python
swift_library(
  deps = [
      "//tensorflow/lite/swift:TensorFlowLite",
  ],
)
```

对于 Objective-C：

```python
objc_library(
  deps = [
      "//tensorflow/lite/objc:TensorFlowLite",
  ],
)
```

当您构建应用项目时，对 TensorFlow Lite 库的任何更改都将被获取并构建到您的应用中。

### 直接修改 Xcode 项目设置

强烈建议使用 CocoaPods 或 Bazel 将 TensorFlow Lite 依赖项添加到您的项目中。如果仍然希望手动添加 `TensorFlowLiteC` 框架，则需要将 `TensorFlowLiteC` 框架作为嵌入式框架添加到您的应用项目中。解压从上述构建生成的 `TensorFlowLiteC_framework.zip`，以获取 `TensorFlowLiteC.framework` 目录。此目录是 Xcode 可以理解的实际框架。

准备好 `TensorFlowLiteC.framework` 后，首先需要将其作为嵌入式二进制文件添加到您的应用目标中。具体的项目设置部分可能会有所不同，具体取决于您的 Xcode 版本。

- Xcode 11：针对您的应用目标，转到项目编辑器的 General 标签页，然后在 Frameworks, Libraries, and Embedded Content 部分下添加 `TensorFlowLiteC.framework`。
- Xcode 10 及更低版本：转到项目编辑器的 General 标签页以找到您的应用目标，然后在 Embedded Binaries 下添加 `TensorFlowLiteC.framework`。此框架也应在 Linked Frameworks and Libraries 部分下自动添加。

当您将框架作为嵌入式二进制文件添加时，Xcode 还会更新 Build Settings 标签页下的 Framework Search Paths 条目，以包括框架的父目录。如果这种情况未自动发生，则应手动添加 `TensorFlowLiteC.framework` 目录的父目录。

完成这两个设置后，您应当能够导入并调用 `TensorFlowLiteC.framework/Headers` 目录下的头文件定义的 TensorFlow Lite 的 C API。
