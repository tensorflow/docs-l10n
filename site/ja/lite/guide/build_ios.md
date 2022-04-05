# iOS 用の TensorFlow Lite を構築する

このドキュメントでは、TensorFlow Lite iOS ライブラリを独自に構築する方法について説明します。通常、TensorFlow Lite iOS ライブラリをローカルで構築する必要はありません。使用するだけであれば、TensorFlow Lite CocoaPods の構築済みの安定リリースまたはナイトリーリリースを使用することが最も簡単な方法といえます。iOS プロジェクトでこれらを使用する方法の詳細については、[iOS クイックスタート](ios.md)を参照してください。

## ローカルで構築する

TensorFlow Lite にローカルの変更を加え、iOS アプリでそれらの変更をテストする場合や提供されている動的フレームワークより静的フレームワークを使用したい場合などに、TensorFlow Lite のローカルビルドを使用する場合があります。TensorFlow Lite のユニバーサル iOS フレームワークをローカルで作成するには、macOS マシンで Bazel を使用して構築する必要があります。

### Xcode をインストールする

Xcode 8 以降と `xcode-select` を使用するツールをインストールしていない場合は、まずそれらをインストールする必要があります。

```sh
xcode-select --install
```

これが新規インストールの場合は、次のコマンドを使用して、すべてのユーザーの使用許諾契約に同意する必要があります。

```sh
sudo xcodebuild -license accept
```

### Bazel をインストールする

Bazel は TensorFlow の主要なビルドシステムです。[Bazel Web サイトの指示]に従って Bazel をインストールします。`tensorflow`リポジトリのルートの [`configure.py` ファイル]から `_TF_MIN_BAZEL_VERSION` または `_TF_MAX_BAZEL_VERSION` を選択します。

### WORKSPACE と .bazelrc の構成

ルート TensorFlow チェックアウトディレクトリで `./configure` スクリプトを実行し、iOS サポートで TensorFlow を構築するかどうかを尋ねられたら、「Yes」と答えます。

### TensorFlowLiteC 動的フレームワークの構築（推奨）

注意: このステップは、（1）アプリに Bazel を使用している場合、または（2）Swift または Objective-C API に対するローカルの変更のみをテストする場合は必要ありません。このような場合は、以下の[独自のアプリで使用する](#use_in_your_own_application)セクションに進んでください。

Bazel が iOS サポートで適切に設定されたら、次のコマンドで `TensorFlowLiteC` フレームワークを構築できます。

```sh
bazel build --config=ios_fat -c opt \
  //tensorflow/lite/ios:TensorFlowLiteC_framework
```

このコマンドは、TensorFlow ルートディレクトリにある `bazel-bin/tensorflow/lite/ios/` ディレクトリに `TensorFlowLiteC_framework.zip` ファイルを生成します。デフォルトでは、生成されたフレームワークには、armv7、arm64、x86_64（i386 は含まない）を含む「ファット」バイナリが含まれています。`--config=ios_fat`を指定するときに使用されるビルドフラグの完全なリストを確認するには、[`.bazelrc` ファイル]の iOS 構成セクションを参照してください。

### TensorFlowLiteC 静的フレームワークの構築

デフォルトでは、Cocoapod 経由でのみ動的フレームワークを配布していますが、代わりに静的フレームワークを使用する場合は、次のコマンドを使って `TensorFlowLiteC` 静的フレームワークを構築できます。

```
bazel build --config=ios_fat -c opt \
  //tensorflow/lite/ios:TensorFlowLiteC_static_framework
```

このコマンドは、`TensorFlowLiteC_static_framework.zip` というファイルを TensorFlow ルートディレクトリの下にある `bazel-bin/tensorflow/lite/ios/` ディレクトリに生成します。この静的フレームワークは、動的フレームワークとまったく同じ方法で使用することができます。

### TFLite フレームワークを選択的に構築する

選択的ビルドを使用すると、モデルセット内の未使用の演算を省略し、特定のモデルセットを実行するために必要な op カーネルのみを含むモデルセットのみをターゲットする、より小型のフレームワークを構築できます。コマンドは以下のようになります。

```sh
bash tensorflow/lite/ios/build_frameworks.sh \
  --input_models=model1.tflite,model2.tflite \
  --target_archs=x86_64,armv7,arm64
```

モデルに Select TensorFlow op が含まれている場合、上記のコマンドによって、TensorFlow Lite のビルトイン op とカスタム op 用に、静的フレームワーク `bazel-bin/tensorflow/lite/ios/tmp/TensorFlowLiteC_framework.zip` が生成されます。`--target_archs` フラグは、デプロイアーキテクチャの指定に使用できることに注意してください。

## 独自のアプリで使用する

### CocoaPods 開発者

TensorFlow Lite には 3 つの CocoaPods があります。

- `TensorFlowLiteSwift`: TensorFlow Lite の Swift API を提供します。
- `TensorFlowLiteObjC`: TensorFlow Lite の Objective-C API を提供します。
- `TensorFlowLiteC`: TensorFlow Lite コアランタイムを組み込み、上記の 2 つのポッドで使用されるベース C API を公開する共通ベースポッド。ユーザーが直接使用するためのものではありません。

開発者は、アプリの記述言語に基づいて、`TensorFlowLiteSwift` または `TensorFlowLiteObjC` ポッドのいずれかを選択する必要がありますが、両方を選択することはできません。TensorFlow Lite のローカルビルドを使用するための正確な手順は、ビルドする正確なパーツによって異なります。

#### ローカル Swift または Objective-C API の使用

CocoaPods を使用していて、TensorFlow Lite の [Swift API](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/experimental/swift) または [Objective-C API](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/experimental/objc) に対するローカルの変更のみをテストする場合は、以下の手順に従います。

1. `tensorflow`チェックアウトの Swift または Objective-C API に変更を加えます。

2. `TensorFlowLite(Swift|ObjC).podspec`ファイルを開き、<br> `s.dependency 'TensorFlowLiteC', "#{s.version}"` <br> を次のように変更します。<br> `s.dependency 'TensorFlowLiteC', "~> 0.0.1-nightly"` <br> これは、安定版ではなく、`TensorFlowLiteC` API の最新の利用可能なナイトリ―版 (毎晩 1 時から午前 4 時の間に構築される) API に対して Swift または Objective-C API を構築していることを確認するためです。安定版はローカルの `tensorflow` チェックアウトと比較して古くなっている可能性があります。または、独自のバージョンの `TensorFlowLiteC` を公開し、そのバージョンを使用することもできます (以下の[ローカル TensorFlow Lite コアの使用](#using_local_tensorflow_lite_core)セクションを参照してください)。

3. iOS プロジェクトの `Podfile` で、依存関係を次のように変更して、`tensorflow` ルートディレクトリへのローカルパスを指すようにします。<br>Swift の場合: <br> `pod 'TensorFlowLiteSwift', :path => '<your_tensorflow_root_dir>'` <br> Objective-C の場合: <br> `pod 'TensorFlowLiteObjC', :path => '<your_tensorflow_root_dir>'`

4. iOS プロジェクトのルートディレクトリからポッドインストールを更新します。<br> `$ pod update`

5. 生成されたワークスペース（`<project>.xcworkspace`）を再度開き、Xcode 内でアプリを再ビルドします。

#### ローカル TensorFlow Lite コアの使用

プライベート CocoaPods 仕様リポジトリをセットアップし、カスタム `TensorFlowLiteC` フレームワークをプライベートリポジトリに公開します。この [podspe ファイル](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/experimental/ios/TensorFlowLiteC.podspec)をコピーして、いくつかの値を変更できます。

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

独自の `TensorFlowLiteC.podspec` ファイルを作成したら、[プライベート CocoaPods の使用に関する指示]に従って、独自のプロジェクトで使用できます。また、`TensorFlowLite(Swift|ObjC).podspec` を変更して、カスタムの `TensorFlowLiteC` ポッドをポイントし、アプリプロジェクトで Swift または Objective-C ポッドを使用することもできます。

### Bazel 開発者

メインビルドツールとして Bazel を使用している場合は、`BUILD` ファイルのターゲットに `TensorFlow Lite` 依存関係を追加するだけです。

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

TensorFlow Lite の依存関係をプロジェクトに追加するには、CocoaPods または Bazel を使用することを強くお勧めします。手動で `TensorFlowLiteC` フレームワークを追加する場合は、`TensorFlowLiteC` フレームワークを埋め込みフレームワークとしてアプリプロジェクトに追加する必要があります。上記のビルドで生成された`TensorFlowLiteC_framework.zip` を解凍して、`TensorFlowLiteC.framework` ディレクトリを取得します。このディレクトリは、Xcode が理解できる実際のフレームワークです。

`TensorFlowLiteC.framework` を準備したら、まずそれを埋め込みバイナリとしてアプリターゲットに追加する必要があります。このための正確なプロジェクト設定セクションは、Xcode のバージョンによって異なる場合があります。

- Xcode 11: アプリターゲットのプロジェクトエディタの General タブに移動し、Franeworks, Libraries, and Embedded Contentセクションに `TensorFlowLiteC.framework` を追加します。
- Xcode 10 以前: アプリターゲットのプロジェクトエディタのGeneral タブに移動し、Embedded Binaries の下に `TensorFlow Lite C.framework` を追加します。フレームワークは、Linked Frameworks and Libraries セクションの下にも自動的に追加されます。

フレームワークを埋め込みバイナリとして追加すると、Xcode はフレームワークの親ディレクトリを含むように Build Settings タブの Framework Search Paths エントリも更新します。これが自動的に行われない場合は、`TensorFlowLiteC.framework` ディレクトリの親ディレクトリを手動で追加する必要があります。

これら 2 つの設定が完了すると、`TensorFlowLiteC.framework/Headers` ディレクトリにあるヘッダーファイルで定義された TensorFlow Lite の C API をインポートして呼び出すことができるようになります。


[Bazel Web サイトの指示]: https://docs.bazel.build/versions/master/install-os-x.html
[`.bazelrc` ファイル]: https://github.com/tensorflow/tensorflow/blob/master/.bazelrc
[`configure.py` ファイル]: https://github.com/tensorflow/tensorflow/blob/master/configure.py
[プライベート CocoaPods の使用に関する指示]: https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/objc