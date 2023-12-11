# C++ ライブラリを理解する

TensorFlow Lite for Microcontrollers C++ ライブラリは [TensorFlow リポジトリ](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro)の一部です。読みやすさ、変更しやさ、統合しやすさが考慮されており、徹底してテストすることで、標準の TensorFlow Lite との互換性を維持できるように設計されています。

以下のドキュメントでは、C++ ライブラリの基本構造の概要を説明し、独自のプロジェクト作成に関する情報を提供します。

## ファイル構造

[`micro`](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro) ルートディレクトリの構造は比較的単純ですが、ルートディレクトリは広範に渡る TensorFlow リポジトリの中に配置されているため、関連するソースファイルを提供するスクリプトと事前生成済みのプロジェクトファイルは、多様な組み込み開発環境内で隔離して作成しました。

### 重要なファイル

TensorFlow Lite for Microcontrollers インタプリタを使用する上で最も重要なファイルは、プロジェクトのルートにテストと共に配置されています。

```
[`micro_mutable_op_resolver.h`](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/micro_mutable_op_resolver.h)
can be used to provide the operations used by the interpreter to run the
model.
```

- [`micro_error_reporter.h`](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h) はデバッグ情報を出力します。
- [`micro_interpreter.h`](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/micro_interpreter.h) にはモデルを処理して実行するコードが含まれます。

一般的な使用方法のウォークスルーについては、[マイクロコントローラの基礎](get_started_low_level.md)をご覧ください。

ビルドシステムは、特定のファイルのプラットフォーム固有の実装を提供します。これらは、[`cortex-m`](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/cortex_m_generic) のようなプラットフォーム名をもつディレクトリに配置されています。

その他、次のようなディレクトリがあります。

- [`kernel`](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/kernels): 演算の実装と関連するコードが含まれます。
- [`tools`](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/tools): ビルドツールとその出力が含まれます。
- [`examples`](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/examples): サンプルコードが含まれます。

## 新規プロジェクトを始める

新しいプロジェクトには、*Hello World* サンプルをテンプレートととして使うことをお勧めします。このセクションの手順に従って、プラットフォームに適したバージョンのサンプルを入手してください。

### Arduino ライブラリを使用する

Arduino を使用している場合、*Hello World* の例は `Arduino_TensorFlowLite` Arduino ライブラリに含まれており、Arduino IDE および [Arduino Create](https://create.arduino.cc/) に手動でインストールできます。

ライブラリが追加されたら、`File -> Examples` を選択します。リストの下の方に `TensorFlowLite:hello_world` というサンプルがあります。それを選択し、`hello_world` をクリックしてサンプルを読み込みます。すると、サンプルのコピーを保存して独自のプロジェクトの基礎として使用できるようになります。

### ほかのプラットフォーム用のプロジェクトを生成する

TensorFlow Lite for Microcontrollers は、`Makefile` を使用して、必要なすべてのソースファイルを含むスタンドアロンのプロジェクトを生成することができます。現在サポートされている環境は、Keil、Make、および Mbed です。

これらのプロジェクトを Make で生成するには、[TensorFlow/tflite-micro リポジトリ](https://github.com/tensorflow/tflite-micro)をクローンして、以下のコマンドを実行します。

```bash
make -f tensorflow/lite/micro/tools/make/Makefile generate_projects
```

依存関係に必要となる大規模なツールチェーンをダウンロードする必要があるため、この作業には数分かかります。完了したら、`gen/linux_x86_64/prj/` といったパス（正確なパスはホストのオペレーティングシステムによって異なります）に複数のフォルダが生成されます。 これらのフォルダには生成されたプロジェクトとソースファイルが含まれます。

コマンドを実行すると、*Hello World* プロジェクトが `gen/linux_x86_64/prj/hello_world` に表示されます。たとえば、`hello_world/keil` には Keil プロジェクトが含まれています。

## テストを実行する

ライブラリをビルドしてすべてのユニットテストを実行するには、以下のコマンドを使用します。

```bash
make -f tensorflow/lite/micro/tools/make/Makefile test
```

テストを個別に実行するには、以下のコマンドを使用します。`<test_name>` をテストの名前に置き換えてください。

```bash
make -f tensorflow/lite/micro/tools/make/Makefile test_<test_name>
```

テスト名はプロジェクトの Makefile で確認できます。たとえば、`examples/hello_world/Makefile.inc` は *Hello World* サンプルのテスト名を示します。

## バイナリをビルドする

特定のプロジェクト（サンプルアプリケーションなど）の実行可能なバイナリをビルドするには、以下のコマンドを使用します。`<project_name>` をビルドするプロジェクトに置き換えてください。

```bash
make -f tensorflow/lite/micro/tools/make/Makefile <project_name>_bin
```

たとえば、以下のコマンドは、*Hello World* アプリケーションのバイナリをビルドします。

```bash
make -f tensorflow/lite/micro/tools/make/Makefile hello_world_bin
```

デフォルトでは、プロジェクトはホストのオペレーティングシステム用にコンパイルされます。別のターゲットアーキテクチャを指定するには、`TARGET=` と `TARGET_ARCH=` を使用します。 以下の例は、*Hello World* サンプルを一般的な cortex-m0 向けにビルドする方法を示しています。

```bash
make -f tensorflow/lite/micro/tools/make/Makefile TARGET=cortex_m_generic TARGET_ARCH=cortex-m0 hello_world_bin
```

ターゲットが指定されている場合、元のコードの代わりに、利用可能なターゲット固有のソースファイルが使用されます。たとえば、`examples/hello_world/cortex_m_generic` サブディレクトリには `constants.cc` ファイルと `output_handler.cc` ファイルの SparkFun Edge 向けの実装が含まれており、ターゲットに `cortex_m_generic` が指定された場合に使用されます。

プロジェクト名はプロジェクトの Makefile で確認できます。たとえば、`examples/hello_world/Makefile.inc` は *Hello World* サンプルのバイナリ名を示します。

## 最適化されたカーネル

`tensorflow/lite/micro/kernels` のルートにある参照カーネルは、純粋な C/C++ で実装されているため、プラットフォーム固有のハードウェア最適化は含まれていません。

最適化されたバージョンのカーネルは、サブディレクトリに提供されています。たとえば、`kernels/cmsis-nn` には Arm の CMSIS-NN ライブラリを使用する最適化された複数のカーネルが含まれています。

最適化されたカーネルを使ってプロジェクトを生成するには、以下のコマンドを使用します。`<subdirectory_name>` を最適化を含むサブディレクトリの名前に置き換えてください。

```bash
make -f tensorflow/lite/micro/tools/make/Makefile TAGS=<subdirectory_name> generate_projects
```

独自の最適化を追加するには、その最適化用の新しいサブフォルダを作成します。新しい最適化済みの実装についてはプルリクエストを作成してください。

## Arduino ライブラリを生成する

ライブラリの新規ビルドを生成する必要がある場合は、TensorFlow レポジトリから以下のスクリプトを実行できます。

```bash
./tensorflow/lite/micro/tools/ci_build/test_arduino.sh
```

生成されたライブラリは、`gen/arduino_x86_64/prj/tensorflow_lite.zip` にあります。

## 新規デバイスに移植する

TensorFlow Lite for Microcontrollers を新しいプラットフォームやデバイスに移植するためのガイドは、[`micro/docs/new_platform_support.md`](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/docs/new_platform_support.md) をご覧ください。
