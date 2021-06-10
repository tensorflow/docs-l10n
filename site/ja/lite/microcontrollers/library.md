# C++ ライブラリを理解する

 TensorFlow Lite for Microcontrollers C++ ライブラリは [TensorFlow リポジトリ](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/experimental/micro)の一部です。 読みやすさ、修正しやさ、統合しやすさが考慮されており、徹底してテストすることで、標準の TensorFlow Lite との互換性を維持できるように設計されています。

以下のドキュメントでは、C++ ライブラリの基本構造の概要を説明し、独自のプロジェクト作成に関する情報を提供します。

## ファイル構造

[`micro`](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/experimental/micro) ルートディレクトリの構造は比較的単純ですが、ルートディレクトリは広範に渡る TensorFlow リポジトリの中に配置されているため、関連するソースファイルを提供するスクリプトと事前生成済みのプロジェクトファイルを、さまざまな組み込み開発環境内の別の場所に作成しました。

### 重要なファイル

TensorFlow Lite for Microcontrollers インタープリタを使用する上で最も重要なファイルは、プロジェクトのルートに配置されており、テストが付属しています。

- [`all_ops_resolver.h`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/micro/kernels/all_ops_resolver.h) または [`micro_mutable_op_resolver.h`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/micro/micro_mutable_op_resolver.h) は、モデルを実行するインタープリタによって使用される演算を提供します。 `all_ops_resolver.h` はすべての利用可能な演算を含んでいるため大量のメモリを使用してしまいます。本番アプリケーションにおいては、モデルが必要とする演算のみを読み込む `micro_mutable_op_resolver.h` を使用することをお勧めします。
- [`micro_error_reporter.h`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/micro/micro_error_reporter.h) はデバッグ情報を出力します。
- [`micro_interpreter.h`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/micro/micro_interpreter.h) にはモデルを処理して実行するコードが含まれます。

一般的な使用方法のウォークスルーについては、[マイクロコントローラの基礎](get_started_low_level.md)をご覧ください。

ビルドシステムは、特定のファイルのプラットフォーム固有の実装を提供します。これらは、[`sparkfun_edge`](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/micro/sparkfun_edge) のようなプラットフォーム名をもつディレクトリに配置されています。

その他、次のようなディレクトリがあります。

- [`kernel`](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/micro/kernels): 演算の実装と関連するコードが含まれます。
- [`tools`](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/micro/tools): ビルドツールとその出力が含まれます。
- [`examples`](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/micro/examples): サンプルコードが含まれます。

## 新規プロジェクトを始める

新しいプロジェクトには、*Hello World* サンプルをテンプレートととして使うことをお勧めします。このセクションの手順に従って、プラットフォームに適したバージョンのサンプルを入手してください。

### Arduino ライブラリを使用する

Arduino を使用している場合、*Hello World* サンプルは、`Arduino_TensorFlowLite` Arduino ライブラリに含まれています。このライブラリは、Arduino IDE または [Arduino Create](https://create.arduino.cc/) からダウンロードできます。

ライブラリが一度追加されたら、`ファイル -> スケッチ例` を選択します。リストの下の方に `TensorFlowLite:hello_world` というサンプルをみつけられるはずです。 それを選択し、`hello_world` をクリックし、サンプルを読み込みます。サンプルのコピーを保存し、あなた自身のプロジェクトの基礎として使用することができます。

### ほかのプラットフォーム用のプロジェクトを生成する

マイクロコントローラ向け TensorFlow Lite は、スタンドアロンなプロジェクトを生成でき、それは必要なソースファイルをすべて含み、`Makefile` を使用します。現在サポートされている環境は、Keil、Make、そしてMbedです。

これらのプロジェクトをMakeと一緒に生成するために、[TensorFlow repository](http://github.com/tensorflow/tensorflow) をクローンし、以下のコマンドを実行します:

```bash
make -f tensorflow/lite/micro/tools/make/Makefile generate_projects
```

これには数分かかります。理由は、依存関係のためにいくつかの大きなツールチェーンをダウンロードする必要があるからです。 一度それが終わると、 `tensorflow/lite/micro/tools/make/gen/linux_x86_64/prj/` （正確なパスはホストのオペレーティングシステムに依存します） のようなパスの中にいくつかのフォルダが生成されていることがわかるはずです。 これらのフォルダは、生成されたプロジェクトとソースファイルを含みます。

コマンド実行後、*Hello World* プロジェクトを `tensorflow/lite/micro/tools/make/gen/linux_x86_64/prj/hello_world` にみつけられるでしょう。 たとえば、`hello_world/keil` は Keil プロジェクトを含むでしょう。

## テストを実行する

ライブラリをビルドし、すべての単体テストを実行するためには、以下のコマンドを使用します:

```bash
make -f tensorflow/lite/micro/tools/make/Makefile test
```

個々のテストを実行するには、以下のコマンドを使用します。`<test_name>` はテスト名に置き換えます。

```bash
make -f tensorflow/lite/micro/tools/make/Makefile test_<test_name>
```

プロジェクトの Makefile からテスト名をみつけられるでしょう。たとえば、`examples/hello_world/Makefile.inc` は *Hello World*  サンプルのテスト名を指定しています。

## バイナリをビルドする

得られたプロジェクト（サンプルアプリケーションのような）の実行可能なバイナリをビルドするために、以下のコマンドを使用します。`<project_name>` はビルドしたいプロジェクトに置き換えます。

```bash
make -f tensorflow/lite/micro/tools/make/Makefile <project_name>_bin
```

たとえば、以下のコマンドは、*Hello World* アプリケーションのバイナリをビルドします。

```bash
make -f tensorflow/lite/micro/tools/make/Makefile hello_world_bin
```

デフォルトでは、プロジェクトはホストのオペレーティングシステム用にコンパイルされるでしょう。異なるターゲットアーキテクチャを指定するには、`TARGET=` を使用します。 以下のサンプルは、*Hello World* サンプルを SparkFun Edge 向けにビルドする方法を示しています。

```bash
make -f tensorflow/lite/micro/tools/make/Makefile TARGET=sparkfun_edge hello_world_bin
```

ターゲットが指定されると、利用可能なターゲット固有のソースファイルが、オリジナルのコードと置き換えられて使用されるでしょう。 たとえば、サブディレクトリ `examples/hello_world/sparkfun_edge` は `constants.cc` と `output_handler.cc` ファイルの SparkFun Edge 向けの実装を含んでおり、 ターゲット `sparkfun_edge` が指定されたときに使用されるでしょう。

プロジェクトの Makefile にプロジェクト名をみつけられるでしょう。たとえば、`examples/hello_world/Makefile.inc` は、*Hello World* サンプルのバイナリ名を指定しています。

## 最適化されたカーネル

`tensorflow/lite/micro/kernels` のルートにあるリファレンス・カーネルは、純粋なC/C++で実装されており、プラットフォーム固有のハードウェア最適化は含みません。

最適化されたカーネルは、サブディレクトリに提供されています。 たとえば、`kernels/cmsis-nn` はいくつかの最適化されたカーネルを含み、Arm のCMSIS-NN ライブラリを使用します。

最適化されたカーネルを使ったプロジェクトを生成するためには、以下のコマンドを使用します。 `<subdirectory_name>` は最適化を含むサブディレクトリの名前に置き換えます。

```bash
make -f tensorflow/lite/micro/tools/make/Makefile TAGS=<subdirectory_name> generate_projects
```

新しいサブフォルダーを作成することで、あなた自身による最適化を追加することができます。我々は、最適化された新しい実装について、プルリクエストを奨励しています。

## Arduino ライブラリを生成する

Arduino ライブラリのナイトリー・ビルドは、Arduino IDE のライブラリマネージャから利用できます。

ライブラリの新規ビルドを生成する必要があるなら、TensorFlow レポジトリから以下のスクリプトを実行できます:

```bash
./tensorflow/lite/micro/tools/ci_build/test_arduino.sh
```

結果として生成されたライブラリは以下にみつけられます。 `tensorflow/lite/micro/tools/make/gen/arduino_x86_64/prj/tensorflow_lite.zip`

## 新規デバイスに移植する

新規プラットフォームやデバイスへ マイクロコントローラ向け TensorFlow Lite を移植をするためのガイドは [`micro/README.md`](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/micro/README.md) にあります。
