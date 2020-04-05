# C++ライブラリを理解する

マイクロコントローラ向け TensorFlow Lite の C++ライブラリ は
[TensorFlow repository](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/experimental/micro)
の一部です。
読みやすく、修正しやすく、よく検証されており、統合しやすく、また、標準のTensorFlow Liteと互換性があるように設計されている。

以下の資料は、C++ライブラリの基本構造の概要を示しており、プロジェクトを作成に関する情報を提供する。

## ファイル構造

[`micro`](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/experimental/micro)
のルートディレクトリは、比較的簡単な構造です。
しかし、広大な TensorFlow レポジトリの中に配置されているので、スクリプトと生成済みプロジェクトファイルを作成しました。そのプロジェクトファイルは、さまざまな組込み開発環境の内部に分離して、関連のあるソースファイルを提供します。

### 重要なファイル

マイクロコントローラ向け TensorFlow Lite のインタープリタを使うために最も重要なファイルは、プロジェクトのルートに配置されており、テストが付属しています。

-   [`all_ops_resolver.h`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/micro/kernels/all_ops_resolver.h)
    または
    [`micro_mutable_op_resolver.h`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/micro/micro_mutable_op_resolver.h)
    は、モデルを実行するインタープリタによって使用される演算を提供します。
    `all_ops_resolver.h` は全ての利用可能な演算を引き入れるので、多くのメモリを使用します。製品アプリケーションにおいては、`micro_mutable_op_resolver.h` を使用するべきであり、モデルが必要とする演算のみを引き入れます。

-   [`micro_error_reporter.h`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/micro/micro_error_reporter.h)
    デバッグ情報を出力します。

-   [`micro_interpreter.h`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/micro/micro_interpreter.h)
    モデルを操作し、実行するコードを含みます。

典型的な使用方法は、[マイクロコントローラから始める](get_started.md) を見てみましょう。

ビルドシステムは、いくつかのファイルから成るプラットフォーム固有の実装を提供します。これらはプラットフォーム名のディレクトリに配置されており、例えば
[`sparkfun_edge`](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/micro/sparkfun_edge) があります。

いくつかの他のディレクトリが存在し、以下を含みます。

-   [`kernel`](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/micro/kernels)、
    操作の実装と関連するコードを含みます。
-   [`tools`](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/micro/tools)、
    ビルドツールとその出力を含みます。
-   [`examples`](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/micro/examples)、
    サンプルコードを含みます。

## 新規プロジェクトを始める

*Hello World* サンプルを新規プロジェクトのテンプレートととして使うことを推奨しています。
この節の以下の指示に従うことで、選択したプラットフォームのサンプルを得られます。

### Arduinoライブラリを使用する

Arduino を使用してるのならば、*Hello World* サンプルは、`Arduino_TensorFlowLite` Arduino ライブラリに含まれており、
Arduino IDE または [Arduino Create](https://create.arduino.cc/) からダウンロードできます。

ライブラリが一度追加されたら、`ファイル -> スケッチ例` を選択します。リストの下の方に `TensorFlowLite:hello_world` と言うサンプルを見つけられるはずです。それを選択し、`hello_world` をクリックし、サンプルを読み込みます。サンプルのコピーを保存し、あなた自身のプロジェクトの基礎として使用することができます。

### 他のプラットフォーム用のプロジェクトを生成する

マイクロコントローラ向け TensorFlow Lite は、スタンドアロンなプロジェクトを生成でき、それは必要なソースファイルを全て含み、`Makefile` を使用します。現在サポートされている環境は、Keil、Make、そしてMbedです。

これらのプロジェクトをMakeと一緒に生成するために、[TensorFlow repository](http://github.com/tensorflow/tensorflow) をクローンし、以下のコマンドを実行する。

```bash
make -f tensorflow/lite/micro/tools/make/Makefile generate_projects
```

これは数分かかります。理由は、依存関係のために幾つかの大きなツールチェーンをダウンロードする必要があるからです。
一度それが終わると、`tensorflow/lite/micro/tools/make/gen/linux_x86_64/prj/` （正確なパスはあなたのホストオペレーティングシステムに依存します）のようなパスの中にいくつかのフォルダが生成されていることがわかるはずです。これらのフォルダは、生成されたプロジェクトとソースファイルを含みます。

コマンド実行後、*Hello World* プロジェクトを `tensorflow/lite/micro/tools/make/gen/linux_x86_64/prj/hello_world` に見つけられるでしょう。例えば、`hello_world/keil` は Keilプロジェクトを含むでしょう。

## テストを実行する

ライブラリをビルドし、全ての単体テストを実行するためには、以下のコマンドを使用します。

```bash
make -f tensorflow/lite/micro/tools/make/Makefile test
```

個々のテストを実行するには、以下のコマンドを使用します。`<test_name>` はテスト名に置き換えます。

```bash
make -f tensorflow/lite/micro/tools/make/Makefile test_<test_name>
```

プロジェクトのMakefileからテスト名を見つけられるでしょう。例えば、`examples/hello_world/Makefile.inc` は *Hello World*  サンプルのテスト名を指定しています。

## バイナリをビルドする

得られたプロジェクト（サンプルアプリケーションのような）の実行可能なバイナリをビルドするために、以下のコマンドを使用します。`<project_name>` はビルドしたいプロジェクトに置き換えます。

```bash
make -f tensorflow/lite/micro/tools/make/Makefile <project_name>_bin
```

例えば、以下のコマンドは、*Hello World* アプリケーションのバイナリをビルドします。

```bash
make -f tensorflow/lite/micro/tools/make/Makefile hello_world_bin
```

デフォルトでは、プロジェクトはホストオペレーティングシステム用にコンパイルされるでしょう。異なるターゲットアーキテクチャを指定するには、`TARGET=` を使用します。以下のサンプルは、*Hello World* サンプルを SparkFun Edge 向けにビルドする方法を示しています。

```bash
make -f tensorflow/lite/micro/tools/make/Makefile TARGET=sparkfun_edge hello_world_bin
```

ターゲットが指定されると、利用可能なターゲット固有のソースファイルが、オリジナルのコードと置き換えられて使用されるでしょう。例えば、サブディレクトリ `examples/hello_world/sparkfun_edge` は `constants.cc` と `output_handler.cc` ファイルの SparkFun Edge 向けの実装を含んでおり、ターゲット `sparkfun_edge` が指定されたときに使用されるでしょう。

プロジェクトの Makefiles にプロジェクト名を見つけられるでしょう。例えば、`examples/hello_world/Makefile.inc`
は、*Hello World*  サンプルのバイナリ名を指定しています。

## 最適化されたカーネル

`tensorflow/lite/micro/kernels` のルートにある参考となるカーネルは、純粋なC/C++で実装されており、プラットフォーム固有のハードウェア最適化は含みません。

最適化されたバージョンのカーネルは、サブディレクトリに提供されています。例えば、`kernels/cmsis-nn` はいくつかの最適化されたカーネルを含み、ArmのCMSIS-NNライブラリを使用します。

最適化されたカーネルを使ったプロジェクトを生成するためには、以下のコマンドを使用します。
`<subdirectory_name>`  は最適化を含むサブディレクトリの名前に置き換えます。

```bash
make -f tensorflow/lite/micro/tools/make/Makefile TAGS=<subdirectory_name> generate_projects
```

新しいサブフォルダーを作成することで、あなた自身による最適化を追加することができます。我々は、新しい最適化実装へのプルリクエスト奨励しています。

## Arduino ライブラリを生成する

Arduinoライブラリのナイトリー・ビルドは、Arduino IDEのライブラリマネージャから利用できます。

ライブラリの新規ビルドを生成する必要があるなら、TensorFlowレポジトリから以下のスクリプトを実行できます。

```bash
./tensorflow/lite/micro/tools/ci_build/test_arduino.sh
```

結果として生成されたライブラリは以下で見つけられます。
`tensorflow/lite/micro/tools/make/gen/arduino_x86_64/prj/tensorflow_lite.zip`

## 新規デバイスに移植する

新規プラットフォームやデバイスへ マイクロコントローラ向けTensorFlow Lite の移植をするためのガイドは [`micro/README.md`](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/micro/README.md) で見つけれます。
