# マイクロコントローラを使ってみる

このドキュメントは、マイクロコントローラを使用してモデルをトレーニングし、推論を実行する方法について説明します。

## Hello World の例

[Hello World](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/micro/examples/hello_world) の例は、マイクロコントローラ向け TensorFlow Lite を使用するための基本を説明するためのものです。サイン関数を複製するモデルをトレーニングして実行します。つまり、単一の数値を入力として受け取り、その数値の[サイン](https://en.wikipedia.org/wiki/Sine)値を出力します。マイクロコントローラにデプロイされると、その予測は LED を点滅させるか、アニメーションを制御するために使用されます。

エンドツーエンドのワークフローには、次の手順が含まれます。

1. [モデルをトレーニングする](#train-a-model) (Python): デバイス上で使用するためにモデルをトレーニング、変換、最適化するための jupyter ノートブック。
2. [推論を実行する](#run-inference) (C++ 11): [C++ライブラリ](library.md)を使用してモデルで推論を実行するエンドツーエンドの単体テスト。

## サポートされているデバイスを入手する

使用するサンプルアプリケーションは、次のデバイスでテストされています。

- [Arduino Nano 33 BLE Sense](https://store.arduino.cc/usa/nano-33-ble-sense-with-headers) (Arduino IDE を使用する)
- [SparkFun Edge](https://www.sparkfun.com/products/15170) (ソースから直接構築する)
- [STM32F746 Discovery kit](https://www.st.com/en/evaluation-tools/32f746gdiscovery.html) (Mbed を使用する)
- [Adafruit EdgeBadge](https://www.adafruit.com/product/4400) (Arduino IDE を使用する)
- [Adafruit TensorFlow Lite for Microcontrollers Kit](https://www.adafruit.com/product/4317) (Arduino IDE を使用する)
- [Adafruit Circuit Playground Bluefruit](https://learn.adafruit.com/tensorflow-lite-for-circuit-playground-bluefruit-quickstart?view=all) (Arduino IDE を使用する)
- [Espressif ESP32-DevKitC](https://www.espressif.com/en/products/hardware/esp32-devkitc/overview) (ESP IDF を使用する)
- [Espressif ESP-EYE](https://www.espressif.com/en/products/hardware/esp-eye/overview) (ESP IDF を使用する)

サポートされているプラットフォームの詳細については、[マイクロコントローラ向け TensorFlow Lite](index.md) をご覧ください。

## モデルをトレーニングする

注：このセクションをスキップして、サンプルコードに含まれているトレーニング済みモデルを使用することもできます。

各アプリケーション例には、`README.md`ファイルがあり、サポートされたプラットフォームへのデプロイの仕方を説明しています。

<a class="button button-primary" href="https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/micro/examples/hello_world/train/README.md">Hello World Training README.md</a>

## 推論を実行する方法

デバイスでモデルを実行するために、`README.md`の手順を説明します。

<a class="button button-primary" href="https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/micro/examples/hello_world/README.md">Hello World README.md</a>

以下の節は *Hello World* サンプルの [`hello_world_test.cc`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/micro/examples/hello_world/hello_world_test.cc)を見ていきます。 これは、マイクロコントローラ向けTensorFlow Liteを使って推論を実行する方法を実演します。

### ライブラリをインクルードする

この例では、モデルは正弦波関数を再現するように訓練されています。 １つの数を入力として、[正弦波](https://en.wikipedia.org/wiki/Sine)の数値を出力します。 マイクロコントローラにデプロイされると、その予測は、LEDを点滅させたりアニメーションを制御したりすることに使用されます。

```C++
#include "tensorflow/lite/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
```

- [`all_ops_resolver.h`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/micro/all_ops_resolver.h)モデルを実行するためにインタープリタが使用する演算を提供します。
- [`micro_error_reporter.h`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/micro/micro_error_reporter.h)はデバッグ情報を出力します。
- [`micro_interpreter.h`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/micro/micro_interpreter.h)にはモデルをロードして実行するためのコードが含まれています。
- [`schema_generated.h`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/schema/schema_generated.h)には、TensorFlow Lite [`FlatBuffer`](https://google.github.io/flatbuffers/)デルファイル形式のスキーマが含まれています。
- [`version.h`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/version.h)は TensorFlow Lite スキーマのバージョン情報を提供します。

### モデルをインクルードする

マイクロコントローラ向けTensorFlow Lite インタープリタは、モデルがC++配列で提供されることを期待しています。*Hellow World* サンプルでは、モデルは `sine_model_data.h` と `sine_model_data.cc` で定義されています。ヘッダーは以下の行で含まれます。

```C++
#include "tensorflow/lite/micro/examples/hello_world/sine_model_data.h"
```

### 単体テストを用意する

見ていくコードは単体テストで、それはマイクロコントローラ向けTensorFlow Liteフレームワークの単体テストフレームワークを使います。 フレームワークを読み込むため、以下のファイルをインクルードします。

```C++
#include "tensorflow/lite/micro/testing/micro_test.h"
```

テストは以下のマクロを使って定義されます。

```C++
TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(LoadModelAndPerformInference) {
  . // add code here
  .
}

TF_LITE_MICRO_TESTS_END
```

コードの残り部分は、モデルの読み込みと推論を実演します。

### ログ取得を準備する

ログ取得の準備をするために、`tflite::MicroErrorReporter` インスタンスへのポインタを持つ、`tflite::ErrorReporter` ポインタが作成されます。

```C++
tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;
```

この変数はインタープリタに渡され、ログに書くことを許可します。 マイクロコントローラはしばしばログ取得のさまざまな機構をもつので、`tflite::MicroErrorReporter` の実装は、 デバイス固有にカスタマイズされるように設計されています。

### モデルを読み込む

以下のコードでは、モデルは `char` 配列、つまり `sine_model_data.h` で宣言された `g_sine_model_data` からのデータを使って実体化されます。 モデルを検査し、そのスキーマ・バージョンが我々が使用しているバージョンと互換性があることを確認します。

```C++
const tflite::Model* model = ::tflite::GetModel(g_sine_model_data);
if (model->version() != TFLITE_SCHEMA_VERSION) {
  error_reporter->Report(
      "得られたモデルはスキーマ・バージョン%dであり、"
      "サポートされたバージョン%dと一致しません。",
      model->version(), TFLITE_SCHEMA_VERSION);
}
```

### 演算子リゾルバを実体化する

[`AllOpsResolver`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/micro/kernels/all_ops_resolver.h) インスタンスが宣言されています。これは、モデルで使用されている演算にアクセスするためにインタープリタが使います。

```C++
tflite::ops::micro::AllOpsResolver resolver;
```

`AllOpsResolver` は、マイクロコントローラ向けTensorFlow Liteで利用可能なすべての演算を読み込むため多くのメモリを使用します。 通常、モデルが必要とするのはこれらの演算のうちの一部のため、現実世界に適用する際には必要な演算のみを読み込むことが推奨されます。

これは別のクラス、`MicroMutableOpResolver` を使用して実施されます。 *Micro speech* [`micro_speech_test.cc`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/micro/examples/micro_speech/micro_speech_test.cc) の例で使い方を見ることができます。

### メモリを割り当てる

適当な量のメモリを入力、出力、そして中間配列に事前に割り当てる必要があります。 これは、`tensor_arena_size` の大きさの `uint8_t` 配列として提供されます。

```C++
const int tensor_arena_size = 2 * 1024;
uint8_t tensor_arena[tensor_arena_size];
```

要求される大きさは使用するモデルに依存し、実験によって決める必要があるかもしれません。

### インタープリタを実体化する

`tflite::MicroInterpreter` インスタンスを作成し、事前に作成した変数を渡します。

```C++
tflite::MicroInterpreter interpreter(model, resolver, tensor_arena,
                                     tensor_arena_size, error_reporter);
```

### テンソルを割り当てる

インタープリタに対し、 `tensor_arena` からモデルのテンソルにメモリを割り当てるように指示します。

```C++
interpreter.AllocateTensors();
```

### 入力の形を検証する

`MicroInterpreter` インスタンスは、`.input(0)` を呼ぶことで、モデルの入力テンソルへのポインタを提供します。 `0` は最初の（そして唯一の）入力テンソルであることを表します。

```C++
  // モデルの入力テンソルへポインタを取得する
  TfLiteTensor* input = interpreter.input(0);
```

このテンソルを検証し、形と型が期待したものであることを確認します。

```C++
// 入力は期待するプロパティを持つことを確認する
TF_LITE_MICRO_EXPECT_NE(nullptr, input);
// プロパティ "dims" はテンソルの形を教えてくれる。
// それは次元ごとに1つの要素を持つ。我々の入力は2次元のテンソルで1つの要素を含むので、
// "dims" の次元数は2であるべきである。
TF_LITE_MICRO_EXPECT_EQ(2, input->dims->size);
// 要素ごとの値は、対応するテンソルの長さを与える。
// 我々は2つのテンソルを期待する。（一方は他方に含まれる。）
TF_LITE_MICRO_EXPECT_EQ(1, input->dims->data[0]);
TF_LITE_MICRO_EXPECT_EQ(1, input->dims->data[1]);
// 入力は32bit浮動小数点数である。
TF_LITE_MICRO_EXPECT_EQ(kTfLiteFloat32, input->type);
```

enum値 `kTfLiteFloat32` は、TensorFlow Lite のデータ型のうちの一つへの参照であり、 [`common.h`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/c/common.h). で定義されています。

### 入力値を提供する

入力をモデルに提供するために、入力テンソルの内容を以下のとおり設定します。

```C++
input->data.f[0] = 0.;
```

この場合、`0` を表す浮動小数点数を入力しています。

### モデルを実行する

モデルを実行するために、 `tflite::MicroInterpreter` インスタンス上で `Invoke()` を呼びます。

```C++
TfLiteStatus invoke_status = interpreter.Invoke();
if (invoke_status != kTfLiteOk) {
  error_reporter->Report("Invoke failed\n");
}
```

戻り値 `TfLiteStatus` を確認でき、実行が成功したかどうか決定できます。 `TfLiteStatus` の取りうる値は、 [`common.h`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/c/common.h)で定義されており、 `kTfLiteOk` と `kTfLiteError` です。

下記コードは値が、推論がうまく実行されたことを意味する、`kTfLiteOk`であることを知らせています。

```C++
TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, invoke_status);
```

### 出力を取得する

モデルの出力テンソルは、`tflite::MicroIntepreter` 上で `output(0)` を呼ぶことで取得できます。 `0` は最初の（そして唯一の）出力テンソルであることを表します。

サンプルでは、モデルの出力は1つの2次元テンソルに含まれる1つの浮動小数点数です。

```C++
TfLiteTensor* output = interpreter.output(0);
TF_LITE_MICRO_EXPECT_EQ(2, output->dims->size);
TF_LITE_MICRO_EXPECT_EQ(1, input->dims->data[0]);
TF_LITE_MICRO_EXPECT_EQ(1, input->dims->data[1]);
TF_LITE_MICRO_EXPECT_EQ(kTfLiteFloat32, output->type);
```

出力テンソルから直接値を読むこともでき、それが期待するものであるか評価することもできます。

```C++
// テンソルから出力値を取得する
float value = output->data.f[0];
// 出力値が期待値から0.05以内であるかを検査する
TF_LITE_MICRO_EXPECT_NEAR(0., value, 0.05);
```

### 推論を再度実行する

コードの残りの部分は、推論をさらに何回も実行します。 インスタンス毎に、入力テンソルに値を割り当て、インタープリタを呼び、そして出力テンソルから結果を読み取ります。

```C++
input->data.f[0] = 1.;
interpreter.Invoke();
value = output->data.f[0];
TF_LITE_MICRO_EXPECT_NEAR(0.841, value, 0.05);

input->data.f[0] = 3.;
interpreter.Invoke();
value = output->data.f[0];
TF_LITE_MICRO_EXPECT_NEAR(0.141, value, 0.05);

input->data.f[0] = 5.;
interpreter.Invoke();
value = output->data.f[0];
TF_LITE_MICRO_EXPECT_NEAR(-0.959, value, 0.05);
```

### アプリケーションのコードを読む

この単体テストを一度ひととおり読み終えたら、 [`main_functions.cc`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/micro/examples/hello_world/main_functions.cc) にあるサンプルアプリケーションのコードを理解できるはずです。 同じような処理を行いますが、実行された推論の数に基づいて入力値を生成し、それからデバイス固有の関数を呼び、モデルの出力をユーザーに表示します。
