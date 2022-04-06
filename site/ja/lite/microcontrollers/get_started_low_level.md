# マイクロコントローラを使ってみる

このドキュメントは、マイクロコントローラを使用してモデルをトレーニングし、推論を実行する方法について説明します。

## Hello World の例

[Hello World](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/examples/hello_world) の例は、マイクロコントローラ向け TensorFlow Lite を使用するための基本を説明するためのものです。サイン関数を複製するモデルをトレーニングして実行します。つまり、単一の数値を入力として受け取り、その数値の[サイン](https://en.wikipedia.org/wiki/Sine)値を出力します。マイクロコントローラにデプロイされると、その予測は LED を点滅させるか、アニメーションを制御するために使用されます。

エンドツーエンドのワークフローには、次の手順が含まれます。

1. [モデルをトレーニングする](#train_a_model) (Python): デバイス上で使用するためにモデルをトレーニング、変換、最適化するための jupyter ノートブック。
2. [Run inference](#run_inference) (in C++ 11): An end-to-end unit test that runs inference on the model using the [C++ library](library.md).

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

Google Colaboratory を使用して、[独自のモデルをトレーニング](https://colab.research.google.com/github/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/examples/hello_world/train/train_hello_world_model.ipynb)します。 詳細については、`README.md`を参照してください。

<a class="button button-primary" href="https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/examples/hello_world/train/README.md">Hello World Training README.md</a>

## 推論を実行する

デバイスでモデルを実行するために、`README.md`の手順を説明します。

<a class="button button-primary" href="https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/examples/hello_world/README.md">Hello World README.md</a>

以下のセクションは <em>Hello World</em> サンプルの <a><code>hello_world_test.cc</code></a>を見ていきます。 この単体テストでは、マイクロコントローラ向け TensorFlow Liteを使って推論を実行する方法を実演します。モデルを読み込み、推論を数回実行します。

### 1. ライブラリをインクルードする

この例では、モデルは正弦波関数を再現するようにトレーニングされています。１つの数を入力として、[正弦波](https://en.wikipedia.org/wiki/Sine)の数値を出力します。マイクロコントローラにデプロイされると、その予測は、LED　を点滅させたりアニメーションを制御したりすることに使用されます。

```C++
#include "tensorflow/lite/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
```

- [`all_ops_resolver.h`](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/all_ops_resolver.h)モデルを実行するためにインタープリタが使用する演算を提供します。
- [`micro_error_reporter.h`](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/micro_error_reporter.h)はデバッグ情報を出力します。
- [`micro_interpreter.h`](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/micro_interpreter.h)にはモデルをロードして実行するためのコードが含まれています。
- [`schema_generated.h`](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/schema/schema_generated.h)には、TensorFlow Lite [`FlatBuffer`](https://google.github.io/flatbuffers/)デルファイル形式のスキーマが含まれています。
- [`version.h`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/version.h)は TensorFlow Lite スキーマのバージョン情報を提供します。

### 2. モデルヘッダーをインクルードする

マイクロコントローラ向け TensorFlow Lite インタープリタは、モデルがC++配列で提供されることを期待しています。*Hellow World* サンプルでは、モデルは `sine_model_data.h`と`sine_model_data.cc`で定義されています。ヘッダーは以下の行で含まれます。

```C++
#include "tensorflow/lite/micro/examples/hello_world/sine_model_data.h"
```

### 3. 単体テストフレームワークヘッダーをインクルードする

見ていくコードは単体テストで、それはマイクロコントローラ向け TensorFlow Lite フレームワークの単体テストフレームワークを使います。フレームワークを読み込むため、以下のファイルをインクルードします。

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

### 4. ログ取得を準備する

ログ取得の準備をするために、`tflite::MicroErrorReporter`インスタンスへのポインタを持つ、`tflite::ErrorReporter`ポインタが作成されます。

```C++
tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;
```

この変数はインタープリタに渡され、ログに書くことを許可します。マイクロコントローラはしばしばログ取得のさまざまな機構をもつので、`tflite::MicroErrorReporter`の実装は、 デバイス固有にカスタマイズされるように設計されています。

### 5. モデルを読み込む

以下のコードでは、モデルは`char`配列、つまり`sine_model_data.h`で宣言された`g_sine_model_data`からのデータを使って実体化されます。モデルを検査し、そのスキーマ・バージョンが我々が使用しているバージョンと互換性があることを確認します。

```C++
const tflite::Model* model = ::tflite::GetModel(g_model);
if (model->version() != TFLITE_SCHEMA_VERSION) {
  TF_LITE_REPORT_ERROR(error_reporter,
      "Model provided is schema version %d not equal "
      "to supported version %d.\n",
      model->version(), TFLITE_SCHEMA_VERSION);
}
```

### 6. 演算子リゾルバを実体化する

[`AllOpsResolver`](github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/all_ops_resolver.h)インスタンスが宣言されています。これは、モデルで使用されている演算にアクセスするためにインタープリタが使います。

```C++
tflite::AllOpsResolver resolver;
```

`AllOpsResolver`は、マイクロコントローラ向けTensorFlow Lite で利用可能なすべての演算を読み込むため多くのメモリを使用します。通常、モデルが必要とするのはこれらの演算のうちの一部のため、現実世界に適用する際には必要な演算のみを読み込むことが推奨されます。

これは別のクラス、`MicroMutableOpResolver`を使用して実施されます。 *Micro speech* [`micro_speech_test.cc`](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/examples/micro_speech/micro_speech_test.cc)の例で使い方を見ることができます。

### 7.メモリを割り当てる

適当な量のメモリを入力、出力、そして中間配列に事前に割り当てる必要があります。これは、`tensor_arena_size`の大きさの`uint8_t`配列として提供されます。

```C++
const int tensor_arena_size = 2 * 1024;
uint8_t tensor_arena[tensor_arena_size];
```

要求される大きさは使用するモデルに依存し、実験によって決める必要があるかもしれません。

### 8. インタプリタをインスタンス化する

`tflite::MicroInterpreter`インスタンスを作成し、事前に作成した変数を渡します。

```C++
tflite::MicroInterpreter interpreter(model, resolver, tensor_arena,
                                     tensor_arena_size, error_reporter);
```

### 9. テンソルを割り当てる

インタープリタに対し、`tensor_arena`からモデルのテンソルにメモリを割り当てるように指示します。

```C++
interpreter.AllocateTensors();
```

### 10. 入力の形状を検証する

`MicroInterpreter`インスタンスは、`.input(0)`を呼ぶことで、モデルの入力テンソルへのポインタを提供します。 `0`は最初の（そして唯一の）入力テンソルであることを表します。

```C++
    // Obtain a pointer to the model's input tensor
  TfLiteTensor* input = interpreter.input(0);
```

このテンソルを検証し、形状と型が期待したものであることを確認します。

```C++
// Make sure the input has the properties we expect
TF_LITE_MICRO_EXPECT_NE(nullptr, input);
// The property "dims" tells us the tensor's shape. It has one element for
// each dimension. Our input is a 2D tensor containing 1 element, so "dims"
// should have size 2.
TF_LITE_MICRO_EXPECT_EQ(2, input->dims->size);
// The value of each element gives the length of the corresponding tensor.
// We should expect two single element tensors (one is contained within the
// other).
TF_LITE_MICRO_EXPECT_EQ(1, input->dims->data[0]);
TF_LITE_MICRO_EXPECT_EQ(1, input->dims->data[1]);
// The input is a 32 bit floating point value
TF_LITE_MICRO_EXPECT_EQ(kTfLiteFloat32, input->type);
```

enum値`kTfLiteFloat32`は、TensorFlow Lite のデータ型のうちの一つへの参照であり、 [`common.h`](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/c/common.h)で定義されています。

### 11. 入力値を提供する

入力をモデルに提供するために、入力テンソルの内容を以下のとおり設定します。

```C++
input->data.f[0] = 0.;
```

この場合、`0`を表す浮動小数点数を入力しています。

### 12. モデルを実行する

モデルを実行するために、`tflite::MicroInterpreter`インスタンス上で`Invoke()`を呼びます。

```C++
TfLiteStatus invoke_status = interpreter.Invoke();
if (invoke_status != kTfLiteOk) {
  error_reporter->Report("Invoke failed\n");
}
```

戻り値`TfLiteStatus`を確認でき、実行が成功したかどうか決定できます。`TfLiteStatus`の取りうる値は、[`common.h`](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/c/common.h)で定義されており、 `kTfLiteOk`と`kTfLiteError` です。

下記コードは値が、推論がうまく実行されたことを意味する、`kTfLiteOk`であることを知らせています。

```C++
TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, invoke_status);
```

### 13. 出力を取得する

モデルの出力テンソルは、`tflite::MicroIntepreter`上で`output(0)`を呼ぶことで取得できます。`0`は最初の（そして唯一の）出力テンソルであることを表します。

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
// Obtain the output value from the tensor
float value = output->data.f[0];
// Check that the output value is within 0.05 of the expected value
TF_LITE_MICRO_EXPECT_NEAR(0., value, 0.05);
```

### 14. 推論を再度実行する

コードの残りの部分は、推論をさらに何回も実行します。インスタンス毎に、入力テンソルに値を割り当て、インタープリタを呼び、そして出力テンソルから結果を読み取ります。

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

### 15. アプリケーションのコードを読む

この単体テストを一度ひととおり読み終えたら、[`main_functions.cc`](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/examples/hello_world/main_functions.cc)にあるサンプルアプリケーションのコードを理解できるはずです。 同じような処理を行いますが、実行された推論の数に基づいて入力値を生成し、それからデバイス固有の関数を呼び、モデルの出力をユーザーに表示します。
