# Introdução aos microcontroladores

Este documento explica como treinar um modelo e executar a inferência usando um microcontrolador.

## Exemplo Hello World

O objetivo do exemplo [Hello World](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/examples/hello_world) (Olá, mundo) é demonstrar o uso básico do TensorFlow Lite para Microcontroladores. Treinamos e executamos um modelo que replica uma função de seno, ou seja, recebe um único número como entrada e gera como saída o valor [seno](https://en.wikipedia.org/wiki/Sine) do número. Quando implantado no microcontrolador, suas previsões são usadas para piscar luzes LED ou controlar uma animação.

O workflow completo é composto pelas seguintes etapas:

1. [Treine um modelo](#train_a_model) (no Python): um arquivo do Python para treinar, converter e otimizar um modelo para uso em dispositivos.
2. [Execute a inferência](#run_inference) (no C++ 17): um teste de unidade fim a fim que executa a inferência no modelo usando a [biblioteca do C++](library.md).

## Use um dispositivo com suporte

O aplicativo de exemplo que usaremos foi testado nos seguintes dispositivos:

- [Arduino Nano 33 BLE Sense](https://store-usa.arduino.cc/products/arduino-nano-33-ble-sense-with-headers) (usando Arduino IDE)
- [SparkFun Edge](https://www.sparkfun.com/products/15170) (compilando diretamente a partir do código-fonte)
- [STM32F746 Discovery kit](https://www.st.com/en/evaluation-tools/32f746gdiscovery.html) (usando Mbed)
- [Adafruit EdgeBadge](https://www.adafruit.com/product/4400) (usando Arduino IDE)
- [Adafruit TensorFlow Lite for Microcontrollers Kit](https://www.adafruit.com/product/4317) (usando Arduino IDE)
- [Adafruit Circuit Playground Bluefruit](https://learn.adafruit.com/tensorflow-lite-for-circuit-playground-bluefruit-quickstart?view=all) (usando Arduino IDE)
- [Espressif ESP32-DevKitC](https://www.espressif.com/en/products/hardware/esp32-devkitc/overview) (usando ESP IDF)
- [Espressif ESP-EYE](https://www.espressif.com/en/products/hardware/esp-eye/overview) (usando ESP IDF)

Saiba mais sofre as plataformas com suporte no [TensorFlow Lite para Microcontroladores](index.md).

## Treine um modelo

Observação: você pode pular esta seção e usar o modelo treinado incluído no código de exemplo.

Use [train.py](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/examples/hello_world/train.py) para treinamento do modelo Hello World de reconhecimento de senoides.

Execute: `bazel build tensorflow/lite/micro/examples/hello_world:train` `bazel-bin/tensorflow/lite/micro/examples/hello_world/train --save_tf_model --save_dir=/tmp/model_created/`

## Execute a inferência

Para executar o modelo em seu dispositivo, falaremos sobre as instruções no arquivo `README.md`:

<a class="button button-primary" href="https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/examples/hello_world/README.md">Hello World README.md</a>

As próximas seções falam sobre o teste de unidade [`evaluate_test.cc`](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/examples/hello_world/evaluate_test.cc) do exemplo, que demonstra como executar a inferência usando o TensorFlow Lite para Microcontroladores. Ele carrega o modelo e executa a inferência diversas vezes.

### 1. Inclua os cabeçalhos da biblioteca

Para usar a biblioteca do TensorFlow Lite para Microcontroladores, precisamos incluir os seguintes arquivos de cabeçalho:

```C++
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
```

- [`micro_mutable_op_resolver.h`](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/micro_mutable_op_resolver.h) – fornece as operações usadas pelo interpretador para executar o modelo.
- [`micro_error_reporter.h`](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h) – gera como saída informações de depuração.
- [`micro_interpreter.h`](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/micro_interpreter.h) – contém código para carregar e executar modelos.
- [`schema_generated.h`](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/schema/schema_generated.h) – contém o esquema do formato de arquivo de modelo [`FlatBuffer`](https://google.github.io/flatbuffers/) do TensorFlow Lite.
- [`version.h`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/version.h) – fornece informações de versionamento do esquema do TensorFlow Lite.

### 2. Inclua o cabeçalho do modelo

O interpretador do TensorFlow Lite para Microcontroladores espera que o modelo seja fornecido como um array do C++. O modelo é definido nos arquivos `model.h` e `model.cc`. O cabeçalho é incluído com a seguinte linha:

```C++
#include "tensorflow/lite/micro/examples/hello_world/model.h"
```

### 3. Inclua o cabeçalho do framework de teste de unidade

Para criar um teste de unidade, incluímos o framework de teste de unidade do TensorFlow Lite para Microcontroladores com a seguinte linha:

```C++
#include "tensorflow/lite/micro/testing/micro_test.h"
```

O teste é definido usando-se as seguintes macros:

```C++
TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(LoadModelAndPerformInference) {
  . // add code here
  .
}

TF_LITE_MICRO_TESTS_END
```

Agora, vamos falar sobre o código incluído na macro acima.

### 4. Configure a gravação de logs

Para configurar a gravação de logs, um ponteiro `tflite::ErrorReporter` é criado usando-se um ponteiro para uma instância de `tflite::MicroErrorReporter`:

```C++
tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;
```

Essa variável será passada ao interpretador, o que permite a ele escrever logs. Como os microcontroladores costumam ter uma variedade de mecanismos de gravação de logs, a implementação de `tflite::MicroErrorReporter` foi criada de forma que possa ser personalizada para seu dispositivo específico.

### 5. Carregue um modelo

No código abaixo, o modelo é instanciado usando dados de um array `char`, `g_model`, que é declarado em `model.h`. Em seguida, verificamos o modelo para garantir que a versão do esquema seja compatível com a versão que estamos usando.

```C++
const tflite::Model* model = ::tflite::GetModel(g_model);
if (model->version() != TFLITE_SCHEMA_VERSION) {
  TF_LITE_REPORT_ERROR(error_reporter,
      "Model provided is schema version %d not equal "
      "to supported version %d.\n",
      model->version(), TFLITE_SCHEMA_VERSION);
}
```

### 6. Instancie o resolvedor de operações

Uma instância de [`MicroMutableOpResolver`](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/micro_mutable_op_resolver.h) é declarada, que será usada pelo interpretador para registrar e acessar as operações usadas pelo modelo:

```C++
using HelloWorldOpResolver = tflite::MicroMutableOpResolver<1>;

TfLiteStatus RegisterOps(HelloWorldOpResolver& op_resolver) {
  TF_LITE_ENSURE_STATUS(op_resolver.AddFullyConnected());
  return kTfLiteOk;

```

`MicroMutableOpResolver` requer um parâmetro de template que indique o número de operações que serão registradas. A função `RegisterOps` registra as operações no resolvedor.

```C++
HelloWorldOpResolver op_resolver;
TF_LITE_ENSURE_STATUS(RegisterOps(op_resolver));

```

### 7. Aloque memória

Precisamos pré-alocar uma determinada quantidade de memória para os arrays de entrada, saída e intermediários. Isso é fornecido como um array `uint8_t` de tamanho `tensor_arena_size`:

```C++
const int tensor_arena_size = 2 * 1024;
uint8_t tensor_arena[tensor_arena_size];
```

O tamanho necessário dependerá do modelo que você estiver usando e talvez precise ser determinado fazendo experimentos.

### 8. Instancie o interpretador

Criamos uma instância de `tflite::MicroInterpreter` passando as variáveis criadas anteriormente:

```C++
tflite::MicroInterpreter interpreter(model, resolver, tensor_arena,
                                     tensor_arena_size, error_reporter);
```

### 9. Aloque tensores

Dizemos ao interpretador para alocar memória de `tensor_arena` para os tensores do modelo:

```C++
interpreter.AllocateTensors();
```

### 10. Valide o formato da entrada

A instância de `MicroInterpreter` pode nos fornecer um ponteiro para o tensor de entrada do modelo, basta chamar `.input(0)`, em que `0` representa o primeiro (e único) tensor de entrada:

```C++
  // Obtain a pointer to the model's input tensor
  TfLiteTensor* input = interpreter.input(0);
```

Em seguida, inspecionamos esse tensor para confirmar que o formato e o tipo sejam os esperados:

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

O valor enum `kTfLiteFloat32` é uma referência a um dos tipos de dados do TensorFlow Lite e é definido em [`common.h`](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/c/common.h).

### 11. Forneça um valor de entrada

Para fornecer um valor de entrada ao modelo, definimos o conteúdo do tensor de entrada da seguinte forma:

```C++
input->data.f[0] = 0.;
```

Neste caso, fornecemos como entrada um valor de ponto flutuante que representa `0`.

### 12. Execute o modelo

Para executar o modelo, chamamos `Invoke()` na instância de `tflite::MicroInterpreter`:

```C++
TfLiteStatus invoke_status = interpreter.Invoke();
if (invoke_status != kTfLiteOk) {
  TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed\n");
}
```

Podemos verificar o valor retornado, um `TfLiteStatus`, para determinar se a execução foi bem-sucedida. Os valores possíveis de `TfLiteStatus`, definidos em [`common.h`](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/c/common.h), são `kTfLiteOk` e `kTfLiteError`.

O código abaixo indica que o valor é `kTfLiteOk`, ou seja, a inferência foi executada com êxito.

```C++
TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, invoke_status);
```

### 13. Obtenha a saída

O tensor de saída do modelo pode ser obtido chamando-se `output(0)` em `tflite::MicroInterpreter`, em que `0` representa o primeiro (e único) tensor de saída.

Neste exemplo, a saída do modelo é um único valor de ponto flutuante contido dentro de um tensor bidimensional:

```C++
TfLiteTensor* output = interpreter.output(0);
TF_LITE_MICRO_EXPECT_EQ(2, output->dims->size);
TF_LITE_MICRO_EXPECT_EQ(1, input->dims->data[0]);
TF_LITE_MICRO_EXPECT_EQ(1, input->dims->data[1]);
TF_LITE_MICRO_EXPECT_EQ(kTfLiteFloat32, output->type);
```

Podemos ler o valor diretamente no tensor de saída e confirmar que é o esperado:

```C++
// Obtain the output value from the tensor
float value = output->data.f[0];
// Check that the output value is within 0.05 of the expected value
TF_LITE_MICRO_EXPECT_NEAR(0., value, 0.05);
```

### 14. Execute a inferência novamente

O restante do código executa a inferência diversas outras vezes. Em cada execução, atribuímos um valor ao tensor de entrada, chamamos o interpretador e lemos o resultado no tensor de saída:

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
