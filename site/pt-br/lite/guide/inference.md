# Inferência com o TensorFlow Lite

O termo *inferência* refere-se ao processo de executar um modelo do TensdorFlow Lite em um dispositivo para fazer previsões com base nos dados de entrada. Para fazer a inferência de um modelo do TensorFlow Lite, você precisa executá-lo por meio de um *interpretador*. O interpretador do TensorFlow Lite é leve, rápido e usa uma ordenação estática de grafos e um alocador personalizado (menos dinâmico) de memória para garantir uma latência mínima de carga, inicialização e execução.

Esta página descreve como acessar o interpretador do TensorFlow Lite e fazer inferência usando o C++, Java e Python, além de incluir links para outros recursos para cada [plataforma com suporte](#supported-platforms).

[TOC]

## Conceitos importantes

Tipicamente, a inferência com o TensorFlow Lite segue estas etapas:

1. **Carregamento do modelo**

    Você precisa adicionar o modelo `.tflite` à memória, que contém o grafo de execução do modelo.

2. **Transformação dos dados**

    Geralmente, os dados de entrada brutos do modelo não coincidem com o formato de dados de entrada esperado por ele. Por exemplo: talvez você precise redimensionar uma imagem ou alterar o formato da imagem para que fique compatível com o modelo.

3. **Execução da inferência**

    Esta etapa envolve o uso da API do TensorFlow Lite para executar o modelo, além de envolver algumas etapas como compilar o interpretador e alocar tensores, conforme descrito nas próximas seções.

4. **Interpretação da saída**

    Ao receber os resultados de inferência do modelo, você precisa interpretar os tensores de uma forma que faça sentido e seja útil para sua aplicação.

    Por exemplo, talvez um modelo retorne somente uma lista de probabilidades. Cabe a você mapeá-las para categorias relevantes e apresentar os resultados ao usuário final.

## Plataformas com suporte

São fornecidas APIs de inferência do TensorFlow para a maioria das plataformas comuns de dispositivos móveis/embarcadas, como [Android](#android-platform), [iOS](#ios-platform) e [Linux](#linux-platform), em diversas linguagens de programação.

Na maioria dos casos, o design da API reflete a preferência por desempenho em vez de facilidade de uso. O TensorFlow Lite foi criado para fazer uma inferência rápida em dispositivos pequenos, então não é surpresa nenhuma que as APIs tentem evitar cópias desnecessárias apenas por questões de conveniência. De maneira similar, a consistência com as APIs do TensorFlow não era um objetivo explícito, e deve-se esperar algumas variações entre as linguagens.

Dentre todas as bibliotecas, a API do TensorFlow Lite permite carregar modelos, alimentar entradas e buscar saídas de inferência.

### Plataforma Android

No Android, a inferência com o TensorFlow Lite pode ser realizada usando APIs do Java ou C++. As APIs do Java são convenientes e podem ser usadas diretamente nas classes de Activity do Android. As APIs do C++ oferecem mais flexibilidade e velocidade, mas podem exigir a programação de encapsuladores JNI para mover dados entre as camadas do Java e do C++.

Confira detalhes sobre o uso do [C++](#load-and-run-a-model-in-c) e do [Java](#load-and-run-a-model-in-java) abaixo ou confira um tutorial e exemplo de código no [Guia de início rápido para Android](../android).

#### Gerador de código do encapsulador Android para o TensorFlow Lite

Observação: o gerador de código do encapsulador para o TensorFlow Lite está em fase experimental (beta) e, no momento, só tem suporte ao Android.

Para o modelo do TensorFlow Lite aprimorado com [metadados](../inference_with_metadata/overview), os desenvolvedores podem usar o gerador de código do encapsulador Android para o TensorFlow Lite para criar o código do encapsulador específico para a plataforma. O código do encapsulador remove a necessidade de interagir diretamente com o `ByteBuffer` no Android. Em vez disso, os desenvolvedores podem interagir com o modelo do TensorFlow Lite usando objetos tipados, como `Bitmap` e `Rect`. Confira mais informações em [Gerador de código do encapsulador Android para o TensorFlow Lite](../inference_with_metadata/codegen.md).

### Plataforma iOS

No iOS, o TensorFlow Lite está disponível com bibliotecas nativas do iOS no [Swift](https://www.tensorflow.org/code/tensorflow/lite/swift) e no [Objective-C](https://www.tensorflow.org/code/tensorflow/lite/objc). Além disso, você pode usar a [API do C](https://www.tensorflow.org/code/tensorflow/lite/c/c_api.h) diretamente em código do Objective-C.

Confira detalhes sobre o uso do [Swift](#load-and-run-a-model-in-swift), do [Objective-C](#load-and-run-a-model-in-objective-c) e da [API do C](#using-c-api-in-objective-c-code) abaixo ou confira um tutorial e exemplo de código no [Guia rápido para iOS](ios.md).

### Plataforma Linux

Em plataformas Linux (incluindo [Raspberry Pi](build_arm)), você pode executar inferência usando as APIs do TensorFlow Lite disponíveis no [C++](#load-and-run-a-model-in-c) e no [Python](#load-and-run-a-model-in-python), conforme exibido nas seções abaixo.

## Como executar um modelo

Para executar um modelo do TensorFlow Lite, é preciso seguir algumas etapas:

1. Carregue o modelo na memória.
2. Compile um `Interpreter` com base em um modelo existente.
3. Defina os valores dos tensores de entrada (opcionalmente, redimensione os tensores de entrada se os tamanhos predefinidos não forem desejáveis).
4. Invoque a inferência.
5. Leia os valores dos tensores de saída.

As próximas seções descrevem como essas etapas podem ser feitas em cada linguagem.

## Carregue e execute um modelo no Java

*Plataforma: Android*

A API do Java para executar inferência com o TensorFlow Lite foi criada principalmente para uso com o Android, então ela está disponível como uma dependência de biblioteca do Android: `org.tensorflow:tensorflow-lite`.

No Java, você usará a classe `Interpreter` para carregar um modelo e fazer a inferência. Em diversos casos, essa poderá ser a única API que você precisará utilizar.

Você pode inicializar um `Interpreter` usando um arquivo `.tflite`:

```java
public Interpreter(@NotNull File modelFile);
```

Ou com um `MappedByteBuffer`:

```java
public Interpreter(@NotNull MappedByteBuffer mappedByteBuffer);
```

Nos dois casos, você precisa fornecer um modelo válido do TensorFlow Lite, ou então a API vai gerar a exceção `IllegalArgumentException`. Se você utilizar `MappedByteBuffer` para inicializar um `Interpreter`, ele precisará permanecer inalterado durante todo o ciclo de vida do `Interpreter`.

A melhor forma de executar a inferência em um modelo é usando assinaturas, disponíveis para modelos convertidos a partir do Tensorflow 2.5.

```Java
try (Interpreter interpreter = new Interpreter(file_of_tensorflowlite_model)) {
  Map<String, Object> inputs = new HashMap<>();
  inputs.put("input_1", input1);
  inputs.put("input_2", input2);
  Map<String, Object> outputs = new HashMap<>();
  outputs.put("output_1", output1);
  interpreter.runSignature(inputs, outputs, "mySignature");
}
```

O método `runSignature` recebe três argumentos:

- **Inputs** (entradas): faz o mapeamento de entradas do nome de entradas na assinatura para um objeto de entrada.

- **Outputs** (saídas): faz o mapeamento de saídas do nome de saída na assinatura para os dados de saída.

- **Signature Name** (nome da assinatura) [opcional]: nome da assinatura (pode ser deixado em branco se o modelo tiver uma única assinatura).

Outra forma de executar a inferência quando o modelo não tiver assinaturas definidas: basta chamar `Interpreter.run()`. Por exemplo:

```java
try (Interpreter interpreter = new Interpreter(file_of_a_tensorflowlite_model)) {
  interpreter.run(input, output);
}
```

O método `run()` recebe somente uma entrada e retorna somente uma saída. Portanto, se o seu modelo tiver diversas entradas ou diversas saídas, use:

```java
interpreter.runForMultipleInputsOutputs(inputs, map_of_indices_to_outputs);
```

Neste caso, cada entrada em `inputs` corresponde a um tensor de entrada, e `map_of_indices_to_outputs` mapeia índices de tensores de saída para os dados de saída correspondentes.

Nos dois casos, os índices dos tensores correspondem aos valores que você forneceu ao [TensorFlow Lite Converter](../models/convert/) (conversor do TF Lite) ao criar o modelo. É importante salientar que a ordem dos tensores em `input` precisa coincidir com a ordem fornecida para o conversor do TensorFlow Lite.

A classe `Interpreter` também conta com funções convenientes para obter o índice de qualquer entrada ou saída do modelo usando um nome de operação:

```java
public int getInputIndex(String opName);
public int getOutputIndex(String opName);
```

Se `opName` não for uma operação válida no modelo, é gerada a exceção `IllegalArgumentException`.

Também é importante salientar que o `Interpreter` é proprietário de recursos. Para evitar vazamento de memória, os recursos precisam ser liberados após o uso da seguinte forma:

```java
interpreter.close();
```

Confira um exemplo de projeto com Java no [exemplo de classificação de imagens no Android](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android).

### Tipos de dados permitidos (no Java)

Para usar o TensorFlow Lite, os tipos de dados dos tensores de entrada e saída precisam ser um dos tipos primitivos abaixo:

- `float`
- `int`
- `long`
- `byte`

Também há suporte aos tipos `String`, mas eles são codificados de forma diferente do que os tipos primitivos. Especificamente, o formato de um tensor String determina o número e a organização das strings no tensor, em que cada elemento é um string de tamanho variável. Nesse sentido, o tamanho (byte) do tensor não pode ser computado somente a partir do formato e do tipo e, consequentemente, as strings não podem ser fornecidas como um único argumento simples `ByteBuffer`. Confira alguns exemplos nesta [página](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/Interpreter).

Se forem usados outros tipos de dados, incluindo tipos boxed, como `Integer` e `Float`, será gerada a exceção `IllegalArgumentException`.

#### Entradas

Cada entrada deve ser um array ou um array multidimensional dos tipos primitivos permitidos, ou um `ByteBuffer` bruto do tamanho adequado. Se a entrada for um array ou um array multidimensional, o tensor de entrada associado será redimensionado implicitamente para as dimensões do array no momento da inferência. Se a entrada for um ByteBuffer, o chamador deverá primeiro redimensionar manualmente o tensor de entrada associado (via `Interpreter.resizeInput()`) antes de executar a inferência.

Ao usar `ByteBuffer`, é melhor usar buffers de byte diretos, pois assim o `Interpreter` poderá evitar cópias desnecessárias. Se o `ByteBuffer` for um buffer de byte direto, sua ordem deve ser `ByteOrder.nativeOrder()`. Após ser usado para inferência do modelo, ele deve permanecer inalterado até a conclusão da inferência.

#### Saídas

Cada saída deve ser um array ou um array multidimensional dos tipos primitivos permitidos, ou um ByteBuffer do tamanho adequado. Alguns modelos têm saídas dinâmicas, em que o formato dos tensores de saída pode variar, dependendo da entrada. Não há uma única forma direta de tratar esse problema com a API atual de inferência do Java, mas extensões futuras deverão permitir isso.

## Carregue e execute um modelo no Swift

*Plataforma: iOS*

A [API do Swift](https://www.tensorflow.org/code/tensorflow/lite/swift) está disponível no pod `TensorFlowLiteSwift` do CocoaPods.

Primeiro, você precisa importar o módulo `TensorFlowLite`.

```swift
import TensorFlowLite
```

```swift
// Getting model path
guard
  let modelPath = Bundle.main.path(forResource: "model", ofType: "tflite")
else {
  // Error handling...
}

do {
  // Initialize an interpreter with the model.
  let interpreter = try Interpreter(modelPath: modelPath)

  // Allocate memory for the model's input `Tensor`s.
  try interpreter.allocateTensors()

  let inputData: Data  // Should be initialized

  // input data preparation...

  // Copy the input data to the input `Tensor`.
  try self.interpreter.copy(inputData, toInputAt: 0)

  // Run inference by invoking the `Interpreter`.
  try self.interpreter.invoke()

  // Get the output `Tensor`
  let outputTensor = try self.interpreter.output(at: 0)

  // Copy output to `Data` to process the inference results.
  let outputSize = outputTensor.shape.dimensions.reduce(1, {x, y in x * y})
  let outputData =
        UnsafeMutableBufferPointer<Float32>.allocate(capacity: outputSize)
  outputTensor.data.copyBytes(to: outputData)

  if (error != nil) { /* Error handling... */ }
} catch error {
  // Error handling...
}
```

## Carregue e execute um modelo no Objective-C

*Plataforma: iOS*

A [API do Objective-C](https://www.tensorflow.org/code/tensorflow/lite/objc) está disponível no pod `TensorFlowLiteObjC` do CocoaPods.

Primeiro, você precisa importar o módulo `TensorFlowLite`.

```objc
@import TensorFlowLite;
```

```objc
NSString *modelPath = [[NSBundle mainBundle] pathForResource:@"model"
                                                      ofType:@"tflite"];
NSError *error;

// Initialize an interpreter with the model.
TFLInterpreter *interpreter = [[TFLInterpreter alloc] initWithModelPath:modelPath
                                                                  error:&error];
if (error != nil) { /* Error handling... */ }

// Allocate memory for the model's input `TFLTensor`s.
[interpreter allocateTensorsWithError:&error];
if (error != nil) { /* Error handling... */ }

NSMutableData *inputData;  // Should be initialized
// input data preparation...

// Get the input `TFLTensor`
TFLTensor *inputTensor = [interpreter inputTensorAtIndex:0 error:&error];
if (error != nil) { /* Error handling... */ }

// Copy the input data to the input `TFLTensor`.
[inputTensor copyData:inputData error:&error];
if (error != nil) { /* Error handling... */ }

// Run inference by invoking the `TFLInterpreter`.
[interpreter invokeWithError:&error];
if (error != nil) { /* Error handling... */ }

// Get the output `TFLTensor`
TFLTensor *outputTensor = [interpreter outputTensorAtIndex:0 error:&error];
if (error != nil) { /* Error handling... */ }

// Copy output to `NSData` to process the inference results.
NSData *outputData = [outputTensor dataWithError:&error];
if (error != nil) { /* Error handling... */ }
```

### Como usar a API do C em código do Objective-C

Atualmente, a API do Objective-C não tem suporte a delegados. Para usar delegados em código do Objective-C, você precisa chamar diretamente a [API do C](https://www.tensorflow.org/code/tensorflow/lite/c/c_api.h) subjacente.

```c
#include "tensorflow/lite/c/c_api.h"
```

```c
TfLiteModel* model = TfLiteModelCreateFromFile([modelPath UTF8String]);
TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();

// Create the interpreter.
TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, options);

// Allocate tensors and populate the input tensor data.
TfLiteInterpreterAllocateTensors(interpreter);
TfLiteTensor* input_tensor =
    TfLiteInterpreterGetInputTensor(interpreter, 0);
TfLiteTensorCopyFromBuffer(input_tensor, input.data(),
                           input.size() * sizeof(float));

// Execute inference.
TfLiteInterpreterInvoke(interpreter);

// Extract the output tensor data.
const TfLiteTensor* output_tensor =
    TfLiteInterpreterGetOutputTensor(interpreter, 0);
TfLiteTensorCopyToBuffer(output_tensor, output.data(),
                         output.size() * sizeof(float));

// Dispose of the model and interpreter objects.
TfLiteInterpreterDelete(interpreter);
TfLiteInterpreterOptionsDelete(options);
TfLiteModelDelete(model);
```

## Carregue e execute um modelo no C++

*Plataformas: Android, iOS e Linux*

Observação: a API do C++ no iOS está disponível somente ao usar o Bazel.

No C++, o modelo é armazenado na classe [`FlatBufferModel`](https://www.tensorflow.org/lite/api_docs/cc/class/tflite/flat-buffer-model.html). Ele encapsula um modelo do TensorFlow Lite, e você pode compilá-lo de diferentes formas, dependendo de onde o modelo é armazenado:

```c++
class FlatBufferModel {
  // Build a model based on a file. Return a nullptr in case of failure.
  static std::unique_ptr<FlatBufferModel> BuildFromFile(
      const char* filename,
      ErrorReporter* error_reporter);

  // Build a model based on a pre-loaded flatbuffer. The caller retains
  // ownership of the buffer and should keep it alive until the returned object
  // is destroyed. Return a nullptr in case of failure.
  static std::unique_ptr<FlatBufferModel> BuildFromBuffer(
      const char* buffer,
      size_t buffer_size,
      ErrorReporter* error_reporter);
};
```

Observação: se o TensorFlow Lite detectar a presença da [NNAPI do Android](https://developer.android.com/ndk/guides/neuralnetworks), vai tentar automaticamente usar a memória compartilhada para armazenar o `FlatBufferModel`.

Agora que você tem o modelo como um objeto `FlatBufferModel`, pode executá-lo com um [`Interpreter`](https://www.tensorflow.org/lite/api_docs/cc/class/tflite/interpreter.html). Um único `FlatBufferModel` pode ser usado simultaneamente por mais de um `Interpreter`.

Atenção: o objeto `FlatBufferModel` precisa permanecer válido até todas as instâncias do `Interpreter` que o utilizem serem destruídas.

As partes importantes da API do `Interpreter` são exibidas no trecho de código abaixo. Deve-se observar que:

- Os tensores são representados por inteiros para evitar comparações entre strings (e qualquer dependência fixa nas bibliotecas de strings).
- Um interpretador não pode ser acessado em threads simultâneos.
- A alocação de memória para os tensores de entrada e saída precisa ser acionada chamando `AllocateTensors()` logo após o redimensionamento dos tensores.

Veja o uso mais simples do TensorFlow Lite com C++:

```c++
// Load the model
std::unique_ptr<tflite::FlatBufferModel> model =
    tflite::FlatBufferModel::BuildFromFile(filename);

// Build the interpreter
tflite::ops::builtin::BuiltinOpResolver resolver;
std::unique_ptr<tflite::Interpreter> interpreter;
tflite::InterpreterBuilder(*model, resolver)(&interpreter);

// Resize input tensors, if desired.
interpreter->AllocateTensors();

float* input = interpreter->typed_input_tensor<float>(0);
// Fill `input`.

interpreter->Invoke();

float* output = interpreter->typed_output_tensor<float>(0);
```

Confira mais códigos de exemplo em [`minimal.cc`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/minimal/minimal.cc) e [`label_image.cc`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/label_image/label_image.cc).

## Carregue e execute um modelo no Python

*Plataforma: Linux*

A API do Python para executar inferência é fornecida no módulo `tf.lite`. Basicamente, você só precisa do [`tf.lite.Interpreter`](https://www.tensorflow.org/api_docs/python/tf/lite/Interpreter) para carregar um modelo e executar a inferência.

O exemplo abaixo mostra como usar o interpretador do Python para carregar um arquivo `.tflite` e executar a inferência com dados de entrada aleatórios:

Este exemplo é recomendado se você estiver convertendo um SavedModel com uma SignatureDef definida. Disponível a partir do TensorFlow 2.5.

```python
class TestModel(tf.Module):
  def __init__(self):
    super(TestModel, self).__init__()

  @tf.function(input_signature=[tf.TensorSpec(shape=[1, 10], dtype=tf.float32)])
  def add(self, x):
    '''
    Simple method that accepts single input 'x' and returns 'x' + 4.
    '''
    # Name the output 'result' for convenience.
    return {'result' : x + 4}


SAVED_MODEL_PATH = 'content/saved_models/test_variable'
TFLITE_FILE_PATH = 'content/test_variable.tflite'

# Save the model
module = TestModel()
# You can omit the signatures argument and a default signature name will be
# created with name 'serving_default'.
tf.saved_model.save(
    module, SAVED_MODEL_PATH,
    signatures={'my_signature':module.add.get_concrete_function()})

# Convert the model using TFLiteConverter
converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_PATH)
tflite_model = converter.convert()
with open(TFLITE_FILE_PATH, 'wb') as f:
  f.write(tflite_model)

# Load the TFLite model in TFLite Interpreter
interpreter = tf.lite.Interpreter(TFLITE_FILE_PATH)
# There is only 1 signature defined in the model,
# so it will return it by default.
# If there are multiple signatures then we can pass the name.
my_signature = interpreter.get_signature_runner()

# my_signature is callable with input as arguments.
output = my_signature(x=tf.constant([1.0], shape=(1,10), dtype=tf.float32))
# 'output' is dictionary with all outputs from the inference.
# In this case we have single output 'result'.
print(output['result'])
```

Outro exemplo se o modelo não tiver SignatureDefs definidas.

```python
import numpy as np
import tensorflow as tf

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="converted_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the model on random input data.
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)
```

Como alternativa para carregar o modelo como um arquivo `.tflite` pré-convertido, você pode combinar seu código com a [API do Python do conversor do TensorFlow Lite](https://www.tensorflow.org/lite/api_docs/python/tf/lite/TFLiteConverter) (`tf.lite.TFLiteConverter`), permitindo converter seu modelo do Keras para o formato do TensorFlow Lite e depois executar a inferência:

```python
import numpy as np
import tensorflow as tf

img = tf.keras.Input(shape=(64, 64, 3), name="img")
const = tf.constant([1., 2., 3.]) + tf.constant([1., 4., 4.])
val = img + const
out = tf.identity(val, name="out")

# Convert to TF Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(tf.keras.models.Model(inputs=[img], outputs=[out]))
tflite_model = converter.convert()

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

# Continue to get tensors and so forth, as shown above...
```

Confira mais exemplos de código do Python em [`label_image.py`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py).

Dica: execute `help(tf.lite.Interpreter)` no terminal do Python para ver a documentação detalhada sobre o interpretador.

## Execute a inferência com um modelo de formato dinâmico

Se você deseja executar um modelo com formato de entrada dinâmico, *redimensione o formato da entrada* antes de executar a inferência. Caso contrário, o formato `None` em modelos do Tensorflow será substituído pelo padrão `1` nos modelos do TF Lite.

Os exemplos abaixo mostram como redimensionar o formato da entrada antes de executar a inferência em diferentes linguagens. Todos os exemplos pressupõem que o formato da entrada é definido como `[1/None, 10]` e precisa ser redimensionado para `[3, 10]`.

Exemplo no C++:

```c++
// Resize input tensors before allocate tensors
interpreter->ResizeInputTensor(/*tensor_index=*/0, std::vector<int>{3,10});
interpreter->AllocateTensors();
```

Exemplo no Python:

```python
# Load the TFLite model in TFLite Interpreter
interpreter = tf.lite.Interpreter(model_path=TFLITE_FILE_PATH)
  
# Resize input shape for dynamic shape model and allocate tensor
interpreter.resize_tensor_input(interpreter.get_input_details()[0]['index'], [3, 10])
interpreter.allocate_tensors()
  
# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
```




## Operações permitidas

O TensorFlow Lite tem suporte a um subconjunto das operações do TensorFlow, com algumas limitações. Veja a lista completa de operações e limitações na [página de operações do TF Lite](https://www.tensorflow.org/mlir/tfl_ops).
