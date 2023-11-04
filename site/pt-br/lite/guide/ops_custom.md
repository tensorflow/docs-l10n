# Operadores personalizados

Como a biblioteca de operadores integrados do TensorFlow Lite só tem suporte a um número limitado de operadores do TensorFlow, nem todo modelo pode ser convertido. Confira mais detalhes em [Compatibilidade de operadores](ops_compatibility.md).

Para possibilitar a conversão, os usuários precisam fornecer sua própria implementação personalizada de um operador do TensorFlow sem suporte no TensorFlow Lite, conhecido como operador personalizado. *Se em vez disso você quiser combinar uma série de operadores do TensorFlow sem suporte (ou com suporte) em um único operador personalizado, otimizado e fundido, confira [Fusão de operadores](https://www.tensorflow.org/lite/models/convert/operation_fusion).*

Para usar operadores personalizados, é preciso seguir quatro etapas:

- [Crie um modelo do TensorFlow.](#create-a-tensorflow-model) O SavedModel (ou GraphDef) precisa referenciar o nome correto do operador do TensorFlow Lite.

- [Converta para um modelo do TensorFlow Lite.](#convert-to-a-tensorflow-lite-model) Defina o atributo correto do conversor do TensorFlow Lite para converter o modelo corretamente.

- [Crie e registre o operador.](#create-and-register-the-operator) Isso é necessário para que o runtime do TensorFlow Lite saiba como mapear o operador e os parâmetros em seu grafo para código executável do C/C++.

- [Teste e faça o profiling do seu operador.](#test-and-profile-your-operator) Se você deseja testar somente seu operador personalizado, é melhor criar um modelo com somente seu operador e usar o programa [benchmark_model](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/tools/benchmark/benchmark_model.cc).

Vamos ver um exemplo completo de execução de um modelo com um operador personalizado `tf.atan` (chamado de `Atan`, confira a seção Crie um modelo do TensorFlow), que tem suporte no TensorFlow, mas não tem suporte no TensorFlow Lite.

Observação: a função `tf.atan` **não** é um operador personalizado. É um operador comum com suporte tanto no TensorFlow quanto no TensorFlow Lite. Porém, vamos **pressupor** que ele seja um operador personalizado no exemplo abaixo para demonstrar um workflow simples.

O operador TensorFlow Text é um exemplo de operador personalizado. Confira um exemplo de código no tutorial <a href="https://tensorflow.org/text/guide/text_tf_lite" class="external">Converta TF Text para o TF Lite</a>.

## Exemplo: operador personalizado `Atan`

Vamos ver um exemplo de como adicionar suporte a um operador do TensorFlow não disponível no TensorFlow Lite. Vamos supor que estejamos usando o operador `Atan` e que estejamos criando um modelo muito simples para a função `y = atan(x + offset)`, em que `offset` é treinável.

### Crie um modelo do TensorFlow

O trecho de código abaixo treina um modelo simples do TensorFlow. Esse modelo contém somente um operador personalizado chamado `Atan`, que é a função `y = atan(x + offset)`, em que `offset` é treinável.

```python
import tensorflow as tf

# Define training dataset and variables
x = [-8, 0.5, 2, 2.2, 201]
y = [-1.4288993, 0.98279375, 1.2490457, 1.2679114, 1.5658458]
offset = tf.Variable(0.0)

# Define a simple model which just contains a custom operator named `Atan`
@tf.function(input_signature=[tf.TensorSpec.from_tensor(tf.constant(x))])
def atan(x):
  return tf.atan(x + offset, name="Atan")

# Train model
optimizer = tf.optimizers.Adam(0.01)
def train(x, y):
    with tf.GradientTape() as t:
      predicted_y = atan(x)
      loss = tf.reduce_sum(tf.square(predicted_y - y))
    grads = t.gradient(loss, [offset])
    optimizer.apply_gradients(zip(grads, [offset]))

for i in range(1000):
    train(x, y)

print("The actual offset is: 1.0")
print("The predicted offset is:", offset.numpy())
```

```python
The actual offset is: 1.0
The predicted offset is: 0.99999905
```

Neste momento, se você tentar gerar um modelo do TensorFlow Lite com os sinalizadores padrão do conversor, verá a seguinte mensagem de erro:

```none
Error:
error: 'tf.Atan' op is neither a custom op nor a flex op.
```

### Converta para um modelo do TensorFlow Lite

Crie um modelo do TensorFlow Lite com operadores personalizados definindo o atributo `allow_custom_ops` do conversor, conforme exibido abaixo:

<pre>converter = tf.lite.TFLiteConverter.from_concrete_functions([atan.get_concrete_function()], atan)
&lt;b&gt;converter.allow_custom_ops = True&lt;/b&gt;
tflite_model = converter.convert()
</pre>

Neste momento, se você executá-lo com o interpretador padrão usando comandos como os seguintes:

```python
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()
```

Será exibido o erro:

```none
Encountered unresolved custom op: Atan.
```

### Crie e registre o operador

Todos os operadores do TensorFlow Lite (tanto personalizados quanto integrados) são definidos usando-se uma interface simples em C puro que é composta por quatro funções:

```c++
typedef struct {
  void* (*init)(TfLiteContext* context, const char* buffer, size_t length);
  void (*free)(TfLiteContext* context, void* buffer);
  TfLiteStatus (*prepare)(TfLiteContext* context, TfLiteNode* node);
  TfLiteStatus (*invoke)(TfLiteContext* context, TfLiteNode* node);
} TfLiteRegistration;
```

Confira [`common.h`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/c/common.h) para ver detalhes sobre `TfLiteContext` e `TfLiteNode`. O primeiro conta com recursos de relatórios de erros e acesso a objetos globais, incluindo todos os tensores. O segundo permite implementações para acessar suas entradas e saídas.

Quando o interpretador carrega um modelo, ele chama `init()` uma vez em cada nó do grafo. Um determinado `init()` será chamado mais de uma vez se a operação for usada diversas vezes no grafo. Para operações personalizadas, será concedido um buffer de configuração contendo um flexbuffer que mapeia os nomes de parâmetros para seus valores. O buffer fica vazio para operações integradas, pois o interpretador já processou seus parâmetros. Implementações de kernels que exijam estado deverão inicializá-lo aqui e transferir a titularidade para o chamador. Para cada chamada a `init()`, haverá uma chamada correspondente a `free()`, o que permite às implementações descartar o buffer que possam ter alocado em `init()`.

Sempre que os tensores de entrada são redimensionados, o interpretador percorre o grafo, notificando as implementações que houve uma mudança. Dessa forma, elas terão a oportunidade de redimensionar o buffer interno, verificar a validade dos formatos e tipos de entrada, além de recalcular os formatos de saída. Isso tudo é feito por meio de `prepare()`, e as implementações podem acessar seu estado utilizando `node->user_data`.

Por fim, a cada execução da inferência, o interpretador percorre o grafo, chamando `invoke()`, e o estado também fica disponível utilizando `node->user_data`.

As operações personalizadas podem ser implementadas exatamente da mesma forma que as operações integradas, basta definir essas quatro funções e uma função global de registro que geralmente é feita da seguinte forma:

```c++
namespace tflite {
namespace ops {
namespace custom {
  TfLiteRegistration* Register_MY_CUSTOM_OP() {
    static TfLiteRegistration r = {my_custom_op::Init,
                                   my_custom_op::Free,
                                   my_custom_op::Prepare,
                                   my_custom_op::Eval};
    return &r;
  }
}  // namespace custom
}  // namespace ops
}  // namespace tflite
```

Note que o registro não é automático, e uma chamada explícita a `Register_MY_CUSTOM_OP` precisa ser feita. Embora o `BuiltinOpResolver` padrão (disponível no alvo `:builtin_ops`) trate o registro de operações integradas, as operações personalizadas precisarão ser coletadas em bibliotecas personalizadas separadas.

### Definição do kernel no runtime do TensorFlow Lite

Para a usar a operação no TensorFlow Lite, basta definir duas funções (`Prepare` e `Eval`) e criar um `TfLiteRegistration`:

```cpp
TfLiteStatus AtanPrepare(TfLiteContext* context, TfLiteNode* node) {
  using namespace tflite;
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* input = GetInput(context, node, 0);
  TfLiteTensor* output = GetOutput(context, node, 0);

  int num_dims = NumDimensions(input);

  TfLiteIntArray* output_size = TfLiteIntArrayCreate(num_dims);
  for (int i=0; i<num_dims; ++i) {
    output_size->data[i] = input->dims->data[i];
  }

  return context->ResizeTensor(context, output, output_size);
}

TfLiteStatus AtanEval(TfLiteContext* context, TfLiteNode* node) {
  using namespace tflite;
  const TfLiteTensor* input = GetInput(context, node, 0);
  TfLiteTensor* output = GetOutput(context, node, 0);

  float* input_data = GetTensorData<float>(input);
  float* output_data = GetTensorData<float>(output);

  size_t count = 1;
  int num_dims = NumDimensions(input);
  for (int i = 0; i < num_dims; ++i) {
    count *= input->dims->data[i];
  }

  for (size_t i=0; i<count; ++i) {
    output_data[i] = atan(input_data[i]);
  }
  return kTfLiteOk;
}

TfLiteRegistration* Register_ATAN() {
  static TfLiteRegistration r = {nullptr, nullptr, AtanPrepare, AtanEval};
  return &r;
}
```

Ao inicializar `OpResolver`, adicione a operação personalizada no resolvedor (confira um exemplo abaixo). Dessa forma, o operador será registrado no Tensorflow Lite para que o TensorFlow Lite possa usar a nova implementação. Os últimos dois argumentos em `TfLiteRegistration` correspondem às funções `AtanPrepare` e `AtanEval` que você definiu para a operação personalizada. Se você tiver usado as funções `AtanInit` e `AtanFree` para inicializar as variáveis usadas na operação e para liberar espaço, respectivamente, elas terão sido adicionadas aos dois primeiros argumentos de `TfLiteRegistration`. Esses argumentos são definidos como `nullptr` neste exemplo.

### Registre o operador na biblioteca do kernel

Agora, precisamos registrar o operador na biblioteca do kernel, o que é feito usando um `OpResolver`. Por trás dos panos, o interpretador vai carregar uma biblioteca de kernels, que terá a atribuição de executar cada um dos operadores do modelo. Embora a biblioteca padrão contenha somente kernels integrados, é possível substituí-la/ampliá-la com operadores de uma biblioteca personalizada.

A classe `OpResolver`, que converte códigos e nomes de operadores em código em si, é definida da seguinte forma:

```c++
class OpResolver {
  virtual TfLiteRegistration* FindOp(tflite::BuiltinOperator op) const = 0;
  virtual TfLiteRegistration* FindOp(const char* op) const = 0;
  virtual void AddBuiltin(tflite::BuiltinOperator op, TfLiteRegistration* registration) = 0;
  virtual void AddCustom(const char* op, TfLiteRegistration* registration) = 0;
};
```

Nos casos de uso comum, é preciso utilizar `BuiltinOpResolver` e escrever:

```c++
tflite::ops::builtin::BuiltinOpResolver resolver;
```

Para adicionar a operação personalizada criada acima, é preciso chamar `AddOp` (antes de passar o resolvedor para `InterpreterBuilder`):

```c++
resolver.AddCustom("Atan", Register_ATAN());
```

Se for determinado que o conjunto de operações integradas é grande demais, um novo `OpResolver` pode ser gerado por código com base em um determinado subconjunto de operações, possivelmente somente aquelas contidas em um determinado modelo. Isso é equivalente ao registro seletivo do TensorFlow (uma versão simples está disponível no diretório `tools`).

Se quiser definir seus operadores personalizados no Java, no momento você precisa criar sua própria camada JNI personalizada e compilar seu próprio AAR [neste código jni](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/java/src/main/native/nativeinterpreterwrapper_jni.cc). De maneira similar, se quiser disponibilizar esses operadores no Python, você pode colocar os registros no [código de encapsulamento do Python](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/python/interpreter_wrapper/interpreter_wrapper.cc).

Um processo similar ao descrito acima pode ser seguido para oferecer suporte a um conjunto de operações em vez de a um único operador, basta adicionar o número de operadores `AddCustom` necessários. Além disso, `BuiltinOpResolver` também permite sobrescrever implementações de operadores integrados usando `AddBuiltin`.

### Teste e faça o profiling do seu operador

Para fazer o profiling da sua operação usando a ferramenta de referencial do TensorFlow Lite, você pode usar a [ferramenta de benchmark de modelos](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark#tflite-model-benchmark-tool) para o TensorFlow Lite. Para fazer testes, você pode criar sua build local do TensorFlow Lite com reconhecimento da sua operação personalizada adicionando a chamada `AddCustom` adequada (conforme mostrado acima) a [register.cc](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/core/kernels/register.cc).

## Práticas recomendadas

1. Otimize as alocações e desalocações de memória com cuidado. Alocar memória em `Prepare` é mais eficiente do que em `Invoke`, e alocar memória antes de um loop é melhor do que alocar em cada iteração. Use dados de tensores temporários em vez fazer a alocação de memória por conta própria (confira o item 2). Na medida do possível, use ponteiros/referências em vez de copiar.

2. Se uma estrutura de dados for existir durante toda a operação, aconselhamos fazer a pré-alocação da memória usando tensores temporários. Talvez você precise usar a estrutura OpData para referenciar os índices dos tensores em outras funções. Confira o exemplo em [Kernel para convolução](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/kernels/conv.cc). Veja abaixo um trecho de código de exemplo:

    ```
    auto* op_data = reinterpret_cast<OpData*>(node->user_data);
    TfLiteIntArrayFree(node->temporaries);
    node->temporaries = TfLiteIntArrayCreate(1);
    node->temporaries->data[0] = op_data->temp_tensor_index;
    TfLiteTensor* temp_tensor = &context->tensors[op_data->temp_tensor_index];
    temp_tensor->type =  kTfLiteFloat32;
    temp_tensor->allocation_type = kTfLiteArenaRw;
    ```

3. Se não houver muito desperdício de memória, opte por usar um array de tamanho fixo estático (ou um `std::vector` pré-alocado em `Resize`) em vez de usar um `std::vector` alocado dinamicamente em cada iteração da execução.

4. Evite instanciar modelos do container da biblioteca padrão que ainda não existam, pois isso afeta o tamanho do binário. Por exemplo: se você precisar de um `std::map` em sua operação que não exista em outros kernels, usar um `std::vector` com mapeamento de indexação direta pode funcionar, mantendo o binário pequeno. Confira quais outros kernels usar para entender melhor (ou pergunte).

5. Confira o ponteiro para a memória retornado por `malloc`. Se esse ponteiro for `nullptr`, nenhuma operação deve ser realizada utilizando-o. Se você fizer a alocação de memória (`malloc`) em uma função e houver um erro na saída, desaloque a memória antes de sair.

6. Use `TF_LITE_ENSURE(context, condition)` para verificar uma condição específica. Seu código não pode deixar memória perdida quando `TF_LITE_ENSURE` é usado, ou seja, essas macros devem ser usadas antes de qualquer recurso que cause vazamento seja alocado.
