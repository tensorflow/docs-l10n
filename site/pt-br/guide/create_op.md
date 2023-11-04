# Criando uma op

Observação: para garantir que suas operações (ops) personalizadas C++ tenham compatibilidade ABI com os pacotes pip oficiais do TensorFlow, siga o guia em [Repositório de ops personalizadas](https://github.com/tensorflow/custom-op). Ele inclui um exemplo de código completo, bem como imagens Docker para construir e distribuir suas ops personalizadas.

Se você quiser criar uma op que não seja coberta pela biblioteca existente do TensorFlow, recomendamos que primeiro tente escrever a op em Python como uma composição de ops ou funções existentes do Python. Se isto não for possível, você pode criar uma op C++ personalizada. Há vários motivos pelos quais você pode querer criar uma op C++ personalizada:

- Não é fácil nem possível expressar sua operação como uma composição de ops existentes.
- Não é eficiente expressar sua operação como uma composição de primitivos existentes.
- Você deseja combinar manualmente uma composição de primitivos que um futuro compilador acharia difícil combinar.

Por exemplo, imagine que você queira implementar algo como "pooling de medianas", semelhante ao operador "MaxPool", mas que computa medianas sobre janelas deslizantes em vez de valores máximos. Fazer isto usando uma composição de operações pode até ser possível (por exemplo, usando ExtractImagePatches e TopK), mas pode não ser tão eficiente em termos de desempenho ou memória quanto uma operação nativa onde você possa fazer algo mais inteligente numa única operação combinada. Como sempre, normalmente vale a pena primeiro tentar expressar o que você deseja usando a composição de operadores, optando apenas por adicionar uma nova operação se isto provar ser difícil ou ineficiente.

Para incorporar sua op personalizada, você precisará fazer o seguinte:

1. Registre a nova op num arquivo C++. O registro de ops define uma interface (especificação) para a funcionalidade da op, que é independente da implementação da op. Por exemplo, o registro da op define o nome da op e suas entradas e saídas. Ele também define a função de formato usada para inferência do formato do tensor.
2. Implemente a op em C++. A implementação de uma op é conhecida como kernel e é a implementação concreta da especificação que você registrou no primeiro passo. Pode haver vários kernels para diferentes tipos ou arquiteturas de entrada/saída (por exemplo, CPUs, GPUs).
3. Crie um wrapper Python (opcional). Este wrapper é a API pública usada para criar a operação em Python. Um wrapper padrão é gerado a partir do registro operacional, que pode ser usado diretamente ou adicionado.
4. Escreva uma função para calcular gradientes para a op (opcional).
5. Teste a op. Geralmente fazemos isso em Python por conveniência, mas você também pode testar a operação em C++. Se você definir gradientes, poderá verificá-los como `tf.test.compute_gradient_error` do Python. Veja [`relu_op_test.py`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/kernel_tests/nn_ops/relu_op_test.py) como um exemplo que testa as funções diretas de operadores do tipo Relu e seus gradientes.

### Pré-requisitos

- Alguma familiaridade com C++.
- Você deve ter instalado o [binário do TensorFlow](https://www.tensorflow.org/install) ou deve ter [baixado a fonte do TensorFlow](https://www.tensorflow.org/install/source) e ser capaz de compilá-la e montá-la.

## Defina a interface da op

Você define a interface de uma op registrando-a no sistema TensorFlow. No registro, você especifica o nome da op, suas entradas (tipos e nomes) e saídas (tipos e nomes), bem como docstrings e quaisquer [attrs](#attrs) que a op possa exigir.

Para ver como isso funciona, suponha que você queira criar uma op que pegue um tensor de `int32s` e produza uma cópia do tensor, com todos os elementos tendo o valor zero, exceto o primeiro. Para fazer isso, crie um arquivo chamado `zero_out.cc`. Em seguida, adicione uma chamada à macro `REGISTER_OP` que define a interface da sua op:

```c++
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("ZeroOut")
    .Input("to_zero: int32")
    .Output("zeroed: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });
```

Esta operação `ZeroOut` recebe um tensor `to_zero` de inteiros de 32 bits como entrada e gera um tensor `zeroed` de inteiros de 32 bits. A operação também usa uma função de formato para garantir que o tensor de saída tenha o mesmo formato que o tensor de entrada. Por exemplo, se a entrada for um tensor de formato [10, 20], então esta função de formato especifica que o formato de saída também será [10, 20].

Nota: O nome da op deve estar em CamelCase e deve ser único entre todas as outras ops registradas no binário.

## Implemente o kernel para a op

Depois de definir a interface, forneça uma ou mais implementações da operação. Para criar um desses kernels, crie uma classe que estenda `OpKernel` e sobreponha o método `Compute`. O método `Compute` fornece um argumento `context` do tipo `OpKernelContext*`, a partir do qual você pode acessar coisas úteis como os tensores de entrada e saída.

Adicione seu kernel ao arquivo que você criou acima. O kernel pode ser parecido com o mostrado a seguir:

```c++
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

class ZeroOutOp : public OpKernel {
 public:
  explicit ZeroOutOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<int32>();

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<int32>();

    // Set all but the first element of the output tensor to 0.
    const int N = input.size();
    for (int i = 1; i < N; i++) {
      output_flat(i) = 0;
    }

    // Preserve the first input value if possible.
    if (N > 0) output_flat(0) = input(0);
  }
};
```

Depois de implementar seu kernel, você o registra no sistema TensorFlow. No registro, você especifica diferentes restrições sob as quais este kernel será executado. Por exemplo, você pode ter um kernel feito para CPUs e outro separado para GPUs.

Para fazer isso na operação `ZeroOut`, adicione o seguinte a `zero_out.cc`:

```c++
REGISTER_KERNEL_BUILDER(Name("ZeroOut").Device(DEVICE_CPU), ZeroOutOp);
```

> Importante: Instâncias do seu OpKernel poderão ser acessadas simultaneamente, portanto seu método `Compute` precisa ser thread-safe. Proteja qualquer acesso aos membros da classe com uma trava mutex. Ou melhor, evite compartilhar estado entre membros da classe! Considere usar um [`ResourceMgr`](https://www.tensorflow.org/code/tensorflow/core/framework/resource_mgr.h) para acompanhar o estado do op.

### Kernels de CPU multithread

Para escrever um kernel de CPU multithread, a função Shard em [`work_sharder.h`](https://www.tensorflow.org/code/tensorflow/core/util/work_sharder.h) pode ser usada. Esta função fragmenta uma função de computação pelos threads configurados para serem usados ​​no paralelismo intra-op (veja intra_op_parallelism_threads em [`config.proto`](https://www.tensorflow.org/code/tensorflow/core/protobuf/config.proto)).

### Kernels de GPU

Um kernel de GPU é implementado em duas partes: o OpKernel e o kernel CUDA, e seu código de inicialização.

Às vezes, a implementação do OpKernel é comum entre um kernel de CPU e de GPU, como na inspeção de entradas e na alocação de saídas. Nesse caso, uma implementação sugerida é:

1. Defina o OpKernel em template no Device e o tipo primitivo do tensor.
2. Para realizar o cálculo real da saída, a função Compute chama uma struct de functor em template.
3. A especialização desse functor para CPUDevice é definida no mesmo arquivo, mas a especialização para GPUDevice é definida em um arquivo .cu.cc, já que será compilado com o compilador CUDA.

Aqui está um exemplo de implementação.

```c++
// kernel_example.h
#ifndef KERNEL_EXAMPLE_H_
#define KERNEL_EXAMPLE_H_

#include <unsupported/Eigen/CXX11/Tensor>

template <typename Device, typename T>
struct ExampleFunctor {
  void operator()(const Device& d, int size, const T* in, T* out);
};

#if GOOGLE_CUDA
// Partially specialize functor for GpuDevice.
template <typename T>
struct ExampleFunctor<Eigen::GpuDevice, T> {
  void operator()(const Eigen::GpuDevice& d, int size, const T* in, T* out);
};
#endif

#endif KERNEL_EXAMPLE_H_
```

```c++
// kernel_example.cc
#include "kernel_example.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

REGISTER_OP("Example")
    .Attr("T: numbertype")
    .Input("input: T")
    .Output("input_times_two: T")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });

// CPU specialization of actual computation.
template <typename T>
struct ExampleFunctor<CPUDevice, T> {
  void operator()(const CPUDevice& d, int size, const T* in, T* out) {
    for (int i = 0; i < size; ++i) {
      out[i] = 2 * in[i];
    }
  }
};

// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
template <typename Device, typename T>
class ExampleOp : public OpKernel {
 public:
  explicit ExampleOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));

    // Do the computation.
    OP_REQUIRES(context, input_tensor.NumElements() <= tensorflow::kint32max,
                errors::InvalidArgument("Too many elements in tensor"));
    ExampleFunctor<Device, T>()(
        context->eigen_device<Device>(),
        static_cast<int>(input_tensor.NumElements()),
        input_tensor.flat<T>().data(),
        output_tensor->flat<T>().data());
  }
};

// Register the CPU kernels.
#define REGISTER_CPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("Example").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      ExampleOp<CPUDevice, T>);
REGISTER_CPU(float);
REGISTER_CPU(int32);

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T)                                          \
  /* Declare explicit instantiations in kernel_example.cu.cc. */ \
  extern template class ExampleFunctor<GPUDevice, T>;            \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("Example").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      ExampleOp<GPUDevice, T>);
REGISTER_GPU(float);
REGISTER_GPU(int32);
#endif  // GOOGLE_CUDA
```

```c++
// kernel_example.cu.cc
#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "kernel_example.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;

// Define the CUDA kernel.
template <typename T>
__global__ void ExampleCudaKernel(const int size, const T* in, T* out) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
       i += blockDim.x * gridDim.x) {
    out[i] = 2 * __ldg(in + i);
  }
}

// Define the GPU implementation that launches the CUDA kernel.
template <typename T>
void ExampleFunctor<GPUDevice, T>::operator()(
    const GPUDevice& d, int size, const T* in, T* out) {
  // Launch the cuda kernel.
  //
  // See core/util/gpu_kernel_helper.h for example of computing
  // block count and thread_per_block count.
  int block_count = 1024;
  int thread_per_block = 20;
  ExampleCudaKernel<T>
      <<<block_count, thread_per_block, 0, d.stream()>>>(size, in, out);
}

// Explicitly instantiate functors for the types of OpKernels registered.
template struct ExampleFunctor<GPUDevice, float>;
template struct ExampleFunctor<GPUDevice, int32>;

#endif  // GOOGLE_CUDA
```

## Construa a biblioteca de ops

### Compile a op usando o compilador do sistema (instalação binária do TensorFlow)

Você deve ser capaz de compilar `zero_out.cc` com um compilador `C++` como `g++` ou `clang` disponível no seu sistema. O pacote binário PIP instala os arquivos de cabeçalho e a biblioteca necessária para compilar sua op em locais específicos do sistema. No entanto, a biblioteca Python do TensorFlow fornece a função `get_include` para obter o diretório de cabeçalho, e o diretório `get_lib` tem um objeto compartilhado que deve ser lincado. Aqui estão os resultados dessas funções em uma máquina Ubuntu.

```bash
$ python
>>> import tensorflow as tf
>>> tf.sysconfig.get_include()
'/usr/local/lib/python3.6/site-packages/tensorflow/include'
>>> tf.sysconfig.get_lib()
'/usr/local/lib/python3.6/site-packages/tensorflow'
```

Supondo que você tenha o `g++` instalado, aqui está a sequência de comandos que você pode usar para compilar sua op numa biblioteca dinâmica.

```bash
TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
g++ -std=c++14 -shared zero_out.cc -o zero_out.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2
```

No macOS, o sinalizador adicional "-undefined dynamic_lookup" é necessário ao criar o arquivo `.so`.

> Nota sobre a versão `gcc` `>=5`: o gcc usa o novo C++ [ABI](https://gcc.gnu.org/gcc-5/changes.html#libstdcxx) desde a versão `5`. O TensorFlow 2.8 e anteriores foram criados com `gcc4` que usa o ABI mais antigo. Se você estiver usando essas versões do TensorFlow e tentando compilar sua biblioteca operacional com `gcc>=5`, adicione `-D_GLIBCXX_USE_CXX11_ABI=0` à linha de comando para tornar a biblioteca compatível com o ABI mais antigo. Os pacotes do TensorFlow 2.9+ são compatíveis com o ABI mais recente por padrão.

### Compile a op usando bazel (instalação com fontes do TensorFlow)

Se você tiver o código-fonte do TensorFlow instalado, poderá usar o sistema de build do TensorFlow para compilar sua op. Coloque um arquivo BUILD com a seguinte regra de build do Bazel no diretório [`tensorflow/core/user_ops`](https://www.tensorflow.org/code/tensorflow/core/user_ops/).

```python
load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")

tf_custom_op_library(
    name = "zero_out.so",
    srcs = ["zero_out.cc"],
)
```

Execute o seguinte comando para criar `zero_out.so`.

```bash
$ bazel build --config opt //tensorflow/core/user_ops:zero_out.so
```

Para compilar a operação `Example`, com o Kernel CUDA, você precisa usar o parâmetro `gpu_srcs` de `tf_custom_op_library`. Coloque um arquivo BUILD com a seguinte regra de build do Bazel numa nova pasta dentro do diretório [`tensorflow/core/user_ops`](https://www.tensorflow.org/code/tensorflow/core/user_ops/) (por exemplo, "example_gpu").

```python
load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")

tf_custom_op_library(
    # kernel_example.cc  kernel_example.cu.cc  kernel_example.h
    name = "kernel_example.so",
    srcs = ["kernel_example.h", "kernel_example.cc"],
    gpu_srcs = ["kernel_example.cu.cc", "kernel_example.h"],
)
```

Execute o seguinte comando para criar o `kernel_example.so`.

```bash
$ bazel build --config opt //tensorflow/core/user_ops/example_gpu:kernel_example.so
```

Observação: Conforme explicado acima, se você estiver compilando com gcc&gt;=5 adicione `--cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0"` aos argumentos da linha de comando do Bazel.

> Nota: Embora você possa criar uma biblioteca compartilhada (um arquivo `.so`) com a regra padrão `cc_library`, recomendamos fortemente que você use a macro `tf_custom_op_library`. Ela adiciona algumas dependências necessárias e realiza verificações para garantir que a biblioteca compartilhada seja compatível com o mecanismo de carregamento de plug-ins do TensorFlow.

## Use a op em Python

A API Python do TensorFlow fornece a função `tf.load_op_library` para carregar a biblioteca dinâmica e registrar a operação com o framework TensorFlow. `load_op_library` retorna um módulo Python que contém os wrappers Python para a op e o kernel. Assim, depois de criar a op, você pode fazer o seguinte para executá-la no Python:

```python
import tensorflow as tf
zero_out_module = tf.load_op_library('./zero_out.so')
print(zero_out_module.zero_out([[1, 2], [3, 4]]).numpy())

# Prints
array([[1, 0], [0, 0]], dtype=int32)
```

Lembre-se de que a função gerada receberá um nome em formato snake_case (para conformidade com [PEP8](https://www.python.org/dev/peps/pep-0008/)). Portanto, se sua operação for chamada `ZeroOut` nos arquivos C++, a função Python será chamada `zero_out`.

Para disponibilizar a op como uma função regular importável via `import` em um módulo Python, talvez seja útil ter a chamada `load_op_library` em um arquivo fonte Python da seguinte forma:

```python
import tensorflow as tf

zero_out_module = tf.load_op_library('./zero_out.so')
zero_out = zero_out_module.zero_out
```

## Verifique se a op funciona

Uma boa maneira de verificar se você implementou sua op com sucesso é escrever um teste para ela. Crie o arquivo `zero_out_op_test.py` com o conteúdo:

```python
import tensorflow as tf

class ZeroOutTest(tf.test.TestCase):
  def testZeroOut(self):
    zero_out_module = tf.load_op_library('./zero_out.so')
    with self.test_session():
      result = zero_out_module.zero_out([5, 4, 3, 2, 1])
      self.assertAllEqual(result.eval(), [5, 0, 0, 0, 0])

if __name__ == "__main__":
  tf.test.main()
```

Em seguida, execute seu teste (supondo que você tenha o TensorFlow instalado):

```sh
$ python zero_out_op_test.py
```

## Crie recursos avançados na sua op

Agora que você sabe como construir uma op e implementação básicas (e um tanto restritas), veremos algumas das coisas mais complicadas que você normalmente precisará incorporar na sua op. Isto inclui:

- [Verificações condicionais e validação](#conditional-checks-and-validation)
- [Registro da op](#op-registration)
    - [Attrs](#attrs)
    - [Tipos de attr](#attr-types)
    - [Polimorfismo](#polymorphism)
    - [Entradas e saídas](#inputs-and-outputs)
    - [Compatibilidade reversa](#backwards-compatibility)
- [Suporte para GPU](#gpu-support)
    - [Compilando o kernel para o dispositivo GPU](#compiling-the-kernel-for-the-gpu-device)
- [Implementação do gradiente em Python](#implement-the-gradient-in-python)
- [Funções de formato em C++](#shape-functions-in-c)

### Verificações condicionais e validação

O exemplo acima assumiu que a op se aplicava a um tensor de qualquer formato. E se fosse aplicado apenas a vetores? Isso significa adicionar uma verificação à implementação do OpKernel acima.

```c++
  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);

    OP_REQUIRES(context, TensorShapeUtils::IsVector(input_tensor.shape()),
                errors::InvalidArgument("ZeroOut expects a 1-D vector."));
    // ...
  }
```

Isso afirma que a entrada é um vetor e retorna definindo o status `InvalidArgument` se não for. A [macro `OP_REQUIRES`](https://www.tensorflow.org/code/tensorflow/core/platform/errors.h) recebe três argumentos:

- O `context`, que tanto pode ser um ponteiro `OpKernelContext` como `OpKernelConstruction` (veja [`tensorflow/core/framework/op_kernel.h`](https://www.tensorflow.org/code/tensorflow/core/framework/op_kernel.h)), para seu método `SetStatus()`.
- A condição. Por exemplo, existem funções para validar o formato de um tensor em [`tensorflow/core/framework/tensor_shape.h`](https://www.tensorflow.org/code/tensorflow/core/framework/tensor_shape.h)
- O erro em si, que é representado por um objeto `Status`, veja [`tensorflow/core/platform/status.h`](https://www.tensorflow.org/code/tensorflow/core/platform/status.h). Um `Status` possui um tipo (geralmente `InvalidArgument`, mas veja a lista de tipos) e uma mensagem. Funções para construir um erro podem ser encontradas em [`tensorflow/core/platform/errors.h`](https://www.tensorflow.org/code/tensorflow/core/platform/errors.h).

Como alternativa, se você quiser testar se um objeto `Status` retornado de alguma função é um erro e, em caso afirmativo, retorná-lo, use [`OP_REQUIRES_OK`](https://www.tensorflow.org/code/tensorflow/core/platform/errors.h). Ambas as macros retornam da função em caso de erro.

### Registro da op

#### Attrs

As ops podem ter attrs, cujos valores são definidos quando a op é adicionada a um grafo. Eles são usados ​​para configurar a op e seus valores podem ser acessados ​​tanto na implementação do kernel quanto nos tipos de entradas e saídas no registro da op. Prefira usar uma entrada em vez de um attr quando possível, pois as entradas são mais flexíveis. Isso ocorre porque os attrs são constantes e precisam ser definidos no momento da construção do grafo. Por outro lado, as entradas são tensores cujos valores podem ser dinâmicos; isto é, as entradas podem mudar a cada passo, podem ser definidas usando um feed, etc. Attrs são usados ​​para coisas que não podem ser feitas com entradas: qualquer configuração que afete a assinatura (número ou tipo de entradas ou saídas) ou que não possa mudar a cada passo.

Você define um attr ao registrar a op, especificando seu nome e tipo usando o método `Attr`, que espera uma especificação da forma:

```
<name>: <attr-type-expr>
```

onde `<name>` começa com uma letra e pode ser composto por caracteres alfanuméricos e sublinhados, e `<attr-type-expr>` é uma expressão de tipo no formato [descrito abaixo](#attr-types).

Por exemplo, se você quiser que a op `ZeroOut` preserve um índice especificado pelo usuário, em vez de apenas o elemento 0, você pode registrar a op da seguinte forma:

```c++
REGISTER_OP("ZeroOut")
    .Attr("preserve_index: int")
    .Input("to_zero: int32")
    .Output("zeroed: int32");
```

(Observe que o conjunto de [tipos de atributos](#attr-types) é diferente do `tf.DType` usado para entradas e saídas.)

Seu kernel pode então acessar esse attr em seu construtor através do parâmetro `context`:

```c++
class ZeroOutOp : public OpKernel {
 public:
  explicit ZeroOutOp(OpKernelConstruction* context) : OpKernel(context) {
    // Get the index of the value to preserve
    OP_REQUIRES_OK(context,
                   context->GetAttr("preserve_index", &preserve_index_));
    // Check that preserve_index is positive
    OP_REQUIRES(context, preserve_index_ >= 0,
                errors::InvalidArgument("Need preserve_index >= 0, got ",
                                        preserve_index_));
  }
  void Compute(OpKernelContext* context) override {
    // ...
  }
 private:
  int preserve_index_;
};
```

que pode então ser usado no método `Compute`:

```c++
  void Compute(OpKernelContext* context) override {
    // ...

    // We're using saved attr to validate potentially dynamic input
    // So we check that preserve_index is in range
    OP_REQUIRES(context, preserve_index_ < input.dimension(0),
                errors::InvalidArgument("preserve_index out of range"));

    // Set all the elements of the output tensor to 0
    const int N = input.size();
    for (int i = 0; i < N; i++) {
      output_flat(i) = 0;
    }

    // Preserve the requested input value
    output_flat(preserve_index_) = input(preserve_index_);
  }
```

#### Tipos de attr

Os seguintes tipos são suportados em um attr:

- `string`: Qualquer sequência de bytes (não é necessário ser UTF8).
- `int`: Um inteiro com sinal.
- `float`: Um número de ponto flutuante.
- `bool`: True ou false.
- `type` : um dos valores (non-ref) de [`DataType`](https://www.tensorflow.org/code/tensorflow/core/framework/types.cc).
- `shape`: Um [`TensorShapeProto`](https://www.tensorflow.org/code/tensorflow/core/framework/tensor_shape.proto).
- `list(<type>)` : Uma lista de `<type>`, onde `<type>` é um dos tipos acima. Observe que `list(list(<type>))` é inválido.

Veja também: [`op_def_builder.cc:FinalizeAttr`](https://www.tensorflow.org/code/tensorflow/core/framework/op_def_builder.cc) para uma lista definitiva.

##### Valores padrão e restrições

Os attrs podem ter valores padrão e alguns tipos de attrs podem ter restrições. Para definir um attr com restrições, você pode usar os seguintes `<attr-type-expr>`:

`{'<string1>', '<string2>'}`: o valor deve ser uma string que tenha o valor `<string1>` ou `<string2>`. O nome do tipo, `string`, está implícito quando você usa essa sintaxe. Isto emula um enum:

```c++
REGISTER_OP("EnumExample")
    .Attr("e: {'apple', 'orange'}");
```

`{<type1>, <type2>}`: o valor é do tipo `type` e deve ser um `<type1>` ou um `<type2>`, onde `<type1>` e `<type2>` são tipos `tf.DType` suportados. Você não especifica que o tipo do attr é `type`. Isso fica implícito quando você tem uma lista de tipos em `{...}`. Por exemplo, neste caso o attr `t` é um tipo que precisa ser um `int32`, `float` ou `bool`:

```c++
REGISTER_OP("RestrictedTypeExample")
    .Attr("t: {int32, float, bool}");
```

Existem atalhos para restrições de tipo comuns:

- `numbertype`: tipo `type` restrito aos tipos numéricos (que não são string nem bool).
- `realnumbertype`: como `numbertype` sem tipos complexos.
- `quantizedtype`: como `numbertype`, mas apenas os tipos de números quantizados.

As listas específicas de tipos permitidos por estes são definidas pelas funções (como `NumberTypes()`) em [`tensorflow/core/framework/types.h`](https://www.tensorflow.org/code/tensorflow/core/framework/types.h). Neste exemplo, o attr `t` deve ser um dos tipos numéricos:

```c++
REGISTER_OP("NumberType")
    .Attr("t: numbertype");
```

Para esta op:

```python
tf.number_type(t=tf.int32)  # Valid
tf.number_type(t=tf.bool)   # Invalid
```

Listas podem ser combinadas com outras listas e tipos simples. A op a seguir permite que o attr `t` seja qualquer um dos tipos numéricos ou do tipo bool:

```c++
REGISTER_OP("NumberOrBooleanType")
    .Attr("t: {numbertype, bool}");
```

Para esta op:

```python
tf.number_or_boolean_type(t=tf.int32)  # Valid
tf.number_or_boolean_type(t=tf.bool)   # Valid
tf.number_or_boolean_type(t=tf.string) # Invalid
```

`int >= <n>`: O valor deve ser um int cujo valor seja maior ou igual a `<n>`, onde `<n>` é um número natural. Por exemplo, o registro da op a seguir especifica que o attr `a` deve ter um valor que seja pelo menos `2`:

```c++
REGISTER_OP("MinIntExample")
    .Attr("a: int >= 2");
```

`list(<type>) >= <n>`: Uma lista do tipo `<type>` cujo comprimento é maior ou igual a `<n>`. Por exemplo, o registro da op a seguir especifica que o attr `a` é uma lista de tipos (`int32` ou `float`) e que deve haver pelo menos 3 deles:

```c++
REGISTER_OP("TypeListExample")
    .Attr("a: list({int32, float}) >= 3");
```

Para definir um valor padrão para um attr (tornando-o opcional no código gerado), adicione `= <default>` no final, como em:

```c++
REGISTER_OP("AttrDefaultExample")
    .Attr("i: int = 0");
```

Além disso, tanto uma restrição quanto um valor padrão podem ser especificados:

```c++
REGISTER_OP("AttrConstraintAndDefaultExample")
    .Attr("i: int >= 1 = 1");
```

A sintaxe suportada do valor padrão é a que seria usada na representação proto da definição GraphDef resultante.

Eis alguns exemplos de como especificar um valor padrão para todos os tipos:

```c++
REGISTER_OP("AttrDefaultExampleForAllTypes")
   .Attr("s: string = 'foo'")
   .Attr("i: int = 0")
   .Attr("f: float = 1.0")
   .Attr("b: bool = true")
   .Attr("ty: type = DT_INT32")
   .Attr("sh: shape = { dim { size: 1 } dim { size: 2 } }")
   .Attr("te: tensor = { dtype: DT_INT32 int_val: 5 }")
   .Attr("l_empty: list(int) = []")
   .Attr("l_int: list(int) = [2, 3, 5, 7]");
```

Observe que os valores do tipo `type` usam `tf.DType`.

#### Polimorfismo

##### Polimorfismo de tipo

Para ops que podem receber diferentes tipos como entrada ou produzir diferentes tipos de saída, você pode especificar [um attr](#attrs) em [um tipo de entrada ou saída](#inputs-and-outputs) no registro da op. Normalmente você registraria um `OpKernel` para cada tipo suportado.

Por exemplo, se você quiser que a operação `ZeroOut` funcione com valores `float` além de valores `int32`, o registro da op deve ser algo similar a:

```c++
REGISTER_OP("ZeroOut")
    .Attr("T: {float, int32}")
    .Input("to_zero: T")
    .Output("zeroed: T");
```

O registro da op agora especifica que o tipo de entrada deve ser `float`, ou `int32`, e que sua saída será do mesmo tipo, já que ambas declaram o tipo `T`.

###### Nomes

Entradas, saídas e attrs geralmente devem receber nomes no formato snake_case. A única exceção são os attrs que são usados ​​como tipo de uma entrada ou de uma saída. Esses attrs podem ser inferidos quando a op é adicionada ao grafo e, portanto, não aparecem na função da op. Por exemplo, esta última definição de ZeroOut irá gerar uma função Python semelhante a:

```python
def zero_out(to_zero, name=None):
  """...
  Args:
    to_zero: A `Tensor`. Must be one of the following types:
        `float32`, `int32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `to_zero`.
  """
```

Se `to_zero` receber um tensor `int32`, então `T` será automaticamente definido como `int32` (bem, na verdade `DT_INT32`). Esses atributos inferidos recebem nomes em maiúscula ou em formato CamelCase.

Compare isto com uma op que possui um tipo attr que determina o tipo de saída:

```c++
REGISTER_OP("StringToNumber")
    .Input("string_tensor: string")
    .Output("output: out_type")
    .Attr("out_type: {float, int32} = DT_FLOAT");
    .Doc(R"doc(
Converts each string in the input Tensor to the specified numeric type.
)doc");
```

Neste caso, o usuário deve especificar o tipo de saída, como no Python gerado:

```python
def string_to_number(string_tensor, out_type=None, name=None):
  """Converts each string in the input Tensor to the specified numeric type.

  Args:
    string_tensor: A `Tensor` of type `string`.
    out_type: An optional `tf.DType` from: `tf.float32, tf.int32`.
      Defaults to `tf.float32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `out_type`.
  """
```

###### Exemplo de polimorfismo de tipo

```c++
#include "tensorflow/core/framework/op_kernel.h"

class ZeroOutInt32Op : public OpKernel {
  // as before
};

class ZeroOutFloatOp : public OpKernel {
 public:
  explicit ZeroOutFloatOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<float>();

    // Create an output tensor
    Tensor* output = NULL;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input_tensor.shape(), &output));
    auto output_flat = output->template flat<float>();

    // Set all the elements of the output tensor to 0
    const int N = input.size();
    for (int i = 0; i < N; i++) {
      output_flat(i) = 0;
    }

    // Preserve the first input value
    if (N > 0) output_flat(0) = input(0);
  }
};

// Note that TypeConstraint<int32>("T") means that attr "T" (defined
// in the op registration above) must be "int32" to use this template
// instantiation.
REGISTER_KERNEL_BUILDER(
    Name("ZeroOut")
    .Device(DEVICE_CPU)
    .TypeConstraint<int32>("T"),
    ZeroOutInt32Op);
REGISTER_KERNEL_BUILDER(
    Name("ZeroOut")
    .Device(DEVICE_CPU)
    .TypeConstraint<float>("T"),
    ZeroOutFloatOp);
```

Para preservar a [compatibilidade reversa](#backwards-compatibility), você deve especificar um [valor padrão](#default-values-and-constraints) ao adicionar um attr a uma op existente:

```c++
REGISTER_OP("ZeroOut")
  .Attr("T: {float, int32} = DT_INT32")
  .Input("to_zero: T")
  .Output("zeroed: T")
```

Vamos supor que você queira adicionar mais tipos, digamos `double`:

```c++
REGISTER_OP("ZeroOut")
    .Attr("T: {float, double, int32}")
    .Input("to_zero: T")
    .Output("zeroed: T");
```

Em vez de escrever outro `OpKernel` com código redundante como acima, muitas vezes você poderá usar um template do C++. Você ainda terá um registro de kernel (chamada `REGISTER_KERNEL_BUILDER`) por sobrecarga.

```c++
template <typename T>
class ZeroOutOp : public OpKernel {
 public:
  explicit ZeroOutOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<T>();

    // Create an output tensor
    Tensor* output = NULL;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input_tensor.shape(), &output));
    auto output_flat = output->template flat<T>();

    // Set all the elements of the output tensor to 0
    const int N = input.size();
    for (int i = 0; i < N; i++) {
      output_flat(i) = 0;
    }

    // Preserve the first input value
    if (N > 0) output_flat(0) = input(0);
  }
};

// Note that TypeConstraint<int32>("T") means that attr "T" (defined
// in the op registration above) must be "int32" to use this template
// instantiation.
REGISTER_KERNEL_BUILDER(
    Name("ZeroOut")
    .Device(DEVICE_CPU)
    .TypeConstraint<int32>("T"),
    ZeroOutOp<int32>);
REGISTER_KERNEL_BUILDER(
    Name("ZeroOut")
    .Device(DEVICE_CPU)
    .TypeConstraint<float>("T"),
    ZeroOutOp<float>);
REGISTER_KERNEL_BUILDER(
    Name("ZeroOut")
    .Device(DEVICE_CPU)
    .TypeConstraint<double>("T"),
    ZeroOutOp<double>);
```

Se você tiver mais do que algumas poucas sobrecargas, poderá colocar o registro numa macro.

```c++
#include "tensorflow/core/framework/op_kernel.h"

#define REGISTER_KERNEL(type)                                       \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("ZeroOut").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      ZeroOutOp<type>)

REGISTER_KERNEL(int32);
REGISTER_KERNEL(float);
REGISTER_KERNEL(double);

#undef REGISTER_KERNEL
```

Dependendo da lista de tipos para os quais você está registrando o kernel, você poderá usar uma macro fornecida por [`tensorflow/core/framework/register_types.h`](https://www.tensorflow.org/code/tensorflow/core/framework/register_types.h):

```c++
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"

REGISTER_OP("ZeroOut")
    .Attr("T: realnumbertype")
    .Input("to_zero: T")
    .Output("zeroed: T");

template <typename T>
class ZeroOutOp : public OpKernel { ... };

#define REGISTER_KERNEL(type)                                       \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("ZeroOut").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      ZeroOutOp<type>)

TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNEL);

#undef REGISTER_KERNEL
```

##### Entradas e saídas de listas

Além de poder aceitar ou produzir diferentes tipos, as ops podem consumir ou produzir um número variável de tensores.

No exemplo a seguir, o attr `T` contém uma *lista* de tipos e é usado como o tipo de `in` e `out`. A entrada e a saída são listas de tensores desse tipo (e o número e os tipos de tensores na saída são iguais aos da entrada, pois ambos declaram o tipo `T`).

```c++
REGISTER_OP("PolymorphicListExample")
    .Attr("T: list(type)")
    .Input("in: T")
    .Output("out: T");
```

Você também pode impor restrições sobre quais tipos podem ser especificados na lista. Neste próximo exemplo, a entrada é uma lista de tensores `float` e `double`. A operação aceita, por exemplo, tipos de entrada `(float, double, float)` e nesse caso o tipo de saída também seria `(float, double, float)`.

```c++
REGISTER_OP("ListTypeRestrictionExample")
    .Attr("T: list({float, double})")
    .Input("in: T")
    .Output("out: T");
```

Se você quiser que todos os tensores de uma lista sejam do mesmo tipo, você pode fazer algo como:

```c++
REGISTER_OP("IntListInputExample")
    .Attr("N: int")
    .Input("in: N * int32")
    .Output("out: int32");
```

Isto aceita uma lista de tensores `int32` e usa um `int` attr `N` para especificar o comprimento da lista.

Isso também pode ser feito com [polimorfismo de tipos](#type-polymorphism). No próximo exemplo, a entrada é uma lista de tensores (com comprimento `"N"`) do mesmo tipo (mas não especificado) (`"T"`), e a saída é um único tensor de tipo correspondente:

```c++
REGISTER_OP("SameListInputExample")
    .Attr("N: int")
    .Attr("T: type")
    .Input("in: N * T")
    .Output("out: T");
```

Por padrão, as listas de tensores têm um comprimento mínimo de 1. Você pode alterar esse padrão usando [uma restrição `">="` no attr correspondente](#default-values-and-constraints). Neste próximo exemplo, a entrada é uma lista de pelo menos 2 tensores `int32`:

```c++
REGISTER_OP("MinLengthIntListExample")
    .Attr("N: int >= 2")
    .Input("in: N * int32")
    .Output("out: int32");
```

A mesma sintaxe funciona com attrs `"list(type)"` :

```c++
REGISTER_OP("MinimumLengthPolymorphicListExample")
    .Attr("T: list(type) >= 3")
    .Input("in: T")
    .Output("out: T");
```

#### Entradas e saídas

Para resumir o que foi visto acima, o registro de uma op pode ter múltiplas entradas e saídas:

```c++
REGISTER_OP("MultipleInsAndOuts")
    .Input("y: int32")
    .Input("z: float")
    .Output("a: string")
    .Output("b: int32");
```

Cada especificação de entrada ou saída tem o formato:

```
<name>: <io-type-expr>
```

onde `<name>` começa com uma letra e pode ser composto por caracteres alfanuméricos e sublinhados. `<io-type-expr>` é uma das seguintes expressões de tipo:

- `<type>`, onde `<type>` é um tipo de entrada suportado (por exemplo, `float`, `int32`, `string`). Istso especifica um único tensor do tipo fornecido.

    Veja `tf.DType`.

    ```c++
    REGISTER_OP("BuiltInTypesExample")
        .Input("integers: int32")
        .Input("complex_numbers: complex64");
    ```

- `<attr-type>`, onde `<attr-type>` é o nome de um [Attr](#attrs) com tipo `type` ou `list(type)` (com uma possível restrição de tipo). Esta sintaxe permite [ops polimórficas](#polymorphism).

    ```c++
    REGISTER_OP("PolymorphicSingleInput")
        .Attr("T: type")
        .Input("in: T");

    REGISTER_OP("RestrictedPolymorphicSingleInput")
        .Attr("T: {int32, int64}")
        .Input("in: T");
    ```

    Referenciando um attr do tipo `list(type)` permite aceitar uma sequência de tensores.

    ```c++
    REGISTER_OP("ArbitraryTensorSequenceExample")
        .Attr("T: list(type)")
        .Input("in: T")
        .Output("out: T");

    REGISTER_OP("RestrictedTensorSequenceExample")
        .Attr("T: list({int32, int64})")
        .Input("in: T")
        .Output("out: T");
    ```

    Observe que a quantidade e os tipos de tensores na saída `out` são os mesmos que na entrada `in`, pois ambos são do tipo `T`.

- Para uma sequência de tensores do mesmo tipo: `<number> * <type>`, onde `<number>` é o nome de um [Attr](#attrs) de tipo `int`. O `<type>` pode ser `tf.DType` ou o nome de um attr de tipo `type`. Como exemplo do primeiro, esta op aceita uma lista de tensores `int32`:

    ```c++
    REGISTER_OP("Int32SequenceExample")
        .Attr("NumTensors: int")
        .Input("in: NumTensors * int32")
    ```

    Enquanto que esta op aceita uma lista de tensores de qualquer tipo, desde que sejam todos iguais:

    ```c++
    REGISTER_OP("SameTypeSequenceExample")
        .Attr("NumTensors: int")
        .Attr("T: type")
        .Input("in: NumTensors * T")
    ```

- Para uma referência a um tensor: `Ref(<type>)`, onde `<type>` é um dos tipos anteriores.

Qualquer attr usado no tipo de entrada será inferido. Por convenção, esses atributos inferidos usam nomes maiúsculos (como `T` ou `N`). Caso contrário, entradas, saídas e atributos terão nomes como parâmetros de função (por exemplo, `num_outputs`). Para mais detalhes, veja a [seção anterior sobre nomes](#naming).

Para mais detalhes, veja [`tensorflow/core/framework/op_def_builder.h`](https://www.tensorflow.org/code/tensorflow/core/framework/op_def_builder.h).

#### Compatibilidade reversa

Vamos supor que você escreveu uma op personalizada e a compartilhou com outras pessoas, portanto você tem clientes satisfeitos usando sua operação. No entanto, você gostaria de fazer alterações na op de alguma forma.

Em geral, as alterações em especificações existentes devem ser compatíveis com versões anteriores: alterar a especificação de uma op não deveria quebrar os buffers de protocolo `GraphDef` previamente serializados e construídos a partir de especificações mais antigas. Os detalhes da compatibilidade com `GraphDef` são [descritos aqui](./versions.md#compatibility_of_graphs_and_checkpoints).

Há várias maneiras de preservar a compatibilidade reversa.

1. Quaisquer novos attrs adicionados a uma operação devem ter valores padrão definidos e, com esse valor padrão, a op deve preservar seu comportamento original. Para alterar uma operação de não polimórfica para polimórfica, você *precisa* fornecer um valor padrão ao novo tipo attr para preservar a assinatura original por padrão. Por exemplo, se sua operação foi:

    ```c++
    REGISTER_OP("MyGeneralUnaryOp")
        .Input("in: float")
        .Output("out: float");
    ```

    você pode torná-la polimórfica de maneira compatível com versões anteriores usando:

    ```c++
    REGISTER_OP("MyGeneralUnaryOp")
        .Input("in: T")
        .Output("out: T")
        .Attr("T: numerictype = DT_FLOAT");
    ```

2. Você pode seguramente tornar uma restrição de um attr menos restritiva. Por exemplo, você pode mudar de `{int32, int64}` para `{int32, int64, float}` ou `type`. Ou você pode mudar de `{"apple", "orange"}` para `{"apple", "banana", "orange"}` ou `string`.

3. Você pode alterar entradas/saídas simples para entradas/saídas de lista, desde que o padrão para o tipo de lista corresponda à assinatura antiga.

4. Você pode adicionar uma nova entrada/saída de lista, se o padrão for vazio.

5. Associe um namespace a todas as novas ops que você criar, prefixando os nomes das ops com algo que seja exclusivo do seu projeto. Isso evitará que sua op colida com quaisquer outras ops que possam ser incluídas em versões futuras do TensorFlow.

6. Planeje com antecedência! Tente antecipar usos futuros para a op. Algumas alterações de assinatura não podem ser feitas de maneira compatível (por exemplo, transformar uma lista de tipos iguais em uma lista de tipos variados).

A lista completa de alterações seguras e inseguras pode ser encontrada em [`tensorflow/core/framework/op_compatibility_test.cc`](https://www.tensorflow.org/code/tensorflow/core/framework/op_compatibility_test.cc). Se você não conseguir fazer a alteração para uma operação compatível com versões anteriores, crie uma nova operação com um novo nome e a nova semântica.

Observe também que, embora essas alterações possam manter a compatibilidade com `GraphDef`, o código Python gerado pode mudar de uma forma que não é compatível com os chamadores antigos. A API Python pode ser mantida compatível por meio de alterações cuidadosas em um wrapper Python escrito manualmente, mantendo a assinatura antiga, exceto possivelmente adicionando novos argumentos opcionais ao final. Geralmente, alterações incompatíveis só podem ser feitas quando o TensorFlow altera versões principais e devem estar em conformidade com a <a href="./versions.md#compatibility_of_graphs_and_checkpoints" data-md-type="link">semântica de versão do `GraphDef`</a>.

### Suporte para GPU

Você pode implementar diferentes OpKernels e registrar um para CPUs e outro para GPUs, assim como você pode [registrar kernels para diferentes tipos](#polymorphism). Existem vários exemplos de kernels com suporte a GPU em [`tensorflow/core/kernels/`](https://www.tensorflow.org/code/tensorflow/core/kernels/). Observe que alguns kernels têm uma versão CPU em um arquivo `.cc`, uma versão GPU em um arquivo que termina em `_gpu.cu.cc` e algum código compartilhado em comum em um arquivo `.h`.

Por exemplo, o `tf.pad` tem tudo, menos o kernel da GPU em [`tensorflow/core/kernels/pad_op.cc`](https://www.tensorflow.org/code/tensorflow/core/kernels/pad_op.cc). O kernel da GPU está em [`tensorflow/core/kernels/pad_op_gpu.cu.cc`](https://www.tensorflow.org/code/tensorflow/core/kernels/pad_op_gpu.cu.cc) e o código compartilhado é uma classe de modelo definida em [`tensorflow/core/kernels/pad_op.h`](https://www.tensorflow.org/code/tensorflow/core/kernels/pad_op.h). Organizamos o código dessa maneira por dois motivos: permite compartilhar código comum entre as implementações de CPU e GPU e coloca a implementação de GPU em um arquivo separado para que possa ser compilada apenas pelo compilador de GPU.

Uma coisa a ser observada: mesmo quando a versão do kernel da GPU do `pad` for usada, ele ainda precisa de sua entrada de preenchimentos `"paddings"` na memória da CPU. Para marcar que entradas ou saídas são mantidas na CPU, adicione uma chamada a `HostMemory()` no registro do kernel, por exemplo:

```c++
#define REGISTER_GPU_KERNEL(T)                         \
  REGISTER_KERNEL_BUILDER(Name("Pad")                  \
                              .Device(DEVICE_GPU)      \
                              .TypeConstraint<T>("T")  \
                              .HostMemory("paddings"), \
                          PadOp<GPUDevice, T>)
```

#### Compilando o kernel para o dispositivo GPU

Veja [cuda_op_kernel.cu.cc](https://www.tensorflow.org/code/tensorflow/examples/adding_an_op/cuda_op_kernel.cu.cc) para um exemplo que usa um kernel CUDA para implementar uma op. A `tf_custom_op_library` aceita um argumento `gpu_srcs` onde a lista de arquivos-fonte contendo os kernels CUDA (arquivos `*.cu.cc`) pode ser especificada. Para uso com uma instalação binária do TensorFlow, os kernels CUDA devem ser compilados com o compilador `nvcc` NVIDIA. Aqui está a sequência de comandos que você pode usar para compilar o [cuda_op_kernel.cu.cc](https://www.tensorflow.org/code/tensorflow/examples/adding_an_op/cuda_op_kernel.cu.cc) e o [cuda_op_kernel.cc](https://www.tensorflow.org/code/tensorflow/examples/adding_an_op/cuda_op_kernel.cc) numa única biblioteca carregável dinamicamente:

```bash
nvcc -std=c++14 -c -o cuda_op_kernel.cu.o cuda_op_kernel.cu.cc \
  ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

g++ -std=c++14 -shared -o cuda_op_kernel.so cuda_op_kernel.cc \
  cuda_op_kernel.cu.o ${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]}
```

O `cuda_op_kernel.so` produzido acima pode ser carregado normalmente em Python, usando a função `tf.load_op_library`.

Observe que se suas bibliotecas CUDA não estiverem instaladas em `/usr/local/lib64`, você precisará especificar o caminho explicitamente no segundo comando (g++) acima. Por exemplo, adicione `-L /usr/local/cuda-8.0/lib64/` se seu CUDA estiver instalado em `/usr/local/cuda-8.0`.

Observação: Em algumas configurações do Linux, são necessárias opções adicionais para a etapa de compilação `nvcc`. Adicione `-D_MWAITXINTRIN_H_INCLUDED` à linha de comando `nvcc` para evitar erros de `mwaitxintrin.h`.

### Implementação do gradiente em Python

Given a graph of ops, TensorFlow uses automatic differentiation (retropropagação) to add new ops representing gradients with respect to the existing ops. To make automatic differentiation work for new ops, you must register a gradient function which computes gradients with respect to the ops' inputs given gradients with respect to the ops' outputs.

Matematicamente, se uma op calcula \(y = f(x)\) a op de gradiente registrada converte gradientes \(\partial L/ \partial y\) de perda \(L\) em relação a \(y\) em gradientes \(\partial L/ \partial x\) em relação a \(x\) através da regra da cadeia:

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \frac{\partial y}{\partial x} = \frac{\partial L}{\partial y} \frac{\partial f}{\partial x}.$$

No caso da `ZeroOut`, apenas um campo da entrada afeta a saída, portanto o gradiente em relação à entrada é um tensor esparso "one hot". Isso é expresso da seguinte forma:

```python
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops

@ops.RegisterGradient("ZeroOut")
def _zero_out_grad(op, grad):
  """The gradients for `zero_out`.

  Args:
    op: The `zero_out` `Operation` that we are differentiating, which we can use
      to find the inputs and outputs of the original op.
    grad: Gradient with respect to the output of the `zero_out` op.

  Returns:
    Gradients with respect to the input of `zero_out`.
  """
  to_zero = op.inputs[0]
  shape = array_ops.shape(to_zero)
  index = array_ops.zeros_like(shape)
  first_grad = array_ops.reshape(grad, [-1])[0]
  to_zero_grad = sparse_ops.sparse_to_dense([index], shape, first_grad, 0)
  return [to_zero_grad]  # List of one Tensor, since we have one input
```

Detalhes sobre o registro de funções gradientes com `tf.RegisterGradient`:

- Para uma op com uma saída, a função gradiente receberá um `tf.Operation`, `op` e um `tf.Tensor` `grad` e criará novas operações a partir dos tensores `op.inputs[i]`, `op.outputs[i]` e `grad`. Informações sobre quaisquer atributos podem ser encontradas em `tf.Operation.get_attr`.

- Se a op tiver múltiplas saídas, a função gradiente usará `op` e `grads`, onde `grads` é uma lista de gradientes em relação a cada saída. O resultado da função gradiente deve ser uma lista de objetos `Tensor` representando os gradientes em relação a cada entrada.

- Se não houver um gradiente bem definido para alguma entrada, como para entradas inteiras usadas como índices, o gradiente retornado correspondente deverá ser `None`. Por exemplo, para uma operação que usa um tensor de ponto flutuante `x` e um índice inteiro `i`, a função gradiente iria retornar None (`return [x_grad, None]`).

- Se não houver nenhum gradiente significativo para a op, muitas vezes você não precisará registrar nenhum gradiente e, desde que o gradiente da operação nunca seja necessário, você ficará bem. Em alguns casos, uma op não possui um gradiente bem definido, mas ela pode estar envolvida no cálculo do gradiente. Aqui você pode usar `ops.NotDifferentiable` para propagar zeros automaticamente para trás.

Observe que no momento em que a função gradiente é chamada, apenas o gráfico de fluxo de dados das ops estará disponível, e não os dados do tensor em si. Assim, toda a computação deve ser realizada usando outras ops do TensorFlow, para serem executadas no tempo de execução do grafo.

Adicione dicas de tipo ao registrar o gradiente personalizado para um tipo de op para deixar o código mais legível, depurável, mais fácil de manter e mais robusto através da validação de dados. Por exemplo, ao usar um `op` como parâmetro numa função, especifique que a função gradiente usará uma <a href="https://www.tensorflow.org/api_docs/python/tf/Operation"><code>tf.Operation</code></a> como tipo de parâmetro.

### Funções de formato em C++

A API TensorFlow possui um recurso chamado "inferência de formato" que fornece informações sobre os formatos dos tensores sem a necessidade de executar o grafo. A inferência de formato é suportada por "funções de formato" que são registradas para cada tipo de op na declaração C++ `REGISTER_OP` e desempenham duas funções: afirmar que os formatos das entradas são compatíveis durante a construção do grafo e especificar os formatos das saídas.

As funções de formato são definidas como operações na classe `shape_inference::InferenceContext`. Por exemplo, na função de formato para ZeroOut:

```c++
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });
```

`c->set_output(0, c->input(0));` declara que o formato da primeira saída deve ser definido como o formato da primeira entrada. Se a saída for selecionada por seu índice como no exemplo acima, o segundo parâmetro de `set_output` deverá ser um objeto `ShapeHandle`. Você pode criar um objeto `ShapeHandle` vazio através do seu construtor padrão. O objeto `ShapeHandle` para uma entrada com índice `idx` pode ser obtido por `c->input(idx)`.

Há uma série de funções de formato comuns que se aplicam a muitas ops, como `shape_inference::UnchangedShape` que pode ser encontrada em [common_shape_fns.h](https://www.tensorflow.org/code/tensorflow/core/framework/common_shape_fns.h) e usada da seguinte forma:

```c++
REGISTER_OP("ZeroOut")
    .Input("to_zero: int32")
    .Output("zeroed: int32")
    .SetShapeFn(::tensorflow::shape_inference::UnchangedShape);
```

Uma função de formato também pode restringir o formato de uma entrada. Para a versão do [`ZeroOut` com restrição de formato vetorial](#conditional-checks-and-validation), a função de formato seria a seguinte:

```c++
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      ::tensorflow::shape_inference::ShapeHandle input;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &input));
      c->set_output(0, input);
      return Status::OK();
    });
```

A chamada `WithRank` valida que o formato de entrada `c->input(0)` tem um formato com exatamente uma dimensão (ou se o formato de entrada for desconhecido, o formato de saída será um vetor com uma dimensão desconhecida).

Se sua op for [polimórfica com múltiplas entradas](#polymorphism), você pode usar membros de `InferenceContext` para determinar o número de formatos a serem verificados e `Merge` para validar se aos formatos são todos compatíveis (como alternativa, acesse atributos que indicam os comprimentos, com `InferenceContext::GetAttr`, que fornece acesso aos atributos da op).

```c++
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      ::tensorflow::shape_inference::ShapeHandle input;
      ::tensorflow::shape_inference::ShapeHandle output;
      for (size_t i = 0; i < c->num_inputs(); ++i) {
        TF_RETURN_IF_ERROR(c->WithRank(c->input(i), 2, &input));
        TF_RETURN_IF_ERROR(c->Merge(output, input, &output));
      }
      c->set_output(0, output);
      return Status::OK();
    });
```

Já que a inferência de formato é um recurso opcional e os formatos dos tensores podem variar dinamicamente, as funções de formato devem ser robustas para informações de formato incompletas para qualquer uma das entradas. O método `Merge` em [`InferenceContext`](https://www.tensorflow.org/code/tensorflow/core/framework/shape_inference.h) permite que o chamador afirme que dois formatos são iguais, mesmo que uma ou ambas não tenham informações completas. As funções de formato são definidas para todas as ops principais do TensorFlow e fornecem muitos exemplos de uso diferentes.

A classe `InferenceContext` possui diversas funções que podem ser usadas para definir manipulações de funções de formato. Por exemplo, você pode validar se uma dimensão específica tem um valor muito específico usando `InferenceContext::Dim` e `InferenceContext::WithValue`; você pode especificar que uma dimensão de saída é a soma/produto de duas dimensões de entrada usando `InferenceContext::Add` e `InferenceContext::Multiply`. Consulte a classe `InferenceContext` para todas as diversas manipulações de formatos que você pode especificar. O exemplo a seguir define o formato da primeira saída como (n, 3), onde a primeira entrada tem formato (n, ...)

```c++
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    c->set_output(0, c->Matrix(c->Dim(c->input(0), 0), 3));
    return Status::OK();
});
```

Se você tiver uma função de formato complicada, considere adicionar um teste para validar se várias combinações de formatos de entrada produzem as combinações de formatos de saída esperadas. Você pode ver exemplos de como escrever esses testes em alguns de nossos [principais testes de operações](https://www.tensorflow.org/code/tensorflow/core/ops/array_ops_test.cc). (A sintaxe de `INFER_OK` e `INFER_ERROR` é um pouco enigmática, mas tente ser compacto ao representar as especificações de formato de entrada e saída em testes. Por enquanto, consulte os comentários em volta desses testes para ter uma noção da especificação da string de formato).

## Crie um pacote pip para sua op personalizada

Para construir um pacote `pip` para sua op, veja o exemplo [tensorflow/custom-op](https://github.com/tensorflow/custom-op). Esse guia mostra como criar ops personalizadas a partir do pacote pip do TensorFlow em vez de criar o TensorFlow a partir do código-fonte.
