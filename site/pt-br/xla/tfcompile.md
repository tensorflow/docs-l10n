# Usando compilação AOT

## O que é tfcompile?

O `tfcompile` é uma ferramenta autônoma que compila grafos do TensorFlow antecipadamente (AOT) em código executável. Ele pode reduzir o tamanho binário total e também evitar algumas sobrecargas de tempo de execução. Um caso de uso típico de `tfcompile` é compilar um grafo de inferência em código executável para dispositivos móveis.

O grafo do TensorFlow normalmente é executado pelo runtime do TensorFlow. Isso incorre em alguma sobrecarga de tempo de execução para a execução de cada nó do grafo. Isso também leva a um tamanho total maior de código binário, já que o código do runtime do TensorFlow precisa estar disponível, além do próprio grafo. O código executável produzido por `tfcompile` não usa o runtime do TensorFlow e só possui dependências de kernels que são realmente usados ​​na computação.

O compilador é construído sobre o framework XLA. O código que liga o TensorFlow ao framework XLA reside em [tensorflow/compiler](https://www.tensorflow.org/code/tensorflow/compiler/).

## O que o tfcompile faz?

O `tfcompile` pega um subgrafo, identificado pelos conceitos de feeds e fetches do TensorFlow, e gera uma função que implementa esse subgrafo. Os `feeds` são argumentos de entrada da função e os `fetches` são os argumentos de saída da função. Todas as entradas devem ser totalmente especificadas pelos feeds; o subgrafo removido resultante não pode conter nós Placeholder ou Variable. É comum especificar todos os espaços reservados e variáveis ​​como feeds, o que garante que o subgrafo resultante não contenha mais esses nós. A função gerada é empacotada como `cc_library`, com um arquivo de cabeçalho exportando a assinatura da função e um arquivo objeto contendo a implementação. O usuário escreve código para chamar a função gerada conforme apropriado.

## Usando tfcompile

Esta seção detalha passos de alto nível para gerar um binário executável com `tfcompile` a partir de um subgrafo do TensorFlow. Os passos são:

- Passo 1: configure o subgrafo para compilar
- Passo 2: use a macro de build `tf_library` para compilar o subgrafo
- Passo 3: escreva o código para chamar o subgrafo
- Passo 4: crie o binário final

### Passo 1: configure o subgrafo para compilar

Identifique os feeds e fetches que correspondem aos argumentos de entrada e saída da função gerada. Em seguida, configure os `feeds` e `fetches` em um proto [`tensorflow.tf2xla.Config`](https://www.tensorflow.org/code/tensorflow/compiler/tf2xla/tf2xla.proto).

```textproto
# Each feed is a positional input argument for the generated function.  The order
# of each entry matches the order of each input argument.  Here “x_hold” and “y_hold”
# refer to the names of placeholder nodes defined in the graph.
feed {
  id { node_name: "x_hold" }
  shape {
    dim { size: 2 }
    dim { size: 3 }
  }
}
feed {
  id { node_name: "y_hold" }
  shape {
    dim { size: 3 }
    dim { size: 2 }
  }
}

# Each fetch is a positional output argument for the generated function.  The order
# of each entry matches the order of each output argument.  Here “x_y_prod”
# refers to the name of a matmul node defined in the graph.
fetch {
  id { node_name: "x_y_prod" }
}
```

### Passo 2: use a macro de build tf_library para compilar o subgrafo

Este passo converte o grafo numa `cc_library` usando a macro de build `tf_library`. A `cc_library` consiste num arquivo objeto contendo o código gerado a partir do grafo, junto com um arquivo de cabeçalho que dá acesso ao código gerado. `tf_library` utiliza `tfcompile` para compilar o grafo do TensorFlow em código executável.

```build
load("//tensorflow/compiler/aot:tfcompile.bzl", "tf_library")

# Use the tf_library macro to compile your graph into executable code.
tf_library(
    # name is used to generate the following underlying build rules:
    # <name>           : cc_library packaging the generated header and object files
    # <name>_test      : cc_test containing a simple test and benchmark
    # <name>_benchmark : cc_binary containing a stand-alone benchmark with minimal deps;
    #                    can be run on a mobile device
    name = "test_graph_tfmatmul",
    # cpp_class specifies the name of the generated C++ class, with namespaces allowed.
    # The class will be generated in the given namespace(s), or if no namespaces are
    # given, within the global namespace.
    cpp_class = "foo::bar::MatMulComp",
    # graph is the input GraphDef proto, by default expected in binary format.  To
    # use the text format instead, just use the ‘.pbtxt’ suffix.  A subgraph will be
    # created from this input graph, with feeds as inputs and fetches as outputs.
    # No Placeholder or Variable ops may exist in this subgraph.
    graph = "test_graph_tfmatmul.pb",
    # config is the input Config proto, by default expected in binary format.  To
    # use the text format instead, use the ‘.pbtxt’ suffix.  This is where the
    # feeds and fetches were specified above, in the previous step.
    config = "test_graph_tfmatmul.config.pbtxt",
)
```

> Para gerar o proto GraphDef (test_graph_tfmatmul.pb) para este exemplo, execute [make_test_graphs.py](https://www.tensorflow.org/code/tensorflow/compiler/aot/tests/make_test_graphs.py) e especifique o local de saída com o sinalizador --out_dir.

Grafos típicos contêm [`Variables`](https://www.tensorflow.org/guide/variables) que representam os pesos que são aprendidos por meio de treinamento, mas `tfcompile` não pode compilar um subgrafo que contenha `Variables`. A ferramenta [freeze_graph.py](https://www.tensorflow.org/code/tensorflow/python/tools/freeze_graph.py) converte variáveis ​​em constantes, usando valores armazenados em um arquivo de checkpoint. Por conveniência, a macro `tf_library` suporta o argumento `freeze_checkpoint`, que executa a ferramenta. Para mais exemplos, consulte [tensorflow/compiler/aot/tests/BUILD](https://www.tensorflow.org/code/tensorflow/compiler/aot/tests/BUILD) .

> As constantes que aparecem no subgrafo compilado são compiladas diretamente no código gerado. Para passar as constantes para a função gerada, em vez de compilá-las, simplesmente passe-as como feeds.

Para detalhes sobre a macro de build `tf_library`, veja [tfcompile.bzl](https://www.tensorflow.org/code/tensorflow/compiler/aot/tfcompile.bzl).

Para detalhes sobre a ferramenta `tfcompile` nativa, veja [tfcompile_main.cc](https://www.tensorflow.org/code/tensorflow/compiler/aot/tfcompile_main.cc).

### Passo 3: escreva o código para chamar o subgrafo

Este passo usa o arquivo de cabeçalho (`test_graph_tfmatmul.h`) gerado pela macro de build `tf_library` no passo anterior para chamar o código gerado. O arquivo de cabeçalho está localizado no diretório `bazel-bin` correspondente ao pacote de build e é nomeado com base no atributo name definido na macro de build `tf_library`. Por exemplo, o cabeçalho gerado para `test_graph_tfmatmul` seria `test_graph_tfmatmul.h`. Abaixo está uma versão abreviada do que é gerado. O arquivo gerado, em `bazel-bin`, contém comentários úteis adicionais.

```c++
namespace foo {
namespace bar {

// MatMulComp represents a computation previously specified in a
// TensorFlow graph, now compiled into executable code.
class MatMulComp {
 public:
  // AllocMode controls the buffer allocation mode.
  enum class AllocMode {
    ARGS_RESULTS_AND_TEMPS,  // Allocate arg, result and temp buffers
    RESULTS_AND_TEMPS_ONLY,  // Only allocate result and temp buffers
  };

  MatMulComp(AllocMode mode = AllocMode::ARGS_RESULTS_AND_TEMPS);
  ~MatMulComp();

  // Runs the computation, with inputs read from arg buffers, and outputs
  // written to result buffers. Returns true on success and false on failure.
  bool Run();

  // Arg methods for managing input buffers. Buffers are in row-major order.
  // There is a set of methods for each positional argument.
  void** args();

  void set_arg0_data(float* data);
  float* arg0_data();
  float& arg0(size_t dim0, size_t dim1);

  void set_arg1_data(float* data);
  float* arg1_data();
  float& arg1(size_t dim0, size_t dim1);

  // Result methods for managing output buffers. Buffers are in row-major order.
  // Must only be called after a successful Run call. There is a set of methods
  // for each positional result.
  void** results();


  float* result0_data();
  float& result0(size_t dim0, size_t dim1);
};

}  // end namespace bar
}  // end namespace foo
```

A classe C++ gerada é chamada `MatMulComp` no namespace `foo::bar`, porque essa era a `cpp_class` especificada na macro `tf_library`. Todas as classes geradas possuem uma API semelhante, com a única diferença sendo os métodos para lidar com buffers de argumentos e resultados. Esses métodos diferem com base no número e nos tipos de buffers, que foram especificados pelos argumentos `feed` e `fetch` para a macro `tf_library`.

Há três tipos de buffers gerenciados na classe gerada: `args` representando as entradas, `results` representando as saídas e `temps` representando buffers temporários usados ​​internamente para realizar o cálculo. Por padrão, cada instância da classe gerada aloca e gerencia todos esses buffers para você. O argumento do construtor `AllocMode` pode ser usado para alterar esse comportamento. Todos os buffers estão alinhados aos limites de 64 bytes.

A classe C++ gerada é apenas um wrapper em torno do código de baixo nível gerado pelo XLA.

Exemplo de chamada da função gerada com base em [`tfcompile_test.cc`](https://www.tensorflow.org/code/tensorflow/compiler/aot/tests/tfcompile_test.cc):

```c++
#define EIGEN_USE_THREADS
#define EIGEN_USE_CUSTOM_THREAD_POOL

#include <iostream>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/compiler/aot/tests/test_graph_tfmatmul.h" // generated

int main(int argc, char** argv) {
  Eigen::ThreadPool tp(2);  // Size the thread pool as appropriate.
  Eigen::ThreadPoolDevice device(&tp, tp.NumThreads());


  foo::bar::MatMulComp matmul;
  matmul.set_thread_pool(&device);

  // Set up args and run the computation.
  const float args[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::copy(args + 0, args + 6, matmul.arg0_data());
  std::copy(args + 6, args + 12, matmul.arg1_data());
  matmul.Run();

  // Check result
  if (matmul.result0(0, 0) == 58) {
    std::cout << "Success" << std::endl;
  } else {
    std::cout << "Failed. Expected value 58 at 0,0. Got:"
              << matmul.result0(0, 0) << std::endl;
  }

  return 0;
}
```

### Passo 4: crie o binário final

Este passo combina a biblioteca gerada por `tf_library` no passo 2 e o código escrito no passo 3 para criar um arquivo binário final. Abaixo está um exemplo de arquivo BUILD do `bazel`.

```build
# Example of linking your binary
# Also see //tensorflow/compiler/aot/tests/BUILD
load("//tensorflow/compiler/aot:tfcompile.bzl", "tf_library")

# The same tf_library call from step 2 above.
tf_library(
    name = "test_graph_tfmatmul",
    ...
)

# The executable code generated by tf_library can then be linked into your code.
cc_binary(
    name = "my_binary",
    srcs = [
        "my_code.cc",  # include test_graph_tfmatmul.h to access the generated header
    ],
    deps = [
        ":test_graph_tfmatmul",  # link in the generated object file
        "//third_party/eigen3",
    ],
    linkopts = [
          "-lpthread",
    ]
)
```
