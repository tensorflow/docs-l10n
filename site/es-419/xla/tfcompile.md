# Cómo usar la compilación AOT

## ¿Qué es tfcompile?

`tfcompile` es una herramienta independiente que compila de forma anticipada (AOT) grafos de TensorFlow en código ejecutable. Puede reducir el tamaño binario total y también evitar algunos gastos generales de tiempo de ejecución. Un caso de uso típico de `tfcompile` es compilar un grafo de inferencia en código ejecutable para dispositivos móviles.

El grafo de TensorFlow normalmente se ejecuta mediante el tiempo de ejecución de TensorFlow. Esto genera cierta sobrecarga de tiempo de ejecución para la ejecución de cada nodo en el grafo. Esto también conduce a un mayor tamaño binario total, ya que el código para el tiempo de ejecución de TensorFlow debe estar disponible, además del grafo en sí. El código ejecutable que produce `tfcompile` no usa el tiempo de ejecución de TensorFlow y solo depende de los núcleos que realmente se usan en el cálculo.

El compilador se basa en el marco de XLA. El código que une TensorFlow con el marco de XLA reside en [tensorflow/compiler](https://www.tensorflow.org/code/tensorflow/compiler/).

## ¿Qué hace tfcompile?

`tfcompile` toma un subgrafo, identificado por los conceptos de fuentes y extracciones de TensorFlow, y genera una función que implementa ese subgrafo. Las `feeds` son los argumentos de entrada de la función y las `fetches` son los argumentos de salida de la función. Todas las entradas deben estar completamente especificadas por las fuentes; el subgrafo podado resultante no puede contener nodos de marcador de posición o de variable. Es común especificar todos los marcadores de posición y variables como fuentes, lo que garantiza que el subgrafo resultante ya no contenga estos nodos. La función generada se empaqueta como `cc_library`, con un archivo de encabezado que exporta la firma de la función y un archivo objeto que contiene la implementación. El usuario escribe código para invocar la función generada según corresponda.

## Cómo usar tfcompile

En esta sección se detallan los pasos de alto nivel necesarios para generar un binario ejecutable con `tfcompile` a partir de un subgrafo de TensorFlow. Los pasos son los siguientes:

- Paso 1: configure el subgrafo para compilar
- Paso 2: use la macro de compilación `tf_library` para compilar el subgrafo
- Paso 3: escriba el código para invocar el subgrafo
- Paso 4: cree el binario final

### Paso 1: configure el subgrafo para compilar

Identifique las fuentes y las extracciones que corresponden a los argumentos de entrada y salida de la función generada. Luego, configure `feeds` y `fetches` en un protocolo [`tensorflow.tf2xla.Config`](https://www.tensorflow.org/code/tensorflow/compiler/tf2xla/tf2xla.proto).

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

### Paso 2: use la macro de compilación tf_library para compilar el subgrafo

En este paso se usa la macro de compilación `tf_library` para convertir el grafo en `cc_library`. `cc_library` consta de un archivo objeto que contiene el código generado a partir del grafo, junto con un archivo de encabezado que da acceso al código generado. `tf_library` usa `tfcompile` para compilar el grafo de TensorFlow en código ejecutable.

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

> Para generar el protocolo GraphDef (test_graph_tfmatmul.pb) para este ejemplo, ejecute [make_test_graphs.py](https://www.tensorflow.org/code/tensorflow/compiler/aot/tests/make_test_graphs.py) y especifique la ubicación de salida con la marca --out_dir.

Los grafos típicos contienen [`Variables`](https://www.tensorflow.org/guide/variables) que representan las ponderaciones que se aprenden a través del entrenamiento, pero `tfcompile` no puede compilar un subgrafo que contenga `Variables`. La herramienta [frozen_graph.py](https://www.tensorflow.org/code/tensorflow/python/tools/freeze_graph.py) usa valores almacenados en un archivo de punto de control para convertir variables en constantes. Para su comodidad, la macro `tf_library` admite el argumento `freeze_checkpoint`, que ejecuta la herramienta. Para obtener más ejemplos, consulte [tensorflow/compiler/aot/tests/BUILD](https://www.tensorflow.org/code/tensorflow/compiler/aot/tests/BUILD).

> Las constantes que aparecen en el subgrafo compilado se compilan directamente en el código generado. Para pasar las constantes a la función generada, en lugar de compilarlas, simplemente páselas como fuentes.

Para obtener más información sobre la macro de compilación `tf_library`, consulte[tfcompile.bzl](https://www.tensorflow.org/code/tensorflow/compiler/aot/tfcompile.bzl).

Para obtener más información sobre la herramienta `tfcompile` subyacente, consulte [tfcompile_main.cc](https://www.tensorflow.org/code/tensorflow/compiler/aot/tfcompile_main.cc).

### Paso 3: escriba el código para invocar el subgrafo

En este paso se usa el archivo de encabezado (`test_graph_tfmatmul.h`) generado por la macro de compilación `tf_library` en el paso anterior para invocar el código generado. El archivo de encabezado se encuentra en el directorio `bazel-bin` correspondiente al paquete de compilación y recibe su nombre a partir del atributo de nombre establecido en la macro de compilación `tf_library`. Por ejemplo, el encabezado generado para `test_graph_tfmatmul` sería `test_graph_tfmatmul.h`. A continuación, se muestra una versión abreviada de lo que se genera. El archivo generado, en `bazel-bin`, contiene comentarios útiles adicionales.

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

La clase C++ generada se llama `MatMulComp` en el espacio de nombres `foo::bar`, porque esa era la `cpp_class` especificada en la macro `tf_library`. Todas las clases generadas tienen una API similar, que solo se diferencia por los métodos que usan para manejar los búferes de argumentos y resultados. Esos métodos difieren según el número y los tipos de búferes, que fueron especificados por los argumentos `feed` y `fetch` de la macro `tf_library`.

Hay tres tipos de búferes que se administran dentro de la clase generada: `args` que representan las entradas, `results` que representan las salidas y `temps` que representan búferes temporales que se usan internamente para ejecutar el cálculo. De forma predeterminada, cada instancia de la clase generada asigna y administra todos estos búferes por usted. El argumento del constructor `AllocMode` se puede usar para cambiar este comportamiento. Todos los búferes están alineados con límites de 64 bytes.

La clase C++ generada es solo un contenedor del código de bajo nivel generado por XLA.

Ejemplo de invocación de la función generada a partir de [`tfcompile_test.cc`](https://www.tensorflow.org/code/tensorflow/compiler/aot/tests/tfcompile_test.cc):

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

### Paso 4: cree el binario final

Este paso combina la biblioteca generada por `tf_library` en el paso 2 y el código escrito en el paso 3 para crear un binario final. A continuación, se muestra un ejemplo de archivo BUILD `bazel`.

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
