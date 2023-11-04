# Crear un op

Nota: Para garantizar que sus ops personalizadas en C++ son compatibles en ABI con los paquetes pip oficiales de TensorFlow, siga la guía que encontrará en [Repositorio de ops personalizadas](https://github.com/tensorflow/custom-op). Contiene un ejemplo de código completo, así como imágenes Docker para crear y distribuir sus operaciones personalizadas.

Si quiere crear una op que no esté cubierta por la librería TensorFlow existente, le recomendamos que primero intente escribir la op en Python como una composición de ops o funciones Python existentes. Si no es posible, puede crear una op personalizada en C++. Hay varias razones para querer crear una op personalizada en C++:

- No es fácil ni posible expresar su operación como una composición de ops existentes.
- No es eficaz expresar su operación como un compuesto de primitivas existentes.
- Quiere fusionar a mano un grupo de primitivas que a un futuro compilador le resultaría difícil fusionar.

Por ejemplo, imagine que quiere implementar algo como la "acumulación de medianas", similar al operador "MaxPool", pero calculando medianas sobre intervalos deslizantes en lugar de valores máximos. Puede ser posible hacerlo usando una combinación de operaciones (por ejemplo, usando ExtractImagePatches y TopK), pero quizá no sea tan eficiente en cuanto a rendimiento o memoria como una operación nativa en la que pueda hacer algo más inteligente en una única operación fusionada. Como siempre, vale la pena intentar primero expresar lo que quiere utilizando la combinación de operadores, y sólo elegir añadir una nueva operación si resulta difícil o ineficaz.

Para incorporar su op personalizada necesitará:

1. Registrar la nueva op en un archivo C++. El registro de una op define una interfaz (especificación) para la funcionalidad de la op, que es independiente de la implementación de la op. Por ejemplo, el registro de una op define su nombre y sus entradas y salidas. También define la función de forma que se usa para inferir la forma del tensor.
2. Implementar la op en C++. La implementación de una op se conoce como kernel, y es la implementación concreta de la especificación que registraste en el Paso 1. Puede haber varios kernels para distintos tipos de entrada/salida o arquitecturas (por ejemplo, CPUs, GPUs).
3. Crear un contenedor Python (opcional). Este contenedor es la API pública que se usa para crear la op en Python. Se genera un contenedor predeterminado a partir del registro de la op, que puede usarse directamente o añadirse.
4. Escribir una función para calcular gradientes para la op (opcional).
5. Probar la op. Normalmente lo hacemos en Python por comodidad, pero también puede probar la op en C++. Si define gradientes, puede comprobarlos con la función de Python `tf.test.compute_gradient_error`. Consulte [`relu_op_test.py`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/kernel_tests/nn_ops/relu_op_test.py) como ejemplo que comprueba las funciones de avance de los operadores tipo Relu y sus gradientes.

### Requisitos previos

- Cierta familiaridad con C++.
- Debes tener instalado el [binario de TensorFlow](https://www.tensorflow.org/install), o debes tener [descargado el código fuente de TensorFlow](https://www.tensorflow.org/install/source), y ser capaz de compilarlo.

## Definir la interfaz op

Define la interfaz de una op al registrarla en el sistema TensorFlow. En el registro, especifica el nombre de su op, sus entradas (tipos y nombres) y salidas (tipos y nombres), así como docstrings y cualquier [attrs](#attrs) que la op pueda necesitar.

Para ver cómo funciona, suponga que quiere crear una op que tome un tensor de `int32`s y produzca una copia del tensor, con todos los elementos a cero menos el primero. Para ello, cree un archivo llamado `zero_out.cc`. Añada una llamada a la macro `REGISTER_OP` que define la interfaz de su op:

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

Esta op `ZeroOut` toma como entrada un tensor `to_zero` de enteros de 32 bits, y devuelve un tensor `to_zero` de enteros de 32 bits. La op también usa una función de forma para asegurarse de que el tensor de salida tiene la misma forma que el tensor de entrada. Por ejemplo, si la entrada es un tensor de forma [10, 20], esta función de forma especifica que la forma de salida también es [10, 20].

Nota: El nombre de la op debe estar en mayúsculas y minúsculas y debe ser único entre todas las demás ops registradas en el binario.

## Implementar el kernel para la op

Después de definir la interfaz, proporcione una o varias implementaciones de la op. Para crear uno de estos kernels, cree una clase que extienda `OpKernel` y anule el método `Compute`. El método `Compute` ofrece un argumento `context` de tipo `OpKernelContext*`, desde el que puedee acceder a cosas útiles como los tensores de entrada y salida.

Añada su kernel al archivo que ha creado anteriormente. El kernel podría terminar viéndose así:

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

Después de implementar su kernel, lo registra en el sistema TensorFlow. En el registro, especifica las diferentes restricciones bajo las que se ejecutará este kernel. Por ejemplo, podría tener un kernel hecho para CPUs, y otro distinto para GPUs.

Para hacer esto con la op `ZeroOut`, añada lo siguiente a `zero_out.cc`:

```c++
REGISTER_KERNEL_BUILDER(Name("ZeroOut").Device(DEVICE_CPU), ZeroOutOp);
```

> Importante: Se puede acceder a las instancias de su OpKernel simultáneamente. Su método `Compute` debe ser seguro para los hilos. Proteja cualquier acceso a los miembros de la clase con un mutex. O mejor aún, ¡no comparta el estado a través de los miembros de la clase! Considere usar un [`ResourceMgr`](https://www.tensorflow.org/code/tensorflow/core/framework/resource_mgr.h) para dar seguimiento al estado op.

### Kernels CPU multihilo

Para escribir un kernel de CPU multihilo, se puede usar la función Shard de [`work_sharder.h`](https://www.tensorflow.org/code/tensorflow/core/util/work_sharder.h). Esta función fragmenta una función de cálculo entre los hilos configurados para usarlos en la gestión intraoperativa de hilos (véase intra_op_parallelism_threads en [`config.proto`](https://www.tensorflow.org/code/tensorflow/core/protobuf/config.proto)).

### Kernel de GPU

Un kernel de GPU se implementa en dos partes: el OpKernel y el kernel CUDA y su código de lanzamiento.

A veces, la implementación del OpKernel es común entre el kernel de la CPU y el de la GPU, por ejemplo, para inspeccionar las entradas y asignar las salidas.  En ese caso, se sugiere implementar lo siguiente:

1. Definir la plantilla OpKernel en el Dispositivo y el tipo primitivo del tensor.
2. Para realizar el cálculo real de la salida, la función Compute llama a un struct functor de plantilla.
3. La especialización de ese functor para el CPUDevice se define en el mismo archivo, pero la especialización para el GPUDevice se define en un archivo .cu.cc, ya que se compilará con el compilador CUDA.

Aquí tienes un ejemplo de implementación.

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

## Construir la librería op

### Compila la op usando el compilador de tu sistema (instalación del binario de TensorFlow).

Debería poder compilar `zero_out.cc` con un compilador `C++` como `g++` o `clang` disponible en su sistema. El paquete binario PIP instala los archivos de cabecera y la librería que necesita para compilar su op en ubicaciones que son específicas del sistema. Sin embargo, la librería python de TensorFlow ofrece la función `get_include` para obtener el directorio header, y el directorio `get_lib` tiene un objeto compartido con el que enlazar. Estas son las salidas de estas funciones en una máquina Ubuntu.

```bash
$ python
>>> import tensorflow as tf
>>> tf.sysconfig.get_include()
'/usr/local/lib/python3.6/site-packages/tensorflow/include'
>>> tf.sysconfig.get_lib()
'/usr/local/lib/python3.6/site-packages/tensorflow'
```

Suponiendo que tenga `g++` instalado, ésta es la secuencia de comandos que puede usar para compilar su op en una librería dinámica.

```bash
TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
g++ -std=c++14 -shared zero_out.cc -o zero_out.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2
```

En macOS, se requiere el Indicador adicional "-undefined dynamic_lookup" al construir el archivo `.so`.

> Nota sobre `gcc` versión `>=5`: gcc usa la nueva [ABI](https://gcc.gnu.org/gcc-5/changes.html#libstdcxx) de C++ desde la versión `5`. TensorFlow 2.8 y anteriores se construyeron con `gcc4`, que usa la ABI anterior. Si utiliza estas versiones de TensorFlow e intenta compilar su librería op con `gcc>=5`, añada `-D_GLIBCXX_USE_CXX11_ABI=0` a la línea de comandos para que la librería sea compatible con la ABI anterior. Los paquetes TensorFlow 2.9+ son compatibles con la ABI más reciente de forma predeterminada.

### Compile la op usando bazel (instalación del código fuente de TensorFlow)

Si tiene instaladas las fuentes de TensorFlow, puede usar el sistema de compilación de TensorFlow para compilar su op. Coloque un archivo BUILD con la siguiente regla de compilación Bazel en el directorio [`tensorflow/core/user_ops`](https://www.tensorflow.org/code/tensorflow/core/user_ops/).

```python
load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")

tf_custom_op_library(
    name = "zero_out.so",
    srcs = ["zero_out.cc"],
)
```

Ejecute el siguiente comando para compilar `zero_out.so`.

```bash
$ bazel build --config opt //tensorflow/core/user_ops:zero_out.so
```

Para compilar la operación `Example`, con el kernel CUDA, debe usar el parámetro `gpu_srcs` de `tf_custom_op_library`. Coloque un archivo BUILD con la siguiente regla de compilación Bazel en una nueva carpeta dentro del directorio [`tensorflow/core/user_ops`](https://www.tensorflow.org/code/tensorflow/core/user_ops/) (por ejemplo, "example_gpu").

```python
load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")

tf_custom_op_library(
    # kernel_example.cc  kernel_example.cu.cc  kernel_example.h
    name = "kernel_example.so",
    srcs = ["kernel_example.h", "kernel_example.cc"],
    gpu_srcs = ["kernel_example.cu.cc", "kernel_example.h"],
)
```

Ejecute el siguiente comando para construir `kernel_ejemplo.so`.

```bash
$ bazel build --config opt //tensorflow/core/user_ops/example_gpu:kernel_example.so
```

Nota: Como ya se ha explicado, si compila con gcc&gt;=5 añada `--cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0"` a los argumentos de la línea de comandos de Bazel.

> Nota: Aunque puede crear una librería compartida (un archivo `.so`) con la regla estándar `cc_library`, le sugerimos mucho que use la macro `tf_custom_op_library`. Añade algunas dependencias necesarias y se asegura de que la librería compartida es compatible con el mecanismo de carga de complementos de TensorFlow.

## Usar la op en Python

La API Python de TensorFlow proporciona la función `tf.load_op_library` para cargar la librería dinámica y registrar la op en el marco TensorFlow. `load_op_library` devuelve un módulo de Python que contiene los contenedores de Python para la op y el kernel. Así, una vez que haya construido la op, puede hacer lo siguiente para ejecutarla desde Python:

```python
import tensorflow as tf
zero_out_module = tf.load_op_library('./zero_out.so')
print(zero_out_module.zero_out([[1, 2], [3, 4]]).numpy())

# Prints
array([[1, 0], [0, 0]], dtype=int32)
```

Tenga en cuenta que a la función generada se le dará un nombre tipo snake_case (para cumplir con [PEP8](https://www.python.org/dev/peps/pep-0008/)). Así, si su op se llama `ZeroOut` en los archivos C++, la función python se llamará `zero_out`.

Para que la op esté disponible como una función normal `import`-able desde un módulo de Python, puede ser útil tener la llamada `load_op_library` en un archivo fuente de Python, como se indica a continuación:

```python
import tensorflow as tf

zero_out_module = tf.load_op_library('./zero_out.so')
zero_out = zero_out_module.zero_out
```

## Comprobar que la op funciona

Una buena forma de comprobar que ha implementado correctamente su op es escribir una prueba para ella. Cree el archivo `zero_out_op_test.py` con el contenido:

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

A continuación, ejecute su prueba (suponiendo que tenga instalado tensorflow):

```sh
$ python zero_out_op_test.py
```

## Añadir funciones avanzadas a tu op

Ahora que ya sabe cómo construir una op y una implementación básicas (y algo restringidas), veremos algunas de las cosas más complicadas que normalmente necesitará incorporar a su op. Esto incluye:

- [Comprobaciones condicionales y validación](#conditional-checks-and-validation)
- [Registro de op](#op-registration)
    - [Attrs](#attrs)
    - [Tipos de attr](#attr-types)
    - [Polimorfismo](#polymorphism)
    - [Entradas y salidas](#inputs-and-outputs)
    - [Retrocompatibilidad](#backwards-compatibility)
- [Soporte para GPU](#gpu-support)
    - [Compilación del kernel para el dispositivo GPU](#compiling-the-kernel-for-the-gpu-device)
- [Implementar el gradiente en Python](#implement-the-gradient-in-python)
- [Funciones de forma en C++](#shape-functions-in-c)

### Comprobaciones condicionales y validación

El ejemplo anterior suponía que la op se aplicaba a un tensor de cualquier forma. ¿Y si sólo se aplicara a vectores? Eso significa añadir una comprobación a la implementación del OpKernel anterior.

```c++
  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);

    OP_REQUIRES(context, TensorShapeUtils::IsVector(input_tensor.shape()),
                errors::InvalidArgument("ZeroOut expects a 1-D vector."));
    // ...
  }
```

Esto afirma que la entrada es un vector, y devuelve habiendo configurado el estatus `InvalidArgument` si no lo es. La macro [`OP_REQUIRES`](https://www.tensorflow.org/code/tensorflow/core/platform/errors.h) toma tres argumentos:

- El `context`, que puede ser un puntero `OpKernelContext` o `OpKernelConstruction` (véase [`tensorflow/core/framework/op_kernel.h`](https://www.tensorflow.org/code/tensorflow/core/framework/op_kernel.h)), para su método `SetStatus()`.
- La condición. Por ejemplo, hay funciones para validar la forma de un tensor en [`tensorflow/core/framework/tensor_shape.h`](https://www.tensorflow.org/code/tensorflow/core/framework/tensor_shape.h)
- El error en sí, que se representa mediante un objeto `Status`, consulta [`tensorflow/core/platform/status.h`](https://www.tensorflow.org/code/tensorflow/core/platform/status.h). Un `Status` tiene tanto un tipo (frecuentemente `InvalidArgument`, pero mira la lista de tipos) como un mensaje. Puede encontrar funciones para construir un error en [`tensorflow/core/platform/errors.h`](https://www.tensorflow.org/code/tensorflow/core/platform/errors.h).

Alternativamente, si quiere comprobar si un objeto `Status` devuelto por alguna función es un error, y en tal caso devolverlo, use [`OP_REQUIRES_OK`](https://www.tensorflow.org/code/tensorflow/core/platform/errors.h).  Ambas macros devuelven de la función en caso de error.

### Registro de op

#### Attrs

Los ops pueden tener attrs, cuyos valores se configuran cuando el op se añade a un grafo. Se usan para configurar la op, y se puede acceder a sus valores tanto dentro de la implementación del kernel como en los tipos de entradas y salidas del registro de la op. Es preferible usar un input en lugar de un attrs siempre que sea posible, ya que los inputs son más flexibles. Esto se debe a que las attrs son constantes y deben definirse en el momento de construir el grafo. En cambio, los inputs son Tensores cuyos valores pueden ser dinámicos; es decir, los inputs pueden cambiar a cada paso, configurarse mediante un feed, etc. Los attrs se usan para cosas que no se pueden hacer con las entradas: cualquier configuración que afecte a la firma (número o tipo de entradas o salidas) o que no pueda cambiar de un paso a otro.

Se define un attrs cuando se registra la op, especificando su nombre y tipo mediante el método `Attr`, que espera una especificación de la forma:

```
<name>: <attr-type-expr>
```

donde `<name>` empieza por una letra y puede estar compuesto por caracteres alfanuméricos y guiones bajos, y `<attr-type-expr>` es una expresión de tipo de la forma [descrita a continuación](#attr-types).

Por ejemplo, si quiere que la op `ZeroOut` conserve un índice especificado por el usuario, en lugar de sólo el elemento 0, puede registrar la op así:

```c++
REGISTER_OP("ZeroOut")
    .Attr("preserve_index: int")
    .Input("to_zero: int32")
    .Output("zeroed: int32");
```

(Tenga en cuenta que el conjunto de [tipos de atributos](#attr-types) es distinto del `tf.DType` que se usa para las entradas y salidas).

Su kernel puede acceder a este attrs en su constructor mediante el parámetro `context`:

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

que puede usarse en el método `Compute`:

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

En un attr se admiten los siguientes tipos:

- `string`: Cualquier secuencia de bytes (no es necesario que sea UTF8).
- `int`: Un entero con signo.
- `float`: Un número de punto flotante.
- `bool`: Verdadero o falso.
- `type`: Uno de los valores (no ref) de [`DataType`](https://www.tensorflow.org/code/tensorflow/core/framework/types.cc).
- `shape`: Un [`TensorShapeProto`](https://www.tensorflow.org/code/tensorflow/core/framework/tensor_shape.proto).
- `list(<type>)`: Una lista de `<type>`, donde `<type>` es uno de los tipos mencionados. Tenga en cuenta que `list(list(<type>))` no es válida.

Consulte también [`op_def_builder.cc:FinalizeAttr`](https://www.tensorflow.org/code/tensorflow/core/framework/op_def_builder.cc) para obtener una lista definitiva.

##### Valores predeterminados y restricciones

Los attrs pueden tener valores predeterminados, y algunos tipos de attrs pueden tener restricciones. Para definir un attr con restricciones, puede usar los siguientes `<attr-type-expr>`:

`{'<string1>', '<string2>'}`: El valor debe ser una cadena que tenga el valor `<string1>` o `<string2>`. El nombre del tipo, `string`, está implícito cuando usa esta sintaxis. Esto emula un enum:

```c++
REGISTER_OP("EnumExample")
    .Attr("e: {'apple', 'orange'}");
```

`{<type1>, <type2>}`: El valor es del tipo `type`, y debe ser o `<type1>` o `<type2>`, donde `<type1>` y `<type2>` son `tf.DType` soportados. No se especifica que el tipo del attrs sea `type`. Esto está implícito cuando tiene una lista de tipos en `{...}`. Por ejemplo, en este caso el attr `t` es un tipo que debe ser un `int32`, un `float`, o un `bool`:

```c++
REGISTER_OP("RestrictedTypeExample")
    .Attr("t: {int32, float, bool}");
```

Existen atajos para las restricciones de tipo habituales:

- `numbertype`: Tipo `type` restringido a los tipos numéricos (no string ni bool).
- `realnumbertype`: Como `numbertype` sin tipos complejos.
- `quantizedtype`: Como `numbertype`, pero sólo los tipos de números cuantizados.

Las listas específicas de tipos permitidos por éstas están definidas por las funciones (como `NumberTypes()`) en [`tensorflow/core/framework/types.h`](https://www.tensorflow.org/code/tensorflow/core/framework/types.h). En este ejemplo, el attr `t` debe ser uno de los tipos numéricos:

```c++
REGISTER_OP("NumberType")
    .Attr("t: numbertype");
```

Para esta op:

```python
tf.number_type(t=tf.int32)  # Valid
tf.number_type(t=tf.bool)   # Invalid
```

Las listas pueden combinarse con otras listas y tipos simples. La siguiente op permite que attr `t` sea cualquiera de los tipos numéricos o el tipo bool:

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

`int >= <n>`: El valor debe ser un int cuyo valor sea mayor o igual que `<n>`, donde `<n>` es un número natural. Por ejemplo, el siguiente registro op especifica que el attr `a` debe tener un valor que sea al menos `2`:

```c++
REGISTER_OP("MinIntExample")
    .Attr("a: int >= 2");
```

`list(<type>) >= <n>`: Una lista de tipo `<type>` cuya longitud es mayor o igual que `<n>`. Por ejemplo, el siguiente registro op especifica que el attr `a` es una lista de tipos (ya sea `int32` o `float`), y que debe haber al menos 3 de ellos:

```c++
REGISTER_OP("TypeListExample")
    .Attr("a: list({int32, float}) >= 3");
```

Para configurar un valor predeterminado para un attrs (haciéndolo opcional en el código generado), añada `= <default>` al final, como en:

```c++
REGISTER_OP("AttrDefaultExample")
    .Attr("i: int = 0");
```

Además, se puede especificar tanto una restricción como un valor predeterminado:

```c++
REGISTER_OP("AttrConstraintAndDefaultExample")
    .Attr("i: int >= 1 = 1");
```

La sintaxis admitida del valor predeterminado es la que se usaría en la protorepresentación de la definición GraphDef resultante.

Aquí hay ejemplos de cómo especificar un valor por default para todos los tipos:

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

Observe, en particular, que los valores de tipo `type` usan `tf.DType`.

#### Polimorfismo

##### Polimorfismo de tipo

Para las ops que pueden tomar distintos tipos como entrada o producir distintos tipos de salida, puedes especificar [un attr](#attrs) en [un tipo de entrada o salida](#inputs-and-outputs) en el registro de la op. Por lo general, entonces registraría un `OpKernel` para cada tipo admitido.

Por ejemplo, si quiere que la op `ZeroOut` funcione con `float`s, además de con `int32`s, su registro de op podría tener el siguiente aspecto:

```c++
REGISTER_OP("ZeroOut")
    .Attr("T: {float, int32}")
    .Input("to_zero: T")
    .Output("zeroed: T");
```

Su registro de op especifica ahora que el tipo de la entrada debe ser `float`, o `int32`, y que su salida será del mismo tipo, ya que ambas tienen el tipo `T`.

###### Nomenclatura

Por lo general, las entradas, salidas y attrs deben tener nombres con snake_case. La única excepción son los attrs que se usan como tipo de una entrada o en el tipo de una salida. Estos attrs pueden deducirse cuando la op se añade al grafo y, por tanto, no aparecen en la función de la op. Por ejemplo, esta última definición de ZeroOut generará una función Python parecida a:

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

Si a `to_zero` se le pasa un tensor `int32`, entonces `T` se configura automáticamente en `int32` (bueno, en realidad `DT_INT32`). Esos attrs inferidos reciben nombres en mayúsculas o en camelCase.

Compárelo con una op que tenga un tipo attr que determine el tipo de salida:

```c++
REGISTER_OP("StringToNumber")
    .Input("string_tensor: string")
    .Output("output: out_type")
    .Attr("out_type: {float, int32} = DT_FLOAT");
    .Doc(R"doc(
Converts each string in the input Tensor to the specified numeric type.
)doc");
```

En este caso, el usuario tiene que especificar el tipo de salida, como en el Python generado:

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

###### Ejemplo de polimorfismo de tipo

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

Para mantener la [retrocompatibilidad](#backwards-compatibility), debe especificar un [valor por defaul](#default-values-and-constraints) cuando añada un attr a un op existente:

```c++
REGISTER_OP("ZeroOut")
  .Attr("T: {float, int32} = DT_INT32")
  .Input("to_zero: T")
  .Output("zeroed: T")
```

Supongamos que quiere añadir más tipos, por ejemplo `double`:

```c++
REGISTER_OP("ZeroOut")
    .Attr("T: {float, double, int32}")
    .Input("to_zero: T")
    .Output("zeroed: T");
```

En lugar de escribir otro `OpKernel` con código redundante como el anterior, a menudo podrá usar una plantilla C++ en su lugar. Seguirá teniendo un registro del kernel (llamada `REGISTER_KERNEL_BUILDER`) por sobrecarga.

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

Si tiene más de un par de sobrecargas, puede poner el registro en una macro.

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

Según la lista de tipos para los que registre el kernel, puede usar una macro proporcionada por [`tensorflow/core/framework/register_types.h`](https://www.tensorflow.org/code/tensorflow/core/framework/register_types.h):

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

##### Entradas y salidas de lista

Además de poder aceptar o producir distintos tipos, las ops pueden consumir o producir un número variable de tensores.

En el siguiente ejemplo, el attr `T` contiene una *lista* de tipos, y se usa como tipo tanto de la entrada `in` como de la salida `out`. La entrada y la salida son listas de tensores de ese tipo (y el número y los tipos de tensores de la salida son los mismos que los de la entrada, ya que ambos tienen el tipo `T`).

```c++
REGISTER_OP("PolymorphicListExample")
    .Attr("T: list(type)")
    .Input("in: T")
    .Output("out: T");
```

También puedes poner restricciones a los tipos que se pueden especificar en la lista. En el siguiente caso, la entrada es una lista de tensores `float` y `double`. La op acepta, por ejemplo, tipos de entrada `(float, double, float)` y en ese caso el tipo de salida también sería `(float, double, float)`.

```c++
REGISTER_OP("ListTypeRestrictionExample")
    .Attr("T: list({float, double})")
    .Input("in: T")
    .Output("out: T");
```

Si quiere que todos los tensores de una lista sean del mismo tipo, puede hacer algo como:

```c++
REGISTER_OP("IntListInputExample")
    .Attr("N: int")
    .Input("in: N * int32")
    .Output("out: int32");
```

Esto acepta una lista de `int32` tensores y usa un `int` attr `N` para especificar la longitud de la lista.

Esto puede hacerse [tipo polimórfico](#type-polymorphism) también. En el siguiente ejemplo, la entrada es una lista de tensores (con longitud `"N"`) del mismo (pero no especificado) tipo (`"T"`), y la salida es un único tensor de tipo coincidente:

```c++
REGISTER_OP("SameListInputExample")
    .Attr("N: int")
    .Attr("T: type")
    .Input("in: N * T")
    .Output("out: T");
```

Por default, las listas de tensores tienen una longitud mínima de 1. Puede cambiar ese valor predeterminado usando [una restricción `">="` en el attr](#default-values-and-constraints) correspondiente. En el siguiente ejemplo, la entrada es una lista de al menos 2 tensores `int32`:

```c++
REGISTER_OP("MinLengthIntListExample")
    .Attr("N: int >= 2")
    .Input("in: N * int32")
    .Output("out: int32");
```

La misma sintaxis funciona con `"list(type)"` attrs:

```c++
REGISTER_OP("MinimumLengthPolymorphicListExample")
    .Attr("T: list(type) >= 3")
    .Input("in: T")
    .Output("out: T");
```

#### Entradas y salidas

Para resumir lo anterior, un registro op puede tener múltiples entradas y salidas:

```c++
REGISTER_OP("MultipleInsAndOuts")
    .Input("y: int32")
    .Input("z: float")
    .Output("a: string")
    .Output("b: int32");
```

Cada especificación de entrada o salida tiene la forma:

```
<name>: <io-type-expr>
```

donde `<name>` comienza con una letra y puede estar formado por caracteres alfanuméricos y guiones bajos. `<io-type-expr>` es una de las siguientes expresiones de tipo:

- `<type>`, donde `<type>` es un tipo de entrada admitido (por ejemplo, `float`, `int32`, `string`). Esto especifica un único tensor del tipo dado.

    Vea `tf.DType`.

    ```c++
    REGISTER_OP("BuiltInTypesExample")
        .Input("integers: int32")
        .Input("complex_numbers: complex64");
    ```

- `<attr-type>`, donde `<attr-type>` es el nombre de una [Attr](#attrs) con tipo `type` o `list(type)` (con una posible restricción de tipo). Esta sintaxis permite [ops polimórficas](#polymorphism).

    ```c++
    REGISTER_OP("PolymorphicSingleInput")
        .Attr("T: type")
        .Input("in: T");

    REGISTER_OP("RestrictedPolymorphicSingleInput")
        .Attr("T: {int32, int64}")
        .Input("in: T");
    ```

    Hacer referencia a un attr de tipo `list(type)` permite aceptar una secuencia de tensores.

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

    Observe que el número y los tipos de tensores en la salida `out` es el mismo que en la entrada `in`, ya que ambos son del tipo `T`.

- Para una secuencia de tensores con el mismo tipo: `<number> * <type>`, donde `<number>` es el nombre de un [Attr](#attrs) con tipo `int`. El `<type>` puede ser un `tf.DType`, o el nombre de un attr con tipo `type`. Como ejemplo de lo primero, esta op acepta una lista de tensores `int32`:

    ```c++
    REGISTER_OP("Int32SequenceExample")
        .Attr("NumTensors: int")
        .Input("in: NumTensors * int32")
    ```

    Mientras que esta op acepta una lista de tensores de cualquier tipo, siempre que sean todos iguales:

    ```c++
    REGISTER_OP("SameTypeSequenceExample")
        .Attr("NumTensors: int")
        .Attr("T: type")
        .Input("in: NumTensors * T")
    ```

- Para una referencia a un tensor: `Ref(<type>)`, donde `<type>` es uno de los tipos anteriores.

Cualquier attr usado en el tipo de una entrada será inferido. Por convención, esos attr inferidos usan nombres en mayúsculas (como `T` o `N`). De lo contrario, las entradas, salidas y attr tienen nombres como parámetros de función (por ejemplo, `num_outputs`). Si desea más detalles, consulte la [sección previa sobre nomenclatura](#naming).

Si desea saber más, vea [`tensorflow/core/framework/op_def_builder.h`](https://www.tensorflow.org/code/tensorflow/core/framework/op_def_builder.h).

#### Retrocompatibilidad

Supongamos que ha escrito una buena op personalizada y la ha compartido con otros, por lo que tiene clientes contentos usando su operación. Sin embargo, le gustaría hacer cambios en la op de alguna manera.

En general, los cambios en las especificaciones existentes y verificadas deben ser retrocompatibles: cambiar la especificación de una op no debe romper los buffers de protocolo de `GraphDef` serializados anteriores construidos a partir de especificaciones más antiguas. Los detalles de la compatibilidad de `GraphDef` se [describen aquí](./versions.md#compatibility_of_graphs_and_checkpoints).

Existen varias formas de preservar la retrocompatibilidad.

1. Todos los nuevos attrs que se añadan a una operación deben tener definidos valores por defecto, y con ese valor por defecto la op debe tener el comportamiento original. Para cambiar una operación de no polimórfica a polimórfica, *debe* dar un valor por defecto al nuevo tipo attr para preservar la firma original por defecto. Por ejemplo, si su operación era:

    ```c++
    REGISTER_OP("MyGeneralUnaryOp")
        .Input("in: float")
        .Output("out: float");
    ```

    puede hacerla polimórfica de forma retrocompatible usando:

    ```c++
    REGISTER_OP("MyGeneralUnaryOp")
        .Input("in: T")
        .Output("out: T")
        .Attr("T: numerictype = DT_FLOAT");
    ```

2. Puede hacer que una restricción en un attr sea menos restrictiva de forma segura. Por ejemplo, puede cambiar de `{int32, int64}` a `{int32, int64, float}` o `type`. O puede cambiar de `{"apple", "orange"}` a `{"apple", "banana", "orange"}` o `string`.

3. Puede cambiar las entradas/salidas simples por entradas/salidas de lista, siempre que el valor por defecto para el tipo de lista coincida con la firma antigua.

4. Puede añadir una nueva entrada / salida de la lista, si el valor predeterminado es vacío.

5. Asigne un namespace a cualquier nueva op que cree, anteponiendo a los nombres de las op algo exclusivo de su proyecto. Esto evita que su op colisione con cualquier op que pueda incluirse en futuras versiones de TensorFlow.

6. ¡Prevea el futuro! Intente anticiparse a futuros usos de la op. Algunos cambios de firma no pueden hacerse de forma compatible (por ejemplo, convertir una lista del mismo tipo en una lista de tipos distintos).

Puede encontrar la lista completa de cambios seguros y no seguros en [`tensorflow/core/framework/op_compatibility_test.cc`](https://www.tensorflow.org/code/tensorflow/core/framework/op_compatibility_test.cc). Si no puede hacer que su cambio en una operación sea retrocompatible, cree una nueva operación con un nuevo nombre con la nueva semántica.

Tenga en cuenta también que, aunque estos cambios pueden mantener la compatibilidad con `GraphDef`, el código Python generado puede cambiar de forma que no sea compatible con los antiguos invocadores. La API de Python puede conservarse compatible mediante cambios cuidadosos en un contenedor de Python escrito a mano, conservando la firma antigua excepto, quizá, añadiendo nuevos argumentos opcionales al final. En general, los cambios incompatibles sólo pueden hacerse cuando TensorFlow cambia de versión principal, y deben ajustarse a la semántica de versión de <a href="./versions.md#compatibility_of_graphs_and_checkpoints" data-md-type="link">`GraphDef`</a>.

### Soporte para GPU

Puede implementar diferentes OpKernels y registrar uno para CPU y otro para GPU, al igual que puede [registrar kernels para diferentes tipos](#polymorphism). Hay varios ejemplos de kernels con soporte para GPU en [`tensorflow/core/kernels/`](https://www.tensorflow.org/code/tensorflow/core/kernels/). Observe que algunos kernels tienen una versión para CPU en un archivo `.cc`, una versión para GPU en un archivo que termina en `_gpu.cu.cc`, y un poco de código compartido en común en un archivo `.h`.

Por ejemplo, el `tf.pad` tiene todo menos el kernel de la GPU en [`tensorflow/core/kernels/pad_op.cc`](https://www.tensorflow.org/code/tensorflow/core/kernels/pad_op.cc). El kernel de la GPU está en [`tensorflow/core/kernels/pad_op_gpu.cu.cc`](https://www.tensorflow.org/code/tensorflow/core/kernels/pad_op_gpu.cu.cc), y el código compartido es una clase de plantilla definida en [`tensorflow/core/kernels/pad_op.h`](https://www.tensorflow.org/code/tensorflow/core/kernels/pad_op.h). Organizamos el código de esta forma por dos motivos: permite compartir código común entre las implementaciones de la CPU y la GPU, y coloca la implementación de la GPU en un archivo independiente para que sólo pueda ser compilada por el compilador de la GPU.

Una cosa a tener en cuenta, incluso cuando se usa la versión del kernel de la GPU de `pad`, sigue necesitando su entrada de `"paddings"` en la memoria de la CPU. Para marcar que las entradas o salidas se conservan en la CPU, añada una llamada `HostMemory()` al registro del kernel, por ejemplo:

```c++
#define REGISTER_GPU_KERNEL(T)                         \
  REGISTER_KERNEL_BUILDER(Name("Pad")                  \
                              .Device(DEVICE_GPU)      \
                              .TypeConstraint<T>("T")  \
                              .HostMemory("paddings"), \
                          PadOp<GPUDevice, T>)
```

#### Compilación del kernel para el dispositivo GPU

Mire en [cuda_op_kernel.cu.cc](https://www.tensorflow.org/code/tensorflow/examples/adding_an_op/cuda_op_kernel.cu.cc) un ejemplo que usa un kernel CUDA para implementar una op. La `tf_custom_op_library` acepta un argumento `gpu_srcs` en el que se puede especificar la lista de archivos fuente que contienen los kernels CUDA (archivos `*.cu.cc`). Para usarlos con una instalación binaria de TensorFlow, los kernels CUDA deben compilarse con el compilador `nvcc` de NVIDIA. Esta es la secuencia de comandos que puede usar para compilar [cuda_op_kernel.cu.cc](https://www.tensorflow.org/code/tensorflow/examples/adding_an_op/cuda_op_kernel.cu.cc) y [cuda_op_kernel.cc](https://www.tensorflow.org/code/tensorflow/examples/adding_an_op/cuda_op_kernel.cc) en una única librería cargable dinámicamente:

```bash
nvcc -std=c++14 -c -o cuda_op_kernel.cu.o cuda_op_kernel.cu.cc \
  ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

g++ -std=c++14 -shared -o cuda_op_kernel.so cuda_op_kernel.cc \
  cuda_op_kernel.cu.o ${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]}
```

El `cuda_op_kernel.so` producido anteriormente puede cargarse como de costumbre en Python, usando la función `tf.load_op_library`.

Tenga en cuenta que si sus librerías CUDA no están instaladas en `/usr/local/lib64`, tendrá que especificar la ruta explícitamente en el segundo comando (g++) anterior. Por ejemplo, añada `-L /usr/local/cuda-8.0/lib64/` si su CUDA está instalada en `/usr/local/cuda-8.0`.

Nota: En algunas configuraciones de Linux, se necesitan opciones adicionales al paso de compilación `nvcc`. Añada `-D_MWAITXINTRIN_H_INCLUDED` a la línea de órdenes `nvcc` para evitar errores de `mwaitxintrin.h`.

### Implementar el gradiente en Python

Dado un grafo de ops, TensorFlow usa la diferenciación automática (retropropagación) para añadir nuevas ops que representen gradientes con respecto a las ops existentes. Para que la diferenciación automática funcione para los nuevos ops, debe registrar una función de gradiente que calcule gradientes con respecto a las entradas de los ops dados gradientes con respecto a las salidas de los ops.

Matemáticamente, si una op calcula \(y = f(x)\) la op de gradiente registrada convierte los gradientes \(\parcial L/ \parcial y\) de pérdida \(L\) con respecto a \(y\) en gradientes \(\parcial L/ \parcial x\) con respecto a \(x\) mediante la regla de la cadena:

$$\frac{\parcial L}{\parcial x} = \frac{\parcial L}{\parcial y} \frac{\parcial y}{\parcial x} = \frac{\parcial L}{\parcial y} \frac{\parcial f}{\parcial x}.$$

En el caso de `ZeroOut`, sólo una entrada en la entrada afecta a la salida, por lo que el gradiente con respecto a la entrada es un tensor "caliente" disperso.  Esto se manifiesta de la siguiente manera:

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

Detalles sobre el registro de funciones de gradiente con `tf.RegisterGradient`:

- Para una op con una salida, la función de gradiente tomará una `tf.Operación`, `op`, y un `grad` de `tf.Tensor` y construirá nuevas ops a partir de las `op.inputs[i]`, `op.outputs[i]`, y `grad` de tensores. Se puede encontrar información sobre cualquier attr a través de `tf.Operation.get_attr`.

- Si la op tiene múltiples salidas, la función gradiente tomará `op` y `grads`, donde `grads` es una lista de gradientes con respecto a cada salida. El resultado de la función gradiente debe ser una lista de objetos `Tensor` que representen los gradientes con respecto a cada entrada.

- Si no existe un gradiente bien definido para alguna entrada, como en el caso de entradas enteras usadas como índices, el gradiente devuelto correspondiente debería ser `None`. Por ejemplo, para una op que toma un tensor de punto flotante `x` y un índice entero `i`, la función de gradiente devolvería `return [x_grad, None]`.

- Si no hay ningún gradiente significativo para el op, a menudo no tendrá que registrar ningún gradiente, y mientras el gradiente del op no se necesite nunca, todo irá bien. En algunos casos, un op no tiene un gradiente bien definido pero puede participar en el cálculo del gradiente. Aquí puede utilizar `ops.NotDifferentiable` para propagar automáticamente ceros hacia atrás.

Tenga en cuenta que en el momento en que se llama a la función de gradiente, sólo está disponible el grafo de flujo de datos de ops, no los datos del tensor en sí. Entonces, todo el cálculo debe realizarse usando otras ops de tensorflow, que se ejecutarán en el momento de ejecución del grafo.

Añada sugerencias de tipo al registrar el gradiente personalizado para un tipo de op para que el código sea más legible, depurable, fácil de mantener y más robusto gracias a la validación de datos. Por ejemplo, al tomar una `op` como parámetro en una función, especifique que la función de gradiente tomará una <a href="https://www.tensorflow.org/api_docs/python/tf/Operation"><code>tf.Operation</code></a> como tipo de parámetro.

### Funciones de forma en C++

La API de TensorFlow tiene una función llamada "inferencia de forma" que aporta información sobre las formas de los tensores sin tener que ejecutar el grafo. La inferencia de forma está soportada por "funciones de forma" que se registran para cada tipo de op en la declaración `REGISTER_OP` de C++, y tienen dos roles: asegurar que las formas de las entradas son compatibles durante la construcción del grafo, y especificar las formas para las salidas.

Las funciones de forma se definen como operaciones sobre la clase `shape_inference::InferenceContext`. Por ejemplo, en la función de forma para ZeroOut:

```c++
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });
```

`c->set_output(0, c->input(0));` declara que la forma de la primera salida debe configurarse con la forma de la primera entrada. Si la salida se selecciona por su índice como en el ejemplo anterior, el segundo parámetro de `set_output` debería ser un objeto `ShapeHandle`. Puede crear un objeto `ShapeHandle` vacío mediante su constructor predeterminado. El objeto `ShapeHandle` para una entrada con índice `idx` puede obtenerse mediante `c->input(idx)`.

Hay una serie de funciones de forma comunes que se aplican a muchas ops, como `shape_inference::UnchangedShape`, que puede encontrarse en [common_shape_fns.h](https://www.tensorflow.org/code/tensorflow/core/framework/common_shape_fns.h) y usarse como sigue:

```c++
REGISTER_OP("ZeroOut")
    .Input("to_zero: int32")
    .Output("zeroed: int32")
    .SetShapeFn(::tensorflow::shape_inference::UnchangedShape);
```

Una función de forma también puede restringir la forma de una entrada. Para la versión de [`ZeroOut` con una restricción de forma vectorial](#conditional-checks-and-validation), la función de forma sería la siguiente:

```c++
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      ::tensorflow::shape_inference::ShapeHandle input;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &input));
      c->set_output(0, input);
      return Status::OK();
    });
```

La llamada `WithRank` verifica que la forma de entrada `c->input(0)` tiene una forma con exactamente una dimensión (o si la forma de entrada es desconocida, la forma de salida será un vector con una dimensión desconocida).

Si su op es [polimórfica con múltiples entradas](#polymorphism), puede usar miembros de `InferenceContext` para determinar el número de formas a comprobar, y `Merge` para validar que las formas son todas compatibles (alternativamente, acceda a los atributos que indican las longitudes, con `InferenceContext::GetAttr`, que da acceso a los atributos de la op).

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

Dado que la inferencia de forma es una característica opcional, y que las formas de los tensores pueden variar dinámicamente, las funciones de forma deben ser robustas a la información de forma incompleta para cualquiera de las entradas. El método `Merge` de [`InferenceContext`](https://www.tensorflow.org/code/tensorflow/core/framework/shape_inference.h) permite afirmar que dos formas son iguales, aunque una de ellas o ambas no dispongan de información completa. Las funciones de forma se definen para todas las ops centrales de TensorFlow y dan muchos ejemplos de uso diferentes.

La clase `InferenceContext` tiene una serie de funciones que pueden usarse para definir manipulaciones de funciones de forma.  Por ejemplo, puede validar que una dimensión concreta tenga un valor muy específico usando `InferenceContext::Dim` y `InferenceContext::WithValue`; puede especificar que una dimensión de salida sea la suma / producto de dos dimensiones de entrada usando `InferenceContext::Add` y `InferenceContext::Multiply`. Consulte la clase `InferenceContext` para conocer todas las manipulaciones de forma que puede especificar. El siguiente ejemplo configura la forma de la primera salida a (n, 3), donde la primera entrada tiene forma (n, ...)

```c++
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    c->set_output(0, c->Matrix(c->Dim(c->input(0), 0), 3));
    return Status::OK();
});
```

Si tiene una función de forma complicada, debería considerar añadir una prueba para validar que varias combinaciones de forma de entrada produzcan las combinaciones de forma de salida esperadas. Puede ver ejemplos de cómo escribir estas pruebas en algunas de nuestras pruebas [core ops](https://www.tensorflow.org/code/tensorflow/core/ops/array_ops_test.cc) (la sintaxis de `INFER_OK` y `INFER_ERROR` es un poco enigmática, pero intente ser compacto a la hora de representar las especificaciones de formas de entrada y salida en las pruebas. Por ahora, vea los comentarios alrededor en esas pruebas para hacerse una idea de la especificación de la cadena de forma).

## Construya un paquete pip para su op personalizada

Para construir un paquete `pip` para su op, vea el ejemplo [tensorflow/custom-op](https://github.com/tensorflow/custom-op). Esta guía muestra cómo construir ops personalizadas a partir del paquete pip de TensorFlow en lugar de construir TensorFlow desde el código fuente.
