# Llamadas personalizadas en XLA

Este documento describe cómo escribir y usar las "llamadas personalizadas" en XLA. Las llamadas personalizadas le permiten invocar código escrito en un lenguaje de programación como C++ o CUDA desde un programa XLA.

Advertencia: Las llamadas personalizadas son una función de usuario avanzado de bajo nivel. Cuando se usan llamadas personalizadas es fácil romper el programa en formas difíciles de depurar (e incluso difíciles de notar). No debería usar llamadas personalizadas a menos que esté preparado para depurar XLA usted mismo cuando algo salga mal y, si tiene problemas, es probable que reciba relativamente menos asistencia de los desarrolladores de XLA.

Advertencia: La API/ABI de llamada personalizada aún no es estable. No pretendemos cambiarla en vano, pero es posible que cambie. A continuación, se describen algunos posibles cambios futuros.

## Llamada personalizada en la CPU

Puede crear una instrucción de HLO que represente una llamada personalizada a través de la API cliente de XLA. Esto no está disponible a través de TensorFlow al momento de escribir este artículo.

Por ejemplo, el siguiente código usa una llamada personalizada para calcular `A[i] = B[i % 128]+ C[i]` en la CPU. (Por supuesto que podría, ¡y debería!, hacer esto con una HLO normal).

```c++
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/service/custom_call_target_registry.h"

void do_it() {
  xla::XlaBuilder b("do_it");
  xla::XlaOp param0 =
      xla::Parameter(&b, 0, xla::ShapeUtil::MakeShape(xla::F32, {128}), "p0");
  xla::XlaOp param1 =
      xla::Parameter(&b, 1, xla::ShapeUtil::MakeShape(xla::F32, {2048}), "p1");
  xla::XlaOp custom_call =
      xla::CustomCall(&b, "do_custom_call", /*operands=*/{param0, param1},
                      /*shape=*/xla::ShapeUtil::MakeShape(xla::F32, {2048}));
}

void do_custom_call(void* out, const void** in) {
  float* out_buf = reinterpret_cast<float*>(out);
  const float* in0 = reinterpret_cast<const float*>(in[0]);
  const float* in1 = reinterpret_cast<const float*>(in[1]);
  for (int i = 0; i < 2048; ++i) {
    out_buf[i] = in0[i % 128] + in1[i];
  }
}
XLA_REGISTER_CUSTOM_CALL_TARGET(do_custom_call, "Host");
```

Observe que la función `do_custom_call` necesita conocer las dimensiones de los búferes sobre los que opera. En este ejemplo codificamos los tamaños 128 y 2048. Si no desea hacer esto, puede pasar las dimensiones como parámetros de la llamada.

## Llamada personalizada en la GPU

El marco de llamadas personalizadas de la GPU es ligeramente distinto al de la CPU. Aquí hay un ejemplo de CUDA que ejecuta el mismo cálculo `A[i] = B[i % 128] + C[i]` que el código de CPU anterior.

```c++
void do_it() { /* same implementation as above */ }

__global__ custom_call_kernel(const float* in0, const float* in1, float* out) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  out[idx] = in0[idx % 128] + in1[idx];
}

void do_custom_call(CUstream stream, void** buffers,
                    const char* opaque, size_t opaque_len) {
  const float* in0 = reinterpret_cast<const float*>(buffers[0]);
  const float* in1 = reinterpret_cast<const float*>(buffers[1]);
  float* out = reinterpret_cast<float*>(buffers[2]);

  const int64_t block_dim = 64;
  const int64_t grid_dim = 2048 / block_dim;
  custom_call_kernel<<<grid_dim, block_dim,
                       /*dynamic_shared_mem_bytes=*/0, stream>>>(in0, in1, out);
}
XLA_REGISTER_CUSTOM_CALL_TARGET(do_custom_call, "CUDA");
```

Observe primero que la función de llamada personalizada de la GPU *sigue siendo una función ejecutada en la CPU*. Nuestra función de CPU `do_custom_call` es responsable de poner en cola el trabajo en la GPU. Aquí lanza un kernel CUDA, pero también podría hacer otra cosa, como llamar a cublas.

`buffers` es un arreglo de punteros que reside en el host, y cada elemento que contiene apunta a la memoria del dispositivo (es decir, GPU). Primero vienen los parámetros, seguidos del valor de salida. Esto es notablemente diferente de la convención de llamadas de CPU, que tiene dos parámetros, `ins` y `out`. La razón principal por la que se diverge es para propiciar un manejo más eficiente de las entradas/salidas en forma de tupla; consulte la sección siguiente.

Como en el ejemplo de la CPU, codificamos los tamaños de los búferes de entrada y salida en nuestra llamada personalizada. Sin embargo, a diferencia del caso de la CPU, pasar los tamaños del búfer como operandos a la llamada personalizada no funcionaría bien. Normalmente necesitamos los tamaños de búfer disponibles en la CPU; por ejemplo, cuando lanzamos un kernel, necesitamos saber las dimensiones del bloque/cuadrícula a usar. Pero si pasáramos los tamaños de búfer como operandos a nuestra llamada personalizada, sus valores residirían en la memoria de la GPU. En ese caso, tendríamos que ejecutar una costosa operación de memoria síncrona de dispositivo a host al inicio de nuestra operación con el único fin de leer los tamaños.

Para permitirle solucionar este problema, le ofrecemos el parámetro `opaque`. Puede configurarlo en una cadena arbitraria de bytes al crear la llamada personalizada:

```c++
std::string opaque = "...";
xla::CustomCall(&b, "do_custom_call", /*operands=*/{param0, param1},
                /*output_shape=*/xla::ShapeUtil::MakeShape(xla::F32, {2048}),
                opaque);
```

Como `xla::Shape` tiene una representación de búfer de protocolo, puede almacenar este prototipo serializado dentro de `opaque` y deserializarlo dentro de su llamada personalizada de GPU. Sin embargo, tenga en cuenta que, aunque `xla::ShapeProto` no cambia con frecuencia, *sí* cambia. Consulte el registro de git para ver cómo ha cambiado hasta ahora.

## Cómo señalar un error

Si su llamada personalizada encuentra un error, puede señalar el error al tiempo de ejecución de XLA (en lugar de, por ejemplo, fallar o devolver un error sin sentido en los búferes de salida) usando la siguiente firma para su función en la CPU:

```c++
#include "tensorflow/compiler/xla/service/custom_call_status.h"

void do_custom_call(void* out, const void** in, XlaCustomCallStatus* status);
```

... y en GPU:

```c++
#include "tensorflow/compiler/xla/service/custom_call_status.h"

void do_custom_call(CUstream stream, void** buffers, const char* opaque,
                    size_t opaque_len, xla::XlaCustomCallStatus* status);
```

Puede señalar un error mediante `XlaCustomCallStatusSetFailure`, por ejemplo:

```c++
void do_custom_call(void* out, const void** in, XlaCustomCallStatus* status) {
  // ... do some work.

  if (bad_condition) {
    char* error_message = "An error occurred";
    XlaCustomCallStatusSetFailure(status, error_message, strlen(error_message));
    return;
  }

  // ... continue.
}
```

También puede usar `XlaCustomCallStatusSetSuccess` para indicar que el proceso se ha completado correctamente, pero `XlaCustomCallStatus` tiene por defecto un estado de éxito, por lo que si lo ignora por completo también indicará que el proceso se ha completado correctamente.

Cuando use funciones de llamada personalizadas con esta firma, debe crear la operación `custom-call` correspondiente con la versión de API adecuada configurada, por ejemplo:

```c++
xla::CustomCall(&b, "do_custom_call", /*operands=*/{param0, param1},
                /*output_shape=*/xla::ShapeUtil::MakeShape(F32, {2048}),
                opaque, /*has_side_effect=*/false,
                /*output_operand_aliasing=*/{}, /*literal=*/nullptr,
                /*schedule=*/xla::CustomCallSchedule::SCHEDULE_NONE,
                /*api_version=*/API_VERSION_STATUS_RETURNING);
```

NOTA: En el futuro, todos los clientes deberán migrar sus funciones de llamada personalizadas a la nueva versión de API y la anterior quedará obsoleta. Para obtener llamadas personalizadas que no pueden fallar, simplemente agregue el nuevo parámetro `XlaCustomCallStatus*` y luego ignórelo.

En caso de error, no se usará ninguna de las salidas de llamadas personalizadas; el tiempo de ejecución de XLA finalizará el cálculo. No es posible que un cálculo de HLO se recupere del error (por ejemplo, detectándolo y manejándolo).

## Cómo pasar tuplas a llamadas personalizadas

Considere la siguiente llamada personalizada.

```c++
using xla::ShapeUtil;
using xla::F32;
Shape p0_shape = ShapeUtil::MakeTuple({
    ShapeUtil::MakeShape(F32, {32}),
    ShapeUtil::MakeTuple({
        ShapeUtil::MakeShape(F32, {64}),
        ShapeUtil::MakeShape(F32, {128}),
    }),
    ShapeUtil::MakeShape(F32, {256}),
});
xla::XlaOp p0 = xla::Parameter(0, p0_shape, "p0");

Shape out_shape = ShapeUtil::MakeTuple({
  ShapeUtil::MakeShape(F32, {512}),
  ShapeUtil::MakeShape(F32, {1024}),
});
xla::CustomCall(&b, "do_custom_call", /*operands=*/{p0}, out_shape);
```

Tanto en la CPU como en la GPU, una tupla se representa en la memoria como un arreglo de punteros. En pseudocódigo C++, el parámetro 0 anterior se presenta de la siguiente manera.

```c++
// In-memory layout of parameter 0 from custom-call above.  True on both CPU
// and GPU.
float* subbuf0 = new float[32];
float* subbuf1 = new float[64];
float* subbuf2 = new float[128]
float* subbuf3 = new float[256];

void* subtuple = new void*[2];
(*subtuple)[0] = subbuf1;
(*subtuple)[1] = subbuf2;

void* p0 = new void*[3];
(*p0)[0] = subbuf0;
(*p0)[1] = subtuple;
(*p0)[2] = subbuf3;
```

Aunque la representación en memoria de las tuplas es la misma en CPU y GPU, se manejan de manera diferente en las convenciones de llamadas personalizadas de CPU y GPU.

### Salidas de tupla como búferes temporales

Las entradas de tuplas para llamadas personalizadas son convenientes, pero no son estrictamente necesarias. Si no admitiéramos la entrada de tuplas en las llamadas personalizadas, siempre se podría desempaquetar las tuplas con get-tuple-element antes de pasarlas a la llamada personalizada.

Por otro lado, *las salidas* de tuplas nos permiten hacer cosas que de otra manera no podríamos hacer.

La razón obvia para tener salidas de tupla es que así es como una llamada personalizada (o cualquier otra operación de XLA) devuelve múltiples arreglos independientes.

Pero, aunque de manera menos evidente, una salida de tupla también es una forma de darle memoria temporal a su llamada personalizada. Sí, una *salida* puede representar un búfer temporal. Recuerde que un búfer de salida tiene la propiedad de que la operación puede escribir en él y puede leerlo después de que se haya escrito. Eso es exactamente lo que queremos de un búfer temporal.

En el ejemplo anterior, supongamos que queremos usar `F32[1024]` como búfer temporal. Luego escribiríamos la HLO como se indicó anteriormente y simplemente nunca leeríamos el índice de tupla 1 de la salida de la llamada personalizada.

### Tuplas en llamadas personalizadas de CPU

En el código de la CPU, tenemos una función `do_custom_call(const void** ins, void* out)`. `ins` es un arreglo con un solo elemento, que apunta a `param0`. Se puede acceder a los subbúferes de `param0` al hacer referencia a ese puntero, y a los subbúferes de `output_tuple` se puede acceder al hacer referencia a `out`.

### Tuplas en llamadas personalizadas de GPU

En el código de la GPU, tenemos una función `do_custom_call(..., void** buffers, ...)`. En este caso, `buffers` es un arreglo de host de *seis* punteros de dispositivo, uno para cada búfer de hoja en la entrada/salida. Para generar la lista plana, iteramos sobre los parámetros y la salida, y para cada uno hacemos una solicitud previa transversal a su forma. Concretamente:

```c++
// Layout of `buffers` parameter to GPU custom call function for custom-call
// above.
buffers[0] == subbuf0
buffers[1] == subbuf1
buffers[2] == subbuf2
buffers[3] == subbuf3
buffers[4] == output_subbuf0
buffers[5] == output_subbuf1
```
