# Problemas conocidos

La compilación con XLA puede mejorar en gran medida el rendimiento de sus programas, pero la interoperabilidad de TensorFlow tiene una serie de puntos débiles conocidos.

## `tf.Variable` en un dispositivo diferente

*Mensaje de error*: `INVALID_ARGUMENT: Trying to access resource <Variable> (defined @ <Loc>) located in device CPU:0 from device GPU:0`

El clúster de XLA se ejecuta exactamente en un dispositivo y no puede leer ni escribir en `tf.Variable` ubicado en un dispositivo diferente. Por lo general, este mensaje de error indica que, para empezar, la variable no se colocó en el dispositivo correcto. El mensaje de error debe especificar con precisión la ubicación de la variable infractora.

NOTA: `tf.Variable` de tipo `int32` siempre se colocan en un host y no se pueden colocar en una GPU. Como solución alternativa, se puede usar `int64`.

## La interconversión TensorArray TF/XLA no es compatible

*Mensaje de error*: `Support for TensorList crossing the XLA/TF boundary is not implemented` .

XLA es compatible con `tf.TensorArray`. Sin embargo, la *interconversión* entre las representaciones TF y XLA aún no se ha implementado. Este error surge a menudo cuando `TensorArray` se usa dentro del bloque compilado, pero la derivada se toma afuera.

*Solución alternativa*: compile el alcance más externo que toma la derivada.

## Los bucles while de TensorFlow deben estar delimitados (o tener la retropropagación desactivada)

*Mensaje de error*: `XLA compilation requires a fixed tensor list size. Set the max number of elements. This could also happen if you're using a TensorArray in a while loop that does not have its maximum_iteration set, you can fix this by setting maximum_iteration to a suitable value` .

Los [bucles](https://www.tensorflow.org/api_docs/python/tf/while_loop) while de TF creados con `tf.while_loop` admiten la retropropagación al acumular todos los resultados intermedios en un `TensorArray`, pero XLA solo admite `TensorArray` limitados.

*Solución alternativa*: todos los bucles while compilados deben tener el parámetro `maximum_iterations` establecido en un valor constante conocido en el momento de la compilación o la retropropagación deshabilitada con ayuda de `back_prop=False`.

## `tf.TensorArray` dinámico no es compatible

Las escrituras en `tf.TensorArray(..., dynamic_size=True)` no son compilables con XLA, ya que dichas escrituras requieren un número desconocido de reasignaciones cuando el arreglo excede el límite original.

*Solución alternativa*: proporcione un límite estáticamente conocido a sus arreglos.

## La generación de números aleatorios ignora la semilla de TF

XLA actualmente ignora las semillas de TF para operaciones aleatorias. Esto afecta las operaciones aleatorias de TF con estado, como `tf.random.normal` o `tf.nn.dropout`. XLA se comportará como si la compilación se sembrara con una nueva semilla única en cada ejecución dentro del mismo proceso (la primera ejecución del proceso siempre arrojará el mismo resultado).

*Solución alternativa*: directamente use los [RNG recomendados](https://www.tensorflow.org/guide/random_numbers#stateless_rngs), como `tf.random.stateless_uniform` o `tf.random.Generator`.

## No se admiten entradas que deben ser constantes y que son funciones de variables de inducción

*Mensaje de error*: `XLA compilation requires that operator arguments that represent shapes or dimensions be evaluated to concrete values at compile time. This error means that a shape or dimension argument could not be evaluated at compile time, usually because the value of the argument depends on a parameter to the computation, on a variable, or on a stateful operation such as a random number generator`

XLA requiere que ciertos valores sean conocidos en tiempo de compilación, como el eje de reducción de una operación de reducción o las dimensiones de transposición. Piense en el caso en el que, por ejemplo, el eje de reducción se defina como una función de una variable de inducción de `tf.range`: resolverlo estáticamente no será posible sin desenrollar todo el bucle, lo que podría no ser conveniente para el usuario.

*Solución alternativa*: deshaga los bucles, por ejemplo, convirtiendo `tf.range` en `range` de Python.

NOTA: El mensaje de error anterior no es exclusivo de este problema y puede surgir debido a otras limitaciones o errores.
