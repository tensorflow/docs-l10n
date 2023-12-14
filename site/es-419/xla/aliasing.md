# Creación de alias ​​en XLA

Este documento describe la API de creación de alias para XLA: al compilar un programa XLA, puede especificar el alias deseado entre los búferes de entrada y salida.

## Definición de alias durante la compilación

Por ejemplo, pensemos en un módulo HLO simple que sencillamente sume `1` a su entrada:

```
HloModule increment

ENTRY entry {
  %p = f32[] parameter(0)
  %c = f32[] constant(1)
  ROOT %out = f32[] add(%p, %c)
}
```

Este módulo asignará dos búferes de 4 bytes: uno para la entrada `%p` y otro para la salida `%out`.

No obstante, a menudo es conveniente ejecutar la actualización in situ (por ejemplo, si en la interfaz que genera la expresión la variable de entrada ya no existe después del cálculo, como en el incremento `p++`).

Para ejecutar dicha actualización de manera eficiente, puede especificar el alias de entrada:

```
HloModule increment, input_output_alias={ {}: 0 }

ENTRY entry {
  %p = f32[] parameter(0)
  %c = f32[] constant(1)
  ROOT %out = f32[] add(%p, %c)
}
```

El formato especifica que toda la salida (marcada por `{}`) tiene un alias con el parámetro de entrada `0`.

Consulte la API [`XlaBuilder::SetUpAlias`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/xla_builder.h) ​​para especificar el alias mediante programación.

## Definición de alias durante el tiempo de ejecución

El alias que se definió en el paso anterior se especifica durante la *compilación*. Durante la ejecución, puede elegir si desea donar el búfer a través de la API [`LocalClient::RunAsync`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/local_client.h).

Los búferes de entrada al programa se incluyen en [`ExecutionInput`](https://www.tensorflow.org/code/tensorflow/compiler/xla/service/executable.h), que a su vez contiene un árbol de `MaybeOwningDeviceMemory`. Si la memoria se especifica como *propietaria* (la propiedad del búfer se pasa al tiempo de ejecución de XLA), el búfer realmente se dona y la actualización se ejecuta in situ, tal y como solicita la API de creación de alias en tiempo de compilación.

Sin embargo, si el búfer que tiene un alias en el momento de la compilación *no* se dona en el tiempo de ejecución, se activa *la protección contra copia*: se asigna un búfer de salida adicional `O` y el contenido del búfer de entrada `P` que debía tener un alias se copia en `O` (de modo que el programa puede ejecutarse como si el búfer `O` se hubiera donado en tiempo de ejecución).

## Interoperabilidad del frontend

### TF/XLA

En los clústeres del programa TensorFlow compilado con XLA, todas las actualizaciones de variables de recursos tienen un alias en el momento de la compilación (la creación de alias en el tiempo de ejecución depende de si algo más contiene una referencia al tensor de variables de recursos).
