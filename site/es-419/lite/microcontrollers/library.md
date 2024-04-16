# Comprender la librería C++

La librería C++ de TensorFlow Lite para microcontroladores forma parte del repositorio [TensorFlow](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro). Está diseñada para ser legible, fácil de modificar, bien probada, fácil de integrar y compatible con TensorFlow Lite normal.

El siguiente documento describe la estructura básica de la librería C++ y ofrece información sobre cómo crear su propio proyecto.

## Estructura de archivos

El directorio raíz [`micro`](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro) tiene una estructura relativamente sencilla. Sin embargo, dado que se encuentra dentro del extenso repositorio de TensorFlow, hemos creado scripts y archivos de proyecto pregenerados que ofrecen los archivos fuente relevantes de forma aislada dentro de varios entornos de desarrollo embebidos.

### Archivos clave

Los archivos más importantes para usar el intérprete de TensorFlow Lite para microcontroladores se encuentran en la raíz del proyecto, acompañados de pruebas:

```
[`micro_mutable_op_resolver.h`](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/micro_mutable_op_resolver.h)
can be used to provide the operations used by the interpreter to run the
model.
```

- [`micro_error_reporter.h`](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h) emite información de depuración.
- [`micro_interpreter.h`](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/micro_interpreter.h) contiene código para manejar y ejecutar modelos.

Consulte [Cómo empezar con los microcontroladores](get_started_low_level.md) para recibir un recorrido por el uso típico.

El sistema de compilación permite implementar determinados archivos específicos de la plataforma. Éstos se encuentran en un directorio con el nombre de la plataforma, por ejemplo [`cortex-m`](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/cortex_m_generic).

Existen varios directorios más, entre ellos:

- [`kernel`](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/kernels), que contiene las implementaciones de las operaciones y el código asociado.
- [`tools`](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/tools), que contiene las herramientas de compilación y sus resultados.
- [`examples`](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/examples), que contiene código de muestra.

## Iniciar un nuevo proyecto

Le recomendamos usar el ejemplo *Hello World* como plantilla para nuevos proyectos. Puede obtener una versión del mismo para la plataforma de su elección siguiendo las instrucciones de esta sección.

### Usar la librería Arduino

Si está usando Arduino, el ejemplo *Hello World* está incluido en la `Arduino_TensorFlowLite` librería de Arduino, que puede instalar manualmente en el IDE de Arduino y en [Arduino Create](https://create.arduino.cc/).

Una vez añadida la librería, vaya a `File -> Examples`. En la parte inferior de la lista debería ver un ejemplo llamado `TensorFlowLite:hello_world`. Selecciónelo y haga clic en `hello_world` para cargar el ejemplo. A continuación, puede guardar una copia del ejemplo y usarlo como base de su propio proyecto.

### Generar proyectos para otras plataformas

TensorFlow Lite para microcontroladores es capaz de generar proyectos independientes que contienen todos los archivos fuente necesarios, usando un `Makefile`. Los entornos soportados actualmente son Keil, Make y Mbed.

Para generar estos proyectos con Make, clone el repositorio [TensorFlow/tflite-micro](https://github.com/tensorflow/tflite-micro) y ejecute el siguiente comando:

```bash
make -f tensorflow/lite/micro/tools/make/Makefile generate_projects
```

Esto tardará unos minutos, ya que tiene que descargar algunas cadenas de herramientas grandes para las dependencias. Una vez que haya terminado, debería ver algunas carpetas creadas dentro de una ruta como `gen/linux_x86_64/prj/` (la ruta exacta depende de su sistema operativo anfitrión). Estas carpetas contienen el proyecto generado y los archivos fuente.

Tras ejecutar la orden, podrá encontrar los proyectos *Hello World* en `gen/linux_x86_64/prj/hello_world`. Por ejemplo, `hello_world/keil` contendrá el proyecto Keil.

## Ejecutar las pruebas

Para construir la librería y ejecutar todas sus pruebas de unidad, use la siguiente orden:

```bash
make -f tensorflow/lite/micro/tools/make/Makefile test
```

Para ejecutar una prueba individual, utilice el siguiente comando, sustituyendo `<test_name>` por el nombre de la prueba:

```bash
make -f tensorflow/lite/micro/tools/make/Makefile test_<test_name>
```

Puede encontrar los nombres de las pruebas en los Makefiles del proyecto. Por ejemplo, `examples/hello_world/Makefile.inc` especifica los nombres de las pruebas para el ejemplo *Hello World*.

## Compilar binarios

Para compilar un binario ejecutable para un proyecto determinado (como una aplicación de ejemplo), use el siguiente comando, sustituyendo `<project_name>` por el proyecto que desea compilar:

```bash
make -f tensorflow/lite/micro/tools/make/Makefile <project_name>_bin
```

Por ejemplo, el siguiente comando compilará un binario para la aplicación *Hello World*:

```bash
make -f tensorflow/lite/micro/tools/make/Makefile hello_world_bin
```

De forma predeterminada, el proyecto se compilará para el sistema operativo anfitrión. Para especificar una arquitectura de objetivo diferente, use `TARGET=` y `TARGET_ARCH=`. El siguiente ejemplo muestra cómo compilar el ejemplo *Hello World* para un cortex-m0 genérico:

```bash
make -f tensorflow/lite/micro/tools/make/Makefile TARGET=cortex_m_generic TARGET_ARCH=cortex-m0 hello_world_bin
```

Cuando se especifica un destino, se usará cualquier archivo fuente específico del destino disponible en lugar del código original. Por ejemplo, el subdirectorio `examples/hello_world/cortex_m_generic` contiene implementaciones de SparkFun Edge de los archivos `constants.cc` y `output_handler.cc`, que se usarán cuando se especifique el destino `cortex_m_generic`.

Puede encontrar los nombres del proyecto en los Makefiles del proyecto. Por ejemplo, `examples/hello_world/Makefile.inc` especifica los nombres binarios para el ejemplo *Hello World*.

## Kernels optimizados

Los kernels de referencia en la raíz de `tensorflow/lite/micro/kernels` están implementados en C/C++ puro, y no incluyen optimizaciones de hardware específicas de la plataforma.

Las versiones optimizadas de los kernels se facilitan en subdirectorios. Por ejemplo, `kernels/cmsis-nn` contiene varios kernels optimizados que usan la librería CMSIS-NN de Arm.

Para generar proyectos utilizando kernels optimizados, use el siguiente comando, reemplazando `<subdirectory_name>` por el nombre del subdirectorio que contiene las optimizaciones:

```bash
make -f tensorflow/lite/micro/tools/make/Makefile TAGS=<subdirectory_name> generate_projects
```

Puede añadir sus propias optimizaciones creando una nueva subcarpeta para ellas. Le animamos a realizar pull requests para nuevas implementaciones optimizadas.

## Generar la librería Arduino

Si necesita generar una nueva compilación de la librería, puede ejecutar el siguiente script desde el repositorio de TensorFlow:

```bash
./tensorflow/lite/micro/tools/ci_build/test_arduino.sh
```

La librería resultante se puede encontrar en `gen/arduino_x86_64/prj/tensorflow_lite.zip`.

## Portar a nuevos dispositivos

[`micro/docs/new_platform_support.md`](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/docs/new_platform_support.md) le puede orientar sobre cómo portar TensorFlow Lite para microcontroladores a nuevas plataformas y dispositivos.
