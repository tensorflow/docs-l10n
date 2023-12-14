# Inicio rápido para dispositivos basados en Linux con Python

Usar TensorFlow Lite con Python es estupendo para dispositivos embebidos basados en Linux, como [Raspberry Pi](https://www.raspberrypi.org/){:.external} y [Dispositivos Coral con Edge TPU](https://coral.withgoogle.com/){:.external}, entre muchos otros.

Esta página muestra cómo puede empezar a ejecutar modelos TensorFlow Lite con Python en sólo unos minutos. Todo lo que necesita es un modelo TensorFlow [convertido a TensorFlow Lite](../models/convert/). (Si aún no tiene un modelo convertido, puede experimentar usando el modelo que se incluye con el ejemplo enlazado a continuación).

## Acerca del paquete runtime de TensorFlow Lite

Para empezar rápidamente a ejecutar modelos de TensorFlow Lite con Python, puede instalar sólo el intérprete de TensorFlow Lite, en lugar de todos los paquetes de TensorFlow. Llamamos a este paquete Python simplificado `tflite_runtime`.

El paquete `tflite_runtime` es una fracción del tamaño del paquete completo `tensorflow` e incluye el código mínimo necesario para ejecutar inferencias con TensorFlow Lite-principalmente la clase de Python [`Interpreter`](https://www.tensorflow.org/api_docs/python/tf/lite/Interpreter). Este pequeño paquete es ideal cuando lo único que se desea es ejecutar modelos `.tflite` y evitar malgastar espacio en disco con la enorme librería TensorFlow.

Nota: Si necesita acceder a otras APIs de Python, como el [Convertidor de TensorFlow Lite](../models/convert/), debe instalar el [paquete completo de TensorFlow](https://www.tensorflow.org/install/). Por ejemplo, las [ops de TensorFlow seleccionadas] (https://www.tensorflow.org/lite/guide/ops_select) no están incluidas en el paquete `tflite_runtime`.Si sus modelos tienen alguna dependencia con las ops de TF seleccionadas, deberá usar en su lugar el paquete TensorFlow completo.

## Instale TensorFlow Lite para Python

Puede instalarlo en Linux con pip:

<pre class="devsite-terminal devsite-click-to-copy">python3 -m pip install tflite-runtime
</pre>

## Plataformas compatibles

Las wheels Python `tflite-runtime` están precompiladas y se proporcionan para estas plataformas:

- Linux armv7l (por ejemplo, Raspberry Pi 2, 3, 4 y Zero 2 con Raspberry Pi OS de 32 bits)
- Linux aarch64 (por ejemplo, Raspberry Pi 3, 4 ejecutando Debian ARM64)
- Linux x86_64

Si desea ejecutar modelos TensorFlow Lite en otras plataformas, debe usar el [paquete completo de TensorFlow](https://www.tensorflow.org/install/), o [compilar el paquete tflite-runtime a partir del código fuente](build_cmake_pip.md).

Si está usando TensorFlow con la TPU Coral Edge, deberá seguir en su lugar la [documentación apropiada de configuración de Coral](https://coral.ai/docs/setup).

Nota: Ya no actualizamos el paquete Debian `python3-tflite-runtime`. El último paquete Debian es para la versión 2.5 de TF, que puede instalar siguiendo [estas instrucciones más antiguas](https://github.com/tensorflow/tensorflow/blob/v2.5.0/tensorflow/lite/g3doc/guide/python.md#install-tensorflow-lite-for-python).

Nota: Ya no publicamos wheels `tflite-runtime` precompiladas para Windows y macOS. Para estas plataformas, debe usar el [paquete completo de TensorFlow](https://www.tensorflow.org/install/), o [compilar el paquete tflite-runtime desde el código fuente](build_cmake_pip.md).

## Ejecutar una inferencia usando tflite_runtime

En lugar de importar `Interpreter` del módulo `tensorflow`, ahora deberá importarlo de `tflite_runtime`.

Por ejemplo, después de instalar el paquete mencionado, copie y ejecute el archivo [`label_image.py`](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/examples/python/). Mandará un error (probablemente) porque no tiene instalada la librería `tensorflow`. Para solucionarlo, edite esta línea del archivo:

```python
import tensorflow as tf
```

para que en su lugar diga:

```python
import tflite_runtime.interpreter as tflite
```

Y luego cambie esta línea:

```python
interpreter = tf.lite.Interpreter(model_path=args.model_file)
```

para que diga:

```python
interpreter = tflite.Interpreter(model_path=args.model_file)
```

Ahora ejecute de nuevo `label_image.py`. Listo. Ya está ejecutando modelos TensorFlow Lite.

## Más información

- Para más detalles sobre la API `Interpreter`, lea [Cargar y ejecutar un modelo en Python](inference.md#load-and-run-a-model-in-python).

- Si tiene una Raspberry Pi, consulte una serie de [vídeos](https://www.youtube.com/watch?v=mNjXEybFn98&list=PLQY2H8rRoyvz_anznBg6y3VhuSMcpN9oe) sobre cómo ejecutar la detección de objetos en Raspberry Pi usando TensorFlow Lite.

- Si usa un acelerador Coral ML, revise los [ejemplos de Coral en GitHub](https://github.com/google-coral/tflite/tree/master/python/examples).

- Para convertir otros modelos TensorFlow a TensorFlow Lite, lea sobre el [Conversor de TensorFlow Lite](../models/convert/).

- Si desea generar la wheel `tflite_runtime`, lea [Generar el paquete de la wheel Python de TensorFlow Lite](build_cmake_pip.md).
