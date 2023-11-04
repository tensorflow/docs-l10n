# Prácticas recomendadas de prueba de TensorFlow

Estas son las prácticas recomendadas para probar código en el [repositorio de TensorFlow](https://github.com/tensorflow/tensorflow) .

## Antes de empezar

Antes de contribuir con código fuente a un proyecto de TensorFlow, revise el archivo `CONTRIBUTING.md` en el repositorio de GitHub del proyecto. Por ejemplo, consulte el archivo [CONTRIBUTING.md](https://github.com/tensorflow/tensorflow/blob/master/CONTRIBUTING.md) en el repositorio básico de TensorFlow. Todos los contribuyentes de código deben firmar un [Acuerdo de licencia de colaborador](https://cla.developers.google.com/clas) (CLA, por sus siglas en inglés).

## Principios generales

### Solo depende de lo que use en sus reglas BUILD

TensorFlow es una enorme biblioteca, y es común depender del paquete completo cuando se escribe una prueba unitaria para sus submódulos. Sin embargo, esto deshabilita el análisis basado en dependencias de `bazel`. Esto significa que los sistemas de integración continua no pueden eliminar inteligentemente las pruebas que no están relacionadas para las ejecuciones presubmit/postsubmit. Si solo se depende de los submódulos que se están probando en el archivo `BUILD`, ahorrará tiempo a todos los desarrolladores de TensorFlow, y una importante cantidad de valiosa potencia de cálculo.

Sin embargo, modificar su dependencia de compilación para omitir los objetivos de TF completos trae algunas limitaciones respecto a lo que puede importar en su código Python. Ya no podrá utilizar el `import tensorflow as tf` en sus pruebas unitarias. Pero esta es una compensación que vale la pena, ya que ahorra a todos los desarrolladores la necesidad de ejecutar miles de pruebas innecesarias.

### Todo el código debe tener pruebas unitarias

Para cualquier código que escriba, también debe escribir sus pruebas unitarias. Si escribe un archivo nuevo `foo.py`, debe colocar sus pruebas unitarias en `foo_test.py` y enviarlo dentro del mismo cambio. Procure alcanzar una cobertura de pruebas incremental &gt;90% para todo su código.

### Evite el uso de reglas de prueba nativas de Bazel en TF

TF tiene muchas sutilezas a la hora de ejecutar pruebas. Hemos trabajado para ocultar todas esas complejidades en nuestras macros de bazel. Para no tener que lidiar con ellas, utilice lo siguiente en lugar de las reglas de prueba nativas. Tenga en cuenta que todas están definidas en `tensorflow/tensorflow.bzl` Para pruebas CC, use `tf_cc_test`, `tf_gpu_cc_test`, `tf_gpu_only_cc_test`. Para pruebas de Python, use `tf_py_test` o `gpu_py_test`. Si necesita algo realmente parecido a la regla nativa `py_test`, use la definida en tensorflow.bzl. Solo tiene que agregar la siguiente línea en la parte superior del archivo BUILD: `load(“tensorflow/tensorflow.bzl”, “py_test”)`

### Tenga en cuenta dónde se ejecuta la prueba

Cuando escribe una prueba, nuestra infraestructura de prueba puede encargarse de ejecutar sus pruebas en CPU, GPU y aceleradores si las escribe correctamente. Contamos con pruebas automatizadas que se ejecutan en Linux, macOS, Windows, que tengan sistemas con o sin GPU. Lo único que tiene que hacer es elegir una de las macros que mencionamos anteriormente y, a continuación, utilizar etiquetas para limitar dónde se ejecutan.

- `manual` impedirá que su prueba se ejecute en cualquier lugar. Esto incluye ejecuciones de pruebas manuales que utilizan patrones como `bazel test tensorflow/…`

- `no_oss` evitará que su prueba se ejecute en la infraestructura oficial de pruebas del software de código abierto de TF.

- `no_mac` o `no_windows` se pueden utilizar para evitar que la prueba se incluya en los paquetes de pruebas de los sistemas operativos correspondientes.

- `no_gpu` se puede usar para evitar que la prueba se ejecute en paquetes de pruebas de GPU.

### Verifique que las pruebas se ejecuten en los conjuntos de pruebas previstos

TF tiene bastantes paquetes de pruebas. A veces, puede resultar confuso configurarlos. Se pueden presentar diferentes problemas que hagan que sus pruebas se omitan de las compilaciones continuas. Por lo tanto, debe verificar que sus pruebas se ejecuten según lo esperado. Para esto, haga lo siguiente:

- Espere a que se completen los presubmit de su solicitud de incorporación (PR).
- Desplácese hasta la parte inferior de su PR para ver las verificaciones de estado.
- Haga clic en el vínculo "Detalles" ubicado en el lado derecho de cualquier control de Kokoro.
- Consulte la lista "Objetivos" para encontrar los objetivos recién agregados.

### Cada clase/unidad debe tener su propio archivo de prueba unitaria

Las clases de prueba independientes nos ayudan a aislar mejor los fallos y los recursos. Generan archivos de prueba mucho más cortos y fáciles de leer. Por lo tanto, todos sus archivos Python deben tener al menos un archivo de prueba correspondiente (Por cada `foo.py`, debe haber un `foo_test.py`). En el caso de las pruebas más elaboradas, como pruebas de integración que requieran distintas configuraciones, es válido agregar más archivos de prueba.

## Velocidad y tiempos de ejecución

### La fragmentación debe usarse lo menos posible

En lugar de fragmentar, haga lo siguiente:

- Reduzca sus pruebas
- Si lo anterior no es posible, divida las pruebas

La fragmentación ayuda a reducir la latencia general de una prueba, pero se puede lograr lo mismo si se dividen las pruebas en objetivos más pequeños. La división de pruebas nos ofrece un mayor nivel de control en cada prueba, lo que minimiza las ejecuciones presubmit innecesarias y reduce la pérdida de cobertura de un buildcop que deshabilita un objetivo completo debido a un caso de prueba que no funciona correctamente. Además, la fragmentación conlleva costos ocultos que no son tan evidentes, como la ejecución de todo el código de inicialización de pruebas para todas las fragmentaciones. Los equipos de infraestructura nos han planteado este problema como el origen de una carga adicional.

### Las pruebas más pequeñas son mejores

Cuanto más rápido se ejecuten sus pruebas, más probable será que la gente las ejecute. Un segundo de más en una prueba puede suponer horas de tiempo extra para los desarrolladores y para nuestra infraestructura. Intente que sus pruebas se ejecuten en menos de 30 segundos (¡en modo no opcional!) y que sean pequeñas. Solo marque sus pruebas como medianas si no tiene otra opción. ¡La infraestructura no ejecuta pruebas importantes como presubmit o postsubmit! Por lo tanto, solo escriba una prueba grande si va a organizar dónde se ejecutará. A continuación, le dejamos algunos consejos para que las pruebas se ejecuten más rápido:

- Ejecute menos iteraciones de entrenamiento en su prueba
- Considere la posibilidad de recurrir a la inyección de dependencias para reemplazar las grandes dependencias del sistema bajo prueba con simples falsificaciones.
- Estudie la posibilidad de usar datos de entrada más pequeños en las pruebas unitarias
- Si nada más funciona, pruebe con dividir el archivo de prueba.

### Los tiempos de prueba deben apuntar a la mitad del tiempo de espera del tamaño de la prueba para evitar la inestabilidad

Con los objetivos de prueba `bazel`, las pruebas pequeñas tienen tiempos de espera de 1 minuto. Las pruebas medianas tienen tiempos de espera de 5 minutos. La infraestructura de pruebas de TensorFlow directamente no ejecuta las pruebas grandes. Sin embargo, muchas pruebas no son deterministas en cuanto a la cantidad de tiempo que tardan. Por diversos motivos, sus pruebas pueden demorar más tiempo de vez en cuando. Y, si marca como pequeña una prueba que se ejecuta durante un promedio de 50 segundos, la prueba fallará si se programa en una máquina con una CPU antigua. Por lo tanto, intente alcanzar un tiempo de ejecución promedio de 30 segundos para pruebas pequeñas. Para las pruebas medianas, intente conseguir un tiempo de ejecución promedio de 2 minutos y 30 segundos.

### Reduzca el número de muestras y aumente las tolerancias para el entrenamiento

Las pruebas de ejecución lenta disuaden a los contribuyentes. La ejecución de entrenamientos en las pruebas puede ser muy lenta. Elija tolerancias más altas para poder utilizar menos muestras en sus pruebas y mantenerlas lo suficientemente rápidas (2,5 minutos como máximo).

## Elimine la falta de determinismo y las inestabilidades

### Escriba pruebas deterministas

Las pruebas unitarias siempre deben ser deterministas. Todas las pruebas que se ejecutan en TAP y Guitar deben ejecutarse de la misma manera cada vez, si no hay ningún cambio de código que las afecte. Para garantizar esto, hay que tener en cuenta algunos puntos.

### Siembre siempre cualquier fuente de estocasticidad

Cualquier generador de números aleatorios, o cualquier otra fuente de estocasticidad puede causar inestabilidad. Por lo tanto, hay que sembrar cada uno de ellos. Además de hacer que las pruebas sean menos complicadas, esto hace que todas las pruebas se puedan reproducir. Las diferentes formas de establecer algunas semillas que puede necesitar en las pruebas de TF son las siguientes:

```python
# Python RNG
import random
random.seed(42)

# Numpy RNG
import numpy as np
np.random.seed(42)

# TF RNG
from tensorflow.python.framework import random_seed
random_seed.set_seed(42)
```

### Evite el uso de `sleep` en pruebas multiproceso

El uso de la función `sleep` en las pruebas puede ser una de las principales causas de inestabilidad. Especialmente cuando se utilizan varios subprocesos, utilizar sleep para esperar a otro subproceso nunca será determinista. Esto se debe a que el sistema no puede garantizar ningún orden de ejecución de diferentes subprocesos o procesos. Por lo tanto, se recomienda el uso de construcciones de sincronización deterministas como las exclusiones mutuas.

### Compruebe si la prueba es inestable

Las inestabilidades hacen que los constructores y los desarrolladores pierdan muchas horas. Son difíciles de detectar y difíciles de depurar. Aunque existen sistemas automatizados para detectar inestabilidades, necesitan acumular cientos de ejecuciones de pruebas antes de poder incluirlas en la lista de forma precisa. Incluso cuando las detectan, rechazan sus pruebas y se pierde la cobertura de la prueba. Por lo tanto, los autores de pruebas deben comprobar si sus pruebas son inestables al escribirlas. Esto se puede hacer fácilmente si se ejecuta la prueba con la marca: `--runs_per_test=1000`

### Use TensorFlowTestCase

TensorFlowTestCase toma las precauciones necesarias, como sembrar todos los generadores de números aleatorios utilizados para reducir la inestabilidad tanto como sea posible. A medida que descubramos y solucionemos más fuentes de inestabilidad, todas ellas se agregarán a TensorFlowTestCase. Por lo tanto, debe usar TensorFlowTestCase al escribir pruebas para tensorflow. TensorFlowTestCase se define aquí: `tensorflow/python/framework/test_util.py`

### Escriba pruebas herméticas

Las pruebas herméticas no necesitan recursos externos. Tienen todo lo que necesitan y simplemente inician cualquier servicio falso que puedan necesitar. Cualquier servicio distinto de sus pruebas es fuente de no determinismo. Incluso con una disponibilidad del 99 % de otros servicios, la red puede fallar, la respuesta de rpc puede retrasarse y podría recibir un mensaje de error inexplicable. Los servicios externos pueden ser, entre otros, GCS, S3 o cualquier sitio web.
