# Conversión de modelos

TensorFlow.js viene con varios modelos previamente entrenados que están listos para usar en el navegador. Puede encontrarlos en nuestro [repositorio de modelos](https://github.com/tensorflow/tfjs-models). Aunque, probablemente ya haya encontrado o escrito otro modelo que se encuentra en alguna otra parte y que quisiera usar en su aplicación web. TensorFlow.js cuenta con un [conversor](https://github.com/tensorflow/tfjs/tree/master/tfjs-converter) de modelos, para estos casos. El conversor de  TensorFlow.js tiene dos componentes:

1. Una utilidad de línea de comandos que convierte los modelos de Keras y TensorFlow para usarlos en TensorFlow.js.
2. Una API para cargar y ejecutar el modelo en el navegador con TensorFlow.js.

## Conversión de su propio modelo

El conversor de TensorFlow.js funciona con muchos formatos diferentes de modelos:

**SavedModel**: es el formato predeterminado en el que se guardan los modelos de TensorFlow. El formato SavedModel está documentado [aquí](https://www.tensorflow.org/guide/saved_model).

**Modelo Keras**: por lo general, los modelos Keras se guardan en un archivo HDF5. Para más información sobre cómo guardar modelos Keras, consulte [aquí](https://keras.io/getting-started/faq/#savingloading-whole-models-architecture-weights-optimizer-state).

**Módulo de TensorFlow Hub**: estos son modelos que han sido empaquetados para su distribución en TensorFlow Hub, una plataforma para compartir y descubrir modelos. Puede encontrar la biblioteca de modelos [aquí](https://tfhub.dev/).

Dependiendo del tipo de modelo que intente convertir, necesitará pasar distintos argumentos al conversor. Por ejemplo, digamos que ha guardado un modelo Keras denominado `model.h5` en su directorio `tmp/`. Para convertir el modelo con el conversor de TensorFlow.js, puede ejecutar el siguiente comando:

```
$ tensorflowjs_converter --input_format=keras /tmp/model.h5 /tmp/tfjs_model
```

De este modo, se convertirá el modelo en `/tmp/model.h5` y saldrá un archivo `model.json` junto con los archivos binarios hacia su directorio `tmp/tfjs_model/`.

Para más detalles sobre los argumentos de la línea de comandos correspondiente a los diferentes formatos de modelos, consulte el archivo [README](https://github.com/tensorflow/tfjs/tree/master/tfjs-converter) del conversor de TensorFlow.js.

Durante el proceso de conversión recorre el grafo modelo y controla que cada una de las operaciones sea compatible con TensorFlow.js. De ser así, escribimos el grafo en un formato que el navegador pueda consumir. Intentamos optimizar el modelo para reciba el servicio en la web mediante el particionamiento horizontal de los pesos en archivos de 4MB. De este modo, los navegadores podrán guardarlos en caché. También intentamos simplificar el grafo del modelo usando el proyecto [Grappler](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/core/grappler) de código abierto. Entre las simplificaciones del grafo se incluye plegar las operaciones adyacentes juntas, eliminar subgrafos comunes, etc. Estos cambios no tienen efecto alguno en la salida del modelo. Para una optimización mayor, los usuarios pueden pasar un argumento que instruya al conversor para que cuantifique el modelo a un tamaño de byte específico. La cuantificación es una técnica que se usa para reducir el tamaño del modelo mediante la representación de pesos con menos bits. Los usuarios deben prestar particular atención a confirmar que el modelo mantenga un grado aceptable de exactitud después de la cuantificación.

Si encontramos una operación incompatible durante la conversión, el proceso falla y le señalamos el nombre de la operación al usuario. En caso de que surja un inconveniente con nuestro [GitHub](https://github.com/tensorflow/tfjs/issues), no dude en hacérnoslo saber; intentaremos implementar operaciones nuevas para responder a las demandas de los usuarios.

### Prácticas recomendadas

A pesar de que hacemos todo lo posible por optimizar el modelo durante la conversión, con frecuencia, la mejor manera de garantizar que el modelo tendrá un buen desempeño es crearlo teniendo en cuenta los entornos de recursos limitados. Significa que debemos evitar las arquitecturas demasiado complejas y minimizar la cantidad de parámetros (pesos) siempre que sea posible.

## Ejecución del modelo

Después de convertir el modelo correctamente, lo que tendrá será un conjunto de archivos de pesos y un archivo con la topología del modelo. TensorFlow.js ofrece varias API para cargar modelos, que se pueden usar para buscar y traer estos activos de modelos y ejecutar la inferencia en el navegador.

A continuación, mostramos cómo se ve la API utilizada para un SavedModel de TensorFlow o un módulo de TensorFlow Hub después de haber sido convertidos:

```js
const model = await tf.loadGraphModel(‘path/to/model.json’);
```

Y así es cómo se ve con un modelo Keras convertido:

```js
const model = await tf.loadLayersModel(‘path/to/model.json’);
```

La API `tf.loadGraphModel` devuelve un `tf.FrozenModel`. Significa que los parámetros son fijos y que no podrá realizar el ajuste fino de su modelo con los nuevos datos. La API `tf.loadLayersModel` devuelve un tf.Model, que se puede entrenar. Para más información sobre cómo entrenar a un tf.Model, consulte la guía sobre cómo [entrenar modelos](train_models.md).

Después de la conversión, es conveniente ejecutar la inferencia algunas veces y tomar como referencia la velocidad del modelo. Contamos con una página de referencias independiente que se puede utilizar con este fin: https://tensorflow.github.io/tfjs/e2e/benchmarks/local-benchmark/index.html. Habrá notado que descartamos las mediciones de una ejecución de preparación inicial. El motivo es que, en general, la primera inferencia del modelo es mucho más lenta que las subsiguientes, debido al sobrecosto de crear texturas y compilar sombreadores.
