# API de capas de TensorFlow.js para usuarios de Keras

La API Layers de TensorFlow.js está modelada como Keras y tratamos de hacer que la [API Layers](https://js.tensorflow.org/api/latest/#Layers) sea lo más parecido posible a Keras, dentro de lo razonable, considerando las diferencias que existen entre JavaScript y Python. De este modo, a los usuarios con experiencia en desarrollo de modelos Keras en Python les resulta más fácil migrar a TensorFlow.js Layers en JavaScript. Por ejemplo, el siguiente código Keras traduce a JavaScript:

```python
# Python:
import keras
import numpy as np

# Build and compile model.
model = keras.Sequential()
model.add(keras.layers.Dense(units=1, input_shape=[1]))
model.compile(optimizer='sgd', loss='mean_squared_error')

# Generate some synthetic data for training.
xs = np.array([[1], [2], [3], [4]])
ys = np.array([[1], [3], [5], [7]])

# Train model with fit().
model.fit(xs, ys, epochs=1000)

# Run inference with predict().
print(model.predict(np.array([[5]])))
```

```js
// JavaScript:
import * as tf from '@tensorflow/tfjs';

// Build and compile model.
const model = tf.sequential();
model.add(tf.layers.dense({units: 1, inputShape: [1]}));
model.compile({optimizer: 'sgd', loss: 'meanSquaredError'});

// Generate some synthetic data for training.
const xs = tf.tensor2d([[1], [2], [3], [4]], [4, 1]);
const ys = tf.tensor2d([[1], [3], [5], [7]], [4, 1]);

// Train model with fit().
await model.fit(xs, ys, {epochs: 1000});

// Run inference with predict().
model.predict(tf.tensor2d([[5]], [1, 1])).print();
```

Sin embargo, hay algunas diferencias que quisiéramos destacar y explicar en este documento. Una vez que entienda estas diferencias y el razonamiento que hay detrás de ellas, la migración de Python a JavaScript (o la migración en la dirección inversa) debería fluir relativamente sin problemas.

## Los constructores toman objetos de JavaScript como configuraciones

Compare las siguientes líneas de Python y JavaScript del ejemplo anterior: ambas crean una capa [densa](https://keras.io/api/layers/core_layers/dense).

```python
# Python:
keras.layers.Dense(units=1, inputShape=[1])
```

```js
// JavaScript:
tf.layers.dense({units: 1, inputShape: [1]});
```

Las funciones de JavaScript no tienen un equivalente de los argumentos de las palabras clave en las funciones de Python. Intentaremos evitar la implementación de opciones constructor como argumentos posicionales en JavaScript, ya que usarlas y recordarlas sería particularmente engorroso para los constructores, por la gran cantidad de argumentos de palabras clave (p. ej., [LSTM](https://keras.io/api/layers/recurrent_layers/lstm)). Es el motivo por el cual utilizamos objetos de configuración de JavaScript. Dichos objetos proporcionan el mismo nivel de invariancia y flexibilidad posicional que los argumentos de palabras clave de Python.

Algunos métodos de clase "modelo", p. ej., [`Model.compile()`](https://keras.io/models/model/#model-class-api), también toman como entrada a un objeto de configuración de JavaScript. Sin embargo, no debemos olvidar que `Model.fit()`, `Model.evaluate()` y `Model.predict()` son algo diferentes. Como estos métodos toman los datos de `x` (características) e `y` (etiquetas o destinos) obligatorios como entradas; `x` e `y` son argumentos posicionales separados del objeto de configuración de garantía que cumple el rol de los argumentos de palabra clave. Por ejemplo:

```js
// JavaScript:
await model.fit(xs, ys, {epochs: 1000});
```

## Model.fit() es asincrónico

`Model.fit()` es el método primario con el que los usuarios realizan el entrenamiento del modelo en TensorFlow.js. En muchos casos este método puede tener una duración prolongada de ejecución y durar segundos o, incluso, minutos. Por lo tanto, utilizamos la característica `async` del lenguaje JavaScript, para que esta función se pueda usar de modo tal que no bloquee al hilo de UI principal, mientras se ejecuta en el navegador. Es similar a otras funciones de potencial larga duración en JavaScript, como `async` [fetch](https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API). Tenga en cuenta que `async` es una construcción que no existe en Python. Mientras que el método [`fit()`](https://keras.io/models/model/#model-class-api) en Keras devuelve un objeto de historia, la contraparte del método `fit()` en JavaScript devuelve una [promesa](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Promise) de historia, que se puede [esperar](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Operators/await) (como en el ejemplo anterior) o usar con el método <em>then()</em>.

## Sin NumPy para TensorFlow.js

Los usuarios de Python y Keras, por lo general, usan [NumPy](http://www.numpy.org/) para realizar operaciones de arreglos y numéricas básicas, como la generación de los tensores 2D del ejemplo anterior.

```python
# Python:
xs = np.array([[1], [2], [3], [4]])
```

En TensorFlow.js, este tipo de operaciones numéricas básicas se hace con el paquete mismo. Por ejemplo:

```js
// JavaScript:
const xs = tf.tensor2d([[1], [2], [3], [4]], [4, 1]);
```

El nombre de espacio `tf.*` también proporciona otras funciones para arreglar y alinear operaciones de álgebra como la matriz de multiplicación. Para más información, consulte la sección sobre [documentación de TensorFlow.js Core](https://js.tensorflow.org/api/latest/).

## Métodos de factoría, no constructores

Esta línea en Python (del ejemplo anterior) es una llamada de constructor:

```python
# Python:
model = keras.Sequential()
```

Si se traduce estrictamente en JavaScript, la llamada del constructor equivalente sería como se muestra a continuación:

```js
// JavaScript:
const model = new tf.Sequential();  // !!! DON'T DO THIS !!!
```

Sin embargo, decidimos no usar los constructores “nuevos” porque 1) la “nueva” palabra clave haría que el código se sobredimensionara y 2) el constructor "nuevo" es considerado una “parte mala” de JavaScript: un potencial inconveniente, tal como se lo describe en [*JavaScript: the Good Parts*](https://www.oreilly.com/library/view/javascript-the-good/9780596517748/) (JavaScript: las partes buenas). Para crear modelos y capas en TensorFlow.js, puede llamar a métodos de factoría que tengan nombres con el formato "*lowerCamelCase*", por ejemplo:

```js
// JavaScript:
const model = tf.sequential();

const layer = tf.layers.batchNormalization({axis: 1});
```

## Las opciones de valores de las strings son lowerCamelCase, no snake_case

En JavaScript, es más común usar la combinación de mayúsculas y minúsculas en los nombres de los símbolos (p. ej., consulte la [Google JavaScript Style Guide](https://google.github.io/styleguide/jsguide.html#naming-camel-case-defined)), en comparación con Python, donde es más común usar todas minúsculas (p. ej., en Keras). Dicho esto, decidimos usar <em>lowerCamelCase</em> (la combinación de minúsculas y mayúsculas) en los valores de las strings para las opciones que incluyen lo siguiente:

- DataFormat; p. ej., **`channelsFirst`** en vez de `channels_first`
- Inicializador; p. ej., **`glorotNormal`** en vez de `glorot_normal`
- Pérdida y métricas; p. ej., **`meanSquaredError`** en vez de `mean_squared_error`, **`categoricalCrossentropy`** en vez de `categorical_crossentropy`.

Como en el ejemplo anterior:

```js
// JavaScript:
model.compile({optimizer: 'sgd', loss: 'meanSquaredError'});
```

Con respecto a la serialización y la deserialización, quédese tranquilo. El mecanismo interno de TensorFlow.js garantiza que las escrituras todo en minúscula (<em>snake_case</em>) en objetos JSON se administran correctamente, p. ej., cuando se cargan modelos previamente entrenados de Keras de Python.

## Ejecución de objetos Layer con apply(), no llamándolos como funciones

En Keras, un objeto Layer tiene el método `__call__` definido. Por lo tanto, el usuario puede invocar la lógica de capa llamando al objeto como una función, p. ej.:

```python
# Python:
my_input = keras.Input(shape=[2, 4])
flatten = keras.layers.Flatten()

print(flatten(my_input).shape)
```

Esta sintaxis <em>sugar</em> de Python se implementa como método <em>apply()</em> en TensorFlow.js:

```js
// JavaScript:
const myInput = tf.input({shape: [2, 4]});
const flatten = tf.layers.flatten();

console.log(flatten.apply(myInput).shape);
```

## Layer.apply() es compatible con la evaluación imperativa (<em>eager</em>) de tensores concretos

Actualmente, en Keras, el método de **llamada** solamente se puede usar en objetos `tf.Tensor` de TensorFlow (Python) (siempre que el backend sea en TensorFlow), que son simbólicos y no contienen valores numéricos. Es lo que se muestra en el ejemplo de la sección anterior. Sin embargo, en TensorFlow.js, el método apply() de capas puede funcionar tanto en modo simbólico como imperativo. Si se invoca `apply()` con un SymbolicTensor (una analogía de tf.Tensor), el valor de retorno será un SymbolicTensor. Es lo que sucede normalmente durante la construcción del modelo. Pero si se invoca `apply()` con un valor de tensor concreto real, devolverá un tensor concreto. Por ejemplo:

```js
// JavaScript:
const flatten = tf.layers.flatten();

flatten.apply(tf.ones([2, 3, 4])).print();
```

Esta característica es una reminiscencia de la [ejecución <em>eager</em>](https://www.tensorflow.org/guide/eager) de TensorFlow (Python). Permite mayor interactividad y posibilidades de depuración durante el desarrollo del modelo, además de abrir puertas para la composición dinámica de redes neuronales.

## Los optimizadores se encuentran en train.*, no en optimizers.*

En Keras, los objetos optimizadores se encuentran en el espacio de nombres `keras.optimizers.*`. En TensorFlow.js Layers, los métodos de factoría para los optimizadores se encuentran en el espacio de nombres `tf.train.*`. Por ejemplo:

```python
# Python:
my_sgd = keras.optimizers.sgd(lr=0.2)
```

```js
// JavaScript:
const mySGD = tf.train.sgd({lr: 0.2});
```

## loadLayersModel() carga desde una URL, no desde un archivo HDF5

En Keras, los modelos, por lo general, se [guardan](https://keras.io/getting_started/faq/#what-are-my-options-for-saving-models) como un archivo HDF5 (.h5), que después se puede cargar con el método `keras.models.load_model()`. Este método sigue una ruta hacia el archivo .h5. La contraparte de `load_model()` en TensorFlow.js es [`tf.loadLayersModel()`](https://js.tensorflow.org/api/latest/#loadLayersModel). Como HDF5 no es un formato de archivo adecuado para navegadores, `tf.loadLayersModel()` toma un formato específico de TensorFlow.js. `tf.loadLayersModel()` toma un archivo model.json como argumento de entrada. El model.json se puede convertir a partir de un archivo HDF5 Keras usando el paquete pip de tensorflowjs.

```js
// JavaScript:
const model = await tf.loadLayersModel('https://foo.bar/model.json');
```

También, cabe destacar que `tf.loadLayersModel()` devuelve una [`Promise`](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Promise) de [`tf.Model`](https://js.tensorflow.org/api/latest/#class:Model).

En general, el guardado y la carga de los `tf.Model` en TensorFlow.js se hace con los métodos `tf.Model.save` y `tf.loadLayersModel`, respectivamente. Diseñamos estas API para que sean similares a la [API de save y load_model](https://keras.io/getting_started/faq/#what-are-my-options-for-saving-models) de Keras. Pero el entorno del navegador es bastante diferente al entorno de backend en el que se ejecutan los marcos de aprendizaje profundo fundamentales como Keras, particularmente en el arreglo de rutas para continuar y transmitir los datos. De allí surgen algunas diferencias interesantes entre las API para guardar o cargar en TensorFlow.js y en Keras. Para más detalles, consulte nuestro tutorial sobre [Guardar y cargar tf.Model](./save_load.md).

## Uso de `fitDataset()` para entrenar modelos con objetos `tf.data.Dataset`

En tf.keras de TensorFlow en Python, un modelo se puede entrenar con un objeto [Conjunto de datos](https://www.tensorflow.org/guide/datasets). El método `fit()` del modelo acepta, directamente, tales objetos. Un modelo TensorFlow.js también se puede entrenar con el equivalente de JavaScript en los objetos del conjunto de datos (consulte [la documentación de la API tf.data en TensorFlow.js](https://js.tensorflow.org/api/latest/#Data)). Sin embargo, a diferencia de lo que sucede en Python, el entrenamiento basado en conjuntos de datos se hace con un método exclusivo, concretamente con [fitDataset](https://js.tensorflow.org/api/0.15.1/#tf.Model.fitDataset). El método [fit()](https://js.tensorflow.org/api/latest/#tf.Model.fitDataset) solamente se utiliza para el entrenamiento de modelos basados en tensores.

## Gestión de memorias de objetos Layer y Model

TensorFlow.js se ejecuta en WebGL en el navegador, donde los pesos de los objetos Layer y Model están respaldados por las texturas de WebGL. Sin embargo, WebGL no tiene ningún soporte integrado para recolección de basura. Los objetos Layer y Model gestionan internamente la memoria del tensor para usarla durante las llamadas de inferencia y entrenamiento. Pero también permiten que el usuario las descarte (<em>dispose</em>) para liberar la memoria de WebGL que ocupan. Esto resulta particularmente útil cuando muchas de las instancias del modelo se crean y liberan dentro de una sola carga de página. Para eliminar un objeto Layer o Model, use el método `dispose()`.
