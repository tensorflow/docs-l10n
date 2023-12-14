# Modelos y capas

En el aprendizaje automático, un *modelo* es una función con [parámetros](https://developers.google.com/machine-learning/glossary/#parameter) *capaces de aprender* que mapea a una entrada con una salida. Los parámetros óptimos se obtienen entrenando al modelo con datos. Un modelo bien entrenado brindará un mapeo exacto desde la entrada hasta la salida deseada.

En TensorFlow.js hay dos maneras de crear un modelo de aprendizaje automático:

1. con la API Layers donde el modelo se crea usando *capas*.
2. con la API Core, con operaciones de bajo nivel como `tf.matMul()`, `tf.add()`, etc.

Primero, observaremos la API Layers, que es una API de nivel más alto diseñada para crear modelos. Luego, mostraremos cómo crear el mismo modelo con la API Core.

## Creación de modelos con la API Layers

Con la API Layers se pueden crear dos tipos de modelo: uno *secuencial* y otro *funcional*. En las siguientes dos secciones observaremos cada uno de estos tipos más de cerca.

### El modelo secuencial

El tipo más común de modelo es el <code>&lt;a href="https://js.tensorflow.org/api/0.15.1/#class:Sequential" data-md-type="link"&gt;Sequential&lt;/a&gt;</code>, que es una pila lineal de capas. Se puede crear un modelo <code>Sequential</code> pasando una lista de capas a la función <code>&lt;a href="https://js.tensorflow.org/api/0.15.1/#sequential" data-md-type="link"&gt;sequential()&lt;/a&gt;</code>:

```js
const model = tf.sequential({
 layers: [
   tf.layers.dense({inputShape: [784], units: 32, activation: 'relu'}),
   tf.layers.dense({units: 10, activation: 'softmax'}),
 ]
});
```

O mediante el método `add()`:

```js
const model = tf.sequential();
model.add(tf.layers.dense({inputShape: [784], units: 32, activation: 'relu'}));
model.add(tf.layers.dense({units: 10, activation: 'softmax'}));
```

> IMPORTANTE: La primera capa del modelo necesita una `inputShape`. Verifique haber excluido el tamaño del lote al aportar la `inputShape`. Por ejemplo, si planea alimentar a los tensores del modelo con forma `[B, 784]`, donde `B` puede ser cualquier tamaño de lote, especifique `inputShape` como `[784]` al crear el modelo.

Puede acceder a las capas del modelo a través de `model.layers` y, más específicamente, mediante `model.inputLayers` y `model.outputLayers`.

### El modelo funcional

Otra manera de crear un `LayersModel` es utilizando la función `tf.model()`. La diferencia entre `tf.model()` y `tf.sequential()` es que `tf.model()` permite crear un grafo arbitrario de capas, siempre y cuando no tengan ciclos.

A continuación, compartimos un fragmento de código en el que se define el mismo modelo que mostramos arriba, pero con la API `tf.model()`:

```js
// Create an arbitrary graph of layers, by connecting them
// via the apply() method.
const input = tf.input({shape: [784]});
const dense1 = tf.layers.dense({units: 32, activation: 'relu'}).apply(input);
const dense2 = tf.layers.dense({units: 10, activation: 'softmax'}).apply(dense1);
const model = tf.model({inputs: input, outputs: dense2});
```

Llamamos a `apply()` en cada una de las capas para conectarlas con la salida de otra capa. El resultado de `apply()` en este caso es un `SymbolicTensor`, que actúa como `Tensor` pero sin ningún valor concreto.

Tenga en cuenta que a diferencia de lo que sucede con el modelo secuencial, creamos un `SymbolicTensor` con `tf.input()`, en vez de proporcionar una `inputShape` a la primera capa.

Si a `apply()` se le pasa un `Tensor` concreto, también puede aportar un `Tensor` concreto:

```js
const t = tf.tensor([-2, 1, 0, 5]);
const o = tf.layers.activation({activation: 'relu'}).apply(t);
o.print(); // [0, 1, 0, 5]
```

Esta acción puede resultar útil para probar capas aisladas y ver su salida.

Al igual que en un modelo secuencial, se puede acceder a las capas del modelo a través de `model.layers` y, más específicamente, mediante `model.inputLayers` y `model.outputLayers`.

## Validación

Tanto el modelo secuencial como el funcional son instancias de la clase `LayersModel`. Uno de los mayores beneficios de trabajar con un `LayersModel` es la validación: lo fuerza a uno a especificar la forma de la entrada, que usará más adelante para validarla. El `LayersModel` también hace una inferencia automática de la forma a medida que los datos fluyen entre las capas. Saber la forma con anticipación sirve para que el modelo cree automáticamente sus parámetros y pueda decir si dos capas consecutivas no son compatibles entre sí.

## Resumen del modelo

Llamamos a `model.summary()` para imprimir un resumen útil del modelo, que incluye lo siguiente:

- El nombre y el tipo de todas las capas del modelo.
- La forma de la salida de cada capa.
- La cantidad de parámetros de peso de cada capa.
- Las entradas que recibe cada capa, si el modelo tiene una topología general (analizado más adelante en este artículo).
- La cantidad total de parámetros entrenables y no entrenables del modelo.

Para el modelo definido arriba, obtenemos la siguiente salida en la consola:

<table>
  <tr>
   <td>Capa (tipo)</td>
   <td>Forma de la salida</td>
   <td>Parám. nro.</td>
  </tr>
  <tr>
   <td>dense_Dense1 (Dense)</td>
   <td>[null,32]</td>
   <td>25120</td>
  </tr>
  <tr>
   <td>dense_Dense2 (Dense)</td>
   <td>[null,10]</td>
   <td>330</td>
  </tr>
  <tr>
   <td colspan="3">Parámetros totales: 25450<br>Parámetros entrenables: 25450<br> Parámetros no entrenables: 0</td>
  </tr>
</table>

Preste atención a los valores `null` en las formas de las salidas de las capas: un recordatorio de que el modelo espera que la entrada tenga un tamaño de lote como la dimensión <br>más externa, que en este caso puede ser flexible debido al valor `null`.

## Serialización

Uno de los principales beneficios de usar un `LayersModel` y no una API de bajo nivel es la posibilidad que ofrece de guardar y cargar un modelo. Un `LayersModel` sabe lo siguiente:

- la arquitectura del modelo, que permite recrear el modelo;
- los pesos del modelo;
- la configuración de entrenamiento (pérdida, optimizador, métricas);
- el estado del optimizador, que permite reanudar el entrenamiento.

Para guardar o cargar un modelo se usa una sola línea de código:

```js
const saveResult = await model.save('localstorage://my-model-1');
const model = await tf.loadLayersModel('localstorage://my-model-1');
```

En el ejemplo anterior, el modelo se guarda en el almacenamiento local del navegador. Consulte la <code>&lt;a href="https://js.tensorflow.org/api/latest/#tf.Model.save" data-md-type="link"&gt;documentación sobre model.save()&lt;/a&gt;</code> y la guía sobre cómo [guardar y cargar](save_load.md), para guardar con diferentes medios (p. ej., con almacenamiento de archivos, <code>IndexedDB</code>, al dispararse una descarga del navegador, etc.).

## Capas personalizadas

Las capas son los ladrillos de un modelo. Si el modelo hace un cálculo personalizado, podrá definir una capa también personalizada que interactúa bien con el resto de las capas. A continuación definimos una capa personalizada que calcula la suma de cuadrados:

```js
class SquaredSumLayer extends tf.layers.Layer {
 constructor() {
   super({});
 }
 // In this case, the output is a scalar.
 computeOutputShape(inputShape) { return []; }

 // call() is where we do the computation.
 call(input, kwargs) { return input.square().sum();}

 // Every layer needs a unique name.
 getClassName() { return 'SquaredSum'; }
}
```

Para probarla, podemos llamar al método `apply()` con un tensor concreto:

```js
const t = tf.tensor([-2, 1, 0, 5]);
const o = new SquaredSumLayer().apply(t);
o.print(); // prints 30
```

> IMPORTANTE: Si se carga una capa personalizada, se pierde la posibilidad de serializar un modelo.

## Creación de modelos con la API Core

Ya mencionamos, al principio de esta guía, que hay dos formas de entrenar a un modelo de aprendizaje automático en TensorFlow.js.

La regla general indica que siempre hay que intentar usar la API Layers primero, ya que está modelada como la reconocida API Keras que sigue las [mejores prácticas y reduce la carga cognitiva](https://keras.io/why-use-keras/). La API Layers también ofrece varias soluciones estándares como la inicialización de peso, la serialización de modelos, el entrenamiento con monitoreo, la portabilidad y las comprobaciones de seguridad.

Probablemente, le convenga usar la API Core cuando sucede lo siguiente:

- Necesita flexibilidad o control máximos.
- No necesita la serialización o puede implementar su propia lógica de serialización.

Los modelos de la API Core son solamente funciones que toman uno o más `Tensors` y devuelven un `Tensor`. El mismo modelo escrito arriba, pero con la API Core se ve de la siguiente manera:

```js
// The weights and biases for the two dense layers.
const w1 = tf.variable(tf.randomNormal([784, 32]));
const b1 = tf.variable(tf.randomNormal([32]));
const w2 = tf.variable(tf.randomNormal([32, 10]));
const b2 = tf.variable(tf.randomNormal([10]));

function model(x) {
  return x.matMul(w1).add(b1).relu().matMul(w2).add(b2).softmax();
}
```

Tenga en cuenta que en la API Core somos responsables de crear e inicializar pesos del modelo. Cada peso está respaldado por una `Variable ` que señala a TensorFlow.js que esos tensores son capaces de aprender. Se puede crear una `Variable` con [tf.variable()](https://js.tensorflow.org/api/latest/#variable) y pasando un `Tensor` que ya exista.

Al leer esta guía se habrá familiarizado con las diferentes formas de crear un modelo con las API Layers y Core. Ahora, lea la guía sobre [entrenamiento de modelos](train_models.md) para entender cómo entrenar un modelo.
