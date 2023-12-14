# Entrenamiento de modelos

Esta guía contiene información basada en el supuesto de que ya se ha leído la guía de [modelos y capas](models_and_layers.md).

En TensorFlow.js hay dos formas de entrenar un modelo de aprendizaje automático:

1. utilizando la API Layers con <code>&lt;a href="https://js.tensorflow.org/api/latest/#tf.Model.fit" data-md-type="link"&gt;LayersModel.fit()&lt;/a&gt;</code> o <code>&lt;a href="https://js.tensorflow.org/api/latest/#tf.Model.fitDataset" data-md-type="link"&gt;LayersModel.fitDataset()&lt;/a&gt;</code>.
2. utilizando la API Core con <code>&lt;a href="https://js.tensorflow.org/api/latest/#tf.train.Optimizer.minimize" data-md-type="link"&gt;Optimizer.minimize()&lt;/a&gt;</code>.

Primero, observaremos la API de capas, que es una API de un nivel superior diseñada para crear y entrenar modelos. Luego, mostraremos cómo entrenar el mismo modelo con la API Core.

## Introducción

Un *modelo* de aprendizaje automático es una función con parámetros que permiten el aprendizaje que mapea una entrada con una salida deseada. Los parámetros óptimos se obtienen mediante el entrenamiento del modelo con los datos.

El entrenamiento incluye varios pasos:

- Obtener un [lote](https://developers.google.com/machine-learning/glossary/#batch) de datos para el modelo.
- Pedir al modelo que haga las predicciones.
- Comparar esa predicción con el valor "verdadero".
- Decidir cuánto hay que cambiar cada parámetro para que el modelo pueda hacer una mejor predicción en el futuro para ese lote.

Un modelo bien entrenado proporciona con exactitud el mapeo desde la entrada a la salida deseada.

## Parámetros del modelo

Definamos un modelo simple de 2 capas con la API Layers:

```js
const model = tf.sequential({
 layers: [
   tf.layers.dense({inputShape: [784], units: 32, activation: 'relu'}),
   tf.layers.dense({units: 10, activation: 'softmax'}),
 ]
});
```

Bajo la superficie, los modelos tienen parámetros (con frecuencia, denominados *pesos*) que pueden aprender entrenando los datos. Imprimamos los nombres de los pesos asociados con este modelo y sus formas:

```js
model.weights.forEach(w => {
 console.log(w.name, w.shape);
});
```

Obtenemos la siguiente salida:

```
> dense_Dense1/kernel [784, 32]
> dense_Dense1/bias [32]
> dense_Dense2/kernel [32, 10]
> dense_Dense2/bias [10]
```

Hay 4 pesos en total, 2 por cada capa densa. Es lo que se espera, ya que las capas densas representan una función que mapea el tensor de entrada `x` con un tensor de salida `y` mediante la ecuación `y = Ax + b` donde `A` (el núcleo) y `b` (el sesgo) son parámetros de la capa densa.

> NOTA: Por defecto, las capas densas incluyen un sesgo, pero se lo puede excluir mediante la especificación de `{useBias: false}` en las opciones, al momento de crearlas.

`model.summary()` es un método útil si lo que se desea es obtener un panorama general del modelo y observar la cantidad total de parámetros:

<table>
  <tr>
   <td>Capa (tipo)</td>
   <td>Forma de la salida</td>
   <td>Parám. nro.</td>
  </tr>
  <tr>
   <td>dense_Dense1 (densa)</td>
   <td>[null,32]</td>
   <td>25120</td>
  </tr>
  <tr>
   <td>dense_Dense2 (densa)</td>
   <td>[null,10]</td>
   <td>330</td>
  </tr>
  <tr>
   <td colspan="3">Parámetros totales: 25450<br>Parámetros entrenables: 25450<br>Parámetros no entrenables: 0</td>
  </tr>
</table>

Cada peso del modelo tiene un backend de un objeto <code>&lt;a href="https://js.tensorflow.org/api/0.14.2/#class:Variable" data-md-type="link"&gt;variable&lt;/a&gt;</code>. En TensorFlow.js, una <code>variable</code> es un <code>Tensor</code> de punto flotante con un método <code>assign()</code> extra usado para actualizar los valores. La API Layers inicializa automáticamente los pesos con las mejores prácticas. Con fines demostrativos, podríamos sobrescribir los pesos llamando a <code>assign()</code> en las variables subyacentes:

```js
model.weights.forEach(w => {
  const newVals = tf.randomNormal(w.shape);
  // w.val is an instance of tf.Variable
  w.val.assign(newVals);
});
```

## Optimizador, pérdida y métrica

Antes de hacer cualquier entrenamiento, debe decidir sobre tres cosas:

1. **Un optimizador**. El trabajo del optimizador es decidir cuánto conviene cambiar cada parámetro del modelo, dada la predicción del modelo actual. Cuando se usa la API Layers, puede proporcionarse tanto un identificador de <em>string</em> de un optimizador existente (como el `'sgd'` o el `'adam'`), como dar una instancia de la clase <code>&lt;a href="https://js.tensorflow.org/api/latest/#Training-Optimizers" data-md-type="link"&gt;optimizador&lt;/a&gt;</code>.
2. <strong>Una función de pérdida</strong>. Es un objetivo que el modelo intentará minimizar. Su meta es aportar un número solo para indicar "cuán errónea" ha resultado la predicción del modelo. La pérdida se calcula para cada lote de datos, de modo que el modelo pueda actualizar sus pesos. Cuando se usa la API Layers, es posible proporcionar tanto un identificador de <em>string</em> de una función de pérdida existente (como <code>'categoricalCrossentropy'</code>) como cualquier función que tome un valor verdadero y uno predicho, y devuelva una pérdida. Consulte una [lista de pérdidas disponibles](https://js.tensorflow.org/api/latest/#Training-Losses) en nuestros documentos de la API.
3. <strong>Lista de métricas.</strong> En forma similar a lo que sucede con las pérdidas, con las métricas también se calcula un número solo, que resume cuán bien está funcionando el modelo. Las métricas, por lo general, se calculan en base a todos los datos, al final de cada época. Como mínimo, deberíamos controlar que nuestra pérdida disminuya cada vez más. Sin embargo, pronto, convendrá aplicar una métrica más adecuada para los humanos, como la exactitud. Cuando se usa la API Layers, es posible proporcionar tanto un identificador de <em>string</em> de una métrica existente (como <code>'accuracy'</code>), como cualquier función que tome un valor verdadero y uno predicho, y devuelva un puntaje. Consulte la [lista de métricas disponibles](https://js.tensorflow.org/api/latest/#Metrics) en nuestros documentos de la API.

Cuando lo haya decidido, compile un <code>LayersModel</code> llamando a <code>model.compile()</code> con las opciones provistas:

```js
model.compile({
  optimizer: 'sgd',
  loss: 'categoricalCrossentropy',
  metrics: ['accuracy']
});
```

Durante la compilación, el modelo hará algunas validaciones para garantizar que las opciones elegidas son compatibles entre sí.

## Entrenamiento

Hay dos maneras de entrenar un `LayersModel`:

- Con `model.fit()` y proporcionando los datos como un tensor grande de datos.
- Con el `model.fitDataset()` y proporcionando los datos a través de un objeto `Dataset`.

### model.fit()

Si el conjunto de datos cabe en la memoria principal y se encuentra en la forma de un tensor solo, es posible entrenar a un modelo llamando al método `fit()`:

```js
// Generate dummy data.
const data = tf.randomNormal([100, 784]);
const labels = tf.randomUniform([100, 10]);

function onBatchEnd(batch, logs) {
  console.log('Accuracy', logs.acc);
}

// Train for 5 epochs with batch size of 32.
model.fit(data, labels, {
   epochs: 5,
   batchSize: 32,
   callbacks: {onBatchEnd}
 }).then(info => {
   console.log('Final accuracy', info.history.acc);
 });
```

Por detrás, `model.fit()` puede hacer mucho por nosotros:

- Separa los datos y los reúne en un conjunto de validación y entrenamiento. Después, usa el conjunto de validación para medir el progreso durante el entrenamiento.
- Aleatoriza los datos, pero recién después de la separación. Para mayor seguridad, debería aleatorizar previamente los datos, antes de pasarlos a `fit()`.
- Separa el tensor de datos grande en otros más pequeños, del tamaño<br>`batchSize.`
- Llama a `optimizer.minimize()` mientras calcula la pérdida del modelo con respecto a los datos del lote.
- Puede enviar notificaciones que usted recibirá al principio y al final de cada época o lote. En nuestro caso, recibimos notificaciones al final de cada lote usando la opción `callbacks.onBatchEnd `. Otras opciones son: `onTrainBegin`, `onTrainEnd`, `onEpochBegin`, `onEpochEnd` y `onBatchBegin`.
- Da curso al hilo principal para garantizar que las tareas en la cola, del ciclo de eventos de JS, se puedan administrar oportunamente.

Para más información, consulte la [documentación](https://js.tensorflow.org/api/latest/#tf.Sequential.fit) sobre `fit()`. Tenga en cuenta que, si elige la API Core, deberá implementar la lógica por su propia cuenta.

### model.fitDataset()

Si los datos no entran por completo en la memoria, o si se están transmitiendo, se puede entrenar un modelo llamándolo con `fitDataset()`, que toma un objeto `Dataset`. A continuación, se encuentra el mismo código de entrenamiento, pero con un conjunto de datos que encapsula a una función generadora:

```js
function* data() {
 for (let i = 0; i < 100; i++) {
   // Generate one sample at a time.
   yield tf.randomNormal([784]);
 }
}

function* labels() {
 for (let i = 0; i < 100; i++) {
   // Generate one sample at a time.
   yield tf.randomUniform([10]);
 }
}

const xs = tf.data.generator(data);
const ys = tf.data.generator(labels);
// We zip the data and labels together, shuffle and batch 32 samples at a time.
const ds = tf.data.zip({xs, ys}).shuffle(100 /* bufferSize */).batch(32);

// Train the model for 5 epochs.
model.fitDataset(ds, {epochs: 5}).then(info => {
 console.log('Accuracy', info.history.acc);
});
```

Para más información sobre los conjuntos de datos, consulte la [documentación](https://js.tensorflow.org/api/latest/#tf.Model.fitDataset) de `model.fitDataset()`.

## Predicción de datos nuevos

Una vez que el modelo ha sido entrenado, podrá llamar a `model.predict()` para hacer predicciones de datos aún no vistos:

```js
// Predict 3 random samples.
const prediction = model.predict(tf.randomNormal([3, 784]));
prediction.print();
```

Nota: Tal como ya lo mencionamos en la guía sobre [modelos y capas](models_and_layers), `LayersModel` espera que la dimensión más superficial de la entrada sea el tamaño del lote. En el ejemplo anterior, el tamaño del lote es 3.

## API Core

Ya mencionamos que hay dos formas de entrenar a un modelo de aprendizaje automático en TensorFlow.js.

La regla general indica que, primero, hay que intentar usar la API Layers, ya que está modelada luego de que la API Keras fuera ampliamente adoptada. La API Layers también ofrece varias soluciones estándares como la inicialización de peso, la serialización de modelos, el entrenamiento con monitoreo, la portabilidad y las comprobaciones de seguridad.

Probablemente le convenga usar la API Core cuando sucede lo siguiente:

- Necesita flexibilidad o control máximos.
- Y no necesita la serialización o puede implementar su propia lógica (de serialización).

Para más información sobre esta API, lea la sección sobre la "API Core" en la guía sobre [modelos y capas](models_and_layers.md).

El mismo modelo anterior escrito con la API Core se ve de la siguiente manera:

```js
// The weights and biases for the two dense layers.
const w1 = tf.variable(tf.randomNormal([784, 32]));
const b1 = tf.variable(tf.randomNormal([32]));
const w2 = tf.variable(tf.randomNormal([32, 10]));
const b2 = tf.variable(tf.randomNormal([10]));

function model(x) {
  return x.matMul(w1).add(b1).relu().matMul(w2).add(b2);
}
```

Además de la API Layers, la API Data también funciona perfectamente con la API Core. Reutilicemos el conjunto datos que definimos antes en la sección sobre [model.fitDataset()](#model.fitDataset()), que hace por nosotros la aleatorización y la agrupación en lotes:

```js
const xs = tf.data.generator(data);
const ys = tf.data.generator(labels);
// Zip the data and labels together, shuffle and batch 32 samples at a time.
const ds = tf.data.zip({xs, ys}).shuffle(100 /* bufferSize */).batch(32);
```

Entrenemos el modelo:

```js
const optimizer = tf.train.sgd(0.1 /* learningRate */);
// Train for 5 epochs.
for (let epoch = 0; epoch < 5; epoch++) {
  await ds.forEachAsync(({xs, ys}) => {
    optimizer.minimize(() => {
      const predYs = model(xs);
      const loss = tf.losses.softmaxCrossEntropy(ys, predYs);
      loss.data().then(l => console.log('Loss', l));
      return loss;
    });
  });
  console.log('Epoch', epoch);
}
```

Este código es una receta estándar para entrenar un modelo con la API Core:

- Itera sobre la cantidad de épocas.
- Dentro de cada época, itere los lotes de datos. Cuando usa un `Dataset`, le convendrá utilizar <code>&lt;a href="https://js.tensorflow.org/api/0.15.1/#tf.data.Dataset.forEachAsync" data-md-type="link"&gt;dataset.forEachAsync()&lt;/a&gt; </code> para iterar los lotes.
- Para cada lote, llame a <code>&lt;a href="https://js.tensorflow.org/api/latest/#tf.train.Optimizer.minimize" data-md-type="link"&gt;optimizer.minimize(f)&lt;/a&gt;</code>, que ejecuta <code>f</code> y minimiza su salida mediante el cálculo de gradientes con respecto a las cuatro variables definidas anteriormente.
- <code>f</code> calcula la pérdida. Llama a una de las funciones de pérdida previamente definidas utilizando la predicción del modelo y el valor verdadero.
