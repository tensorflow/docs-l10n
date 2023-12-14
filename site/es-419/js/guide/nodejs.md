# TensorFlow.js en Node

## CPU de TensorFlow

El paquete para CPU de TensorFlow se puede importar de la siguiente manera:

```js
import * as tf from '@tensorflow/tfjs-node'
```

Cuando importe TensorFlow.js de este paquete, el módulo que obtendrá se acelerará con TensorFlow para C binario y se ejecutará en la CPU. TensorFlow en la CPU utiliza la aceleración de hardware para acelerar el cálculo algebraico lineal subyacente.

Este paquete funciona con Linux, Windows y Mac, con las que TensorFlow es compatible.

> Nota: No es necesario importar '@tensorflow/tfjs' ni agregarlo a su package.json. Se importa indirectamente a través de la biblioteca nodo.

## GPU de TensorFlow

El paquete para GPU de TensorFlow se puede importar de la siguiente manera:

```js
import * as tf from '@tensorflow/tfjs-node-gpu'
```

Al igual que con el paquete para CPU, el módulo que obtenga se acelerará con TensorFlow para C binario. Sin embargo, las operaciones del tensor se ejecutarán en GPU con CUDA y, por lo tanto, solamente en Linux. Este enlace puede ser de, al menos, un orden de magnitud más rápido que el de otras opciones de enlazado.

> Nota: Por el momento, este paquete solamente funciona con CUDA. Antes de elegir esta opción deberá tener CUDA instalado en su máquina con una tarjeta gráfica NVIDIA.

> Nota: No es necesario importar '@tensorflow/tfjs' ni agregarlo a su package.json. Se importa indirectamente a través de la biblioteca nodo.

## CPU vainilla

La versión de TensorFlow.js que se ejecuta con las operaciones de la CPU vainilla se pueden importar de la siguiente manera:

```js
import * as tf from '@tensorflow/tfjs'
```

Este paquete es el mismo que usaría en el navegador. En él, las operaciones se ejecutan en JavaScript vainilla en la CPU. Es mucho más pequeño que otros porque no necesita utilizar TensorFlow binario y también, mucho más lento.

Como el paquete no depende de TensorFlow, se puede usar en más dispositivos compatibles con Node.js y no solamente con Linux, Windows y Mac.

## Consideraciones sobre producción

Los enlaces de Node.js ofrecen un backend para TensorFlow.js que implementa operaciones de manera sincrónica. Significa que cuando se llame a una operación, p. ej., `tf.matMul(a, b)`, se bloqueará el hilo principal hasta que la misma haya terminado.

Por este motivo, los enlaces actuales son adecuados para los scripts y las tareas offline. Si desea utilizar los enlaces de Node.js en una aplicación de producción, como en un webserver, deberá preparar una cola de trabajos o configurar los hilos de trabajadores para que el código de TensorFlow.js no bloquee al hilo principal.

## Las API

Una vez importado el paquete como tf con cualquiera de las opciones citadas arriba, en el módulo importado aparecerán todos los símbolos de TensorFlow.js normales.

### tf.browser

En el paquete de TensorFlow.js normal, los símbolos del espacio de nombres `tf.browser.*` no se podrán usar en Node.js, ya que usan API específicas de navegador.

Actualmente, son:

- tf.browser.fromPixels
- tf.browser.toPixels

### tf.node

Los dos paquetes de Node.js también ofrecen un espacio de nombres, `tf.node`, que contiene las API específicas para Node.

TensorBoard es un ejemplo destacable de API específica para Node.js.

El siguiente es un ejemplo sobre cómo importar resúmenes a TensorBoard en Node.js:

```js
const model = tf.sequential();
model.add(tf.layers.dense({ units: 1, inputShape: [200] }));
model.compile({
  loss: 'meanSquaredError',
  optimizer: 'sgd',
  metrics: ['MAE']
});


// Generate some random fake data for demo purpose.
const xs = tf.randomUniform([10000, 200]);
const ys = tf.randomUniform([10000, 1]);
const valXs = tf.randomUniform([1000, 200]);
const valYs = tf.randomUniform([1000, 1]);


// Start model training process.
async function train() {
  await model.fit(xs, ys, {
    epochs: 100,
    validationData: [valXs, valYs],
    // Add the tensorBoard callback here.
    callbacks: tf.node.tensorBoard('/tmp/fit_logs_1')
  });
}
train();
```
