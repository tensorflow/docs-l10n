# Cambio a TensorFlow.js 3.0

## Qué cambia en TensorFlow.js 3.0

Las notas del lanzamiento se encuentran [disponibles aquí](https://github.com/tensorflow/tfjs/releases). Algunas de las características más destacables que notará el usuario son las siguientes:

### Módulos personalizados

Ofrecemos ayuda para crear módulos tfjs personalizados para lograr producir agrupamientos para navegadores con tamaño optimizado. Se envían menos JavaScript a los usuarios. Para más información sobre este tema, [consulte este tutorial](https://github.com/tensorflow/tfjs-website/blob/master/docs/tutorials/deployment/size_optimized_bundles.md).

Esta característica está prevista para ser implementada en el navegador, sin embargo, su activación provoca algunos de los cambios descriptos a continuación.

### Código ES2017

Además de algunas agrupaciones previamente compiladas, **la opción principal que usamos ahora para enviar nuestro código a NPM es [módulos ES](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Modules) con [sintaxis ES2017](https://2ality.com/2016/02/ecmascript-2017.html)**. Esto les brinda a los desarrolladores la oportunidad de aprovechar las [características del JavaScript moderno](https://web.dev/publish-modern-javascript/) y tener un mayor control sobre qué envían a sus usuarios finales.

Nuestra entrada del `module` de package.json apunta a archivos individuales de la biblioteca en formato ES2017 (es decir, no agrupados). Es lo que permite "agitar el árbol" y un mayor control del desarrollador sobre la transpilación downstream.

Proporcionamos algunos formatos alternativos como agrupamientos previamente compilados, para mantener la compatibilidad con navegadores anteriores y otros sistemas de módulos. Estos formatos siguen las convenciones de normalización de nombres descriptas en la tabla a continuación y puede cargarlas desde alguna de las CDN populares como JsDelivr o Unpkg.

<table>
  <tr>
   <td>Nombre de archivo</td>
   <td>Formato de módulo</td>
   <td>Versión de idioma</td>
  </tr>
  <tr>
   <td>tf[-package].[min].js*</td>
   <td>UMD</td>
   <td>ES5</td>
  </tr>
  <tr>
   <td>tf[-package].es2017.[min].js</td>
   <td>UMD</td>
   <td>ES2017</td>
  </tr>
  <tr>
   <td>tf[-package].node.js**</td>
   <td>CommonJS</td>
   <td>ES5</td>
  </tr>
  <tr>
   <td>tf[-package].es2017.fesm.[min].js</td>
   <td>ESM (archivo plano único)</td>
   <td>ES2017</td>
  </tr>
  <tr>
   <td>index.js***</td>
   <td>ESM</td>
   <td>ES2017</td>
  </tr>
</table>

* [package] se refiere a los nombres como core (central)/conversor/capas para subpaquetes del paquete principal tf.js. [min] describe dónde proporcionamos los archivos minificados además de los no minificados.

** Nuestra entrada `main` de package.json apunta a este archivo.

*** Nuestra entrada `module` de package.json apunta a este archivo.

Si usa tensorflow.js a través de npm y, a la vez, usa un agrupador, probablemente necesite ajustar la configuración del agrupador para asegurarse de que consuma los módulos ES2017 o apunte a otra de las entradas en package.json.

### @tensorflow/tfjs-core es más delgado por defecto

Para permitir [agitar el árbol](https://developers.google.com/web/fundamentals/performance/optimizing-javascript/tree-shaking) mejor, ya no incluimos la API de encadenamiento/fluida en tensores por defecto en @tensorflow/tfjs-core. Recomendamos usar directamente las operaciones (ops) para obtener el agrupamiento más pequeño posible. Proporcionamos un `import '@tensorflow/tfjs-core/dist/public/chained_ops/register_all_chained_ops';` que restablece la API de encadenamiento.

Ya no registramos más los gradientes para los núcleos (<em>kernel</em>), por defecto. Si quiere contar con compatibilidad con gradientes/entrenamiento puede usar `import '@tensorflow/tfjs-core/dist/register_all_gradients';`

> Nota: si está usando @tensorflow/tfjs, @tensorflow/tfjs-layers o cualquier otro paquete de alto nivel, lo anterior se logra automáticamente.

### Registros de reorganización de código, núcleo y gradientes

Hemos reorganizado nuestro código para facilitar tanto la contribución de las operaciones y los núcleos, como la implementación de las operaciones, los núcleos y los gradientes personalizados. [Para más información, consulte esta guía](https://www.tensorflow.org/js/guide/custom_ops_kernels_gradients).

### Cambios importantes

[Aquí](https://github.com/tensorflow/tfjs/releases) encontrará una lista completa de cambios importantes (<em>breaking changes</em>). Tenga en cuenta que incluye la eliminación de todas las operaciones *Strict como mulStrict o addStrict.

## Actualización del código desde 2.x

### Usuarios de @tensorflow/tfjs

Para cualquier cambio importante de esta lista: https://github.com/tensorflow/tfjs/releases

### Usuarios de @tensorflow/tfjs-core

Con cualquier cambio importante de esta lista (https://github.com/tensorflow/tfjs/releases), haga lo siguiente:

#### Agregue aumentadores de operaciones encadenadas o use directamente las operaciones

En vez de

```
import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-webgl';

const a = tf.tensor([1,2,3,4]);
const b = a.sum(); // this is a 'chained' op.
```

Deberá

```
import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-webgl';
import '@tensorflow/tfjs-core/dist/public/chained_ops/sum'; // add the 'sum' chained op to all tensors

const a = tf.tensor([1,2,3,4]);
const b = a.sum();
```

También puede importar todas las API de encadenamiento/fluidas con lo siguiente:

```
import '@tensorflow/tfjs-core/dist/public/chained_ops/register_all_chained_ops';
```

Como alternativa puede usar directamente la operación (aquí también podría usar las importaciones con los nombres predefinidos)

```
import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-webgl';

const a = tf.tensor([1,2,3,4]);
const b = tf.sum(a);
```

#### Importe el código de inicialización

Si utiliza exclusivamente importaciones con nombre predefinido (en vez de `import * as ...`), entonces, en algunos casos puede tener que hacer lo siguiente:

```
import @tensorflow/tfjs-core
```

cerca de la parte superior de su programa, esto evita que los "agitadores de árboles" agresivos hagan caer cualquier inicialización necesaria.

## Actualización del código desde 1.x

### Usuarios de @tensorflow/tfjs

Tome cualquiera de los cambios importantes que figuran en [esta lista](https://github.com/tensorflow/tfjs/releases/tag/tfjs-v2.0.0). Luego, siga las instrucciones para actualizar a partir de 2x.

### Usuarios de @tensorflow/tfjs-core

Tome cualquiera de los cambios importantes que figuran en [esta lista](https://github.com/tensorflow/tfjs/releases/tag/tfjs-v2.0.0). Luego, seleccione un backend, tal como se describe a continuación. Finalmente, siga los pasos necesarios para actualizar a partir de 2x.

#### Selección de los backend

En TensorFlow.js 2.0 quitamos los backend cpu y webgl. Consulte [@tensorflow/tfjs-backend-cpu](https://www.npmjs.com/package/@tensorflow/tfjs-backend-cpu), [@tensorflow/tfjs-backend-webgl](https://www.npmjs.com/package/@tensorflow/tfjs-backend-webgl), [@tensorflow/tfjs-backend-wasm](https://www.npmjs.com/package/@tensorflow/tfjs-backend-wasm) y [@tensorflow/tfjs-backend-webgpu](https://www.npmjs.com/package/@tensorflow/tfjs-backend-webgpu) para acceder a instrucciones sobre cómo incluir estos backends.
