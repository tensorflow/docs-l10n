# Guardar y cargar modelos

TensorFlow.js proporciona una funcionalidad para guardar y cargar modelos que han sido creados con la API [`Layers`](https://js.tensorflow.org/api/0.14.2/#Models) o que se han convertido a partir de modelos de TensorFlow. Pueden ser modelos que ha entrenado solo o que han entrenado otros. Un beneficio clave de utilizar la API Layers es que los modelos creados con ella son serializables y eso es lo que analizaremos en este tutorial.

Este tutorial se centrará en guardar y cargar los modelos TensorFlow.js (identificable por los archivos JSON). También podemos importar modelos de TensorFlow Python. La carga de estos modelos se trata en los siguientes dos tutoriales:

- [Cómo importar modelos Keras](../tutorials/conversion/import_keras.md)
- [Cómo importar modelos Graphdef](../tutorials/conversion/import_saved_model.md)

## Save a tf.Model

[`tf.Model`](https://js.tensorflow.org/api/0.14.2/#class:Model) y [`tf.Sequential`](https://js.tensorflow.org/api/0.14.2/#class:Model), ambos, proporcionan una función [`model.save`](https://js.tensorflow.org/api/0.14.2/#tf.Model.save) que le permitirá guardar la *topología* y los *pesos* de un modelo.

- Topología: es un archivo en el que se describe la arquitectura de un modelo (es decir, qué operaciones usa). Contiene referencias a los pesos del modelo que se almacenan externamente.

- Pesos: son archivos binarios en los que se almacenan los pesos de un modelo dado en un formato eficiente. Por lo general, se almacenan en la misma carpeta que la topología.

Echemos un vistazo a cómo luce el código para guardar un modelo

```js
const saveResult = await model.save('localstorage://my-model-1');
```

Algunos datos para tener en cuenta:

- El método `save` toma un argumento de string similar a una URL que empieza con un **esquema**. Describe el tipo de destino en el que se intenta guardar un modelo. En el ejemplo anterior, el esquema es `localstorage://`
- Al esquema lo sigue una **ruta**. En el ejemplo anterior la ruta es `my-model-1`.
- El método `save` es asincrónico.
- El valor de retorno de `model.save` es un objeto JSON, que contiene información como la de los tamaños en bytes de la topología y los pesos del modelo.
- El entorno usado para guardar el modelo no afecta a los entornos que puedan cargar el modelo. Que el modelo se guarde en node.js no impide que se pueda cargar en el navegador.

A continuación, examinaremos los diferentes esquemas disponibles.

### Almacenamiento local (solamente para navegadores)

**Esquema:** `localstorage://`

```js
await model.save('localstorage://my-model');
```

Esto guarda un modelo con el nombre `my-model` en el [almacenamiento local](https://developer.mozilla.org/en-US/docs/Web/API/Window/localStorage) del navegador (persistirá entre actualizaciones). Pero los usuarios o el mismo navegador pueden borrar el almacenamiento local en caso de que el espacio se transforme en un problema. Cada navegador también define su propio límite con respecto a cuántos datos, de un dominio dado, se pueden guardar en el almacenamiento local.

### IndexedDB (solamente para navegador)

**Esquema:** `indexeddb://`

```js
await model.save('indexeddb://my-model');
```

Esta opción guarda un modelo en el almacenamiento [IndexedDB](https://developer.mozilla.org/en-US/docs/Web/API/IndexedDB_API) del navegador. Al igual que el almacenamiento local, continúa entre una actualización y otra del navegador, y también tiende a tener límites más amplios con respecto al tamaño de los objetos almacenados.

### Descargas de archivos (solamente para navegador)

**Esquema:** `downloads://`

```js
await model.save('downloads://my-model');
```

Esto hará que el navegador descargue los archivos del modelo en la máquina del usuario. Se producirán dos archivos:

1. Un archivo de texto JSON denominado `[my-model].json`, que contiene la tipología y la referencia al archivo de pesos que se describe a continuación.
2. Un archivo binario que contiene los valores de los pesos, con el nombre `[my-model].weights.bin`.

Es posible cambiar el nombre `[my-model]` para obtener archivos con un nombre diferente.

Como el archivo `.json` apunta al `.bin` con una ruta relativa, ambos archivos deberían guardarse en la misma carpeta.

> NOTA: En algunos navegadores hace falta que los usuarios otorguen los permisos antes de poder descargar más de un archivo al mismo tiempo.

### Solicitud HTTP(S)

**Esquema:** `http://` o `https://`

```js
await model.save('http://model-server.domain/upload')
```

Esto creará una solicitud web para guardar un modelo en un servidor remoto. Debemos tener al servidor remoto bajo control, para poder garantizar que sea capaz de administrar la solicitud.

El modelo se enviará a un servidor HTTP específico mediante una solicitud [POST](https://developer.mozilla.org/en-US/docs/Web/HTTP/Methods/POST). El cuerpo de la POST está en formato `multipart/form-data` y contiene dos archivos

1. Un archivo de texto JSON denominado `model.json`, que contiene la tipología y la referencia al archivo de pesos que se describe a continuación.
2. Un archivo binario que contiene los valores de los pesos, con el nombre `model.weights.bin`.

Tenga en cuenta que los nombres de los dos archivos siempre serán exactamente como los especificados arriba (el nombre se integra a la función). Este [documento de API](https://js.tensorflow.org/api/latest/#tf.io.browserHTTPRequest) contiene un fragmento de código Python que demuestra cómo se puede usar el marco de trabajo web [Flask](http://flask.pocoo.org/) para gestionar la solicitud originada a partir de `save`.

En muchos casos, habrá que pasar más argumentos o encabezados de solicitudes al servidor HTTP (p. ej. para la autenticación o si deseara especificar una carpeta en la que se debería guardar el modelo). Se puede lograr un control más detallado de estos aspectos de las solicitudes de `save` reemplazando el argumento de la string de la URL en `tf.io.browserHTTPRequest`. Esta API permite una mayor flexibilidad para controlar las solicitudes HTTP.

Por ejemplo:

```js
await model.save(tf.io.browserHTTPRequest(
    'http://model-server.domain/upload',
    {method: 'PUT', headers: {'header_key_1': 'header_value_1'}}));
```

### Sistema de archivo nativo (solamente para Node.js)

**Esquema:** `file://`

```js
await model.save('file:///path/to/my-model');
```

Cuando le ejecución se hace en Node.js, también tenemos acceso al sistema de archivos y podemos guardar los modelos allí. Con el comando de arriba se guardarán dos archivos en la  `path` especificada después del `scheme`.

1. Un archivo de texto JSON denominado `[model].json`, que contiene la tipología y la referencia al archivo de pesos que se describe a continuación.
2. Un archivo binario que contiene los valores de los pesos, con el nombre `[model].weights.bin`.

Tenga en cuenta que los nombres de los dos archivos siempre serán exactamente como los especificados arriba ( el nombre se integra a la función).

## Carga de un tf.Model

Si tenemos un modelo que se guardó con alguno de los métodos descriptos arriba, podemos cargarlo con la API `tf.loadLayersModel`.

Echemos un vistazo a cómo luce el código para cargar un modelo

```js
const model = await tf.loadLayersModel('localstorage://my-model-1');
```

Algunos datos para tener en cuenta:

- Tal como `model.save()`, la función `loadLayersModel` toma un argumento de string similar a una URL que comienza con un **scheme**. Esto describe el tipo de destino del que pretendemos cargar un modelo.
- Al esquema lo sigue una **ruta**. En el ejemplo anterior la ruta es `my-model-1`.
- La string similar a una URL se puede reemplazar con un objeto que coincida con la interfaz IOHandler.
- La función `tf.loadLayersModel()` es asincrónica.
- El valor de retorno de `tf.loadLayersModel` es `tf.Model`

A continuación, examinaremos los diferentes esquemas disponibles.

### Almacenamiento local (solamente para navegadores)

**Esquema:** `localstorage://`

```js
const model = await tf.loadLayersModel('localstorage://my-model');
```

Esto carga un modelo denominado `my-model` del [almacenamiento local](https://developer.mozilla.org/en-US/docs/Web/API/Window/localStorage) del navegador.

### IndexedDB (solamente para navegador)

**Esquema:** `indexeddb://`

```js
const model = await tf.loadLayersModel('indexeddb://my-model');
```

Esto carga un modelo denominado del almacenamiento [IndexedDB](https://developer.mozilla.org/en-US/docs/Web/API/IndexedDB_API) del navegador.

### HTTP(S)

**Esquema:** `http://` o `https://`

```js
const model = await tf.loadLayersModel('http://model-server.domain/download/model.json');
```

Esto carga un modelo desde un endpoint http. Después de cargar el archivo `json`, la función hará las solicitudes de los archivos `.bin` correspondientes a los que referencia el archivo `json`.

> NOTA: Esta implementación depende de la presencia del método [`fetch`](https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API/Using_Fetch). Si uno está en un entorno que no proporciona el método <em>fetch</em> nativamente, se puede proporcionar un [`fetch`](https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API/Using_Fetch) de nombres (de método global) que satisfaga esa interfaz o que use una biblioteca como [`node-fetch`](https://www.npmjs.com/package/node-fetch).

### Sistema de archivo nativo (solamente para Node.js)

**Esquema:** `file://`

```js
const model = await tf.loadLayersModel('file://path/to/my-model/model.json');
```

Cuando la ejecución se hace en Node.js, también tenemos acceso al sistema de archivos y podemos cargar los modelos desde allí. Tenga en cuenta que en la función llamada referenciamos al archivo model.json (mientras que cuando guardamos, especificamos la carpeta). El o los archivos `.bin` correspondientes deberían estar en la misma carpeta que el archivo `json`.

## Carga de modelos con IOHandlers

Si los esquemas aquí presentados no son suficientes para cubrir sus necesidades, puede implementar un comportamiento de carga personalizada con un `IOHandler`. Uno de los `IOHandler` que proporciona TensorFlow.js es [`tf.io.browserFiles`](https://js.tensorflow.org/api/latest/#io.browserFiles), que permite a los usuarios de un navegador cargar archivos de modelo en el navegador. Para más información, consulte la [documentación](https://js.tensorflow.org/api/latest/#io.browserFiles).

# Guardar y cargar modelos con IOHandlers personalizados

Si los esquemas anteriores no son suficientes para cubrir sus necesidades de carga o guardado, lo que puede hacer es utilizar un comportamiento de serialización personalizado mediante la implementación de un `IOHandler`.

Un `IOHandler` es un objetivo con un método `save` y `load`.

La función `save` toma un parámetro que coincide con la interfaz [ModelArtifacts](https://github.com/tensorflow/tfjs-core/blob/master/src/io/types.ts#L165) y debería devolver una promesa que se resolviera en un objeto [SaveResult](https://github.com/tensorflow/tfjs-core/blob/master/src/io/types.ts#L107).

La función `load` no toma parámetros y debería devolver una promesa resuelve a un objeto [ModelArtifacts](https://github.com/tensorflow/tfjs-core/blob/master/src/io/types.ts#L165). Es el mismo objeto pasado a `save`.

Para ver un ejemplo sobre cómo implementar un IOHandler, consulte [BrowserHTTPRequest](https://github.com/tensorflow/tfjs-core/blob/master/src/io/browser_http.ts).
