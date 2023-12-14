# Plataforma y entorno

TensorFlow.js funciona en el navegador Node.js y en ambas plataformas hay muchas configuraciones diferentes disponibles. Cada plataforma tiene un conjunto único de consideraciones que afectará la forma en que se desarrollen las aplicaciones.

En el navegador, TensorFlow.js es compatible tanto con dispositivos móviles como con los de escritorio. Cada dispositivo tiene un conjunto específico de restricciones, como las API WebGL, que se determinan y configuran automáticamente.

En Node.js, TensorFlow.js admite la vinculación directa con la API de TensorFlow o la ejecución más lenta de las implementaciones en CPU vainilla.

## Entornos

Cada vez que se ejecuta un programa TensorFlow.js, a la configuración específica se la denomina entorno.  El entorno está compuesto por un backend global específico y un conjunto de marcas que controlan las características de granulado fino de TensorFlow.js.

### Backends

TensorFlow.js es compatible con muchos backends diferentes que implementan las operaciones matemáticas y de almacenamiento en tensores. En todo momento hay solamente un backend activo.<br>La mayor parte del tiempo, TensorFlow.js elegirá automáticamente el mejor backend, según el entorno actual del momento. Sin embargo, a veces, es importante saber qué backend se está usando y cómo cambiarlo.

Para saber qué backend se está usando:

```js
console.log(tf.getBackend());
```

Si desea cambiar el backend de forma manual:

```js
tf.setBackend('cpu');
console.log(tf.getBackend());
```

#### Backend WebGL

El backend WebGL, 'webgl', actualmente, es el más potente para el navegador. Este backend es hasta 100 veces más rápido que el de CPU vainilla. Los tensores se almacenan como texturas WebGL y las operaciones matemáticas se implementan en sombreadores de WebGL. A continuación, compartimos algunos puntos útiles que le convendrá saber cuando utilice este backend:  \

##### Evitar bloquear el hilo de UI

Cuando se llama a la operación, como con tf.matMul(a, b), el tf.Tensor resultante se devuelve sincrónicamente, pero el cómputo de la multiplicación de la matriz puede no estar listo para ser leído. Significa que el tf.Tensor devuelto es solamente un <em>handle</em> para el cálculo. Cuando se llame a `x.data()` o `x.array()`, los valores resolverán cuándo el cálculo haya terminado realmente. Por este motivo es importante usar los métodos asincrónicos `x.data()` y `x.array()` en vez de sus contrapartes sincrónicas `x.dataSync()` y `x.arraySync()`, para evitar bloquear el hilo de la UI mientras se completa el cálculo.

##### Gestión de la memoria

Una salvedad, cuando se usa el backend WebGL hace falta una gestión de la memoria explícita. WebGLTextures (que es donde, en definitiva, se almacenan los datos del tensor) no es basura que se recolecta automáticamente con el navegador.

Para destruir la memoria de un `tf.Tensor`, se puede usar el método `dispose()`:

```js
const a = tf.tensor([[1, 2], [3, 4]]);
a.dispose();
```

Es muy común, la formación de una cadena con varias operaciones juntas en una aplicación. Fijar una referencia en la cual disponer todas las variables intermedias puede reducir la legibilidad del código. Para resolver este problema, TensorFlow.js ofrece un método `tf.tidy()` con el que se limpian todos los `tf.Tensor` que no son devueltos por una función después de su ejecución, similar a la forma en que las variables locales se limpian cuando se ejecuta una función:

```js
const a = tf.tensor([[1, 2], [3, 4]]);
const y = tf.tidy(() => {
  const result = a.square().log().neg();
  return result;
});
```

> Nota: No hay inconvenientes con usar `dispose()` o `tidy()` en entornos no WebGL (como Node.js o un backend de CPU) que tenga una colección automática de basura. De hecho, con frecuencia puede ser favorable al desempeño para liberar la memoria del tensor más rápido de lo que hubiese sido naturalmente con la colección de basura.

##### Precisión

En dispositivos móviles, WebGL, solamente podría admitir texturas de punto flotante de 16 bits. Sin embargo, la mayoría de los modelos de aprendizaje automático se entrenan con activaciones y pesos de punto flotante de 32 bits. La consecuencia es que puede causar problemas de precisión al portar un modelo a un dispositivo móvil, ya que los números flotantes de 16 bits solamente representan números dentro del rango de `[0.000000059605, 65504]`. Significa que habría que prestar particular atención a que los pesos y las activaciones del modelo no excedan ese rango. Para verificar si el dispositivo es compatible con texturas de 32 bits, observe el valor de `tf.ENV.getBool('WEBGL_RENDER_FLOAT32_CAPABLE')`; si es falso, el dispositivo solamente admite texturas de punto flotante de 16 bits. Para controlar si TensorFlow.js está usando texturas de 32 bits, puede usar <br>`tf.ENV.getBool('WEBGL_RENDER_FLOAT32_ENABLED')`.

##### Compilación de sombreador y cargas de textura

TensorFlow.js ejecuta operaciones en la GPU mediante la ejecución de programas sombreadores de WebGL. Estos sombreadores se ensamblan y compilan con más lentitud cuando el usuario pide ejecutar una operación. La compilación de un sombreador se produce en la CPU, en el hilo principal, y puede ser lenta. TensorFlow.js almacena automáticamente en caché los sombreadores compilados, en consecuencia, la segunda llamada a la misma operación con los tensores de entrada y salida de la misma forma será mucho más rápida. Normalmente, las aplicaciones de TensorFlow.js usan las mismas operaciones varias veces durante la vida útil de la aplicación, de modo que el segundo pasaje por el modelo de aprendizaje automático será mucho más rápido.

TensorFlow.js también almacena datos de tf.Tensor WebGLTextures. Cuando se crea un `tf.Tensor`, no cargamos inmediatamente los datos en la GPU, sino que guardamos los datos en la CPU hasta que el `tf.Tensor` se use en una operación. Si el `tf.Tensor` se utiliza una segunda vez, los datos ya estarán en la GPU, por lo tanto, no habrá costo de carga. En un modelo típico de aprendizaje automático, significa que los pesos se cargan durante la primera predicción a través del modelo y que ya el segundo paso a través del modelo será mucho más rápido.

Si le interesa conocer más acerca del rendimiento de la primera predicción a través del modelo o de código de TensorFlow.js, le recomendamos precalentar el modelo pasando un tensor de entrada de la misma forma antes de usar los datos reales.

Por ejemplo:

```js
const model = await tf.loadLayersModel(modelUrl);

// Warmup the model before using real data.
const warmupResult = model.predict(tf.zeros(inputShape));
warmupResult.dataSync();
warmupResult.dispose();

// The second predict() will be much faster
const result = model.predict(userData);
```

#### Backend de TensorFlow Node.js

En el backend de TensorFlow Node.js, el 'nodo', la API C de TensorFlow se usa para acelerar las operaciones. Utilizará la aceleración del hardware disponible de la máquina, como CUDA, en caso de estar disponible.

En este backend, igual que con el WebGL, las operaciones devuelven tensores `tf.Tensor` sincrónicamente. Sin embargo, a diferencia de lo que sucede con el backend WebGL, la operación se completa antes de recibir el tensor devuelto. Significa que una llamada a `tf.matMul(a, b)` bloqueará el hilo de la UI.

Por este motivo, si lo que se pretende es usarlo en una aplicación de producción, habría que ejecutar TensorFlow.js en hilos trabajadores para no bloquear el hilo principal.

Para más información sobre Node.js, consulte la siguiente guía.

#### Backend WASM

TensorFlow.js ofrece un [backend WebAssembly](https://github.com/tensorflow/tfjs/blob/master/tfjs-backend-wasm/README.md) (`wasm`), que brinda la aceleración de la CPU y que puede usarse como alternativa a los backend de las CPU JavaScript vainilla (`cpu`) y WebGL acelerado (`webgl`).  Para usarlo:

```js
// Set the backend to WASM and wait for the module to be ready.
tf.setBackend('wasm');
tf.ready().then(() => {...});
```

Si su servidor opera el archivo `.wasm` con una ruta o nombre diferente, use `setWasmPath` antes de inicializar el backend. Para más información, consulte la sección ["Using Bundlers"](https://github.com/tensorflow/tfjs/tree/master/tfjs-backend-wasm#using-bundlers) en el archivo README:

```js
import {setWasmPath} from '@tensorflow/tfjs-backend-wasm';
setWasmPath(yourCustomPath);
tf.setBackend('wasm');
tf.ready().then(() => {...});
```

> Nota: TensorFlow.js define una prioridad para cada backend y elegirá automáticamente al mejor backend compatible para un entorno dado. Para utilizar explícitamente el backend WASM, debemos llamar a `tf.setBackend('wasm')`.

##### Por qué usar WASM

[WASM](https://webassembly.org/) se presentó por primera vez en 2015 como un nuevo formato binario basado en la web, que aportaba programas escritos en JavaScript, C, C++, etc. con un objetivo de compilación para funcionar en la web. WASM es [compatible](https://webassembly.org/roadmap/) con Chrome, Safari, Firefox y Edge desde 2017 y es compatible con el [90 % de los dispositivos](https://caniuse.com/#feat=wasm) de todo el mundo.

**Desempeño**

El backend WASM aprovecha la [biblioteca XNNPACK](https://github.com/google/XNNPACK) para la implementación optimizada de operadores de redes neuronales.

*Versus JavaScript*: generalmente, la carga, análisis y ejecución de los navegadores es mucho más rápida con los WASM binarios que con los <em>bundles</em> JavaScript. Con JavaScript el tipado es dinámico y hay recolección de basura, lo que puede causar ralentizaciones en el tiempo de ejecución.

*Versus WebGL*: WebGL es más rápida que WASM para la mayoría de los modelos, pero en modelos pequeños WASM pueden hacer rendir de más WebGL debido a los costos extra fijos que conlleva ejecutar los sombreadores de WebGL. En la sección “Cuándo conviene usar WASM” que se encuentra a continuación, se discute sobre la heurística para tomar esta decisión.

**Portabilidad y estabilidad**

WASM tiene una aritmética flotante de 32 bits portátil que ofrece paridad de precisión en todos los dispositivos. WebGL, por otra parte, es específico para hardware y la precisión puede variar según los dispositivos (p.ej., cambiar a flotantes de 16 bits en dispositivos iOS).

Al igual que WebGL, WASM es oficialmente compatible con todos los navegadores principales. A diferencia de lo que sucede con WebGL, WASM se puede ejecutar en Node.js y se puede usar del lado del servidor sin ninguna necesidad de compilar las bibliotecas nativas.

##### Cuándo conviene usar WASM

**Tamaño del modelo y demanda computacional**

En general, WASM es una buena opción cuando los modelos son más pequeños o cuando resultan importantes los dispositivos de gama baja no compatibles con WebGL<br>(extensión `OES_texture_float`) o que las GPU que tienen son menos potentes. En el gráfico a continuación, se muestran tiempos de inferencia (a partir de TensorFlow.js 1.5.2) en Chrome en una MacBook Pro 2018 para 5 de nuestros [modelos](https://github.com/tensorflow/tfjs-models) con compatibilidad oficial en los backends WebGL, WASM y CPU:

**Modelos más pequeños**

Modelo | WebGL | WASM | CPU | Memoria
--- | --- | --- | --- | ---
BlazeFace | 22.5 ms | 15.6 ms | 315.2 ms | .4 MB
FaceMesh | 19.3 ms | 19.2 ms | 335 ms | 2.8 MB

**Modelos más grandes**

Modelo | WebGL | WASM | CPU | Memoria
--- | --- | --- | --- | ---
PoseNet | 42.5 ms | 173.9 ms | 1514.7 ms | 4.5 MB
BodyPix | 77 ms | 188.4 ms | 2683 ms | 4.6 MB
MobileNet v2 | 37 ms | 94 ms | 923.6 ms | 13 MB

En la tabla anterior se muestra que WASM es entre 10 y 30 veces más rápido que el backend JS CPU simple en los diferentes modelos y que compite con WebGL para modelos más pequeños como el [BlazeFace](https://github.com/tensorflow/tfjs-models/tree/master/blazeface), que tiene un peso liviano (400KB), pero que aun así tiene una cantidad decente de operaciones (~140). El hecho de que los programas con WebGL tengan un costo extra fijo por ejecución de operación, explica por qué los modelos como BlazeFace son más rápidos en WASM.

**Estos resultados variarán dependiendo del dispositivo. La mejor manera de determinar si WASM es la opción correcta para una aplicación es probándolo en nuestros diferentes backends.**

##### Inferencia vs. entrenamiento

Para abordar el principal caso de uso para desarrollo de modelos previamente entrenados, el desarrollo del backend WASM priorizará la compatibilidad con la *inferencia* sobre la del *entrenamiento*. Consulte la [lista actualizada](https://github.com/tensorflow/tfjs/blob/master/tfjs-backend-wasm/src/register_all_kernels.ts) de operaciones compatibles en WASM y [díganos](https://github.com/tensorflow/tfjs/issues?q=is%3Aissue+is%3Aopen+sort%3Aupdated-desc) si su modelo tiene una operación incompatible. Para los modelos de entrenamiento recomendamos usar el backend Node (TensorFlow C++) o el WebGL.

#### Backend CPU

El backend CPU, "cpu", es el de menor rendimiento, sin embargo, es el más simple. Las operaciones se implementan en JavaScript vainilla, lo que hace menos paralelizables. También bloquean el hilo de UI.

Este backend puede ser muy útil para pruebas o para usarlo en dispositivos en los que WebGL no está disponible.

### Flags

TensorFlow.js tiene un conjunto de indicadores o flags que se evalúan automáticamente y que determinan la mejor configuración en la plataforma presente. Estos indicadores son principalmente internos, pero algunos indicadores globales se pueden controlar con API pública.

- `tf.enableProdMode():` permite el modo de producción, que quitará el modelo de validación, los controles de NaN y otras verificaciones de correcciones para favorecer el rendimiento.
- `tf.enableDebugMode()`: habilita el modo de depuración, que registrará en la consola cada una de las operaciones que se ejecuten, además de la información sobre el rendimiento del tiempo de ejecución, como el impacto en la memoria y el tiempo total de ejecución en núcleo (kernel). Tenga en cuenta que esta acción ralentizará mucho su aplicación, no la use en producción.

> Nota: Estos dos métodos deberían usarse antes de aplicar cualquier código TensorFlow.js, ya que afectan los valores de otros indicadores (<em>flags</em>) que se almacenarán en caché. Por el mismo motivo, no hay ninguna función análoga para "deshabilitar".

> Nota: Se pueden ver todos los indicadores que han sido evaluados al registrar `tf.ENV.features` en la consola. Si bien es cierto que **no son parte de la API pública **(y, por lo tanto, no tienen garantía de estabilidad entre distintas versiones), sí pueden resultar útiles para depurar o hacer el ajuste fino del comportamiento en distintas plataformas y dispositivos. Para sobrescribir el valor del indicador, puede usar `tf.ENV.set`.
