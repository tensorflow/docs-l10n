# Generación de agrupamientos del navegador de tamaño optimizado con TensorFlow.js

## Descripción general

TensorFlow.js 3.0 aporta lo necesario para crear *{nbsp}agrupamientos (bundling) para el navegador orientados a producción y con tamaño optimizado*. Pongámoslo de otro modo, queremos facilitar el envío de menos JavaScript al navegador.

Esta característica está orientada a usuarios con casos de uso de producción que se beneficiarían particularmente de reducir el número de bytes a una cantidad muy pequeña de su carga útil (y que, por lo tanto, pretenden dedicar el esfuerzo a lograr esto). Para usar esta característica deberá estar familiarizado con los [módulos ES](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Modules), con herramientas para agrupamientos de JavaScript como [webpack](https://webpack.js.org/) o [rollup](https://rollupjs.org/guide/en/) y con conceptos como el de la [eliminación de código muerto o agitar el árbol](https://developers.google.com/web/fundamentals/performance/optimizing-javascript/tree-shaking).

En este tutorial se demuestra cómo crear un módulo tensorflow.js personalizado que se pueda utilizar con un agrupador (<em>bundler</em>) para generar construcciones de tamaño optimizado para un programa usando tensorflow.js.

### Terminología

En el contexto de este documento hay algunos términos que usaremos:

**[Módulos ES](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Modules)**: **el sistema estándar de módulos de JavaScript**. Presentado en ES6/ES2015. Identificable mediante el uso de instrucciones para **importar** y **exportar**.

**Agrupamiento (<em>bundling</em>)**: consiste en tomar un conjunto de activos de JavaScript y agruparlos en uno o más activos de JavaScript que se puedan usar en un navegador. Es el paso que, por lo común, producen los activos finales que se sirven al navegador. ***Por lo general, las aplicaciones hacen su propio agrupamiento directamente a partir de fuentes tipo biblioteca transpiladas*.** Entre los **agrupadores** (<em>bundlers</em>) más comunes se incluyen el *rollup* y el *webpack*. El resultado final de agrupar de este modo es lo que se conoce como **agrupamiento (<em>bundle</em>)** (o, a veces, como **fragmento (<em>chunk</em>)** si se divide en muchas partes).

**[Agitar el árbol / eliminación de código muerto](https://developers.google.com/web/fundamentals/performance/optimizing-javascript/tree-shaking)**: se trata de la eliminación de código que la aplicación escrita final no usa. Se realiza durante el agrupamiento, *normalmente* en el paso de "minificación".

**Operaciones (Ops)**: una operación matemática en uno o más tensores que produce uno o más tensores como salida. Las operaciones son código de "alto nivel" y pueden usar otras operaciones para definir su lógica.

**Núcleo (<em>Kernel</em>)**: es una implementación específica de una operación ligada a capacidades de hardware también específicas. Los núcleos son de "bajo nivel"" y son específicos de un backend. Algunas operaciones tienen un mapeo uno a uno desde la operación al núcleo, mientras que otras usan varios núcleos.

## Alcance y casos de uso

### Inferencia solamente en modelos de grafos

El principal caso de uso que conocemos por los comentarios de usuarios relacionados con este tema y que apoyamos en este lanzamiento es el de hacer **inferencias con *modelos de grafos de TensorFlow.js***. Si se está usando un *modelo de capas de TensorFlow.js*, se puede convertir a un formato de modelo de grafo con el convertidor [tfjs-converter](https://www.npmjs.com/package/@tensorflow/tfjs-converter). El formato de modelo de grafo es más eficiente para el caso de uso de inferencia.

### Manipulación de tensores de bajo nivel con tfjs-core

El otro caso de uso que apoyamos es el de los programas que utilizan directamente el paquete @tensorflow/tjfs-core para manipulación de tensores de bajo nivel.

## Nuestro método para la creación personalizada

Nuestros principios esenciales para el diseño de esta funcionalidad incluyen lo siguiente:

- Usar al máximo del sistema de módulo de JavaScript (ESM) y permitir que los usuarios de TensorFlow.js hagan lo mismo.
- Hacer que TensorFlow.js permita *agitar el árbol* lo más posible con los *agrupadores* (p. ej., con webpack, rollup, etc.). Gracias a esto, los usuarios pueden aprovechar todas las capacidades de los agrupadores, incluidas características como la de división de código.
- Mantener lo máximo posible la *facilidad de uso para los usuarios que no son tan sensibles al tamaño del agrupamiento*. Significa que para las construcciones de producción se requerirá de un mayor esfuerzo, ya que muchas de las opciones predeterminadas de nuestras bibliotecas favorecen la facilidad de uso por sobre las construcciones de tamaño optimizado.

El objetivo principal de nuestro flujo de trabajo es producir un *módulo de JavaScript* personalizado para TensorFlow.js que contenga solamente la funcionalidad requerida para el programa que intentamos optimizar. A fin de lograr la optimización, dependemos de los agrupadores existentes.

Si bien es cierto que, en primer lugar, dependemos del sistema de módulo de JavaScript, también cabe considerar que proporcionamos una *herramienta CLI* *personalizada* para administrar partes que no son fáciles de especificar a través del sistema del módulo en código de cara al usuario. A continuación, dos ejemplos:

- Especificaciones del modelo almacenadas en archivos `model.json`
- La operación que usamos para el sistema de distribución de núcleo específico por backend.

Todo esto hace que la generación de una construcción tfjs personalizada necesite un poco más de participación que la de solamente apuntar un agrupador al paquete @tensorflow/tfjs regular.

## Cómo crear agrupamientos personalizados con tamaño optimizado

### Paso 1: Determine qué núcleos (kernels) usa su programa

**Este paso nos permitirá determinar cuáles son todos los núcleos usados por cualquiera de los modelos que ejecute o el código de pre o posprocesamiento, según el backend que haya seleccionado.**

Use tf.profile para ejecutar las partes de su aplicación que utilicen tensorflow.js y obtenga los kernels. Obtendrá algo similar a lo que se muestra a continuación:

```
const profileInfo = await tf.profile(() => {
  // You must profile all uses of tf symbols.
  runAllMyTfjsCode();
});

const kernelNames = profileInfo.kernelNames
console.log(kernelNames);
```

Copie esa lista de núcleos en el portapapeles para guardarlo para el paso siguiente.

> Deberá perfilar el código con los mismos backend que quiera usar en el agrupamiento personalizado.

> Si su modelo o el código de pre/posprocesamiento cambian, deberá repetir este paso.

### Paso 2. Escriba un archivo de configuración para el módulo tfjs personalizado

Compartimos un ejemplo del archivo de configuración.

Así es como se ve:

```
{
  "kernels": ["Reshape", "_FusedMatMul", "Identity"],
  "backends": [
      "cpu"
  ],
  "models": [
      "./model/model.json"
  ],
  "outputPath": "./custom_tfjs",
  "forwardModeOnly": true
}
```

- Los núcleos (kernels): la lista de núcleos del agrupamiento. Es una copia de la salida del paso 1.
- Los backend: la lista de los backend que desea incluir. Entre las opciones válidas se encuentran "cpu", "webgl" y “wasm”.
- Los modelos (models): una lista de los archivos model.json para los modelos que cargue en su aplicación. Puede estar vacía si el programa no usa tfjs_converter para cargar un modelo de grafo.
- La ruta de salida (outputPath): una ruta a una carpeta donde se pondrán los módulos generados.
- Solo el modo hacia adelante (forwardModeOnly): configure esta opción como falsa si lo que desea es incluir gradientes para los núcleos del listado.

### Paso 3. Genere el módulo tfjs personalizado

Ejecute la herramienta personalizada con el archivo config como argumento. Deberá tener el paquete **@tensorflow/tfjs** instalado para tener acceso a esta herramienta.

```
npx tfjs-custom-module  --config custom_tfjs_config.json
```

De este modo, se creará una carpeta en `outputPath` con algunos archivos nuevos.

### Paso 4. Configure el agrupador con el alias tfjs para el nuevo módulo personalizado.

En agrupadores como webpack y rollup podemos asignar alias a las referencias de los módulos tfjs que ya existían para apuntar a nuestros nuevos módulos tfjs personalizados recién generados. Hay tres módulos a los que se les deben asignar alias para lograr un ahorro máximo en el tamaño del agrupamiento.

El siguiente es un fragmento de cómo luce en webpack ([ver el ejemplo completo aquí](https://github.com/tensorflow/tfjs/blob/master/e2e/custom_module/dense_model/webpack.config.js)):

```
...

config.resolve = {
  alias: {
    '@tensorflow/tfjs$':
        path.resolve(__dirname, './custom_tfjs/custom_tfjs.js'),
    '@tensorflow/tfjs-core$': path.resolve(
        __dirname, './custom_tfjs/custom_tfjs_core.js'),
    '@tensorflow/tfjs-core/dist/ops/ops_for_converter': path.resolve(
        __dirname, './custom_tfjs/custom_ops_for_converter.js'),
  }
}

...
```

Y el que sigue, es el fragmento de código equivalente para rollup ([ver el ejemplo completo aquí](https://github.com/tensorflow/tfjs/blob/master/e2e/custom_module/dense_model/rollup.config.js)):

```
import alias from '@rollup/plugin-alias';

...

alias({
  entries: [
    {
      find: /@tensorflow\/tfjs$/,
      replacement: path.resolve(__dirname, './custom_tfjs/custom_tfjs.js'),
    },
    {
      find: /@tensorflow\/tfjs-core$/,
      replacement: path.resolve(__dirname, './custom_tfjs/custom_tfjs_core.js'),
    },
    {
      find: '@tensorflow/tfjs-core/dist/ops/ops_for_converter',
      replacement: path.resolve(__dirname, './custom_tfjs/custom_ops_for_converter.js'),
    },
  ],
}));

...
```

> Si su agrupador no es compatible con la asignación de alias a módulos, deberá cambiar las instrucciones de `import` para poder importar tensorflow.js a partir del `custom_tfjs.js` generado que se creó en el paso 3. A las definiciones de operaciones no se les podrá "agitar el árbol", pero a los núcleos sí se les "agitará el árbol". Por lo general, los núcleos que pasan por este proceso son los que ofrecen los mayores ahorros para el tamaño de agrupamiento final.

> Si solamente usa el paquete @tensoflow/tfjs-core, deberá asignar únicamente el de este paquete.

### Paso 5. Cree su propio agrupamiento

Ejecute su agrupador (p. ej. `webpack` o `rollup`) para producir su propio agrupamiento. El tamaño del agrupamiento debería ser más pequeño que el que obtendría si ejecutara el agrupador sin asignarle un alias al módulo. También puede usar visualizadores como [este](https://www.npmjs.com/package/rollup-plugin-visualizer) para ver qué hizo en el agrupamiento final.

### Paso 6. Pruebe la aplicación

No olvide probar que la aplicación funcione según lo esperado.
