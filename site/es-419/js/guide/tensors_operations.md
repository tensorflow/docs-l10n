# Tensores y operaciones

TensorFlow.js es un marco de trabajo que sirve para definir y ejecutar cálculos con tensores en JavaScript. Un *tensor* es una generalización de vectores y matrices para dimensiones más altas.

## Tensores

La unidad central de datos en TensorFlow.js es el `tf.Tensor`: un conjunto de valores que forman un arreglo de una o más dimensiones. Los `tf.Tensor` son muy similares a los arreglos multidimensionales.

Un `tf.Tensor` también contiene las siguientes propiedades:

- `rank` (rango): define cuántas dimensiones contiene el tensor.
- `shape` (forma): define el tamaño de cada dimensión de los datos.
- `dtype` (tipo d): define el tipo de datos del tensor.

Nota: Usaremos los términos "dimensión" y "rango" indistintamente. A veces, en aprendizaje automático, la "dimensionalidad" de un tensor también puede referirse al tamaño de una dimensión en particular (p. ej., una matriz de forma [10, 5] es un tensor de rango 2 o un tensor bidimensional. La dimensionalidad de la primera dimensión es 10. Es cierto que puede resultar confuso, pero ponemos aquí esta nota, porque probablemente se encuentre con estos usos duales del término).

Un `tf.Tensor` se puede crear a partir de un arreglo con el método `tf.tensor()`:

```js
// Create a rank-2 tensor (matrix) matrix tensor from a multidimensional array.
const a = tf.tensor([[1, 2], [3, 4]]);
console.log('shape:', a.shape);
a.print();

// Or you can create a tensor from a flat array and specify a shape.
const shape = [2, 2];
const b = tf.tensor([1, 2, 3, 4], shape);
console.log('shape:', b.shape);
b.print();
```

Por defecto, los `tf.Tensor` tendrán `float32` `dtype.` Los `tf.Tensor` se pueden crear con booleanos, int32, complex64 y tipos d en string:

```js
const a = tf.tensor([[1, 2], [3, 4]], [2, 2], 'int32');
console.log('shape:', a.shape);
console.log('dtype', a.dtype);
a.print();
```

El TensorFlow.js también proporciona un conjunto de métodos que facilitan la creación de tensores aleatorios, tensores rellenados con un valor en particular, tensores que parten de `HTMLImageElement` y muchos otros que podrá encontrar [aquí](https://js.tensorflow.org/api/latest/#Tensors-Creation).

#### Cambio de la forma de un tensor

La cantidad de elementos que hay en `tf.Tensor` es el producto de los tamaños que se encuentran en su forma. Como en muchos casos puede haber varias formas con el mismo tamaño, por lo general, resulta útil poder cambiar la forma de un `tf.Tensor` por otra con el mismo tamaño. Esto se logra con el método `reshape()`:

```js
const a = tf.tensor([[1, 2], [3, 4]]);
console.log('a shape:', a.shape);
a.print();

const b = a.reshape([4, 1]);
console.log('b shape:', b.shape);
b.print();
```

#### Obtención de los valores de un tensor

También puede obtener los valores de un `tf.Tensor` con los métodos `Tensor.array()` o `Tensor.data()`:

```js
 const a = tf.tensor([[1, 2], [3, 4]]);
 // Returns the multi dimensional array of values.
 a.array().then(array => console.log(array));
 // Returns the flattened data that backs the tensor.
 a.data().then(data => console.log(data));
```

Además, brindamos versiones sincrónicas de estos métodos, que son más simples de usar, pero que causarán problemas de desempeño en su aplicación. En las aplicaciones de producción siempre conviene optar por los métodos asincrónicos.

```js
const a = tf.tensor([[1, 2], [3, 4]]);
// Returns the multi dimensional array of values.
console.log(a.arraySync());
// Returns the flattened data that backs the tensor.
console.log(a.dataSync());
```

## Operaciones

Si bien es cierto que los tensores permiten almacenar datos, las operaciones (ops) lo que permiten es manipular esos datos. TensorFlow.js también ofrece una amplia variedad de operaciones adecuadas para álgebra lineal y aprendizaje automático que se pueden utilizar en los tensores.

Ejemplo: para el cálculo x<sup>2</sup> de todos los elementos de un `tf.Tensor`:

```js
const x = tf.tensor([1, 2, 3, 4]);
const y = x.square();  // equivalent to tf.square(x)
y.print();
```

Ejemplo: agregamos los elementos de dos `tf.Tensor` con correspondencia entre elementos (<em>element-wise</em>):

```js
const a = tf.tensor([1, 2, 3, 4]);
const b = tf.tensor([10, 20, 30, 40]);
const y = a.add(b);  // equivalent to tf.add(a, b)
y.print();
```

Como los tensores son inmutables, estas operaciones no cambian sus valores. En cambio, los retornos de las operaciones siempre devuelven `tf.Tensor` nuevos.

> Nota: La mayoría de las operaciones devuelven tensores `tf.Tensor`, sin embargo el resultado puede no estar del todo listo. Significa que el `tf.Tensor` que se obtiene, en realidad, es un <em>handle</em> para el cálculo. Cuando se llama a `Tensor.data()` o a `Tensor.array()`, estos métodos devuelven promesas que se resolverán con valores recién cuando haya terminado el cálculo. Cuando la ejecución se lleve a cabo en un contexto de UI (como la aplicación del navegador), siempre convendrá preferir las versiones asincrónicas de estos métodos en vez de sus contrapartes sincrónicas, para evitar bloquear el hilo de UI hasta que se complete el cálculo.

Puede encontrar una lista de las operaciones compatibles con TensorFlow.js [aquí](https://js.tensorflow.org/api/latest/#Operations).

## Memoria

Cuando use el backend WebGL, la memoria de `tf.Tensor` debe gestionarse explícitamente (**no basta** con dejar que un `tf.Tensor` se salga del alcance para que su memoria se libere).

Para destruir la memoria de un tf.Tensor, se pueden usar los métodos `dispose() ` o `tf.dispose()`:

```js
const a = tf.tensor([[1, 2], [3, 4]]);
a.dispose(); // Equivalent to tf.dispose(a)
```

Es muy común, la formación de una cadena con varias operaciones juntas en una aplicación. Fijar una referencia en la cual disponer todas las variables intermedias puede reducir la legibilidad del código. Para resolver este problema, TensorFlow.js ofrece un método `tf.tidy()` con el que se limpian todos los `tf.Tensor` que no son devueltos por una función después de su ejecución, similar a la forma en que las variables locales se limpian cuando se ejecuta una función:

```js
const a = tf.tensor([[1, 2], [3, 4]]);
const y = tf.tidy(() => {
  const result = a.square().log().neg();
  return result;
});
```

En este ejemplo, el resultado de `square()` y `log()` se eliminará automáticamente. El resultado de `neg()` no se eliminará ya que es el valor de retorno de tf.tidy().

También se puede obtener la cantidad de tensores rastreados por TensorFlow.js:

```js
console.log(tf.memory());
```

El objeto impreso por `tf.memory()` contendrá información sobre cuánta memoria está ocupada. Encontrará más información al respecto [aquí](https://js.tensorflow.org/api/latest/#memory).
