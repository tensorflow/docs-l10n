# Tensores e Operações

O TensorFlow.js é um framework para definir e executar computações usando tensores em JavaScript. Um *tensor* é uma generalização de vetores e matrizes com dimensões superiores.

## Tensores

A unidade central de dados no TensorFlow.js é o `tf.Tensor`, um conjunto de valores formatados em um array com uma ou mais dimensões. Os `tf.Tensor`s são muito similares a arrays multidimensionais.

Um `tf.Tensor` também contém as seguintes propriedades:

- `rank`: define quantas dimensões o tensor contém
- `shape`: define o tamanho de cada dimensão dos dados
- `dtype`: define o tipo de dados do tensor

Observação: usaremos o termo "dimensão" de forma intercambiável com posto. Às vezes, em aprendizado de máquina, a "dimensionalidade" de um tensor também pode se referir ao tamanho de uma dimensão específica (por exemplo: uma matriz de formato [10, 5] é um tensor de posto 2 ou um tensor bidimensional. A dimensionalidade da primeira dimensão é 10. Isso pode ser confuso, mas adicionamos esta observação porque provavelmente você vai se deparar com esses dois usos do termo).

Um `tf.Tensor` pode ser criado a partir de um array com o método `tf.tensor()`:

```js
// Cria um tensor de posto 2 (matriz) a partir de um array multidimensional.
const a = tf.tensor([[1, 2], [3, 4]]);
console.log('shape:', a.shape);
a.print();

// Ou você pode criar um tensor a partir de um array "plano" e especificar o formato.
const shape = [2, 2];
const b = tf.tensor([1, 2, 3, 4], shape);
console.log('shape:', b.shape);
b.print();
```

Por padrão, os `tf.Tensor`s terão `float32` como o `dtype`. Os   `tf.Tensor`s também podem ser criados com o dtypes bool, int32, complex64 e string:

```js
const a = tf.tensor([[1, 2], [3, 4]], [2, 2], 'int32');
console.log('shape:', a.shape);
console.log('dtype:', a.dtype);
a.print();
```

O TensorFlow.js também oferece um conjunto de métodos convenientes para criar tensores aleatórios, tensores preenchidos com um valor específico, tensores a partir de `HTMLImageElement`s e muitos outros, que você pode conferir [aqui](https://js.tensorflow.org/api/latest/#Tensors-Creation).

#### Alteração do formato de um tensor

O número de elementos em um `tf.Tensor` é o produto dos tamanhos em seu formato. Como muitas vezes podem existir múltiplos formatos com o mesmo tamanho, é útil poder reformatar um `tf.Tensor` para outro formato com o mesmo tamanho. Isso pode ser feito usando o método `reshape()`:

```js
const a = tf.tensor([[1, 2], [3, 4]]);
console.log('a shape:', a.shape);
a.print();

const b = a.reshape([4, 1]);
console.log('b shape:', b.shape);
b.print();
```

#### Obtenção de valores de um tensor

Também é possível obter os valores de um `tf.Tensor` usando o método `Tensor.array()` ou `Tensor.data()`:

```js
 const a = tf.tensor([[1, 2], [3, 4]]);
 // Retorna o array multidimensional de valores.
 a.array().then(array => console.log(array));
 // Retorna os dados vindos do tensor de forma "planificada".
 a.data().then(data => console.log(data));
```

Também oferecemos versões síncronas desses métodos, que são mais simples de usar, mas causarão problemas de desempenho em sua aplicação. Sempre prefira os métodos assíncronos para aplicações em produção.

```js
const a = tf.tensor([[1, 2], [3, 4]]);
// Retorna o array de valores multidimensional.
console.log(a.arraySync());
// Retorna os dados vindos do tensor de forma "planificada".
console.log(a.dataSync());
```

## Operações

Embora os tensores permitam armazenar dados, as operações (ops) permitem manipular esses dados. O TensorFlow.js também oferece diversas operações adequadas para álgebra linear e aprendizado de máquina que podem ser realizadas em tensores.

Exemplo: computar x<sup>2</sup> de todos os elementos em um `tf.Tensor`:

```js
const x = tf.tensor([1, 2, 3, 4]);
const y = x.square();  // Equivalennte à tf.square(x)
y.print();
```

Exemplo: adicionar elementos de dois `tf.Tensor`s com reconhecimento de elementos:

```js
const a = tf.tensor([1, 2, 3, 4]);
const b = tf.tensor([10, 20, 30, 40]);
const y = a.add(b);  // Equivalente à tf.add(a, b)
y.print();
```

Como os tensores são imutáveis, essas operações não mudam os valores deles. Em vez disso, elas sempre retornam novos `tf.Tensor`s.

> Observação: a maioria das operações retorna `tf.Tensor`s. Entretanto, o resultado pode ainda não estar pronto. Isso significa que o `tf.Tensor` obtido é na verdade um identificador da computação. Quando você chama `Tensor.data()` ou `Tensor.array()`, esses métodos retornam promises que resolvem para valores somente quando a computação é concluída. Ao executar em um contexto de interface gráfica (como uma aplicação para navegador), você sempre deve preferir usar as versões assíncronas desses métodos em vez das contrapartes síncronas para evitar bloquear o thread de interface gráfica até que a computação seja concluída.

Confira a lista das operações compatíveis com o TensorFlow.js [aqui](https://js.tensorflow.org/api/latest/#Operations).

## Memória

Ao usar o back-end WebGEL, a memória do `tf.Tensor` precisa ser gerenciada explicitamente (**não é suficiente** deixar o `tf.Tensor` sair do escopo para que sua memória seja liberada).

Para destruir a memória de um tf.Tensor, você pode usar o método `dispose()` ou `tf.dispose()`:

```js
const a = tf.tensor([[1, 2], [3, 4]]);
a.dispose(); // Equivalente à tf.dispose(a)
```

É muito comum encadear diversas operações em uma aplicação. Armazenar uma referência a todas as variáveis intermediárias para descartá-las depois pode reduzir a legibilidade do código. Para resolver esse problema, o TensorFlow.js conta com o método `tf.tidy()`, que elimina todos os `tf.Tensor`s que não são retornados por uma função após a execução, similar à forma como as variáveis locais são eliminadas quando uma função é executada:

```js
const a = tf.tensor([[1, 2], [3, 4]]);
const y = tf.tidy(() => {
  const result = a.square().log().neg();
  return result;
});
```

Neste exemplo, o resultado de `square()` e `log()` será descartado automaticamente. O resultado de `neg()` não será descartado, pois é o valor de retorno de tf.tidy().

Também é possível obter o número de tensores controlados pelo TensorFlow.js:

```js
console.log(tf.memory());
```

O objeto exibido via print por `tf.memory()` conterá informações de quanta memória está alocada no momento. Confira mais informações [aqui](https://js.tensorflow.org/api/latest/#memory).
