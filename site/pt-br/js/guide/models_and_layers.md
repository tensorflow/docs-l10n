# Modelos e camadas

Em aprendizado de máquina, um *modelo* é uma função com [parâmetros](https://developers.google.com/machine-learning/glossary/#parameter) *que podem ser aprendidos* e que mapeia uma entrada para uma saída. Os parâmetros ideais são obtidos ao treinar o modelo com dados. Um modelo bem treinado fornecerá um mapeamento preciso da entrada para a saída desejada.

No TensorFlow.js, há duas maneiras de criar um modelo de aprendizado de máquina:

1. Usando a API Layers, em que você cria um modelo usando *camadas*.
2. Usando a API Core com operações de baixo nível, como `tf.matMul()`, `tf.add()`, etc.

Primeiros, veremos a API Layers, uma API de alto nível para criar modelos. Em seguida, mostraremos como criar o mesmo modelo usando a API Core.

## Criando modelos com a API Layers

Existem duas formas de criar um modelo usando a API Layers: um modelo *sequencial* e um modelo *funcional*. As próximas duas seções apresentam cada tipo com maiores detalhes.

### Modelo sequencial

O tipo mais comum de modelo é o modelo <code>sequencial</code>, que é uma pilha linear de camadas. Para criar um modelo <code>sequencial</code>, basta passar uma lista de camadas à função <code>sequential()</code>:

```js
const model = tf.sequential({
 layers: [
   tf.layers.dense({inputShape: [784], units: 32, activation: 'relu'}),
   tf.layers.dense({units: 10, activation: 'softmax'}),
 ]
});
```

Ou pelo método `add()`:

```js
const model = tf.sequential();
model.add(tf.layers.dense({inputShape: [784], units: 32, activation: 'relu'}));
model.add(tf.layers.dense({units: 10, activation: 'softmax'}));
```

> IMPORTANTE: a primeira camada no modelo precisa de um `inputShape` (formato de entrada). Você deve excluir o tamanho do lote ao fornecer `inputShape`. Por exemplo: se você planeja alimentar o modelo com tensores de formato `[B, 784]`, em que `B` pode ser qualquer tamanho de lote, especifique `inputShape` como `[784]` ao criar o modelo.

É possível acessar as camadas do modelo por meio de `model.layers` e, mais especificamente, por meio de `model.inputLayers` e `model.outputLayers`.

### O modelo funcional

Outra maneira de criar um `LayersModel` é pela função `tf.model()`. A principal diferença entre `tf.model()` e `tf.sequential()` é que `tf.model()` permite criar um grafo arbitrário de camadas, desde que elas não tenham ciclos.

Veja o trecho de código que define o mesmo modelo que o acima usando a API `tf.model()`:

```js
// Create an arbitrary graph of layers, by connecting them
// via the apply() method.
const input = tf.input({shape: [784]});
const dense1 = tf.layers.dense({units: 32, activation: 'relu'}).apply(input);
const dense2 = tf.layers.dense({units: 10, activation: 'softmax'}).apply(dense1);
const model = tf.model({inputs: input, outputs: dense2});
```

Chamamos `apply()` em cada camada para conectá-la à saída de outra camada. Neste caso, o resultado de `apply()` é um `SymbolicTensor`, que funciona como um `Tensor`, mas sem valores concretos.

Observe que, diferentemente do modelo sequencial, criamos um `SymbolicTensor` por meio de `tf.input()` em vez de fornecer um `inputShape` (formato de entrada) à primeira camada.

A função `apply()` também pode fornecer um `Tensor` concreto se você passar um `Tensor` a ela:

```js
const t = tf.tensor([-2, 1, 0, 5]);
const o = tf.layers.activation({activation: 'relu'}).apply(t);
o.print(); // [0, 1, 0, 5]
```

Isso pode ser útil ao testar camadas isoladamente para ver sua saída.

Tal como em um modelo sequencial, é possível acessar as camadas do modelo por meio de `model.layers` e, mais especificamente, por meio de `model.inputLayers` e `model.outputLayers`.

## Validação

Tanto o modelo sequencial quanto o funcional são instâncias da classe `LayersModel`. Um dos grandes benefícios de trabalhar com um `LayersModel` é a validação: esse modelo força você a especificar o formato da entrada e utilizará esse formato para validar sua entrada. `LayersModel` também faz inferência automática do formato à medida que os dados fluem pelas camadas. Ao saber o formato com antecedência, o modelo pode criar seus parâmetros automaticamente e informar se duas camadas consecutivas não forem compatíveis entre si.

## Resumo do modelo

Chame `model.summary()` para exibir via print um resumo útil do modelo, que inclui:

- Nome e tipo de todas as camadas do modelo.
- Formato de saída de cada camada.
- Número de parâmetros de peso de cada camada.
- Se o modelo tiver uma topologia geral (discutida abaixo), as entradas que cada camada recebe.
- Número total de parâmetros treináveis e não treináveis do modelo.

Para o modelo que definimos acima, termos a seguinte saída no console:

<table>
  <tr>
   <td>Camada (tipo)</td>
   <td>Formato da saída</td>
   <td>Nº de parâm.</td>
  </tr>
  <tr>
   <td>dense_Dense1 (Dense)</td>
   <td>[null,32]</td>
   <td>25.120</td>
  </tr>
  <tr>
   <td>dense_Dense2 (Dense)</td>
   <td>[null,10]</td>
   <td>330</td>
  </tr>
  <tr>
   <td colspan="3">Total de parâmetros: 25.450<br>Parâmetros treináveis: 25.450<br> Parâmetros não treináveis: 0</td>
  </tr>
</table>

Observe os valores `null` nos formatos da saída das camadas: um lembrete de que o modelo espera que a entrada tenha um tamanho de lote como a dimensão mais externa, que, neste caso, pode ser flexível devido ao valor `null`.

## Serialização

Um dos grandes benefícios de usar um `LayersModel` em vez da API de baixo nível é a capacidade de salvar e carregar um modelo. Um `LayersModel` sabe:

- A arquitetura do modelo, permitindo que você recrie-o.
- Os pesos do modelo.
- A configuração de treinamento (perda, otimizador, métricas).
- O estado do otimizador, o que permite retomar o treinamento.

Para salvar ou carregar um modelo, basta uma linha de código:

```js
const saveResult = await model.save('localstorage://my-model-1');
const model = await tf.loadLayersModel('localstorage://my-model-1');
```

O exemplo acima salva o modelo no armazenamento local do navegador. Confira a &lt;a href="https://js.tensorflow.org/api/latest/#tf.Model.save" data-md-type="link"&gt;documentação de <code>model.save()</code>&lt;/a0&gt; e o guia de como [salvar e carregar](save_load.md) para ver como salvar em diferentes locais (por exemplo, armazenamento de arquivos, <code>IndexedDB</code>, iniciar download no navegador, etc.)

## Camadas personalizadas

As camadas são blocos de construção de um modelo. Se o seu modelo estiver fazendo uma computação personalizada, você pode definir uma camada personalizada, que interage bem com o restante das camadas. Definimos abaixo uma camada personalizada que computa a soma de quadrados:

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

Para testar, podemos chamar o método `apply()` com um tensor concreto:

```js
const t = tf.tensor([-2, 1, 0, 5]);
const o = new SquaredSumLayer().apply(t);
o.print(); // prints 30
```

> IMPORTANTE: se você adicionar uma camada personalizada, perde a capacidade de serializar um modelo.

## Criando modelos com a API Core

No começo deste guia, mencionamos que existem duas maneiras de criar um modelo de aprendizado de máquina no TensorFlow.js.

A regra geral é sempre tentar usar a API Layers primeiro, já que ela é modelada de acordo com a API Keras amplamente adotada, que segue [as práticas recomendadas e reduz a carga cognitiva](https://keras.io/why-use-keras/). A API Layers também oferece diversas soluções prontas para uso, como inicialização de pesos, serialização de modelos, monitoramento de treinamento, portabilidade e checagem de segurança.

Talvez você queira usar a API Core sempre que:

- Precisar de flexibilidade ou controle máximos.
- Não precisar de serialização ou puder implementar sua própria lógica de serialização.

Os modelos da API Core são apenas funções que recebem um ou mais `Tensors` e retornam um `Tensor`. Confira abaixo o mesmo modelo conforme escrito acima usando a API Core:

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

Observe que, na API Core, somos responsáveis por criar e inicializar os pesos do modelo. Cada peso tem uma `Variable` correspondente, que sinaliza ao TensorFlow.js que esses tensores podem ser aprendidos. Você pode criar uma `Variable` usando [tf.variable()](https://js.tensorflow.org/api/latest/#variable) e passando-a para um `Tensor` existente.

Neste guia, você viu as diferentes formas de criar um modelo usando as APIs Layers e Core. Agora, confira o guia [Treinamento de modelos](train_models.md) para ver como treinar um modelo.
