# Como fazer upgrade para o TensorFlow.js 3.0

## O que mudou no TensorFlow.js 3.0

As notas de versão estão [disponíveis aqui](https://github.com/tensorflow/tfjs/releases). Veja alguns recursos importantes voltados para o usuário:

### Módulos personalizados

Oferecemos suporte à criação de módulos tfjs personalizados para permitir pacotes de navegador otimizados para produção. Envie menos código JavaScript aos usuários. Para saber mais, [confira este tutorial](https://github.com/tensorflow/docs-l10n/blob/master/site/pt-br/tfjs-website/tutorials/deployment/size_optimized_bundles.md).

Este recurso é destinado à implantação no navegador. Porém, o suporte a essa funcionalidade motivou algumas das mudanças descritas abaixo.

### Código ES2017

Além de alguns conjuntos pré-compilados, **a principal maneira de enviarmos código ao NPM agora é como [Módulos ES](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Modules) com [sintaxe ES2017](https://2ality.com/2016/02/ecmascript-2017.html)**, o que permite aos desenvolvedores usar [recursos modernos do JavaScript](https://web.dev/publish-modern-javascript/) e ter um controle maior do que enviam aos usuários finais.

Nossa entrada `module` do package.json aponta para arquivos individuais da biblioteca no formato ES2017 (ou seja, não é um pacote), o que permite tree shaking e um maior controle do desenvolvedor sobre a transpilação downstream.

Oferecemos alguns formatos alternativos como pacotes pré-compilados para dar suporte a navegadores antigos e outros sistemas modulares. Eles seguem a convenção de nomenclatura descrita na tabela abaixo e podem ser carregados a partir de CDNs populares, como JsDelivr e Unpkg.

<table>
  <tr>
   <td>Nome do arquivo</td>
   <td>Formato do módulo</td>
   <td>Versão da linguagem</td>
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
   <td>ESM (arquivo simples único)</td>
   <td>ES2017</td>
  </tr>
  <tr>
   <td>index.js***</td>
   <td>ESM</td>
   <td>ES2017</td>
  </tr>
</table>

* [package] refere-se a nomes como core/converter/layers de subpacotes do pacote principal tf.js. [min] indica se fornecemos arquivos minificados além dos arquivos não minificados.

** A entrada `main` do package.json aponta para este arquivo.

*** A entrada`module` do package.json aponta para este arquivo.

Se você estiver usando o tensorflow.js via npm e estiver usando o bundler, talvez precise ajustar a configuração do bundler para garantir que ele possa consumir os módulos ES2017 ou apontar para uma das outras entradas em package.json.

### @tensorflow/tfjs-core é mais leve por padrão

Para permitir um [tree-shaking](https://developers.google.com/web/fundamentals/performance/optimizing-javascript/tree-shaking) melhor, por padrão não incluímos mais a API chaining/fluent nos tensores em @tensorflow/tfjs-core. Recomendamos usar operações (ops) diretamente para obter o menor pacote. Fornecemos a importação `import '@tensorflow/tfjs-core/dist/public/chained_ops/register_all_chained_ops';` que restaura a API chaining.

Também não registramos mais os gradientes para os kernels por padrão. Se você quiser suporte a gradiente/treinamento, pode usar `import '@tensorflow/tfjs-core/dist/register_all_gradients';`

> Observação: se você estiver usando @tensorflow/tfjs, @tensorflow/tfjs-layers ou qualquer outro pacote de nível mais alto, isso é feito automaticamente.

### Reorganização do código, registros de kernel e gradiente

Reorganizamos o código para facilitar a contribuição com operações e kernels e também para implementar operações, kernels e gradientes personalizados. [Confira mais informações neste guia](https://www.tensorflow.org/js/guide/custom_ops_kernels_gradients).

### Alterações interruptivas

Confira a lista completa de alterações interruptivas [aqui](https://github.com/tensorflow/tfjs/releases), que incluem a remoção de *Operações estritas, como mulStrict ou addStrict.

## Como fazer upgrade de código 2.x

### Usuários de @tensorflow/tfjs

Resolva todas as alterações interruptivas descritas aqui (https://github.com/tensorflow/tfjs/releases)

### Usuários de @tensorflow/tfjs-core

Resolva todas as alterações interruptivas descritas aqui (https://github.com/tensorflow/tfjs/releases) e depois faça o seguinte:

#### Adicione ampliadores de operações encadeadas ou use operações diretamente

Em vez de:

```
import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-webgl';

const a = tf.tensor([1,2,3,4]);
const b = a.sum(); // this is a 'chained' op.
```

Você precisa fazer o seguinte:

```
import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-webgl';
import '@tensorflow/tfjs-core/dist/public/chained_ops/sum'; // add the 'sum' chained op to all tensors

const a = tf.tensor([1,2,3,4]);
const b = a.sum();
```

Você também pode importar todas as APIs chaining/fluent com a seguinte importação:

```
import '@tensorflow/tfjs-core/dist/public/chained_ops/register_all_chained_ops';
```

Outra opção é usar a operação diretamente (você pode usar as importações nomeadas aqui também)

```
import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-webgl';

const a = tf.tensor([1,2,3,4]);
const b = tf.sum(a);
```

#### Importe o código de inicialização

Se você estiver usando exclusivamente importações nomeadas (em vez de `import * as ...`), então, em alguns casos, talvez você precise fazer o seguinte:

```
import @tensorflow/tfjs-core
```

perto da parte superior do programa, o que impede que tree-shakers agressivos eliminem qualquer inicialização necessária.

## Como fazer upgrade de código 1.x

### Usuários de @tensorflow/tfjs

Resolva todas as alterações interruptivas indicadas [aqui](https://github.com/tensorflow/tfjs/releases/tag/tfjs-v2.0.0). Depois, siga as instruções de upgrade de código 2.x.

### Usuários de @tensorflow/tfjs-core

Resolva todas as alterações interruptivas indicadas [aqui](https://github.com/tensorflow/tfjs/releases/tag/tfjs-v2.0.0), selecione um backend conforme descrito abaixo e depois siga as etapas de upgrade de código 2.x.

#### Como selecionar backends

No TensorFlow.js 2.0, removemos os backends CPU e Webgl e os colocamos em seus próprios pacotes. Confira as instruções de como incluir esses backends em [@tensorflow/tfjs-backend-cpu](https://www.npmjs.com/package/@tensorflow/tfjs-backend-cpu), [@tensorflow/tfjs-backend-webgl](https://www.npmjs.com/package/@tensorflow/tfjs-backend-webgl), [@tensorflow/tfjs-backend-wasm](https://www.npmjs.com/package/@tensorflow/tfjs-backend-wasm) e [@tensorflow/tfjs-backend-webgpu](https://www.npmjs.com/package/@tensorflow/tfjs-backend-webgpu).
