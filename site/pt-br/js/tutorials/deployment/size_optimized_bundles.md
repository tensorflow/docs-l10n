# Como gerar pacotes de navegador com tamanho otimizado usando TensorFlow.js

## Visão geral

O TensorFlow.js 3.0 oferece suporte à criação de *pacotes de navegador para produção com tamanho otimizado*. Dizendo de outra forma, queremos que seja mais fácil para você enviar menos JavaScript para o navegador.

O recurso é destinado a usuários com casos de uso em produção que se beneficiariam da remoção de alguns bytes do conteúdo (e, portanto, estão dispostos a empreender esforços para atingir esse objetivo). Para usar esse recurso, você já deve conhecer os [módulos ES](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Modules), as ferramentas de empacotamento JavaScrjpt, como [webpack](https://webpack.js.org/) ou [rollup](https://rollupjs.org/guide/en/), além de conceitos como [tree-shaking/eliminação de código morto](https://developers.google.com/web/fundamentals/performance/optimizing-javascript/tree-shaking).

Este tutorial demonstra como criar um módulo tensorflow.js personalizado que possa ser usado com um bundler para gerar uma build de um programa com tamanho otimizado usando tensorflow.js.

### Terminologia

No contexto deste documento, usaremos alguns termos chave:

**[Módulos ES](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Modules)** – **Sistema padrão de módulos JavaScript**. Lançado no ES6/ES2015. Pode ser identificação pelo uso das declarações **import** e **export**.

**Empacotamento** – Significa reunir um conjunto de ativos JavaScript e agrupá-los/empacotá-los em um ou mais ativos JavaScript que podem ser usados em um navegador. Essa é a etapa que costuma gerar os ativos finais enviados ao navegador. ***Geralmente, as aplicações fazem seu próprio empacotamento diretamente a partir de fontes de biblioteca transpiladas*.** Exemplos comuns de **bundlers**: *rollup* e *webpack*. O resultado final do empacotamento é conhecido como **pacote** (ou às vezes como **parte**, caso ele seja dividido em diversas partes).

**[Tree-shaking/Eliminação de código morto](https://developers.google.com/web/fundamentals/performance/optimizing-javascript/tree-shaking)** – Remoção de código que não é usado pela aplicação escrita final. Isso é feito durante o empacotamento, *geralmente* na etapa de minificação.

**Operações (ops)** – Operação matemática feita em um ou mais tensores que produz um ou mais tensores como saída. As operações são código de "alto nível" e podem usar outras operações para definirem sua lógica.

**Kernel** – Implementação específica de uma operação ligada a capacidades de hardwares específicos. Os kernels são de "baixo nível" e específicos de backends. Algumas operações têm um mapeamento de operação para kernel um para um, enquanto outras operações utilizam diversos kernels.

## Escopo e casos de uso

### Modelos de grafo somente para inferência

O principal caso de uso relatado por usuários está relacionado a isso. Nesta versão, temos suporte a **inferência com *modelos de grafo de TensorFlow.js***. Se você estiver usando um *modelo de camadas de TensorFlow.js*, pode convertê-lo para o formato de modelo de grafo usando [tfjs-converter](https://www.npmjs.com/package/@tensorflow/tfjs-converter). O formato de modelo de grafo é mais eficiente para inferência.

### Manipulação de Tensor de baixo nível com tfjs-core

O outro caso de uso que oferecemos são programas que usam diretamente o pacote @tensorflow/tjfs-core para fazer manipulação de tensor de baixo nível.

## Nossa estratégia para criar builds

Veja os principais princípios ao desenvolver essa funcionalidade:

- Fazer um uso mínimo do sistema de módulos JavaScript (ESM, na sigla em inglês) e permitir que os usuários de TensorFlow.js façam o mesmo.
- Permitir o maior nível possível de tree-shaking para TensorFlow.js*{nbsp}por meio dos bundlers existentes* (por exemplo, webpack, rollup, etc). Dessa forma, os usuários podem usar todas as funcionalidades desses bundlers, incluindo recursos como divisão de código.
- Na medida do possível, manter *a facilidade de uso para usuários que não tenham tanto problema com o tamanho dos pacotes*. Isso significa que gerar builds de produção vai exigir mais esforços, pois diversos dos padrões em nossas bibliotecas têm suporte à facilidade de uso em detrimento de builds com tamanho otimizado.

O principal objetivo do nosso workflow é gerar um *módulo JavaScript* personalizado para TensorFlow.js que contenha somente a funcionalidade necessária para o programa que estamos tentando otimizar. Contamos com os bundlers existentes para fazer a otimização em si.

Embora contemos principalmente com o sistema de módulos JavaScript, também fornecemos uma *ferramenta CLI* *personalizada* para lidar com as partes que não são fáceis de especificar por meio do sistema de módulos no código voltado para o usuário. Confira dois exemplos:

- Especificações de modelos armazenadas em arquivos `model.json`.
- Operação do sistema de envio de kernel específico de backends que usamos.

Dessa forma, gerar uma build tfjs personalizada é um pouco mais complicado do que simplesmente apontar um bundler para o pacote @tensorflow/tfjs comum.

## Como criar pacotes personalizados com tamanho otimizado

### Etapa 1 – Determine quais kernels seu programa está usando

**Com esta etapa, podemos determinar todos os kernels usados por qualquer modelo que você execute ou código pré-pós-processado dado o backend que você selecionou.**

Use tf.profile para executar as partes da sua aplicação que usem tensorflow.js e obtenha os kernels. Veja como fica:

```
const profileInfo = await tf.profile(() => {
  // You must profile all uses of tf symbols.
  runAllMyTfjsCode();
});

const kernelNames = profileInfo.kernelNames
console.log(kernelNames);
```

Copie essa lista de kernels para a área de transferência, pois serão necessários na próxima etapa.

> Você precisa fazer o profiling do código usando os mesmos backends que deseja usar no pacote personalizado.

> Você precisará repetir essa etapa se o seu modelo alterar ou se o código de pré-pós-processamento mudar.

### Etapa 2 – Escreva um arquivo de configuração para o módulo tjfs personalizado

Confira um exemplo de arquivo configuração.

Veja como fica:

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

- kernels: lista dos kernels a incluir no pacote. Copie da saída gerada na etapa 1.
- backends: lista dos backends que você deseja incluir. Veja algumas opções válidas:"cpu", "webgl" e “wasm”.
- models: lista de arquivos model.json para modelos que você carrega em sua aplicação. Pode ficar vazia caso o seu programa não use tfjs_converter para carregar um modelo de grafo.
- outputPath: caminho para uma pasta onde serão colocados os módulos gerados.
- forwardModeOnly: defina como false se você quiser incluir gradientes para os kernels listados acima.

### Etapa 3 – Gere o módulo tfjs personalizado

Execute a ferramenta de compilação personalizada usando o arquivo de configuração como argumento. Você precisa ter o pacote **@tensorflow/tfjs** instalado para ter acesso a essa ferramenta.

```
npx tfjs-custom-module  --config custom_tfjs_config.json
```

Será criada uma pasta em `outputPath` com alguns arquivos novos.

### Etapa 4 – Configure seu bundler para fazer um alias de tfjs para o novo módulo personalizado.

Em bundlers como o webpack e o rollup, podemos fazer o alias de referências existentes para módulos tfjs a fim de apontar para os módulos tfjs personalizados recém-gerados. É preciso fazer o alias de três módulos gerados para ter economia máxima no tamanho do pacote.

Veja abaixo um trecho de código para o webpack ([confira o exemplo completo aqui](https://github.com/tensorflow/tfjs/blob/master/e2e/custom_module/dense_model/webpack.config.js)):

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

E aqui está o trecho de código equivalente para o rollup ([confira o exemplo completo aqui](https://github.com/tensorflow/tfjs/blob/master/e2e/custom_module/dense_model/rollup.config.js)):

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

> Caso o seu bundler não tenha suporte ao alias de módulos, você precisará alterar suas declarações `import` para importar tensorflow.js do `custom_tfjs.js` gerado, criado na etapa 3. As definições de operações não passarão por tree-shaking, mas os kernels ainda passarão. Geralmente, fazer o tree-shaking de kernels é o que proporciona a maior economia no tamanho final do pacote.

> Se você estiver usando somente o pacote @tensoflow/tfjs-core, só precisa fazer alias desse único pacote.

### Etapa 5 – Crie o pacote

Execute o bundler (e.g. `webpack` ou `rollup`) para gerar o pacote. O tamanho do pacote deverá ser menor do que se você executar o bundler sem fazer o alias dos módulos. Você também pode usar visualizadores como [este](https://www.npmjs.com/package/rollup-plugin-visualizer) para ver o que ficou no pacote final.

### Etapa 6 – Teste a aplicação

Confirme se a sua aplicação está funcionando conforme o esperado.
