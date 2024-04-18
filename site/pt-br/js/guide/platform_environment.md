# Plataforma e ambiente

O TensorFlow.js funciona no navegador e no Node.js, e, em ambas as plataformas, há várias configurações diferentes disponíveis. Cada plataforma tem um conjunto único de considerações que afetarão a forma como as aplicações são desenvolvidas.

No navegador, o TensorFlow.js tem suporte a dispositivos móveis bem como a computadores. Cada dispositivo tem um conjunto específico de restrições, como as APIs do WebGL disponíveis, que são determinadas automaticamente e configuradas para você.

No Node.js, o TensorFlow.js oferece suporte à conexão diretamente à API do TensorFlow ou à execução mais lenta com implementações vanilla de CPU.

## Ambientes

Quando um programa do TensorFlow.js é executado, a configuração específica é chamada de ambiente. O ambiente é composto por um único back-end global, bem como por um conjunto de sinalizadores que controlam os recursos finos do TensorFlow.js.

### Back-ends

O TensorFlow.js tem suporte a diversos back-ends diferentes que implementam armazenamento de tensores e operações matemáticas. Em um dado momento qualquer, somente um back-end está ativo. Na maior parte do tempo, o TensorFlow.js escolherá automaticamente o melhor back-end para você dado o ambiente atual. Entretanto, às vezes é importante saber qual back-end está sendo usado e como alterá-lo.

Para descobrir qual back-end você está usando:

```js
console.log(tf.getBackend());
```

Se você quiser alterar o back-end manualmente:

```js
tf.setBackend('cpu');
console.log(tf.getBackend());
```

#### Back-end WebGL

O back-end WebGL, 'webgl', é atualmente o back-end mais poderoso para navegadores. Ele é até 100 vezes mais rápido do que o back-end vanilla de CPU. Os tensores são armazenados como texturas do WebGL, e as operações matemáticas são implementadas como shaders do WebGL. Veja alguns aspectos interessantes que você deve saber ao usar este back-end:  \

##### Evite bloquear o thread de interface gráfica

Quando uma operação é chamada, como tf.matMul(a, b), o tf.Tensor resultante é retornado de maneira síncrona. Entretanto, talvez a computação da multiplicação de matrizes ainda não esteja pronta. Portanto, o tf.Tensor retornado é apenas um identificador da computação. Quando você chamar `x.data()` ou `x.array()`, os valores serão resolvidos quando a computação for concluída. Por isso, é importante usar os métodos assíncronos `x.data()` e `x.array()` em vez das suas contrapartes síncronas `x.dataSync()` e `x.arraySync()` para evitar bloquear o thread de interface gráfica até a computação ser concluída.

##### Gerenciamento de memória

Uma ressalva ao usar o back-end WebGL é a necessidade de gerenciamento explícito de memória. O WebGLTextures, o local onde os dados de tensores são armazenados, não é coletado automaticamente como lixo pelo navegador.

Para destruir a memória de um `tf.Tensor`, você pode usar o método `dispose()`:

```js
const a = tf.tensor([[1, 2], [3, 4]]);
a.dispose();
```

É muito comum encadear diversas operações em uma aplicação. Armazenar uma referência a todas as variáveis intermediárias para descartá-las depois pode reduzir a legibilidade do código. Para resolver esse problema, o TensorFlow.js conta com o método `tf.tidy()`, que elimina todos os `tf.Tensor`s que não são retornados por uma função após a execução, similar à forma como as variáveis locais são eliminadas quando uma função é executada:

```js
const a = tf.tensor([[1, 2], [3, 4]]);
const y = tf.tidy(() => {
  const result = a.square().log().neg();
  return result;
});
```

> Observação: não existe uma desvantagem de se usar `dispose()` ou `tidy()` em ambientes não WebGL (como Node.js ou back-end de CPU) que tenham coleta automática de lixo. De fato, geralmente há ganhos de desempenho ao liberar a memória de tensores mais rápido do que ocorreria naturalmente com a coleta de lixo.

##### Precisão

Em dispositivos móveis, o WebGL pode ter suporte apenas a texturas de ponto flutuante de 16 bits. Porém, a maioria dos modelos de aprendizado de máquina são treinados usando-se ativações e pesos de ponto flutuante de 32 bits, o que pode causar problemas de precisão ao fazer a portabilidade de um modelo para dispositivos móveis, pois os números flutuantes de 16 bits podem representar somente números no intervalo `[0.000000059605, 65504]`. Portanto, você deve ter cuidado para os pesos e as ativações do seu modelo não excederem esse intervalo. Para verificar se o dispositivo tem suporte a texturas de 32 bits, confira o valor de `tf.ENV.getBool('WEBGL_RENDER_FLOAT32_CAPABLE')`. Se o valor for false, então o dispositivo tem suporte somente a texturas de ponto flutuante de 16 bits. Você pode usar `tf.ENV.getBool('WEBGL_RENDER_FLOAT32_ENABLED')` para verificar se o TensorFlow.js está usando atualmente texturas de 32 bits.

##### Compilação de shaders e uploads de texturas

O TensorFlow.js executa operações na GPU executando programas de shader do WebGL. Esses shaders são montados e compilados de maneira lazy quando o usuário solicita a execução de uma operação. A compilação de um shader ocorre na CPU do thread principal e pode ser lenta. O TensorFlow.js fará o cache dos shaders compilados automaticamente, fazendo a segunda chamada à mesma operação com tensores de entrada e saída com o mesmo formato de forma muito mais rápida. Tipicamente, aplicações do TensorFlow.js usarão as mesmas operações diversas vezes durante o ciclo de vida da aplicação, então o segundo passo de um modelo de aprendizado de máquina é bem mais rápido.

O TensorFlow.js também armazena dados do tf.Tensor como WebGLTextures. Quando um `tf.Tensor` é criado, não fazemos o upload dos dados imediatamente para a GPU. Em vez disso, mantemos os dados na CPU até que o `tf.Tensor` seja usado em uma operação. Se o `tf.Tensor` for usado uma segunda vez, os dados já estarão na GPU, então não há custo de upload. Em um modelo típico de aprendizado de máquina, isso significa que é feito upload dos pesos durante a previsão do modelo, e o segundo passo do modelo será muito mais rápido.

Se o desempenho da primeira previsão do modelo ou o código do TensorFlow.js são importantes para você, recomendamos fazer uma inicialização do modelo passando um tensor de entrada com o mesmo formato antes de usar dados reais.

Por exemplo:

```js
const model = await tf.loadLayersModel(modelUrl);

// Aqueça o modelo antes de usar dados reais.
const warmupResult = model.predict(tf.zeros(inputShape));
warmupResult.dataSync();
warmupResult.dispose();

// O segundo predict() será muito mais rápido.
const result = model.predict(userData);
```

#### Back-end TensorFlow Node.js

No back-end TensorFlow Node.js, 'node', a API C do TensorFlow é usada para acelerar as operações. É utilizada a aceleração de hardware que estiver disponível na máquina, como CUDA.

Nesse back-end, assim como no back-end WebGL, as operações retornam `tf.Tensor`s de maneira síncrona. Entretanto, diferentemente do back-end WebGL, a operação é concluída antes que você receba o tensor de volta. Portanto, uma chamada a `tf.matMul(a, b)` bloqueará o thread de interface gráfica.

Por esse motivo, se você pretende usá-lo em uma aplicação de produção, deve executar o TensorFlow.js em threads worker que não bloqueiem o thread principal.

Confira mais informações sobre o Node.js neste guia.

#### Back-end WASM

O TensorFlow.js conta com um [back-end WebAssembly](https://github.com/tensorflow/tfjs/blob/master/tfjs-backend-wasm/README.md)  (`wasm`), que oferece aceleração de CPU e pode ser usado como alternativa aos back-ends de CPU vanilla JavaScript (`cpu`) e WebGL acelerado (`webgl`). Para utilizá-lo:

```js
// Set the backend to WASM and wait for the module to be ready.
tf.setBackend('wasm');
tf.ready().then(() => {...});
```

Se o seu servidor colocar o arquivo de produção `.wasm` em um caminho diferente ou nome diferente, use `setWasmPath` antes de inicializar o back-end. Confira mais informações na seção ["Using Bundlers"](https://github.com/tensorflow/tfjs/tree/master/tfjs-backend-wasm#using-bundlers) no arquivo README.

```js
import {setWasmPath} from '@tensorflow/tfjs-backend-wasm';
setWasmPath(yourCustomPath);
tf.setBackend('wasm');
tf.ready().then(() => {...});
```

> Observação: o TensorFlow.js define uma prioridade para cada back-end e escolherá automaticamente o melhor back-end compatível com um dado ambiente. Para usar explicitamente o back-end WASM, precisamos fazer a chamada `tf.setBackend('wasm')`.

##### Por que usar o WASM?

O [WASM](https://webassembly.org/) foi lançado em 2015 como um novo formato de binário baseado na web, fornecendo programas escritos em JavaScript, C, C++, etc. com um alvo de compilação para executar na web. O WASM é [compatível](https://webassembly.org/roadmap/) com o Chrome, Safari, Firefox e Edge desde 2017, e também é compatível com [90% dos dispositivos](https://caniuse.com/#feat=wasm) mundialmente.

**Desempenho**

O back-end WASM usa a [biblioteca XNNPACK](https://github.com/google/XNNPACK) para implementação otimizada de operadores de redes neurais.

*Comparação com o JavaScript*: os binários do WASM geralmente têm um carregamento, processamento e execução mais rápidos do que os pacotes JavaScript no navegador. O JavaScript tem tipagem dinâmica e conta com coleta de lixo, o que pode causar lentidões em tempo de execução.

*Comparação com o WebGL*: o WebGL é mais rápido do que o WASM para a maioria dos modelos. Porém, para modelos muito pequenos, o WASM pode ter desempenho superior ao WebGL devido aos custos de sobrecarga fixos de execução dos shaders do WebGL. A seção “Quando devo usar o WASM?” abaixo discute a heurística para tomar essa decisão.

**Portabilidade e estabilidade**

O WASM tem uma aritmética de ponto flutuante de 32 bits portável, oferecendo paridade de precisão entre todos os dispositivos. Por outro lado, o WebGL é específico ao hardware, e diferentes dispositivos podem ter precisão variada (como o fallback para pontos flutuantes de 16 bits em dispositivos iOS).

Assim como o WebGL, o WASM tem suporte oficial de todos os grandes navegadores. Diferentemente do WebGL, o WASM pode ser executado no Node.js e pode ser usado no lado do servidor sem qualquer necessidade de compilar bibliotecas nativas.

##### Quando devo usar o WASM?

**Tamanho do modelo e demanda computacional**

De forma geral, o WASM é uma boa opção quando os modelos são menores ou quando dispositivos inferiores sem suporte ao WebGL (extensão `OES_texture_float`) são importantes para você, ou quando eles têm GPUs menos poderosas. A tabela abaixo mostra o tempo de inferência (no TensorFlow.js 1.5.2) no Chrome em um 2018 MacBook Pro para 5 dos nossos [modelos](https://github.com/tensorflow/tfjs-models) com suporte oficial nos back-ends WebGL, WASM e CPU:

**Modelos menores**

Modelo | WebGL | WASM | CPU | Memória
--- | --- | --- | --- | ---
BlazeFace | 22,5 ms | 15,6 ms | 315,2 ms | 0,4 MB
FaceMesh | 19,3 ms | 19,2 ms | 335 ms | 2,8 MB

**Modelos maiores**

Modelo | WebGL | WASM | CPU | Memória
--- | --- | --- | --- | ---
PoseNet | 42,5 ms | 173,9 ms | 1.514,7 ms | 4,5 MB
BodyPix | 77 ms | 188,4 ms | 2.683 ms | 4,6 MB
MobileNet v2 | 37 ms | 94 ms | 923,6 ms | 13 MB

A tabela acima mostra que o WASM é de 10 a 30 vezes mais rápido do que o back-end de CPU no JS entre os modelos, e é competitivo com o WebGL para modelos menores, como [BlazeFace](https://github.com/tensorflow/tfjs-models/tree/master/blazeface), que é leve (400 KB), mas ainda tem um número razoável de operações (cerca de 140). Como os programas do WebGL têm um custo de sobrecarga fixo por execução de operação, isso explica por que modelos como o BlazeFace são mais rápidos no WASM.

**Esses resultados variam dependendo do dispositivo. A melhor forma de determinar se o WASM é a opção certa para sua aplicação e testá-la em diferentes back-ends.**

##### Inferência versus treinamento

Para tratar o caso de uso principal para implantação de modelos pré-treinados, o desenvolvimento do back-end WASM priorizará suporte à *inferência* em detrimento do *treinamento*. Confira uma [lista atualizada](https://github.com/tensorflow/tfjs/blob/master/tfjs-backend-wasm/src/register_all_kernels.ts) das operações com suporte no WASM e [nos avise](https://github.com/tensorflow/tfjs/issues?q=is%3Aissue+is%3Aopen+sort%3Aupdated-desc) se o seu modelo tiver uma operação sem suporte. Para modelos de treinamento, recomendamos usar o back-end Node (TensorFlow C++) ou o back-end WebGL.

#### Back-end de CPU

O back-end de CPU, 'cpu', é o que tem o pior desempenho, mas é o mais simples de todos. Todas as operações são implementadas no JavaScript vanilla, o que as torna menos paralelizáveis. Além disso, elas bloqueiam o thread de interface gráfica.

Esse back-end pode ser muito útil para testes ou em dispositivos nos quais o WebGL não estiver disponível.

### Sinalizadores

O TensorFlow.js tem um conjunto de sinalizadores de ambiente que são avaliados automaticamente e determinam a melhor configuração na plataforma atual. Esses sinalizadores são principalmente internos, mas alguns sinalizadores globais podem ser controlados por meio da API pública.

- `tf.enableProdMode()` – ativa o modo de produção, que remove a validação do modelo, as verificações de NaN e outras checagens para melhorar o desempenho.
- `tf.enableDebugMode()` – ativa o modo de depuração, que grava no console cada operação executada, bem como as informações de desempenho do runtime, como pegada de memória e tempo total de execução dos kernels. Observe que isso reduz bastante a velocidade da aplicação e não deve ser usado em produção.

> Observação: esses dois métodos devem ser usados antes de utilizar qualquer código do TensorFlow.js, pois eles afetam os valores de outros sinalizadores que serão armazenados em cache. Por esse mesmo motivo, não existe uma função análoga "disable" para desativá-los.

> Observação: você pode ver todos os sinalizadores que foram avaliados gravando `tf.ENV.features` no console. Embora eles **não façam parte da API pública** (e, portanto, não há garantias de estabilidade entre as versões), podem ser úteis para fazer a depuração ou os ajustes finos entre plataformas e dispositivos. Você pode usar `tf.ENV.set` para sobrescrever o valor de um sinalizador.
