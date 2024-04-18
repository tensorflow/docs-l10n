# Conversão de modelos

O TensorFlow.js vem com uma variedade de modelos pré-treinados que estão prontos para uso no navegador – eles podem ser encontrados em nosso [repositório de modelos](https://github.com/tensorflow/tfjs-models). No entanto, você pode ter encontrado ou criado um modelo TensorFlow em outro lugar que gostaria de usar em seu aplicativo web. O TensorFlow.js fornece um modelo [conversor](https://github.com/tensorflow/tfjs/tree/master/tfjs-converter) para essa finalidade. O conversor possui dois componentes:

1. Um utilitário de linha de comando que converte modelos do Keras e TensorFlow para uso no TensorFlow.js.
2. Uma API para carregar e executar o modelo no navegador com o TensorFlow.js.

## Converta seu modelo

O conversor do TensorFlow.js funciona com diversos formatos de modelo diferentes:

**SavedModel** – É o formato padrão no qual os modelos do TensorFlow são salvos. O formato SavedModel está documentado [aqui](https://www.tensorflow.org/guide/saved_model).

**Modelos do Keras** – Os modelos do Keras geralmente são salvos como um arquivo HDF5. Confira mais informações sobre como salvar modelos do Keras [aqui](https://keras.io/getting-started/faq/#savingloading-whole-models-architecture-weights-optimizer-state).

**Módulo do TensorFlow Hub** – São modelos que foram empacotados para distribuição no TensorFlow Hub, uma plataforma para compartilhamento e descoberta de modelos. A biblioteca de modelos está disponível [aqui](https://tfhub.dev/).

Dependendo do tipo de modelo que você está tentando converter, será preciso passar diferentes argumentos ao conversor. Por exemplo: digamos que você tenha salvado um modelo do Keras com o nome `model.h5` no diretório `tmp/`. Para converter o modelo usando o conversor do TensorFlow.js, você pode executar o seguinte comando:

```
$ tensorflowjs_converter --input_format=keras /tmp/model.h5 /tmp/tfjs_model
```

Dessa forma, o modelo em `/tmp/model.h5` será convertido, e será gerado um arquivo `model.json` juntamente com os arquivos binários de pesos no diretório `tmp/tfjs_model/`.

Confira mais detalhes sobre os argumentos da linha de comando correspondentes aos diferentes formatos de modelos no arquivo [README](https://github.com/tensorflow/tfjs/tree/master/tfjs-converter) do conversor do TensorFlow.js.

Durante o processo de conversão, percorremos o grafo do modelo e verificamos se cada operação é compatível com o TensorFlow.js. Em caso afirmativo, escrevemos o grafo em um formato que o navegador consiga consumir. Tentamos otimizar o modelo para que possa ser usado na web por meio da fragmentação dos pesos em arquivos de 4 MB – dessa forma, os navegadores podem fazer o cache deles. Também tentamos simplificar o modelo do grafo em si usando o projeto de código aberto [Grappler](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/core/grappler). As simplificações de grafo incluem a combinação de operações adjacentes, a eliminação de subgrafos comuns, etc. Essas alterações não acarretam efeitos na saída do modelo. Para otimizar ainda mais, os usuários podem passar um argumento que instrui o conversor a quantizar o modelo em um determinado tamanho de bytes. A quantização é uma técnica para reduzir o tamanho do modelo por meio da representação dos pesos com menos bits. Os usuários precisam ter cuidado para garantir que o modelo mantenha um nível de exatidão aceitável após a quantização.

Se identificarmos uma operação incompatível durante a conversão, haverá falha no processo, e exibiremos via print o nome da operação para o usuário. Fique à vontade para criar um issue no [GitHub](https://github.com/tensorflow/tfjs/issues) para nos avisar. Tentamos implementar novas operações em resposta à demanda dos usuários.

### Práticas recomendadas

Embora nos esforcemos para otimizar seu modelo durante a conversão, geralmente a melhor forma de garantir que ele tenha um bom desempenho é criá-lo já tendo em mente ambientes com restrição de recursos. Portanto, é bom evitar arquiteturas complexas demais e minimizar o número de parâmetros (pesos), quando possível.

## Execute o modelo

Ao converter seu modelo com êxito, você terá um conjunto de arquivos de pesos e um arquivo de topologia do modelo. O TensorFlow.js conta com APIs de carregamento de modelos que podem ser usadas para buscar esses arquivos de modelos e executar a inferência no navegador.

Veja a API para um SavedModel do TensorFlow convertido ou para um módulo do TensorFlow Hub:

```js
const model = await tf.loadGraphModel(‘path/to/model.json’);
```

E veja a API para um modelo do Keras convertido:

```js
const model = await tf.loadLayersModel(‘path/to/model.json’);
```

A API `tf.loadGraphModel` retorna um `tf.FrozenModel`, ou seja, os parâmetros são fixos, e você não poderá fazer os ajustes finos do modelo com novos dados. A API `tf.loadLayersModel` retorna um  tf.Model, que pode ser treinado. Confira mais informações sobre como treinar um tf.Model no [guia de treinamento de modelos](train_models.md).

Após a conversão, é uma boa ideia executar a inferência algumas vezes e fazer o benchmarking da velocidade do modelo. Temos uma página de benchmarking independente que pode ser usada para esse propósito: https://tensorflow.github.io/tfjs/e2e/benchmarks/local-benchmark/index.html. Talvez você perceba que descartamos as medidas de uma execução preparatória inicial. Fazemos isso porque, de forma geral, a primeira inferência do modelo será muitas vezes mais lenta do que inferências subsequentes devido à sobrecarga de criação de texturas e compilação de shaders.
