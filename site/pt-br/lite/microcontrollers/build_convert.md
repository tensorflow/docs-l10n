# Compile e converta modelos

Os microcontroladores têm RAM e armazenamento limitados, o que restringe o tamanho dos modelos de aprendizado de máquina. Além disso, no momento, o TensorFlow Lite para Microcontroladores oferece suporte a um subconjunto limitado de operações e, portanto, nem todas as arquiteturas de modelo são possíveis.

Este documento explica o processo de converter um modelo do TensorFlow para ser executado em microcontroladores, além de descrever as operações com suporte e oferecer orientações sobre como conceber e treinar um modelo de forma que caiba na memória limitada.

Para ver um exemplo executável completo de compilação e conversão de um modelo, confira o seguinte Colab, que faz parte do exemplo *Hello World* (Olá, mundo):

<a class="button button-primary" href="https://colab.research.google.com/github/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/examples/hello_world/train/train_hello_world_model.ipynb">train_hello_world_model.ipynb</a>

## Conversão do modelo

Para converter um modelo do TensorFlow de forma que possa ser executado em microcontroladores, você deve usar a [API do Python de conversão para o TensorFlow Lite](https://www.tensorflow.org/lite/models/convert/), que vai converter o modelo em um [`FlatBuffer`](https://google.github.io/flatbuffers/), reduzindo o tamanho do modelo, e modificá-lo para usar as operações do TensorFlow Lite.

Para conseguir o menor tamanho possível para o modelo, considere usar [quantização pós-treinamento](https://www.tensorflow.org/lite/performance/post_training_quantization).

### Converta em um array do C

Diversas plataformas de microcontroladores não têm suporte a sistemas de arquivo nativos. A maneira mais fácil de usar um modelo do seu programa é incluí-lo como um array do C e compilá-lo em seu programa.

O comando UNIX abaixo gera um arquivo fonte do C que contém o modelo do TensorFlow Lite como um array `char`:

```bash
xxd -i converted_model.tflite > model_data.cc
```

A saída será parecida com a seguinte:

```c
unsigned char converted_model_tflite[] = {
  0x18, 0x00, 0x00, 0x00, 0x54, 0x46, 0x4c, 0x33, 0x00, 0x00, 0x0e, 0x00,
  // <Lines omitted>
};
unsigned int converted_model_tflite_len = 18200;
```

Após gerar o arquivo, você pode incluí-lo em seu programa. É importante alterar a declaração do array para `const` a fim de conseguir uma eficiência de memória melhor em plataformas embarcadas.

Para ver um exemplo de como incluir e usar um modelo em seu programa, confira [`evaluate_test.cc`](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/examples/hello_world/evaluate_test.cc) no exemplo *Hello World*.

## Arquitetura do modelo e treinamento

Ao conceber um modelo para uso em microcontroladores, é importante considerar o tamanho do modelo, a carga de trabalho e as operações que serão usadas.

### Tamanho do modelo

O modelo precisa ser pequeno o suficiente para caber na memória do dispositivo escolhido juntamente com o restante do programa, tanto como binário quanto no runtime.

Para criar um modelo menor, você pode usar menos camadas e camadas menores em sua arquitetura. Porém, modelos menores estão mais propensos a sofrer underfitting. Portanto, para muitos problemas, faz sentido tentar usar o maior modelo possível que caiba na memória. Entretanto, usar modelos maiores também levará a um aumento da carga de trabalho dos processadores.

Observação: o runtime core do TensorFlow Lite para Microcontroladores cabe em 16 KB em um Cortex M3.

### Carga de trabalho

O tamanho e a complexidade do modelo impactam a carga de trabalho. Modelos maiores e mais complexos resultam em um ciclo de trabalho maior e, portanto, o processador do dispositivo passa mais tempo trabalhando e menos tempo ocioso, o que aumenta o consumo de energia e a geração de calor, o que pode ser um problema, dependendo da aplicação.

### Suporte a operações

No momento, o TensorFlow Lite para Microcontroladores tem suporte a um subconjunto limitado de operações do TensorFlow, impactando as possíveis arquiteturas do modelo que podem ser usadas. Estamos trabalhando na expansão do suporte às operações, tanto em termos das implementações de referência quanto otimizações para arquiteturas específicas.

Confira as operações com suporte no arquivo [`micro_mutable_ops_resolver.h`](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/micro_mutable_op_resolver.h).
