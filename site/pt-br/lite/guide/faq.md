# Perguntas frequentes

Se você não encontrar uma resposta para sua pergunta aqui, confira nossa documentação detalhada sobre o tópico ou crie um [issue no GitHub](https://github.com/tensorflow/tensorflow/issues).

## Conversão de modelos

#### Quais são os formatos disponíveis para conversão do TensorFlow para o TensorFlow Lite?

Os formatos disponíveis estão indicados [aqui](../models/convert/index#python_api).

#### Por que algumas operações não são implementadas no TensorFlow Lite?

Para manter o TF Lite leve, somente alguns operadores do TF (indicados na [lista de permissões](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/guide/op_select_allowlist.md)) têm suporte no TF Lite.

#### Por que meu modelo não está sendo convertido?

Como o número de operações no TensorFlow Lite é menor do que no TensorFlow, não é possível converter alguns modelos. Alguns erros comuns são indicados [aqui](../models/convert/index#conversion-errors).

Para problemas de conversão não relacionados a operações ausentes ou operações de fluxo de controle, confira os [issues do GitHub](https://github.com/tensorflow/tensorflow/issues?q=label%3Acomp%3Alite+) ou crie um [novo](https://github.com/tensorflow/tensorflow/issues).

#### Como faço para testar se um modelo do TensorFlow Lite se comporta da mesma forma que o modelo original do TensorFlow?

A melhor forma de fazer esse teste é comparando as saídas dos modelos do TensorFlow e do TensorFlow Lite dadas as mesmas entradas (dados de teste ou entradas aleatórias), conforme mostrado [aqui](inference#load-and-run-a-model-in-python).

#### Como faço para determinar as entradas/saídas do buffer do protocolo GraphDef?

A forma mais fácil de inspecionar um grafo de um arquivo `.pb` é usando o [Netron](https://github.com/lutzroeder/netron), um visualizador de código aberto para modelos de aprendizado de máquina.

Se o Netron não conseguir abrir o grafo, você pode tentar usar a ferramenta [summarize_graph](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/graph_transforms/README.md#inspecting-graphs).

Se a ferramenta summarize_graph gerar um erro, você pode visualizar o GraphDef no [TensorBoard](https://www.tensorflow.org/guide/summaries_and_tensorboard) e procurar as entradas e saídas no grafo. Para visualizar um arquivo `.pb`, use o script [`import_pb_to_tensorboard.py`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/import_pb_to_tensorboard.py) da seguinte forma:

```shell
python import_pb_to_tensorboard.py --model_dir <model path> --log_dir <log dir path>
```

#### Como faço para inspecionar um arquivo `.tflite`?

O [Netron](https://github.com/lutzroeder/netron) é a forma mais fácil de visualizar um modelo do TensorFlow Lite.

Se o Netron não conseguir abrir seu modelo do TensorFlow Lite, você pode tentar o script [visualize.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/tools/visualize.py) em nosso repositório.

Se você estiver usando o TF 2.5 ou superior:

```shell
python -m tensorflow.lite.tools.visualize model.tflite visualized_model.html
```

Caso contrário, execute este script com o Bazel:

- [Clone o repositório do TensorFlow](https://www.tensorflow.org/install/source)
- Execute o script `visualize.py` com o Bazel:

```shell
bazel run //tensorflow/lite/tools:visualize model.tflite visualized_model.html
```

## Otimização

#### Como faço para reduzir o tamanho do meu modelo convertido para TensorFlow Lite?

É possível usar [quantização pós-treinamento](../performance/post_training_quantization) durante a conversão para o TensorFlow Lite a fim de reduzir o tamanho do modelo. A quantização pós-treinamento quantiza os pesos de ponto flutuante para 8 bits de precisão e desfaz a quantização durante o tempo de execução para fazer computações com pontos flutuantes. Porém, isso pode impactar a exatidão.

Se treinar novamente o modelo for uma opção, considere fazer o [treinamento com reconhecimento de quantização](https://github.com/tensorflow/tensorflow/tree/r1.13/tensorflow/contrib/quantize). Porém, esse tipo de treinamento só está disponível para um subconjunto das arquiteturas de redes neurais convolucionais.

Para compreender melhor os diferentes métodos de otimização, confira [Otimização de modelos](../performance/model_optimization).

#### Como faço para otimizar o desempenho do TensorFlow Lite para minha tarefa de aprendizado de máquina?

Confira o processo geral para otimizar o desempenho do TensorFlow Lite:

- *Confirme se você tem o modelo certo para a tarefa.* Para classificação de imagens, confira o [TensorFlow Hub](https://tfhub.dev/s?deployment-format=lite&module-type=image-classification).
- *Ajuste o número de threads.* Diversos operadores do TensorFlow Lite têm suporte a kernels multithread. Você pode usar `SetNumThreads()` da [API do C++](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/core/interpreter_builder.h#L110) para fazer isso. Porém, aumentar o número de threads faz o desempenho variar, dependendo do ambiente.
- *Use aceleradores de hardware.* O TensorFlow Lite tem suporte à aceleração de modelos para hardwares específicos usando delegados. Confira nosso guia sobre [delegados](../performance/delegates) para ver mais informações sobre quais aceleradores têm suporte e como usá-los em seu modelo nos dispositivos.
- *(Avançado) Profiling de modelos.* A [ferramenta de benchmark](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark) do Tensorflow Lite tem um profiler integrado que mostra estatísticas por operador. Se você souber como otimizar o desempenho de um operador para sua plataforma específica, pode implementar um [operador personalizado](ops_custom).

Confira mais detalhes sobre como otimizar o desempenho nas [Práticas recomendadas](../performance/best_practices).
