# Como começar com a otimização de modelos do TensorFlow

## 1. Escolha o melhor modelo para a tarefa

Dependendo da tarefa, você precisará fazer um trade-off entre a complexidade e o tamanho do modelo. Caso sua tarefa exija alta exatidão, talvez seja necessário um modelo maior e complexo. Para tarefas que exigem menos exatidão, é melhor usar um modelo menor, já que ocupa menos espaço em disco e memória, além de ser mais rápido e ter mais eficiência energética.

## 2. Modelos pré-otimizados

Veja se algum [modelo pré-otimizado existente do TensorFlow Lite](https://www.tensorflow.org/lite/models) oferece a eficiência exigida pelo seu aplicativo.

## 3. Ferramentas pós-treinamento

Se você não puder usar um modelo pré-treinado para seu aplicativo, tente usar as [ferramentas de quantização pós-treinamento do TensorFlow Lite](./quantization/post_training) durante a [conversão do TensorFlow Lite](https://www.tensorflow.org/lite/convert), que pode otimizar seu modelo do TensorFlow já treinado.

Saiba mais neste [tutorial de quantização pós-treinamento](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/performance/post_training_quant.ipynb).

## Próximas etapas: ferramentas de tempo de treinamento

Se as soluções simples acima não atenderem às suas necessidades, talvez você precise usar técnicas de otimização de tempo de treinamento. [Otimize ainda mais](optimize_further.md) com nossas ferramentas de tempo de treinamento e vá mais a fundo.
