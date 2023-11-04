# Práticas recomendadas de desempenho

Dispositivos móveis e embarcados têm recursos computacionais limitados, então é importante manter a eficiência de recursos do seu aplicativo. Reunimos uma lista de práticas recomendadas e estratégias que você pode usar para melhorar o desempenho do seu modelo do TensorFlow Lite.

## Escolha o melhor modelo para a tarefa

Dependendo da tarefa, você precisará fazer um trade-off entre a complexidade e o tamanho do modelo. Caso sua tarefa exija alta exatidão, talvez seja necessário um modelo maior e complexo. Para tarefas que exigem menos exatidão, é melhor usar um modelo menor, já que ocupa menos espaço em disco e memória, além de ser mais rápido e ter mais eficiência energética. Por exemplo, os grafos abaixo mostram trade-offs de exatidão e latência para alguns modelos comuns de classificação de imagens.

![Grafo de tamanho x precisão do modelo](../images/performance/model_size_vs_accuracy.png "Model Size vs Accuracy")

![Grafo de exatidão x latência](../images/performance/accuracy_vs_latency.png "Accuracy vs Latency")

Um exemplo de modelo otimizado para dispositivos móveis são os [MobileNets](https://arxiv.org/abs/1704.04861), específicos para aplicativos de visão. O [TensorFlow Hub](https://tfhub.dev/s?deployment-format=lite) lista vários outros modelos que foram otimizados especialmente para dispositivos móveis e embarcados.

Você pode treinar novamente os modelos listados com seu próprio dataset ao usar o aprendizado por transferência. Confira os tutoriais de aprendizado por transferência usando o [Model Maker](../models/modify/model_maker/) do TensorFlow Lite.

## Analise o perfil do seu modelo

Depois de selecionar uma opção de modelo ideal para sua tarefa, é recomendável analisar o perfil e fazer o benchmarking do modelo. A [ferramenta de benchmarking](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark) do TensorFlow Lite tem um profiler integrado que mostra estatísticas de profiling por operador. Isso pode ajudar a entender gargalos de desempenho e saber quais operadores dominam o tempo computacional.

Você também pode usar o [tracing do TensorFlow Lite](measurement#trace_tensorflow_lite_internals_in_android) para analisar o perfil do modelo no seu aplicativo Android, com o tracing do sistema Android padrão, e para visualizar as invocações de operador por tempo com ferramentas de profiling baseadas na GUI.

## Otimize operadores após analisar o perfil deles no grafo

Se um operador específico aparecer com frequência no modelo e, com base no profiling, você achar que esse operador consome a maior parte do tempo, considere otimizá-lo. Essa situação deve ser rara, porque o TensorFlow Lite tem versões otimizadas para a maioria dos operadores. No entanto, talvez você consiga escrever uma versão mais rápida de uma operação personalizada se souber as restrições da execução do operador. Consulte o [guia de operadores personalizados](../guide/ops_custom).

## Otimize seu modelo

A otimização do modelo visa criar modelos menores, que geralmente são mais rápidos e têm mais eficiência energética, para que sejam implantados em dispositivos móveis. O TensorFlow Lite é compatível com várias técnicas de otimização, como quantização.

Confira mais detalhes na [documentação sobre a otimização de modelo](model_optimization).

## Ajuste o número de threads

O TensorFlow Lite aceita kernels de vários threads para vários operadores. Você pode aumentar o número de threads e acelerar a execução dos operadores. No entanto, ao aumentar o número de threads, o modelo usará mais recursos e energia.

Para alguns aplicativos, a latência pode ser mais importante do que a eficiência energética. Você pode aumentar o número de threads ao configurar o número de [threads](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/interpreter.h#L346) do interpretador. Porém, a desvantagem da execução de vários threads é que ela aumenta a variabilidade do desempenho dependendo do que está sendo executado ao mesmo tempo. Esse é o caso especialmente para aplicativos móveis. Por exemplo, testes isolados podem mostrar o dobro de velocidade em comparação com um único thread, mas, se outro aplicativo estiver sendo executado simultaneamente, pode resultar em um desempenho inferior a um único thread.

## Elimine cópias redundantes

Se o seu aplicativo não for projetado com cuidado, pode haver cópias redundantes ao alimentar a entrada e ler a saída do modelo. Não deixe de eliminar as cópias redundantes. Se você estiver usando APIs de nível superior, como Java, verifique cuidadosamente as ressalvas de desempenho na documentação. Por exemplo, a API Java é muito mais rápida ao usar `ByteBuffers` como [entradas](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/java/src/main/java/org/tensorflow/lite/Interpreter.java#L175).

## Analise o perfil do seu aplicativo com ferramentas da plataforma

As ferramentas específicas da plataforma, como [Android profiler](https://developer.android.com/studio/profile/android-profiler) e [Instruments](https://help.apple.com/instruments/mac/current/), fornecem uma abundância de informações de profiling que podem ser usadas para depurar seu aplicativo. Às vezes, o bug de desempenho pode não estar no modelo, e sim em partes do código do aplicativo que interagem com o modelo. Familiarize-se com as ferramentas de profiling específicas e as práticas recomendadas para sua plataforma.

## Avalie se o modelo pode obter proveito de aceleradores de hardware disponíveis no dispositivo

O TensorFlow Lite adicionou novas maneiras de acelerar modelos com hardware mais rápido, como GPUs, DSPs e aceleradores neurais. Geralmente, esses aceleradores são expostos por submódulos de [delegados](delegates) que assumem partes da execução do interpretador. O TensorFlow Lite pode usar delegados:

- Ao utilizar a [API Neural Networks](https://developer.android.com/ndk/guides/neuralnetworks/) do Android. Você pode usar esses back-ends de acelerador de hardware para melhorar a velocidade e eficiência do seu modelo. Para habilitar a API Neural Networks, consulte o guia [delegado NNAPI](https://www.tensorflow.org/lite/android/delegates/nnapi).
- O delegado de GPU está disponível no Android e no iOS, usando OpenGL/OpenCL e Metal, respectivamente. Para testar, confira o [tutorial](gpu) e a [documentação sobre o delegado de GPU](gpu_advanced).
- O delegado Hexagon está disponível no Android. Ele aproveita o DSP Qualcomm Hexagon, caso esteja disponível no dispositivo. Veja mais informações no [tutorial do delegado Hexagon](https://www.tensorflow.org/lite/android/delegates/hexagon).
- É possível criar seu próprio delegado se você tiver acesso a hardware não padrão. Saiba mais em [delegados do TensorFlow Lite](delegates).

Tenha em mente que alguns aceleradores funcionam melhor para diferentes tipos de modelos. Alguns delegados só aceitam modelos float ou otimizados de maneira específica. É importante fazer o [benchmarking](measurement) de cada delegado para ver se é uma boa escolha para seu aplicativo. Por exemplo, se você tiver um modelo muito pequeno, pode não valer a pena delegar o modelo à API NN ou GPU. Por outro lado, os aceleradores são uma ótima opção para modelos grandes com uma alta intensidade aritmética.

## Precisa de mais ajuda?

A equipe do TensorFlow fica feliz em ajudar a diagnosticar e resolver os problemas específicos de desempenho que você estiver enfrentando. Crie um issue no [GitHub](https://github.com/tensorflow/tensorflow/issues) com detalhes sobre o problema.
