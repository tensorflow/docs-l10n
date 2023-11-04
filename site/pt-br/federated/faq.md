# Perguntas frequentes

## O TensorFlow Federated pode ser usado em ambientes de produção, por exemplo, em telefones celulares?

No momento, não. Embora tenhamos projetado o TFF pensando na implantação em dispositivos reais, neste estágio não fornecemos nenhuma ferramenta para essa finalidade. A versão atual destina-se a usos experimentais, como expressar novos algoritmos federados ou testar o aprendizado federado com seus próprios datasets, usando o runtime de simulação incluído.

Prevemos que, com o tempo, o ecossistema de código aberto em torno do TFF evoluirá para incluir runtimes direcionados a plataformas de implantação física.

## Como uso o TFF para fazer experimentos com grandes datasets?

O runtime padrão incluído na versão inicial do TFF destina-se apenas a pequenos experimentos, como aqueles descritos em nossos tutoriais, nos quais todos os seus dados (em todos os clientes simulados) cabem simultaneamente na memória de uma única máquina e todo o experimento é executado localmente. dentro do notebook colab.

Nosso roteiro futuro de curto prazo inclui um runtime de alto desempenho para experimentos com datasets muito grandes e grandes números de clientes.

## Como posso garantir que a aleatoriedade no TFF corresponda às minhas expectativas?

Já que o TFF tem computação federada incorporada em seu núcleo, o autor do TFF não deve assumir o controle sobre onde e como as `Session` do TensorFlow são inseridas ou `run` é chamada nessas sessões. A semântica da aleatoriedade pode depender da entrada e saída das `Session` do TensorFlow se as sementes forem definidas. Recomendamos usar a aleatoriedade no estilo TensorFlow 2, usando, por exemplo `tf.random.experimental.Generator` a partir do TF 1.14. Isto usa um `tf.Variable` para gerenciar seu estado interno.

Para ajudar a gerenciar as expectativas, o TFF permite que o TensorFlow que ele serializa tenha sementes de nível operacional definidas, mas não sementes de nível de grafo. Isto ocorre porque a semântica das sementes no nível da operação deve ser mais clara na configuração do TFF: uma sequência determinística será gerada em cada chamada de uma função empacotada como `tf_computation`, e somente dentro desta chamada quaisquer garantias feitas pelo gerador de números pseudoaleatórios serão válidas. Observe que isto não é exatamente o mesmo que a semântica de chamar uma `tf.function` no modo eager; O TFF efetivamente entra e sai de um `tf.Session` exclusivo cada vez que `tf_computation` é invocado, enquanto chamar repetidamente uma função no modo eager é análogo a chamar `sess.run` no tensor de saída repetidamente dentro da mesma sessão.

## Como posso contribuir?

Veja o [README](https://github.com/tensorflow/federated/blob/main/README.md), diretrizes [de contribuição](https://github.com/tensorflow/federated/blob/main/CONTRIBUTING.md) e [colaborações](collaborations/README.md).

## Qual é a relação entre FedJAX e TensorFlow Federated?

O TensorFlow Federated (TFF) é um framework completo para aprendizagem e análise federada, projetado para facilitar a composição de diferentes algoritmos e recursos e para permitir a portabilidade de código em diferentes cenários de simulação e implantação. O TFF fornece um runtime escalonável e oferece suporte a muitos algoritmos de privacidade, compactação e otimização por meio de suas APIs padrão. O TFF também suporta [diversos tipos de pesquisa de FL](https://www.tensorflow.org/federated/tff_for_research), com uma coleção de exemplos de artigos publicados do Google aparecendo no [repositório google-research](https://github.com/google-research/federated).

Em contraste, o [FedJAX](https://github.com/google/fedjax) é uma biblioteca leve de simulação baseada em Python e JAX que foca na facilidade de uso e na prototipagem rápida de algoritmos de aprendizagem federados para fins de pesquisa. TensorFlow Federated e FedJAX são desenvolvidos como projetos separados, sem expectativa de portabilidade de código.
