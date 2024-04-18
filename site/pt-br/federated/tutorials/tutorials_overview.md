# Tutoriais do TensorFlow Federated

Estes tutoriais [baseados no Colab](https://colab.research.google.com/) apresentam os principais conceitos do TFF e das APIs usando exemplos práticos. Confira a documentação de referência nos [guias do TFF](../get_started.md).

Observação: no momento, o TFF requer o Python 3.9 ou posterior, mas os runtimes hospedados no [Google Colaboratory](https://research.google.com/colaboratory/) usam atualmente o Python 3.7. Portanto, para executar estes notebooks, você precisará usar um [runtime local personalizado](https://research.google.com/colaboratory/local-runtimes.html).

**Introdução ao aprendizado federado**

- O tutorial [Aprendizado federado para classificação de imagens](federated_learning_for_image_classification.ipynb) apresenta as partes essenciais da API Federated Learning (FL) e demonstra como usar o TFF para simular aprendizado federado com dados federados tipo MNIST.
- O tutorial [Aprendizado federado para geração de texto](federated_learning_for_text_generation.ipynb) aprofunda a demonstração de como usar a API FL do TFF para refinar um modelo pré-treinado serializado para uma tarefa de modelagem de linguagem.
- O tutorial [Ajustes de agregações recomendadas para aprendizado](tuning_recommended_aggregators.ipynb) mostra como as computações básicas do FL no `tff.learning` podem ser combinadas com rotinas de agregação especializadas que oferecem robustez, privacidade diferencial, compressão e muito mais.
- O tutorial [Reconstrução federada para fatoração de matriz](federated_reconstruction_for_matrix_factorization.ipynb) apresenta o aprendizado federado parcialmente local, em que alguns dos parâmetros de clientes nunca são agregados no servidor. O tutorial demonstra como usar a API Federated Learning para treinar um modelo de fatoração de matriz parcialmente local.

**Introdução à análise federada**

- O tutorial [Heavy hitters privados](private_heavy_hitters.ipynb) mostra como usar `tff.analytics.heavy_hitters` para construir uma computação de análise federada com o objetivo de descobrir heavy hitters privados.

**Como escrever computações federadas personalizadas**

- O tutorial [Construindo seu próprio algoritmo de aprendizado federado](building_your_own_federated_learning_algorithm.ipynb) mostra como usar as APIs Core do TFF para implementar algoritmos de aprendizado federado, usando cálculo federado de médias como exemplo.
- O tutorial [Combinando algoritmos de aprendizado](composing_learning_algorithms.ipynb) mostra como usar a API Learning do TFF para implementar facilmente novos algoritmos de aprendizado federado, especialmente variantes do cálculo federado de médias.
- O tutorial [Algoritmo federado personalizado com otimizadores do TFF](custom_federated_algorithm_with_tff_optimizers.ipynb) mostra como usar `tff.learning.optimizers` para criar um processo iterativo personalizado para cálculo federado de médias.
- Os tutoriais [Algoritmos federados personalizados, parte 1: Introdução ao Federated Core](custom_federated_algorithms_1.ipynb) e [parte 2: Implementando o cálculo federado de médias](custom_federated_algorithms_2.ipynb) apresentam os conceitos e interfaces essenciais oferecidos pela API Federated Core (API FC).
- O tutorial [Implementando agregações personalizadas](custom_aggregators.ipynb) explica os princípios de design por trás do módulo `tff.aggregators` e as práticas recomendadas para implementar agregação personalizada de valores dos clientes para o servidor.

**Práticas recomendadas de simulação**

- O tutorial [Simulação do TFF com aceleradores (GPU)](simulations_with_accelerators.ipynb) mostra como o runtime de alto desempenho do TFF pode ser usado com GPUs.

- O tutorial [Trabalhando com ClientData](working_with_client_data.ipynb) apresenta as práticas recomendadas para integrar os datasets de simulação baseados em [ClientData](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/ClientData) do TFF às computações do TFF.

**Tutoriais intermediários e avançados**

- O tutorial [Geração de ruído aleatório](random_noise_generation.ipynb) destaca algumas sutilezas ao usar aleatoriedade em computações descentralizadas e recomenda práticas e padrões.

- O tutorial [Enviando dados diferentes para determinados clientes com tff.federated_select](federated_select.ipynb) apresenta o operador `tff.federated_select` e fornece um exemplo simples de algoritmo federado personalizado que envia dados diferentes a clientes diferentes.

- O tutorial [Aprendizado federado de modelos grandes eficiente nos clientes via federated_selec e agregação esparsa](sparse_federated_learning.ipynb) mostra como o TFF pode ser usado para treinar um modelo muito grande em que cada dispositivo cliente baixa e atualiza somente uma pequena parte do modelo usando `tff.federated_select` e agregação esparsa.

- O tutorial [TFF para pesquisa de aprendizado federado – Compressão de modelo e atualização](tff_for_federated_learning_research_compression.ipynb) demonstra como agregações personalizadas construídas utilizando a [API tensor_encoding](https://github.com/tensorflow/model-optimization/tree/master/tensorflow_model_optimization/python/core/internal/tensor_encoding) podem ser usadas no TFF.

- O tutorial [Aprendizado federado com privacidade diferencial no TFF](federated_learning_with_differential_privacy.ipynb) demonstra como usar o TFF para treinar modelos com privacidade diferencial no nível de usuário.

- O tutorial [Suporte ao JAX no TFF](../tutorials/jax_support.ipynb) mostra como as computações do [JAX](https://github.com/google/jax) podem ser usadas no TFF, demonstrando como o TFF foi criado para poder fazer a interoperabilidade com outros frameworks de aprendizado de máquina de front-end e back-end.
