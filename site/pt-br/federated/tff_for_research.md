# Usando TFF para Pesquisa de Aprendizagem Federada

<!-- Note that some section headings are used as deep links into the document.
     If you update those section headings, please make sure you also update
     any links to the section. -->

## Visão geral

O TFF é um framework extensível e poderoso para conduzir pesquisas de aprendizagem federada (FL), simulando cálculos federados em conjuntos de dados proxy realistas. Esta página descreve os principais conceitos e componentes relevantes para simulações de pesquisa, bem como orientações detalhadas para a realização de diferentes tipos de pesquisa em TFF.

## A estrutura típica do código de pesquisa no TFF

Uma simulação de pesquisa FL implementada em TFF normalmente consiste em três tipos principais de lógica.

1. Partes individuais do código do TensorFlow, normalmente `tf.function`, que encapsulam a lógica executada num único local (por exemplo, em clientes ou num servidor). Este código normalmente é escrito e testado sem nenhuma referência `tff.*` e pode ser reutilizado fora do TFF. Por exemplo, o [ciclo de treinamento do cliente no Federated Averaging](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/simple_fedavg/simple_fedavg_tf.py#L184-L222) é implementado neste nível.

2. Lógica de orquestração do TensorFlow Federated, que une os `tf.function` individuais de 1. encapsulando-os como `tff.tf_computation` e, em seguida, orquestrando-os usando abstrações como `tff.federated_broadcast` e `tff.federated_mean` dentro de um `tff.federated_computation`. Veja, por exemplo, esta [orquestração para Federated Averaging](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/simple_fedavg/simple_fedavg_tff.py#L112-L140) .

3. Um script de driver externo que simula a lógica de controle de um sistema FL de produção, selecionando clientes simulados de um dataset e, em seguida, executando cálculos federados definidos em 2. nesses clientes. Por exemplo, [Federated EMNIST experiment driver](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/simple_fedavg/emnist_fedavg_main.py) .

## Datasets de aprendizagem federada

O TensorFlow Federated [hospeda vários datasets](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets) que representam as características de problemas do mundo real que poderiam ser resolvidos com o aprendizado federado.

Observação: esses datasets também podem ser consumidos por qualquer framework de ML baseado em Python como arrays Numpy, conforme documentado na [API ClientData](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/ClientData).

Os datasets incluem:

- [**StackOverflow**.](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/stackoverflow/load_data) Um dataset de texto realista para modelagem de linguagem ou tarefas de aprendizagem supervisionada, com 342.477 usuários unívocos com 135.818.730 exemplos (frases) no conjunto de treinamento.

- [**Federated EMNIST**.](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/emnist/load_data) Um pré-processamento federado do dataset de caracteres e dígitos EMNIST, onde cada cliente corresponde a um escritor diferente. O conjunto completo de treinamento contém 3.400 usuários com 671.585 exemplos de 62 rótulos.

- [**Shakespeare** .](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/shakespeare/load_data) Um dataset de texto menor em nível de caracteres baseado nas obras completas de William Shakespeare. O dataset é composto por 715 usuários (personagens de peças de Shakespeare), onde cada exemplo corresponde a um conjunto contíguo de falas faladas pelo personagem numa determinada peça.

- [**CIFAR-100**.](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/cifar100/load_data) Um particionamento federado do dataset CIFAR-100 em 500 clientes de treinamento e 100 clientes de teste. Cada cliente tem 100 exemplos unívocos. O particionamento é feito de forma a criar uma heterogeneidade mais realista entre os clientes. Para mais detalhes, consulte a [API](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/cifar100/load_data).

- [**Google Landmark v2 dataset**](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/gldv2/load_data) O dataset consiste em fotos de vários pontos de referência mundiais, com imagens agrupadas por fotógrafo para obter um particionamento federado dos dados. Dois tipos de dataset estão disponíveis: um dataset menor com 233 clientes e 23.080 imagens, e um dataset maior com 1.262 clientes e 164.172 imagens.

- [**CelebA**](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/celeba/load_data) Um dataset de exemplos (imagem e atributos faciais) de rostos de celebridades. O dataset federado agrupa os exemplos de cada celebridade para formar um cliente. Existem 9.343 clientes, cada um com pelo menos 5 exemplos. O dataset pode ser dividido em grupos de treinamento e teste por clientes ou por exemplos.

- [**iNaturalist**](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/inaturalist/load_data) Um dataset que consiste em fotos de várias espécies. O dataset contém 120.300 imagens para 1.203 espécies. Sete versões do dataset estão disponíveis. Uma delas é agrupada pelo fotógrafo e é composto por 9.257 clientes. O restante dos datasets são agrupados pela localização geográfica onde a foto foi tirada. Esses seis tipos do dataset consistem de 11 a 3.606 clientes.

## Simulações de alto desempenho

Embora o tempo de uma *simulação* de FL não seja uma métrica relevante para avaliar algoritmos (já que o hardware de simulação não é representativo de ambientes reais de implantação de FL), ser capaz de executar simulações de FL rapidamente é fundamental para a produtividade da pesquisa. Conseqüentemente, o TFF investiu pesadamente no fornecimento de runtimes de alto desempenho rodando em uma ou múltiplas máquinas. A documentação está em desenvolvimento, mas por enquanto veja as instruções sobre [simulações TFF com aceleradores](https://www.tensorflow.org/federated/tutorials/simulations_with_accelerators) e instruções sobre [como configurar simulações com TFF no GCP](https://www.tensorflow.org/federated/gcp_setup). O runtime TFF de alto desempenho é habilitado por padrão.

## TFF para diferentes áreas de pesquisa

### Algoritmos de otimização federados

A pesquisa em algoritmos de otimização federados pode ser feita de diferentes maneiras no TFF, dependendo do nível de customização desejado.

Uma implementação stand-alone mínima do algoritmo [Federated Averaging](https://arxiv.org/abs/1602.05629) é fornecida [aqui](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/simple_fedavg). O código inclui [funções TF](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/simple_fedavg/simple_fedavg_tf.py) para computação local, [computações TFF](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/simple_fedavg/simple_fedavg_tff.py) para orquestração e um [script de driver](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/simple_fedavg/emnist_fedavg_main.py) no dataset EMNIST como exemplo. Esses arquivos podem ser facilmente adaptados para aplicações personalizadas e alterações algorítmicas seguindo instruções detalhadas no [README](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/simple_fedavg/README.md).

Uma implementação mais geral da Federated Averaging pode ser encontrada [aqui](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/learning/algorithms/fed_avg.py). Esta implementação permite técnicas de otimização mais sofisticadas, incluindo o uso de diferentes otimizadores tanto no servidor quanto no cliente. Outros algoritmos de aprendizado federado, incluindo clustering k-means federado, podem ser encontrados [aqui](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/learning/algorithms/).

### Compressão da atualização de modelos

A compressão com perdas de atualizações de modelos pode levar à redução dos custos de comunicação, o que, por sua vez, pode levar à redução do tempo geral de treinamento.

Para reproduzir um [artigo](https://arxiv.org/abs/2201.02664) recente, consulte [este projeto de pesquisa](https://github.com/google-research/federated/tree/master/compressed_communication). Para implementar um algoritmo de compactação personalizado, consulte [compare_methods](https://github.com/google-research/federated/tree/master/compressed_communication/aggregators/comparison_methods) no projeto para obter linhas de base como exemplo o [tutorial de TFF Aggregators](https://www.tensorflow.org/federated/tutorials/custom_aggregators), se ainda não estiver familiarizado.

### Privacidade diferencial

O TFF é interoperável com a biblioteca [TensorFlow Privacy](https://github.com/tensorflow/privacy) para permitir pesquisas em novos algoritmos para treinamento federado de modelos com privacidade diferencial. Para obter um exemplo de treinamento com DP usando [o algoritmo DP-FedAvg básico](https://arxiv.org/abs/1710.06963) e [extensões](https://arxiv.org/abs/1812.06210), consulte [este driver de experimento](https://github.com/google-research/federated/blob/master/differential_privacy/stackoverflow/run_federated.py).

Se quiser implementar um algoritmo DP personalizado e aplicá-lo às atualizações agregadas da média federada, você pode implementar um novo algoritmo de média DP como uma subclasse de [`tensorflow_privacy.DPQuery`](https://github.com/tensorflow/privacy/blob/master/tensorflow_privacy/privacy/dp_query/dp_query.py#L54) e criar um `tff.aggregators.DifferentiallyPrivateFactory` com uma instância de sua consulta. Um exemplo de implementação do [algoritmo DP-FTRL](https://arxiv.org/abs/2103.00039) pode ser encontrado [aqui](https://github.com/google-research/federated/blob/master/dp_ftrl/dp_fedavg.py)

GANs federadas (descritas [abaixo](#generative_adversarial_networks)) são outro exemplo de projeto TFF que implementa privacidade diferencial no nível do usuário (por exemplo, [aqui em código](https://github.com/google-research/federated/blob/master/gans/tff_gans.py#L144)).

### Robustez e ataques

O TFF também pode ser usado para simular os ataques direcionados a sistemas de aprendizagem federados e defesas diferenciais baseadas em privacidade considerados em *[Can You Really Back door Federated Learning?](https://arxiv.org/abs/1911.07963)* (Você pode realmente fazer backdoor no Federated Learning?). Isto é feito construindo um processo iterativo com clientes potencialmente maliciosos (consulte [`build_federated_averaging_process_attacked`](https://github.com/tensorflow/federated/blob/6477a3dba6e7d852191bfd733f651fad84b82eab/federated_research/targeted_attack/attacked_fedavg.py#L412)). O diretório [target_attack](https://github.com/tensorflow/federated/tree/6477a3dba6e7d852191bfd733f651fad84b82eab/federated_research/targeted_attack) contém mais detalhes.

- Novos algoritmos de ataque podem ser implementados escrevendo uma função de atualização do cliente que é uma função Tensorflow, veja [`ClientProjectBoost`](https://github.com/tensorflow/federated/blob/6477a3dba6e7d852191bfd733f651fad84b82eab/federated_research/targeted_attack/attacked_fedavg.py#L460) para um exemplo.
- Novas defesas podem ser implementadas personalizando ['tff.utils.StatefulAggregateFn'](https://github.com/tensorflow/federated/blob/6477a3dba6e7d852191bfd733f651fad84b82eab/tensorflow_federated/python/core/utils/computation_utils.py#L103) que agrega as saídas do cliente para obter uma atualização global.

Para obter um exemplo de script para simulação, veja [`emnist_with_targeted_attack.py`](https://github.com/tensorflow/federated/blob/6477a3dba6e7d852191bfd733f651fad84b82eab/federated_research/targeted_attack/emnist_with_targeted_attack.py).

### Redes Adversariais Gerativas

As GANs criam um [padrão de orquestração federada](https://github.com/google-research/federated/blob/master/gans/tff_gans.py#L266-L316) interessante que parece um pouco diferente da média federada padrão. Elas envolvem duas redes distintas (o gerador e o discriminador), cada uma treinada com sua própria etapa de otimização.

O TFF pode ser usado para pesquisas sobre treinamento federado de GANs. Por exemplo, o algoritmo DP-FedAvg-GAN apresentado em [trabalhos recentes](https://arxiv.org/abs/1911.06679) é [implementado em TFF](https://github.com/tensorflow/federated/tree/main/federated_research/gans). Este trabalho demonstra a eficácia da combinação de aprendizagem federada, modelos gerativos e [privacidade diferencial](#differential_privacy).

### Personalização

A personalização no cenário da aprendizagem federada é uma área de pesquisa ativa. O objetivo da personalização é fornecer diferentes modelos de inferência para diferentes usuários. Existem abordagens potencialmente diferentes para este problema.

Uma abordagem é permitir que cada cliente ajuste um único modelo global (treinado usando aprendizagem federada) com seus dados locais. Esta abordagem tem conexões com a meta-aprendizagem, veja, por exemplo, [este artigo](https://arxiv.org/abs/1909.12488). Um exemplo desta abordagem é dado em [`emnist_p13n_main.py`](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/personalization/emnist_p13n_main.py). Para explorar e comparar diferentes estratégias de personalização, você pode:

- Definir uma estratégia de personalização implementando uma `tf.function` que parte de um modelo inicial, treina e avalia um modelo personalizado usando os datasets locais de cada cliente. Um exemplo é dado por [`build_personalize_fn`](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/personalization/p13n_utils.py).

- Definir um `OrderedDict` que mapeie nomes de estratégias para as estratégias de personalização correspondentes e use-o como o argumento `personalize_fn_dict` em [`tff.learning.build_personalization_eval_computation`](https://www.tensorflow.org/federated/api_docs/python/tff/learning/build_personalization_eval_computation).

Outra abordagem é evitar treinar um modelo totalmente global treinando parte de um modelo totalmente local. Uma instanciação dessa abordagem é descrita [nesta postagem do blog](https://ai.googleblog.com/2021/12/a-scalable-approach-for-partially-local.html). Essa abordagem também está ligada ao meta-aprendizado, veja [este artigo](https://arxiv.org/abs/2102.03448). Para explorar a aprendizagem federada parcialmente local, você pode:

- Confira o [tutorial](https://www.tensorflow.org/federated/tutorials/federated_reconstruction_for_matrix_factorization) para obter um exemplo de código completo aplicando a Reconstrução Federada e [exercícios de acompanhamento](https://www.tensorflow.org/federated/tutorials/federated_reconstruction_for_matrix_factorization#further_explorations).

- Crie um processo de treinamento parcialmente local usando [`tff.learning.reconstruction.build_training_process`](https://www.tensorflow.org/federated/api_docs/python/tff/learning/reconstruction/build_training_process), modificando `dataset_split_fn` para personalizar o comportamento do processo.
