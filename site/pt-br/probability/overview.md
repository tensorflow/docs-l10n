# TensorFlow Probability

O TensorFlow Probability é uma biblioteca para raciocínio probabilístico e análise estatística no TensorFlow. O TensorFlow Probability faz parte do ecossistema do TensorFlow e conta com integração de métodos probabilísticos com redes profundas, inferência baseada em gradientes usando diferenciação automática e escalabilidade para datasets e modelos grandes com aceleração de hardware (GPUs) e computação distribuída.

Para começar a usar o TensorFlow Probability, confira o [guia de instalação](./install.md) e veja os [tutoriais em notebook do Python](https://github.com/tensorflow/probability/blob/main/tensorflow_probability/examples/jupyter_notebooks/){:.external}.

## Componentes

As ferramentas de aprendizado de máquina probabilístico são estruturadas da seguinte forma:

### Camada 0: TensorFlow

As *operações numéricas* — especificamente, a classe `LinearOperator` — permitem implementações sem matrizes que podem aproveitar uma estrutura específica (diagonal, posto baixo, etc.) para deixar a computação eficiente. A classe é construída e mantida pela equipe do TensorFlow Probability e faz parte do [`tf.linalg`](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/python/ops/linalg) no TensorFlow core.

### Camada 1: blocos de construção estatística

- *Distribuições* ([`tfp.distributions`](https://github.com/tensorflow/probability/tree/main/tensorflow_probability/python/distributions)): coleção grande de distribuições de probabilidade e estatísticas relacionadas com lote e semântica de [broadcasting](https://docs.scipy.org/doc/numpy-1.14.0/user/basics.broadcasting.html){:.external}.
- *Bijetores* ([`tfp.bijectors`](https://github.com/tensorflow/probability/tree/main/tensorflow_probability/python/bijectors)): transformações de variáveis aleatórias que são reversíveis e podem ser compostas. Os bijetores oferecem uma classe rica de distribuições transformadas, desde exemplos clássicos, como a [distribuição log-normal](https://en.wikipedia.org/wiki/Log-normal_distribution){:.external} até modelos de aprendizado profundo sofisticados, como [fluxos autorregressivos com máscara](https://arxiv.org/abs/1705.07057){:.external}.

### Camada 2: criação do modelo

- Distribuições conjuntas (por exemplo: [`tfp.distributions.JointDistributionSequential`](https://github.com/tensorflow/probability/tree/main/tensorflow_probability/python/distributions/joint_distribution_sequential.py)): distribuições conjuntas de uma ou mais distribuições possivelmente interdependentes. Confira uma introdução da modelagem com `JointDistribution`s do TFP [neste colab](https://github.com/tensorflow/docs-l10n/blob/master/site/pt-br/probability/examples/Modeling_with_JointDistribution.ipynb).
- *Camadas probabilísticas* ([`tfp.layers`](https://github.com/tensorflow/probability/tree/main/tensorflow_probability/python/layers)): camadas de redes neurais com incerteza sobre as funções representadas, que estendem as camadas do TensorFlow.

### Camada 3: inferência probabilística

- *Monte Carlo via cadeias de Markov* ([`tfp.mcmc`](https://github.com/tensorflow/probability/tree/main/tensorflow_probability/python/mcmc)): algoritmos para aproximação de integrais via amostragem. Inclui [Monte Carlo Hamiltoniano](https://en.wikipedia.org/wiki/Hamiltonian_Monte_Carlo){:.external}, algoritmo de Metropolis-Hastings random-walk e a capacidade de compilar kernels de transição personalizados.
- *Inferência variacional* ([`tfp.vi`](https://github.com/tensorflow/probability/tree/main/tensorflow_probability/python/vi)): algoritmos para aproximação de integrais por meio de otimização.
- *Otimizadores* ([`tfp.optimizer`](https://github.com/tensorflow/probability/tree/main/tensorflow_probability/python/optimizer)): métodos de otimização estocástica que estendem os TensorFlow Optimizers. Inclui o [Método do gradiente estocástico de Langevin](http://www.icml-2011.org/papers/398_icmlpaper.pdf){:.external}.
- *Monte Carlo* ([`tfp.monte_carlo`](https://github.com/tensorflow/probability/blob/main/tensorflow_probability/python/monte_carlo)): ferramentas para computar as expectativas de Monte Carlo.

O TensorFlow Probability está em constante desenvolvimento, e as interfaces podem ser alteradas.

## Exemplos

Além dos [tutoriais em notebook do Python](https://github.com/tensorflow/probability/blob/main/tensorflow_probability/examples/jupyter_notebooks/){:.external} indicados na navegação, confira alguns scripts de exemplo disponíveis:

- [Autoencoders variacionais](https://github.com/tensorflow/probability/tree/main/tensorflow_probability/examples/vae.py) — representam o aprendizado com um código latente e inferência variacional.
- [Autoencoder quantizado em vetores](https://github.com/tensorflow/probability/tree/main/tensorflow_probability/examples/vq_vae.py) — aprendizado com representação discreta e quantização em vetores.
- [Redes neurais bayesianas](https://github.com/tensorflow/probability/tree/main/tensorflow_probability/examples/bayesian_neural_network.py) — redes neurais com incerteza sobre os pesos.
- [Regressão logística bayesiana](https://github.com/tensorflow/probability/tree/main/tensorflow_probability/examples/logistic_regression.py) — inferência bayesiana para classificação binária.

## Comunique problemas

Comunique bugs ou solicite recursos usando o [issue tracker do TensorFlow Probability](https://github.com/tensorflow/probability/issues).
