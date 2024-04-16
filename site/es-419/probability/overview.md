# TensorFlow Probability

TensorFlow Probability es una biblioteca de razonamiento probabilístico y análisis estadístico de TensorFlow. Como parte del ecosistema TensorFlow, TensorFlow Probability proporciona integración de métodos probabilísticos con redes profundas, inferencia basada en gradientes mediante diferenciación automática y escalabilidad a grandes conjuntos de datos y modelos con aceleración de hardware (GPU) y computación distribuida.

Para comenzar a usar TensorFlow Probability, consulte la [guía de instalación](./install.md) y los [tutoriales del bloc de notas de Python](https://github.com/tensorflow/probability/blob/main/tensorflow_probability/examples/jupyter_notebooks/) {:.external}.

## Componentes

Nuestras herramientas probabilísticas de aprendizaje automático están estructuradas de la siguiente manera:

### Capa 0: TensorFlow

*Las operaciones numéricas* (en particular, la clase `LinearOperator`) permiten ejecutar implementaciones sin matrices que pueden explotar una estructura particular (diagonal, de rango bajo, etc.) para un cálculo eficiente. El equipo de TensorFlow Probability se ha encargado de compilarlo y mantenerlo, y es parte de [`tf.linalg`](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/python/ops/linalg) en el núcleo de TensorFlow.

### Capa 1: Bloques de creación estadísticos

- *Distributions* ([`tfp.distributions`](https://github.com/tensorflow/probability/tree/main/tensorflow_probability/python/distributions)): una gran colección de distribuciones de probabilidad y estadísticas relacionadas con semántica {:.external} por lotes y de [difusión](https://docs.scipy.org/doc/numpy-1.14.0/user/basics.broadcasting.html).
- *Bijectors* ([`tfp.bijectors`](https://github.com/tensorflow/probability/tree/main/tensorflow_probability/python/bijectors)): transformaciones reversibles y componibles de variables aleatorias. Los biyectores ofrecen una rica clase de distribuciones transformadas, desde ejemplos clásicos como la [distribución log-normal](https://en.wikipedia.org/wiki/Log-normal_distribution) {:.external} hasta modelos sofisticados de aprendizaje profundo como los [flujos autorregresivos enmascarados](https://arxiv.org/abs/1705.07057) {:.external}.

### Capa 2: Compilación de modelos

- Distribuciones conjuntas (p. ej., [`tfp.distributions.JointDistributionSequential`](https://github.com/tensorflow/probability/tree/main/tensorflow_probability/python/distributions/joint_distribution_sequential.py)): distribuciones conjuntas sobre una o más distribuciones posiblemente interdependientes. Para obtener una introducción al modelado con `JointDistribution` de TFP, consulte [esta colaboración](https://github.com/tensorflow/probability/blob/main/tensorflow_probability/examples/jupyter_notebooks/Modeling_with_JointDistribution.ipynb)
- *Capas probabilísticas* ([`tfp.layers`](https://github.com/tensorflow/probability/tree/main/tensorflow_probability/python/layers)): capas de redes neuronales con incertidumbre sobre las funciones que representan, extendiendo las capas de TensorFlow.

### Capa 3: Inferencia probabilística

- *Método de Monte Carlo basado en cadenas de Markov* ([`tfp.mcmc`](https://github.com/tensorflow/probability/tree/main/tensorflow_probability/python/mcmc)): algoritmos para aproximar integrales mediante muestreo. Incluye [algoritmo hamiltoniano de Monte Carlo](https://en.wikipedia.org/wiki/Hamiltonian_Monte_Carlo) {:.external}, algoritmo Metrópolis-Hastings de paseo aleatorio y la capacidad de crear kernels de transición personalizados.
- *Inferencia variacional* ([`tfp.vi`](https://github.com/tensorflow/probability/tree/main/tensorflow_probability/python/vi)): algoritmos para aproximar integrales mediante optimización.
- *Optimizadores* ([`tfp.optimizer`](https://github.com/tensorflow/probability/tree/main/tensorflow_probability/python/optimizer)): métodos de optimización estocástica que amplían los optimizadores de TensorFlow. Incluye [dinámica de Langevin con gradiente estocástico](http://www.icml-2011.org/papers/398_icmlpaper.pdf) {:.external}.
- *Monte Carlo* ([`tfp.monte_carlo`](https://github.com/tensorflow/probability/blob/main/tensorflow_probability/python/monte_carlo)): herramientas para calcular las expectativas de Monte Carlo.

TensorFlow Probability se encuentra en fase de desarrollo activo y las interfaces pueden cambiar.

## Ejemplos

Además de los [tutoriales del bloc de notas de Python](https://github.com/tensorflow/probability/blob/main/tensorflow_probability/examples/jupyter_notebooks/) {:.external} que aparecen en la navegación, hay algunos scripts de ejemplo disponibles:

- [Autocodificadores variacionales](https://github.com/tensorflow/probability/tree/main/tensorflow_probability/examples/vae.py): aprendizaje de representación con un código latente e inferencia variacional.
- [Autocodificador con cuantización vectorial:](https://github.com/tensorflow/probability/tree/main/tensorflow_probability/examples/vq_vae.py) aprendizaje de representación discreta con cuantización vectorial.
- [Redes neuronales bayesianas](https://github.com/tensorflow/probability/tree/main/tensorflow_probability/examples/bayesian_neural_network.py): redes neuronales con incertidumbre sobre sus ponderaciones.
- [Regresión logística bayesiana](https://github.com/tensorflow/probability/tree/main/tensorflow_probability/examples/logistic_regression.py): inferencia bayesiana para clasificación binaria.

## Informe de problemas

Informe errores o solicitudes de características mediante el uso del [rastreador de problemas de TensorFlow Probability](https://github.com/tensorflow/probability/issues).
