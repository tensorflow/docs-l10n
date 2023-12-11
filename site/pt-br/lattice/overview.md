# TensorFlow Lattice (TFL)

O TensorFlow Lattice é uma biblioteca que implementa modelos baseados em lattice flexíveis, controlados e interpretáveis. A biblioteca permite que você injete conhecimento de domínio no processo de aprendizado pelas [restrições de formato](tutorials/shape_constraints.ipynb) de senso comum ou baseadas na política. Isso é realizado usando uma coleção de [camadas do Keras](tutorials/keras_layers.ipynb) que podem satisfazer restrições como monotonicidade, convexidade e confiança em pares. A biblioteca também fornece [estimadores predefinidos](tutorials/canned_estimators.ipynb) fáceis de configurar.

## Conceitos

Esta seção é uma versão simplificada da descrição em [Monotonic Calibrated Interpolated Look-Up Tables](http://jmlr.org/papers/v17/15-243.html) (Tabelas de consulta interpoladas e calibradas monotônicas), JMLR 2016.

### Lattices

Um *lattice* é uma tabela de consulta que consegue aproximar relações arbitrárias de entrada-saída nos seus dados. Ele sobrepõe uma grade regular no seu espaço de entrada e aprende valores para a saída nos vértices da grade. Para um ponto de teste $x$, $f(x)$ é interpolado de maneira linear a partir dos valores lattice em volta de $x$.

<img src="images/2d_lattice.png" style="display:block; margin:auto;">

O exemplo simples acima é uma função com 2 características de entrada e 4 parâmetros: $\theta=[0, 0.2, 0.4, 1]$, que são os valores da função nos cantos do espaço de entrada. O resto da função é interpolada a partir desses parâmetros.

A função $f(x)$ consegue capturar interações não lineares entre as características. Você pode pensar nos parâmetros lattice como a altura de postes colocados no chão de uma grade regular, e a função resultante é como um tecido puxado com força entre os quatro postes.

Com as características $D$ e os 2 vértices ao longo de cada dimensão, um lattice regular terá $2^D$ parâmetros. Para se adaptar a uma função mais flexível, você pode especificar um lattice mais refinado no espaço da característica com mais vértices ao longo de cada dimensão. As funções de regressão lattice são contínuas e têm partes infinitamente diferenciáveis.

### Calibração

Digamos que o lattice de amostra anterior representa a *satisfação do usuário* aprendida com uma recomendação de café local calculada usando características:

- preço do café, de 0 a 20 dólares
- distância do usuário, de 0 a 30 quilômetros

Queremos que nosso modelo aprenda a satisfação do usuário com uma recomendação de café local. Os modelos do TensorFlow Lattice podem usar *funções lineares por partes* (com `tfl.layers.PWLCalibration`) para calibrar e normalizar as características de entrada para o intervalo aceito pelo lattice: 0.0 a 1.0 no lattice de exemplo acima. O seguinte mostra exemplos, como funções de calibração com 10 keypoints:

<p align="center"> <img src="images/pwl_calibration_distance.png"> <img src="images/pwl_calibration_price.png"></p>

Geralmente, é uma boa ideia usar os quantis das características como keypoints de entrada. Os [estimadores predefinidos](tutorials/canned_estimators.ipynb) do TensorFlow Lattice podem configurar automaticamente os keypoints de entrada para os quantis de características.

Para as características categóricas, o TensorFlow Lattice fornece calibração categórica (com `tfl.layers.CategoricalCalibration`) com delimitação de saída semelhante para alimentar o lattice.

### Ensembles

O número de parâmetros de uma camada de lattice aumenta exponencialmente com o número de características de entrada. Portanto, não é bem escalado para dimensões muito altas. O TensorFlow Lattice oferece ensembles de lattices que combinam vários *pequenos* lattices (regulares), o que permite ao modelo crescer linearmente no número de características.

A biblioteca oferece duas variações destes ensembles:

- **Pequenos Lattices Aleatórios** (RTL): cada submodelo usa um subconjunto aleatório de características (com substituição).

- **Crystals** : o algoritmo Crystals primeiro treina um modelo de *prefitting* que estima interações de características em pares. Em seguida, ele organiza o ensemble final para que as características com mais interações não lineares estejam nos mesmos lattices.

## Por que o TensorFlow Lattice?

Confira uma breve introdução ao TensorFlow Lattice nesta [postagem de blog do TF](https://blog.tensorflow.org/2020/02/tensorflow-lattice-flexible-controlled-and-interpretable-ML.html).

### Interpretabilidade

Como os parâmetros de cada camada são a saída dessa camada, é fácil analisar, entender e depurar cada parte do modelo.

### Modelos exatos e flexíveis

Usando lattices mais granulares, você pode obter funções *arbitrariamente complexas* com uma única camada de lattice. O uso de várias camadas de calibradores e lattices geralmente funciona bem na prática e tem um desempenho igual ou superior aos modelos de DNN de tamanho semelhante.

### Restrições de formato de senso comum

Os dados de treinamento do mundo real podem não representar suficientemente os dados do runtime. Soluções de ML flexíveis, como DNNs ou florestas, geralmente agem de maneira inesperada e até frenética nas partes do espaço de entrada não cobertas pelos dados de treinamento. Esse comportamento é especialmente problemático quando as restrições de política ou imparcialidade podem ser violadas.

<img src="images/model_comparison.png" style="display:block; margin:auto;">

Embora formas comuns de regularização possam resultar em extrapolação mais sensata, os regularizadores padrão não podem garantir um comportamento de modelo razoável em todo o espaço de entrada, especialmente com entradas de muitas dimensões. Trocar para modelos mais simples com um comportamento mais controlado e previsível pode custar bastante para a exatidão do modelo.

O TF Lattice possibilita continuar usando modelos flexíveis, mas fornece várias opções para injetar conhecimento de domínio no processo de aprendizado por [restrições de formato](tutorials/shape_constraints.ipynb) de senso comum ou baseadas na política semanticamente significativas:

- **Monotonicidade**: você pode especificar que a saída só deve aumentar/diminuir em relação a uma entrada. Em nosso exemplo, você deve especificar que uma maior distância até um café só deve diminuir a preferência de usuário prevista.

<p align="center"> <img src="images/linear_fit.png"> <img src="images/flexible_fit.png"> <img src="images/regularized_fit.png"> <img src="images/monotonic_fit.png"></p>

- **Convexidade/concavidade**: você pode especificar que o formato da função pode ser convexo ou côncavo. Combinado com a monotonicidade, isso pode forçar a função a representar retornos decrescentes em relação a uma determinada característica.

- **Unimodalidade**: você pode especificar que a função deve ter um pico ou vale exclusivo. Isso permite que você represente funções que têm um *ponto favorável* em relação a uma característica.

- **Confiança em pares**: essa restrição funciona em um par de características e sugere que uma característica de entrada reflete semanticamente a confiança em outra característica. Por exemplo, um maior número de avaliações deixa você mais confiante na classificação média de estrelas de um restaurante. O modelo será mais sensível em relação à classificação de estrelas (ou seja, terá um declive maior em relação à classificação) quando o número de avaliações for maior.

### Flexibilidade controlada com regularizadores

Além das restrições de formato, o lattice do TensorFlow oferece um número de regularizadores para controlar a flexibilidade e a suavidade da função para cada camada.

- **Laplacian Regularizer**: as saídas dos vértices/keypoints de lattice/calibração são regularizados em relação aos valores dos respectivos vizinhos. Isso resulta em uma função *mais achatada*.

- **Hessian Regularizer**: penaliza a primeira derivada da camada de calibração de PWL para deixar a função *mais linear*.

- **Wrinkle Regularizer**: penaliza a segunda derivada da camada de calibração de PWL para evitar mudanças inesperadas na curvatura. Deixa a função mais suave.

- **Torsion Regularizer**: as saídas do lattice serão regularizadas para prevenir a torsão entre as características. Em outras palavras, o modelo será regularizado para a independência entre as contribuições das características.

### Combine com outras camadas do Keras

Você pode usar as camadas do TF Lattice em combinação com outras camadas do Keras para construir modelos parcialmente restritos ou regularizados. Por exemplo, as camadas de calibração de lattice ou PWL podem ser usadas na última camada de redes mais profundas que incluem embeddings ou outras camadas do Keras.

## Artigos

- [Deontological Ethics By Monotonicity Shape Constraints](https://arxiv.org/abs/2001.11990) (Ética deontológica por restrições de formato de monotonicidade), Serena Wang, Maya Gupta, International Conference on Artificial Intelligence and Statistics (AISTATS), 2020
- [Shape Constraints for Set Functions](http://proceedings.mlr.press/v97/cotter19a.html) (Restrições de formato para funções de conjuntos), Andrew Cotter, Maya Gupta, H. Jiang, Erez Louidor, Jim Muller, Taman Narayan, Serena Wang, Tao Zhu. International Conference on Machine Learning (ICML), 2019
- [Diminishing Returns Shape Constraints for Interpretability and Regularization](https://papers.nips.cc/paper/7916-diminishing-returns-shape-constraints-for-interpretability-and-regularization) (Restrições de formato de retornos decrescentes para interpretabilidade e regularização), Maya Gupta, Dara Bahri, Andrew Cotter, Kevin Canini, Advances in Neural Information Processing Systems (NeurIPS), 2018
- [Deep Lattice Networks and Partial Monotonic Functions](https://research.google.com/pubs/pub46327.html) (Redes lattice profundas e funções monotônicas parciais), Seungil You, Kevin Canini, David Ding, Jan Pfeifer, Maya R. Gupta, Advances in Neural Information Processing Systems (NeurIPS), 2017
- [Fast and Flexible Monotonic Functions with Ensembles of Lattices](https://papers.nips.cc/paper/6377-fast-and-flexible-monotonic-functions-with-ensembles-of-lattices) (Funções monotônicas rápidas e flexíveis com ensembles de lattices), Mahdi Milani Fard, Kevin Canini, Andrew Cotter, Jan Pfeifer, Maya Gupta, Advances in Neural Information Processing Systems (NeurIPS), 2016
- [Monotonic Calibrated Interpolated Look-Up Tables](http://jmlr.org/papers/v17/15-243.html), (Tabelas de consulta interpoladas e calibradas monotônicas), Maya Gupta, Andrew Cotter, Jan Pfeifer, Konstantin Voevodski, Kevin Canini, Alexander Mangylov, Wojciech Moczydlowski, Alexander van Esbroeck, Journal of Machine Learning Research (JMLR), 2016
- [Optimized Regression for Efficient Function Evaluation](http://ieeexplore.ieee.org/document/6203580/) (Regressão otimizada para avaliação de função eficiente), Eric Garcia, Raman Arora, Maya R. Gupta, IEEE Transactions on Image Processing, 2012
- [Lattice Regression](https://papers.nips.cc/paper/3694-lattice-regression) (Regressão de lattice), Eric Garcia, Maya Gupta, Advances in Neural Information Processing Systems (NeurIPS), 2009

## Tutoriais e documentação da API

Para arquiteturas de modelo comuns, você pode usar [modelos pré-fabricados do Keras](tutorials/premade_models.ipynb) ou [Estimadores predefinidos](tutorials/canned_estimators.ipynb). Você também pode criar modelos personalizados usando [camadas Keras do TF Lattice](tutorials/keras_layers.ipynb) ou combinar com outras camadas do Keras. Confira a [documentação completa da API](https://www.tensorflow.org/lattice/api_docs/python/tfl) para mais detalhes.
