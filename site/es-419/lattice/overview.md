# TensorFlow Lattice (TFL)

TensorFlow Lattice es una biblioteca que implementa modelos basados ​​en cuadrículas flexibles, controlados e interpretables. La biblioteca le permite inyectar conocimiento del dominio en el proceso de aprendizaje a través de [restricciones de forma](tutorials/shape_constraints.ipynb) impulsadas por directivas o el sentido común. Esto se hace mediante el uso de una colección de [capas de Keras](tutorials/keras_layers.ipynb) que pueden satisfacer restricciones como monotonicidad, convexidad y confianza por pares. La biblioteca también proporciona [estimadores prediseñados](tutorials/canned_estimators.ipynb) fáciles de configurar.

## Conceptos

Esta sección es una versión simplificada de la descripción en [Tablas de búsqueda interpoladas calibradas monótonas](http://jmlr.org/papers/v17/15-243.html), JMLR 2016.

### Cuadrículas

Una *cuadrícula* es una tabla de búsqueda interpolada que puede aproximarse a relaciones arbitrarias de entrada y salida en sus datos. Superpone una cuadrícula regular en su espacio de entrada y aprende valores para la salida en los vértices de la cuadrícula. Para un punto de prueba $x$, $f(x)$ se interpola linealmente a partir de los valores de la cuadrícula que rodean a $x$.


<img src="images/2d_lattice.png" style="display:block; margin:auto;">

El ejemplo simple anterior es una función con 2 características de entrada y 4 parámetros: $\theta=[0, 0.2, 0.4, 1]$, que son los valores de la función en las esquinas del espacio de entrada; el resto de la función se interpola a partir de estos parámetros.

La función $f(x)$ puede capturar interacciones no lineales entre características. Puede pensar en los parámetros de la cuadrícula como la altura de postes colocados en el suelo en una cuadrícula común, y la función resultante es como una tela estirada desde los cuatro postes.

Con características $D$ y 2 vértices a lo largo de cada dimensión, una cuadrícula regular tendrá parámetros $2^D$. Para ajustar una función más flexible, puede especificar una cuadrícula más fina sobre el espacio de características con más vértices a lo largo de cada dimensión. Las funciones de regresión reticular son continuas e infinitamente diferenciables por partes.

### Calibración

Digamos que el ejemplo de la cuadrícula anterior representa *la felicidad del usuario* aprendida con respecto a una cafetería local recomendada y que se calcula con las características:

- precio del café, en un rango de 0 a 20 dólares
- distancia al usuario, en un rango de 0 a 30 kilómetros

Queremos que nuestro modelo aprenda sobre la felicidad del usuario con una cafetería local recomendada. Los modelos de TensorFlow Lattice pueden usar *funciones lineales por partes* (con `tfl.layers.PWLCalibration`) para calibrar y normalizar las características de entrada al rango aceptado por la cuadrícula: 0,0 a 1,0 en la cuadrícula del ejemplo anterior. A continuación se muestran ejemplos de funciones de calibraciones con 10 puntos clave:

<p align="center"><img src="images/model_comparison.png" style="display:block; margin:auto;"> </p>

Suele ser una buena idea usar los cuantiles de las características como puntos clave de entrada. [Los estimadores prediseñados](tutorials/canned_estimators.ipynb) de TensorFlow Lattice pueden establecer automáticamente los puntos clave de entrada en los cuantiles de las características.

Para funciones categóricas, TensorFlow Lattice proporciona calibración categórica (con `tfl.layers.CategoricalCalibration`) con límites de salida similares para alimentar una cuadrícula.

### Conjuntos

La cantidad de parámetros de una capa cuadricular aumenta exponencialmente con la cantidad de características de entrada, por lo que no se escala bien a dimensiones muy altas. Para superar esta limitación, TensorFlow Lattice ofrece conjuntos de cuadrículas que combinan (en promedio) varias cuadrículas *pequeñas*, lo que permite que el modelo crezca linealmente en la cantidad de características.

La biblioteca ofrece dos variaciones de estos conjuntos:

- **Cuadrículas pequeñas aleatorias** (RTL, por sus siglas en inglés): cada submodelo usa un subconjunto aleatorio de características (con reemplazo).

- **Crystals**: el algoritmo Crystals primero entrena un modelo de *preajuste* que estima las interacciones de características por pares. Luego organiza el conjunto final de manera que las entidades con más interacciones no lineales estén en las mismas cuadrículas.

## ¿Por qué TensorFlow Lattice?

Puede encontrar una breve introducción a TensorFlow Lattice en esta [publicación del blog de TF](https://blog.tensorflow.org/2020/02/tensorflow-lattice-flexible-controlled-and-interpretable-ML.html).

### Interpretabilidad

Dado que los parámetros de cada capa son la salida de esa capa, es fácil analizar, comprender y depurar cada parte del modelo.

### Modelos precisos y flexibles

Al usar cuadrículas ajustadas, puede obtener funciones *arbitrariamente complejas* con una sola capa de cuadrícula. El uso de varias capas de calibradores y cuadrículas suele funcionar bien en la práctica y puede igualar o superar a los modelos DNN de tamaños similares.

### Restricciones de forma de sentido común

Es posible que los datos de entrenamiento del mundo real no representen suficientemente los datos del tiempo de ejecución. Las soluciones de aprendizaje automático flexibles, como las DNN o los bosques, suelen actuar de forma inesperada e incluso desenfrenada en partes del espacio de entrada que no cubren los datos de entrenamiento. Este comportamiento es especialmente problemático cuando existe el riesgo de violar restricciones de directivas o de equidad.

 <img src="images/pwl_calibration_distance.png"> <img src="images/pwl_calibration_price.png">


Aunque las formas comunes de regularización pueden dar como resultado una extrapolación más sensata, los regularizadores estándar no pueden garantizar un comportamiento razonable del modelo en todo el espacio de entrada, especialmente con entradas de alta dimensión. Cambiar a modelos más simples con un comportamiento más controlado y predecible puede tener un costo severo para la precisión del modelo.

TF Lattice permite seguir usando modelos flexibles, pero proporciona varias opciones para inyectar conocimiento del dominio en el proceso de aprendizaje a través de [restricciones de forma](tutorials/shape_constraints.ipynb) semánticamente significativas, impulsadas por directivas o el sentido común.

- **Monotonicidad**: puede especificar que la salida solo debe aumentar/disminuir con respecto a una entrada. En nuestro ejemplo, es posible que desee especificar que una mayor distancia a una cafetería solo debería disminuir la preferencia prevista del usuario.

<p align="center"> <img src="images/linear_fit.png"> <img src="images/flexible_fit.png"> <img src="images/regularized_fit.png"> <img src="images/monotonic_fit.png"></p>

- **Convexidad/Concavidad**: puede especificar que la forma de la función puede ser convexa o cóncava. Combinado con la monotonicidad, esto puede obligar a la función a representar retornos decrecientes con respecto a una característica determinada.

- **Unimodalidad**: puede especificar que la función debe tener un pico único o un valle único. Esto le permite representar funciones que tienen un *punto óptimo* con respecto a una característica.

- **Confianza por pares**: esta restricción funciona en un par de características y sugiere que una característica de entrada refleja semánticamente la confianza en otra característica. Por ejemplo, una mayor cantidad de reseñas le da más confianza en la calificación promedio de estrellas de un restaurante. El modelo será más sensible con respecto a la calificación de estrellas (es decir, tendrá una mayor pendiente con respecto a la calificación) cuando el número de reseñas sea mayor.

### Flexibilidad controlada con regularizadores

Además de las restricciones de forma, la cuadrícula de TensorFlow proporciona una serie de regularizadores para controlar la flexibilidad y suavidad de la función para cada capa.

- **Regularizador laplaciano**: las salidas de cuadrículas/vértices de calibración/puntos clave se regularizan hacia los valores de sus respectivos vecinos. Esto da como resultado una función *más plana*.

- **Regularizador de Hesse**: penaliza la primera derivada de la capa de calibración PWL para hacer la función *más lineal*.

- **Regularizador de pliegues**: penaliza la segunda derivada de la capa de calibración PWL para evitar cambios bruscos en la curvatura. Hace que la función sea más fluida.

- **Regularizador de torsión**: las salidas de la cuadrícula se regularizarán para evitar la torsión entre las funciones. En otras palabras, el modelo se regularizará hacia la independencia entre las contribuciones de las características.

### Mezclar y combinar con otras capas de Keras

Puede usar capas TF Lattice en combinación con otras capas de Keras para construir modelos parcialmente restringidos o regularizados. Por ejemplo, se pueden usar capas de calibración de cuadrícula o PWL en la última capa de redes más profundas que incluyen incorporaciones u otras capas de Keras.

## Artículos

- [Deontological Ethics By Monotonicity Shape Constraints](https://arxiv.org/abs/2001.11990), Serena Wang, Maya Gupta, International Conference on Artificial Intelligence and Statistics (AISTATS), 2020
- [Shape Constraints for Set Functions](http://proceedings.mlr.press/v97/cotter19a.html), Andrew Cotter, Maya Gupta, H. Jiang, Erez Louidor, Jim Muller, Taman Narayan, Serena Wang, Tao Zhu. International Conference on Machine Learning (ICML), 2019
- [Diminishing Returns Shape Constraints for Interpretability and Regularization](https://papers.nips.cc/paper/7916-diminishing-returns-shape-constraints-for-interpretability-and-regularization), Maya Gupta, Dara Bahri, Andrew Cotter, Kevin Canini, Advances in Neural Information Processing Systems (NeurIPS), 2018
- [Deep Lattice Networks and Partial Monotonic Functions](https://research.google.com/pubs/pub46327.html), Seungil You, Kevin Canini, David Ding, Jan Pfeifer, Maya R. Gupta, Advances in Neural Information Processing Systems (NeurIPS), 2017
- [Fast and Flexible Monotonic Functions with Ensembles of Lattices](https://papers.nips.cc/paper/6377-fast-and-flexible-monotonic-functions-with-ensembles-of-lattices), Mahdi Milani Fard, Kevin Canini, Andrew Cotter, Jan Pfeifer, Maya Gupta, Advances in Neural Information Processing Systems (NeurIPS), 2016
- [Monotonic Calibrated Interpolated Look-Up Tables](http://jmlr.org/papers/v17/15-243.html), Maya Gupta, Andrew Cotter, Jan Pfeifer, Konstantin Voevodski, Kevin Canini, Alexander Mangylov, Wojciech Moczydlowski, Alexander van Esbroeck, Journal of Machine Learning Research (JMLR), 2016
- [Optimized Regression for Efficient Function Evaluation](http://ieeexplore.ieee.org/document/6203580/), Eric Garcia, Raman Arora, Maya R. Gupta, IEEE Transactions on Image Processing, 2012
- [Lattice Regression](https://papers.nips.cc/paper/3694-lattice-regression), Eric Garcia, Maya Gupta, Advances in Neural Information Processing Systems (NeurIPS), 2009

## Tutoriales y documentos de la API

Para arquitecturas comunes de modelos, puede usar [modelos prefabricados de Keras](tutorials/premade_models.ipynb) o [estimadores prediseñados](tutorials/canned_estimators.ipynb). También puede crear modelos personalizados con [capas de TF Lattice Keras](tutorials/keras_layers.ipynb) o mezclarlos y combinarlos con otras capas de Keras. Consulte los [documentos completos de la API](https://www.tensorflow.org/lattice/api_docs/python/tfl) para obtener más detalles.
