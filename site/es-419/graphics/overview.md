# Descripción general

En los últimos años se han multiplicado las capas gráficas diferenciables que pueden insertarse en arquitecturas de redes neuronales. Desde transformadores espaciales hasta renderizadores de gráficos diferenciables, estas nuevas capas aprovechan los conocimientos adquiridos a lo largo de años de investigación en visión artificial y computación gráfica para construir arquitecturas de red nuevas y más eficientes. El modelado explícito de las restricciones y los antecedentes geométricos en redes neuronales abre la puerta a arquitecturas que pueden entrenarse de manera sólida, eficiente y, lo que es más importante, de forma autosupervisada.

A un alto nivel, una canalización de computación gráfica requiere una representación de objetos 3D y su posicionamiento absoluto en la escena, una descripción del material del que están hechos, luces y una cámara. Luego, un renderizador interpreta esta descripción de escena para generar una representación sintética.

<div align="center">   <img border="0" src="https://storage.googleapis.com/tensorflow-graphics/git/readme/graphics.jpg" width="600">
</div>

En contraste, un sistema de visión artificial parte de una imagen e intenta deducir los parámetros de la escena. Esto permite predecir qué objetos hay en la escena, de qué materiales están hechos y su posición y orientación tridimensionales.

<div align="center">   <img border="0" src="https://storage.googleapis.com/tensorflow-graphics/git/readme/cv.jpg" width="600">
</div>

Entrenar sistemas de aprendizaje automático capaces de resolver estas complejas tareas de visión 3D suele requerir grandes cantidades de datos. Como etiquetar datos es un proceso complejo y costoso, es importante contar con mecanismos necesarios para diseñar modelos de aprendizaje automático que puedan comprender el mundo tridimensional mientras se entrenan sin mucha supervisión. La combinación de técnicas de visión artificial y computación gráfica brinda una oportunidad única de aprovechar las grandes cantidades de datos sin etiquetar disponibles. Como se ilustra en la siguiente imagen, esto se puede lograr, por ejemplo, mediante análisis por síntesis en el que el sistema de visión extrae los parámetros de la escena y el sistema de gráficos genera una imagen basada en ellos. Si la representación coincide con la imagen original, el sistema de visión ha extraído con precisión los parámetros de la escena. En esta configuración, la visión artificial y la computación gráfica van de la mano y forman un único sistema de aprendizaje automático similar a un codificador automático, que puede entrenarse de manera autosupervisada.

<div align="center">   <img border="0" src="https://storage.googleapis.com/tensorflow-graphics/git/readme/cv_graphics.jpg" width="600">
</div>

Tensorflow Graphics se está desarrollando para ayudar a abordar este tipo de desafíos y, para hacerlo, proporciona un conjunto de gráficos y capas de geometría diferenciables (por ejemplo, cámaras, modelos de reflectancia, transformaciones espaciales, convoluciones de malla) y funcionalidades de visor 3D (por ejemplo, 3D TensorBoard) que se pueden usar para entrenar y depurar los modelos de aprendizaje automático que elija.
