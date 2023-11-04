# Citando TensorFlow

TensorFlow publica un identificador de objetos digitales (DOI por las siglas en inglés) para la base de código abierto mediante Zenodo.org: [10.5281/zenodo.4724125](https://doi.org/10.5281/zenodo.4724125)

Las guías de TensorFlow están enumeradas para menciones más abajo.

## Aprendizaje automático a gran escala en Sistemas distribuidos heterogéneos:

[Acceder a esta guía.](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45166.pdf)

**Resumen:** TensorFlow es una interfaz para expresar algoritmos de aprendizaje automático y una implementación para ejecutar estos algoritmos. Una computación expresada usando TensorFlow puede ser ejecutada con pocos o ningún cambio en una gran variedad de sistemas heterogéneos, que van desde dispositivos móviles y tablets hasta sistemas distribuidos a gran escala formados por cientos de máquinas y miles de dispositivos para la computación como pueden ser las tarjetas gráficas. El sistema es flexible y puede usarse para expresar una gran variedad de algoritmos, incluyendo algoritmos de entrenamiento e inferencia para redes neuronales profundas, y se ha usado para llevar a cabo investigación y para poner sistemas de aprendizaje automático en producción a través de más de una docena de áreas de ciencias de la computación y otros campos, incluyendo: reconocimiento de voz, visión artificial, robótica, recuperación de información, procesamiento del lenguaje natural, extracción de información geográfica o descubrimiento computacional de medicamentos. Este artículo describe la interfaz de TensorFlow y una implementación de la misma que hemos creado en Google. La interfaz de programación de aplicaciones (API) de TensorFlow y una implementación de referencia fueron publicadas como un paquete de código abierto bajo la licencia de Apache 2.0 en Noviembre del 2015 y están disponibles en www.tensorflow.org.

### <br><br>En formato BibTeX

Si usa TensorFlow en su investigación y desea citar el sistema, le sugerimos que cite este documento técnico.

<pre>@misc{tensorflow2015-whitepaper,
title={ {TensorFlow}: Large-Scale Machine Learning on Heterogeneous Systems},
url={https://www.tensorflow.org/},
note={Software available from tensorflow.org},
author={
    Mart\'{i}n~Abadi and
    Ashish~Agarwal and
    Paul~Barham and
    Eugene~Brevdo and
    Zhifeng~Chen and
    Craig~Citro and
    Greg~S.~Corrado and
    Andy~Davis and
    Jeffrey~Dean and
    Matthieu~Devin and
    Sanjay~Ghemawat and
    Ian~Goodfellow and
    Andrew~Harp and
    Geoffrey~Irving and
    Michael~Isard and
    Yangqing Jia and
    Rafal~Jozefowicz and
    Lukasz~Kaiser and
    Manjunath~Kudlur and
    Josh~Levenberg and
    Dandelion~Man\'{e} and
    Rajat~Monga and
    Sherry~Moore and
    Derek~Murray and
    Chris~Olah and
    Mike~Schuster and
    Jonathon~Shlens and
    Benoit~Steiner and
    Ilya~Sutskever and
    Kunal~Talwar and
    Paul~Tucker and
    Vincent~Vanhoucke and
    Vijay~Vasudevan and
    Fernanda~Vi\'{e}gas and
    Oriol~Vinyals and
    Pete~Warden and
    Martin~Wattenberg and
    Martin~Wicke and
    Yuan~Yu and
    Xiaoqiang~Zheng},
  year={2015},
}
</pre>

O en forma textual:

<pre>Martín Abadi, Ashish Agarwal, Paul Barham, Eugene Brevdo,
Zhifeng Chen, Craig Citro, Greg S. Corrado, Andy Davis,
Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Ian Goodfellow,
Andrew Harp, Geoffrey Irving, Michael Isard, Rafal Jozefowicz, Yangqing Jia,
Lukasz Kaiser, Manjunath Kudlur, Josh Levenberg, Dan Mané, Mike Schuster,
Rajat Monga, Sherry Moore, Derek Murray, Chris Olah, Jonathon Shlens,
Benoit Steiner, Ilya Sutskever, Kunal Talwar, Paul Tucker,
Vincent Vanhoucke, Vijay Vasudevan, Fernanda Viégas,
Oriol Vinyals, Pete Warden, Martin Wattenberg, Martin Wicke,
Yuan Yu, and Xiaoqiang Zheng.
TensorFlow: Large-scale machine learning on heterogeneous systems,
2015. Software available from tensorflow.org.
</pre>

## TensorFlow: Un sistema para el aprendizaje automático a gran escala

[Acceda a esta guía.](https://www.usenix.org/system/files/conference/osdi16/osdi16-abadi.pdf)

**Resumen:** TensorFlow es un sistema de aprendizaje automático que opera a gran escala y en entornos heterogéneos. Usa gráficos de flujo de datos para representar el cálculo, el estado compartido y las operaciones que mutan ese estado. Mapea los nodos de un gráfico de flujo de datos en muchas máquinas en un clúster y dentro de una máquina en múltiples dispositivos computacionales, incluidas CPUs multinúcleo, GPUs de propósito general y ASICs de diseño personalizado, conocidas como Unidades de Procesamiento de Tensor (TPUs). Esta arquitectura brinda flexibilidad al desarrollador de aplicaciones: mientras que en los diseños anteriores de "servidor de parámetros" la gestión del estado compartido está integrada en el sistema, TensorFlow permite a los desarrolladores experimentar con optimizaciones novedosas y algoritmos de entrenamiento. Adicionalmente, admite una variedad de aplicaciones con un enfoque en el entrenamiento y la inferencia en redes neuronales profundas. Varios servicios de Google usan TensorFlow en producción, lo hemos lanzado como un proyecto de código abierto y se ha vuelto ampliamente utilizado para la investigación de aprendizaje automático. En este documento, describimos el modelo de flujo de datos de TensorFlow y demostramos el rendimiento convincente que logra para varias aplicaciones del mundo real.
