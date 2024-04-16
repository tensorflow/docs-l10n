# Conceptos fundamentales del aprendizaje automático cuántico

En el <a href="https://ai.googleblog.com/2019/10/quantum-supremacy-using-programmable.html" class="external">experimento cuántico, más allá de lo clásico</a> de Google se usaron 53 bits cuánticos *ruidosos* para demostrar que, en 200 segundos, se puede hacer un cálculo en una computadora cuántica, que tardaría 10 000 años en la computadora clásica más grande del mundo con los algoritmos existentes hasta el momento. Este hito marca el comienzo de la era de la computación <a href="https://quantum-journal.org/papers/q-2018-08-06-79/" class="external">cuántica de escala intermedia ruidosa</a> (NISQ). En los años siguientes, se espera que los dispositivos cuánticos con decenas de miles de bits cuánticos (cúbits) se vuelvan realidad.

## Computación cuántica

La computación cuántica depende de propiedades de la mecánica cuántica para calcular problemas que estarían fuera del alcance de las computadoras clásicas. Una computadora cuántica usa *bits cuánticos*. Los bits cuánticos o cúbits son como los bits comunes de las computadoras, pero con la habilidad extra de que se pueden *superponer* y pueden compartir *entrelazamiento* entre ellos.

Las computadoras clásicas realizan operaciones clásicas deterministas o pueden emular procesos probabilísticos con métodos de muestreo. Al aprovechar la superposición y el entrelazamiento, las computadoras cuánticas pueden realizar operaciones cuánticas que son difíciles de emular a escala con computadoras clásicas. Las ideas para aprovechar la computación cuántica (NISQ) incluye la optimización, la simulación cuántica, la criptografía y el aprendizaje automático.

## Aprendizaje automático cuántico

El *aprendizaje automático cuántico* (QML) se basa en dos conceptos: el de los *datos cuánticos* y el de los *modelos híbridos cuántico-clásicos*.

### Datos cuánticos

Los *datos cuánticos* son cualquier fuente de datos que ocurre en un sistema cuántico natural o artificial. Pueden ser datos generados por una computadora cuántica como las muestras reunidas por el <a href="https://www.nature.com/articles/s41586-019-1666-5" class="external">procesador Sycamore</a> para la demostración de la supremacía cuántica de Google. Los datos cuánticos exhiben superposición y entrelazamiento, lo que deriva en distribuciones probabilísticas conjuntas para las que, en el caso de las computadoras clásicas, habría que contar con una cantidad exponencial de datos que las representara y almacenara. Con el experimento sobre supremacía cuántica se demostró que es posible tomar una muestra de una distribución de probabilidad conjunta extremadamente compleja de espacio de Hilbert 2^53.

Los datos cuánticos generados por los procesadores NISQ son ruidosos y están típicamente entrelazados justo antes de que se produzca la medición. Con las técnicas heurísticas de aprendizaje automático se pueden crear modelos que maximicen la extracción de información clásica útil a partir de los datos entrelazados ruidosos. En la biblioteca de TensorFlow Quantum (TFQ) se ofrecen modelos primitivos para desarrollar que desenlazan y generalizan las correlaciones de los datos cuánticos; hecho que abre las puertas a otras oportunidades de mejorar los algoritmos cuánticos existentes o, incluso, de descubrir nuevos algoritmos cuánticos.

Los siguientes son ejemplos de datos cuánticos que se pueden generar o simular en un dispositivo cuántico:

- *Simulación química*: extrae la información sobre las estructuras y dinámicas químicas con aplicaciones potenciales en ciencias de los materiales, química computacional, biología computacional y descubrimiento de medicamentos.
- *Simulación de materia cuántica*: el modelo y diseño de la superconductividad a temperatura alta u otros estados exóticos de la materia que exhiben efectos cuánticos de muchos cuerpos.
- *Control cuántico*: los modelos híbridos cuánticos-clásicos pueden entrenarse variablemente para realizar un control óptimo de ciclo cerrado o abierto y, también, para la calibración y la mitigación de errores. Incluye estrategias de detección y corrección de errores para dispositivos y procesadores cuánticos.
- *Redes de comunicaciones cuánticas*: usan el aprendizaje automático para discriminar entre estados cuánticos no ortogonales, con aplicación en el diseño y la construcción de repetidores cuánticos estructurados, receptores cuánticos y unidades de purificación.
- *Metrología cuántica*: mediciones de alta precisión mejoradas con cuántica, como el sensado cuántico y el procesamiento cuántico de imágenes, que se hacen de manera inherente en sondeos con dispositivos cuánticos a pequeña escala, y se podrían diseñar o mejorar con modelos cuánticos variables.

### Modelos híbridos cuántico-clásicos

Un modelo puede representar y generalizar datos con un origen mecánico cuántico. Debido a que los procesadores cuánticos de corto plazo todavía son bastante pequeños y ruidosos, los modelos cuánticos no pueden generalizar datos cuánticos solamente con procesadores cuánticos. Los procesadores NISQ deben trabajar en forma común y coordinada con los coprocesadores clásicos para volverse más efectivos. Debido a que TensorFlow ya trabaja con la computación heterogénea (en las CPU, las GPU y las TPU) se usa como plataforma de base para experimentar con algoritmos híbridos cuántico-clásicos.

Se usa una *red neuronal cuántica* (QNN) para describir un modelo computacional cuántico parametrizado que se ejecute mejor en computadoras cuánticas. Por lo general, este término es intercambiable con el *circuito cuántico parametrizado* (PQC).

## Investigación

Durante la era NISQ, los algoritmos cuánticos con velocidades mayores conocidas a las de los algoritmos clásicos, como el <a href="https://arxiv.org/abs/quant-ph/9508027" class="external">algoritmo de factorización de Shor</a> o el <a href="https://arxiv.org/abs/quant-ph/9605043" class="external">algoritmo de investigación de Grover</a>, todavía no son posibles a una escala significativa.

Uno de los objetivos de TensorFlow Quantum es ayudar a descubrir algoritmos para la era NISQ, con un interés en particular en lo siguiente:

1. El *uso del aprendizaje automático clásico para mejorar los algoritmos NISQ.* Existe la esperanza de que las técnicas de aprendizaje automático clásico mejoren nuestra capacidad de comprender la computación cuántica. En el <a href="https://arxiv.org/abs/1907.05415" class="external">metaaprendizaje para redes neuronales cuánticas a través de redes neuronales recurrentes clásicas</a>, se usa una red neuronal recurrente (RNN) para descubrir la optimización de los parámetros de control para que algoritmos como el QAOA y el VQE sean más eficientes que los optimizadores simples existentes. Y el <a href="https://www.nature.com/articles/s41534-019-0141-3" class="external">aprendizaje automático para control cuántico</a> usa el aprendizaje por refuerzo para ayudar a mitigar errores y producir puertas cuánticas de mayor calidad.
2. *Modelado de datos cuánticos con circuitos cuánticos.* El modelado clásico de datos cuánticos es posible si contamos con una descripción exacta de la fuente de los datos, aunque a veces es imposible. Para resolver este problema, podemos probar el modelado en la misma computadora cuántica y medir u observar las estadísticas importantes. Las <a href="https://www.nature.com/articles/s41567-019-0648-8" class="external">redes neuronales convolucionales cuánticas</a> muestran un circuito diseñado con una estructura análoga a la de una red neuronal convolucional (CNN) para detectar diferentes fases topológicas de la materia. La computadora cuántica contiene los datos y el modelo. El procesador clásico solamente ve las muestras de medición de las salidas del modelo, pero nunca ve los datos en sí mismos. En una <a href="https://arxiv.org/abs/1711.07500" class="external">renormalización de entrelazado robusto hecha en una computadora cuántica</a>, los autores aprenden a comprimir la información sobre los sistemas cuánticos de muchos cuerpos con un modelo DMERA.

Las siguientes son otras áreas de interés relacionadas con el aprendizaje automático cuántico:

- El modelado de datos puramente clásicos en computadoras cuánticas.
- Los algoritmos clásicos inspirados en la cuántica.
- El <a href="https://arxiv.org/abs/1810.03787" class="external">aprendizaje supervisado con clasificadores cuánticos</a>.
- El aprendizaje adaptativo sensible a las capas, para redes neuronales cuánticas.
- El <a href="https://arxiv.org/abs/1909.12264" class="external">aprendizaje automático cuántico</a>.
- El <a href="https://arxiv.org/abs/1910.02071" class="external">modelado generativo de estados cuánticos mezclados</a> .
- La <a href="https://arxiv.org/abs/1802.06002" class="external">clasificación con redes neuronales cuánticas en procesadores de corto plazo</a>.
