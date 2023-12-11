# Cómo usar TFF para la investigación sobre el aprendizaje federado

<!-- Note that some section headings are used as deep links into the document.
     If you update those section headings, please make sure you also update
     any links to the section. -->

## Descripción general

TFF es un marco de trabajo extensible y potente para llevar a cabo investigaciones sobre aprendizaje federado (FL) por medio de la simulación de cálculos federados en conjuntos de datos proxy realistas. Esta página describe los principales conceptos y componentes relevantes para las simulaciones de investigación, así como una guía detallada para llevar a cabo diferentes tipos de investigación en TFF.

## La estructura típica del código de investigación en TFF.

Una simulación de FL de investigación que se implementa en TFF generalmente consta de tres tipos principales de lógica.

1. Piezas individuales de código TensorFlow, comúnmente `tf.function`s, que encapsulan la lógica que se ejecuta en una única ubicación (por ejemplo, en clientes o en un servidor). Este código suele escribirse y probarse sin referencias `tff.*`, y se puede reutilizar fuera de TFF. Por ejemplo, el [bucle de entrenamiento del cliente en el promediado federado](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/simple_fedavg/simple_fedavg_tf.py#L184-L222) se implementa en este nivel.

2. Lógica de orquestación federada de TensorFlow, que une las `tf.function`s individuales de 1. al envolverlas como `tff.tf_computation`s y luego orquestarlas mediante abstracciones como `tff.federated_broadcast` y `tff.federated_mean` dentro de una `tff.federated_computation`. Consulte, por ejemplo, esta [orquestación para el promediado federado](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/simple_fedavg/simple_fedavg_tff.py#L112-L140).

3. Un script de controlador externo que simula la lógica de control de un sistema de FL de producción, seleccionando clientes simulados de un conjunto de datos para luego ejecutar cálculos federados definidos en 2. en esos clientes. Por ejemplo, [un controlador de experimento EMNIST federado](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/simple_fedavg/emnist_fedavg_main.py).

## Conjuntos de datos de aprendizaje federado

TensorFlow federado [aloja múltiples conjuntos de datos](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets) que son representativos de las características de los problemas del mundo real que podrían resolverse gracias al aprendizaje federado.

Nota: Cualquier marco de aprendizaje automático basado en Python puede consumir estos conjuntos de datos como arreglos Numpy, tal y como se documenta en la [ClientData API](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/ClientData).

Entre los conjuntos de datos se incluyen los siguientes:

- [**StackOverflow**.](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/stackoverflow/load_data) Un conjunto de datos de texto realista para tareas de modelado lingüístico o aprendizaje supervisado, con 342 477 usuarios únicos 135 818 730 ejemplos (frases) en el conjunto de entrenamiento.

- [**EMNIST federado**.](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/emnist/load_data) Un preprocesamiento federado del conjunto de datos de caracteres y dígitos de EMNIST, en el que cada cliente corresponde a un escritor diferente. El conjunto de entrenamiento completo contiene 3400 usuarios con 671 585 ejemplos de 62 etiquetas.

- [**Shakespeare**.](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/shakespeare/load_data) Un conjunto de datos de texto más pequeño a nivel de personaje que se basa en las obras completas de William Shakespeare. El conjunto de datos consta de 715 usuarios (personajes de obras de Shakespeare), donde cada ejemplo corresponde a un conjunto contiguo de líneas que dice el personaje en una obra en particular.

- [**CIFAR-100**.](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/cifar100/load_data) Una partición federada del conjunto de datos CIFAR-100 entre 500 clientes de entrenamiento y 100 clientes de prueba. Cada cliente tiene 100 ejemplos únicos. La partición se lleva a cabo de forma que se cree una heterogeneidad más realista entre los clientes. Si desea obtener más detalles, consulte la [API](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/cifar100/load_data).

- [**Conjunto de datos Google Landmark v2.**](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/gldv2/load_data) El conjunto de datos consta de fotos de varios lugares emblemáticos del mundo, con imágenes agrupadas por fotógrafo para lograr una partición federada de los datos. Hay dos tipos de conjuntos de datos disponibles: uno más pequeño, con 233 clientes que tiene 23 080 imágenes, y uno más grande con 1262 clientes y 164 172 imágenes.

- [**CelebA.**](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/celeba/load_data) Un conjunto de datos de ejemplos (imagen y atributos faciales) de rostros de famosos. El conjunto de datos federado agrupa los ejemplos de cada famoso para formar un cliente. Hay 9343 clientes, cada uno con al menos 5 ejemplos. El conjunto de datos se puede dividir en grupos de entrenamiento y de prueba, ya sea por clientes o por ejemplos.

- [**iNaturalist.**](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/inaturalist/load_data) Un conjunto de datos consta de fotos de diferentes especies. El conjunto de datos contiene 120 300 imágenes de 1203 especies. Hay siete versiones disponibles del conjunto de datos. Una de ellas se agrupa por fotógrafo y consta de 9257 clientes. Los demás conjuntos de datos se agrupan por la ubicación geográfica en la que se tomó la foto. Estas seis versiones del conjunto de datos constan de entre 11 y 3606 clientes.

## Simulaciones de alto rendimiento

Aunque el tiempo de ejecución de una *simulación* de FL no es una medida relevante para evaluar algoritmos (ya que el hardware de simulación no es representativo de los entornos reales de implementación del FL), la capacidad de ejecutar simulaciones de FL con rapidez resulta fundamental para la productividad de la investigación. En este sentido, TFF dedica grandes esfuerzos a mejorar el rendimiento de los tiempos de ejecución en una o varias máquinas. La documentación está en desarrollo, pero por ahora consulte las instrucciones sobre [simulaciones de TFF con aceleradores](https://www.tensorflow.org/federated/tutorials/simulations_with_accelerators) y las instrucciones de [configuración de simulaciones con TFF en GCP](https://www.tensorflow.org/federated/gcp_setup). El tiempo de ejecución de TFF de alto rendimiento se encuentra activado por defecto.

## TFF para diferentes áreas de investigación

### Algoritmos de optimización federados

La investigación sobre algoritmos de optimización federados se puede abordar de diferentes maneras en TFF, en función del nivel de personalización que se desee.

Puede acceder a una implementación autónoma mínima del algoritmo de [promediado federado](https://arxiv.org/abs/1602.05629) [aquí](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/simple_fedavg). El código incluye [funciones de TF](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/simple_fedavg/simple_fedavg_tf.py) para el cálculo local, [cálculos de TFF](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/simple_fedavg/simple_fedavg_tff.py) para la orquestación y un [script de controlador](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/simple_fedavg/emnist_fedavg_main.py) en el conjunto de datos EMNIST como ejemplo. Estos archivos pueden adaptarse fácilmente para aplicaciones personalizadas y cambios algorítmicos siguiendo las instrucciones detalladas en el [README](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/simple_fedavg/README.md).

Puede acceder a una implementación más general del promediado federado [aquí](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/learning/algorithms/fed_avg.py). Esta implementación admite técnicas de optimización más sofisticadas, como el uso de diferentes optimizadores tanto en el servidor como en el cliente. [Aquí](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/learning/algorithms/) se pueden encontrar otros algoritmos de aprendizaje federado, incluida la agrupación en clústeres de k-medias federados.

### Compresión de actualización del modelo

La compresión con pérdidas de las actualizaciones del modelo puede traducirse en una reducción de los costos de comunicación, lo que a su vez puede generar una reducción del tiempo total de entrenamiento.

Para reproducir un [artículo](https://arxiv.org/abs/2201.02664), consulte [este proyecto de investigación](https://github.com/google-research/federated/tree/master/compressed_communication). Para implementar un algoritmo de compresión personalizado, consulte [comparison_methods](https://github.com/google-research/federated/tree/master/compressed_communication/aggregators/comparison_methods) en el proyecto para conocer las líneas de base como ejemplo, y el [tutorial sobre agregadores de TFF](https://www.tensorflow.org/federated/tutorials/custom_aggregators) en caso de que aún no esté familiarizado con este tema.

### Privacidad diferencial

TFF se puede combinar con la biblioteca [TensorFlow Privacy](https://github.com/tensorflow/privacy) para permitir la investigación de nuevos algoritmos para el entrenamiento federado de modelos con privacidad diferencial. Si desea ver un ejemplo de entrenamiento con DP en el que se utiliza [el algoritmo básico DP-FedAvg](https://arxiv.org/abs/1710.06963) y sus [extensiones](https://arxiv.org/abs/1812.06210), consulte [este controlador de experimentos](https://github.com/google-research/federated/blob/master/differential_privacy/stackoverflow/run_federated.py).

Si desea implementar un algoritmo DP personalizado y aplicarlo a las actualizaciones agregadas de promediado federado, puede implementar un nuevo algoritmo DP promedio como una subclase de [`tensorflow_privacy.DPQuery`](https://github.com/tensorflow/privacy/blob/master/tensorflow_privacy/privacy/dp_query/dp_query.py#L54) y crear un `tff.aggregators.DifferentiallyPrivateFactory` con una instancia de su consulta. Puede consultar un ejemplo de implementación del [algoritmo DP-FTRL](https://arxiv.org/abs/2103.00039) [aquí](https://github.com/google-research/federated/blob/master/dp_ftrl/dp_fedavg.py)

Las GAN federadas (que se describen [a continuación](#generative_adversarial_networks)) son otro ejemplo de un proyecto de TFF que aplica la privacidad diferencial a nivel de usuario (por ejemplo, [aquí en código](https://github.com/google-research/federated/blob/master/gans/tff_gans.py#L144)).

### Robustez y ataques

TFF también se puede usar para simular los ataques dirigidos a sistemas de aprendizaje federados y defensas diferenciales basadas en la privacidad que se consideran en *[¿Puede realmente vulnerarse el aprendizaje federado?](https://arxiv.org/abs/1911.07963)*. Esto se hace mediante la creación de un proceso iterativo con clientes potencialmente maliciosos (consulte [`build_federated_averaging_process_attacked`](https://github.com/tensorflow/federated/blob/6477a3dba6e7d852191bfd733f651fad84b82eab/federated_research/targeted_attack/attacked_fedavg.py#L412)). El directorio [target_attack](https://github.com/tensorflow/federated/tree/6477a3dba6e7d852191bfd733f651fad84b82eab/federated_research/targeted_attack) contiene más detalles.

- Se pueden implementar nuevos algoritmos de ataque si se escribe una función de actualización del cliente que sea una función de TensorFlow. Consulte [`ClientProjectBoost`](https://github.com/tensorflow/federated/blob/6477a3dba6e7d852191bfd733f651fad84b82eab/federated_research/targeted_attack/attacked_fedavg.py#L460) para acceder a un ejemplo.
- Se pueden implementar nuevas defensas al personalizar ['tff.utils.StatefulAggregateFn'](https://github.com/tensorflow/federated/blob/6477a3dba6e7d852191bfd733f651fad84b82eab/tensorflow_federated/python/core/utils/computation_utils.py#L103) que agrega las salidas del cliente para obtener una actualización global.

Si desea ver un script de ejemplo para la simulación, consulte [`emnist_with_targeted_attack.py`](https://github.com/tensorflow/federated/blob/6477a3dba6e7d852191bfd733f651fad84b82eab/federated_research/targeted_attack/emnist_with_targeted_attack.py).

### Redes generativas adversativas

Las GAN ofrecen un [patrón de orquestación federada](https://github.com/google-research/federated/blob/master/gans/tff_gans.py#L266-L316) interesante que difiere un poco del promediado federado estándar. Involucran dos redes distintas (el generador y el discriminador) y cada una de ellas se entrena con su propio paso de optimización.

TFF se puede usar para la investigación sobre el entrenamiento federado de GAN. Por ejemplo, el algoritmo DP-FedAvg-GAN que se presentó en un [trabajo reciente](https://arxiv.org/abs/1911.06679) se [implementa en TFF](https://github.com/tensorflow/federated/tree/main/federated_research/gans). Este trabajo demuestra la eficacia de combinar el aprendizaje federado, los modelos generativos y la [privacidad diferencial](#differential_privacy).

### Personalización

La personalización en el marco del aprendizaje federado es un campo de investigación activo. El objetivo de la personalización es facilitar diferentes modelos de inferencia a diferentes usuarios. Este problema se puede abordar de distintas maneras.

Un enfoque consiste en dejar que cada cliente ajuste un único modelo global (entrenado mediante aprendizaje federado) con sus datos locales. Este enfoque está vinculado al metaaprendizaje (consulte, por ejemplo, [este artículo](https://arxiv.org/abs/1909.12488). En [`emnist_p13n_main.py`](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/personalization/emnist_p13n_main.py) se ofrece un ejemplo de este enfoque. Para explorar y comparar diferentes estrategias de personalización, puede hacer lo siguiente:

- Definir una estrategia de personalización mediante la implementación de una `tf.function` que parta de un modelo inicial, entrene y evalúe un modelo personalizado a partir de los conjuntos de datos locales de cada cliente. Puede ver un ejemplo en [`build_personalize_fn`](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/personalization/p13n_utils.py).

- Definir un `OrderedDict` que asigne los nombres de las estrategias a las estrategias de personalización correspondientes y usarlo como argumento de `personalize_fn_dict` en [`tff.learning.build_personalization_eval_computation`](https://www.tensorflow.org/federated/api_docs/python/tff/learning/build_personalization_eval_computation).

Otro enfoque consiste en evitar el entrenamiento de un modelo totalmente global y entrenar una parte del modelo de forma totalmente local. En [esta entrada del blog](https://ai.googleblog.com/2021/12/a-scalable-approach-for-partially-local.html) se describe un ejemplo de este enfoque. Esta estrategia también está vinculada al metaaprendizaje, consulte [este artículo](https://arxiv.org/abs/2102.03448). Si desea explorar el aprendizaje federado parcialmente local, puede hacer lo siguiente:

- Consulte este [tutorial](https://www.tensorflow.org/federated/tutorials/federated_reconstruction_for_matrix_factorization) para acceder a un ejemplo de código completo en el que se apliquen la reconstrucción federada y los [ejercicios de seguimiento](https://www.tensorflow.org/federated/tutorials/federated_reconstruction_for_matrix_factorization#further_explorations).

- Use [`tff.learning.reconstruction.build_training_process`](https://www.tensorflow.org/federated/api_docs/python/tff/learning/reconstruction/build_training_process) para crear un proceso de entrenamiento parcialmente local y modifique `dataset_split_fn` para personalizar el comportamiento del proceso.
