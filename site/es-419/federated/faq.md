# Preguntas frecuentes

## ¿Se puede usar TensorFlow Federated en entornos de producción, por ejemplo, en teléfonos móviles?

Por el momento, no. Si bien desarrollamos TFF pensando en su implementación en dispositivos reales, por el momento no ofrecemos ninguna herramienta para este fin. La versión actual está pensada para usos experimentales, como expresar nuevos algoritmos federados o probar el aprendizaje federado con sus propios conjuntos de datos, utilizando el tiempo de ejecución de simulación incluido.

Esperamos que, con el tiempo, el ecosistema de código abierto en torno a TFF evolucione e incorpore tiempos de ejecución dirigidos a plataformas de implementación física.

## ¿Cómo se usa TFF para experimentar con grandes conjuntos de datos?

El tiempo de ejecución que se incluye por defecto en la versión inicial de TFF está pensado exclusivamente para experimentos pequeños, como los que se describen en nuestros tutoriales, en los que todos los datos (de todos los clientes simulados) caben simultáneamente en la memoria de una sola máquina, y todo el experimento se ejecuta a nivel local dentro del bloc de notas colab.

Nuestra planificación a corto plazo incluye un tiempo de ejecución de alto rendimiento para experimentos con conjuntos de datos muy grandes y un gran número de clientes.

## ¿Cómo puedo asegurarme de que la aleatoriedad en TFF se ajuste a mis expectativas?

Dado que TFF tiene computación federada incorporada en su núcleo, el autor de TFF no debe asumir control sobre dónde y cómo se ingresan `Session` de TensorFlow o se llama `run` dentro de esas sesiones. La semántica de la aleatoriedad puede depender de la entrada y salida de las `Session` de TensorFlow si se establecen semillas. Recomendamos usar aleatoriedad estilo TensorFlow 2, usando, por ejemplo `tf.random.experimental.Generator` a partir de TF 1.14. Esto utiliza una `tf.Variable` para gestionar su estado interno.

Para ayudar a gestionar las expectativas, TFF permite que el TensorFlow que serializa tenga semillas de nivel de operación, pero no semillas de nivel de gráfico. Esto se debe a que la semántica de las semillas de nivel de operación debería ser más clara en la configuración de TFF: se generará una secuencia determinista en cada invocación de una función envuelta como `tf_computation`, y solo dentro de esta invocación se mantendrán las garantías otorgadas por el generador de números pseudoaleatorios. Tenga en cuenta que esto no es exactamente lo mismo que la semántica de llamar a una `tf.function` en modo eager; TFF efectivamente entra y sale de una `tf.Session` única cada vez que se invoca `tf_computation`, mientras que llamar repetidamente a una función en modo eager es análogo a llamar `sess.run` en el tensor de salida varias veces dentro de la misma sesión.

## ¿Cómo puedo contribuir?

Consulte el archivo [README](https://github.com/tensorflow/federated/blob/main/CONTRIBUTING.md), las pautas de [contribución](https://github.com/tensorflow/federated/blob/main/CONTRIBUTING.md) y las [colaboraciones](collaborations/README.md).

## ¿Qué relación hay entre FedJAX y TensorFlow Federated?

TensorFlow Federated (TFF) es un marco completo para el aprendizaje y el análisis federados que se diseñó para facilitar la composición de diferentes algoritmos y características, y para permitir la portabilidad de códigos a diferentes escenarios de simulación e implementación. TFF proporciona un tiempo de ejecución escalable y admite muchos algoritmos de privacidad, compresión y optimización a través de sus API estándar. TFF también admite [muchos tipos de investigación de FL](https://www.tensorflow.org/federated/tff_for_research), con una colección de ejemplos de artículos publicados por Google que aparecen en el [repositorio de investigación de Google](https://github.com/google-research/federated).

Por el contrario, [FedJAX](https://github.com/google/fedjax) es una biblioteca de simulación ligera basada en Python y JAX que se centra en la facilidad de uso y la creación rápida de prototipos de algoritmos de aprendizaje federado con fines de investigación. TensorFlow Federated y FedJAX se desarrollan como proyectos separados, sin expectativas de portabilidad de código.
