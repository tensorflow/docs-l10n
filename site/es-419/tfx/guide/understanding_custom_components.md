# Explicación de los componentes personalizados de TFX

Las canalizaciones de TFX le permiten orquestar su flujo de trabajo de aprendizaje automático (ML) en orquestadores, como: Apache Airflow, Apache Beam y Kubeflow Pipelines. Las canalizaciones organizan su flujo de trabajo en una secuencia de componentes, donde cada componente ejecuta un paso en su flujo de trabajo de ML. Los componentes estándar de TFX ofrecen una funcionalidad comprobada para ayudarlo a comenzar a compilar fácilmente un flujo de trabajo de aprendizaje automático. También puede incluir componentes personalizados en su flujo de trabajo. Los componentes personalizados le permiten ampliar su flujo de trabajo de ML al ejecutar las siguientes acciones:

- Compilar componentes que se adapten a sus necesidades, como la ingesta de datos desde un sistema propietario.
- Aplicar aumento de datos, muestreo ascendente o descendente.
- Detectar anomalías en función de intervalos de confianza o errores de reproducción del codificador automático.
- Interconectar con sistemas externos, como los soportes técnicos, para alertar y monitorear.
- Aplicar etiquetas a ejemplos sin etiquetar.
- Integrar herramientas compiladas con lenguajes distintos de Python en su flujo de trabajo de aprendizaje automático, como la ejecución de análisis de datos con R.

Al combinar componentes estándar y componentes personalizados, puede compilar un flujo de trabajo de aprendizaje automático que satisfaga sus necesidades y, al mismo tiempo, aprovechar las prácticas recomendadas integradas en los componentes estándar de TFX.

Esta guía describe los conceptos necesarios para comprender los componentes personalizados de TFX y las diferentes formas en que se pueden compilar componentes personalizados.

## Anatomía de un componente de TFX

Esta sección ofrece una descripción general de alto nivel de la composición de un componente de TFX. Si es nuevo en las canalizaciones de TFX, puede [aprender los conceptos básicos al leer la guía para comprender las canalizaciones de TFX](understanding_tfx_pipelines.md).

Los componentes de TFX se componen de una especificación de componente y una clase ejecutora que están empaquetadas en una clase de interfaz de componente.

Una *especificación de componente* define el contrato de entrada y salida del componente. Este contrato especifica los artefactos de entrada y salida del componente, y los parámetros que se usan para la ejecución del componente.

La clase *ejecutor* de un componente ofrece la implementación del trabajo que lleva a cabo el componente.

Una clase *interfaz de componente* combina la especificación del componente con el ejecutor para su uso como componente en una canalización de TFX.

### Componentes de TFX en tiempo de ejecución

Cuando una canalización ejecuta un componente de TFX, el componente se ejecuta en tres fases:

1. En primer lugar, el Driver usa la especificación del componente para recuperar los artefactos necesarios del almacén de metadatos y pasarlos al componente.
2. A continuación, el Executor hace el trabajo del componente.
3. Luego, el Publisher usa la especificación del componente y los resultados del ejecutor para almacenar las salidas del componente en el almacén de metadatos.

![Anatomía de los componentes](images/component.png)

La mayoría de las implementaciones de componentes personalizados no requieren que personalice el Driver o el Publisher. Normalmente, las modificaciones al Driver y al Publisher solo deberían ser necesarias si desea cambiar la interacción entre los componentes de su canalización y el almacén de metadatos. Si solo desea cambiar las entradas, salidas o parámetros de su componente, simplemente tiene que modificar la *especificación del componente*.

## Tipos de componentes personalizados

Hay tres tipos de componentes personalizados: componentes basados ​​en funciones de Python, componentes basados ​​en contenedores y componentes totalmente personalizados. En las próximas secciones se describen los diferentes tipos de componentes y los casos en los que se debe aplicar cada enfoque.

### Componentes basados ​​en funciones de Python

Los componentes basados ​​en funciones de Python son más fáciles de construir que los componentes basados ​​en contenedores o los componentes totalmente personalizados. La especificación del componente se define en los argumentos de la función de Python mediante anotaciones de tipo que describen si un argumento es un artefacto de entrada, un artefacto de salida o un parámetro. El cuerpo de la función define el ejecutor del componente. La interfaz del componente se define al agregar el [decorador `@component`](https://github.com/tensorflow/tfx/blob/master/tfx/dsl/component/experimental/decorators.py) {: .external} a la función.

Al decorar su función con el decorador `@component` y definir los argumentos de la función con anotaciones de tipo, puede compilar un componente sin la complejidad de compilar una especificación de componente, un ejecutor y una interfaz de componente.

Aprenda a [compilar componentes basados ​​en funciones de Python](custom_function_component.md).

### Componentes basados ​​en contenedores

Los componentes basados ​​en contenedores ofrecen la flexibilidad de integrar código escrito en cualquier lenguaje en su canalización, siempre que ese código se pueda ejecutar en un contenedor Docker. Para compilar un componente basado en contenedores, debe compilar una imagen de contenedor Docker que contenga el código ejecutable de su componente. Luego, tiene que llamar a la [función `create_container_component`](https://github.com/tensorflow/tfx/blob/master/tfx/dsl/component/experimental/container_component.py) {:.external} para definir lo siguiente:

- Las entradas, las salidas y los parámetros de la especificación de su componente.
- La imagen del contenedor y el comando que ejecuta el ejecutor del componente.

Esta función devuelve una instancia de un componente que puede incluir en su definición de canalización.

Este enfoque es más complejo que compilar un componente basado en funciones de Python, ya que requiere empaquetar su código como una imagen de contenedor. Este enfoque es más adecuado para incluir código que no sea Python en su canalización o para compilar componentes de Python con dependencias o entornos de ejecución complejos.

Aprenda a [compilar componentes basados ​​en contenedores](container_component.md).

### Componentes totalmente personalizados

Los componentes totalmente personalizados le permiten compilar componentes a través de la definición de la especificación del componente, el ejecutor y las clases de interfaz del componente. Este enfoque le permite reutilizar y ampliar un componente estándar para adaptarlo a sus necesidades.

Si un componente existente se define con las mismas entradas y salidas que el componente personalizado que se está desarrollando, basta con anular la clase Executor del componente existente. Esto significa que puede reutilizar una especificación de componente e implementar un nuevo ejecutor que derive de un componente existente. De este modo, se reutiliza la funcionalidad incorporada en los componentes existentes y se implementa solo la funcionalidad necesaria.

Sin embargo, si las entradas y salidas de su nuevo componente son únicas, puede definir una *especificación de componentes* completamente nueva.

Este enfoque es mejor para reutilizar las especificaciones y los ejecutores de componentes existentes.

Aprenda a [compilar componentes totalmente personalizados](custom_component.md).
