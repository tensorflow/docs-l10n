# TensorFlow Data Validation: verificación y análisis de los datos

Una vez que sus datos estén en una canalización de TFX, puede usar componentes de TFX para analizarlos y transformarlos. Puede usar estas herramientas incluso antes de entrenar un modelo.

Hay muchas razones para analizar y transformar los datos:

- Para encontrar problemas en los datos. Los problemas comunes incluyen los siguientes:
    - Datos faltantes, como características con valores vacíos.
    - Las etiquetas se tratan como características, para que su modelo pueda ver la respuesta correcta durante el entrenamiento.
    - Características con valores fuera del rango esperado.
    - Anomalías en los datos.
    - El modelo de transferencia aprendido tiene un preprocesamiento que no coincide con los datos de entrenamiento.
- Para diseñar conjuntos de características más eficaces. Por ejemplo, se puede identificar lo siguiente:
    - Características especialmente informativas.
    - Características redundantes.
    - Características que varían tanto en escala que podrían ralentizar el aprendizaje.
    - Características con poca o ninguna información predictiva única.

Las herramientas de TFX pueden servir tanto para encontrar errores en los datos como para ayudar a diseñar características.

## TensorFlow Data Validation

- [Descripción general](#overview)
- [Validación de ejemplo basada en esquemas](#schema_based_example_validation)
- [Detección de sesgos entre entrenamiento y servicio](#skewdetect)
- [Detección de desviaciones](#drift_detection)

### Descripción general

TensorFlow Data Validation identifica anomalías en los datos de entrenamiento y servicio, y puede crear automáticamente un esquema a partir del análisis de los datos. El componente se puede configurar para detectar diferentes clases de anomalías en los datos. Permite ejecutar las siguientes acciones:

1. Llevar a cabo comprobaciones de validez mediante la comparación de estadísticas de datos con un esquema que codifica las expectativas del usuario.
2. Detectar el sesgo entrenamiento-servicio mediante la comparación de los datos de entrenamiento y servicio.
3. Detectar la desviación de datos mediante la observación de una serie de datos.

Documentamos cada una de estas funcionalidades de forma independiente:

- [Validación de ejemplo basada en esquemas](#schema_based_example_validation)
- [Detección de sesgos entre entrenamiento y servicio](#skewdetect)
- [Detección de desviaciones](#drift_detection)

### Validación de ejemplo basada en esquemas

TensorFlow Data Validation identifica cualquier anomalía en los datos de entrada al comparar las estadísticas de los datos con un esquema. El esquema codifica propiedades que se espera que satisfagan los datos de entrada, como tipos de datos o valores categóricos, y el usuario puede modificarlos o reemplazarlos.

Tensorflow Data Validation generalmente se invoca varias veces dentro del contexto de la canalización de TFX: (i) para cada división obtenida de ExampleGen, (ii) para todos los datos usados por Transform antes de la transformación y (iii) para todos los datos generados por Transform después de la transformación. Cuando se invoca en el contexto de Transform (ii-iii), las opciones estadísticas y las restricciones basadas en esquemas se pueden establecer mediante la definición de [`stats_options_updater_fn`](tft.md). Esto es particularmente útil al validar datos no estructurados (por ejemplo, características de texto). Consulte el [código de usuario](https://github.com/tensorflow/tfx/blob/master/tfx/examples/bert/mrpc/bert_mrpc_utils.py) para ver un ejemplo.

#### Características avanzadas de los esquemas

Esta sección cubre una configuración de esquema más avanzada que nos permite establecer configuraciones especiales.

##### Características dispersas

La codificación de características dispersas en los ejemplos generalmente introduce múltiples características que se espera que tengan la misma valencia para todos los ejemplos. Por ejemplo, la característica dispersa:

<pre><code>
WeightedCategories = [('CategoryA', 0.3), ('CategoryX', 0.7)]
</code></pre>

se codificaría con características separadas para índice y valor:

<pre><code>
WeightedCategoriesIndex = ['CategoryA', 'CategoryX']
WeightedCategoriesValue = [0.3, 0.7]
</code></pre>

con la restricción de que la valencia del índice y la característica de valor deben coincidir para todos los ejemplos. Esta restricción se puede hacer explícita en el esquema si se define una característica dispersa:

<pre><code class="lang-proto">
sparse_feature {
  name: 'WeightedCategories'
  index_feature { name: 'WeightedCategoriesIndex' }
  value_feature { name: 'WeightedCategoriesValue' }
}
</code></pre>

La definición de característica dispersa requiere uno o más índices y una característica de valor que se refieren a características que existen en el esquema. La definición explícita de características dispersas permite a TFDV verificar que las valencias de todas las características referidas coincidan.

Algunos casos de uso introducen restricciones de valencia similares entre características, pero no necesariamente codifican una característica dispersa. El uso de la característica dispersa debería desbloquearlo, pero no es lo ideal.

##### Entornos de esquema

De forma predeterminada, las validaciones suponen que todos los ejemplos de una canalización adhieren a un único esquema. En algunos casos, es necesario introducir ligeras variaciones en el esquema; por ejemplo, las características que se usan como etiquetas son necesarias durante el entrenamiento (y deben validarse), pero faltan durante el servicio. Los entornos se pueden utilizar para expresar dichos requisitos, en particular `default_environment()`, `in_environment()`, `not_in_environment()`.

Por ejemplo, supongamos que se requiere una característica llamada "LABEL" para la capacitación, pero se espera que no esté presente en el servicio. Esto se puede expresar así:

- Defina dos entornos distintos en el esquema: ["SERVING", "TRAINING"] y asocie 'LABEL' únicamente con el entorno "TRAINING".
- Asocie los datos de entrenamiento con el entorno "TRAINING" y los datos de servicio con el entorno "SERVING".

##### Generación de esquemas

El esquema de datos de entrada se especifica como una instancia del [esquema](https://github.com/tensorflow/metadata/blob/master/tensorflow_metadata/proto/v0/schema.proto) de TensorFlow.

En lugar de construir manualmente un esquema desde cero, un desarrollador puede confiar en la construcción automática de esquemas de TensorFlow Data Validation. Específicamente, TensorFlow Data Validation construye automáticamente un esquema inicial basado en estadísticas calculadas sobre los datos de entrenamiento disponibles en la canalización. Los usuarios pueden limitarse a revisar este esquema de generación automática, modificarlo según sea necesario, registrarlo en un sistema de control de versiones e insertarlo explícitamente en el proceso para su posterior validación.

TFDV incluye `infer_schema()` para generar un esquema automáticamente. Por ejemplo:

```python
schema = tfdv.infer_schema(statistics=train_stats)
tfdv.display_schema(schema=schema)
```

Esto desencadena una generación automática de esquemas basada en las siguientes reglas:

- Si un esquema ya se ha generado automáticamente, se utiliza tal cual.

- De lo contrario, TensorFlow Data Validation examina las estadísticas de datos disponibles y calcula un esquema adecuado para los datos.

*Nota: El esquema generado automáticamente es el de mejor esfuerzo y solo intenta inferir propiedades básicas de los datos. Se espera que los usuarios lo revisen y modifiquen según sea necesario.*

### Detección de sesgos entre entrenamiento y servicio<a name="skewdetect"></a>

#### Descripción general

TensorFlow Data Validation puede detectar un sesgo de distribución entre los datos de entrenamiento y de servicio. El sesgo de distribución ocurre cuando la distribución de los valores de las características para los datos de entrenamiento es significativamente diferente de los datos de servicio. Una de las causas clave del sesgo en la distribución es el uso de un corpus completamente diferente para entrenar la generación de datos con el fin de superar la falta de datos iniciales en el corpus deseado. Otra razón es un mecanismo de muestreo defectuoso que solo elige una submuestra de los datos de servicio para entrenar.

##### Escenario de ejemplo

Nota: Por ejemplo, para compensar un segmento de datos escasamente representado, si se utiliza un muestreo sesgado sin aumentar adecuadamente los ejemplos sometidos a un muestreo reducido, la distribución de los valores de las características entre los datos de entrenamiento y de servicio se sesga artificialmente.

Consulte la [Guía de introducción a TensorFlow Data Validation](https://www.tensorflow.org/tfx/data_validation/get_started#checking_data_skew_and_drift) para obtener información sobre cómo configurar la detección de sesgos entrenamiento-servicio.

### Detección de desviaciones

La detección de desviaciones se admite entre intervalos de datos consecutivos (es decir, entre el intervalo N y el intervalo N+1), como por ejemplo entre diferentes días de datos de entrenamiento. Expresamos la desviación en términos de [distancia L-infinito](https://en.wikipedia.org/wiki/Chebyshev_distance) para características categóricas y [divergencia aproximada de Jensen-Shannon](https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence) para características numéricas. Puede establecer la distancia umbral para recibir advertencias cuando la desviación sea mayor de lo aceptable. Establecer la distancia correcta suele ser un proceso iterativo que requiere experimentación y conocimiento del dominio.

Consulte la [Guía de introducción a TensorFlow Data Validation](https://www.tensorflow.org/tfx/data_validation/get_started#checking_data_skew_and_drift) para obtener información sobre cómo configurar la detección de desviación.

## Cómo usar visualizaciones para verificar sus datos

TensorFlow Data Validation proporciona herramientas para visualizar la distribución de valores de características. Al examinar estas distribuciones en un bloc de notas Jupyter usando [facetas](https://pair-code.github.io/facets/), puede detectar problemas comunes con los datos.

![Estadísticas de características](images/feature_stats.png)

### Identificación de distribuciones sospechosas

Puede identificar errores comunes en sus datos si usa una pantalla de descripción general de facetas para buscar distribuciones sospechosas de valores de características.

#### Datos desequilibrados

Una característica desequilibrada es una característica en la que predomina un valor. Las características desequilibradas pueden ocurrir naturalmente, pero si una característica siempre tiene el mismo valor, es posible que tenga un error de datos. Para detectar características desequilibradas en una descripción general de facetas, elija "Sin uniformidad" en el menú desplegable "Ordenar por".

Las características más desequilibradas aparecerán en la parte superior de cada lista de tipos de características. Por ejemplo, la siguiente captura de pantalla muestra una característica que es todo ceros y una segunda que está muy desequilibrada, en la parte superior de la lista "Características numéricas":

![Visualización de datos desequilibrados](images/unbalanced.png)

#### Datos distribuidos uniformemente

Una característica distribuida uniformemente es aquella para la cual todos los valores posibles aparecen casi con la misma frecuencia. Al igual que con los datos desequilibrados, esta distribución puede ocurrir de forma natural, pero también puede deberse a errores en los datos.

Para detectar características distribuidas uniformemente en una descripción general de facetas, elija "Sin uniformidad" en el menú desplegable "Ordenar por" y marque la casilla de verificación "Invertir orden":

![Histograma de datos uniformes](images/uniform.png)

Los datos de cadena se representan mediante gráficos de barras si hay 20 valores únicos o menos, y como un gráfico de distribución acumulativa si hay más de 20 valores únicos. Entonces, para datos de cadenas, las distribuciones uniformes pueden aparecer como gráficos de barras planas como el de arriba o como líneas rectas como el de abajo:

![Gráfico lineal: distribución acumulativa de datos uniformes](images/uniform_cumulative.png)

##### Errores que pueden generar datos distribuidos uniformemente

A continuación, se indican algunos errores comunes que pueden generar datos distribuidos uniformemente:

- Uso de cadenas para representar tipos de datos que no son cadenas, como las fechas. Por ejemplo, tendrá muchos valores únicos para una característica de fecha y hora con representaciones como "2017-03-01-11-45-03". Los valores únicos se distribuirán uniformemente.

- Incorporación de índices como "número de fila" a modo de características. Aquí nuevamente aparecen muchos valores únicos.

#### Datos faltantes

Para comprobar si a una característica le faltan valores por completo:

1. Elija "Cantidad faltante/cero" en el menú desplegable "Ordenar por".
2. Marque la casilla de verificación "Invertir orden".
3. Mire la columna "faltante" para ver el porcentaje de instancias con valores faltantes para una característica.

Un error de datos también puede causar valores de características incompletos. Por ejemplo, puede esperar que la lista de valores de una característica siempre tenga tres elementos y descubrir que a veces solo tiene uno. Para comprobar si hay valores incompletos u otros casos en los que las listas de valores de características no tienen la cantidad esperada de elementos:

1. Elija "Longitud de la lista de valores" en el menú desplegable "Gráfico para mostrar" a la derecha.

2. Mire el gráfico a la derecha de cada fila de características. El gráfico muestra el rango de longitudes de la lista de valores para la característica. Por ejemplo, la fila resaltada en la siguiente captura de pantalla muestra una característica que tiene algunas listas de valores de longitud cero:

![Visualización de descripción general de facetas con funciones con listas de valores de funciones de longitud cero](images/zero_length.png)

#### Grandes diferencias de escala entre características

Si sus características varían mucho en escala, entonces el modelo puede tener dificultades para aprender. Por ejemplo, si algunas características varían de 0 a 1 y otras varían de 0 a 1 000 000 000, tiene una gran diferencia de escala. Compare las columnas "máximo" y "mínimo" de las características para encontrar escalas que varían ampliamente.

Considere la posibilidad de normalizar los valores de las características para reducir estas amplias variaciones.

#### Etiquetas con etiquetas no válidas

Los estimadores de TensorFlow tienen restricciones sobre el tipo de datos que aceptan como etiquetas. Por ejemplo, los clasificadores binarios normalmente solo funcionan con etiquetas {0, 1}.

Revise los valores de las etiquetas en la descripción general de facetas y asegúrese de que cumplan con los [requisitos de Estimator](https://github.com/tensorflow/docs/blob/master/site/en/r1/guide/feature_columns.md).
