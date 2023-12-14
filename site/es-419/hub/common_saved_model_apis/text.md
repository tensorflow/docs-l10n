# API SavedModel comunes para tareas de texto

En esta página se describe cómo se debería implementar la [API SavedModel reutilizable](../reusable_saved_models.md) con [TF2 SavedModels](../tf2_saved_model.md) para tareas relacionadas con textos. (Reemplaza y extiende las [Firmas comunes para texto](../common_signatures/text.md) del [formato TF1 Hub ](../tf1_hub_module), hoy en desuso).

## Descripción general

Hay muchas API para calcular las **(embeddings) incorporaciones de textos** (también conocidas como representaciones densas de texto o vectores de características del texto).

- La API para *la incorporación de textos a partir de entradas (de textos)* se implementa mediante un SavedModel que vincula un lote de secuencias a un lote de vectores incorporados. Es muy fácil de usar y muchos modelos de TF Hub ya lo tienen implementado. Sin embargo, no permite el ajuste fino del modelo en TPU.

- La API para *la incorporación de textos con entradas preprocesadas* resuelve la misma tarea, pero se implementa con dos SavedModels separados:

    - un *preprocesador* que se puede ejecutar dentro de una canalización de entrada de tf.data y que convierte secuencias y otros datos con longitudes variables en tensores numéricos,
    - un *codificador* que acepta resultados de un preprocesador y realiza la parte entrenable del cálculo de incorporación.

    Esta división permite que las entradas se procesen asincrónicamente antes de ingresar en el ciclo de entrenamiento. En particular, favorece la construcción de codificadores que se puedan ejecutar y ajustar en [TPU](https://www.tensorflow.org/guide/tpu).

- La API para *incorporaciones de texto con codificadores Transformer* extiende la API para dichas incorporaciones a partir de las entradas preprocesadas al caso particular BERT y otros codificadores Transformer.

    - El *preprocesador* se extiende para construir entradas del codificador a partir de más de un segmento de texto de entrada.
    - El *codificador Transformer* expone las incorporaciones relacionadas con el contexto de tokens individuales.

En cada caso, las entradas de texto son secuencias codificadas UTF-8, normalmente de texto sin formato, a menos que la documentación del modelo indique lo contrario.

Independientemente de la API, los diferentes modelos se han preentrenado con textos de diferentes tareas que uno tiene en mente. Por lo tanto, no todos los modelos de incorporación de texto son adecuados para cualquier problema.

<a name="feature-vector"></a>
<a name="text-embeddings-from-text"></a>

## Incorporación de textos a partir de entradas de texto

Un SavedModel para **incorporación de textos a partir de entradas (de textos)** acepta un lote de entradas en un tensor de secuencia de forma `[batch_size]` y las mapea a un tensor float32 de forma `[batch_size, dim]` con representaciones densas (vectores de características) de las entradas.

### Sinopsis de uso

```python
obj = hub.load("path/to/model")
text_input = ["A long sentence.",
              "single-word",
              "http://example.com"]
embeddings = obj(text_input)
```

Recordemos que en la [API de SavedModel reutilizable](../reusable_saved_models.md) si el modelo se ejecuta en el modo de entrenamiento (p. ej., para abandonar) puede requerir un argumento con palabra clave `obj(..., training=True)` y que `obj` proporcione atributos `.variables`, `.trainable_variables` y `.regularization_losses` según corresponda.

En Keras, lo que lleva a cabo todo esto es lo siguiente:

```python
embeddings = hub.KerasLayer("path/to/model", trainable=...)(text_input)
```

### Entrenamiento distribuido

Si la incorporación de texto se utiliza como parte de un modelo que se entrena con una estrategia distribuida, las llamadas a `hub.load("path/to/model")` o `hub.KerasLayer("path/to/model", ...)`, respectivamente, deben producirse dentro del alcance de DistributionStrategy para crear las variables del modelo de un modo distribuido. Por ejemplo:

```python
  with strategy.scope():
    ...
    model = hub.load("path/to/model")
    ...
```

### Ejemplos

- Tutorial de Colab sobre [clasificación de textos con revisiones de películas](https://colab.research.google.com/github/tensorflow/docs/blob/master/g3doc/en/hub/tutorials/tf2_text_classification.ipynb).

<a name="text-embeddings-preprocessed"></a>

## Incorporaciones de textos con entradas preprocesadas

Una **incorporación de textos con entradas preprocesadas** se implementa mediante dos SavedModels separados:

- un **preprocesador** que mapea un tensor de secuencia de forma `[batch_size]` con un diccionario de tensores numéricos,
- un **codificador** que acepta un diccionario de tensores como devueltos por el preprocesador, realiza la parte entrenable del cálculo de incorporación y devuelve un diccionario de salidas. La salida con la clave `"default"` es un tensor float32 de forma `[batch_size, dim]`.

Gracias a ello es posible ejecutar el preprocesador en una canalización de entradas y realizar el ajuste fino de las incorporaciones calculadas por el codificador como parte de un modelo más grande. En particular, permite construir codificadores que se pueden ejecutar o a los que se les puede hacer el ajuste fino en [TPU](https://www.tensorflow.org/guide/tpu).

Es un detalle de la implementación saber qué tensores se encuentran dentro de la salida del preprocesador y cuáles (si es que hay alguno) de los tensores adicionales, además de `"default"`, están dentro de la salida del codificador.

La documentación del codificador debe especificar qué procesador se usará. Por lo común, hay una sola opción correcta exacta.

### Sinopsis de uso

```python
text_input = tf.constant(["A long sentence.",
                          "single-word",
                          "http://example.com"])
preprocessor = hub.load("path/to/preprocessor")  # Must match `encoder`.
encoder_inputs = preprocessor(text_input)

encoder = hub.load("path/to/encoder")
enocder_outputs = encoder(encoder_inputs)
embeddings = enocder_outputs["default"]
```

Recordemos que en la [API de SavedModel reutilizable](../reusable_saved_models.md) si el codificador se ejecuta en el modo de entrenamiento (p. ej., para abandonar) puede requerir un argumento con palabra clave `encoder(..., training=True)` y que `encoder` proporcione atributos `.variables`, `.trainable_variables` y `.regularization_losses` según corresponda.

El modelo `preprocessor` puede tener `.variables` pero no está previsto para recibir más entrenamiento. El preprocesamiento no es dependiente del modo: si `preprocessor()` tiene un argumento `training=...`, no tiene efecto.

En Keras, lo que lleva a cabo todo esto es lo siguiente:

```python
encoder_inputs = hub.KerasLayer("path/to/preprocessor")(text_input)
encoder_outputs = hub.KerasLayer("path/to/encoder", trainable=True)(encoder_inputs)
embeddings = encoder_outputs["default"]
```

### Entrenamiento distribuido

Si el codificador se usa como parte de un modelo que se entrena con una estrategia de distribución, las llamadas a `hub.load("path/to/encoder")` o `hub.KerasLayer("path/to/encoder", ...)`, respectivamente, deben producirse dentro,

```python
  with strategy.scope():
    ...
```

para recrear las variables del codificador de un modo distribuido.

Del mismo modo, si el preprocesador es parte del modelo entrenado (como se da en el ejemplo simple anterior), también se debe cargar dentro del alcance de la estrategia de distribución. Si, a pesar de todo, el preprocesador se usa en una canalización de entrada (p. ej., en una canalización invocable pasada a `tf.data.Dataset.map()`), la carga debe producirse *fuera* del alcance de la estrategia de distribución, para colocar sus variables (en caso de que haya alguna) en la CPU <em>host</em>.

### Ejemplos

- Tutorial de Colab sobre [clasificación de textos con BERT](https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/text/classify_text_with_bert.ipynb).

<a name="transformer-encoders"></a>

## Incorporación de textos con codificadores Transformer

Los codificadores Transformer para texto operan en lotes de secuencias de entradas, cada secuencia compuesta por segmentos *n* ≥ 1 de texto *tokenizado*, dentro de algún enlace específico del modelo en *n*. En el caso de BERT y de muchas otras de sus extensiones, ese enlace es 2, por lo tanto, se aceptan segmentos simples y en pares.

La API para <em>incorporaciones de texto con codificadores Transformer</em> extiende la API para dichas incorporaciones de texto a este entorno con entradas preprocesadas.

### Preprocesador

Un preprocesador SavedModel para las incorporaciones con codificadores transformer implementa la API de un preprocesador SavedModel para incorporaciones de texto con entradas preprocesadas (ver arriba), que brinda una forma de mapear entradas de texto de un solo segmento directamente con las entradas del codificador.

Además, el preprocesador SavedModel ofrece subobjetos invocables `tokenize` para la <em>tokenización</em> (separado por segmentos) y `bert_pack_inputs` para empacar los segmentos <em>tokenizados</em> *n* en una secuencia de entrada para el codificador. Cada subobjeto sigue la [API de SavedModel reutilizable](../reusable_saved_models.md).

#### Sinopsis de uso

Como ejemplo concreto de dos segmentos de texto, observemos una tarea de inferencia (<em>entailment</em>) que pregunta si una premisa (el primer segmento) implica o no una hipótesis (segundo segmento).

```python
preprocessor = hub.load("path/to/preprocessor")

# Tokenize batches of both text inputs.
text_premises = tf.constant(["The quick brown fox jumped over the lazy dog.",
                             "Good day."])
tokenized_premises = preprocessor.tokenize(text_premises)
text_hypotheses = tf.constant(["The dog was lazy.",  # Implied.
                               "Axe handle!"])       # Not implied.
tokenized_hypotheses = preprocessor.tokenize(text_hypotheses)

# Pack input sequences for the Transformer encoder.
seq_length = 128
encoder_inputs = preprocessor.bert_pack_inputs(
    [tokenized_premises, tokenized_hypotheses],
    seq_length=seq_length)  # Optional argument.
```

En Keras, este cálculo se puede expresar de la siguiente manera

```python
tokenize = hub.KerasLayer(preprocessor.tokenize)
tokenized_hypotheses = tokenize(text_hypotheses)
tokenized_premises = tokenize(text_premises)

bert_pack_inputs = hub.KerasLayer(
    preprocessor.bert_pack_inputs,
    arguments=dict(seq_length=seq_length))  # Optional argument.
encoder_inputs = bert_pack_inputs([tokenized_premises, tokenized_hypotheses])
```

#### Detalles sobre `tokenize`

Una llamada a `preprocessor.tokenize()` acepta a un tensor secuencial de forma `[batch_size]` y devuelve un [RaggedTensor](https://www.tensorflow.org/guide/ragged_tensor) de forma `[batch_size, ...]` cuyos valores son ID de token int32 que representan las secuencias de entrada (<em>strings</em>). Puede haber *r* ≥ 1 dimensiones irregulares de `batch_size`, pero no otras dimensiones uniformes.

- Si *r*=1, la forma es `[batch_size, (tokens)]` y cada entrada, simplemente, se *tokeniza* en una secuencia plana de tokens.
- Si *r*&gt;1, hay *r*-1 niveles adicionales de agrupamiento. Por ejemplo, [tensorflow_text.BertTokenizer](https://github.com/tensorflow/text/blob/v2.3.0/tensorflow_text/python/ops/bert_tokenizer.py#L138) usa *r*=2 para agrupar tokens por palabras y produce la forma `[batch_size, (words), (tokens_per_word)]`. Dependerá del modelo que se tenga a la mano, la cantidad de niveles extra de este tipo que haya (en caso de que haya alguno) y a qué agrupaciones representen.

El usuario puede (aunque no es necesario que lo haga) modificar las entradas <em>tokenizadas</em>; p. ej., para ajustar el límite de seq_length que se asignará a entradas del codificador para empaquetamiento. En este caso, las dimensiones extra de salida del <em>tokenizador</em> pueden resultar útiles (p. ej., con respecto a los límites para las palabras) pero volverse insignificantes para el paso siguiente.

En cuanto a la [API SavedModel reutilizable](../reusable_saved_models.md), el objeto <code>preprocessor.tokenize</code> puede tener `.variables` pero no está previsto para recibir más entrenamiento. La tokenización no es dependiente del modo: si `preprocessor.tokenize()` tiene un argumento `training=...`, no tiene efecto.

#### Detalles de `bert_pack_inputs`

Una llamada a `preprocessor.bert_pack_inputs()` acepta una lista de Python de entradas tokenizadas (en lotes separados para cada segmento de entrada) y devuelve un diccionario de tensores que representa un lote de secuencias de entrada con longitudes fijas para el modelo codificador Transformer.

Cada entrada *tokenizada* es un RaggedTensor int32 de forma `[batch_size, ...]`, donde la cantidad *r* de dimensiones irregulares de batch_size es 1 o la misma de la salida de `preprocessor.tokenize().` (Esta última, solamente por comodidad; las dimensiones extra se aplanan antes de empaquetarse.)

El empaque agrega tokens especiales en torno a los segmentos de entrada, tal como lo espera el codificador. Con la llamada `bert_pack_inputs()` se implementa exactamente el mismo esquema de empaquetamiento utilizado por los modelos BERT originales y muchas otras de sus extensiones: la secuencia empaquetada comienza con un token de inicio de secuencia, seguido por los segmentos <em>tokenizados</em>, cada uno terminado por un token de fin de segmento. Las posiciones restantes hasta seq_length (en caso de que haya alguna) se completan con tokens de amortiguación (<em>padding</em>).

Si una secuencia empaquetada excediera la seq_length, `bert_pack_inputs()` truncará sus segmentos hasta convertirlos en prefijos de tamaños aproximadamente iguales para que la secuencia quepa exactamente dentro de la seq_length.

El empaquetamiento no depende del modelo: si `preprocessor.bert_pack_inputs()` tiene un argumento `training=...`, no tiene efecto. Además, no se espera que `preprocessor.bert_pack_inputs` tenga variables ni que admita el ajuste fino.

### Codificador

El codificador se llama en el diccionario de `encoder_inputs` del mismo modo en que lo hace la API para incorporaciones de texto con entradas preprocesadas (ver arriba), incluidas las provisiones de la [API del SavedModel reutilizable](../reusable_saved_models.md).

#### Sinopsis de uso

```python
enocder = hub.load("path/to/encoder")
enocder_outputs = encoder(encoder_inputs)
```

o su equivalente en Keras:

```python
encoder = hub.KerasLayer("path/to/encoder", trainable=True)
encoder_outputs = encoder(encoder_inputs)
```

#### Detalles

Las `encoder_outputs` son un diccionario de tensores con las siguientes claves.

<!-- TODO(b/172561269): More guidance for models trained without poolers. -->

- `"sequence_output"`: un tensor float32 de forma `[batch_size, seq_length, dim]` con la incorporación (acorde al contexto) de cada token de cada una de las secuencias de entrada empaquetadas.
- `"pooled_output"`: un tensor float32 de forma `[batch_size, dim]` con la incorporación de cada secuencia de entrada como un todo, derivada de una sequence_output de alguna manera entrenable.
- `"default"`, tal como lo requiere la API para incorporaciones de texto con entradas preprocesadas: un tensor float32 de forma `[batch_size, dim]` con incorporaciones de cada secuencia de entrada. (Podría ser solamente un alias de pooled_output).

El contenido de `encoder_inputs` no es indispensable para esta definición de API. Sin embargo, para los codificadores que utilizan entradas de estilo BERT se recomienda usar los siguientes nombres (del [kit de herramientas para modelado con NLP de TensorFlow Model Garden](https://github.com/tensorflow/models/tree/master/official/nlp)) a fin de minimizar la fricción del intercambio de codificadores y para reutilizar los modelos del preprocesador:

- `"input_word_ids"`: un tensor int32 de forma `[batch_size, seq_length]` con los ID de token de la secuencia de entrada empaquetada (es decir, que incluye un token de inicio de secuencia, tokens de fin de segmento y la amortiguación).
- `"input_mask"`: un tensor int32 de forma `[batch_size, seq_length]` con valor 1 en la posición de todos los token de entrada presentes antes de amortiguar y un valor 0 para los token de amortiguación.
- `"input_type_ids"`: un tensor int32 de forma `[batch_size, seq_length]` con el índice del segmento de entrada que dio origen al token de entrada en la posición respectiva. El primer segmento de entrada (índice 0) incluye el token de inicio de secuencia y su token de fin de segmento. Los segmentos segundo y último (si hay) incluyen los respectivos tokens de fin de segmento. Los tokens de amortiguación vuelven a recibir el índice 0.

### Entrenamiento distribuido

Para cargar los objetos del preprocesador y del codificador dentro o fuera de un alcance estratégico de distribución se aplican las mismas reglas que en la API para incorporaciones con entradas preprocesadas (ver arriba).

### Ejemplos

- Tutorial de Colab sobre [cómo resolver tareas GLUE con BERT en TPU](https://colab.research.google.com/github/tensorflow/text/blob/master/docs/tutorials/bert_glue.ipynb).
