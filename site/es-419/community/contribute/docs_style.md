# Guía de estilo de la documentación de TensorFlow

## Prácticas recomendadas

- Céntrese en la intención del usuario y en el público.
- Use palabras cotidianas y frases cortas.
- Mantenga la coherencia en la construcción de las frases, la redacción y las mayúsculas.
- Use títulos y listas para facilitar la lectura de los documentos.
- La [Guía de estilo de documentos para desarrolladores de Google](https://developers.google.com/style/highlights) es útil.

## Markdown

Con algunas excepciones, TensorFlow emplea una sintaxis Markdown similar a la de [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/) (GFM). En esta sección se explican las diferencias entre la sintaxis de GFM Markdown y el Markdown que se usa para la documentación de TensorFlow.

### Cómo escribir sobre código

#### Menciones en línea de código

Coloque <code>`backticks`</code> alrededor de los siguientes símbolos cuando los use en texto:

- Nombres de argumentos: <code>`input`</code>, <code>`x`</code>, <code>`tensor`</code>
- Nombres de tensor devueltos: <code>`output`</code>, <code>`idx`</code>, <code>`out`</code>
- Tipos de datos: <code>`int32`</code>, <code>`float`</code>, <code>`uint8`</code>
- Otros nombres de operadores mencionados en el texto: <code>`list_diff()`</code>, <code>`shuffle()`</code>
- Nombres de clase: <code>`tf.Tensor`</code>, <code>`Strategy`</code>
- Nombre de archivo: <code>`image_ops.py`</code>, <code>`/path_to_dir/file_name`</code>
- Expresiones matemáticas o condiciones: <code>`-1-input.dims() &lt;= dim &lt;=     input.dims()`</code>

#### Bloques de código

Use tres comillas invertidas para abrir y cerrar un bloque de código. Si lo desea, especifique el lenguaje de programación después del primer grupo de comillas, por ejemplo:

<pre><code>
```python
# some python code here
```
</code></pre>

### Vínculos en Markdown y blocs de notas

#### Vínculos entre archivos en un repositorio

Utilice vínculos relativos entre archivos en un mismo repositorio de GitHub. Incluya la extensión del archivo.

Por ejemplo, **este archivo que está leyendo** pertenece al repositorio [https://github.com/tensorflow/docs](https://github.com/tensorflow/docs). Por lo tanto, puede usar rutas relativas para vincular a otros archivos en el mismo repositorio, como se muestra a continuación:

- <code>\[Basics\]\(../../guide/basics.ipynb\)</code> produce [Basics](../../guide/basics.ipynb).

Este es el enfoque preferido porque de esta forma funcionan todos los vínculos en [tensorflow.org](https://www.tensorflow.org), [GitHub](https://github.com/tensorflow/docs){:.external} y [Colab](https://github.com/tensorflow/docs/tree/master/site/en/guide/bazics.ipynb){:.external}. Además, el lector permanece en el mismo sitio cuando hace clic en un vínculo.

Nota: Debe incluir la extensión del archivo, como `.ipynb` o `.md`, para los vínculos relativos. Se representará en `tensorflow.org` sin una extensión.

#### Vínculos externos

Para los vínculos a archivos que no se encuentren en el repositorio actual, utilice vínculos Markdown estándar con el URI completo. Es preferible vincular al URI de [tensorflow.org](https://www.tensorflow.org) si está disponible.

Para vincular al código fuente, use un vínculo que comience con <var>https://www.github.com/tensorflow/tensorflow/blob/master/</var>, seguido del nombre del archivo que comienza en la raíz de GitHub.

Al crear un vínculo fuera de [tensorflow.org](https://www.tensorflow.org), incluya `{:.external}` en el vínculo de Markdown para que se muestre el símbolo de "vínculo externo".

- `[GitHub](https://github.com/tensorflow/docs){:.external}` produce [GitHub](https://github.com/tensorflow/docs){:.external}

No incluya parámetros de consulta de URI en el vínculo:

- Use: `https://www.tensorflow.org/guide/data`
- No: `https://www.tensorflow.org/guide/data?hl=en`

#### Imágenes

Los consejos de la sección anterior se refieren a los vínculos a páginas. Las imágenes se manejan de otra manera.

En general, no debería registrar imágenes; en lugar de eso, debería agregar el [equipo de TensorFlow-Docs](https://github.com/tensorflow/docs) a su PR y pedirles que alojen las imágenes en [tensorflow.org](https://www.tensorflow.org). Esto ayuda a evitar que el tamaño de su repositorio sea demasiado grande.

Si envía imágenes a su repositorio, tenga en cuenta que algunos sistemas no manejan rutas relativas a las imágenes. Es preferible utilizar una URL completa que remita a la ubicación final de la imagen en [tensorflow.org](https://www.tensorflow.org).

#### Vínculos a la documentación sobre las API

Los vínculos a la API se convierten cuando se publica el sitio. Para establecer un vínculo con la página de referencia de la API de un símbolo, encierre la ruta del símbolo entre comillas invertidas:

- <code>`tf.data.Dataset`</code> produce [`tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset)

Se da preferencia a las rutas completas, excepto cuando son largas. Las rutas se pueden abreviar si se eliminan sus componentes iniciales. Las rutas parciales se convertirán en vínculos si se cumplen estas condiciones:

- Hay al menos un `.` en la ruta y
- la ruta parcial es única dentro del proyecto.

Las rutas de la API están vinculadas **para cada proyecto** con una API de Python publicada en [tensorflow.org](https://www.tensorflow.org). Puede vincular fácilmente varios subproyectos desde un mismo archivo si envuelve los nombres de las API con comillas invertidas. Por ejemplo:

- <code>`tf.metrics`</code>, <code>`tf_agents.metrics`</code>, <code>`text.metrics`</code> produce: `tf.metrics`, `tf_agents.metrics`, `text.metrics`.

Para símbolos con múltiples alias de ruta, existe una ligera preferencia por la ruta que coincida con la página de la API en [tensorflow.org](https://www.tensorflow.org). Todos los alias redireccionarán a la página correcta.

### Matemáticas en Markdown

Puede usar MathJax dentro de TensorFlow al editar archivos Markdown, pero tenga en cuenta lo siguiente:

- MathJax se procesa correctamente en [tensorflow.org](https://www.tensorflow.org).
- MathJax no se procesa correctamente en GitHub.
- Esta anotación puede resultar desconcertante para desarrolladores que no estén familiarizados.
- Para mantener la coherencia, [tensorflow.org](https://www.tensorflow.org) sigue las mismas reglas que Jupyter/Colab.

Utilice <code>$$</code> alrededor de un bloque de MathJax:

<pre><code>$$
E=\frac{1}{2n}\sum_x\lVert (y(x)-y'(x)) \rVert^2
$$</code></pre>

$$ E=\frac{1}{2n}\sum_x\lVert (y(x)-y'(x)) \rVert^2 $$

Envuelva expresiones MathJax en línea con <code>$ ... $</code>:

<pre><code>
This is an example of an inline MathJax expression: $ 2 \times 2 = 4 $
</code></pre>

Este es un ejemplo de una expresión MathJax en línea: $ 2 \times 2 = 4 $

Los delimitadores <code>\( ... \)</code> también funcionan para matemáticas en línea, pero la forma $ a veces es más legible.

Nota: Si necesita utilizar un signo de dólar en texto o expresiones de MathJax, escríbalo con una barra diagonal inicial: `\$` . No es necesario utilizar caracteres de escape en los signos de dólar dentro de los bloques de código (como los nombres de las variables Bash).

## Estilo de prosa

Si se dispone a escribir o editar partes sustanciales de la documentación narrativa, lea la [Guía de estilo de la documentación para desarrolladores de Google](https://developers.google.com/style/highlights).

### Principios del buen estilo

- *Revise la ortografía y gramática de sus contribuciones.* La mayoría de los editores incluyen un corrector ortográfico o tienen un complemento de corrección ortográfica disponible. También puede pegar el texto en un documento de Google Documentos u otro software de documentos para una revisión ortográfica y gramatical más completa.
- *Use una voz informal y amigable.* Escriba documentación de TensorFlow como si fuera una conversación, como si estuviera hablando personalmente con otra persona. Use un tono de comprensivo en el artículo.

Nota: Ser menos formal no significa ser menos técnico. Simplifique su prosa, no el contenido técnico.

- *Evite las exenciones de responsabilidad, las opiniones y los juicios de valor.* Palabras como "fácilmente", " simplemente" y "sencillo" están llenas de suposiciones. Algo puede parecerle fácil a usted, pero difícil a otra persona. Evite usarlas en la medida de lo posible.
- *Use frases sencillas y directas, sin jerga complicada.* Las frases compuestas, las cadenas de cláusulas y los modismos propios de un lugar pueden complicar la comprensión y traducción de un texto. Si una frase puede dividirse en dos, probablemente debería hacerlo. Evite el punto y coma. Use listas de viñetas cuando corresponda.
- *Proporcione contexto.* No utilice abreviaturas sin explicarlas. No mencione proyectos que no sean de TensorFlow sin vincularlos. Explique por qué el código está escrito de esa forma.

## Guía de uso

### Operaciones

En archivos markdown, use `# ⇒` en lugar de un solo signo igual cuando quiera mostrar lo que devuelve una operación.

```python
# 'input' is a tensor of shape [2, 3, 5]
tf.expand_dims(input, 0)  # ⇒ [1, 2, 3, 5]
```

En los blocs de notas, muestre el resultado en lugar de agregar un comentario (si la última expresión en una celda del bloc de notas no está asignada a una variable, se muestra automáticamente).

En los documentos de referencia de la API, opte por el uso de [doctest](docs_ref.md#doctest) para mostrar los resultados.

### Tensores

Cuando se habla de un tensor en términos generales, no se escribe la palabra *tensor* en mayúsculas. Cuando se habla del objeto específico que se proporciona a una operación, se debe escribir la palabra *Tensor* en mayúsculas y agregar comillas invertidas alrededor porque se está hablando de un objeto `Tensor`.

No use la palabra *Tensores* (plural) para describir múltiples objetos `Tensor` a menos que realmente esté hablando de un objeto `Tensors`. En lugar de eso, diga "una lista (o colección) de objetos `Tensor`".

Use la palabra *forma* para detallar los ejes de un tensor y muestre la forma entre corchetes con comillas invertidas. Por ejemplo:

<pre><code>
If `input` is a three-axis `Tensor` with shape `[3, 4, 3]`, this operation
returns a three-axis `Tensor` with shape `[6, 8, 6]`.
</code></pre>

Como en el caso anterior, es preferible utilizar "eje" o "índice" en lugar de "dimensión" cuando se habla de los elementos de la forma de un `Tensor`. De lo contrario, se podría confundir "dimensión" con la dimensión de un espacio vectorial. Un "vector tridimensional" tiene un único eje de longitud 3.
