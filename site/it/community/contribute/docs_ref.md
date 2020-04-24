# Contribuire alla documentazione delle API di TensorFlow

## Docstring verificabili

TensorFlow usa [DocTest](https://docs.python.org/3/library/doctest.html) per
verificare frammenti di codice in docstring Python. I frammenti devono essere codice Python
eseguibile. Per abilitare il test, aggiungere il prefisso `>>>` alla linea (tre parentesi 
angolate sinistre). Per esempio, qui c'è un estratto dalla funzione `tf.concat` nel file sorgente
[array_ops.py](https://www.tensorflow.org/code/tensorflow/python/ops/array_ops.py):

```
def concat(values, axis, name="concat"):
  """Concatenates tensors along one dimension.
  ...

  >>> t1 = [[1, 2, 3], [4, 5, 6]]
  >>> t2 = [[7, 8, 9], [10, 11, 12]]
  >>> concat([t1, t2], 0)
  <tf.Tensor: shape=(4, 3), dtype=int32, numpy=
  array([[ 1,  2,  3],
         [ 4,  5,  6],
         [ 7,  8,  9],
         [10, 11, 12]], dtype=int32)>

  <... more description or code snippets ...>

  Args:
    values: A list of `tf.Tensor` objects or a single `tf.Tensor`.
    axis: 0-D `int32` `Tensor`.  Dimension along which to concatenate. Must be
      in the range `[-rank(values), rank(values))`. As in Python, indexing for
      axis is 0-based. Positive axis in the rage of `[0, rank(values))` refers
      to `axis`-th dimension. And negative axis refers to `axis +
      rank(values)`-th dimension.
    name: A name for the operation (optional).

    Returns:
      A `tf.Tensor` resulting from concatenation of the input tensors.
  """

  <code here>
```

Notare: TensorFlow DocTest usa TensorFlow 2 e Python 3.

### Rendere verificabile il codice con DocTest

Attualmente, molti docstring usano apici inversi (```) per identificare il codice. Per rendere
il codice verificabile con DocTest:

*   Rimuovere gli apici inversi (```) usare pe parentesi angolate sinistre (>>>) davanti a tutte
    le linee. Usare (...) all'inizio delle linee continuazione.
*   Aggiungere un a capo, per separate i segmenti DocTest dal testo Markdown per
    visualizzarli in modo opportuno su tensorflow.org.

### Considerazioni sui docstring

*   *Formato di uscita*: Il formato dell'output del frammento deve essere direttamente al di sotto 
    del codice che genera l'output. Inoltre, l'output nella docstring deve essere
    esattamente uguale a come dovrebbe essere l'output dopo l'esecuzione del codice. Vedere
    l'esempio sopra. Inoltre, prelevare
    [questa parte](https://docs.python.org/3/library/doctest.html#warnings) dalla
    documentazione DocTest. Se l'output supera il limite di 80 linee, potete mettere
    il resto dell'output su una nuova linea e DocTest lo riconoscerà. Per esempio,
    vedere i blocchi multi-linea qui sotto.
*   *Globals*: i moduli <code>&#96;tf&#96;</code>, `np` ed `os` sono sempre
    disponibili nel DocTest di TensorFlow.
*   *Use symbols*: In DocTest potete accedere direttamente ai simboli definiti nello
    stesso file. Per usare un simbolo che non è definito nel file attuale,
    usare l'API pubblica di TensorFlow `tf.xxx` invece di `xxx`. come potete vedere nell'
    esempio sotto, <code>&#96;random.normal&#96;</code> viene acceduto attraverso
    <code>&#96;tf.random.normal&#96;</code>. Questo perché
    <code>&#96;random.normal&#96;</code> non è visibile in `NewLayer`.

    ```
    def NewLayer():
      “””This layer does cool stuff.

      Example usage:

      >>> x = tf.random.normal((1, 28, 28, 3))
      >>> new_layer = NewLayer(x)
      >>> new_layer
      <tf.Tensor: shape=(1, 14, 14, 3), dtype=int32, numpy=...>
      “””
    ```

*   *Valori in virgola mobile*: Il doctest di TensorFlow estrae i valori decimali
    dalle stringhe dei risultati, e le confronta tramite `np.allclose` con tolleranze
    ragionevoli (`atol=1e-6`, `rtol=1e-6`). In questo modo gli autori non devono preoccuparsi
    che docstring eccessivamente precise, causino malfunzionamenti dovuti a problemi aritmetici.
    Incollate semplicemente il valore atteso.

*   *Output non-deterministici*: Usare i punti di sospensione(`...`) per le parti incerte e
    e DocTest ignorerà quelle sotto-stringhe.

    ```
    >>> x = tf.random.normal((1,))
    >>> print(x)
    <tf.Tensor: shape=(1,), dtype=float32, numpy=..., dtype=float32)>
    ```

*   *Blocchi multi-linea*: DocTest è rigido circa la differenza tra un enunciato su una linea singola
    ed un enunziato su più linee. Notate l'uso di (...) qui sotto:

    ```
    >>> if x > 0:
    ...   print("X is positive")
    >>> model.compile(
    ...   loss="mse",
    ...   optimizer="adam")
    ```

*   *Eccezioni*: I dettagli dell'eccezione vengono ignorati eccetto l'Eccezione che
    viene sollevata. Vedere
    [questo](https://docs.python.org/3/library/doctest.html#doctest.IGNORE_EXCEPTION_DETAIL)
    per maggiori dettagli.

    ```
    >>> np_var = np.array([1, 2])
    >>> tf.keras.backend.is_keras_tensor(np_var)
    Traceback (most recent call last):
    ...
    ValueError: Unexpectedly found an instance of type `<class 'numpy.ndarray'>`.
    ```

### Fare test sulla vostra macchina

Ci sono due modi per fare, in locale, il test del codice nelle docstring:

*   Se state solo cambiando la docstring di una classe/funzione/metodo, allora
    potete provarla passando il percorso del file a
    [tf_doctest.py](https://www.tensorflow.org/code/tensorflow/tools/docs/tf_doctest.py).
    Per esempio:

    <pre class="prettyprint lang-bsh">
    <code class="devsite-terminal">python tf_doctest.py --file=&lt;file_path&gt;</code>
    </pre>

    Questo girerà usando la versione installata di TensorFlow. Affinché siate sicuri
    di stare eseguendo lo stesso codice che state provando:

    *   Usare un [tf-nightly](https://pypi.org/project/tf-nightly/) aggiornato
        `pip install -U tf-nightly`
    *   Cambiare base alle vostre richieste di pull sui pull recenti del ramo
        principale di [TensorFlow](https://github.com/tensorflow/tensorflow).

*   Se state cambiando il codice e la docstring di una classe/funzione/metodo,
    allora avrete bisogno di
    [compilare TensorFlow dal sorgente](../../install/source.md). Una volta eseguita la 
    compilazione dal sorgente, potete eseguire il test:

    <pre class="prettyprint lang-bsh">
    <code class="devsite-terminal">bazel run //tensorflow/tools/docs:tf_doctest</code>
    </pre>

    o

    <pre class="prettyprint lang-bsh">
    <code class="devsite-terminal">bazel run //tensorflow/tools/docs:tf_doctest -- --module=ops.array_ops</code>
    </pre>

    Il `--module` è relativo a `tensorflow.python`.
