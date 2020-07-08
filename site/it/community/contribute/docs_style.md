# Guida di stile della documentazione TensorFlow

## Best practice

*   Focalizzarsi sull'obiettivo degli utilizzatori e l'audience.
*   Usare il linguaggio di tutti i giorni e periodi brevi.
*   Usare costruzione e formulazione del periodo consistenti tra loro e con l'impiego delle maiuscole.
*   Usare titoli e punti elenco in modo da rendere il vostro testo facile da scorrere.
*   La
    [Guida di Stile dello Sviluppatore Google](https://developers.google.com/style/highlights)
    può essere d'aiuto.

## Markdown

A meno di poche eccezioni,TensorFlow usa una sintassi di Markdown simile alla
[GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/)
(GFM). Questo capitolo spiega le differenze tra la sintassi del Markdown GFM ed il 
Markdown usato per la documentazione TensorFlow.


### Scrivere del codice

#### Menzione di codice inline

Racchiudere tra <code>&#96;apici inversi&#96;</code> i simboli seguenti nel testo:

*   Nomi di argomenti: <code>&#96;input&#96;</code>, <code>&#96;x&#96;</code>,
    <code>&#96;tensor&#96;</code>
*   Nomi di tensori restituiti: <code>&#96;output&#96;</code>,
    <code>&#96;idx&#96;</code>, <code>&#96;out&#96;</code>
*   Tipi di dato: <code>&#96;int32&#96;</code>, <code>&#96;float&#96;</code>,
    <code>&#96;uint8&#96;</code>
*   Richiami ad operazioni nel testo: <code>&#96;list_diff()&#96;</code>,
    <code>&#96;shuffle()&#96;</code>
*   Nomi di classi: <code>&#96;tf.Tensor&#96;</code>, <code>&#96;Strategy&#96;</code>
*   Nomi di file: <code>&#96;image_ops.py&#96;</code>,
    <code>&#96;/path_to_dir/file_name&#96;</code>
*   Espressioni matematiche o condizioni: <code>&#96;-1-input.dims() &lt;= dim &lt;=
    input.dims()&#96;</code>

#### Blocchi di codice

Usare tre apici inversi per aprire e chiudere un blocco di codice. Eventualmente, specificare il linguaggio 
di programmazione dopo il primo gruppo di apici, per esempio:
<pre><code>
&#96;&#96;&#96;python
&#35; some python code here
&#96;&#96;&#96;
</code></pre>

### Collegamenti in Markdown

#### Collegamenti tra file in questo repository

Usare collegamenti relativi tra i file in un repository. Così funzioneranno in
[tensorflow.org](https://www.tensorflow.org) e
[GitHub](https://github.com/tensorflow/docs/tree/master/site/en):<br/>
<code>\[Custom layers\]\(../tutorials/eager/custom_layers.ipynb\)</code> produce
[Custom layers](https://www.tensorflow.org/tutorials/eager/custom_layers) sul sito.

#### Collegamenti alla documentazione sulle API

Quando il sito viene pubblicato, i collegamenti alle API vengono convertiti. Per collegarsi ad un simbolo 
in una pagina di riferimento di un'API, racchiudere il percorso completo tra apici inversi:

*   <code>&#96;tf.data.Dataset&#96;</code> produce
    [`tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset)

Per API C++, usare il percorso del namespace:

*   `tensorflow::Tensor` produce
    [tensorflow::Tensor](https://www.tensorflow.org/api_docs/cc/class/tensorflow/tensor)

#### Collegamenti esterni

Per i collegamenti esterni, inclusi i file su <var>https://www.tensorflow.org</var>
che non si trovano nel repository `tensorflow/docs`, usare i collegamenti Markdown standard
con l'URI completo.

Per collegamenti al codice sorgente, usare un collegamenti che inizi con
<var>https://www.github.com/tensorflow/tensorflow/blob/master/</var>, seguito 
dal nome del file iniziando dalla radice di GitHub.

Questo schema di nomenclatura degli URI assicura che <var>https://www.tensorflow.org</var> possa
indirizzare il collegamento al ramo di codice corrispondente alla versione della
documentazione che state guardando.

Non includere i parametri delle query di un URI in un collegamento.

Nei percorsi dei file usare il trattino basso (underscore n.d.t.) al posto degli spazi, per esempio, `custom_layers.ipynb`.

Includere le estensioni dei file nei collegamenti da usare sul sito *e* in GitHub, per esempio,<br/>
<code>\[Custom layers\]\(../tutorials/eager/custom_layers.ipynb\)</code>.

### Matematica nei Markdown

In TensorFlow potete usare MathJax, quando dovete redigere file Markdown, ma fate attenzione a quanto segue:

*   MathJax viene visualizzato bene su [tensorflow.org](https://www.tensorflow.org).
*   MathJax non viene visualizzato bene su GitHub.
*   La notazione può scoraggiare sviluppatori che non hanno familiarità con essa.
*   Per consistenza [tensorflow.org](https://www.tensorflow.org) segue le stesse
    regole di Jupyter/Colab.

Racchiudere un blocco MathJax tra <code>&#36;&#36;</code>:

<pre><code>&#36;&#36;
E=\frac{1}{2n}\sum_x\lVert (y(x)-y'(x)) \rVert^2
&#36;&#36;</code></pre>

$$
E=\frac{1}{2n}\sum_x\lVert (y(x)-y'(x)) \rVert^2
$$

Racchiudere le espressioni MathJax inline con <code>&#36; ... &#36;</code>:

<pre><code>
Questo è un esempio di espressione MathJax inline: &#36; 2 \times 2 = 4 &#36;
</code></pre>

Questo è un esempio di espressione MathJax inline: $ 2 \times 2 = 4 $

I delimitatori <code>&#92;&#92;( ... &#92;&#92;)</code> funzionano anche per espressioni matematiche inline,
anche se la forma \$ talvolta è più leggibile.

Nota: Se avete bisogno di un segno dollaro in un testo o in un'espressione MathJax, fatelo precedere
da uno slash: `\$`. I segni dollaro dentro i blocchi di codice (come i nomi delle variabili Bash)
possono essere lasciati invariati.


## Stile della prosa

Se devi scrivere o modificare una porzione consistente della documentazione narrativa,
leggi la 
[Guida di Stile della Documentazione per lo Sviluppatore Google](https://developers.google.com/style/highlights).

### Principi di buon stile

*   *Controlla l'ortografia e la grammatica tuoi vostri contributi.* Molti editor
    contengono un correttore ortografico o dispongono di un plugin per il controllo.
    Puoi anche copiare il tuo testo in un Google Doc o in un altro software per documenti
    per un controllo ortografico e grammaticale più robusto.
*   *Usa una voce informale ed amichevole.* Scrivi la documentazione come una
    conversazione uno-ad-uno che tu stia tenendo con un'altra persona. Usa un tono
    di supporto.

Nota: Essere meno formali non significa essere meno tecnici. Semplifica la tua prosa,
non il contenuto tecnico.

*   *Evita disconoscimenti, opinioni, e giudizi di valore.* Parole come "facilmente",
    "solo", e "semplice" presuppongono assunzioni. Qualcosa può sembrare semplice 
    a te, ma essere difficile per un'altra persona. Quando possibile, 
    cerca di evitarle.
*   *Usa frasi semplici che vanno dirette al punto, senza gerghi complicati.* 
    Frasi composte, catene di clausole, e idiomi locali possono rendere il testo
    difficile da capire e tradurre. Se una frase può essere divisa in due,
    va probabilmente fatto. Evita i punti e virgola. Usa i punti elenco quando appropriato.
*   *Fornisci il contesto.* Non usare abbreviazioni senza spiegarle. Non citare
    progetti che non appartengano a TensorFlow senza collegamenti ad essi. Spiega perché 
    il codice è scritto nel modo in cui lo è.

## Guida all'uso

### Operazioni

Usa `# ⇒` invece di un segno singolo di uguale quando vuoi far vedere ciò che ritorna 
un'operazione.

```python
# 'input' is a tensor of shape [2, 3, 5] 
(tf.expand_dims(input, 0))  # ⇒ [1, 2, 3, 5]
```

### Tensori

Quando stai parlando di un tensore in genere, non usare la lettera maiuscola per la parola
*tensore*. Quando stai parlando dell'oggetto specifico passato o ritornato da un'operazione,
la parola *Tensor* dovrebbe avere la lettera maiuscola ed essere tra apici inversi, 
perché stai parlando di un oggetto `Tensor`.

Non usare la parola *Tensors* (plurale) per descrivere più oggetti `Tensor`
a meno che tu non stia realmente parlando di un oggetto `Tensors` object. Scrivi, invece "una lista
(o una collezione) di oggetti `Tensor`".

Usa la parola *shape* per dettagliare le dimensioni di un tensore, e mostra lo shape
tra parentesi quadre con apici inversi. Per esempio:

<pre><code>
If `input` is a three-dimensional tensor with shape `[3, 4, 3]`, this operation
returns a three-dimensional tensor with shape `[6, 8, 6]`.
</code></pre>
