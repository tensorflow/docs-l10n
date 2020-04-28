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

*   `tensorflow::Tensor` produces
    [tensorflow::Tensor](https://www.tensorflow.org/api_docs/cc/class/tensorflow/tensor)

#### External links

For external links, including files on <var>https://www.tensorflow.org</var>
that are not in the `tensorflow/docs` repository, use standard Markdown links
with the full URI.

To link to source code, use a link starting with
<var>https://www.github.com/tensorflow/tensorflow/blob/master/</var>, followed
by the file name starting at the GitHub root.

This URI naming scheme ensures that <var>https://www.tensorflow.org</var> can
forward the link to the branch of the code corresponding to the version of the
documentation you're viewing.

Do not include URI query parameters in the link.

File paths use underscores for spaces, for example, `custom_layers.ipynb`.

Include the file extension in links to use on the site *and* GitHub, for example,<br/>
<code>\[Custom layers\]\(../tutorials/eager/custom_layers.ipynb\)</code>.

### Math in Markdown

You may use MathJax within TensorFlow when editing Markdown files, but note the
following:

*   MathJax renders properly on [tensorflow.org](https://www.tensorflow.org).
*   MathJax does not render properly on GitHub.
*   This notation can be off-putting to unfamiliar developers.
*   For consistency [tensorflow.org](https://www.tensorflow.org) follows the
    same  rules as Jupyter/Colab.

Use <code>&#36;&#36;</code> around a block of MathJax:

<pre><code>&#36;&#36;
E=\frac{1}{2n}\sum_x\lVert (y(x)-y'(x)) \rVert^2
&#36;&#36;</code></pre>

$$
E=\frac{1}{2n}\sum_x\lVert (y(x)-y'(x)) \rVert^2
$$

Wrap inline MathJax expressions with <code>&#36; ... &#36;</code>:

<pre><code>
This is an example of an inline MathJax expression: &#36; 2 \times 2 = 4 &#36;
</code></pre>

This is an example of an inline MathJax expression: $ 2 \times 2 = 4 $

<code>&#92;&#92;( ... &#92;&#92;)</code> delimiters also work for inline math,
but the \$ form is sometimes more readable.

Note: If you need to use a dollar sign in text or MathJax expressions, escape it
with a leading slash: `\$`. Dollar signs within code blocks (such as Bash
variable names) do not need to be escaped.


## Prose style

If you are going to write or edit substantial portions of the narrative
documentation, please read the
[Google Developer Documentation Style Guide](https://developers.google.com/style/highlights).

### Principles of good style

*   *Check the spelling and grammar in your contributions.* Most editors
    include a spell checker or have an available spell-checking plugin. You can
    also paste your text into a Google Doc or other document software for a more
    robust spelling and grammar check.
*   *Use a casual and friendly voice.* Write TensorFlow documentation like a
    conversation—as if you're talking to another person one-on-one. Use a
    supportive tone in the article.

Note: Being less formal does not mean being less technical. Simplify your prose,
not the technical content.

*   *Avoid disclaimers, opinions, and value judgements.* Words like "easily",
    "just", and "simple" are loaded with assumptions. Something might seem easy
    to you, but be difficult for another person. Try to avoid these whenever
    possible.
*   *Use simple, to the point sentences without complicated jargon.* Compound
    sentences, chains of clauses, and location-specific idioms can make text
    hard to understand and translate. If a sentence can be split in two, it
    probably should. Avoid semicolons. Use bullet lists when appropriate.
*   *Provide context.* Don't use abbreviations without explaining them. Don't
    mention non-TensorFlow projects without linking to them. Explain why the
    code is written the way it is.

## Usage guide

### Ops

Use `# ⇒` instead of a single equal sign when you want to show what an op
returns.

```python
# 'input' is a tensor of shape [2, 3, 5] 
(tf.expand_dims(input, 0))  # ⇒ [1, 2, 3, 5]
```

### Tensors

When you're talking about a tensor in general, don't capitalize the word
*tensor*. When you're talking about the specific object that's provided to or
returned from an op, then you should capitalize the word *Tensor* and add
backticks around it because you're talking about a `Tensor` object.

Don't use the word *Tensors* (plural) to describe multiple `Tensor` objects
unless you really are talking about a `Tensors` object. Instead, say "a list (or
collection) of `Tensor` objects".

Use the word *shape* to detail the dimensions of a tensor, and show the shape in
square brackets with backticks. For example:

<pre><code>
If `input` is a three-dimensional tensor with shape `[3, 4, 3]`, this operation
returns a three-dimensional tensor with shape `[6, 8, 6]`.
</code></pre>
