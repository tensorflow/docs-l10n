# TensorFlow 설명서 스타일 가이드

## 모범 사례

- Focus on user intent and audience.
- Use every-day words and keep sentences short.
- Use consistent sentence construction, wording, and capitalization.
- Use headings and lists to make your docs easier to scan.
- The [Google Developer Docs Style Guide](https://developers.google.com/style/highlights) is helpful.

## Markdown

With a few exceptions, TensorFlow uses a Markdown syntax similar to [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/) (GFM). This section explains differences between GFM Markdown syntax and the Markdown used for TensorFlow documentation.

### 코드 작성

#### 인라인 코드 언급

Put <code>`backticks`</code> around the following symbols when used in text:

- Argument names: <code>`input`</code>, <code>`x`</code>, <code>`tensor`</code>
- Returned tensor names: <code>`output`</code>, <code>`idx`</code>, <code>`out`</code>
- Data types: <code>`int32`</code>, <code>`float`</code>, <code>`uint8`</code>
- Other op names reference in text: <code>`list_diff()`</code>, <code>`shuffle()`</code>
- Class names: <code>`tf.Tensor`</code>, <code>`Strategy`</code>
- File name: <code>`image_ops.py`</code>, <code>`/path_to_dir/file_name`</code>
- Math expressions or conditions: <code>`-1-input.dims() &lt;= dim &lt;=     input.dims()`</code>

#### 코드 블록

Use three backticks to open and close a code block. Optionally, specify the programming language after the first backtick group, for example:

<pre><code>
```python
# some python code here
```
</code></pre>

### Links in Markdown and notebooks

#### Links between files in a repository

Use relative links between files in a single GitHub repository. Include the file extension.

For example, **this file you're reading** is from the [https://github.com/tensorflow/docs](https://github.com/tensorflow/docs) repository. Therefore, it can use relative paths to link to other files in the same repository like this:

- <code>\[Basics\]\(../../guide/basics.ipynb\)</code> produces [Basics](../../guide/basics.ipynb).

This is the prefered approach because this way the links on [tensorflow.org](https://www.tensorflow.org), [GitHub](https://github.com/tensorflow/docs){:.external} and [Colab](https://github.com/tensorflow/docs/tree/master/site/en/guide/bazics.ipynb){:.external} all work. Also, the reader stays in the same site when they click a link.

Note: You should include the file extension—such as `.ipynb` or `.md`—for relative links. It will rendered on `tensorflow.org` without an extension.

#### 외부 링크

For links to files that are not in the current repository, use standard Markdown links with the full URI. Prefer to link to the [tensorflow.org](https://www.tensorflow.org) URI if it's available.

To link to source code, use a link starting with <var>https://www.github.com/tensorflow/tensorflow/blob/master/</var>, followed by the file name starting at the GitHub root.

When linking off of [tensorflow.org](https://www.tensorflow.org), include a `{:.external}` on the Markdown link so that the "external link" symbol is shown.

- `[GitHub](https://github.com/tensorflow/docs){:.external}` produces [GitHub](https://github.com/tensorflow/docs){:.external}

Do not include URI query parameters in the link:

- Use: `https://www.tensorflow.org/guide/data`
- Not: `https://www.tensorflow.org/guide/data?hl=en`

#### Images

The advice in the previous section is for links to pages. Images are handled differently.

Generally, you should not check in images, and instead add the [TensorFlow-Docs team](https://github.com/tensorflow/docs) to your PR, and ask them to host the images on [tensorflow.org](https://www.tensorflow.org). This helps keep the size of your repository down.

If you do submit images to your repository, note that some systems do not handle relative paths to images. Prefer to use a full URL pointing to the image's eventual location on [tensorflow.org](https://www.tensorflow.org).

#### Links to API documentation

API links are converted when the site is published. To link to a symbol's API reference page, enclose the symbol path in backticks:

- <code>`tf.data.Dataset`</code> produces [`tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset)

Full paths are slightly preferred except for long paths. Paths can be abbreviated by dropping the leading path components. Partial paths will be converted to links if:

- There is at least one `.` in the path, and
- The partial path is unique within the project.

API paths are linked **for every project** with a Python API published on [tensorflow.org](https://www.tensorflow.org). You can easily link to multiple subprojects from a single file by wrapping the API names with backticks. For example:

- <code>`tf.metrics`</code>, <code>`tf_agents.metrics`</code>, <code>`text.metrics`</code> produces: `tf.metrics`, `tf_agents.metrics`, `text.metrics`.

For symbols with multiple path aliases there is a slight preference for the path that matches the API-page on [tensorflow.org](https://www.tensorflow.org). All aliases will redirect to the correct page.

### Math in Markdown

You may use MathJax within TensorFlow when editing Markdown files, but note the following:

- MathJax renders properly on [tensorflow.org](https://www.tensorflow.org).
- MathJax는 GitHub에서 올바르게 렌더링되지 않습니다.
- 이 표기법은 친숙하지 않은 개발자에게 적합하지 않습니다.
- For consistency [tensorflow.org](https://www.tensorflow.org) follows the same  rules as Jupyter/Colab.

Use <code>$$</code> around a block of MathJax:

<pre><code>$$
E=\frac{1}{2n}\sum_x\lVert (y(x)-y'(x)) \rVert^2
$$</code></pre>

$$ E=\frac{1}{2n}\sum_x\lVert (y(x)-y'(x)) \rVert^2 $$

Wrap inline MathJax expressions with <code>$ ... $</code>:

<pre><code>
This is an example of an inline MathJax expression: $ 2 \times 2 = 4 $
</code></pre>

This is an example of an inline MathJax expression: $ 2 \times 2 = 4 $

<code>\( ... \)</code> delimiters also work for inline math, but the $ form is sometimes more readable.

Note: If you need to use a dollar sign in text or MathJax expressions, escape it with a leading slash: `\$`. Dollar signs within code blocks (such as Bash variable names) do not need to be escaped.

## Prose style

If you are going to write or edit substantial portions of the narrative documentation, please read the [Google Developer Documentation Style Guide](https://developers.google.com/style/highlights).

### 좋은 스타일의 원리

- *Check the spelling and grammar in your contributions.* Most editors include a spell checker or have an available spell-checking plugin. You can also paste your text into a Google Doc or other document software for a more robust spelling and grammar check.
- *Use a casual and friendly voice.* Write TensorFlow documentation like a conversation—as if you're talking to another person one-on-one. Use a supportive tone in the article.

Note: Being less formal does not mean being less technical. Simplify your prose, not the technical content.

- *Avoid disclaimers, opinions, and value judgements.* Words like "easily", "just", and "simple" are loaded with assumptions. Something might seem easy to you, but be difficult for another person. Try to avoid these whenever possible.
- *Use simple, to the point sentences without complicated jargon.* Compound sentences, chains of clauses, and location-specific idioms can make text hard to understand and translate. If a sentence can be split in two, it probably should. Avoid semicolons. Use bullet lists when appropriate.
- *Provide context.* Don't use abbreviations without explaining them. Don't mention non-TensorFlow projects without linking to them. Explain why the code is written the way it is.

## Usage guide

### Ops

In markdown files, use `# ⇒` instead of a single equal sign when you want to show what an op returns.

```python
# 'input' is a tensor of shape [2, 3, 5]
tf.expand_dims(input, 0)  # ⇒ [1, 2, 3, 5]
```

In notebooks, display the result instead of adding a comment (If the last expression in a notebook cell is not assigned to a variable, it is automatically displayed.)

In API reference docs prefer using [doctest](docs_ref.md#doctest) to show results.

### 텐서

When you're talking about a tensor in general, don't capitalize the word *tensor*. When you're talking about the specific object that's provided to or returned from an op, then you should capitalize the word *Tensor* and add backticks around it because you're talking about a `Tensor` object.

Don't use the word *Tensors* (plural) to describe multiple `Tensor` objects unless you really are talking about a `Tensors` object. Instead, say "a list (or collection) of `Tensor` objects".

Use the word *shape* to detail the axes of a tensor, and show the shape in square brackets with backticks. For example:

<pre><code>
If `input` is a three-axis `Tensor` with shape `[3, 4, 3]`, this operation
returns a three-axis `Tensor` with shape `[6, 8, 6]`.
</code></pre>

As above, prefer "axis" or "index" over "dimension" when talking about the elements of a `Tensor`'s shape. Otherwise it's easy to confuse "dimension" with the dimension of a vector space. A "three-dimensional vector" has a single axis with length 3.
