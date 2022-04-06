# TensorFlow ドキュメントスタイルガイド

## ベストプラクティス

- ユーザーの意図と読者に焦点を絞る。
- 日頃よく使う言葉を使用し、文章を簡潔にする。
- 文章の構成、言い回し、大文字の使用には一貫性を持たせる。
- 見出しやリストを使用し、文書を検索しやすくする。
- [Google 開発者ドキュメントスタイルガイド](https://developers.google.com/style/highlights)を参考にする。

## Markdown

いくつかの例外を除き、TensorFlow は [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/)（GFM）に似た Markdown 構文を使用しています。ここでは、GFM Markdown の構文と、TensorFlow のドキュメントに使用される Markdown との違いを説明します。

### コードの記述

#### コードの記述をインライン化

Put <code>`backticks`</code> around the following symbols when used in text:

- Argument names: <code>`input`</code>, <code>`x`</code>, <code>`tensor`</code>
- Returned tensor names: <code>`output`</code>, <code>`idx`</code>, <code>`out`</code>
- Data types: <code>`int32`</code>, <code>`float`</code>, <code>`uint8`</code>
- Other op names reference in text: <code>`list_diff()`</code>, <code>`shuffle()`</code>
- Class names: <code>`tf.Tensor`</code>, <code>`Strategy`</code>
- File name: <code>`image_ops.py`</code>, <code>`/path_to_dir/file_name`</code>
- Math expressions or conditions: <code>`-1-input.dims() &lt;= dim &lt;=     input.dims()`</code>

#### コードブロック

3 つのバッククォートを使用して、コードブロックを開いたり閉じたりします。オプションで、最初のバッククォートグループの後にプログラミング言語を指定します。例を示します。

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

#### 外部リンク

For links to files that are not in the current repository, use standard Markdown links with the full URI. Prefer to link to the [tensorflow.org](https://www.tensorflow.org) URI if it's available.

ソースコードをリンクするには、<var>https://www.github.com/tensorflow/tensorflow/blob/master/</var>  で始まるリンクを使用し、その後に GitHub のルートで始まるファイル名をつけます。

When linking off of [tensorflow.org](https://www.tensorflow.org), include a `{:.external}` on the Markdown link so that the "external link" symbol is shown.

- `[GitHub](https://github.com/tensorflow/docs){:.external}` produces [GitHub](https://github.com/tensorflow/docs){:.external}

Do not include URI query parameters in the link:

- Use: `https://www.tensorflow.org/guide/data`
- Not: `https://www.tensorflow.org/guide/data?hl=en`

#### Images

The advice in the previous section is for links to pages. Images are handled differently.

Generally, you should not check in images, and instead add the [TensorFlow-Docs team](https://github.com/tensorflow/docs) to your PR, and ask them to host the images on [tensorflow.org](https://www.tensorflow.org). This helps keep the size of your repository down.

If you do submit images to your repository, note that some systems do not handle relative paths to images. Prefer to use a full URL pointing to the image's eventual location on [tensorflow.org](https://www.tensorflow.org).

#### API ドキュメントへのリンク

API links are converted when the site is published. To link to a symbol's API reference page, enclose the symbol path in backticks:

- *貢献内容のスペルと文法をチェックします。*ほとんどのエディタにはスペルチェッカーが含まれているか、利用可能なスペルチェックプラグインがあります。また、Google ドキュメントや他の文書ソフトにテキストを貼り付けて、より強力なスペルチェックや文法チェックをすることもできます。

Full paths are slightly preferred except for long paths. Paths can be abbreviated by dropping the leading path components. Partial paths will be converted to links if:

- There is at least one `.` in the path, and
- *複雑な専門用語は使わず、簡潔で要点を押さえた文章を使用します。*重文、節の連鎖、場所特有の慣用句は、テキストの理解や翻訳を難しくする可能性があります。文章を 2 つに分割できる場合は、分割した方がよいでしょう。セミコロンは避け、必要に応じて箇条書きを使用します。

API paths are linked **for every project** with a Python API published on [tensorflow.org](https://www.tensorflow.org). You can easily link to multiple subprojects from a single file by wrapping the API names with backticks. For example:

- <code>`tf.metrics`</code>, <code>`tf_agents.metrics`</code>, <code>`text.metrics`</code> produces: `tf.metrics`, `tf_agents.metrics`, `text.metrics`.

For symbols with multiple path aliases there is a slight preference for the path that matches the API-page on [tensorflow.org](https://www.tensorflow.org). All aliases will redirect to the correct page.

### Markdown の数学

Markdown ファイルを編集する際には TensorFlow 内で MathJax を使用することができますが、以下の点に注意してください。

- MathJax は [tensorflow.org](https://www.tensorflow.org) で適切にレンダリングします。
- MathJax は GitHub では適切にレンダリングできません。
- この表記法は、経験の浅い開発者には難しいかもしれません。
- 一貫性を保つために、[tensorflow.org](https://www.tensorflow.org) は Jupyter や Colab と同じルールに従っています。

MathJax のブロックの前後に<code>$$</code>を使用します。

<pre><code>$$
E=\frac{1}{2n}\sum_x\lVert (y(x)-y'(x)) \rVert^2
$$</code></pre>

$$ E=\frac{1}{2n}\sum_x\lVert (y(x)-y'(x)) \rVert^2 $$

インラインの MathJax の式を<code>$ ... $</code>でラップします。

<pre><code>
This is an example of an inline MathJax expression: $ 2 \times 2 = 4 $
</code></pre>

これはインライン MathJax の式の例で、$ 2 \times 2 = 4 $ です。

<code>\( ... \)</code> delimiters also work for inline math, but the $ form is sometimes more readable.

注意: テキストや MathJax の式の中でドル記号を使う必要がある場合は、先頭のスラッシュでエスケープします（`\$`）。コードブロック内のドル記号（Bash の変数名など）はエスケープする必要はありません。

## 文章のスタイル

ナラティブドキュメントの大部分を記述したり編集したりする場合には、[Google 開発者向けドキュメントスタイルガイド](https://developers.google.com/style/highlights)をお読みください。

### 良いスタイルの原則

- *貢献内容のスペルと文法をチェックします。*ほとんどのエディタにはスペルチェッカーが含まれているか、利用可能なスペルチェックプラグインがあります。また、Google ドキュメントや他の文書ソフトにテキストを貼り付けて、より強力なスペルチェックや文法チェックをすることもできます。
- *堅くない話し言葉のような文調を使用します。*TensorFlow のドキュメントは会話のように、まるで相手に 1 対 1 で話しているかのように書きます。記事には支持的な口調を使用します。

注意: 堅苦しい表現を避けるというは、難しい技術的な内容を避けるという意味ではありません。技術的な内容を分かりやすく平易な言葉遣いで説明するこということです。

- *否定、意見、価値の判断は避けるようにします。*「easily（簡単）」「just（ただ）」「simple（単純）」などの言葉には、思い込みが込められています。自分には簡単に見えても、他の人には難しいことがあるかもしれません。これらは可能な限り避けるようにします。
- *複雑な専門用語は使わず、簡潔で要点を押さえた文章を使用します。*重文、節の連鎖、場所特有の慣用句は、テキストの理解や翻訳を難しくする可能性があります。文章を 2 つに分割できる場合は、分割した方がよいでしょう。セミコロンは避け、必要に応じて箇条書きを使用します。
- *文脈を提供します。*説明せずに略語を使用しないでください。TensorFlow 以外のプロジェクトについては、そのリンクを張らずに言及してはいけません。コードがそのように記述されている理由を説明してください。

## 使用ガイド

### 演算子

演算子が何を返すかを表示する場合は、単一の等号の代わりに `# ⇒` を使用します。

```python
# 'input' is a tensor of shape [2, 3, 5]
(tf.expand_dims(input, 0))  # ⇒ [1, 2, 3, 5]
```

ノートブックでは、コメントを追加する代わりに結果を表示します（ノートブックセルの最後の式が変数に割り当てられていない場合は、自動的にそれが表示されます）。

API リファレンスドキュメントでは、[doctest](docs_ref.md#doctest) を使って結果を表示することが推奨されています。

### テンソル

一般的にテンソルについて話す際は、*tensor* という単語を大文字にしてはいけません。演算子に提供されたり演算子から返されたりする特定のオブジェクトについて話す際は、*Tensor* と単語の頭文字を大文字にし、さらにこれは`Tensor` オブジェクトについて話しているため、その前後にバッククォートを追加します。

複数の `Tensor` オブジェクトを表現する際に、`Tensors` オブジェクトについて言及していない場合に *Tensors*（複数形）を使用してはいけません。その代わりに "a list (or collection) of `Tensor` objects"（「Tensor オブジェクトのリスト（またはコレクション）」）と表記します。

*shape* という単語を使用してテンソルの軸を詳しく説明し、バッククォート付きの角括弧で形状を示します。例を示します。

<pre><code>
If `input` is a three-dimensional tensor with shape `[3, 4, 3]`, this operation
returns a three-dimensional tensor with shape `[6, 8, 6]`.
</code></pre>

上述のとおり、`Tensor` の形状の要素について話す場合には、「dimension（次元）」ではなく、「axis（軸）」または「index（インデックス）」を使用することを推奨します。そうでない場合、「dimension」をベクトル空間の次元と混同する可能性があるためです。「three-dimensional vector（3 次元ベクトル)」は長さ 3 の単一の軸です。
