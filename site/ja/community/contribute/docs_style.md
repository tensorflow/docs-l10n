# TensorFlow ドキュメントスタイルガイド

## ベストプラクティス

- ユーザーの意図と読者に焦点を絞る。
- 日頃よく使う言葉を使用し、文章を簡潔にする。
- 文章の構成、言い回し、大文字の使用には一貫性を持たせる。
- 見出しやリストを使用し、文書を検索しやすくする。
- [Google 開発者ドキュメントスタイルガイド](https://developers.google.com/style/highlights)を参考にする。

## Markdown

いくつかの例外を除き、TensorFlow は [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/)（GFM）に似ている Markdown 構文を使用しています。本セクションでは、GFM Markdown の構文と TensorFlow のドキュメントに使用される Markdown の違いを説明します。

### コードの記述

#### コードの記述をインライン化

テキストで使用する場合には、以下のシンボルの前後に<code>backticks</code>（バッククォート）を配置します。

- 引数名 : <code>input</code>、<code>x</code>、<code>tensor</code>
- 返されたテンソル名 : <code>output</code>、<code>idx</code>、<code>out</code>
- データ型 : <code>int32</code>、<code>float</code>、<code>uint8</code>
- テキストで参照する他の演算子名 : <code>list_diff()</code>、<code>shuffle()</code>
- クラス名 : <code>tf.Tensor</code>、<code>Strategy</code>
- ファイル名 : <code>image_ops.py</code>、<code>/path_to_dir/file_name</code>
- 数式や条件: <code>-1-input.dims() <= dim <= input.dims()</code>

#### コードブロック

3 つのバッククォートを使用して、コードブロックを開いたり閉じたりします。オプションで、最初のバッククォートグループの後にプログラミング言語を指定します。例を示します。

<pre><code>
```python
# some python code here
```
</code></pre>

### Markdown のリンク

#### このリポジトリ内のファイル間のリンク

リポジトリ内のファイル間の相対リンクを使用します。これは <br>[tensorflow.org](https://www.tensorflow.org) と [GitHub](https://github.com/tensorflow/docs/tree/master/site/en) で動作します。<br> <code>[Custom layers](../tutorials/eager/custom_layers.ipynb)</code>は、サイトに[カスタムレイヤー](https://www.tensorflow.org/tutorials/eager/custom_layers)を生成します。

#### API ドキュメントへのリンク

API リンクはサイトの公開時に変換されます。特定のシンボルの API リファレンスページにリンクするには、シンボルパス全体をバッククォートで囲みます。

- <code>tf.data.Dataset</code>は [tf.data.Dataset](https://www.tensorflow.org/api_docs/python/tf/data/Dataset) を生成します。

C++ API の場合は、ネームスペースパスを使用します。

- `tensorflow::Tensor`は [tensorflow::Tensor](https://www.tensorflow.org/api_docs/cc/class/tensorflow/tensor) を生成します。

#### 外部リンク

`tensorflow/docs`リポジトリにない <var>https://www.tensorflow.org</var> 上のファイルを含む外部リンクには、完全な URI を含んだ標準の Markdown リンクを使用します。

ソースコードをリンクするには、<var>https://www.github.com/tensorflow/tensorflow/blob/master/</var>  で始まるリンクを使用し、その後に GitHub のルートで始まるファイル名をつけます。

この URI 命名方式によって、表示しているドキュメントのバージョンに対応するコードのブランチへのリンクを <var>https://www.tensorflow.org</var> が転送できるようになります。

リンクには、URI クエリパラメータを含めてはいけません。

ファイルパスは、例えば`custom_layers.ipynb`のように、スペースにアンダースコアを使用します。

リンクにファイル拡張子を含めて、サイト*および* GitHub で使用できるようにします。例を示します。<br> <code>[Custom layers](../tutorials/eager/custom_layers.ipynb)</code>

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

<code>\( ... \)</code>区切り文字はインライン数学でも使えますが、$ 形式の方が読みやすい場合があります。

注意: テキストや MathJax の式の中でドル記号を使う必要がある場合は、先頭のスラッシュでエスケープします（`\$`）。コードブロック内のドル記号（Bash の変数名など）はエスケープする必要はありません。

## 文章のスタイル

ナラティブドキュメントの大部分を記述したり編集したりする場合には、[Google 開発者向けドキュメントスタイルガイド](https://developers.google.com/style/highlights)をお読みください。

### 良いスタイルの原則

- *貢献する内容のスペルと文法をチェックします。*ほとんどのエディタにはスペルチェッカーが含まれているか、利用可能なスペルチェックプラグインがあります。また、Google ドキュメントや他の文書ソフトにテキストを貼り付けて、より強力なスペルチェックや文法チェックをすることもできます。
- *堅くない話し言葉のような文調を使用します。*TensorFlow のドキュメントは会話のように、まるで相手に 1 対 1 で話しているかのように書きます。記事には支持的な口調を使用します。

注意: 堅苦しい表現を避けるというは、難しい技術的な内容を避けるという意味ではありません。技術的な内容を分かりやすく平易な言葉遣いで説明するこということです。

- *否定、意見、価値の判断は避けるようにします。*「easily（簡単）」「just（ただ）」「simple（単純）」などの言葉には、思い込みが込められています。自分には簡単に見えても、他の人には難しいことがあるかもしれません。これらは可能な限り避けるようにします。
- *複雑な専門用語は使わず、簡潔で要点を押さえた文章を使用します。*重文、節の連鎖、地域特有の慣用句は、テキストの理解や翻訳を難しくしてしまう可能性があります。文章を 2 つに分割できる場合は、分割した方がよいでしょう。セミコロンは避け、必要に応じて箇条書きを使用します。
- *文脈を提供します。*説明なしで略語を使用しないでください。TensorFlow 以外のプロジェクトについては、そのリンクを張らずに言及してはいけません。なぜコードをそのように記述するのか、理由を説明してください。

## 使用ガイド

### 演算子

演算子が何を返すかを表示する場合は、単一の等号の代わりに`# ⇒`を使用します。

```python
# 'input' is a tensor of shape [2, 3, 5]
(tf.expand_dims(input, 0))  # ⇒ [1, 2, 3, 5]
```

### テンソル

一般的なテンソルについて話す際には、*tensor* という英単語に大文字は使用しないでください。演算子に提供されたり演算子から返されたりする特定のオブジェクトについて話す際は、*Tensor* と頭文字を大文字にします。また、これは`Tensor`オブジェクトについて話しているため、その前後にバッククォートを追加します。

複数の`Tensor`オブジェクトを表現する際に、`Tensors`オブジェクトについて言及しているのでなければ *Tensors*（複数形）は使用しないでください。その代わりに "a list (or collection) of `Tensor` objects"（「Tensor オブジェクトのリスト（またはコレクション）」）と表記します。

*shape* という単語を使用して、テンソルの次元を詳しく説明し、バッククォート付きの角括弧で形状を示します。例を示します。

<pre><code>
If `input` is a three-dimensional tensor with shape `[3, 4, 3]`, this operation
returns a three-dimensional tensor with shape `[6, 8, 6]`.
</code></pre>
