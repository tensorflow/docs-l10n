# TensorFlow 文档风格指南

## 最佳做法

- 关注用户意图和受众。
- 使用日常用词，保持句子简短。
- 使用一致的句子结构、措辞和大写。
- 使用标题和列表使文档更易于浏览。
- [Google 开发者文档风格指南](https://developers.google.com/style/highlights)很有帮助。

## Markdown

除了少数例外，TensorFlow 使用与 [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/) (GFM) 类似的 Markdown 语法。本部分说明了 GFM Markdown 语法与 TensorFlow 文档所使用的 Markdown 之间的区别。

### 撰写关于代码的内容

#### 代码的内嵌提及

在文本中使用以下符号时，请在符号周围加上 <code>&lt;code data-md-type="codespan"&gt;backticks</code>：

- 参数名称：<code>&lt;code data-md-type="codespan"&gt;input</code>、<code>&lt;code data-md-type="codespan"&gt;x</code>、<code>&lt;code data-md-type="codespan"&gt;tensor</code>
- 返回的张量名称：<code>&lt;code data-md-type="codespan"&gt;output</code>、<code>&lt;code data-md-type="codespan"&gt;idx</code>、<code>&lt;code data-md-type="codespan"&gt;out</code>
- 数据类型：<code>&lt;code data-md-type="codespan"&gt;int32</code>、<code>&lt;code data-md-type="codespan"&gt;float</code>、<code>&lt;code data-md-type="codespan"&gt;uint8</code>
- 文本中引用的其他运算名称：<code>&lt;code data-md-type="codespan"&gt;list_diff()</code>、<code>&lt;code data-md-type="codespan"&gt;shuffle()</code>
- 类名：<code>&lt;code data-md-type="codespan"&gt;tf.Tensor</code>、<code>&lt;code data-md-type="codespan"&gt;Strategy</code>
- 文件名：<code>&lt;code data-md-type="codespan"&gt;image_ops.py</code>、<code>&lt;code data-md-type="codespan"&gt;/path_to_dir/file_name</code>
- 数学表达式或条件：<code>&lt;code data-md-type="codespan"&gt;-1-input.dims() &lt;= dim &lt;= input.dims()</code>

#### 代码块

使用三个反引号来打开和关闭代码块。另外，也可以选择在第一个反引号组后面指定编程语言，例如：

<pre><code>
```python
# some python code here
```</code></pre>

### Markdown 中的链接

#### 此仓库中各文件之间的链接

在仓库中的各文件之间使用相对链接。这适用于 [tensorflow.org](https://www.tensorflow.org) 和 [GitHub](https://github.com/tensorflow/docs/tree/master/site/en)：<br><code>[Custom layers](../tutorials/eager/custom_layers.ipynb)</code> 会在网站上生成[自定义层](https://www.tensorflow.org/tutorials/eager/custom_layers)。

#### API 文档的链接

网站发布后会转换 API 链接。要链接到符号的 API 参考页面，请在反引号中包含完整的符号路径：

- <code>&lt;code data-md-type="codespan"&gt;tf.data.Dataset</code> 生成 [`tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset)

对于 C++ API，请使用命名空间路径：

- `tensorflow::Tensor` 生成 [tensorflow::Tensor](https://www.tensorflow.org/api_docs/cc/class/tensorflow/tensor)

#### 外部链接

对于外部链接（包括 <var>https://www.tensorflow.org</var> 上不在 `tensorflow/docs` 仓库中的文件），请使用具有完整 URI 的标准 Markdown 链接。

要链接到源代码，请使用以 <var>https://www.github.com/tensorflow/tensorflow/blob/master/</var> 开头的链接，后接以 GitHub 根开头的文件名。

这种 URI 命名方案确保 <var>https://www.tensorflow.org</var> 可以将链接转发到与您正在查看的文档版本相对应的代码分支。

不要在链接中包含 URI 查询参数。

文件路径使用下划线表示空格，例如 `custom_layers.ipynb`。

在链接中包含要在网站*和* GitHub 上使用的文件扩展名，例如，<br><code>[Custom layers](../tutorials/eager/custom_layers.ipynb)</code>。

### Markdown 中的数学表达式

编辑 Markdown 文件时，您可以在 TensorFlow 中使用 MathJax，但请注意以下几点：

- MathJax 可在 [tensorflow.org](https://www.tensorflow.org) 上正确呈现。
- MathJax 无法在 GitHub 上正确呈现。
- 对于不熟悉的开发者而言，这种表示法可能会令人不快。
- 为了保持一致性，[tensorflow.org](https://www.tensorflow.org) 遵循与 Jupyter/Colab 相同的规则。

在 MathJax 块周围使用 <code>$$</code>：

<pre><code>$$
E=\frac{1}{2n}\sum_x\lVert (y(x)-y'(x)) \rVert^2
$$</code></pre>

$$ E=\frac{1}{2n}\sum_x\lVert (y(x)-y'(x)) \rVert^2 $$

使用 <code>$ ... $</code> 包装内嵌 MathJax 表达式：

<pre><code>
This is an example of an inline MathJax expression: $ 2 \times 2 = 4 $</code></pre>

这是一个内嵌 MathJax 表达式的示例：$ 2 \times 2 = 4 $

<code>\( ... \)</code> 分隔符也适用于内嵌数学公式，但 $ 格式有时更易于阅读。

注：如果您需要在文本或 MathJax 表达式中使用美元符号，请使用前导斜杠对其进行转义：`\$`。代码块内的美元符号（例如 Bash 变量名称）不需要转义。

## 散文风格

如果您要编写或编辑相当一部分叙述性文档，请阅读 [Google 开发者文档风格指南](https://developers.google.com/style/highlights)。

### 良好风格的原则

- *检查文稿中的拼写和语法*。大多数编辑器都包含拼写检查器或提供了拼写检查插件。您还可以将文本粘贴到 Google 文档或其他文档软件中，以进行更可靠的拼写和语法检查。
- *使用轻松而友好的表达*。像对话一样编写 TensorFlow 文档，就像您正在与其他人一对一交谈一样。在文章中使用支持性语气。

注：不太正式并不意味着技术性不强。简化您的文章，而不是技术内容。

- *避免免责声明、意见和价值判断*。像“轻松”、“仅仅”和“简单”这样的词都带有假设意味。某件事对您来说似乎很容易，但对另一个人来说却很难。尽可能避免这些情况。
- *使用简明扼要的句子，不要使用复杂的行话*。复合句、从句链和特定于位置的习语会使文本难以理解和翻译。如果一个句子可以拆分为两句，则应当这样做。避免使用分号。在适当的时候使用项目符号列表。
- *提供上下文*。不要在未加以解释的情况下使用缩写。不要在未提供链接的情况下提及非 TensorFlow 项目。解释这样编写代码的原因。

## 使用指南

### 运算

在 Markdown 文件中，如果要显示运算返回的内容，请使用 `# ⇒` 而不是单个等号。

```python
# 'input' is a tensor of shape [2, 3, 5]
tf.expand_dims(input, 0)  # ⇒ [1, 2, 3, 5]
```

在笔记本中，显示结果而不是添加注释（如果笔记本单元中的最后一个表达式未分配给变量，则会自动显示。）

在 API 参考文档中，优先使用  [doctest](docs_ref.md#doctest) 来显示结果。

### 张量

一般而言，在谈论张量时，请不要把 *tensor* 一词大写。当您谈论提供给运算或从运算返回的特定对象时，应当把 *Tensor* 一词大写并在其周围加上反引号，因为您是在讨论 `Tensor` 对象。

除非您真的在讨论 `Tensors` 对象，否则不要使用 *Tensors*（复数）一词来描述多个 `Tensor` 对象。相反，您可以说“`Tensor` 对象的列表（或集合）”。

使用 *shape* 一词来详细描述张量的轴，并在带反引号的方括号中显示该形状。例如：

<pre><code>
If `input` is a three-axis `Tensor` with shape `[3, 4, 3]`, this operation
returns a three-axis `Tensor` with shape `[6, 8, 6]`.
</code></pre>

如上所述，在讨论 `Tensor` 形状的元素时，优先使用“轴”或“索引”而不是“维度”。否则，很容易将“维度”与向量空间的维度混淆。“三维向量”具有一个长度为 3 的轴。
