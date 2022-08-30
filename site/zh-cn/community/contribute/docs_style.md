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

在文本中使用以下符号时，请在符号周围加上 <code>`backticks`</code>：

- 参数名称：<code>`input`</code>、<code>`x`</code>、<code>`tensor`</code>
- 返回的张量名称：<code>`output`</code>、<code>`idx`</code>、<code>`out`</code>
- 数据类型：<code>`int32`</code>、<code>`float`</code>、<code>`uint8`</code>
- 文本中引用的其他运算名称：<code>`list_diff()`</code>、<code>`shuffle()`</code>
- 类名：<code>`tf.Tensor`</code>、<code>`Strategy`</code>
- 文件名：<code>`image_ops.py`</code>、<code>`/path_to_dir/file_name`</code>
- 数学表达式或条件：<code>`-1-input.dims() &lt;= dim &lt;=     input.dims()`</code>

#### 代码块

使用三个反引号来打开和关闭代码块。另外，也可以选择在第一个反引号组后面指定编程语言，例如：

<pre><code>
```python
# some python code here
```</code></pre>

### Markdown 和笔记本中的链接

#### 仓库中各文件之间的链接

在单个 GitHub 仓库中的各文件之间使用相对链接。包括文件扩展名。

例如，**您正在阅读的这个文件**来自 [https://github.com/tensorflow/docs](https://github.com/tensorflow/docs) 仓库。因此，它可以使用相对路径链接到同一仓库中的其他文件，如下所示：

- <code>\[Basics\]\(../../guide/basics.ipynb\)</code> 产生 [Basics](../../guide/basics.ipynb)。

这是首选方式，因为这样可让 [tensorflow.org](https://www.tensorflow.org)、[GitHub](https://github.com/tensorflow/docs){:.external} 和 [Colab](https://github.com/tensorflow/docs/tree/master/site/en/guide/bazics.ipynb){:.external} 上的链接都正常工作。此外，当读者点击链接时，他们会停留在同一个站点。

注：您应当为相关链接包含文件扩展名，例如 `.ipynb` 或 `.md`。它将在没有扩展名的情况下呈现在 `tensorflow.org` 上。

#### 外部链接

对于不在当前仓库中的文件的链接，请使用包含完整 URI 的标准 Markdown 链接。如果可用，首选链接到 [tensorflow.org](https://www.tensorflow.org) URI。

要链接到源代码，请使用以 <var>https://www.github.com/tensorflow/tensorflow/blob/master/</var> 开头的链接，后接以 GitHub 根开头的文件名。

从 [tensorflow.org](https://www.tensorflow.org) 链接时，请在 Markdown 链接上包含 `{:.external}` 以便显示“外部链接”符号。

- `[GitHub](https://github.com/tensorflow/docs){:.external}` 产生 [GitHub](https://github.com/tensorflow/docs){:.external}

不要在链接中包含 URI 查询参数：

- 使用：`https://www.tensorflow.org/guide/data`
- 不使用：`https://www.tensorflow.org/guide/data?hl=en`

#### 图片

上一部分中的建议适用于页面的链接。图片的处理方式不同。

通常，您不应签入图片，而是将 [TensorFlow-Docs 团队](https://github.com/tensorflow/docs)添加到您的 PR，并要求他们在 [tensorflow.org](https://www.tensorflow.org) 上托管图片。这有助于减小仓库的大小。

如果您确实将图片提交到仓库，请注意某些系统不会处理图片的相对路径。首选使用指向图片在 [tensorflow.org](https://www.tensorflow.org) 上的最终位置的完整 URL。

#### API 文档的链接

网站发布后会转换 API 链接。要链接到符号的 API 参考页面，请使用反引号括起符号路径：

- <code>`tf.data.Dataset`</code> 产生 [`tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset)

除了长路径外，完整路径是略好的选择。可以通过删除前导路径组件来缩写路径。如果出现以下情况，部分路径将被转换为链接：

- 路径中至少有一个 `.`，并且
- 部分路径在项目中是唯一的。

**每个项目的** API 路径都与 [tensorflow.org](https://www.tensorflow.org) 上发布的 Python API 相链接。通过用反引号包装 API 名称，可以轻松地从单个文件链接到多个子项目。例如：

- <code>`tf.metrics`</code>、<code>`tf_agents.metrics`</code>、<code>`text.metrics`</code> 产生：`tf.metrics`、`tf_agents.metrics`、`text.metrics`。

对于具有多个路径别名的符号，更好的选择是与 [tensorflow.org](https://www.tensorflow.org) 上的 API 页面匹配的路径。所有别名都将重定向到正确的页面。

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
