<!--* freshness: { owner: 'kempy' reviewed: '2021-03-09' } *-->

# 文本任务的通用 SavedModel API

本页面介绍用于文本相关任务的 [TF2 SavedModel](../tf2_saved_model.md) 应当如何实现[可重用的 SavedModel API](../reusable_saved_models.md)。（这会替换现已弃用的 [TF1 Hub 格式](../common_signatures/text.md)的[通用文本签名](../tf1_hub_module)。）

## 概述

有几个 API 可用于计算**文本嵌入向量**（也称为文本的密集表示或文本特征向量）。

- *来自文本输入的文本嵌入向量*的 API 由可将一批字符串映射到一批嵌入向量的 SavedModel 实现。此 API 非常易用，TF Hub 上的许多模型都已实现它。但是，此 API 不允许在 TPU 上微调模型。

- *包含预处理输入的文本嵌入向量*的 API 可解决相同的任务，但它由两个单独的 SavedModel 实现：

    - 一个*预处理程序*，可以在 tf.data 输入流水线中运行并将字符串和其他可变长度数据转换为数值张量，
    - 一个*编码器*，接受预处理程序的结果并执行嵌入向量计算的可训练部分。

    这种拆分允许在馈送到训练循环之前对输入进行异步预处理。特别是，它允许构建可在 [TPU](https://www.tensorflow.org/guide/tpu) 上运行和微调的编码器。

- *包含 Transformer 编码器的文本嵌入向量*的 API 可将预处理输入中文本嵌入向量的 API 扩展为 BERT 和其他 Transformer 编码器的特殊情况。

    - *预处理程序*扩展为从多段输入文本中构建编码器输入。
    - *Transformer 编码器*公开各个词例的上下文感知嵌入向量。

在每种情况下，除非模型文档另有规定，否则文本输入均为 UTF-8 编码的字符串，通常为纯文本。

无论使用哪种 API，都已经针对来自不同语言和域的文本预训练了不同的模型，并考虑了不同的任务。因此，并非每个文本嵌入向量模型都适用于所有问题。

<a name="feature-vector"></a>
<a name="text-embeddings-from-text"></a>

## 来自文本输入的文本嵌入向量

**来自文本输入的文本嵌入向量**的 SavedModel 接受形状为 `[batch_size]` 的字符串张量中的一批输入，并将它们映射到形状为 `[batch_size, dim]` 的 float32 张量，其中包含输入的密集表示（特征向量）。

### 用法概要

```python
obj = hub.load("path/to/model")
text_input = ["A long sentence.",
              "single-word",
              "http://example.com"]
embeddings = obj(text_input)
```

从在训练模式下运行模型的[可重用 SavedModel API](../reusable_saved_models.md) 召回（例如，用于随机失活）可能需要关键字参数 `obj(..., training=True)`，并且该 `obj` 在适用时提供特性 `.variables`、`.trainable_variables` 和 `.regularization_losses`。

在 Keras 中，所有这些工作都由以下代码完成：

```python
embeddings = hub.KerasLayer("path/to/model", trainable=...)(text_input)
```

### 分布式训练

如果将文本嵌入向量用作通过分布策略进行训练的模型的一部分，则对 `hub.load("path/to/model")` 或 `hub.KerasLayer("path/to/model", ...)` 的调用必须在 DistributionStrategy 范围内发生，以便以分布式方式创建模型的变量。例如：

```python
  with strategy.scope():
    ...
    model = hub.load("path/to/model")
    ...
```

### 示例

- Colab 教程[影评文本分类](https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/tf2_text_classification.ipynb)。

<a name="text-embeddings-preprocessed"></a>

## 包含预处理输入的文本嵌入向量

**包含预处理输入的文本嵌入向量**由两个单独的 SavedModel 实现：

- 一个**预处理程序**，可将形状为 `[batch_size]` 的字符串张量映射到数值张量词典，
- 一个**编码器**，接受预处理程序返回的张量字典，执行嵌入向量计算的可训练部分，并返回一个输出字典。键 `"default"` 下的输出是一个形状为 `[batch_size, dim]` 的 float32 张量。

这允许在输入流水线中运行预处理程序，但会将编码器计算的嵌入向量作为更大模型的一部分进行微调。特别是，它允许构建可在 [TPU](https://www.tensorflow.org/guide/tpu) 上运行和微调的编码器。

它是一个实现细节，涉及到哪些张量包含在预处理程序的输出中，以及除了 `"default"` 外，哪些额外的张量（如果有）包含在编码器的输出中。

编码器的文档必须指定使用哪个预处理程序。通常，只有一个正确选择。

### 用法概要

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

从在训练模式下运行编码器的[可重用 SavedModel API](../reusable_saved_models.md) 召回（例如，用于随机失活）可能需要关键字参数 `encoder(..., training=True)`，并且该 `encoder` 在适用时提供特性 `.variables`、`.trainable_variables` 和 `.regularization_losses`。

`preprocessor` 模型可以具有 `.variables`，但并不意味着需要进一步训练。预处理不依赖于模式：如果 `preprocessor()` 具有完整的 `training=...` 参数，则它没有任何影响。

在 Keras 中，所有这些工作都由以下代码完成：

```python
encoder_inputs = hub.KerasLayer("path/to/preprocessor")(text_input)
encoder_outputs = hub.KerasLayer("path/to/encoder", trainable=True)(encoder_inputs)
embeddings = encoder_outputs["default"]
```

### 分布式训练

如果编码器用作通过分布策略进行训练的模型的一部分，则对 `hub.load("path/to/encoder")` 或 `hub.KerasLayer("path/to/encoder", ...)` 的调用必须发生在以下代码内部，

```python
  with strategy.scope():
    ...
```

以便以分布式方式重新创建编码器变量。

同样，如果预处理程序是已训练模型的一部分（如上面的简单示例所示），则也需要将其加载到分布策略范围内。但是，如果在输入流水线中使用了预处理程序（例如，在传递给 `tf.data.Dataset.map()` 的可调用对象中），则其加载必须发生在分布策略范围*以外*，以便将其变量（如果有）放置在主机 CPU 上。

### 示例

- Colab 教程[使用 BERT 进行文本分类](https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/text/classify_text_with_bert.ipynb)。

<a name="transformer-encoders"></a>

## 包含 Transfrmer 编码器的文本嵌入向量

文本的 Transformer 编码器在一批输入序列上运行，每个序列由 *n* 上的一些模型特定边界内的 *n* ≥ 1 个词例化文本段构成。对于 BERT 及其许多扩展，该边界为 2，因此它们接受单个段和段对。

**包含 Transformer 编码器的文本嵌入向量**的 API 可将包含预处理输入的文本嵌入向量的 API 扩展为此设置。

### 预处理程序

包含 Transformer 编码器的文本嵌入向量的预处理程序 SavedModel 实现包含预处理输入的文本嵌入向量的预处理程序 SavedModel 的 API（请参阅上文），此 API 提供了一种将单段文本输入直接映射到编码器输入的方法。

此外，预处理程序 SavedModel 还提供了两个可调用子对象 `tokenize` 和 `bert_pack_inputs`，它们分别用于词例化（每个段单独处理）以及将 *n* 个词例化段打包到编码器的一个输入序列中。每个子对象都遵循[可重用 SavedModel API](../reusable_saved_models.md)。

#### 用法概要

作为两个文本段的具体示例，我们看一个句子蕴涵任务，该任务询问前提（第一段）是否暗含假设（第二段）。

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

在 Keras 中，计算过程可表示为：

```python
tokenize = hub.KerasLayer(preprocessor.tokenize)
tokenized_hypotheses = tokenize(text_hypotheses)
tokenized_premises = tokenize(text_premises)

bert_pack_inputs = hub.KerasLayer(
    preprocessor.bert_pack_inputs,
    arguments=dict(seq_length=seq_length))  # Optional argument.
encoder_inputs = bert_pack_inputs([tokenized_premises, tokenized_hypotheses])
```

#### `tokenize` 的详细信息

对 `preprocessor.tokenize()` 的调用接受形状为 `[batch_size]` 的字符串张量，并返回形状为 `[batch_size, ...]` 的 [RaggedTensor](https://www.tensorflow.org/guide/ragged_tensor)，其值为表示输入字符串的 int32 词例 ID。`batch_size` 之后可以有 *r* ≥ 1 个不规则维度，但没有其他统一维度。

- 如果 *r*=1，则形状为 `[batch_size, (tokens)]`，每个输入都会简单地词例化为一个扁平词例序列。
- 如果 *r*&gt;1，则还有 *r*-1 个额外的分组级别。例如，[tensorflow_text.BertTokenizer](https://github.com/tensorflow/text/blob/v2.3.0/tensorflow_text/python/ops/bert_tokenizer.py#L138) 使用 *r*=2 来按词语对词例进行分组并产生形状 `[batch_size, (words), (tokens_per_word)]`。这取决于现有模型有多少额外的级别存在（如果有）以及它们代表哪些组。

用户可以（但不需要）修改词例化输入，例如，目的是适应将在打包编码器输入中强制执行的 seq_length 限制。分词器输出中的额外维度在这里能提供一些帮助（例如，遵守词边界），但在下一个步骤中会变得毫无意义。

就[可重用 SavedModel API](../reusable_saved_models.md) 而言，`preprocessor.tokenize` 对象可以具有 `.variables`，但并不意味着需要进一步训练。词例化不依赖于模式：如果 `preprocessor.tokenize()` 具有完整的 `training=...` 参数，则它没有任何影响。

#### `bert_pack_inputs` 的详细信息

对 `preprocessor.bert_pack_inputs()` 的调用接受词例化输入的 Python 列表（对每个输入段单独进行批处理），并返回一个张量字典，表示用于 Transformer 编码器模型的一批固定长度输入序列。

每个词例化输入都是一个形状为 `[batch_size, ...]` 的 int32 RaggedTensor，其中 batch_size 之后的不规则维度数量 *r* 为 1 或与 `preprocessor.tokenize()` 的输出中的数量相同（后者仅为方便起见；额外的维度在打包前已展平）。

打包过程会按照编码器的预期在输入段周围添加特殊词例。`bert_pack_inputs()` 调用完全实现了原始 BERT 模型及其许多扩展所使用的打包方案：打包序列以一个序列开始词例开头，随后是词例化段，每个段都以一个段结束词例终止。seq_length 之前的剩余位置（如果有）会被填充词例填满。

如果打包序列超过 seq_length，则 `bert_pack_inputs()` 会将其段截断为大小近似相等的前缀，以便使打包序列恰好适合 seq_length。

打包不依赖于模式：如果 `preprocessor.bert_pack_inputs()` 具有完整的 `training=...` 参数，则它没有任何影响。此外，`preprocessor.bert_pack_inputs` 不应具有变量或支持微调。

### 编码器

编码器在 `encoder_inputs` 的字典上调用，其方式与包含预处理输入的文本嵌入向量的 API 相同（请参阅上文），包括[可重用 SavedModel API](../reusable_saved_models.md) 中的规定 。

#### 用法概要

```python
enocder = hub.load("path/to/encoder")
enocder_outputs = encoder(encoder_inputs)
```

在 Keras 中的等效代码为：

```python
encoder = hub.KerasLayer("path/to/encoder", trainable=True)
encoder_outputs = encoder(encoder_inputs)
```

#### 详细信息

`encoder_outputs` 是具有以下键的张量字典。

<!-- TODO(b/172561269): More guidance for models trained without poolers. -->

- `"sequence_output"`：一个形状为 `[batch_size, seq_length, dim]` 的 float32 张量，包含对每个打包输入序列中每个词例的上下文感知嵌入向量。
- `"pooled_output"`：一个形状为 `[batch_size, dim]` 的 float32 张量，包含作为整体的每个输入序列的嵌入向量，以某种可训练方式派生自 sequence_output。
- `"default"`，包含预处理输入的文本嵌入向量的 API 需要此键：它是一个形状为 `[batch_size, dim]` 的 float32 张量，包含每个输入序列的嵌入向量（这可能只是 pooled_output 的别名）。

此 API 定义并未严格要求 `encoder_inputs` 的内容。但是，对于使用 BERT 样式输入的编码器，建议使用以下名称（来自 [TensorFlow Model Garden 的 NLP Modeling Toolkit](https://github.com/tensorflow/models/tree/master/official/nlp)）最大程度减少互换编码器和重用预处理程序模型时的摩擦：

- `"input_word_ids"`：一个形状为 `[batch_size, seq_length]` 的 int32 张量，包含打包输入序列（即，包括序列开始词例、段结束词例和填充）的词例 ID。
- `"input_mask"`：一个形状为 `[batch_size, seq_length]` 的 int32 张量，填充之前存在的所有输入词例的位置处的值为 1，填充词例的值为 0。
- `"input_type_ids"`：一个形状为 `[batch_size, seq_length]` 的 int32 张量，包含在相应位置处产生输入词例的输入段的索引。第一个输入段（索引 0）包括序列开始词例及其段结束词例。第二段和后续段（如果存在）包括其重复的段结束词例。填充词例再次获得索引 0。

### 分布式训练

对于在分布策略范围以内或以外加载预处理程序和编码器对象，同样的规则也适用于包含预处理输入的文本嵌入向量的 API（请参阅上文）。

### 示例

- Colab 教程[在 TPU 上使用 BERT 解决 GLUE 任务](https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/text/solve_glue_tasks_using_bert_on_tpu.ipynb)。
