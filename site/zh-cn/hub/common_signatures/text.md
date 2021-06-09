<!--* freshness: { owner: 'kempy' reviewed: '2021-03-09' } *-->

# 文本的常用签名

本页面介绍应由 [TF1 Hub 格式](../tf1_hub_module.md)的模块为接受文本输入的任务实现的常用签名。（有关 [TF2 SavedModel 格式](../tf2_saved_model.md)，请参阅具有类似功能的 [SavedModel API](../common_saved_model_apis/text.md)。）

## 文本特征向量

**文本特征向量**模块可以根据文本特征创建密集向量表示。该模块接受一批形状为 `[batch_size]` 的字符串，并将其映射至形状为 `[batch_size, N]` 的 `float32` 张量。此张量通常被称为 `N` 维**文本嵌入向量**。

### 基本用法

```python
  embed = hub.Module("path/to/module")
  representations = embed([
      "A long sentence.",
      "single-word",
      "http://example.com"])
```

### 特征列用法

```python
    feature_columns = [
      hub.text_embedding_column("comment", "path/to/module", trainable=False),
    ]
    input_fn = tf.estimator.inputs.numpy_input_fn(features, labels, shuffle=True)
    estimator = tf.estimator.DNNClassifier(hidden_units, feature_columns)
    estimator.train(input_fn, max_steps=100)
```

## 说明

模块已在不同的领域和/或任务中进行了预训练，因此并非每个文本特征向量模块都适用于您的问题。例如，某些模块可能仅使用一种语言进行训练。

此接口不支持在 TPU 上微调文本表示，因为它要求模块同时实例化字符串处理和可训练变量。
