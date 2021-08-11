<!--* freshness: { owner: 'kempy' } *-->

# TensorFlow Hub 库概述

借助 [`tensorflow_hub`](https://github.com/tensorflow/hub) 库，您能够以最少的代码量在 TensorFlow 程序中下载并重用经过训练的模型。加载训练的模型的主要方式是使用 `hub.KerasLayer` API。

```python
import tensorflow_hub as hub

embed = hub.KerasLayer("https://tfhub.dev/google/nnlm-en-dim128/2")
embeddings = embed(["A long sentence.", "single-word", "http://example.com"])
print(embeddings.shape, embeddings.dtype)
```

## 设置下载的缓存位置。

默认情况下，`tensorflow_hub` 使用系统范围的临时目录来缓存下载和未压缩的模型。有关使用其他可能更持久位置的选项，请参阅[缓存](caching.md)。

## API 稳定性

尽管我们希望避免重大变更，但此项目仍在积极开发中，尚不能保证具有一个稳定的 API 或模型格式。

## 公平性

与所有机器学习一样，[公平性](http://ml-fairness.com)是一个[重要](https://research.googleblog.com/2016/10/equality-of-opportunity-in-machine.html)考量因素。许多预训练的模型都是基于大型数据集训练的。在重用任何模型时，请务必牢记该模型基于哪些数据进行了训练（以及其中是否存在任何现有偏差）与这些数据如何影响您的使用。

## 安全性

由于它们包含任意 TensorFlow 计算图，因此可以将模型视为程序。[安全地使用 TensorFlow](https://github.com/tensorflow/tensorflow/blob/master/SECURITY.md) 描述了从不受信任的来源引用模型带来的安全隐患。

## 后续步骤

- [使用库](tf2_saved_model.md)
- [可重用的 SavedModel](reusable_saved_models.md)
