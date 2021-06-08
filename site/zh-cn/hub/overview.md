<!--* freshness: { owner: 'kempy' reviewed: '2020-09-14' } *-->

# TensorFlow Hub

TensorFlow Hub 是用于存储可重用机器学习资产的开放仓库和库。[tfhub.dev](https://tfhub.dev) 仓库中提供了许多预训练模型：文本嵌入向量、图像分类模型、TF.js/TFLite 模型等。该仓库向[社区贡献者](https://tfhub.dev/s?subtype=publisher)开放。

借助 [`tensorflow_hub`](https://github.com/tensorflow/hub) 库，您可以下载并以最少的代码量在 TensorFlow 程序中重用这些模型。

```python
import tensorflow_hub as hub

model = hub.KerasLayer("https://tfhub.dev/google/nnlm-en-dim128/2")
embeddings = model(["The rain in Spain.", "falls",
                    "mainly", "In the plain!"])

print(embeddings.shape)  #(4,128)
```

## 后续步骤

- [在 tfhub.dev 上查找模型](https://tfhub.dev)
- [在 tfhub.dev 上发布模型](publish.md)
- TensorFlow Hub 库
    - [安装 TensorFlow Hub](installation.md)
    - [库概览](lib_overview.md)
- [关注教程](tutorials)
