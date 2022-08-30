# biosses

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/biosses)
- [Huggingface](https://huggingface.co/datasets/biosses)

Use the following command to load this dataset in TFDS:

```python
ds = tfds.load('huggingface:biosses')
```

- **Description**:

```
BIOSSES is a benchmark dataset for biomedical sentence similarity estimation. The dataset comprises 100 sentence pairs, in which each sentence was selected from the TAC (Text Analysis Conference) Biomedical Summarization Track Training Dataset containing articles from the biomedical domain. The sentence pairs were evaluated by five different human experts that judged their similarity and gave scores ranging from 0 (no relation) to 4 (equivalent).
```

- **许可**：BIOSSES 根据 The GNU Common Public License v.3.0 的条款提供。

- **Version**: 0.0.0

- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 100

- **特征**：

```json
{
    "sentence1": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "sentence2": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "score": {
        "dtype": "float32",
        "id": null,
        "_type": "Value"
    }
}
```
