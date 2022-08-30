# gigaword

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/gigaword)
- [Huggingface](https://huggingface.co/datasets/gigaword)

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:gigaword')
```

- **说明**：

```
Headline-generation on a corpus of article pairs from Gigaword consisting of
around 4 million articles. Use the 'org_data' provided by
https://github.com/microsoft/unilm/ which is identical to
https://github.com/harvardnlp/sent-summary but with better format.

There are two features:
  - document: article.
  - summary: headline.
```

- **许可**：无已知许可
- **版本**：1.2.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 1951
`'train'` | 3803957
`'validation'` | 189651

- **特征**：

```json
{
    "document": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "summary": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```
