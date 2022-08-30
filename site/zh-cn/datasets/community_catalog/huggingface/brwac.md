# brwac

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/brwac)
- [Huggingface](https://huggingface.co/datasets/brwac)

Use the following command to load this dataset in TFDS:

```python
ds = tfds.load('huggingface:brwac')
```

- **Description**:

```
The BrWaC (Brazilian Portuguese Web as Corpus) is a large corpus constructed following the Wacky framework,
which was made public for research purposes. The current corpus version, released in January 2017, is composed by
3.53 million documents, 2.68 billion tokens and 5.79 million types. Please note that this resource is available
solely for academic research purposes, and you agreed not to use it for any commercial applications.
Manually download at https://www.inf.ufrgs.br/pln/wiki/index.php?title=BrWaC
```

- **许可**：无已知许可
- **Version**: 1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 3530796

- **特征**：

```json
{
    "doc_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "uri": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "text": {
        "feature": {
            "paragraphs": {
                "feature": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                },
                "length": -1,
                "id": null,
                "_type": "Sequence"
            }
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```
