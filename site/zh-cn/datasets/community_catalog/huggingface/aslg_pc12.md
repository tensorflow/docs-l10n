# aslg_pc12

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/aslg_pc12)
- [Huggingface](https://huggingface.co/datasets/aslg_pc12)

Use the following command to load this dataset in TFDS:

```python
ds = tfds.load('huggingface:aslg_pc12')
```

- **Description**:

```
Synthetic English-ASL Gloss Parallel Corpus 2012
```

- **许可**：无已知许可
- **版本**：0.0.1
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 87710

- **特征**：

```json
{
    "gloss": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "text": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```
