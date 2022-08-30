# gutenberg_time

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/gutenberg_time)
- [Huggingface](https://huggingface.co/datasets/gutenberg_time)

## gutenberg

Use the following command to load this dataset in TFDS:

```python
ds = tfds.load('huggingface:gutenberg_time/gutenberg')
```

- **Description**:

```
A clean data resource containing all explicit time references in a dataset of 52,183 novels whose full text is available via Project Gutenberg.
```

- **许可**：[需要更多信息]
- **Version**: 0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 120694

- **特征**：

```json
{
    "guten_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "hour_reference": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "time_phrase": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "is_ambiguous": {
        "dtype": "bool_",
        "id": null,
        "_type": "Value"
    },
    "time_pos_start": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "time_pos_end": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "tok_context": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```
