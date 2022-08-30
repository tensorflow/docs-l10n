# id_puisi

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/id_puisi)
- [Huggingface](https://huggingface.co/datasets/id_puisi)

Use the following command to load this dataset in TFDS:

```python
ds = tfds.load('huggingface:id_puisi')
```

- **Description**:

```
Puisi (poem) is an Indonesian poetic form. The dataset contains 7223 Indonesian puisi with its title and author.
```

- **许可**：无已知许可
- **Version**: 0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 7223

- **特征**：

```json
{
    "title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "author": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "puisi": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "puisi_with_header": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```
