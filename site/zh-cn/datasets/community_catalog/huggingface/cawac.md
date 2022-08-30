# cawac

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/cawac)
- [Huggingface](https://huggingface.co/datasets/cawac)

Use the following command to load this dataset in TFDS:

```python
ds = tfds.load('huggingface:cawac')
```

- **Description**:

```
caWaC is a 780-million-token web corpus of Catalan built from the .cat top-level-domain in late 2013.
```

- **许可**：CC BY-SA 3.0
- **Version**: 0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 24745986

- **特征**：

```json
{
    "sentence": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```
