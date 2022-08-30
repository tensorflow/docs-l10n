# cawac

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/cawac)
- [Huggingface](https://huggingface.co/datasets/cawac)

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:cawac')
```

- **说明**：

```
caWaC is a 780-million-token web corpus of Catalan built from the .cat top-level-domain in late 2013.
```

- **许可**：CC BY-SA 3.0
- **版本**：0.0.0
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
