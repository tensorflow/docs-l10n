# enwik8

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/enwik8)
- [Huggingface](https://huggingface.co/datasets/enwik8)

## enwik8

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:enwik8/enwik8')
```

- **说明**：

```
The dataset is based on the Hutter Prize (http://prize.hutter1.net) and contains the first 10^8 byte of Wikipedia
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 1128024

- **特征**：

```json
{
    "text": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## enwik8-raw

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:enwik8/enwik8-raw')
```

- **说明**：

```
The dataset is based on the Hutter Prize (http://prize.hutter1.net) and contains the first 10^8 byte of Wikipedia
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 1

- **特征**：

```json
{
    "text": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```
