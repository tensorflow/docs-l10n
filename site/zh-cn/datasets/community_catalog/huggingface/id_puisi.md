# id_puisi

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/id_puisi)
- [Huggingface](https://huggingface.co/datasets/id_puisi)

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:id_puisi')
```

- **说明**：

```
Puisi (poem) is an Indonesian poetic form. The dataset contains 7223 Indonesian puisi with its title and author.
```

- **许可**：无已知许可
- **版本**：0.0.0
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
