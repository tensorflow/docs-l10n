# cats_vs_dogs

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/cats_vs_dogs)
- [Huggingface](https://huggingface.co/datasets/cats_vs_dogs)

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:cats_vs_dogs')
```

- **说明**：

```
A large set of images of cats and dogs. There are 1738 corrupted images that are dropped.
```

- **许可**：无已知许可
- **版本**：0.0.1
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 23410

- **特征**：

```json
{
    "image_file_path": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "image": {
        "decode": true,
        "id": null,
        "_type": "Image"
    },
    "labels": {
        "num_classes": 2,
        "names": [
            "cat",
            "dog"
        ],
        "id": null,
        "_type": "ClassLabel"
    }
}
```
