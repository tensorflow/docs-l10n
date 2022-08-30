# beans

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/beans)
- [Huggingface](https://huggingface.co/datasets/beans)

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:beans')
```

- **说明**：

```
Beans is a dataset of images of beans taken in the field using smartphone
cameras. It consists of 3 classes: 2 disease classes and the healthy class.
Diseases depicted include Angular Leaf Spot and Bean Rust. Data was annotated
by experts from the National Crops Resources Research Institute (NaCRRI) in
Uganda and collected by the Makerere AI research lab.
```

- **许可**：无已知许可
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 128
`'train'` | 1034
`'validation'` | 133

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
        "num_classes": 3,
        "names": [
            "angular_leaf_spot",
            "bean_rust",
            "healthy"
        ],
        "id": null,
        "_type": "ClassLabel"
    }
}
```
