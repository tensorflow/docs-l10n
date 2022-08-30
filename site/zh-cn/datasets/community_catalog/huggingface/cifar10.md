# cifar10

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/cifar10)
- [Huggingface](https://huggingface.co/datasets/cifar10)

## plain_text

Use the following command to load this dataset in TFDS:

```python
ds = tfds.load('huggingface:cifar10/plain_text')
```

- **Description**:

```
The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images
per class. There are 50000 training images and 10000 test images.
```

- **许可**：无已知许可
- **Version**: 1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 10000
`'train'` | 50000

- **特征**：

```json
{
    "img": {
        "id": null,
        "_type": "Image"
    },
    "label": {
        "num_classes": 10,
        "names": [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```
