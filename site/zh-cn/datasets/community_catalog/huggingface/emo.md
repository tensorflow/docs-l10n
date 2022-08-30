# emo

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/emo)
- [Huggingface](https://huggingface.co/datasets/emo)

## emo2019

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:emo/emo2019')
```

- **说明**：

```
In this dataset, given a textual dialogue i.e. an utterance along with two previous turns of context, the goal was to infer the underlying emotion of the utterance by choosing from four emotion classes - Happy, Sad, Angry and Others.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 5509
`'train'` | 30160

- **特征**：

```json
{
    "text": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "label": {
        "num_classes": 4,
        "names": [
            "others",
            "happy",
            "sad",
            "angry"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```
