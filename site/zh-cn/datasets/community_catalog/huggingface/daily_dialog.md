# daily_dialog

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/daily_dialog)
- [Huggingface](https://huggingface.co/datasets/daily_dialog)

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:daily_dialog')
```

- **说明**：

```
We develop a high-quality multi-turn dialog dataset, DailyDialog, which is intriguing in several aspects.
The language is human-written and less noisy. The dialogues in the dataset reflect our daily communication way
and cover various topics about our daily life. We also manually label the developed dataset with communication
intention and emotion information. Then, we evaluate existing approaches on DailyDialog dataset and hope it
benefit the research field of dialog systems.
```

- **许可**：cc-by-nc-sa-4.0
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 1000
`'train'` | 11118
`'validation'` | 1000

- **特征**：

```json
{
    "dialog": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "act": {
        "feature": {
            "num_classes": 5,
            "names": [
                "__dummy__",
                "inform",
                "question",
                "directive",
                "commissive"
            ],
            "names_file": null,
            "id": null,
            "_type": "ClassLabel"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "emotion": {
        "feature": {
            "num_classes": 7,
            "names": [
                "no emotion",
                "anger",
                "disgust",
                "fear",
                "happiness",
                "sadness",
                "surprise"
            ],
            "names_file": null,
            "id": null,
            "_type": "ClassLabel"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```
