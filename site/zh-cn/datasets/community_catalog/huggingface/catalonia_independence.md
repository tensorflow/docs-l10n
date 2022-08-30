# catalonia_independence

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/catalonia_independence)
- [Huggingface](https://huggingface.co/datasets/catalonia_independence)

## catalan

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:catalonia_independence/catalan')
```

- **说明**：

```
This dataset contains two corpora in Spanish and Catalan that consist of annotated Twitter messages for automatic stance detection. The data was collected over 12 days during February and March of 2019 from tweets posted in Barcelona, and during September of 2018 from tweets posted in the town of Terrassa, Catalonia.

Each corpus is annotated with three classes: AGAINST, FAVOR and NEUTRAL, which express the stance towards the target - independence of Catalonia.
```

- **许可**：CC BY-NC-SA 4.0
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 2010
`'train'` | 6028
`'validation'` | 2010

- **特征**：

```json
{
    "id_str": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "TWEET": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "LABEL": {
        "num_classes": 3,
        "names": [
            "AGAINST",
            "FAVOR",
            "NEUTRAL"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```

## spanish

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:catalonia_independence/spanish')
```

- **说明**：

```
This dataset contains two corpora in Spanish and Catalan that consist of annotated Twitter messages for automatic stance detection. The data was collected over 12 days during February and March of 2019 from tweets posted in Barcelona, and during September of 2018 from tweets posted in the town of Terrassa, Catalonia.

Each corpus is annotated with three classes: AGAINST, FAVOR and NEUTRAL, which express the stance towards the target - independence of Catalonia.
```

- **许可**：CC BY-NC-SA 4.0
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 2016
`'train'` | 6046
`'validation'` | 2015

- **特征**：

```json
{
    "id_str": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "TWEET": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "LABEL": {
        "num_classes": 3,
        "names": [
            "AGAINST",
            "FAVOR",
            "NEUTRAL"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```
