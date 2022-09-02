# hda_nli_hindi

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/hda_nli_hindi)
- [Huggingface](https://huggingface.co/datasets/hda_nli_hindi)

## HDA hindi nli

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:hda_nli_hindi/HDA hindi nli')
```

- **说明**：

```
This dataset is a recasted version of the Hindi Discourse Analysis Dataset used to train models for Natural Language Inference Tasks in Low-Resource Languages like Hindi.
```

- **许可**：MIT License
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 9970
`'train'` | 31892
`'validation'` | 9460

- **特征**：

```json
{
    "premise": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "hypothesis": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "label": {
        "num_classes": 2,
        "names": [
            "not-entailment",
            "entailment"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    },
    "topic": {
        "num_classes": 5,
        "names": [
            "Argumentative",
            "Descriptive",
            "Dialogic",
            "Informative",
            "Narrative"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```

## hda nli hindi

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:hda_nli_hindi/hda nli hindi')
```

- **说明**：

```
This dataset is a recasted version of the Hindi Discourse Analysis Dataset used to train models for Natural Language Inference Tasks in Low-Resource Languages like Hindi.
```

- **许可**：MIT License
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 9970
`'train'` | 31892
`'validation'` | 9460

- **特征**：

```json
{
    "premise": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "hypothesis": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "label": {
        "num_classes": 2,
        "names": [
            "not-entailment",
            "entailment"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    },
    "topic": {
        "num_classes": 5,
        "names": [
            "Argumentative",
            "Descriptive",
            "Dialogic",
            "Informative",
            "Narrative"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```
