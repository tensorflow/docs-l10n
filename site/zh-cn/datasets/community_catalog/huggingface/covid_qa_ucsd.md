# covid_qa_ucsd

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/covid_qa_ucsd)
- [Huggingface](https://huggingface.co/datasets/covid_qa_ucsd)

## en

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:covid_qa_ucsd/en')
```

- **说明**：

```
COVID-Dialogue-Dataset is amedical dialogue dataset about COVID-19 and other types of pneumonia.
Patients who are concerned that they may be infected by COVID-19 or other pneumonia consult doctors and doctors provide advice.
There are 603 consultations in English and 1393 consultations in Chinese.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 572

- **特征**：

```json
{
    "dialogue_id": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "dialogue_url": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "dialogue_turns": {
        "feature": {
            "speaker": {
                "num_classes": 2,
                "names": [
                    "Patient",
                    "Doctor"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "utterance": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            }
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## zh

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:covid_qa_ucsd/zh')
```

- **说明**：

```
COVID-Dialogue-Dataset is amedical dialogue dataset about COVID-19 and other types of pneumonia.
Patients who are concerned that they may be infected by COVID-19 or other pneumonia consult doctors and doctors provide advice.
There are 603 consultations in English and 1393 consultations in Chinese.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 1088

- **特征**：

```json
{
    "dialogue_id": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "dialogue_url": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "dialogue_turns": {
        "feature": {
            "speaker": {
                "num_classes": 2,
                "names": [
                    "\u75c5\u4eba",
                    "\u533b\u751f"
                ],
                "names_file": null,
                "id": null,
                "_type": "ClassLabel"
            },
            "utterance": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            }
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```
