# common_language

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/common_language)
- [Huggingface](https://huggingface.co/datasets/common_language)

## full

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:common_language/full')
```

- **说明**：

```
This dataset is composed of speech recordings from languages that were carefully selected from the CommonVoice database.
The total duration of audio recordings is 45.1 hours (i.e., 1 hour of material for each language).
The dataset has been extracted from CommonVoice to train language-id systems.
```

- **许可**：https://creativecommons.org/licenses/by/4.0/legalcode
- **版本**：0.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 5963
`'train'` | 22194
`'validation'` | 5888

- **特征**：

```json
{
    "client_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "path": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "sentence": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "age": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "gender": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "language": {
        "num_classes": 45,
        "names": [
            "Arabic",
            "Basque",
            "Breton",
            "Catalan",
            "Chinese_China",
            "Chinese_Hongkong",
            "Chinese_Taiwan",
            "Chuvash",
            "Czech",
            "Dhivehi",
            "Dutch",
            "English",
            "Esperanto",
            "Estonian",
            "French",
            "Frisian",
            "Georgian",
            "German",
            "Greek",
            "Hakha_Chin",
            "Indonesian",
            "Interlingua",
            "Italian",
            "Japanese",
            "Kabyle",
            "Kinyarwanda",
            "Kyrgyz",
            "Latvian",
            "Maltese",
            "Mangolian",
            "Persian",
            "Polish",
            "Portuguese",
            "Romanian",
            "Romansh_Sursilvan",
            "Russian",
            "Sakha",
            "Slovenian",
            "Spanish",
            "Swedish",
            "Tamil",
            "Tatar",
            "Turkish",
            "Ukranian",
            "Welsh"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```
