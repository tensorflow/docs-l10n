# hope_edi

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/hope_edi)
- [Huggingface](https://huggingface.co/datasets/hope_edi)

## english

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:hope_edi/english')
```

- **说明**：

```
A Hope Speech dataset for Equality, Diversity and Inclusion (HopeEDI) containing user-generated comments from the social media platform YouTube with 28,451, 20,198 and 10,705 comments in English, Tamil and Malayalam, respectively, manually labelled as containing hope speech or not.
```

- **许可**：Creative Commons Attribution 4.0 International Licence
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 22762
`'validation'` | 2843

- **特征**：

```json
{
    "text": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "label": {
        "num_classes": 3,
        "names": [
            "Hope_speech",
            "Non_hope_speech",
            "not-English"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```

## tamil

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:hope_edi/tamil')
```

- **说明**：

```
A Hope Speech dataset for Equality, Diversity and Inclusion (HopeEDI) containing user-generated comments from the social media platform YouTube with 28,451, 20,198 and 10,705 comments in English, Tamil and Malayalam, respectively, manually labelled as containing hope speech or not.
```

- **许可**：Creative Commons Attribution 4.0 International Licence
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 16160
`'validation'` | 2018

- **特征**：

```json
{
    "text": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "label": {
        "num_classes": 3,
        "names": [
            "Hope_speech",
            "Non_hope_speech",
            "not-Tamil"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```

## malayalam

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:hope_edi/malayalam')
```

- **说明**：

```
A Hope Speech dataset for Equality, Diversity and Inclusion (HopeEDI) containing user-generated comments from the social media platform YouTube with 28,451, 20,198 and 10,705 comments in English, Tamil and Malayalam, respectively, manually labelled as containing hope speech or not.
```

- **许可**：Creative Commons Attribution 4.0 International Licence
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 8564
`'validation'` | 1070

- **特征**：

```json
{
    "text": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "label": {
        "num_classes": 3,
        "names": [
            "Hope_speech",
            "Non_hope_speech",
            "not-malayalam"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```
