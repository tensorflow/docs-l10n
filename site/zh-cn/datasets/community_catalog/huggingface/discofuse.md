# discofuse

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/discofuse)
- [Huggingface](https://huggingface.co/datasets/discofuse)

## discofuse-sport

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:discofuse/discofuse-sport')
```

- **说明**：

```
DISCOFUSE is a large scale dataset for discourse-based sentence fusion.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 445521
`'train'` | 43291020
`'validation'` | 440902

- **特征**：

```json
{
    "connective_string": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "discourse_type": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "coherent_second_sentence": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "has_coref_type_pronoun": {
        "dtype": "float32",
        "id": null,
        "_type": "Value"
    },
    "incoherent_first_sentence": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "incoherent_second_sentence": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "has_coref_type_nominal": {
        "dtype": "float32",
        "id": null,
        "_type": "Value"
    },
    "coherent_first_sentence": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## discofuse-wikipedia

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:discofuse/discofuse-wikipedia')
```

- **说明**：

```
DISCOFUSE is a large scale dataset for discourse-based sentence fusion.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 163657
`'train'` | 16310585
`'validation'` | 168081

- **特征**：

```json
{
    "connective_string": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "discourse_type": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "coherent_second_sentence": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "has_coref_type_pronoun": {
        "dtype": "float32",
        "id": null,
        "_type": "Value"
    },
    "incoherent_first_sentence": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "incoherent_second_sentence": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "has_coref_type_nominal": {
        "dtype": "float32",
        "id": null,
        "_type": "Value"
    },
    "coherent_first_sentence": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```
