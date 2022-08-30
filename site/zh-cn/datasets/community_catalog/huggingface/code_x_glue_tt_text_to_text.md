# code_x_glue_tt_text_to_text

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/code_x_glue_tt_text_to_text)
- [Huggingface](https://huggingface.co/datasets/code_x_glue_tt_text_to_text)

## da_en

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:code_x_glue_tt_text_to_text/da_en')
```

- **说明**：

```
CodeXGLUE text-to-text dataset, available at https://github.com/microsoft/CodeXGLUE/tree/main/Text-Text/text-to-text

The dataset we use is crawled and filtered from Microsoft Documentation, whose document located at https://github.com/MicrosoftDocs/.
```

- **许可**：无已知许可
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 1000
`'train'` | 42701
`'validation'` | 1000

- **特征**：

```json
{
    "id": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "source": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "target": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## lv_en

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:code_x_glue_tt_text_to_text/lv_en')
```

- **说明**：

```
CodeXGLUE text-to-text dataset, available at https://github.com/microsoft/CodeXGLUE/tree/main/Text-Text/text-to-text

The dataset we use is crawled and filtered from Microsoft Documentation, whose document located at https://github.com/MicrosoftDocs/.
```

- **许可**：无已知许可
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 1000
`'train'` | 18749
`'validation'` | 1000

- **特征**：

```json
{
    "id": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "source": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "target": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## no_en

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:code_x_glue_tt_text_to_text/no_en')
```

- **说明**：

```
CodeXGLUE text-to-text dataset, available at https://github.com/microsoft/CodeXGLUE/tree/main/Text-Text/text-to-text

The dataset we use is crawled and filtered from Microsoft Documentation, whose document located at https://github.com/MicrosoftDocs/.
```

- **许可**：无已知许可
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 1000
`'train'` | 44322
`'validation'` | 1000

- **特征**：

```json
{
    "id": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "source": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "target": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## zh_en

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:code_x_glue_tt_text_to_text/zh_en')
```

- **说明**：

```
CodeXGLUE text-to-text dataset, available at https://github.com/microsoft/CodeXGLUE/tree/main/Text-Text/text-to-text

The dataset we use is crawled and filtered from Microsoft Documentation, whose document located at https://github.com/MicrosoftDocs/.
```

- **许可**：无已知许可
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 1000
`'train'` | 50154
`'validation'` | 1000

- **特征**：

```json
{
    "id": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "source": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "target": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```
