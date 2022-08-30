# code_x_glue_ct_code_to_text

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/code_x_glue_ct_code_to_text)
- [Huggingface](https://huggingface.co/datasets/code_x_glue_ct_code_to_text)

## go

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:code_x_glue_ct_code_to_text/go')
```

- **说明**：

```
CodeXGLUE code-to-text dataset, available at https://github.com/microsoft/CodeXGLUE/tree/main/Code-Text/code-to-text

The dataset we use comes from CodeSearchNet and we filter the dataset as the following:
- Remove examples that codes cannot be parsed into an abstract syntax tree.
- Remove examples that #tokens of documents is < 3 or >256
- Remove examples that documents contain special tokens (e.g. <img ...> or https:...)
- Remove examples that documents are not English.
```

- **许可**：无已知许可
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 8122
`'train'` | 167288
`'validation'` | 7325

- **特征**：

```json
{
    "id": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "repo": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "path": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "func_name": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "original_string": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "language": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "code": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "code_tokens": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "docstring": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "docstring_tokens": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "sha": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "url": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## java

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:code_x_glue_ct_code_to_text/java')
```

- **说明**：

```
CodeXGLUE code-to-text dataset, available at https://github.com/microsoft/CodeXGLUE/tree/main/Code-Text/code-to-text

The dataset we use comes from CodeSearchNet and we filter the dataset as the following:
- Remove examples that codes cannot be parsed into an abstract syntax tree.
- Remove examples that #tokens of documents is < 3 or >256
- Remove examples that documents contain special tokens (e.g. <img ...> or https:...)
- Remove examples that documents are not English.
```

- **许可**：无已知许可
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 10955
`'train'` | 164923
`'validation'` | 5183

- **特征**：

```json
{
    "id": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "repo": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "path": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "func_name": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "original_string": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "language": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "code": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "code_tokens": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "docstring": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "docstring_tokens": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "sha": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "url": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## javascript

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:code_x_glue_ct_code_to_text/javascript')
```

- **说明**：

```
CodeXGLUE code-to-text dataset, available at https://github.com/microsoft/CodeXGLUE/tree/main/Code-Text/code-to-text

The dataset we use comes from CodeSearchNet and we filter the dataset as the following:
- Remove examples that codes cannot be parsed into an abstract syntax tree.
- Remove examples that #tokens of documents is < 3 or >256
- Remove examples that documents contain special tokens (e.g. <img ...> or https:...)
- Remove examples that documents are not English.
```

- **许可**：无已知许可
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 3291
`'train'` | 58025
`'validation'` | 3885

- **特征**：

```json
{
    "id": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "repo": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "path": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "func_name": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "original_string": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "language": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "code": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "code_tokens": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "docstring": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "docstring_tokens": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "sha": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "url": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## php

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:code_x_glue_ct_code_to_text/php')
```

- **说明**：

```
CodeXGLUE code-to-text dataset, available at https://github.com/microsoft/CodeXGLUE/tree/main/Code-Text/code-to-text

The dataset we use comes from CodeSearchNet and we filter the dataset as the following:
- Remove examples that codes cannot be parsed into an abstract syntax tree.
- Remove examples that #tokens of documents is < 3 or >256
- Remove examples that documents contain special tokens (e.g. <img ...> or https:...)
- Remove examples that documents are not English.
```

- **许可**：无已知许可
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 14014
`'train'` | 241241
`'validation'` | 12982

- **特征**：

```json
{
    "id": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "repo": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "path": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "func_name": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "original_string": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "language": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "code": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "code_tokens": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "docstring": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "docstring_tokens": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "sha": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "url": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## python

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:code_x_glue_ct_code_to_text/python')
```

- **说明**：

```
CodeXGLUE code-to-text dataset, available at https://github.com/microsoft/CodeXGLUE/tree/main/Code-Text/code-to-text

The dataset we use comes from CodeSearchNet and we filter the dataset as the following:
- Remove examples that codes cannot be parsed into an abstract syntax tree.
- Remove examples that #tokens of documents is < 3 or >256
- Remove examples that documents contain special tokens (e.g. <img ...> or https:...)
- Remove examples that documents are not English.
```

- **许可**：无已知许可
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 14918
`'train'` | 251820
`'validation'` | 13914

- **特征**：

```json
{
    "id": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "repo": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "path": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "func_name": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "original_string": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "language": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "code": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "code_tokens": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "docstring": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "docstring_tokens": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "sha": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "url": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## ruby

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:code_x_glue_ct_code_to_text/ruby')
```

- **说明**：

```
CodeXGLUE code-to-text dataset, available at https://github.com/microsoft/CodeXGLUE/tree/main/Code-Text/code-to-text

The dataset we use comes from CodeSearchNet and we filter the dataset as the following:
- Remove examples that codes cannot be parsed into an abstract syntax tree.
- Remove examples that #tokens of documents is < 3 or >256
- Remove examples that documents contain special tokens (e.g. <img ...> or https:...)
- Remove examples that documents are not English.
```

- **许可**：无已知许可
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 1261
`'train'` | 24927
`'validation'` | 1400

- **特征**：

```json
{
    "id": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "repo": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "path": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "func_name": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "original_string": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "language": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "code": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "code_tokens": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "docstring": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "docstring_tokens": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "sha": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "url": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```
