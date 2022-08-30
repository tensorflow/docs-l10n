# code_x_glue_cc_cloze_testing_maxmin

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/code_x_glue_cc_cloze_testing_maxmin)
- [Huggingface](https://huggingface.co/datasets/code_x_glue_cc_cloze_testing_maxmin)

## go

Use the following command to load this dataset in TFDS:

```python
ds = tfds.load('huggingface:code_x_glue_cc_cloze_testing_maxmin/go')
```

- **Description**:

```
CodeXGLUE ClozeTesting-maxmin dataset, available at https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/ClozeTesting-maxmin

Cloze tests are widely adopted in Natural Languages Processing to evaluate the performance of the trained language models. The task is aimed to predict the answers for the blank with the context of the blank, which can be formulated as a multi-choice classification problem.
Here we present the two cloze testing datasets in code domain with six different programming languages: ClozeTest-maxmin and ClozeTest-all. Each instance in the dataset contains a masked code function, its docstring and the target word.
The only difference between ClozeTest-maxmin and ClozeTest-all is their selected words sets, where ClozeTest-maxmin only contains two words while ClozeTest-all contains 930 words.
```

- **许可**：无已知许可
- **Version**: 0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 152

- **特征**：

```json
{
    "id": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "idx": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "nl_tokens": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "pl_tokens": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## java

Use the following command to load this dataset in TFDS:

```python
ds = tfds.load('huggingface:code_x_glue_cc_cloze_testing_maxmin/java')
```

- **Description**:

```
CodeXGLUE ClozeTesting-maxmin dataset, available at https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/ClozeTesting-maxmin

Cloze tests are widely adopted in Natural Languages Processing to evaluate the performance of the trained language models. The task is aimed to predict the answers for the blank with the context of the blank, which can be formulated as a multi-choice classification problem.
Here we present the two cloze testing datasets in code domain with six different programming languages: ClozeTest-maxmin and ClozeTest-all. Each instance in the dataset contains a masked code function, its docstring and the target word.
The only difference between ClozeTest-maxmin and ClozeTest-all is their selected words sets, where ClozeTest-maxmin only contains two words while ClozeTest-all contains 930 words.
```

- **许可**：无已知许可
- **Version**: 0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 482

- **特征**：

```json
{
    "id": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "idx": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "nl_tokens": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "pl_tokens": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## javascript

Use the following command to load this dataset in TFDS:

```python
ds = tfds.load('huggingface:code_x_glue_cc_cloze_testing_maxmin/javascript')
```

- **Description**:

```
CodeXGLUE ClozeTesting-maxmin dataset, available at https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/ClozeTesting-maxmin

Cloze tests are widely adopted in Natural Languages Processing to evaluate the performance of the trained language models. The task is aimed to predict the answers for the blank with the context of the blank, which can be formulated as a multi-choice classification problem.
Here we present the two cloze testing datasets in code domain with six different programming languages: ClozeTest-maxmin and ClozeTest-all. Each instance in the dataset contains a masked code function, its docstring and the target word.
The only difference between ClozeTest-maxmin and ClozeTest-all is their selected words sets, where ClozeTest-maxmin only contains two words while ClozeTest-all contains 930 words.
```

- **许可**：无已知许可
- **Version**: 0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 272

- **特征**：

```json
{
    "id": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "idx": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "nl_tokens": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "pl_tokens": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## php

Use the following command to load this dataset in TFDS:

```python
ds = tfds.load('huggingface:code_x_glue_cc_cloze_testing_maxmin/php')
```

- **Description**:

```
CodeXGLUE ClozeTesting-maxmin dataset, available at https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/ClozeTesting-maxmin

Cloze tests are widely adopted in Natural Languages Processing to evaluate the performance of the trained language models. The task is aimed to predict the answers for the blank with the context of the blank, which can be formulated as a multi-choice classification problem.
Here we present the two cloze testing datasets in code domain with six different programming languages: ClozeTest-maxmin and ClozeTest-all. Each instance in the dataset contains a masked code function, its docstring and the target word.
The only difference between ClozeTest-maxmin and ClozeTest-all is their selected words sets, where ClozeTest-maxmin only contains two words while ClozeTest-all contains 930 words.
```

- **许可**：无已知许可
- **Version**: 0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 407

- **特征**：

```json
{
    "id": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "idx": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "nl_tokens": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "pl_tokens": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## python

Use the following command to load this dataset in TFDS:

```python
ds = tfds.load('huggingface:code_x_glue_cc_cloze_testing_maxmin/python')
```

- **Description**:

```
CodeXGLUE ClozeTesting-maxmin dataset, available at https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/ClozeTesting-maxmin

Cloze tests are widely adopted in Natural Languages Processing to evaluate the performance of the trained language models. The task is aimed to predict the answers for the blank with the context of the blank, which can be formulated as a multi-choice classification problem.
Here we present the two cloze testing datasets in code domain with six different programming languages: ClozeTest-maxmin and ClozeTest-all. Each instance in the dataset contains a masked code function, its docstring and the target word.
The only difference between ClozeTest-maxmin and ClozeTest-all is their selected words sets, where ClozeTest-maxmin only contains two words while ClozeTest-all contains 930 words.
```

- **许可**：无已知许可
- **Version**: 0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 1264

- **特征**：

```json
{
    "id": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "idx": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "nl_tokens": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "pl_tokens": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## ruby

Use the following command to load this dataset in TFDS:

```python
ds = tfds.load('huggingface:code_x_glue_cc_cloze_testing_maxmin/ruby')
```

- **Description**:

```
CodeXGLUE ClozeTesting-maxmin dataset, available at https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/ClozeTesting-maxmin

Cloze tests are widely adopted in Natural Languages Processing to evaluate the performance of the trained language models. The task is aimed to predict the answers for the blank with the context of the blank, which can be formulated as a multi-choice classification problem.
Here we present the two cloze testing datasets in code domain with six different programming languages: ClozeTest-maxmin and ClozeTest-all. Each instance in the dataset contains a masked code function, its docstring and the target word.
The only difference between ClozeTest-maxmin and ClozeTest-all is their selected words sets, where ClozeTest-maxmin only contains two words while ClozeTest-all contains 930 words.
```

- **许可**：无已知许可
- **Version**: 0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 38

- **特征**：

```json
{
    "id": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "idx": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "nl_tokens": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "pl_tokens": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```
