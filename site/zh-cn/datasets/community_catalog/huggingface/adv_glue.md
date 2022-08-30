# adv_glue

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/adv_glue)
- [Huggingface](https://huggingface.co/datasets/adv_glue)

## adv_sst2

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:adv_glue/adv_sst2')
```

- **说明**：

```
Adversarial GLUE Benchmark (AdvGLUE) is a comprehensive robustness evaluation benchmark
that focuses on the adversarial robustness evaluation of language models. It covers five
natural language understanding tasks from the famous GLUE tasks and is an adversarial
version of GLUE benchmark.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'validation'` | 148

- **特征**：

```json
{
    "sentence": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    }
}
```

## adv_qqp

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:adv_glue/adv_qqp')
```

- **说明**：

```
Adversarial GLUE Benchmark (AdvGLUE) is a comprehensive robustness evaluation benchmark
that focuses on the adversarial robustness evaluation of language models. It covers five
natural language understanding tasks from the famous GLUE tasks and is an adversarial
version of GLUE benchmark.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'validation'` | 78

- **特征**：

```json
{
    "question1": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "question2": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    }
}
```

## adv_mnli

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:adv_glue/adv_mnli')
```

- **说明**：

```
Adversarial GLUE Benchmark (AdvGLUE) is a comprehensive robustness evaluation benchmark
that focuses on the adversarial robustness evaluation of language models. It covers five
natural language understanding tasks from the famous GLUE tasks and is an adversarial
version of GLUE benchmark.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'validation'` | 121

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
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    }
}
```

## adv_mnli_mismatched

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:adv_glue/adv_mnli_mismatched')
```

- **说明**：

```
Adversarial GLUE Benchmark (AdvGLUE) is a comprehensive robustness evaluation benchmark
that focuses on the adversarial robustness evaluation of language models. It covers five
natural language understanding tasks from the famous GLUE tasks and is an adversarial
version of GLUE benchmark.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'validation'` | 162

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
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    }
}
```

## adv_qnli

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:adv_glue/adv_qnli')
```

- **说明**：

```
Adversarial GLUE Benchmark (AdvGLUE) is a comprehensive robustness evaluation benchmark
that focuses on the adversarial robustness evaluation of language models. It covers five
natural language understanding tasks from the famous GLUE tasks and is an adversarial
version of GLUE benchmark.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'validation'` | 148

- **特征**：

```json
{
    "question": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "sentence": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    }
}
```

## adv_rte

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:adv_glue/adv_rte')
```

- **说明**：

```
Adversarial GLUE Benchmark (AdvGLUE) is a comprehensive robustness evaluation benchmark
that focuses on the adversarial robustness evaluation of language models. It covers five
natural language understanding tasks from the famous GLUE tasks and is an adversarial
version of GLUE benchmark.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'validation'` | 81

- **特征**：

```json
{
    "sentence1": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "sentence2": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    }
}
```
