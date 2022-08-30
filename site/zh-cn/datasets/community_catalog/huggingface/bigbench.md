# bigbench

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/bigbench)
- [Huggingface](https://huggingface.co/datasets/bigbench)

## abstract_narrative_understanding

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/abstract_narrative_understanding')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 3000
`'train'` | 2400
`'validation'` | 600

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## anachronisms

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/anachronisms')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 230
`'train'` | 184
`'validation'` | 46

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## analogical_similarity

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/analogical_similarity')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 323
`'train'` | 259
`'validation'` | 64

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## analytic_entailment

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/analytic_entailment')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 70
`'train'` | 54
`'validation'` | 16

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## arithmetic

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/arithmetic')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 15023
`'train'` | 12019
`'validation'` | 3004

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## ascii_word_recognition

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/ascii_word_recognition')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 5000
`'train'` | 4000
`'validation'` | 1000

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## authorship_verification

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/authorship_verification')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 880
`'train'` | 704
`'validation'` | 176

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## auto_categorization

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/auto_categorization')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 328
`'train'` | 263
`'validation'` | 65

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## auto_debugging

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/auto_debugging')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 34
`'train'` | 18
`'validation'` | 16

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## bbq_lite_json

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/bbq_lite_json')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 16076
`'train'` | 12866
`'validation'` | 3210

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## bridging_anaphora_resolution_barqa

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/bridging_anaphora_resolution_barqa')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 648
`'train'` | 519
`'validation'` | 129

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## causal_judgment

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/causal_judgment')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 190
`'train'` | 152
`'validation'` | 38

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## cause_and_effect

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/cause_and_effect')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 153
`'train'` | 123
`'validation'` | 30

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## checkmate_in_one

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/checkmate_in_one')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 3498
`'train'` | 2799
`'validation'` | 699

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## chess_state_tracking

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/chess_state_tracking')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 6000
`'train'` | 4800
`'validation'` | 1200

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## chinese_remainder_theorem

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/chinese_remainder_theorem')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 500
`'train'` | 400
`'validation'` | 100

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## cifar10_classification

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/cifar10_classification')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 20000
`'train'` | 16000
`'validation'` | 4000

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## code_line_description

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/code_line_description')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 60
`'train'` | 44
`'validation'` | 16

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## codenames

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/codenames')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 85
`'train'` | 68
`'validation'` | 17

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## color

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/color')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 4000
`'train'` | 3200
`'validation'` | 800

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## common_morpheme

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/common_morpheme')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 50
`'train'` | 34
`'validation'` | 16

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## conceptual_combinations

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/conceptual_combinations')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 103
`'train'` | 84
`'validation'` | 19

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## conlang_translation

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/conlang_translation')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 164
`'train'` | 132
`'validation'` | 32

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## contextual_parametric_knowledge_conflicts

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/contextual_parametric_knowledge_conflicts')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 17528
`'train'` | 14023
`'validation'` | 3505

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## crash_blossom

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/crash_blossom')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 38
`'train'` | 22
`'validation'` | 16

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## crass_ai

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/crass_ai')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 44
`'train'` | 28
`'validation'` | 16

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## cryobiology_spanish

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/cryobiology_spanish')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 146
`'train'` | 117
`'validation'` | 29

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## cryptonite

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/cryptonite')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 26157
`'train'` | 20926
`'validation'` | 5231

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## cs_algorithms

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/cs_algorithms')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 1320
`'train'` | 1056
`'validation'` | 264

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## dark_humor_detection

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/dark_humor_detection')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 80
`'train'` | 64
`'validation'` | 16

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## date_understanding

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/date_understanding')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 369
`'train'` | 296
`'validation'` | 73

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## disambiguation_qa

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/disambiguation_qa')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 258
`'train'` | 207
`'validation'` | 51

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## discourse_marker_prediction

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/discourse_marker_prediction')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 857
`'train'` | 686
`'validation'` | 171

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## disfl_qa

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/disfl_qa')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 8000
`'train'` | 6400
`'validation'` | 1600

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## dyck_languages

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/dyck_languages')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 1000
`'train'` | 800
`'validation'` | 200

- **Features**:

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## elementary_math_qa

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/elementary_math_qa')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 38160
`'train'` | 30531
`'validation'` | 7629

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## emoji_movie

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/emoji_movie')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 100
`'train'` | 80
`'validation'` | 20

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## emojis_emotion_prediction

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/emojis_emotion_prediction')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 131
`'train'` | 105
`'validation'` | 26

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## empirical_judgments

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/empirical_judgments')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 99
`'train'` | 80
`'validation'` | 19

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## english_proverbs

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/english_proverbs')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 34
`'train'` | 18
`'validation'` | 16

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## english_russian_proverbs

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/english_russian_proverbs')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 80
`'train'` | 64
`'validation'` | 16

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## entailed_polarity

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/entailed_polarity')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 148
`'train'` | 119
`'validation'` | 29

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## entailed_polarity_hindi

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/entailed_polarity_hindi')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 138
`'train'` | 111
`'validation'` | 27

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## epistemic_reasoning

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/epistemic_reasoning')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 2000
`'train'` | 1600
`'validation'` | 400

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## evaluating_information_essentiality

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/evaluating_information_essentiality')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 68
`'train'` | 52
`'validation'` | 16

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## fact_checker

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/fact_checker')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 7154
`'train'` | 5724
`'validation'` | 1430

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## fantasy_reasoning

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/fantasy_reasoning')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 201
`'train'` | 161
`'validation'` | 40

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## few_shot_nlg

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/few_shot_nlg')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 153
`'train'` | 123
`'validation'` | 30

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## figure_of_speech_detection

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/figure_of_speech_detection')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 59
`'train'` | 43
`'validation'` | 16

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## formal_fallacies_syllogisms_negation

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/formal_fallacies_syllogisms_negation')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 14200
`'train'` | 11360
`'validation'` | 2840

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## gem

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/gem')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 14802
`'train'` | 11845
`'validation'` | 2957

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## gender_inclusive_sentences_german

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/gender_inclusive_sentences_german')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 200
`'train'` | 160
`'validation'` | 40

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## general_knowledge

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/general_knowledge')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 70
`'train'` | 54
`'validation'` | 16

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## geometric_shapes

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/geometric_shapes')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 359
`'train'` | 288
`'validation'` | 71

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## goal_step_wikihow

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/goal_step_wikihow')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 7053
`'train'` | 5643
`'validation'` | 1410

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## gre_reading_comprehension

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/gre_reading_comprehension')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 31
`'train'` | 15
`'validation'` | 16

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## hhh_alignment

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/hhh_alignment')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 221
`'train'` | 179
`'validation'` | 42

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## hindi_question_answering

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/hindi_question_answering')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 6610
`'train'` | 5288
`'validation'` | 1322

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## hindu_knowledge

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/hindu_knowledge')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 175
`'train'` | 140
`'validation'` | 35

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## hinglish_toxicity

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/hinglish_toxicity')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 200
`'train'` | 160
`'validation'` | 40

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## human_organs_senses

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/human_organs_senses')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 42
`'train'` | 26
`'validation'` | 16

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## hyperbaton

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/hyperbaton')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 50000
`'train'` | 40000
`'validation'` | 10000

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## identify_math_theorems

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/identify_math_theorems')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 53
`'train'` | 37
`'validation'` | 16

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## identify_odd_metaphor

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/identify_odd_metaphor')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 47
`'train'` | 31
`'validation'` | 16

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## implicatures

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/implicatures')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 492
`'train'` | 394
`'validation'` | 98

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## implicit_relations

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/implicit_relations')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 85
`'train'` | 68
`'validation'` | 17

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## intent_recognition

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/intent_recognition')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 693
`'train'` | 555
`'validation'` | 138

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## international_phonetic_alphabet_nli

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/international_phonetic_alphabet_nli')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 126
`'train'` | 101
`'validation'` | 25

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## international_phonetic_alphabet_transliterate

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/international_phonetic_alphabet_transliterate')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 1003
`'train'` | 803
`'validation'` | 200

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## intersect_geometry

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/intersect_geometry')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 249999
`'train'` | 200000
`'validation'` | 49999

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## irony_identification

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/irony_identification')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 99
`'train'` | 80
`'validation'` | 19

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## kanji_ascii

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/kanji_ascii')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 1092
`'train'` | 875
`'validation'` | 217

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## kannada

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/kannada')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 316
`'train'` | 253
`'validation'` | 63

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## key_value_maps

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/key_value_maps')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 101
`'train'` | 80
`'validation'` | 21

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## known_unknowns

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/known_unknowns')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 46
`'train'` | 30
`'validation'` | 16

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## language_games

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/language_games')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 2128
`'train'` | 1704
`'validation'` | 424

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## language_identification

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/language_identification')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 10000
`'train'` | 8000
`'validation'` | 2000

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## linguistic_mappings

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/linguistic_mappings')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 15527
`'train'` | 12426
`'validation'` | 3101

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## linguistics_puzzles

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/linguistics_puzzles')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 2000
`'train'` | 1600
`'validation'` | 400

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## list_functions

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/list_functions')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 10750
`'train'` | 8700
`'validation'` | 2050

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## logic_grid_puzzle

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/logic_grid_puzzle')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 1000
`'train'` | 800
`'validation'` | 200

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## logical_args

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/logical_args')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 32
`'train'` | 16
`'validation'` | 16

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## logical_deduction

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/logical_deduction')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 1500
`'train'` | 1200
`'validation'` | 300

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## logical_fallacy_detection

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/logical_fallacy_detection')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 2800
`'train'` | 2240
`'validation'` | 560

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## logical_sequence

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/logical_sequence')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 39
`'train'` | 23
`'validation'` | 16

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## mathematical_induction

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/mathematical_induction')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 69
`'train'` | 53
`'validation'` | 16

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## matrixshapes

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/matrixshapes')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 4462
`'train'` | 3570
`'validation'` | 892

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## metaphor_boolean

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/metaphor_boolean')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 680
`'train'` | 544
`'validation'` | 136

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## metaphor_understanding

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/metaphor_understanding')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 234
`'train'` | 188
`'validation'` | 46

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## minute_mysteries_qa

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/minute_mysteries_qa')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 477
`'train'` | 383
`'validation'` | 94

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## misconceptions

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/misconceptions')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 219
`'train'` | 176
`'validation'` | 43

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## misconceptions_russian

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/misconceptions_russian')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 49
`'train'` | 33
`'validation'` | 16

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## mnist_ascii

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/mnist_ascii')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 69984
`'train'` | 55988
`'validation'` | 13996

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## modified_arithmetic

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/modified_arithmetic')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 6000
`'train'` | 4800
`'validation'` | 1200

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## moral_permissibility

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/moral_permissibility')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 342
`'train'` | 274
`'validation'` | 68

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## movie_dialog_same_or_different

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/movie_dialog_same_or_different')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 50000
`'train'` | 40000
`'validation'` | 10000

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## movie_recommendation

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/movie_recommendation')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 500
`'train'` | 400
`'validation'` | 100

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## mult_data_wrangling

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/mult_data_wrangling')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 7854
`'train'` | 6380
`'validation'` | 1474

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## multiemo

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/multiemo')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 1437281
`'train'` | 1149873
`'validation'` | 287408

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## natural_instructions

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/natural_instructions')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 193250
`'train'` | 154615
`'validation'` | 38635

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## navigate

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/navigate')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 1000
`'train'` | 800
`'validation'` | 200

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## nonsense_words_grammar

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/nonsense_words_grammar')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 50
`'train'` | 34
`'validation'` | 16

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## novel_concepts

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/novel_concepts')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 32
`'train'` | 16
`'validation'` | 16

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## object_counting

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/object_counting')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 1000
`'train'` | 800
`'validation'` | 200

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## odd_one_out

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/odd_one_out')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 86
`'train'` | 69
`'validation'` | 17

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## operators

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/operators')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 210
`'train'` | 168
`'validation'` | 42

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## paragraph_segmentation

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/paragraph_segmentation')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 9000
`'train'` | 7200
`'validation'` | 1800

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## parsinlu_qa

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/parsinlu_qa')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 1050
`'train'` | 840
`'validation'` | 210

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## parsinlu_reading_comprehension

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/parsinlu_reading_comprehension')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 518
`'train'` | 415
`'validation'` | 103

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## penguins_in_a_table

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/penguins_in_a_table')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 149
`'train'` | 120
`'validation'` | 29

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## periodic_elements

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/periodic_elements')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 654
`'train'` | 524
`'validation'` | 130

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## persian_idioms

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/persian_idioms')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 66
`'train'` | 50
`'validation'` | 16

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## phrase_relatedness

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/phrase_relatedness')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 100
`'train'` | 80
`'validation'` | 20

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## physical_intuition

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/physical_intuition')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 81
`'train'` | 65
`'validation'` | 16

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## physics

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/physics')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 229
`'train'` | 184
`'validation'` | 45

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## physics_questions

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/physics_questions')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 54
`'train'` | 38
`'validation'` | 16

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## play_dialog_same_or_different

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/play_dialog_same_or_different')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 3264
`'train'` | 2612
`'validation'` | 652

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## polish_sequence_labeling

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/polish_sequence_labeling')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 12812
`'train'` | 10250
`'validation'` | 2562

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## presuppositions_as_nli

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/presuppositions_as_nli')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 735
`'train'` | 588
`'validation'` | 147

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## qa_wikidata

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/qa_wikidata')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 20321
`'train'` | 16257
`'validation'` | 4064

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## question_selection

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/question_selection')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 1582
`'train'` | 1266
`'validation'` | 316

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## real_or_fake_text

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/real_or_fake_text')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 15088
`'train'` | 12072
`'validation'` | 3016

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## reasoning_about_colored_objects

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/reasoning_about_colored_objects')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 2000
`'train'` | 1600
`'validation'` | 400

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## repeat_copy_logic

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/repeat_copy_logic')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 32
`'train'` | 16
`'validation'` | 16

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## rephrase

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/rephrase')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 78
`'train'` | 62
`'validation'` | 16

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## riddle_sense

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/riddle_sense')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 49
`'train'` | 33
`'validation'` | 16

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## ruin_names

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/ruin_names')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 448
`'train'` | 359
`'validation'` | 89

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## salient_translation_error_detection

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/salient_translation_error_detection')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 998
`'train'` | 799
`'validation'` | 199

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## scientific_press_release

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/scientific_press_release')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 50
`'train'` | 34
`'validation'` | 16

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## semantic_parsing_in_context_sparc

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/semantic_parsing_in_context_sparc')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 1155
`'train'` | 924
`'validation'` | 231

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## semantic_parsing_spider

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/semantic_parsing_spider')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 1034
`'train'` | 828
`'validation'` | 206

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## sentence_ambiguity

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/sentence_ambiguity')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 60
`'train'` | 44
`'validation'` | 16

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## similarities_abstraction

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/similarities_abstraction')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 76
`'train'` | 60
`'validation'` | 16

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## simp_turing_concept

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/simp_turing_concept')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 6390
`'train'` | 5112
`'validation'` | 1278

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## simple_arithmetic_json

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/simple_arithmetic_json')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 30
`'train'` | 14
`'validation'` | 16

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## simple_arithmetic_json_multiple_choice

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/simple_arithmetic_json_multiple_choice')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 8
`'train'` | 0
`'validation'` | 0

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## simple_arithmetic_json_subtasks

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/simple_arithmetic_json_subtasks')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 30
`'train'` | 15
`'validation'` | 15

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## simple_arithmetic_multiple_targets_json

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/simple_arithmetic_multiple_targets_json')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 10
`'train'` | 0
`'validation'` | 0

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## simple_ethical_questions

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/simple_ethical_questions')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 115
`'train'` | 92
`'validation'` | 23

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## simple_text_editing

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/simple_text_editing')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 47
`'train'` | 31
`'validation'` | 16

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## snarks

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/snarks')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 181
`'train'` | 145
`'validation'` | 36

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## social_iqa

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/social_iqa')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 1935
`'train'` | 1548
`'validation'` | 387

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## social_support

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/social_support')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 897
`'train'` | 718
`'validation'` | 179

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## sports_understanding

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/sports_understanding')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 986
`'train'` | 789
`'validation'` | 197

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## strange_stories

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/strange_stories')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 174
`'train'` | 140
`'validation'` | 34

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## strategyqa

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/strategyqa')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 2289
`'train'` | 1832
`'validation'` | 457

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## sufficient_information

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/sufficient_information')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 39
`'train'` | 23
`'validation'` | 16

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## suicide_risk

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/suicide_risk')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 40
`'train'` | 24
`'validation'` | 16

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## swahili_english_proverbs

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/swahili_english_proverbs')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 153
`'train'` | 123
`'validation'` | 30

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## swedish_to_german_proverbs

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/swedish_to_german_proverbs')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 72
`'train'` | 56
`'validation'` | 16

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## symbol_interpretation

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/symbol_interpretation')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 990
`'train'` | 795
`'validation'` | 195

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## temporal_sequences

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/temporal_sequences')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 1000
`'train'` | 800
`'validation'` | 200

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## tense

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/tense')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 286
`'train'` | 229
`'validation'` | 57

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## timedial

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/timedial')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 2550
`'train'` | 2040
`'validation'` | 510

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## topical_chat

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/topical_chat')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 22295
`'train'` | 17836
`'validation'` | 4459

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## tracking_shuffled_objects

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/tracking_shuffled_objects')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 3750
`'train'` | 3000
`'validation'` | 750

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## understanding_fables

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/understanding_fables')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 189
`'train'` | 152
`'validation'` | 37

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## undo_permutation

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/undo_permutation')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 300
`'train'` | 240
`'validation'` | 60

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## unit_conversion

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/unit_conversion')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 23936
`'train'` | 19151
`'validation'` | 4785

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## unit_interpretation

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/unit_interpretation')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 100
`'train'` | 80
`'validation'` | 20

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## unnatural_in_context_learning

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/unnatural_in_context_learning')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 73420
`'train'` | 58736
`'validation'` | 14684

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## vitaminc_fact_verification

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/vitaminc_fact_verification')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 54668
`'train'` | 43735
`'validation'` | 10933

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## what_is_the_tao

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/what_is_the_tao')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 36
`'train'` | 20
`'validation'` | 16

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## which_wiki_edit

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/which_wiki_edit')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 571
`'train'` | 457
`'validation'` | 114

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## winowhy

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/winowhy')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 2862
`'train'` | 2290
`'validation'` | 572

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## word_sorting

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/word_sorting')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 1900
`'train'` | 1520
`'validation'` | 380

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```

## word_unscrambling

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bigbench/word_unscrambling')
```

- **说明**：

```
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
```

- **许可**：Apache License 2.0
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'default'` | 8917
`'train'` | 7134
`'validation'` | 1783

- **特征**：

```json
{
    "idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "inputs": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_targets": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "multiple_choice_scores": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```
