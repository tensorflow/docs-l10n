# biomrc

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/biomrc)
- [Huggingface](https://huggingface.co/datasets/biomrc)

## plain_text

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:biomrc/plain_text')
```

- **说明**：

```
We introduce BIOMRC, a large-scale cloze-style biomedical MRC dataset. Care was taken to reduce noise, compared to the previous BIOREAD dataset of Pappas et al. (2018). Experiments show that simple heuristics do not perform well on the new dataset and that two neural MRC models that had been tested on BIOREAD perform much better on BIOMRC, indicating that the new dataset is indeed less noisy or at least that its task is more feasible. Non-expert human performance is also higher on the new dataset compared to BIOREAD, and biomedical experts perform even better. We also introduce a new BERT-based MRC model, the best version of which substantially outperforms all other methods tested, reaching or surpassing the accuracy of biomedical experts in some experiments. We make the new dataset available in three different sizes, also releasing our code, and providing a leaderboard.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 62707
`'train'` | 700000
`'validation'` | 50000

- **特征**：

```json
{
    "abstract": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "entities_list": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "answer": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## biomrc_large_A

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:biomrc/biomrc_large_A')
```

- **说明**：

```
We introduce BIOMRC, a large-scale cloze-style biomedical MRC dataset. Care was taken to reduce noise, compared to the previous BIOREAD dataset of Pappas et al. (2018). Experiments show that simple heuristics do not perform well on the new dataset and that two neural MRC models that had been tested on BIOREAD perform much better on BIOMRC, indicating that the new dataset is indeed less noisy or at least that its task is more feasible. Non-expert human performance is also higher on the new dataset compared to BIOREAD, and biomedical experts perform even better. We also introduce a new BERT-based MRC model, the best version of which substantially outperforms all other methods tested, reaching or surpassing the accuracy of biomedical experts in some experiments. We make the new dataset available in three different sizes, also releasing our code, and providing a leaderboard.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 62707
`'train'` | 700000
`'validation'` | 50000

- **特征**：

```json
{
    "abstract": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "entities_list": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "answer": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## biomrc_large_B

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:biomrc/biomrc_large_B')
```

- **说明**：

```
We introduce BIOMRC, a large-scale cloze-style biomedical MRC dataset. Care was taken to reduce noise, compared to the previous BIOREAD dataset of Pappas et al. (2018). Experiments show that simple heuristics do not perform well on the new dataset and that two neural MRC models that had been tested on BIOREAD perform much better on BIOMRC, indicating that the new dataset is indeed less noisy or at least that its task is more feasible. Non-expert human performance is also higher on the new dataset compared to BIOREAD, and biomedical experts perform even better. We also introduce a new BERT-based MRC model, the best version of which substantially outperforms all other methods tested, reaching or surpassing the accuracy of biomedical experts in some experiments. We make the new dataset available in three different sizes, also releasing our code, and providing a leaderboard.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 62707
`'train'` | 700000
`'validation'` | 50000

- **特征**：

```json
{
    "abstract": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "entities_list": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "answer": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## biomrc_small_A

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:biomrc/biomrc_small_A')
```

- **说明**：

```
We introduce BIOMRC, a large-scale cloze-style biomedical MRC dataset. Care was taken to reduce noise, compared to the previous BIOREAD dataset of Pappas et al. (2018). Experiments show that simple heuristics do not perform well on the new dataset and that two neural MRC models that had been tested on BIOREAD perform much better on BIOMRC, indicating that the new dataset is indeed less noisy or at least that its task is more feasible. Non-expert human performance is also higher on the new dataset compared to BIOREAD, and biomedical experts perform even better. We also introduce a new BERT-based MRC model, the best version of which substantially outperforms all other methods tested, reaching or surpassing the accuracy of biomedical experts in some experiments. We make the new dataset available in three different sizes, also releasing our code, and providing a leaderboard.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 6250
`'train'` | 87500
`'validation'` | 6250

- **特征**：

```json
{
    "abstract": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "entities_list": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "answer": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## biomrc_small_B

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:biomrc/biomrc_small_B')
```

- **说明**：

```
We introduce BIOMRC, a large-scale cloze-style biomedical MRC dataset. Care was taken to reduce noise, compared to the previous BIOREAD dataset of Pappas et al. (2018). Experiments show that simple heuristics do not perform well on the new dataset and that two neural MRC models that had been tested on BIOREAD perform much better on BIOMRC, indicating that the new dataset is indeed less noisy or at least that its task is more feasible. Non-expert human performance is also higher on the new dataset compared to BIOREAD, and biomedical experts perform even better. We also introduce a new BERT-based MRC model, the best version of which substantially outperforms all other methods tested, reaching or surpassing the accuracy of biomedical experts in some experiments. We make the new dataset available in three different sizes, also releasing our code, and providing a leaderboard.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 6250
`'train'` | 87500
`'validation'` | 6250

- **特征**：

```json
{
    "abstract": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "entities_list": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "answer": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## biomrc_tiny_A

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:biomrc/biomrc_tiny_A')
```

- **说明**：

```
We introduce BIOMRC, a large-scale cloze-style biomedical MRC dataset. Care was taken to reduce noise, compared to the previous BIOREAD dataset of Pappas et al. (2018). Experiments show that simple heuristics do not perform well on the new dataset and that two neural MRC models that had been tested on BIOREAD perform much better on BIOMRC, indicating that the new dataset is indeed less noisy or at least that its task is more feasible. Non-expert human performance is also higher on the new dataset compared to BIOREAD, and biomedical experts perform even better. We also introduce a new BERT-based MRC model, the best version of which substantially outperforms all other methods tested, reaching or surpassing the accuracy of biomedical experts in some experiments. We make the new dataset available in three different sizes, also releasing our code, and providing a leaderboard.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 30

- **特征**：

```json
{
    "abstract": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "entities_list": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "answer": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## biomrc_tiny_B

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:biomrc/biomrc_tiny_B')
```

- **说明**：

```
We introduce BIOMRC, a large-scale cloze-style biomedical MRC dataset. Care was taken to reduce noise, compared to the previous BIOREAD dataset of Pappas et al. (2018). Experiments show that simple heuristics do not perform well on the new dataset and that two neural MRC models that had been tested on BIOREAD perform much better on BIOMRC, indicating that the new dataset is indeed less noisy or at least that its task is more feasible. Non-expert human performance is also higher on the new dataset compared to BIOREAD, and biomedical experts perform even better. We also introduce a new BERT-based MRC model, the best version of which substantially outperforms all other methods tested, reaching or surpassing the accuracy of biomedical experts in some experiments. We make the new dataset available in three different sizes, also releasing our code, and providing a leaderboard.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 30

- **特征**：

```json
{
    "abstract": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "entities_list": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "answer": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```
