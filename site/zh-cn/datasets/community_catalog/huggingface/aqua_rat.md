# aqua_rat

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/aqua_rat)
- [Huggingface](https://huggingface.co/datasets/aqua_rat)

## raw

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:aqua_rat/raw')
```

- **说明**：

```
A large-scale dataset consisting of approximately 100,000 algebraic word problems.
The solution to each question is explained step-by-step using natural language.
This data is used to train a program generation model that learns to generate the explanation,
while generating the program that solves the question.
```

- **许可**：Copyright 2017 Google Inc.

根据 Apache 许可 2.0（“许可”）获得许可；除非遵循许可要求，否则您不得使用此文件。您可在以下网址获得许可的副本：

```
http://www.apache.org/licenses/LICENSE-2.0
```

除非适用法律要求或以书面形式同意，否则在本许可下分发的软件将在“按原样”的基础上分发，不存在任何明示或暗示的任何类型的保证或条件。有关在本许可下管理权限和限制的特定语言，请参阅本许可。

- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 254
`'train'` | 97467
`'validation'` | 254

- **特征**：

```json
{
    "question": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "options": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "rationale": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "correct": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## tokenized

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:aqua_rat/tokenized')
```

- **说明**：

```
A large-scale dataset consisting of approximately 100,000 algebraic word problems.
The solution to each question is explained step-by-step using natural language.
This data is used to train a program generation model that learns to generate the explanation,
while generating the program that solves the question.
```

- **许可**：Copyright 2017 Google Inc.

根据 Apache 许可 2.0（“许可”）获得许可；除非遵循许可要求，否则您不得使用此文件。您可在以下网址获得许可的副本：

```
http://www.apache.org/licenses/LICENSE-2.0
```

除非适用法律要求或以书面形式同意，否则在本许可下分发的软件将在“按原样”的基础上分发，不存在任何明示或暗示的任何类型的保证或条件。有关在本许可下管理权限和限制的特定语言，请参阅本许可。

- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 254
`'train'` | 97467
`'validation'` | 254

- **特征**：

```json
{
    "question": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "options": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "rationale": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "correct": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```
