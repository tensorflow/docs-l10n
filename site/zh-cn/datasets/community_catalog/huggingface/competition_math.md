# competition_math

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/competition_math)
- [Huggingface](https://huggingface.co/datasets/competition_math)

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:competition_math')
```

- **说明**：

```
The Mathematics Aptitude Test of Heuristics (MATH) dataset consists of problems
from mathematics competitions, including the AMC 10, AMC 12, AIME, and more.
Each problem in MATH has a full step-by-step solution, which can be used to teach
models to generate answer derivations and explanations.
```

- **许可**：https://github.com/hendrycks/math/blob/main/LICENSE
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 5000
`'train'` | 7500

- **特征**：

```json
{
    "problem": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "level": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "type": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "solution": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```
