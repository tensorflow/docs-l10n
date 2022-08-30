# e2e_nlg

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/e2e_nlg)
- [Huggingface](https://huggingface.co/datasets/e2e_nlg)

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:e2e_nlg')
```

- **说明**：

```
The E2E dataset is used for training end-to-end, data-driven natural language generation systems in the restaurant domain, which is ten times bigger than existing, frequently used datasets in this area.
The E2E dataset poses new challenges:
(1) its human reference texts show more lexical richness and syntactic variation, including discourse phenomena;
(2) generating from this set requires content selection. As such, learning from this dataset promises more natural, varied and less template-like system utterances.


E2E is released in the following paper where you can find more details and baseline results:
https://arxiv.org/abs/1706.09254
```

- **许可**：无已知许可
- **版本**：0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 4693
`'train'` | 42061
`'validation'` | 4672

- **特征**：

```json
{
    "meaning_representation": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "human_reference": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```
