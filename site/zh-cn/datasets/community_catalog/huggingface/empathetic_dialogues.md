# empathetic_dialogues

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/empathetic_dialogues)
- [Huggingface](https://huggingface.co/datasets/empathetic_dialogues)

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:empathetic_dialogues')
```

- **说明**：

```
PyTorch original implementation of Towards Empathetic Open-domain Conversation Models: a New Benchmark and Dataset
```

- **许可**：无已知许可
- **版本**：0.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 10943
`'train'` | 76673
`'validation'` | 12030

- **特征**：

```json
{
    "conv_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "utterance_idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "context": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "prompt": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "speaker_idx": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "utterance": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "selfeval": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "tags": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```
