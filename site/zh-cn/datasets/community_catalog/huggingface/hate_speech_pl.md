# hate_speech_pl

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/hate_speech_pl)
- [Huggingface](https://huggingface.co/datasets/hate_speech_pl)

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:hate_speech_pl')
```

- **说明**：

```
HateSpeech corpus in the current version contains over 2000 posts crawled from public Polish web. They represent various types and degrees of offensive language, expressed toward minorities (eg. ethnical, racial). The data were annotated manually.
```

- **许可**：CC BY-NC-SA
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 13887

- **特征**：

```json
{
    "id": {
        "dtype": "uint16",
        "id": null,
        "_type": "Value"
    },
    "text_id": {
        "dtype": "uint32",
        "id": null,
        "_type": "Value"
    },
    "annotator_id": {
        "dtype": "uint8",
        "id": null,
        "_type": "Value"
    },
    "minority_id": {
        "dtype": "uint8",
        "id": null,
        "_type": "Value"
    },
    "negative_emotions": {
        "dtype": "bool",
        "id": null,
        "_type": "Value"
    },
    "call_to_action": {
        "dtype": "bool",
        "id": null,
        "_type": "Value"
    },
    "source_of_knowledge": {
        "dtype": "uint8",
        "id": null,
        "_type": "Value"
    },
    "irony_sarcasm": {
        "dtype": "bool",
        "id": null,
        "_type": "Value"
    },
    "topic": {
        "dtype": "uint8",
        "id": null,
        "_type": "Value"
    },
    "text": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "rating": {
        "dtype": "uint8",
        "id": null,
        "_type": "Value"
    }
}
```
