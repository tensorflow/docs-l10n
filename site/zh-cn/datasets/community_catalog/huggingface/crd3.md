# crd3

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/crd3)
- [Huggingface](https://huggingface.co/datasets/crd3)

Use the following command to load this dataset in TFDS:

```python
ds = tfds.load('huggingface:crd3')
```

- **Description**:

```
Storytelling with Dialogue: A Critical Role Dungeons and Dragons Dataset.
Critical Role is an unscripted, live-streamed show where a fixed group of people play Dungeons and Dragons, an open-ended role-playing game.
The dataset is collected from 159 Critical Role episodes transcribed to text dialogues, consisting of 398,682 turns. It also includes corresponding
abstractive summaries collected from the Fandom wiki. The dataset is linguistically unique in that the narratives are generated entirely through player
collaboration and spoken interaction. For each dialogue, there are a large number of turns, multiple abstractive summaries with varying levels of detail,
and semantic ties to the previous dialogues.
```

- **许可**：无已知许可
- **Version**: 0.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 52796
`'train'` | 52796
`'validation'` | 52796

- **特征**：

```json
{
    "chunk": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "chunk_id": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "turn_start": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "turn_end": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "alignment_score": {
        "dtype": "float32",
        "id": null,
        "_type": "Value"
    },
    "turns": {
        "feature": {
            "names": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "utterances": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "number": {
                "dtype": "int32",
                "id": null,
                "_type": "Value"
            }
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    }
}
```
