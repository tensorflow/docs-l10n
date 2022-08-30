# craigslist_bargains

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/craigslist_bargains)
- [Huggingface](https://huggingface.co/datasets/craigslist_bargains)

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:craigslist_bargains')
```

- **说明**：

```
We study negotiation dialogues where two agents, a buyer and a seller,
negotiate over the price of an time for sale. We collected a dataset of more
than 6K negotiation dialogues over multiple categories of products scraped from Craigslist.
Our goal is to develop an agent that negotiates with humans through such conversations.
The challenge is to handle both the negotiation strategy and the rich language for bargaining.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 838
`'train'` | 5247
`'validation'` | 597

- **特征**：

```json
{
    "agent_info": {
        "feature": {
            "Bottomline": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "Role": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "Target": {
                "dtype": "float32",
                "id": null,
                "_type": "Value"
            }
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "agent_turn": {
        "feature": {
            "dtype": "int32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "dialogue_acts": {
        "feature": {
            "intent": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "price": {
                "dtype": "float32",
                "id": null,
                "_type": "Value"
            }
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "utterance": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "items": {
        "feature": {
            "Category": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "Images": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "Price": {
                "dtype": "float32",
                "id": null,
                "_type": "Value"
            },
            "Description": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "Title": {
                "dtype": "string",
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
