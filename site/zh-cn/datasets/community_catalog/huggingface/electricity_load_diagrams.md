# electricity_load_diagrams

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/electricity_load_diagrams)
- [Huggingface](https://huggingface.co/datasets/electricity_load_diagrams)

## uci

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:electricity_load_diagrams/uci')
```

- **说明**：

```
This new dataset contains hourly kW electricity consumption time series of 370 Portuguese clients from 2011 to 2014.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 2590
`'train'` | 370
`'validation'` | 370

- **特征**：

```json
{
    "start": {
        "dtype": "timestamp[s]",
        "id": null,
        "_type": "Value"
    },
    "target": {
        "feature": {
            "dtype": "float32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "feat_static_cat": {
        "feature": {
            "dtype": "uint64",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "item_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## lstnet

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:electricity_load_diagrams/lstnet')
```

- **说明**：

```
This new dataset contains hourly kW electricity consumption time series of 370 Portuguese clients from 2011 to 2014.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 2240
`'train'` | 320
`'validation'` | 320

- **特征**：

```json
{
    "start": {
        "dtype": "timestamp[s]",
        "id": null,
        "_type": "Value"
    },
    "target": {
        "feature": {
            "dtype": "float32",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "feat_static_cat": {
        "feature": {
            "dtype": "uint64",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "item_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```
