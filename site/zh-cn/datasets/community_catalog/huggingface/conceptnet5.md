# conceptnet5

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/conceptnet5)
- [Huggingface](https://huggingface.co/datasets/conceptnet5)

## conceptnet5

Use the following command to load this dataset in TFDS:

```python
ds = tfds.load('huggingface:conceptnet5/conceptnet5')
```

- **Description**:

```
\ This dataset is designed to provide training data
for common sense relationships pulls together from various
sources.

The dataset is multi-lingual. See langauge codes and language info
here: https://github.com/commonsense/conceptnet5/wiki/Languages


This dataset provides an interface for the conceptnet5 csv file, and
some (but not all) of the raw text data used to build conceptnet5:
omcsnet_sentences_free.txt, and omcsnet_sentences_more.txt.

One use of this dataset would be to learn to extract the conceptnet
relationship from the omcsnet sentences.

Conceptnet5 has 34,074,917 relationships. Of those relationships,
there are 2,176,099 surface text sentences related to those 2M
entries.

omcsnet_sentences_free has 898,161 lines. omcsnet_sentences_more has
2,001,736 lines.

Original downloads are available here
https://github.com/commonsense/conceptnet5/wiki/Downloads. For more
information, see: https://github.com/commonsense/conceptnet5/wiki

The omcsnet data comes with the following warning from the authors of
the above site:

Remember: this data comes from various forms of
crowdsourcing. Sentences in these files are not necessarily true,
useful, or appropriate.
```

- **许可**：这项工作包括来自 ConceptNet 5 的数据，这些数据由 Commonsense Computing Initiative 编译。ConceptNet 5 在 Creative Commons Attribution-ShareAlike license (CC BY SA 3.0) 下免费提供，网址为 http://conceptnet.io。

包含的数据由 Commonsense Computing 项目的贡献者、Wikimedia 项目的贡献者、DBPedia、OpenCyc、Games with a Purpose、普林斯顿大学的 WordNet、Francis Bond 的 Open Multilingual WordNet 和 Jim Breen 的 JMDict 创建。

还有各种其他许可。请参阅：https://github.com/commonsense/conceptnet5/wiki/Copying-and-sharing-ConceptNet

- **版本**：5.7.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 34074917

- **特征**：

```json
{
    "sentence": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "full_rel": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "rel": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "arg1": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "arg2": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "lang": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "extra_info": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "weight": {
        "dtype": "float32",
        "id": null,
        "_type": "Value"
    }
}
```

## omcs_sentences_free

Use the following command to load this dataset in TFDS:

```python
ds = tfds.load('huggingface:conceptnet5/omcs_sentences_free')
```

- **Description**:

```
\ This dataset is designed to provide training data
for common sense relationships pulls together from various
sources.

The dataset is multi-lingual. See langauge codes and language info
here: https://github.com/commonsense/conceptnet5/wiki/Languages


This dataset provides an interface for the conceptnet5 csv file, and
some (but not all) of the raw text data used to build conceptnet5:
omcsnet_sentences_free.txt, and omcsnet_sentences_more.txt.

One use of this dataset would be to learn to extract the conceptnet
relationship from the omcsnet sentences.

Conceptnet5 has 34,074,917 relationships. Of those relationships,
there are 2,176,099 surface text sentences related to those 2M
entries.

omcsnet_sentences_free has 898,161 lines. omcsnet_sentences_more has
2,001,736 lines.

Original downloads are available here
https://github.com/commonsense/conceptnet5/wiki/Downloads. For more
information, see: https://github.com/commonsense/conceptnet5/wiki

The omcsnet data comes with the following warning from the authors of
the above site:

Remember: this data comes from various forms of
crowdsourcing. Sentences in these files are not necessarily true,
useful, or appropriate.
```

- **许可**：这项工作包括来自 ConceptNet 5 的数据，这些数据由 Commonsense Computing Initiative 编译。ConceptNet 5 在 Creative Commons Attribution-ShareAlike license (CC BY SA 3.0) 下免费提供，网址为 http://conceptnet.io。

包含的数据由 Commonsense Computing 项目的贡献者、Wikimedia 项目的贡献者、DBPedia、OpenCyc、Games with a Purpose、普林斯顿大学的 WordNet、Francis Bond 的 Open Multilingual WordNet 和 Jim Breen 的 JMDict 创建。

还有各种其他许可。请参阅：https://github.com/commonsense/conceptnet5/wiki/Copying-and-sharing-ConceptNet

- **版本**：5.7.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 898160

- **特征**：

```json
{
    "sentence": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "raw_data": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "lang": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## omcs_sentences_more

Use the following command to load this dataset in TFDS:

```python
ds = tfds.load('huggingface:conceptnet5/omcs_sentences_more')
```

- **Description**:

```
\ This dataset is designed to provide training data
for common sense relationships pulls together from various
sources.

The dataset is multi-lingual. See langauge codes and language info
here: https://github.com/commonsense/conceptnet5/wiki/Languages


This dataset provides an interface for the conceptnet5 csv file, and
some (but not all) of the raw text data used to build conceptnet5:
omcsnet_sentences_free.txt, and omcsnet_sentences_more.txt.

One use of this dataset would be to learn to extract the conceptnet
relationship from the omcsnet sentences.

Conceptnet5 has 34,074,917 relationships. Of those relationships,
there are 2,176,099 surface text sentences related to those 2M
entries.

omcsnet_sentences_free has 898,161 lines. omcsnet_sentences_more has
2,001,736 lines.

Original downloads are available here
https://github.com/commonsense/conceptnet5/wiki/Downloads. For more
information, see: https://github.com/commonsense/conceptnet5/wiki

The omcsnet data comes with the following warning from the authors of
the above site:

Remember: this data comes from various forms of
crowdsourcing. Sentences in these files are not necessarily true,
useful, or appropriate.
```

- **许可**：这项工作包括来自 ConceptNet 5 的数据，这些数据由 Commonsense Computing Initiative 编译。ConceptNet 5 在 Creative Commons Attribution-ShareAlike license (CC BY SA 3.0) 下免费提供，网址为 http://conceptnet.io。

包含的数据由 Commonsense Computing 项目的贡献者、Wikimedia 项目的贡献者、DBPedia、OpenCyc、Games with a Purpose、普林斯顿大学的 WordNet、Francis Bond 的 Open Multilingual WordNet 和 Jim Breen 的 JMDict 创建。

还有各种其他许可。请参阅：https://github.com/commonsense/conceptnet5/wiki/Copying-and-sharing-ConceptNet

- **版本**：5.7.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 2001735

- **特征**：

```json
{
    "sentence": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "raw_data": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "lang": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```
