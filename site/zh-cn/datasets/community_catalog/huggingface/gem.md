# gem

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/gem)
- [Huggingface](https://huggingface.co/datasets/gem)

## mlsum_de

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:gem/mlsum_de')
```

- **说明**：

```
GEM is a benchmark environment for Natural Language Generation with a focus on its Evaluation,
both through human annotations and automated Metrics.

GEM aims to:
- measure NLG progress across 13 datasets spanning many NLG tasks and languages.
- provide an in-depth analysis of data and models presented via data statements and challenge sets.
- develop standards for evaluation of generated text using both automated and human metrics.

It is our goal to regularly update GEM and to encourage toward more inclusive practices in dataset development
by extending existing data or developing datasets for additional languages.
```

- **许可**：CC-BY-SA-4.0
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'challenge_test_covid'` | 5058
`'challenge_train_sample'` | 500
`'challenge_validation_sample'` | 500
`'test'` | 10695
`'train'` | 220748
`'validation'` | 11392

- **特征**：

```json
{
    "gem_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "gem_parent_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "text": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "topic": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "url": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "date": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "target": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "references": [
        {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        }
    ]
}
```

## mlsum_es

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:gem/mlsum_es')
```

- **说明**：

```
GEM is a benchmark environment for Natural Language Generation with a focus on its Evaluation,
both through human annotations and automated Metrics.

GEM aims to:
- measure NLG progress across 13 datasets spanning many NLG tasks and languages.
- provide an in-depth analysis of data and models presented via data statements and challenge sets.
- develop standards for evaluation of generated text using both automated and human metrics.

It is our goal to regularly update GEM and to encourage toward more inclusive practices in dataset development
by extending existing data or developing datasets for additional languages.
```

- **许可**：CC-BY-SA-4.0
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'challenge_test_covid'` | 1938
`'challenge_train_sample'` | 500
`'challenge_validation_sample'` | 500
`'test'` | 13366
`'train'` | 259888
`'validation'` | 9977

- **特征**：

```json
{
    "gem_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "gem_parent_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "text": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "topic": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "url": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "date": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "target": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "references": [
        {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        }
    ]
}
```

## wiki_lingua_es_en_v0

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:gem/wiki_lingua_es_en_v0')
```

- **说明**：

```
GEM is a benchmark environment for Natural Language Generation with a focus on its Evaluation,
both through human annotations and automated Metrics.

GEM aims to:
- measure NLG progress across 13 datasets spanning many NLG tasks and languages.
- provide an in-depth analysis of data and models presented via data statements and challenge sets.
- develop standards for evaluation of generated text using both automated and human metrics.

It is our goal to regularly update GEM and to encourage toward more inclusive practices in dataset development
by extending existing data or developing datasets for additional languages.
```

- **许可**：CC-BY-SA-4.0
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 19797
`'train'` | 79515
`'validation'` | 8835

- **特征**：

```json
{
    "gem_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "gem_parent_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "source": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "target": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "references": [
        {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        }
    ]
}
```

## wiki_lingua_ru_en_v0

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:gem/wiki_lingua_ru_en_v0')
```

- **说明**：

```
GEM is a benchmark environment for Natural Language Generation with a focus on its Evaluation,
both through human annotations and automated Metrics.

GEM aims to:
- measure NLG progress across 13 datasets spanning many NLG tasks and languages.
- provide an in-depth analysis of data and models presented via data statements and challenge sets.
- develop standards for evaluation of generated text using both automated and human metrics.

It is our goal to regularly update GEM and to encourage toward more inclusive practices in dataset development
by extending existing data or developing datasets for additional languages.
```

- **许可**：CC-BY-SA-4.0
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 9094
`'train'` | 36898
`'validation'` | 4100

- **特征**：

```json
{
    "gem_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "gem_parent_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "source": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "target": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "references": [
        {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        }
    ]
}
```

## wiki_lingua_tr_en_v0

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:gem/wiki_lingua_tr_en_v0')
```

- **说明**：

```
GEM is a benchmark environment for Natural Language Generation with a focus on its Evaluation,
both through human annotations and automated Metrics.

GEM aims to:
- measure NLG progress across 13 datasets spanning many NLG tasks and languages.
- provide an in-depth analysis of data and models presented via data statements and challenge sets.
- develop standards for evaluation of generated text using both automated and human metrics.

It is our goal to regularly update GEM and to encourage toward more inclusive practices in dataset development
by extending existing data or developing datasets for additional languages.
```

- **许可**：CC-BY-SA-4.0
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 808
`'train'` | 3193
`'validation'` | 355

- **特征**：

```json
{
    "gem_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "gem_parent_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "source": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "target": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "references": [
        {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        }
    ]
}
```

## wiki_lingua_vi_en_v0

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:gem/wiki_lingua_vi_en_v0')
```

- **说明**：

```
GEM is a benchmark environment for Natural Language Generation with a focus on its Evaluation,
both through human annotations and automated Metrics.

GEM aims to:
- measure NLG progress across 13 datasets spanning many NLG tasks and languages.
- provide an in-depth analysis of data and models presented via data statements and challenge sets.
- develop standards for evaluation of generated text using both automated and human metrics.

It is our goal to regularly update GEM and to encourage toward more inclusive practices in dataset development
by extending existing data or developing datasets for additional languages.
```

- **许可**：CC-BY-SA-4.0
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 2167
`'train'` | 9206
`'validation'` | 1023

- **特征**：

```json
{
    "gem_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "gem_parent_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "source": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "target": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "references": [
        {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        }
    ]
}
```

## wiki_lingua_arabic_ar

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:gem/wiki_lingua_arabic_ar')
```

- **说明**：

```
GEM is a benchmark environment for Natural Language Generation with a focus on its Evaluation,
both through human annotations and automated Metrics.

GEM aims to:
- measure NLG progress across 13 datasets spanning many NLG tasks and languages.
- provide an in-depth analysis of data and models presented via data statements and challenge sets.
- develop standards for evaluation of generated text using both automated and human metrics.

It is our goal to regularly update GEM and to encourage toward more inclusive practices in dataset development
by extending existing data or developing datasets for additional languages.
```

- **许可**：CC-BY-SA-4.0
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 5841
`'train'` | 20441
`'validation'` | 2919

- **特征**：

```json
{
    "gem_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "gem_parent_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "source_aligned": {
        "languages": [
            "ar",
            "en"
        ],
        "id": null,
        "_type": "Translation"
    },
    "target_aligned": {
        "languages": [
            "ar",
            "en"
        ],
        "id": null,
        "_type": "Translation"
    },
    "source": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "target": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "references": [
        {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        }
    ]
}
```

## wiki_lingua_chinese_zh

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:gem/wiki_lingua_chinese_zh')
```

- **说明**：

```
GEM is a benchmark environment for Natural Language Generation with a focus on its Evaluation,
both through human annotations and automated Metrics.

GEM aims to:
- measure NLG progress across 13 datasets spanning many NLG tasks and languages.
- provide an in-depth analysis of data and models presented via data statements and challenge sets.
- develop standards for evaluation of generated text using both automated and human metrics.

It is our goal to regularly update GEM and to encourage toward more inclusive practices in dataset development
by extending existing data or developing datasets for additional languages.
```

- **许可**：CC-BY-SA-4.0
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 3775
`'train'` | 13211
`'validation'` | 1886

- **特征**：

```json
{
    "gem_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "gem_parent_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "source_aligned": {
        "languages": [
            "zh",
            "en"
        ],
        "id": null,
        "_type": "Translation"
    },
    "target_aligned": {
        "languages": [
            "zh",
            "en"
        ],
        "id": null,
        "_type": "Translation"
    },
    "source": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "target": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "references": [
        {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        }
    ]
}
```

## wiki_lingua_czech_cs

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:gem/wiki_lingua_czech_cs')
```

- **说明**：

```
GEM is a benchmark environment for Natural Language Generation with a focus on its Evaluation,
both through human annotations and automated Metrics.

GEM aims to:
- measure NLG progress across 13 datasets spanning many NLG tasks and languages.
- provide an in-depth analysis of data and models presented via data statements and challenge sets.
- develop standards for evaluation of generated text using both automated and human metrics.

It is our goal to regularly update GEM and to encourage toward more inclusive practices in dataset development
by extending existing data or developing datasets for additional languages.
```

- **许可**：CC-BY-SA-4.0
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 1438
`'train'` | 5033
`'validation'` | 718

- **特征**：

```json
{
    "gem_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "gem_parent_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "source_aligned": {
        "languages": [
            "cs",
            "en"
        ],
        "id": null,
        "_type": "Translation"
    },
    "target_aligned": {
        "languages": [
            "cs",
            "en"
        ],
        "id": null,
        "_type": "Translation"
    },
    "source": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "target": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "references": [
        {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        }
    ]
}
```

## wiki_lingua_dutch_nl

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:gem/wiki_lingua_dutch_nl')
```

- **说明**：

```
GEM is a benchmark environment for Natural Language Generation with a focus on its Evaluation,
both through human annotations and automated Metrics.

GEM aims to:
- measure NLG progress across 13 datasets spanning many NLG tasks and languages.
- provide an in-depth analysis of data and models presented via data statements and challenge sets.
- develop standards for evaluation of generated text using both automated and human metrics.

It is our goal to regularly update GEM and to encourage toward more inclusive practices in dataset development
by extending existing data or developing datasets for additional languages.
```

- **许可**：CC-BY-SA-4.0
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 6248
`'train'` | 21866
`'validation'` | 3123

- **特征**：

```json
{
    "gem_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "gem_parent_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "source_aligned": {
        "languages": [
            "nl",
            "en"
        ],
        "id": null,
        "_type": "Translation"
    },
    "target_aligned": {
        "languages": [
            "nl",
            "en"
        ],
        "id": null,
        "_type": "Translation"
    },
    "source": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "target": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "references": [
        {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        }
    ]
}
```

## wiki_lingua_english_en

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:gem/wiki_lingua_english_en')
```

- **说明**：

```
GEM is a benchmark environment for Natural Language Generation with a focus on its Evaluation,
both through human annotations and automated Metrics.

GEM aims to:
- measure NLG progress across 13 datasets spanning many NLG tasks and languages.
- provide an in-depth analysis of data and models presented via data statements and challenge sets.
- develop standards for evaluation of generated text using both automated and human metrics.

It is our goal to regularly update GEM and to encourage toward more inclusive practices in dataset development
by extending existing data or developing datasets for additional languages.
```

- **许可**：CC-BY-SA-4.0
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 28614
`'train'` | 99020
`'validation'` | 13823

- **特征**：

```json
{
    "gem_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "gem_parent_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "source_aligned": {
        "languages": [
            "en",
            "en"
        ],
        "id": null,
        "_type": "Translation"
    },
    "target_aligned": {
        "languages": [
            "en",
            "en"
        ],
        "id": null,
        "_type": "Translation"
    },
    "source": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "target": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "references": [
        {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        }
    ]
}
```

## wiki_lingua_french_fr

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:gem/wiki_lingua_french_fr')
```

- **说明**：

```
GEM is a benchmark environment for Natural Language Generation with a focus on its Evaluation,
both through human annotations and automated Metrics.

GEM aims to:
- measure NLG progress across 13 datasets spanning many NLG tasks and languages.
- provide an in-depth analysis of data and models presented via data statements and challenge sets.
- develop standards for evaluation of generated text using both automated and human metrics.

It is our goal to regularly update GEM and to encourage toward more inclusive practices in dataset development
by extending existing data or developing datasets for additional languages.
```

- **许可**：CC-BY-SA-4.0
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 12731
`'train'` | 44556
`'validation'` | 6364

- **特征**：

```json
{
    "gem_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "gem_parent_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "source_aligned": {
        "languages": [
            "fr",
            "en"
        ],
        "id": null,
        "_type": "Translation"
    },
    "target_aligned": {
        "languages": [
            "fr",
            "en"
        ],
        "id": null,
        "_type": "Translation"
    },
    "source": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "target": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "references": [
        {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        }
    ]
}
```

## wiki_lingua_german_de

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:gem/wiki_lingua_german_de')
```

- **说明**：

```
GEM is a benchmark environment for Natural Language Generation with a focus on its Evaluation,
both through human annotations and automated Metrics.

GEM aims to:
- measure NLG progress across 13 datasets spanning many NLG tasks and languages.
- provide an in-depth analysis of data and models presented via data statements and challenge sets.
- develop standards for evaluation of generated text using both automated and human metrics.

It is our goal to regularly update GEM and to encourage toward more inclusive practices in dataset development
by extending existing data or developing datasets for additional languages.
```

- **许可**：CC-BY-SA-4.0
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 11669
`'train'` | 40839
`'validation'` | 5833

- **特征**：

```json
{
    "gem_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "gem_parent_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "source_aligned": {
        "languages": [
            "de",
            "en"
        ],
        "id": null,
        "_type": "Translation"
    },
    "target_aligned": {
        "languages": [
            "de",
            "en"
        ],
        "id": null,
        "_type": "Translation"
    },
    "source": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "target": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "references": [
        {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        }
    ]
}
```

## wiki_lingua_hindi_hi

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:gem/wiki_lingua_hindi_hi')
```

- **说明**：

```
GEM is a benchmark environment for Natural Language Generation with a focus on its Evaluation,
both through human annotations and automated Metrics.

GEM aims to:
- measure NLG progress across 13 datasets spanning many NLG tasks and languages.
- provide an in-depth analysis of data and models presented via data statements and challenge sets.
- develop standards for evaluation of generated text using both automated and human metrics.

It is our goal to regularly update GEM and to encourage toward more inclusive practices in dataset development
by extending existing data or developing datasets for additional languages.
```

- **许可**：CC-BY-SA-4.0
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 1984
`'train'` | 6942
`'validation'` | 991

- **特征**：

```json
{
    "gem_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "gem_parent_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "source_aligned": {
        "languages": [
            "hi",
            "en"
        ],
        "id": null,
        "_type": "Translation"
    },
    "target_aligned": {
        "languages": [
            "hi",
            "en"
        ],
        "id": null,
        "_type": "Translation"
    },
    "source": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "target": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "references": [
        {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        }
    ]
}
```

## wiki_lingua_indonesian_id

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:gem/wiki_lingua_indonesian_id')
```

- **说明**：

```
GEM is a benchmark environment for Natural Language Generation with a focus on its Evaluation,
both through human annotations and automated Metrics.

GEM aims to:
- measure NLG progress across 13 datasets spanning many NLG tasks and languages.
- provide an in-depth analysis of data and models presented via data statements and challenge sets.
- develop standards for evaluation of generated text using both automated and human metrics.

It is our goal to regularly update GEM and to encourage toward more inclusive practices in dataset development
by extending existing data or developing datasets for additional languages.
```

- **许可**：CC-BY-SA-4.0
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 9497
`'train'` | 33237
`'validation'` | 4747

- **特征**：

```json
{
    "gem_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "gem_parent_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "source_aligned": {
        "languages": [
            "id",
            "en"
        ],
        "id": null,
        "_type": "Translation"
    },
    "target_aligned": {
        "languages": [
            "id",
            "en"
        ],
        "id": null,
        "_type": "Translation"
    },
    "source": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "target": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "references": [
        {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        }
    ]
}
```

## wiki_lingua_italian_it

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:gem/wiki_lingua_italian_it')
```

- **说明**：

```
GEM is a benchmark environment for Natural Language Generation with a focus on its Evaluation,
both through human annotations and automated Metrics.

GEM aims to:
- measure NLG progress across 13 datasets spanning many NLG tasks and languages.
- provide an in-depth analysis of data and models presented via data statements and challenge sets.
- develop standards for evaluation of generated text using both automated and human metrics.

It is our goal to regularly update GEM and to encourage toward more inclusive practices in dataset development
by extending existing data or developing datasets for additional languages.
```

- **许可**：CC-BY-SA-4.0
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 10189
`'train'` | 35661
`'validation'` | 5093

- **特征**：

```json
{
    "gem_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "gem_parent_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "source_aligned": {
        "languages": [
            "it",
            "en"
        ],
        "id": null,
        "_type": "Translation"
    },
    "target_aligned": {
        "languages": [
            "it",
            "en"
        ],
        "id": null,
        "_type": "Translation"
    },
    "source": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "target": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "references": [
        {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        }
    ]
}
```

## wiki_lingua_japanese_ja

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:gem/wiki_lingua_japanese_ja')
```

- **说明**：

```
GEM is a benchmark environment for Natural Language Generation with a focus on its Evaluation,
both through human annotations and automated Metrics.

GEM aims to:
- measure NLG progress across 13 datasets spanning many NLG tasks and languages.
- provide an in-depth analysis of data and models presented via data statements and challenge sets.
- develop standards for evaluation of generated text using both automated and human metrics.

It is our goal to regularly update GEM and to encourage toward more inclusive practices in dataset development
by extending existing data or developing datasets for additional languages.
```

- **许可**：CC-BY-SA-4.0
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 2530
`'train'` | 8853
`'validation'` | 1264

- **特征**：

```json
{
    "gem_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "gem_parent_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "source_aligned": {
        "languages": [
            "ja",
            "en"
        ],
        "id": null,
        "_type": "Translation"
    },
    "target_aligned": {
        "languages": [
            "ja",
            "en"
        ],
        "id": null,
        "_type": "Translation"
    },
    "source": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "target": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "references": [
        {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        }
    ]
}
```

## wiki_lingua_korean_ko

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:gem/wiki_lingua_korean_ko')
```

- **说明**：

```
GEM is a benchmark environment for Natural Language Generation with a focus on its Evaluation,
both through human annotations and automated Metrics.

GEM aims to:
- measure NLG progress across 13 datasets spanning many NLG tasks and languages.
- provide an in-depth analysis of data and models presented via data statements and challenge sets.
- develop standards for evaluation of generated text using both automated and human metrics.

It is our goal to regularly update GEM and to encourage toward more inclusive practices in dataset development
by extending existing data or developing datasets for additional languages.
```

- **许可**：CC-BY-SA-4.0
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 2436
`'train'` | 8524
`'validation'` | 1216

- **特征**：

```json
{
    "gem_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "gem_parent_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "source_aligned": {
        "languages": [
            "ko",
            "en"
        ],
        "id": null,
        "_type": "Translation"
    },
    "target_aligned": {
        "languages": [
            "ko",
            "en"
        ],
        "id": null,
        "_type": "Translation"
    },
    "source": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "target": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "references": [
        {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        }
    ]
}
```

## wiki_lingua_portuguese_pt

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:gem/wiki_lingua_portuguese_pt')
```

- **说明**：

```
GEM is a benchmark environment for Natural Language Generation with a focus on its Evaluation,
both through human annotations and automated Metrics.

GEM aims to:
- measure NLG progress across 13 datasets spanning many NLG tasks and languages.
- provide an in-depth analysis of data and models presented via data statements and challenge sets.
- develop standards for evaluation of generated text using both automated and human metrics.

It is our goal to regularly update GEM and to encourage toward more inclusive practices in dataset development
by extending existing data or developing datasets for additional languages.
```

- **许可**：CC-BY-SA-4.0
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 16331
`'train'` | 57159
`'validation'` | 8165

- **特征**：

```json
{
    "gem_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "gem_parent_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "source_aligned": {
        "languages": [
            "pt",
            "en"
        ],
        "id": null,
        "_type": "Translation"
    },
    "target_aligned": {
        "languages": [
            "pt",
            "en"
        ],
        "id": null,
        "_type": "Translation"
    },
    "source": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "target": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "references": [
        {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        }
    ]
}
```

## wiki_lingua_russian_ru

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:gem/wiki_lingua_russian_ru')
```

- **说明**：

```
GEM is a benchmark environment for Natural Language Generation with a focus on its Evaluation,
both through human annotations and automated Metrics.

GEM aims to:
- measure NLG progress across 13 datasets spanning many NLG tasks and languages.
- provide an in-depth analysis of data and models presented via data statements and challenge sets.
- develop standards for evaluation of generated text using both automated and human metrics.

It is our goal to regularly update GEM and to encourage toward more inclusive practices in dataset development
by extending existing data or developing datasets for additional languages.
```

- **许可**：CC-BY-SA-4.0
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 10580
`'train'` | 37028
`'validation'` | 5288

- **特征**：

```json
{
    "gem_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "gem_parent_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "source_aligned": {
        "languages": [
            "ru",
            "en"
        ],
        "id": null,
        "_type": "Translation"
    },
    "target_aligned": {
        "languages": [
            "ru",
            "en"
        ],
        "id": null,
        "_type": "Translation"
    },
    "source": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "target": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "references": [
        {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        }
    ]
}
```

## wiki_lingua_spanish_es

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:gem/wiki_lingua_spanish_es')
```

- **说明**：

```
GEM is a benchmark environment for Natural Language Generation with a focus on its Evaluation,
both through human annotations and automated Metrics.

GEM aims to:
- measure NLG progress across 13 datasets spanning many NLG tasks and languages.
- provide an in-depth analysis of data and models presented via data statements and challenge sets.
- develop standards for evaluation of generated text using both automated and human metrics.

It is our goal to regularly update GEM and to encourage toward more inclusive practices in dataset development
by extending existing data or developing datasets for additional languages.
```

- **许可**：CC-BY-SA-4.0
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 22632
`'train'` | 79212
`'validation'` | 11316

- **特征**：

```json
{
    "gem_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "gem_parent_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "source_aligned": {
        "languages": [
            "es",
            "en"
        ],
        "id": null,
        "_type": "Translation"
    },
    "target_aligned": {
        "languages": [
            "es",
            "en"
        ],
        "id": null,
        "_type": "Translation"
    },
    "source": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "target": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "references": [
        {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        }
    ]
}
```

## wiki_lingua_thai_th

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:gem/wiki_lingua_thai_th')
```

- **说明**：

```
GEM is a benchmark environment for Natural Language Generation with a focus on its Evaluation,
both through human annotations and automated Metrics.

GEM aims to:
- measure NLG progress across 13 datasets spanning many NLG tasks and languages.
- provide an in-depth analysis of data and models presented via data statements and challenge sets.
- develop standards for evaluation of generated text using both automated and human metrics.

It is our goal to regularly update GEM and to encourage toward more inclusive practices in dataset development
by extending existing data or developing datasets for additional languages.
```

- **许可**：CC-BY-SA-4.0
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 2950
`'train'` | 10325
`'validation'` | 1475

- **特征**：

```json
{
    "gem_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "gem_parent_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "source_aligned": {
        "languages": [
            "th",
            "en"
        ],
        "id": null,
        "_type": "Translation"
    },
    "target_aligned": {
        "languages": [
            "th",
            "en"
        ],
        "id": null,
        "_type": "Translation"
    },
    "source": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "target": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "references": [
        {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        }
    ]
}
```

## wiki_lingua_turkish_tr

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:gem/wiki_lingua_turkish_tr')
```

- **说明**：

```
GEM is a benchmark environment for Natural Language Generation with a focus on its Evaluation,
both through human annotations and automated Metrics.

GEM aims to:
- measure NLG progress across 13 datasets spanning many NLG tasks and languages.
- provide an in-depth analysis of data and models presented via data statements and challenge sets.
- develop standards for evaluation of generated text using both automated and human metrics.

It is our goal to regularly update GEM and to encourage toward more inclusive practices in dataset development
by extending existing data or developing datasets for additional languages.
```

- **许可**：CC-BY-SA-4.0
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 900
`'train'` | 3148
`'validation'` | 449

- **特征**：

```json
{
    "gem_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "gem_parent_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "source_aligned": {
        "languages": [
            "tr",
            "en"
        ],
        "id": null,
        "_type": "Translation"
    },
    "target_aligned": {
        "languages": [
            "tr",
            "en"
        ],
        "id": null,
        "_type": "Translation"
    },
    "source": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "target": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "references": [
        {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        }
    ]
}
```

## wiki_lingua_vietnamese_vi

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:gem/wiki_lingua_vietnamese_vi')
```

- **说明**：

```
GEM is a benchmark environment for Natural Language Generation with a focus on its Evaluation,
both through human annotations and automated Metrics.

GEM aims to:
- measure NLG progress across 13 datasets spanning many NLG tasks and languages.
- provide an in-depth analysis of data and models presented via data statements and challenge sets.
- develop standards for evaluation of generated text using both automated and human metrics.

It is our goal to regularly update GEM and to encourage toward more inclusive practices in dataset development
by extending existing data or developing datasets for additional languages.
```

- **许可**：CC-BY-SA-4.0
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 3917
`'train'` | 13707
`'validation'` | 1957

- **特征**：

```json
{
    "gem_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "gem_parent_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "source_aligned": {
        "languages": [
            "vi",
            "en"
        ],
        "id": null,
        "_type": "Translation"
    },
    "target_aligned": {
        "languages": [
            "vi",
            "en"
        ],
        "id": null,
        "_type": "Translation"
    },
    "source": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "target": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "references": [
        {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        }
    ]
}
```

## xsum

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:gem/xsum')
```

- **说明**：

```
GEM is a benchmark environment for Natural Language Generation with a focus on its Evaluation,
both through human annotations and automated Metrics.

GEM aims to:
- measure NLG progress across 13 datasets spanning many NLG tasks and languages.
- provide an in-depth analysis of data and models presented via data statements and challenge sets.
- develop standards for evaluation of generated text using both automated and human metrics.

It is our goal to regularly update GEM and to encourage toward more inclusive practices in dataset development
by extending existing data or developing datasets for additional languages.
```

- **许可**：CC-BY-SA-4.0
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'challenge_test_backtranslation'` | 500
`'challenge_test_bfp_02'` | 500
`'challenge_test_bfp_05'` | 500
`'challenge_test_covid'` | 401
`'challenge_test_nopunc'` | 500
`'challenge_train_sample'` | 500
`'challenge_validation_sample'` | 500
`'test'` | 1166
`'train'` | 23206
`'validation'` | 1117

- **特征**：

```json
{
    "gem_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "gem_parent_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "xsum_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "document": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "target": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "references": [
        {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        }
    ]
}
```

## common_gen

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:gem/common_gen')
```

- **说明**：

```
GEM is a benchmark environment for Natural Language Generation with a focus on its Evaluation,
both through human annotations and automated Metrics.

GEM aims to:
- measure NLG progress across 13 datasets spanning many NLG tasks and languages.
- provide an in-depth analysis of data and models presented via data statements and challenge sets.
- develop standards for evaluation of generated text using both automated and human metrics.

It is our goal to regularly update GEM and to encourage toward more inclusive practices in dataset development
by extending existing data or developing datasets for additional languages.
```

- **许可**：CC-BY-SA-4.0
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'challenge_test_scramble'` | 500
`'challenge_train_sample'` | 500
`'challenge_validation_sample'` | 500
`'test'` | 1497
`'train'` | 67389
`'validation'` | 993

- **特征**：

```json
{
    "gem_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "gem_parent_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "concept_set_id": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "concepts": [
        {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        }
    ],
    "target": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "references": [
        {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        }
    ]
}
```

## cs_restaurants

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:gem/cs_restaurants')
```

- **说明**：

```
GEM is a benchmark environment for Natural Language Generation with a focus on its Evaluation,
both through human annotations and automated Metrics.

GEM aims to:
- measure NLG progress across 13 datasets spanning many NLG tasks and languages.
- provide an in-depth analysis of data and models presented via data statements and challenge sets.
- develop standards for evaluation of generated text using both automated and human metrics.

It is our goal to regularly update GEM and to encourage toward more inclusive practices in dataset development
by extending existing data or developing datasets for additional languages.
```

- **许可**：CC-BY-SA-4.0
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'challenge_test_scramble'` | 500
`'challenge_train_sample'` | 500
`'challenge_validation_sample'` | 500
`'test'` | 842
`'train'` | 3569
`'validation'` | 781

- **特征**：

```json
{
    "gem_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "gem_parent_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "dialog_act": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "dialog_act_delexicalized": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "target_delexicalized": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "target": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "references": [
        {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        }
    ]
}
```

## dart

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:gem/dart')
```

- **说明**：

```
GEM is a benchmark environment for Natural Language Generation with a focus on its Evaluation,
both through human annotations and automated Metrics.

GEM aims to:
- measure NLG progress across 13 datasets spanning many NLG tasks and languages.
- provide an in-depth analysis of data and models presented via data statements and challenge sets.
- develop standards for evaluation of generated text using both automated and human metrics.

It is our goal to regularly update GEM and to encourage toward more inclusive practices in dataset development
by extending existing data or developing datasets for additional languages.
```

- **许可**：CC-BY-SA-4.0
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 5097
`'train'` | 62659
`'validation'` | 2768

- **特征**：

```json
{
    "gem_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "gem_parent_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "dart_id": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "tripleset": [
        [
            {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            }
        ]
    ],
    "subtree_was_extended": {
        "dtype": "bool",
        "id": null,
        "_type": "Value"
    },
    "target_sources": [
        {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        }
    ],
    "target": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "references": [
        {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        }
    ]
}
```

## e2e_nlg

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:gem/e2e_nlg')
```

- **说明**：

```
GEM is a benchmark environment for Natural Language Generation with a focus on its Evaluation,
both through human annotations and automated Metrics.

GEM aims to:
- measure NLG progress across 13 datasets spanning many NLG tasks and languages.
- provide an in-depth analysis of data and models presented via data statements and challenge sets.
- develop standards for evaluation of generated text using both automated and human metrics.

It is our goal to regularly update GEM and to encourage toward more inclusive practices in dataset development
by extending existing data or developing datasets for additional languages.
```

- **许可**：CC-BY-SA-4.0
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'challenge_test_scramble'` | 500
`'challenge_train_sample'` | 500
`'challenge_validation_sample'` | 500
`'test'` | 4693
`'train'` | 33525
`'validation'` | 4299

- **特征**：

```json
{
    "gem_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "gem_parent_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "meaning_representation": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "target": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "references": [
        {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        }
    ]
}
```

## totto

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:gem/totto')
```

- **说明**：

```
GEM is a benchmark environment for Natural Language Generation with a focus on its Evaluation,
both through human annotations and automated Metrics.

GEM aims to:
- measure NLG progress across 13 datasets spanning many NLG tasks and languages.
- provide an in-depth analysis of data and models presented via data statements and challenge sets.
- develop standards for evaluation of generated text using both automated and human metrics.

It is our goal to regularly update GEM and to encourage toward more inclusive practices in dataset development
by extending existing data or developing datasets for additional languages.
```

- **许可**：CC-BY-SA-4.0
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'challenge_test_scramble'` | 500
`'challenge_train_sample'` | 500
`'challenge_validation_sample'` | 500
`'test'` | 7700
`'train'` | 121153
`'validation'` | 7700

- **特征**：

```json
{
    "gem_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "gem_parent_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "totto_id": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "table_page_title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "table_webpage_url": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "table_section_title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "table_section_text": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "table": [
        [
            {
                "column_span": {
                    "dtype": "int32",
                    "id": null,
                    "_type": "Value"
                },
                "is_header": {
                    "dtype": "bool",
                    "id": null,
                    "_type": "Value"
                },
                "row_span": {
                    "dtype": "int32",
                    "id": null,
                    "_type": "Value"
                },
                "value": {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                }
            }
        ]
    ],
    "highlighted_cells": [
        [
            {
                "dtype": "int32",
                "id": null,
                "_type": "Value"
            }
        ]
    ],
    "example_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "sentence_annotations": [
        {
            "original_sentence": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "sentence_after_deletion": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "sentence_after_ambiguity": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "final_sentence": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            }
        }
    ],
    "overlap_subset": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "target": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "references": [
        {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        }
    ]
}
```

## web_nlg_en

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:gem/web_nlg_en')
```

- **说明**：

```
GEM is a benchmark environment for Natural Language Generation with a focus on its Evaluation,
both through human annotations and automated Metrics.

GEM aims to:
- measure NLG progress across 13 datasets spanning many NLG tasks and languages.
- provide an in-depth analysis of data and models presented via data statements and challenge sets.
- develop standards for evaluation of generated text using both automated and human metrics.

It is our goal to regularly update GEM and to encourage toward more inclusive practices in dataset development
by extending existing data or developing datasets for additional languages.
```

- **许可**：CC-BY-SA-4.0
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'challenge_test_numbers'` | 500
`'challenge_test_scramble'` | 500
`'challenge_train_sample'` | 502
`'challenge_validation_sample'` | 499
`'test'` | 1779
`'train'` | 35426
`'validation'` | 1667

- **特征**：

```json
{
    "gem_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "gem_parent_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "input": [
        {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        }
    ],
    "target": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "references": [
        {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        }
    ],
    "category": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "webnlg_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## web_nlg_ru

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:gem/web_nlg_ru')
```

- **说明**：

```
GEM is a benchmark environment for Natural Language Generation with a focus on its Evaluation,
both through human annotations and automated Metrics.

GEM aims to:
- measure NLG progress across 13 datasets spanning many NLG tasks and languages.
- provide an in-depth analysis of data and models presented via data statements and challenge sets.
- develop standards for evaluation of generated text using both automated and human metrics.

It is our goal to regularly update GEM and to encourage toward more inclusive practices in dataset development
by extending existing data or developing datasets for additional languages.
```

- **许可**：CC-BY-SA-4.0
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'challenge_test_scramble'` | 500
`'challenge_train_sample'` | 501
`'challenge_validation_sample'` | 500
`'test'` | 1102
`'train'` | 14630
`'validation'` | 790

- **特征**：

```json
{
    "gem_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "gem_parent_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "input": [
        {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        }
    ],
    "target": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "references": [
        {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        }
    ],
    "category": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "webnlg_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## wiki_auto_asset_turk

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:gem/wiki_auto_asset_turk')
```

- **说明**：

```
GEM is a benchmark environment for Natural Language Generation with a focus on its Evaluation,
both through human annotations and automated Metrics.

GEM aims to:
- measure NLG progress across 13 datasets spanning many NLG tasks and languages.
- provide an in-depth analysis of data and models presented via data statements and challenge sets.
- develop standards for evaluation of generated text using both automated and human metrics.

It is our goal to regularly update GEM and to encourage toward more inclusive practices in dataset development
by extending existing data or developing datasets for additional languages.
```

- **许可**：CC-BY-SA-4.0
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'challenge_test_asset_backtranslation'` | 359
`'challenge_test_asset_bfp02'` | 359
`'challenge_test_asset_bfp05'` | 359
`'challenge_test_asset_nopunc'` | 359
`'challenge_test_turk_backtranslation'` | 359
`'challenge_test_turk_bfp02'` | 359
`'challenge_test_turk_bfp05'` | 359
`'challenge_test_turk_nopunc'` | 359
`'challenge_train_sample'` | 500
`'challenge_validation_sample'` | 500
`'test_asset'` | 359
`'test_turk'` | 359
`'train'` | 483801
`'validation'` | 20000

- **特征**：

```json
{
    "gem_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "gem_parent_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "source": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "target": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "references": [
        {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        }
    ]
}
```

## schema_guided_dialog

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:gem/schema_guided_dialog')
```

- **说明**：

```
GEM is a benchmark environment for Natural Language Generation with a focus on its Evaluation,
both through human annotations and automated Metrics.

GEM aims to:
- measure NLG progress across 13 datasets spanning many NLG tasks and languages.
- provide an in-depth analysis of data and models presented via data statements and challenge sets.
- develop standards for evaluation of generated text using both automated and human metrics.

It is our goal to regularly update GEM and to encourage toward more inclusive practices in dataset development
by extending existing data or developing datasets for additional languages.
```

- **许可**：CC-BY-SA-4.0
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'challenge_test_backtranslation'` | 500
`'challenge_test_bfp02'` | 500
`'challenge_test_bfp05'` | 500
`'challenge_test_nopunc'` | 500
`'challenge_test_scramble'` | 500
`'challenge_train_sample'` | 500
`'challenge_validation_sample'` | 500
`'test'` | 10000
`'train'` | 164982
`'validation'` | 10000

- **特征**：

```json
{
    "gem_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "gem_parent_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "dialog_acts": [
        {
            "act": {
                "num_classes": 18,
                "names": [
                    "AFFIRM",
                    "AFFIRM_INTENT",
                    "CONFIRM",
                    "GOODBYE",
                    "INFORM",
                    "INFORM_COUNT",
                    "INFORM_INTENT",
                    "NEGATE",
                    "NEGATE_INTENT",
                    "NOTIFY_FAILURE",
                    "NOTIFY_SUCCESS",
                    "OFFER",
                    "OFFER_INTENT",
                    "REQUEST",
                    "REQUEST_ALTS",
                    "REQ_MORE",
                    "SELECT",
                    "THANK_YOU"
                ],
                "id": null,
                "_type": "ClassLabel"
            },
            "slot": {
                "dtype": "string",
                "id": null,
                "_type": "Value"
            },
            "values": [
                {
                    "dtype": "string",
                    "id": null,
                    "_type": "Value"
                }
            ]
        }
    ],
    "context": [
        {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        }
    ],
    "dialog_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "service": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "turn_id": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "prompt": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "target": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "references": [
        {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        }
    ]
}
```
