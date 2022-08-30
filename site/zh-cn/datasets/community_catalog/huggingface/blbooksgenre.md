# blbooksgenre

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/blbooksgenre)
- [Huggingface](https://huggingface.co/datasets/blbooksgenre)

## title_genre_classifiction

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:blbooksgenre/title_genre_classifiction')
```

- **说明**：

```
This dataset contains metadata for resources belonging to the British Library’s digitised printed books (18th-19th century) collection (bl.uk/collection-guides/digitised-printed-books).
This metadata has been extracted from British Library catalogue records.
The metadata held within our main catalogue is updated regularly.
This metadata dataset should be considered a snapshot of this metadata.
```

- **许可**：CC0 1.0 Universal Public Domain
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 1736

- **特征**：

```json
{
    "BL record ID": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "label": {
        "num_classes": 2,
        "names": [
            "Fiction",
            "Non-fiction"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```

## annotated_raw

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:blbooksgenre/annotated_raw')
```

- **说明**：

```
This dataset contains metadata for resources belonging to the British Library’s digitised printed books (18th-19th century) collection (bl.uk/collection-guides/digitised-printed-books).
This metadata has been extracted from British Library catalogue records.
The metadata held within our main catalogue is updated regularly.
This metadata dataset should be considered a snapshot of this metadata.
```

- **许可**：CC0 1.0 Universal Public Domain
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 4398

- **特征**：

```json
{
    "BL record ID": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Name": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Dates associated with name": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Type of name": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Role": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "All names": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "Title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Variant titles": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Series title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Number within series": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Country of publication": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "Place of publication": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "Publisher": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Date of publication": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Edition": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Physical description": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Dewey classification": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "BL shelfmark": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Topics": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Genre": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Languages": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "Notes": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "BL record ID for physical resource": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "classification_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "user_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "subject_ids": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "annotator_date_pub": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "annotator_normalised_date_pub": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "annotator_edition_statement": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "annotator_FAST_genre_terms": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "annotator_FAST_subject_terms": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "annotator_comments": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "annotator_main_language": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "annotator_other_languages_summaries": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "annotator_summaries_language": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "annotator_translation": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "annotator_original_language": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "annotator_publisher": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "annotator_place_pub": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "annotator_country": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "annotator_title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Link to digitised book": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "annotated": {
        "dtype": "bool",
        "id": null,
        "_type": "Value"
    },
    "Type of resource": {
        "num_classes": 2,
        "names": [
            "Monograph",
            "Serial"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    },
    "created_at": {
        "dtype": "timestamp[s]",
        "id": null,
        "_type": "Value"
    },
    "annotator_genre": {
        "num_classes": 4,
        "names": [
            "Fiction",
            "Can't tell",
            "Non-fiction",
            "The book contains both Fiction and Non-Fiction"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```

## raw

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:blbooksgenre/raw')
```

- **说明**：

```
This dataset contains metadata for resources belonging to the British Library’s digitised printed books (18th-19th century) collection (bl.uk/collection-guides/digitised-printed-books).
This metadata has been extracted from British Library catalogue records.
The metadata held within our main catalogue is updated regularly.
This metadata dataset should be considered a snapshot of this metadata.
```

- **许可**：CC0 1.0 Universal Public Domain
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 55343

- **特征**：

```json
{
    "BL record ID": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Name": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Dates associated with name": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Type of name": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Role": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "All names": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "Title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Variant titles": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Series title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Number within series": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Country of publication": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "Place of publication": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "Publisher": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Date of publication": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Edition": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Physical description": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Dewey classification": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "BL shelfmark": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Topics": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Genre": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Languages": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "Notes": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "BL record ID for physical resource": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "classification_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "user_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "subject_ids": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "annotator_date_pub": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "annotator_normalised_date_pub": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "annotator_edition_statement": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "annotator_FAST_genre_terms": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "annotator_FAST_subject_terms": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "annotator_comments": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "annotator_main_language": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "annotator_other_languages_summaries": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "annotator_summaries_language": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "annotator_translation": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "annotator_original_language": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "annotator_publisher": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "annotator_place_pub": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "annotator_country": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "annotator_title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Link to digitised book": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "annotated": {
        "dtype": "bool",
        "id": null,
        "_type": "Value"
    },
    "Type of resource": {
        "num_classes": 3,
        "names": [
            "Monograph",
            "Serial",
            "Monographic component part"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    },
    "created_at": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "annotator_genre": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```
