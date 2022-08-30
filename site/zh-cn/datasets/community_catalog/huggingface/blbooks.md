# blbooks

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/blbooks)
- [Huggingface](https://huggingface.co/datasets/blbooks)

## all

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:blbooks/all')
```

- **说明**：

```
A dataset comprising of text created by OCR from the 49,455 digitised books, equating to 65,227 volumes (25+ million pages), published between c. 1510 - c. 1900.
The books cover a wide range of subject areas including philosophy, history, poetry and literature.
```

- **许可**：无已知许可
- **版本**：1.0.2
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 14011953

- **特征**：

```json
{
    "record_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "date": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "raw_date": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "place": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "empty_pg": {
        "dtype": "bool",
        "id": null,
        "_type": "Value"
    },
    "text": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "pg": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "mean_wc_ocr": {
        "dtype": "float32",
        "id": null,
        "_type": "Value"
    },
    "std_wc_ocr": {
        "dtype": "float64",
        "id": null,
        "_type": "Value"
    },
    "name": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "all_names": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Publisher": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Country of publication 1": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "all Countries of publication": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Physical description": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Language_1": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Language_2": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Language_3": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Language_4": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "multi_language": {
        "dtype": "bool",
        "id": null,
        "_type": "Value"
    }
}
```

## 1800s

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:blbooks/1800s')
```

- **说明**：

```
A dataset comprising of text created by OCR from the 49,455 digitised books, equating to 65,227 volumes (25+ million pages), published between c. 1510 - c. 1900.
The books cover a wide range of subject areas including philosophy, history, poetry and literature.
```

- **许可**：无已知许可
- **版本**：1.0.2
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 13781747

- **特征**：

```json
{
    "record_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "date": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "raw_date": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "place": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "empty_pg": {
        "dtype": "bool",
        "id": null,
        "_type": "Value"
    },
    "text": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "pg": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "mean_wc_ocr": {
        "dtype": "float32",
        "id": null,
        "_type": "Value"
    },
    "std_wc_ocr": {
        "dtype": "float64",
        "id": null,
        "_type": "Value"
    },
    "name": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "all_names": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Publisher": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Country of publication 1": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "all Countries of publication": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Physical description": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Language_1": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Language_2": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Language_3": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Language_4": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "multi_language": {
        "dtype": "bool",
        "id": null,
        "_type": "Value"
    }
}
```

## 1700s

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:blbooks/1700s')
```

- **说明**：

```
A dataset comprising of text created by OCR from the 49,455 digitised books, equating to 65,227 volumes (25+ million pages), published between c. 1510 - c. 1900.
The books cover a wide range of subject areas including philosophy, history, poetry and literature.
```

- **许可**：无已知许可
- **版本**：1.0.2
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 178224

- **特征**：

```json
{
    "record_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "date": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "raw_date": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "place": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "empty_pg": {
        "dtype": "bool",
        "id": null,
        "_type": "Value"
    },
    "text": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "pg": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "mean_wc_ocr": {
        "dtype": "float32",
        "id": null,
        "_type": "Value"
    },
    "std_wc_ocr": {
        "dtype": "float64",
        "id": null,
        "_type": "Value"
    },
    "name": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "all_names": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Publisher": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Country of publication 1": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "all Countries of publication": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Physical description": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Language_1": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Language_2": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Language_3": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Language_4": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "multi_language": {
        "dtype": "bool",
        "id": null,
        "_type": "Value"
    }
}
```

## 1510_1699

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:blbooks/1510_1699')
```

- **说明**：

```
A dataset comprising of text created by OCR from the 49,455 digitised books, equating to 65,227 volumes (25+ million pages), published between c. 1510 - c. 1900.
The books cover a wide range of subject areas including philosophy, history, poetry and literature.
```

- **许可**：无已知许可
- **版本**：1.0.2
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 51982

- **特征**：

```json
{
    "record_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "date": {
        "dtype": "timestamp[s]",
        "id": null,
        "_type": "Value"
    },
    "raw_date": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "place": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "empty_pg": {
        "dtype": "bool",
        "id": null,
        "_type": "Value"
    },
    "text": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "pg": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "mean_wc_ocr": {
        "dtype": "float32",
        "id": null,
        "_type": "Value"
    },
    "std_wc_ocr": {
        "dtype": "float64",
        "id": null,
        "_type": "Value"
    },
    "name": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "all_names": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Publisher": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Country of publication 1": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "all Countries of publication": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Physical description": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Language_1": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Language_2": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Language_3": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Language_4": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "multi_language": {
        "dtype": "bool",
        "id": null,
        "_type": "Value"
    }
}
```

## 1500_1899

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:blbooks/1500_1899')
```

- **说明**：

```
A dataset comprising of text created by OCR from the 49,455 digitised books, equating to 65,227 volumes (25+ million pages), published between c. 1510 - c. 1900.
The books cover a wide range of subject areas including philosophy, history, poetry and literature.
```

- **许可**：无已知许可
- **版本**：1.0.2
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 14011953

- **特征**：

```json
{
    "record_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "date": {
        "dtype": "timestamp[s]",
        "id": null,
        "_type": "Value"
    },
    "raw_date": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "place": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "empty_pg": {
        "dtype": "bool",
        "id": null,
        "_type": "Value"
    },
    "text": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "pg": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "mean_wc_ocr": {
        "dtype": "float32",
        "id": null,
        "_type": "Value"
    },
    "std_wc_ocr": {
        "dtype": "float64",
        "id": null,
        "_type": "Value"
    },
    "name": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "all_names": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Publisher": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Country of publication 1": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "all Countries of publication": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Physical description": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Language_1": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Language_2": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Language_3": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Language_4": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "multi_language": {
        "dtype": "bool",
        "id": null,
        "_type": "Value"
    }
}
```

## 1800_1899

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:blbooks/1800_1899')
```

- **说明**：

```
A dataset comprising of text created by OCR from the 49,455 digitised books, equating to 65,227 volumes (25+ million pages), published between c. 1510 - c. 1900.
The books cover a wide range of subject areas including philosophy, history, poetry and literature.
```

- **许可**：无已知许可
- **版本**：1.0.2
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 13781747

- **特征**：

```json
{
    "record_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "date": {
        "dtype": "timestamp[s]",
        "id": null,
        "_type": "Value"
    },
    "raw_date": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "place": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "empty_pg": {
        "dtype": "bool",
        "id": null,
        "_type": "Value"
    },
    "text": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "pg": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "mean_wc_ocr": {
        "dtype": "float32",
        "id": null,
        "_type": "Value"
    },
    "std_wc_ocr": {
        "dtype": "float64",
        "id": null,
        "_type": "Value"
    },
    "name": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "all_names": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Publisher": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Country of publication 1": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "all Countries of publication": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Physical description": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Language_1": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Language_2": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Language_3": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Language_4": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "multi_language": {
        "dtype": "bool",
        "id": null,
        "_type": "Value"
    }
}
```

## 1700_1799

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:blbooks/1700_1799')
```

- **说明**：

```
A dataset comprising of text created by OCR from the 49,455 digitised books, equating to 65,227 volumes (25+ million pages), published between c. 1510 - c. 1900.
The books cover a wide range of subject areas including philosophy, history, poetry and literature.
```

- **许可**：无已知许可
- **版本**：1.0.2
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 178224

- **特征**：

```json
{
    "record_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "date": {
        "dtype": "timestamp[s]",
        "id": null,
        "_type": "Value"
    },
    "raw_date": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "place": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "empty_pg": {
        "dtype": "bool",
        "id": null,
        "_type": "Value"
    },
    "text": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "pg": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "mean_wc_ocr": {
        "dtype": "float32",
        "id": null,
        "_type": "Value"
    },
    "std_wc_ocr": {
        "dtype": "float64",
        "id": null,
        "_type": "Value"
    },
    "name": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "all_names": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Publisher": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Country of publication 1": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "all Countries of publication": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Physical description": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Language_1": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Language_2": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Language_3": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "Language_4": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "multi_language": {
        "dtype": "bool",
        "id": null,
        "_type": "Value"
    }
}
```
