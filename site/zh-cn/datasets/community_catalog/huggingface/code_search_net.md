# code_search_net

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/code_search_net)
- [Huggingface](https://huggingface.co/datasets/code_search_net)

## all

Use the following command to load this dataset in TFDS:

```python
ds = tfds.load('huggingface:code_search_net/all')
```

- **Description**:

```
CodeSearchNet corpus contains about 6 million functions from open-source code spanning six programming languages (Go, Java, JavaScript, PHP, Python, and Ruby). The CodeSearchNet Corpus also contains automatically generated query-like natural language for 2 million functions, obtained from mechanically scraping and preprocessing associated function documentation.
```

- **许可**：各种
- **Version**: 1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 100529
`'train'` | 1880853
`'validation'` | 89154

- **特征**：

```json
{
    "repository_name": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "func_path_in_repository": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "func_name": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "whole_func_string": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "language": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "func_code_string": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "func_code_tokens": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "func_documentation_string": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "func_documentation_tokens": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "split_name": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "func_code_url": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## java

Use the following command to load this dataset in TFDS:

```python
ds = tfds.load('huggingface:code_search_net/java')
```

- **Description**:

```
CodeSearchNet corpus contains about 6 million functions from open-source code spanning six programming languages (Go, Java, JavaScript, PHP, Python, and Ruby). The CodeSearchNet Corpus also contains automatically generated query-like natural language for 2 million functions, obtained from mechanically scraping and preprocessing associated function documentation.
```

- **许可**：各种
- **Version**: 1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 26909
`'train'` | 454451
`'validation'` | 15328

- **特征**：

```json
{
    "repository_name": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "func_path_in_repository": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "func_name": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "whole_func_string": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "language": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "func_code_string": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "func_code_tokens": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "func_documentation_string": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "func_documentation_tokens": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "split_name": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "func_code_url": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## go

Use the following command to load this dataset in TFDS:

```python
ds = tfds.load('huggingface:code_search_net/go')
```

- **Description**:

```
CodeSearchNet corpus contains about 6 million functions from open-source code spanning six programming languages (Go, Java, JavaScript, PHP, Python, and Ruby). The CodeSearchNet Corpus also contains automatically generated query-like natural language for 2 million functions, obtained from mechanically scraping and preprocessing associated function documentation.
```

- **许可**：各种
- **Version**: 1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 14291
`'train'` | 317832
`'validation'` | 14242

- **特征**：

```json
{
    "repository_name": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "func_path_in_repository": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "func_name": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "whole_func_string": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "language": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "func_code_string": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "func_code_tokens": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "func_documentation_string": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "func_documentation_tokens": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "split_name": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "func_code_url": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## python

Use the following command to load this dataset in TFDS:

```python
ds = tfds.load('huggingface:code_search_net/python')
```

- **Description**:

```
CodeSearchNet corpus contains about 6 million functions from open-source code spanning six programming languages (Go, Java, JavaScript, PHP, Python, and Ruby). The CodeSearchNet Corpus also contains automatically generated query-like natural language for 2 million functions, obtained from mechanically scraping and preprocessing associated function documentation.
```

- **许可**：各种
- **Version**: 1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 22176
`'train'` | 412178
`'validation'` | 23107

- **特征**：

```json
{
    "repository_name": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "func_path_in_repository": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "func_name": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "whole_func_string": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "language": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "func_code_string": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "func_code_tokens": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "func_documentation_string": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "func_documentation_tokens": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "split_name": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "func_code_url": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## javascript

Use the following command to load this dataset in TFDS:

```python
ds = tfds.load('huggingface:code_search_net/javascript')
```

- **Description**:

```
CodeSearchNet corpus contains about 6 million functions from open-source code spanning six programming languages (Go, Java, JavaScript, PHP, Python, and Ruby). The CodeSearchNet Corpus also contains automatically generated query-like natural language for 2 million functions, obtained from mechanically scraping and preprocessing associated function documentation.
```

- **许可**：各种
- **Version**: 1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 6483
`'train'` | 123889
`'validation'` | 8253

- **特征**：

```json
{
    "repository_name": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "func_path_in_repository": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "func_name": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "whole_func_string": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "language": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "func_code_string": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "func_code_tokens": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "func_documentation_string": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "func_documentation_tokens": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "split_name": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "func_code_url": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## ruby

Use the following command to load this dataset in TFDS:

```python
ds = tfds.load('huggingface:code_search_net/ruby')
```

- **Description**:

```
CodeSearchNet corpus contains about 6 million functions from open-source code spanning six programming languages (Go, Java, JavaScript, PHP, Python, and Ruby). The CodeSearchNet Corpus also contains automatically generated query-like natural language for 2 million functions, obtained from mechanically scraping and preprocessing associated function documentation.
```

- **许可**：各种
- **Version**: 1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 2279
`'train'` | 48791
`'validation'` | 2209

- **特征**：

```json
{
    "repository_name": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "func_path_in_repository": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "func_name": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "whole_func_string": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "language": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "func_code_string": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "func_code_tokens": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "func_documentation_string": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "func_documentation_tokens": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "split_name": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "func_code_url": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## php

Use the following command to load this dataset in TFDS:

```python
ds = tfds.load('huggingface:code_search_net/php')
```

- **Description**:

```
CodeSearchNet corpus contains about 6 million functions from open-source code spanning six programming languages (Go, Java, JavaScript, PHP, Python, and Ruby). The CodeSearchNet Corpus also contains automatically generated query-like natural language for 2 million functions, obtained from mechanically scraping and preprocessing associated function documentation.
```

- **许可**：各种
- **Version**: 1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 28391
`'train'` | 523712
`'validation'` | 26015

- **特征**：

```json
{
    "repository_name": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "func_path_in_repository": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "func_name": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "whole_func_string": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "language": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "func_code_string": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "func_code_tokens": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "func_documentation_string": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "func_documentation_tokens": {
        "feature": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        },
        "length": -1,
        "id": null,
        "_type": "Sequence"
    },
    "split_name": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "func_code_url": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```
