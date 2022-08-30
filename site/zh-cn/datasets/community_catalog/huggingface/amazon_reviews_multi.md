# amazon_reviews_multi

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/amazon_reviews_multi)
- [Huggingface](https://huggingface.co/datasets/amazon_reviews_multi)

## all_languages

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:amazon_reviews_multi/all_languages')
```

- **说明**：

```
We provide an Amazon product reviews dataset for multilingual text classification. The dataset contains reviews in English, Japanese, German, French, Chinese and Spanish, collected between November 1, 2015 and November 1, 2019. Each record in the dataset contains the review text, the review title, the star rating, an anonymized reviewer ID, an anonymized product ID and the coarse-grained product category (e.g. ‘books’, ‘appliances’, etc.) The corpus is balanced across stars, so each star rating constitutes 20% of the reviews in each language.

For each language, there are 200,000, 5,000 and 5,000 reviews in the training, development and test sets respectively. The maximum number of reviews per reviewer is 20 and the maximum number of reviews per product is 20. All reviews are truncated after 2,000 characters, and all reviews are at least 20 characters long.

Note that the language of a review does not necessarily match the language of its marketplace (e.g. reviews from amazon.de are primarily written in German, but could also be written in English, etc.). For this reason, we applied a language detection algorithm based on the work in Bojanowski et al. (2017) to determine the language of the review text and we removed reviews that were not written in the expected language.
```

- **许可**：访问多语种 Amazon 评论语料库（“评论语料库”），即您同意评论语料库是受 Amazon.com 使用条件 (https://www.amazon.com/gp/help/customer/display.html/ref=footer_cou?ie=UTF8&amp;nodeId=508088) 约束的 Amazon 服务并且您同意受其约束，同时附带以下条件：

除了根据《使用条件》授予的许可权利外，Amazon 或其内容提供商还授予您有限、非排他性、不可转让、不可再许可、可撤销的许可，允许您出于学术研究目的访问和使用评论语料库。您不得转售、重新发布或者将评论语料库或其内容用于任何商业用途，包括将评论语料库用于商业研究，例如与资助或咨询合同、实习或者有偿提供或交付给营利性组织的其他关系相关的研究。您不得 (a) 将评论语料库中的内容与任何个人信息（包括 Amazon 客户账户）链接或关联，或 (b) 尝试确定评论语料库中任何内容的作者的身份。如果您违反上述任何条件，您访问和使用评论语料库的许可将自动终止，但不影响 Amazon 可能拥有的任何其他权利或补救措施。

- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 30000
`'train'` | 1200000
`'validation'` | 30000

- **特征**：

```json
{
    "review_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "reviewer_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "stars": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "review_body": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "language": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_category": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## de

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:amazon_reviews_multi/de')
```

- **说明**：

```
We provide an Amazon product reviews dataset for multilingual text classification. The dataset contains reviews in English, Japanese, German, French, Chinese and Spanish, collected between November 1, 2015 and November 1, 2019. Each record in the dataset contains the review text, the review title, the star rating, an anonymized reviewer ID, an anonymized product ID and the coarse-grained product category (e.g. ‘books’, ‘appliances’, etc.) The corpus is balanced across stars, so each star rating constitutes 20% of the reviews in each language.

For each language, there are 200,000, 5,000 and 5,000 reviews in the training, development and test sets respectively. The maximum number of reviews per reviewer is 20 and the maximum number of reviews per product is 20. All reviews are truncated after 2,000 characters, and all reviews are at least 20 characters long.

Note that the language of a review does not necessarily match the language of its marketplace (e.g. reviews from amazon.de are primarily written in German, but could also be written in English, etc.). For this reason, we applied a language detection algorithm based on the work in Bojanowski et al. (2017) to determine the language of the review text and we removed reviews that were not written in the expected language.
```

- **许可**：访问多语种 Amazon 评论语料库（“评论语料库”），即您同意评论语料库是受 Amazon.com 使用条件 (https://www.amazon.com/gp/help/customer/display.html/ref=footer_cou?ie=UTF8&amp;nodeId=508088) 约束的 Amazon 服务并且您同意受其约束，同时附带以下条件：

除了根据《使用条件》授予的许可权利外，Amazon 或其内容提供商还授予您有限、非排他性、不可转让、不可再许可、可撤销的许可，允许您出于学术研究目的访问和使用评论语料库。您不得转售、重新发布或者将评论语料库或其内容用于任何商业用途，包括将评论语料库用于商业研究，例如与资助或咨询合同、实习或者有偿提供或交付给营利性组织的其他关系相关的研究。您不得 (a) 将评论语料库中的内容与任何个人信息（包括 Amazon 客户账户）链接或关联，或 (b) 尝试确定评论语料库中任何内容的作者的身份。如果您违反上述任何条件，您访问和使用评论语料库的许可将自动终止，但不影响 Amazon 可能拥有的任何其他权利或补救措施。

- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 5000
`'train'` | 200000
`'validation'` | 5000

- **特征**：

```json
{
    "review_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "reviewer_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "stars": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "review_body": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "language": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_category": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## en

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:amazon_reviews_multi/en')
```

- **说明**：

```
We provide an Amazon product reviews dataset for multilingual text classification. The dataset contains reviews in English, Japanese, German, French, Chinese and Spanish, collected between November 1, 2015 and November 1, 2019. Each record in the dataset contains the review text, the review title, the star rating, an anonymized reviewer ID, an anonymized product ID and the coarse-grained product category (e.g. ‘books’, ‘appliances’, etc.) The corpus is balanced across stars, so each star rating constitutes 20% of the reviews in each language.

For each language, there are 200,000, 5,000 and 5,000 reviews in the training, development and test sets respectively. The maximum number of reviews per reviewer is 20 and the maximum number of reviews per product is 20. All reviews are truncated after 2,000 characters, and all reviews are at least 20 characters long.

Note that the language of a review does not necessarily match the language of its marketplace (e.g. reviews from amazon.de are primarily written in German, but could also be written in English, etc.). For this reason, we applied a language detection algorithm based on the work in Bojanowski et al. (2017) to determine the language of the review text and we removed reviews that were not written in the expected language.
```

- **许可**：访问多语种 Amazon 评论语料库（“评论语料库”），即您同意评论语料库是受 Amazon.com 使用条件 (https://www.amazon.com/gp/help/customer/display.html/ref=footer_cou?ie=UTF8&amp;nodeId=508088) 约束的 Amazon 服务并且您同意受其约束，同时附带以下条件：

除了根据《使用条件》授予的许可权利外，Amazon 或其内容提供商还授予您有限、非排他性、不可转让、不可再许可、可撤销的许可，允许您出于学术研究目的访问和使用评论语料库。您不得转售、重新发布或者将评论语料库或其内容用于任何商业用途，包括将评论语料库用于商业研究，例如与资助或咨询合同、实习或者有偿提供或交付给营利性组织的其他关系相关的研究。您不得 (a) 将评论语料库中的内容与任何个人信息（包括 Amazon 客户账户）链接或关联，或 (b) 尝试确定评论语料库中任何内容的作者的身份。如果您违反上述任何条件，您访问和使用评论语料库的许可将自动终止，但不影响 Amazon 可能拥有的任何其他权利或补救措施。

- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 5000
`'train'` | 200000
`'validation'` | 5000

- **特征**：

```json
{
    "review_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "reviewer_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "stars": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "review_body": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "language": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_category": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## es

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:amazon_reviews_multi/es')
```

- **说明**：

```
We provide an Amazon product reviews dataset for multilingual text classification. The dataset contains reviews in English, Japanese, German, French, Chinese and Spanish, collected between November 1, 2015 and November 1, 2019. Each record in the dataset contains the review text, the review title, the star rating, an anonymized reviewer ID, an anonymized product ID and the coarse-grained product category (e.g. ‘books’, ‘appliances’, etc.) The corpus is balanced across stars, so each star rating constitutes 20% of the reviews in each language.

For each language, there are 200,000, 5,000 and 5,000 reviews in the training, development and test sets respectively. The maximum number of reviews per reviewer is 20 and the maximum number of reviews per product is 20. All reviews are truncated after 2,000 characters, and all reviews are at least 20 characters long.

Note that the language of a review does not necessarily match the language of its marketplace (e.g. reviews from amazon.de are primarily written in German, but could also be written in English, etc.). For this reason, we applied a language detection algorithm based on the work in Bojanowski et al. (2017) to determine the language of the review text and we removed reviews that were not written in the expected language.
```

- **许可**：访问多语种 Amazon 评论语料库（“评论语料库”），即您同意评论语料库是受 Amazon.com 使用条件 (https://www.amazon.com/gp/help/customer/display.html/ref=footer_cou?ie=UTF8&amp;nodeId=508088) 约束的 Amazon 服务并且您同意受其约束，同时附带以下条件：

除了根据《使用条件》授予的许可权利外，Amazon 或其内容提供商还授予您有限、非排他性、不可转让、不可再许可、可撤销的许可，允许您出于学术研究目的访问和使用评论语料库。您不得转售、重新发布或者将评论语料库或其内容用于任何商业用途，包括将评论语料库用于商业研究，例如与资助或咨询合同、实习或者有偿提供或交付给营利性组织的其他关系相关的研究。您不得 (a) 将评论语料库中的内容与任何个人信息（包括 Amazon 客户账户）链接或关联，或 (b) 尝试确定评论语料库中任何内容的作者的身份。如果您违反上述任何条件，您访问和使用评论语料库的许可将自动终止，但不影响 Amazon 可能拥有的任何其他权利或补救措施。

- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 5000
`'train'` | 200000
`'validation'` | 5000

- **特征**：

```json
{
    "review_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "reviewer_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "stars": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "review_body": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "language": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_category": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## fr

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:amazon_reviews_multi/fr')
```

- **说明**：

```
We provide an Amazon product reviews dataset for multilingual text classification. The dataset contains reviews in English, Japanese, German, French, Chinese and Spanish, collected between November 1, 2015 and November 1, 2019. Each record in the dataset contains the review text, the review title, the star rating, an anonymized reviewer ID, an anonymized product ID and the coarse-grained product category (e.g. ‘books’, ‘appliances’, etc.) The corpus is balanced across stars, so each star rating constitutes 20% of the reviews in each language.

For each language, there are 200,000, 5,000 and 5,000 reviews in the training, development and test sets respectively. The maximum number of reviews per reviewer is 20 and the maximum number of reviews per product is 20. All reviews are truncated after 2,000 characters, and all reviews are at least 20 characters long.

Note that the language of a review does not necessarily match the language of its marketplace (e.g. reviews from amazon.de are primarily written in German, but could also be written in English, etc.). For this reason, we applied a language detection algorithm based on the work in Bojanowski et al. (2017) to determine the language of the review text and we removed reviews that were not written in the expected language.
```

- **许可**：访问多语种 Amazon 评论语料库（“评论语料库”），即您同意评论语料库是受 Amazon.com 使用条件 (https://www.amazon.com/gp/help/customer/display.html/ref=footer_cou?ie=UTF8&amp;nodeId=508088) 约束的 Amazon 服务并且您同意受其约束，同时附带以下条件：

除了根据《使用条件》授予的许可权利外，Amazon 或其内容提供商还授予您有限、非排他性、不可转让、不可再许可、可撤销的许可，允许您出于学术研究目的访问和使用评论语料库。您不得转售、重新发布或者将评论语料库或其内容用于任何商业用途，包括将评论语料库用于商业研究，例如与资助或咨询合同、实习或者有偿提供或交付给营利性组织的其他关系相关的研究。您不得 (a) 将评论语料库中的内容与任何个人信息（包括 Amazon 客户账户）链接或关联，或 (b) 尝试确定评论语料库中任何内容的作者的身份。如果您违反上述任何条件，您访问和使用评论语料库的许可将自动终止，但不影响 Amazon 可能拥有的任何其他权利或补救措施。

- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 5000
`'train'` | 200000
`'validation'` | 5000

- **特征**：

```json
{
    "review_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "reviewer_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "stars": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "review_body": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "language": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_category": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## ja

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:amazon_reviews_multi/ja')
```

- **说明**：

```
We provide an Amazon product reviews dataset for multilingual text classification. The dataset contains reviews in English, Japanese, German, French, Chinese and Spanish, collected between November 1, 2015 and November 1, 2019. Each record in the dataset contains the review text, the review title, the star rating, an anonymized reviewer ID, an anonymized product ID and the coarse-grained product category (e.g. ‘books’, ‘appliances’, etc.) The corpus is balanced across stars, so each star rating constitutes 20% of the reviews in each language.

For each language, there are 200,000, 5,000 and 5,000 reviews in the training, development and test sets respectively. The maximum number of reviews per reviewer is 20 and the maximum number of reviews per product is 20. All reviews are truncated after 2,000 characters, and all reviews are at least 20 characters long.

Note that the language of a review does not necessarily match the language of its marketplace (e.g. reviews from amazon.de are primarily written in German, but could also be written in English, etc.). For this reason, we applied a language detection algorithm based on the work in Bojanowski et al. (2017) to determine the language of the review text and we removed reviews that were not written in the expected language.
```

- **许可**：访问多语种 Amazon 评论语料库（“评论语料库”），即您同意评论语料库是受 Amazon.com 使用条件 (https://www.amazon.com/gp/help/customer/display.html/ref=footer_cou?ie=UTF8&amp;nodeId=508088) 约束的 Amazon 服务并且您同意受其约束，同时附带以下条件：

除了根据《使用条件》授予的许可权利外，Amazon 或其内容提供商还授予您有限、非排他性、不可转让、不可再许可、可撤销的许可，允许您出于学术研究目的访问和使用评论语料库。您不得转售、重新发布或者将评论语料库或其内容用于任何商业用途，包括将评论语料库用于商业研究，例如与资助或咨询合同、实习或者有偿提供或交付给营利性组织的其他关系相关的研究。您不得 (a) 将评论语料库中的内容与任何个人信息（包括 Amazon 客户账户）链接或关联，或 (b) 尝试确定评论语料库中任何内容的作者的身份。如果您违反上述任何条件，您访问和使用评论语料库的许可将自动终止，但不影响 Amazon 可能拥有的任何其他权利或补救措施。

- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 5000
`'train'` | 200000
`'validation'` | 5000

- **特征**：

```json
{
    "review_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "reviewer_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "stars": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "review_body": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "language": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_category": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## zh

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:amazon_reviews_multi/zh')
```

- **说明**：

```
We provide an Amazon product reviews dataset for multilingual text classification. The dataset contains reviews in English, Japanese, German, French, Chinese and Spanish, collected between November 1, 2015 and November 1, 2019. Each record in the dataset contains the review text, the review title, the star rating, an anonymized reviewer ID, an anonymized product ID and the coarse-grained product category (e.g. ‘books’, ‘appliances’, etc.) The corpus is balanced across stars, so each star rating constitutes 20% of the reviews in each language.

For each language, there are 200,000, 5,000 and 5,000 reviews in the training, development and test sets respectively. The maximum number of reviews per reviewer is 20 and the maximum number of reviews per product is 20. All reviews are truncated after 2,000 characters, and all reviews are at least 20 characters long.

Note that the language of a review does not necessarily match the language of its marketplace (e.g. reviews from amazon.de are primarily written in German, but could also be written in English, etc.). For this reason, we applied a language detection algorithm based on the work in Bojanowski et al. (2017) to determine the language of the review text and we removed reviews that were not written in the expected language.
```

- **许可**：访问多语种 Amazon 评论语料库（“评论语料库”），即您同意评论语料库是受 Amazon.com 使用条件 (https://www.amazon.com/gp/help/customer/display.html/ref=footer_cou?ie=UTF8&amp;nodeId=508088) 约束的 Amazon 服务并且您同意受其约束，同时附带以下条件：

除了根据《使用条件》授予的许可权利外，Amazon 或其内容提供商还授予您有限、非排他性、不可转让、不可再许可、可撤销的许可，允许您出于学术研究目的访问和使用评论语料库。您不得转售、重新发布或者将评论语料库或其内容用于任何商业用途，包括将评论语料库用于商业研究，例如与资助或咨询合同、实习或者有偿提供或交付给营利性组织的其他关系相关的研究。您不得 (a) 将评论语料库中的内容与任何个人信息（包括 Amazon 客户账户）链接或关联，或 (b) 尝试确定评论语料库中任何内容的作者的身份。如果您违反上述任何条件，您访问和使用评论语料库的许可将自动终止，但不影响 Amazon 可能拥有的任何其他权利或补救措施。

- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 5000
`'train'` | 200000
`'validation'` | 5000

- **特征**：

```json
{
    "review_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "reviewer_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "stars": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "review_body": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "language": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_category": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```
