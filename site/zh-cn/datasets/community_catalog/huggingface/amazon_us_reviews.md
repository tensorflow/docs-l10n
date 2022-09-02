# amazon_us_reviews

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/amazon_us_reviews)
- [Huggingface](https://huggingface.co/datasets/amazon_us_reviews)

## Books_v1_01

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:amazon_us_reviews/Books_v1_01')
```

- **说明**：

```
Amazon Customer Reviews (a.k.a. Product Reviews) is one of Amazons iconic products. In a period of over two decades since the first review in 1995, millions of Amazon customers have contributed over a hundred million reviews to express opinions and describe their experiences regarding products on the Amazon.com website. This makes Amazon Customer Reviews a rich source of information for academic researchers in the fields of Natural Language Processing (NLP), Information Retrieval (IR), and Machine Learning (ML), amongst others. Accordingly, we are releasing this data to further research in multiple disciplines related to understanding customer product experiences. Specifically, this dataset was constructed to represent a sample of customer evaluations and opinions, variation in the perception of a product across geographical regions, and promotional intent or bias in reviews.

Over 130+ million customer reviews are available to researchers as part of this release. The data is available in TSV files in the amazon-reviews-pds S3 bucket in AWS US East Region. Each line in the data files corresponds to an individual review (tab delimited, with no quote and escape characters).

Each Dataset contains the following columns:

- marketplace: 2 letter country code of the marketplace where the review was written.
- customer_id: Random identifier that can be used to aggregate reviews written by a single author.
- review_id: The unique ID of the review.
- product_id: The unique Product ID the review pertains to. In the multilingual dataset the reviews for the same product in different countries can be grouped by the same product_id.
- product_parent: Random identifier that can be used to aggregate reviews for the same product.
- product_title: Title of the product.
- product_category: Broad product category that can be used to group reviews (also used to group the dataset into coherent parts).
- star_rating: The 1-5 star rating of the review.
- helpful_votes: Number of helpful votes.
- total_votes: Number of total votes the review received.
- vine: Review was written as part of the Vine program.
- verified_purchase: The review is on a verified purchase.
- review_headline: The title of the review.
- review_body: The review text.
- review_date: The date the review was written.
```

- **许可**：无已知许可
- **版本**：0.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 6106719

- **特征**：

```json
{
    "marketplace": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "customer_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
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
    "product_parent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_category": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "star_rating": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "helpful_votes": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "total_votes": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "vine": {
        "num_classes": 2,
        "names": [
            "N",
            "Y"
        ],
        "id": null,
        "_type": "ClassLabel"
    },
    "verified_purchase": {
        "num_classes": 2,
        "names": [
            "N",
            "Y"
        ],
        "id": null,
        "_type": "ClassLabel"
    },
    "review_headline": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_body": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_date": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## Watches_v1_00

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:amazon_us_reviews/Watches_v1_00')
```

- **说明**：

```
Amazon Customer Reviews (a.k.a. Product Reviews) is one of Amazons iconic products. In a period of over two decades since the first review in 1995, millions of Amazon customers have contributed over a hundred million reviews to express opinions and describe their experiences regarding products on the Amazon.com website. This makes Amazon Customer Reviews a rich source of information for academic researchers in the fields of Natural Language Processing (NLP), Information Retrieval (IR), and Machine Learning (ML), amongst others. Accordingly, we are releasing this data to further research in multiple disciplines related to understanding customer product experiences. Specifically, this dataset was constructed to represent a sample of customer evaluations and opinions, variation in the perception of a product across geographical regions, and promotional intent or bias in reviews.

Over 130+ million customer reviews are available to researchers as part of this release. The data is available in TSV files in the amazon-reviews-pds S3 bucket in AWS US East Region. Each line in the data files corresponds to an individual review (tab delimited, with no quote and escape characters).

Each Dataset contains the following columns:

- marketplace: 2 letter country code of the marketplace where the review was written.
- customer_id: Random identifier that can be used to aggregate reviews written by a single author.
- review_id: The unique ID of the review.
- product_id: The unique Product ID the review pertains to. In the multilingual dataset the reviews for the same product in different countries can be grouped by the same product_id.
- product_parent: Random identifier that can be used to aggregate reviews for the same product.
- product_title: Title of the product.
- product_category: Broad product category that can be used to group reviews (also used to group the dataset into coherent parts).
- star_rating: The 1-5 star rating of the review.
- helpful_votes: Number of helpful votes.
- total_votes: Number of total votes the review received.
- vine: Review was written as part of the Vine program.
- verified_purchase: The review is on a verified purchase.
- review_headline: The title of the review.
- review_body: The review text.
- review_date: The date the review was written.
```

- **许可**：无已知许可
- **版本**：0.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 960872

- **特征**：

```json
{
    "marketplace": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "customer_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
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
    "product_parent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_category": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "star_rating": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "helpful_votes": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "total_votes": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "vine": {
        "num_classes": 2,
        "names": [
            "N",
            "Y"
        ],
        "id": null,
        "_type": "ClassLabel"
    },
    "verified_purchase": {
        "num_classes": 2,
        "names": [
            "N",
            "Y"
        ],
        "id": null,
        "_type": "ClassLabel"
    },
    "review_headline": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_body": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_date": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## Personal_Care_Appliances_v1_00

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:amazon_us_reviews/Personal_Care_Appliances_v1_00')
```

- **说明**：

```
Amazon Customer Reviews (a.k.a. Product Reviews) is one of Amazons iconic products. In a period of over two decades since the first review in 1995, millions of Amazon customers have contributed over a hundred million reviews to express opinions and describe their experiences regarding products on the Amazon.com website. This makes Amazon Customer Reviews a rich source of information for academic researchers in the fields of Natural Language Processing (NLP), Information Retrieval (IR), and Machine Learning (ML), amongst others. Accordingly, we are releasing this data to further research in multiple disciplines related to understanding customer product experiences. Specifically, this dataset was constructed to represent a sample of customer evaluations and opinions, variation in the perception of a product across geographical regions, and promotional intent or bias in reviews.

Over 130+ million customer reviews are available to researchers as part of this release. The data is available in TSV files in the amazon-reviews-pds S3 bucket in AWS US East Region. Each line in the data files corresponds to an individual review (tab delimited, with no quote and escape characters).

Each Dataset contains the following columns:

- marketplace: 2 letter country code of the marketplace where the review was written.
- customer_id: Random identifier that can be used to aggregate reviews written by a single author.
- review_id: The unique ID of the review.
- product_id: The unique Product ID the review pertains to. In the multilingual dataset the reviews for the same product in different countries can be grouped by the same product_id.
- product_parent: Random identifier that can be used to aggregate reviews for the same product.
- product_title: Title of the product.
- product_category: Broad product category that can be used to group reviews (also used to group the dataset into coherent parts).
- star_rating: The 1-5 star rating of the review.
- helpful_votes: Number of helpful votes.
- total_votes: Number of total votes the review received.
- vine: Review was written as part of the Vine program.
- verified_purchase: The review is on a verified purchase.
- review_headline: The title of the review.
- review_body: The review text.
- review_date: The date the review was written.
```

- **许可**：无已知许可
- **版本**：0.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 85981

- **特征**：

```json
{
    "marketplace": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "customer_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
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
    "product_parent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_category": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "star_rating": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "helpful_votes": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "total_votes": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "vine": {
        "num_classes": 2,
        "names": [
            "N",
            "Y"
        ],
        "id": null,
        "_type": "ClassLabel"
    },
    "verified_purchase": {
        "num_classes": 2,
        "names": [
            "N",
            "Y"
        ],
        "id": null,
        "_type": "ClassLabel"
    },
    "review_headline": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_body": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_date": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## Mobile_Electronics_v1_00

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:amazon_us_reviews/Mobile_Electronics_v1_00')
```

- **说明**：

```
Amazon Customer Reviews (a.k.a. Product Reviews) is one of Amazons iconic products. In a period of over two decades since the first review in 1995, millions of Amazon customers have contributed over a hundred million reviews to express opinions and describe their experiences regarding products on the Amazon.com website. This makes Amazon Customer Reviews a rich source of information for academic researchers in the fields of Natural Language Processing (NLP), Information Retrieval (IR), and Machine Learning (ML), amongst others. Accordingly, we are releasing this data to further research in multiple disciplines related to understanding customer product experiences. Specifically, this dataset was constructed to represent a sample of customer evaluations and opinions, variation in the perception of a product across geographical regions, and promotional intent or bias in reviews.

Over 130+ million customer reviews are available to researchers as part of this release. The data is available in TSV files in the amazon-reviews-pds S3 bucket in AWS US East Region. Each line in the data files corresponds to an individual review (tab delimited, with no quote and escape characters).

Each Dataset contains the following columns:

- marketplace: 2 letter country code of the marketplace where the review was written.
- customer_id: Random identifier that can be used to aggregate reviews written by a single author.
- review_id: The unique ID of the review.
- product_id: The unique Product ID the review pertains to. In the multilingual dataset the reviews for the same product in different countries can be grouped by the same product_id.
- product_parent: Random identifier that can be used to aggregate reviews for the same product.
- product_title: Title of the product.
- product_category: Broad product category that can be used to group reviews (also used to group the dataset into coherent parts).
- star_rating: The 1-5 star rating of the review.
- helpful_votes: Number of helpful votes.
- total_votes: Number of total votes the review received.
- vine: Review was written as part of the Vine program.
- verified_purchase: The review is on a verified purchase.
- review_headline: The title of the review.
- review_body: The review text.
- review_date: The date the review was written.
```

- **许可**：无已知许可
- **版本**：0.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 104975

- **特征**：

```json
{
    "marketplace": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "customer_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
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
    "product_parent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_category": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "star_rating": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "helpful_votes": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "total_votes": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "vine": {
        "num_classes": 2,
        "names": [
            "N",
            "Y"
        ],
        "id": null,
        "_type": "ClassLabel"
    },
    "verified_purchase": {
        "num_classes": 2,
        "names": [
            "N",
            "Y"
        ],
        "id": null,
        "_type": "ClassLabel"
    },
    "review_headline": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_body": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_date": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## Digital_Video_Games_v1_00

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:amazon_us_reviews/Digital_Video_Games_v1_00')
```

- **说明**：

```
Amazon Customer Reviews (a.k.a. Product Reviews) is one of Amazons iconic products. In a period of over two decades since the first review in 1995, millions of Amazon customers have contributed over a hundred million reviews to express opinions and describe their experiences regarding products on the Amazon.com website. This makes Amazon Customer Reviews a rich source of information for academic researchers in the fields of Natural Language Processing (NLP), Information Retrieval (IR), and Machine Learning (ML), amongst others. Accordingly, we are releasing this data to further research in multiple disciplines related to understanding customer product experiences. Specifically, this dataset was constructed to represent a sample of customer evaluations and opinions, variation in the perception of a product across geographical regions, and promotional intent or bias in reviews.

Over 130+ million customer reviews are available to researchers as part of this release. The data is available in TSV files in the amazon-reviews-pds S3 bucket in AWS US East Region. Each line in the data files corresponds to an individual review (tab delimited, with no quote and escape characters).

Each Dataset contains the following columns:

- marketplace: 2 letter country code of the marketplace where the review was written.
- customer_id: Random identifier that can be used to aggregate reviews written by a single author.
- review_id: The unique ID of the review.
- product_id: The unique Product ID the review pertains to. In the multilingual dataset the reviews for the same product in different countries can be grouped by the same product_id.
- product_parent: Random identifier that can be used to aggregate reviews for the same product.
- product_title: Title of the product.
- product_category: Broad product category that can be used to group reviews (also used to group the dataset into coherent parts).
- star_rating: The 1-5 star rating of the review.
- helpful_votes: Number of helpful votes.
- total_votes: Number of total votes the review received.
- vine: Review was written as part of the Vine program.
- verified_purchase: The review is on a verified purchase.
- review_headline: The title of the review.
- review_body: The review text.
- review_date: The date the review was written.
```

- **许可**：无已知许可
- **版本**：0.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 145431

- **特征**：

```json
{
    "marketplace": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "customer_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
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
    "product_parent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_category": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "star_rating": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "helpful_votes": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "total_votes": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "vine": {
        "num_classes": 2,
        "names": [
            "N",
            "Y"
        ],
        "id": null,
        "_type": "ClassLabel"
    },
    "verified_purchase": {
        "num_classes": 2,
        "names": [
            "N",
            "Y"
        ],
        "id": null,
        "_type": "ClassLabel"
    },
    "review_headline": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_body": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_date": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## Digital_Software_v1_00

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:amazon_us_reviews/Digital_Software_v1_00')
```

- **说明**：

```
Amazon Customer Reviews (a.k.a. Product Reviews) is one of Amazons iconic products. In a period of over two decades since the first review in 1995, millions of Amazon customers have contributed over a hundred million reviews to express opinions and describe their experiences regarding products on the Amazon.com website. This makes Amazon Customer Reviews a rich source of information for academic researchers in the fields of Natural Language Processing (NLP), Information Retrieval (IR), and Machine Learning (ML), amongst others. Accordingly, we are releasing this data to further research in multiple disciplines related to understanding customer product experiences. Specifically, this dataset was constructed to represent a sample of customer evaluations and opinions, variation in the perception of a product across geographical regions, and promotional intent or bias in reviews.

Over 130+ million customer reviews are available to researchers as part of this release. The data is available in TSV files in the amazon-reviews-pds S3 bucket in AWS US East Region. Each line in the data files corresponds to an individual review (tab delimited, with no quote and escape characters).

Each Dataset contains the following columns:

- marketplace: 2 letter country code of the marketplace where the review was written.
- customer_id: Random identifier that can be used to aggregate reviews written by a single author.
- review_id: The unique ID of the review.
- product_id: The unique Product ID the review pertains to. In the multilingual dataset the reviews for the same product in different countries can be grouped by the same product_id.
- product_parent: Random identifier that can be used to aggregate reviews for the same product.
- product_title: Title of the product.
- product_category: Broad product category that can be used to group reviews (also used to group the dataset into coherent parts).
- star_rating: The 1-5 star rating of the review.
- helpful_votes: Number of helpful votes.
- total_votes: Number of total votes the review received.
- vine: Review was written as part of the Vine program.
- verified_purchase: The review is on a verified purchase.
- review_headline: The title of the review.
- review_body: The review text.
- review_date: The date the review was written.
```

- **许可**：无已知许可
- **版本**：0.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 102084

- **特征**：

```json
{
    "marketplace": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "customer_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
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
    "product_parent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_category": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "star_rating": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "helpful_votes": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "total_votes": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "vine": {
        "num_classes": 2,
        "names": [
            "N",
            "Y"
        ],
        "id": null,
        "_type": "ClassLabel"
    },
    "verified_purchase": {
        "num_classes": 2,
        "names": [
            "N",
            "Y"
        ],
        "id": null,
        "_type": "ClassLabel"
    },
    "review_headline": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_body": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_date": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## Major_Appliances_v1_00

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:amazon_us_reviews/Major_Appliances_v1_00')
```

- **说明**：

```
Amazon Customer Reviews (a.k.a. Product Reviews) is one of Amazons iconic products. In a period of over two decades since the first review in 1995, millions of Amazon customers have contributed over a hundred million reviews to express opinions and describe their experiences regarding products on the Amazon.com website. This makes Amazon Customer Reviews a rich source of information for academic researchers in the fields of Natural Language Processing (NLP), Information Retrieval (IR), and Machine Learning (ML), amongst others. Accordingly, we are releasing this data to further research in multiple disciplines related to understanding customer product experiences. Specifically, this dataset was constructed to represent a sample of customer evaluations and opinions, variation in the perception of a product across geographical regions, and promotional intent or bias in reviews.

Over 130+ million customer reviews are available to researchers as part of this release. The data is available in TSV files in the amazon-reviews-pds S3 bucket in AWS US East Region. Each line in the data files corresponds to an individual review (tab delimited, with no quote and escape characters).

Each Dataset contains the following columns:

- marketplace: 2 letter country code of the marketplace where the review was written.
- customer_id: Random identifier that can be used to aggregate reviews written by a single author.
- review_id: The unique ID of the review.
- product_id: The unique Product ID the review pertains to. In the multilingual dataset the reviews for the same product in different countries can be grouped by the same product_id.
- product_parent: Random identifier that can be used to aggregate reviews for the same product.
- product_title: Title of the product.
- product_category: Broad product category that can be used to group reviews (also used to group the dataset into coherent parts).
- star_rating: The 1-5 star rating of the review.
- helpful_votes: Number of helpful votes.
- total_votes: Number of total votes the review received.
- vine: Review was written as part of the Vine program.
- verified_purchase: The review is on a verified purchase.
- review_headline: The title of the review.
- review_body: The review text.
- review_date: The date the review was written.
```

- **许可**：无已知许可
- **版本**：0.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 96901

- **特征**：

```json
{
    "marketplace": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "customer_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
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
    "product_parent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_category": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "star_rating": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "helpful_votes": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "total_votes": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "vine": {
        "num_classes": 2,
        "names": [
            "N",
            "Y"
        ],
        "id": null,
        "_type": "ClassLabel"
    },
    "verified_purchase": {
        "num_classes": 2,
        "names": [
            "N",
            "Y"
        ],
        "id": null,
        "_type": "ClassLabel"
    },
    "review_headline": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_body": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_date": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## Gift_Card_v1_00

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:amazon_us_reviews/Gift_Card_v1_00')
```

- **说明**：

```
Amazon Customer Reviews (a.k.a. Product Reviews) is one of Amazons iconic products. In a period of over two decades since the first review in 1995, millions of Amazon customers have contributed over a hundred million reviews to express opinions and describe their experiences regarding products on the Amazon.com website. This makes Amazon Customer Reviews a rich source of information for academic researchers in the fields of Natural Language Processing (NLP), Information Retrieval (IR), and Machine Learning (ML), amongst others. Accordingly, we are releasing this data to further research in multiple disciplines related to understanding customer product experiences. Specifically, this dataset was constructed to represent a sample of customer evaluations and opinions, variation in the perception of a product across geographical regions, and promotional intent or bias in reviews.

Over 130+ million customer reviews are available to researchers as part of this release. The data is available in TSV files in the amazon-reviews-pds S3 bucket in AWS US East Region. Each line in the data files corresponds to an individual review (tab delimited, with no quote and escape characters).

Each Dataset contains the following columns:

- marketplace: 2 letter country code of the marketplace where the review was written.
- customer_id: Random identifier that can be used to aggregate reviews written by a single author.
- review_id: The unique ID of the review.
- product_id: The unique Product ID the review pertains to. In the multilingual dataset the reviews for the same product in different countries can be grouped by the same product_id.
- product_parent: Random identifier that can be used to aggregate reviews for the same product.
- product_title: Title of the product.
- product_category: Broad product category that can be used to group reviews (also used to group the dataset into coherent parts).
- star_rating: The 1-5 star rating of the review.
- helpful_votes: Number of helpful votes.
- total_votes: Number of total votes the review received.
- vine: Review was written as part of the Vine program.
- verified_purchase: The review is on a verified purchase.
- review_headline: The title of the review.
- review_body: The review text.
- review_date: The date the review was written.
```

- **许可**：无已知许可
- **版本**：0.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 149086

- **特征**：

```json
{
    "marketplace": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "customer_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
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
    "product_parent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_category": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "star_rating": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "helpful_votes": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "total_votes": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "vine": {
        "num_classes": 2,
        "names": [
            "N",
            "Y"
        ],
        "id": null,
        "_type": "ClassLabel"
    },
    "verified_purchase": {
        "num_classes": 2,
        "names": [
            "N",
            "Y"
        ],
        "id": null,
        "_type": "ClassLabel"
    },
    "review_headline": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_body": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_date": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## Video_v1_00

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:amazon_us_reviews/Video_v1_00')
```

- **说明**：

```
Amazon Customer Reviews (a.k.a. Product Reviews) is one of Amazons iconic products. In a period of over two decades since the first review in 1995, millions of Amazon customers have contributed over a hundred million reviews to express opinions and describe their experiences regarding products on the Amazon.com website. This makes Amazon Customer Reviews a rich source of information for academic researchers in the fields of Natural Language Processing (NLP), Information Retrieval (IR), and Machine Learning (ML), amongst others. Accordingly, we are releasing this data to further research in multiple disciplines related to understanding customer product experiences. Specifically, this dataset was constructed to represent a sample of customer evaluations and opinions, variation in the perception of a product across geographical regions, and promotional intent or bias in reviews.

Over 130+ million customer reviews are available to researchers as part of this release. The data is available in TSV files in the amazon-reviews-pds S3 bucket in AWS US East Region. Each line in the data files corresponds to an individual review (tab delimited, with no quote and escape characters).

Each Dataset contains the following columns:

- marketplace: 2 letter country code of the marketplace where the review was written.
- customer_id: Random identifier that can be used to aggregate reviews written by a single author.
- review_id: The unique ID of the review.
- product_id: The unique Product ID the review pertains to. In the multilingual dataset the reviews for the same product in different countries can be grouped by the same product_id.
- product_parent: Random identifier that can be used to aggregate reviews for the same product.
- product_title: Title of the product.
- product_category: Broad product category that can be used to group reviews (also used to group the dataset into coherent parts).
- star_rating: The 1-5 star rating of the review.
- helpful_votes: Number of helpful votes.
- total_votes: Number of total votes the review received.
- vine: Review was written as part of the Vine program.
- verified_purchase: The review is on a verified purchase.
- review_headline: The title of the review.
- review_body: The review text.
- review_date: The date the review was written.
```

- **许可**：无已知许可
- **版本**：0.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 380604

- **特征**：

```json
{
    "marketplace": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "customer_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
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
    "product_parent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_category": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "star_rating": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "helpful_votes": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "total_votes": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "vine": {
        "num_classes": 2,
        "names": [
            "N",
            "Y"
        ],
        "id": null,
        "_type": "ClassLabel"
    },
    "verified_purchase": {
        "num_classes": 2,
        "names": [
            "N",
            "Y"
        ],
        "id": null,
        "_type": "ClassLabel"
    },
    "review_headline": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_body": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_date": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## Luggage_v1_00

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:amazon_us_reviews/Luggage_v1_00')
```

- **说明**：

```
Amazon Customer Reviews (a.k.a. Product Reviews) is one of Amazons iconic products. In a period of over two decades since the first review in 1995, millions of Amazon customers have contributed over a hundred million reviews to express opinions and describe their experiences regarding products on the Amazon.com website. This makes Amazon Customer Reviews a rich source of information for academic researchers in the fields of Natural Language Processing (NLP), Information Retrieval (IR), and Machine Learning (ML), amongst others. Accordingly, we are releasing this data to further research in multiple disciplines related to understanding customer product experiences. Specifically, this dataset was constructed to represent a sample of customer evaluations and opinions, variation in the perception of a product across geographical regions, and promotional intent or bias in reviews.

Over 130+ million customer reviews are available to researchers as part of this release. The data is available in TSV files in the amazon-reviews-pds S3 bucket in AWS US East Region. Each line in the data files corresponds to an individual review (tab delimited, with no quote and escape characters).

Each Dataset contains the following columns:

- marketplace: 2 letter country code of the marketplace where the review was written.
- customer_id: Random identifier that can be used to aggregate reviews written by a single author.
- review_id: The unique ID of the review.
- product_id: The unique Product ID the review pertains to. In the multilingual dataset the reviews for the same product in different countries can be grouped by the same product_id.
- product_parent: Random identifier that can be used to aggregate reviews for the same product.
- product_title: Title of the product.
- product_category: Broad product category that can be used to group reviews (also used to group the dataset into coherent parts).
- star_rating: The 1-5 star rating of the review.
- helpful_votes: Number of helpful votes.
- total_votes: Number of total votes the review received.
- vine: Review was written as part of the Vine program.
- verified_purchase: The review is on a verified purchase.
- review_headline: The title of the review.
- review_body: The review text.
- review_date: The date the review was written.
```

- **许可**：无已知许可
- **版本**：0.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 348657

- **特征**：

```json
{
    "marketplace": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "customer_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
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
    "product_parent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_category": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "star_rating": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "helpful_votes": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "total_votes": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "vine": {
        "num_classes": 2,
        "names": [
            "N",
            "Y"
        ],
        "id": null,
        "_type": "ClassLabel"
    },
    "verified_purchase": {
        "num_classes": 2,
        "names": [
            "N",
            "Y"
        ],
        "id": null,
        "_type": "ClassLabel"
    },
    "review_headline": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_body": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_date": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## Software_v1_00

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:amazon_us_reviews/Software_v1_00')
```

- **说明**：

```
Amazon Customer Reviews (a.k.a. Product Reviews) is one of Amazons iconic products. In a period of over two decades since the first review in 1995, millions of Amazon customers have contributed over a hundred million reviews to express opinions and describe their experiences regarding products on the Amazon.com website. This makes Amazon Customer Reviews a rich source of information for academic researchers in the fields of Natural Language Processing (NLP), Information Retrieval (IR), and Machine Learning (ML), amongst others. Accordingly, we are releasing this data to further research in multiple disciplines related to understanding customer product experiences. Specifically, this dataset was constructed to represent a sample of customer evaluations and opinions, variation in the perception of a product across geographical regions, and promotional intent or bias in reviews.

Over 130+ million customer reviews are available to researchers as part of this release. The data is available in TSV files in the amazon-reviews-pds S3 bucket in AWS US East Region. Each line in the data files corresponds to an individual review (tab delimited, with no quote and escape characters).

Each Dataset contains the following columns:

- marketplace: 2 letter country code of the marketplace where the review was written.
- customer_id: Random identifier that can be used to aggregate reviews written by a single author.
- review_id: The unique ID of the review.
- product_id: The unique Product ID the review pertains to. In the multilingual dataset the reviews for the same product in different countries can be grouped by the same product_id.
- product_parent: Random identifier that can be used to aggregate reviews for the same product.
- product_title: Title of the product.
- product_category: Broad product category that can be used to group reviews (also used to group the dataset into coherent parts).
- star_rating: The 1-5 star rating of the review.
- helpful_votes: Number of helpful votes.
- total_votes: Number of total votes the review received.
- vine: Review was written as part of the Vine program.
- verified_purchase: The review is on a verified purchase.
- review_headline: The title of the review.
- review_body: The review text.
- review_date: The date the review was written.
```

- **许可**：无已知许可
- **版本**：0.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 341931

- **特征**：

```json
{
    "marketplace": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "customer_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
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
    "product_parent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_category": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "star_rating": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "helpful_votes": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "total_votes": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "vine": {
        "num_classes": 2,
        "names": [
            "N",
            "Y"
        ],
        "id": null,
        "_type": "ClassLabel"
    },
    "verified_purchase": {
        "num_classes": 2,
        "names": [
            "N",
            "Y"
        ],
        "id": null,
        "_type": "ClassLabel"
    },
    "review_headline": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_body": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_date": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## Video_Games_v1_00

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:amazon_us_reviews/Video_Games_v1_00')
```

- **说明**：

```
Amazon Customer Reviews (a.k.a. Product Reviews) is one of Amazons iconic products. In a period of over two decades since the first review in 1995, millions of Amazon customers have contributed over a hundred million reviews to express opinions and describe their experiences regarding products on the Amazon.com website. This makes Amazon Customer Reviews a rich source of information for academic researchers in the fields of Natural Language Processing (NLP), Information Retrieval (IR), and Machine Learning (ML), amongst others. Accordingly, we are releasing this data to further research in multiple disciplines related to understanding customer product experiences. Specifically, this dataset was constructed to represent a sample of customer evaluations and opinions, variation in the perception of a product across geographical regions, and promotional intent or bias in reviews.

Over 130+ million customer reviews are available to researchers as part of this release. The data is available in TSV files in the amazon-reviews-pds S3 bucket in AWS US East Region. Each line in the data files corresponds to an individual review (tab delimited, with no quote and escape characters).

Each Dataset contains the following columns:

- marketplace: 2 letter country code of the marketplace where the review was written.
- customer_id: Random identifier that can be used to aggregate reviews written by a single author.
- review_id: The unique ID of the review.
- product_id: The unique Product ID the review pertains to. In the multilingual dataset the reviews for the same product in different countries can be grouped by the same product_id.
- product_parent: Random identifier that can be used to aggregate reviews for the same product.
- product_title: Title of the product.
- product_category: Broad product category that can be used to group reviews (also used to group the dataset into coherent parts).
- star_rating: The 1-5 star rating of the review.
- helpful_votes: Number of helpful votes.
- total_votes: Number of total votes the review received.
- vine: Review was written as part of the Vine program.
- verified_purchase: The review is on a verified purchase.
- review_headline: The title of the review.
- review_body: The review text.
- review_date: The date the review was written.
```

- **许可**：无已知许可
- **版本**：0.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 1785997

- **特征**：

```json
{
    "marketplace": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "customer_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
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
    "product_parent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_category": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "star_rating": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "helpful_votes": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "total_votes": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "vine": {
        "num_classes": 2,
        "names": [
            "N",
            "Y"
        ],
        "id": null,
        "_type": "ClassLabel"
    },
    "verified_purchase": {
        "num_classes": 2,
        "names": [
            "N",
            "Y"
        ],
        "id": null,
        "_type": "ClassLabel"
    },
    "review_headline": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_body": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_date": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## Furniture_v1_00

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:amazon_us_reviews/Furniture_v1_00')
```

- **说明**：

```
Amazon Customer Reviews (a.k.a. Product Reviews) is one of Amazons iconic products. In a period of over two decades since the first review in 1995, millions of Amazon customers have contributed over a hundred million reviews to express opinions and describe their experiences regarding products on the Amazon.com website. This makes Amazon Customer Reviews a rich source of information for academic researchers in the fields of Natural Language Processing (NLP), Information Retrieval (IR), and Machine Learning (ML), amongst others. Accordingly, we are releasing this data to further research in multiple disciplines related to understanding customer product experiences. Specifically, this dataset was constructed to represent a sample of customer evaluations and opinions, variation in the perception of a product across geographical regions, and promotional intent or bias in reviews.

Over 130+ million customer reviews are available to researchers as part of this release. The data is available in TSV files in the amazon-reviews-pds S3 bucket in AWS US East Region. Each line in the data files corresponds to an individual review (tab delimited, with no quote and escape characters).

Each Dataset contains the following columns:

- marketplace: 2 letter country code of the marketplace where the review was written.
- customer_id: Random identifier that can be used to aggregate reviews written by a single author.
- review_id: The unique ID of the review.
- product_id: The unique Product ID the review pertains to. In the multilingual dataset the reviews for the same product in different countries can be grouped by the same product_id.
- product_parent: Random identifier that can be used to aggregate reviews for the same product.
- product_title: Title of the product.
- product_category: Broad product category that can be used to group reviews (also used to group the dataset into coherent parts).
- star_rating: The 1-5 star rating of the review.
- helpful_votes: Number of helpful votes.
- total_votes: Number of total votes the review received.
- vine: Review was written as part of the Vine program.
- verified_purchase: The review is on a verified purchase.
- review_headline: The title of the review.
- review_body: The review text.
- review_date: The date the review was written.
```

- **许可**：无已知许可
- **版本**：0.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 792113

- **特征**：

```json
{
    "marketplace": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "customer_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
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
    "product_parent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_category": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "star_rating": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "helpful_votes": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "total_votes": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "vine": {
        "num_classes": 2,
        "names": [
            "N",
            "Y"
        ],
        "id": null,
        "_type": "ClassLabel"
    },
    "verified_purchase": {
        "num_classes": 2,
        "names": [
            "N",
            "Y"
        ],
        "id": null,
        "_type": "ClassLabel"
    },
    "review_headline": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_body": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_date": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## Musical_Instruments_v1_00

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:amazon_us_reviews/Musical_Instruments_v1_00')
```

- **说明**：

```
Amazon Customer Reviews (a.k.a. Product Reviews) is one of Amazons iconic products. In a period of over two decades since the first review in 1995, millions of Amazon customers have contributed over a hundred million reviews to express opinions and describe their experiences regarding products on the Amazon.com website. This makes Amazon Customer Reviews a rich source of information for academic researchers in the fields of Natural Language Processing (NLP), Information Retrieval (IR), and Machine Learning (ML), amongst others. Accordingly, we are releasing this data to further research in multiple disciplines related to understanding customer product experiences. Specifically, this dataset was constructed to represent a sample of customer evaluations and opinions, variation in the perception of a product across geographical regions, and promotional intent or bias in reviews.

Over 130+ million customer reviews are available to researchers as part of this release. The data is available in TSV files in the amazon-reviews-pds S3 bucket in AWS US East Region. Each line in the data files corresponds to an individual review (tab delimited, with no quote and escape characters).

Each Dataset contains the following columns:

- marketplace: 2 letter country code of the marketplace where the review was written.
- customer_id: Random identifier that can be used to aggregate reviews written by a single author.
- review_id: The unique ID of the review.
- product_id: The unique Product ID the review pertains to. In the multilingual dataset the reviews for the same product in different countries can be grouped by the same product_id.
- product_parent: Random identifier that can be used to aggregate reviews for the same product.
- product_title: Title of the product.
- product_category: Broad product category that can be used to group reviews (also used to group the dataset into coherent parts).
- star_rating: The 1-5 star rating of the review.
- helpful_votes: Number of helpful votes.
- total_votes: Number of total votes the review received.
- vine: Review was written as part of the Vine program.
- verified_purchase: The review is on a verified purchase.
- review_headline: The title of the review.
- review_body: The review text.
- review_date: The date the review was written.
```

- **许可**：无已知许可
- **版本**：0.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 904765

- **特征**：

```json
{
    "marketplace": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "customer_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
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
    "product_parent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_category": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "star_rating": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "helpful_votes": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "total_votes": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "vine": {
        "num_classes": 2,
        "names": [
            "N",
            "Y"
        ],
        "id": null,
        "_type": "ClassLabel"
    },
    "verified_purchase": {
        "num_classes": 2,
        "names": [
            "N",
            "Y"
        ],
        "id": null,
        "_type": "ClassLabel"
    },
    "review_headline": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_body": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_date": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## Digital_Music_Purchase_v1_00

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:amazon_us_reviews/Digital_Music_Purchase_v1_00')
```

- **说明**：

```
Amazon Customer Reviews (a.k.a. Product Reviews) is one of Amazons iconic products. In a period of over two decades since the first review in 1995, millions of Amazon customers have contributed over a hundred million reviews to express opinions and describe their experiences regarding products on the Amazon.com website. This makes Amazon Customer Reviews a rich source of information for academic researchers in the fields of Natural Language Processing (NLP), Information Retrieval (IR), and Machine Learning (ML), amongst others. Accordingly, we are releasing this data to further research in multiple disciplines related to understanding customer product experiences. Specifically, this dataset was constructed to represent a sample of customer evaluations and opinions, variation in the perception of a product across geographical regions, and promotional intent or bias in reviews.

Over 130+ million customer reviews are available to researchers as part of this release. The data is available in TSV files in the amazon-reviews-pds S3 bucket in AWS US East Region. Each line in the data files corresponds to an individual review (tab delimited, with no quote and escape characters).

Each Dataset contains the following columns:

- marketplace: 2 letter country code of the marketplace where the review was written.
- customer_id: Random identifier that can be used to aggregate reviews written by a single author.
- review_id: The unique ID of the review.
- product_id: The unique Product ID the review pertains to. In the multilingual dataset the reviews for the same product in different countries can be grouped by the same product_id.
- product_parent: Random identifier that can be used to aggregate reviews for the same product.
- product_title: Title of the product.
- product_category: Broad product category that can be used to group reviews (also used to group the dataset into coherent parts).
- star_rating: The 1-5 star rating of the review.
- helpful_votes: Number of helpful votes.
- total_votes: Number of total votes the review received.
- vine: Review was written as part of the Vine program.
- verified_purchase: The review is on a verified purchase.
- review_headline: The title of the review.
- review_body: The review text.
- review_date: The date the review was written.
```

- **许可**：无已知许可
- **版本**：0.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 1688884

- **特征**：

```json
{
    "marketplace": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "customer_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
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
    "product_parent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_category": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "star_rating": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "helpful_votes": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "total_votes": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "vine": {
        "num_classes": 2,
        "names": [
            "N",
            "Y"
        ],
        "id": null,
        "_type": "ClassLabel"
    },
    "verified_purchase": {
        "num_classes": 2,
        "names": [
            "N",
            "Y"
        ],
        "id": null,
        "_type": "ClassLabel"
    },
    "review_headline": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_body": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_date": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## Books_v1_02

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:amazon_us_reviews/Books_v1_02')
```

- **说明**：

```
Amazon Customer Reviews (a.k.a. Product Reviews) is one of Amazons iconic products. In a period of over two decades since the first review in 1995, millions of Amazon customers have contributed over a hundred million reviews to express opinions and describe their experiences regarding products on the Amazon.com website. This makes Amazon Customer Reviews a rich source of information for academic researchers in the fields of Natural Language Processing (NLP), Information Retrieval (IR), and Machine Learning (ML), amongst others. Accordingly, we are releasing this data to further research in multiple disciplines related to understanding customer product experiences. Specifically, this dataset was constructed to represent a sample of customer evaluations and opinions, variation in the perception of a product across geographical regions, and promotional intent or bias in reviews.

Over 130+ million customer reviews are available to researchers as part of this release. The data is available in TSV files in the amazon-reviews-pds S3 bucket in AWS US East Region. Each line in the data files corresponds to an individual review (tab delimited, with no quote and escape characters).

Each Dataset contains the following columns:

- marketplace: 2 letter country code of the marketplace where the review was written.
- customer_id: Random identifier that can be used to aggregate reviews written by a single author.
- review_id: The unique ID of the review.
- product_id: The unique Product ID the review pertains to. In the multilingual dataset the reviews for the same product in different countries can be grouped by the same product_id.
- product_parent: Random identifier that can be used to aggregate reviews for the same product.
- product_title: Title of the product.
- product_category: Broad product category that can be used to group reviews (also used to group the dataset into coherent parts).
- star_rating: The 1-5 star rating of the review.
- helpful_votes: Number of helpful votes.
- total_votes: Number of total votes the review received.
- vine: Review was written as part of the Vine program.
- verified_purchase: The review is on a verified purchase.
- review_headline: The title of the review.
- review_body: The review text.
- review_date: The date the review was written.
```

- **许可**：无已知许可
- **版本**：0.1.0
- **拆分**:

拆分 | 样本
:-- | --:
`'train'` | 3105520

- **特征**：

```json
{
    "marketplace": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "customer_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
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
    "product_parent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_category": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "star_rating": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "helpful_votes": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "total_votes": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "vine": {
        "num_classes": 2,
        "names": [
            "N",
            "Y"
        ],
        "id": null,
        "_type": "ClassLabel"
    },
    "verified_purchase": {
        "num_classes": 2,
        "names": [
            "N",
            "Y"
        ],
        "id": null,
        "_type": "ClassLabel"
    },
    "review_headline": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_body": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_date": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## Home_Entertainment_v1_00

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:amazon_us_reviews/Home_Entertainment_v1_00')
```

- **说明**：

```
Amazon Customer Reviews (a.k.a. Product Reviews) is one of Amazons iconic products. In a period of over two decades since the first review in 1995, millions of Amazon customers have contributed over a hundred million reviews to express opinions and describe their experiences regarding products on the Amazon.com website. This makes Amazon Customer Reviews a rich source of information for academic researchers in the fields of Natural Language Processing (NLP), Information Retrieval (IR), and Machine Learning (ML), amongst others. Accordingly, we are releasing this data to further research in multiple disciplines related to understanding customer product experiences. Specifically, this dataset was constructed to represent a sample of customer evaluations and opinions, variation in the perception of a product across geographical regions, and promotional intent or bias in reviews.

Over 130+ million customer reviews are available to researchers as part of this release. The data is available in TSV files in the amazon-reviews-pds S3 bucket in AWS US East Region. Each line in the data files corresponds to an individual review (tab delimited, with no quote and escape characters).

Each Dataset contains the following columns:

- marketplace: 2 letter country code of the marketplace where the review was written.
- customer_id: Random identifier that can be used to aggregate reviews written by a single author.
- review_id: The unique ID of the review.
- product_id: The unique Product ID the review pertains to. In the multilingual dataset the reviews for the same product in different countries can be grouped by the same product_id.
- product_parent: Random identifier that can be used to aggregate reviews for the same product.
- product_title: Title of the product.
- product_category: Broad product category that can be used to group reviews (also used to group the dataset into coherent parts).
- star_rating: The 1-5 star rating of the review.
- helpful_votes: Number of helpful votes.
- total_votes: Number of total votes the review received.
- vine: Review was written as part of the Vine program.
- verified_purchase: The review is on a verified purchase.
- review_headline: The title of the review.
- review_body: The review text.
- review_date: The date the review was written.
```

- **许可**：无已知许可
- **版本**：0.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 705889

- **特征**：

```json
{
    "marketplace": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "customer_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
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
    "product_parent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_category": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "star_rating": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "helpful_votes": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "total_votes": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "vine": {
        "num_classes": 2,
        "names": [
            "N",
            "Y"
        ],
        "id": null,
        "_type": "ClassLabel"
    },
    "verified_purchase": {
        "num_classes": 2,
        "names": [
            "N",
            "Y"
        ],
        "id": null,
        "_type": "ClassLabel"
    },
    "review_headline": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_body": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_date": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## Grocery_v1_00

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:amazon_us_reviews/Grocery_v1_00')
```

- **说明**：

```
Amazon Customer Reviews (a.k.a. Product Reviews) is one of Amazons iconic products. In a period of over two decades since the first review in 1995, millions of Amazon customers have contributed over a hundred million reviews to express opinions and describe their experiences regarding products on the Amazon.com website. This makes Amazon Customer Reviews a rich source of information for academic researchers in the fields of Natural Language Processing (NLP), Information Retrieval (IR), and Machine Learning (ML), amongst others. Accordingly, we are releasing this data to further research in multiple disciplines related to understanding customer product experiences. Specifically, this dataset was constructed to represent a sample of customer evaluations and opinions, variation in the perception of a product across geographical regions, and promotional intent or bias in reviews.

Over 130+ million customer reviews are available to researchers as part of this release. The data is available in TSV files in the amazon-reviews-pds S3 bucket in AWS US East Region. Each line in the data files corresponds to an individual review (tab delimited, with no quote and escape characters).

Each Dataset contains the following columns:

- marketplace: 2 letter country code of the marketplace where the review was written.
- customer_id: Random identifier that can be used to aggregate reviews written by a single author.
- review_id: The unique ID of the review.
- product_id: The unique Product ID the review pertains to. In the multilingual dataset the reviews for the same product in different countries can be grouped by the same product_id.
- product_parent: Random identifier that can be used to aggregate reviews for the same product.
- product_title: Title of the product.
- product_category: Broad product category that can be used to group reviews (also used to group the dataset into coherent parts).
- star_rating: The 1-5 star rating of the review.
- helpful_votes: Number of helpful votes.
- total_votes: Number of total votes the review received.
- vine: Review was written as part of the Vine program.
- verified_purchase: The review is on a verified purchase.
- review_headline: The title of the review.
- review_body: The review text.
- review_date: The date the review was written.
```

- **许可**：无已知许可
- **版本**：0.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 2402458

- **特征**：

```json
{
    "marketplace": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "customer_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
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
    "product_parent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_category": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "star_rating": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "helpful_votes": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "total_votes": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "vine": {
        "num_classes": 2,
        "names": [
            "N",
            "Y"
        ],
        "id": null,
        "_type": "ClassLabel"
    },
    "verified_purchase": {
        "num_classes": 2,
        "names": [
            "N",
            "Y"
        ],
        "id": null,
        "_type": "ClassLabel"
    },
    "review_headline": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_body": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_date": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## Outdoors_v1_00

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:amazon_us_reviews/Outdoors_v1_00')
```

- **说明**：

```
Amazon Customer Reviews (a.k.a. Product Reviews) is one of Amazons iconic products. In a period of over two decades since the first review in 1995, millions of Amazon customers have contributed over a hundred million reviews to express opinions and describe their experiences regarding products on the Amazon.com website. This makes Amazon Customer Reviews a rich source of information for academic researchers in the fields of Natural Language Processing (NLP), Information Retrieval (IR), and Machine Learning (ML), amongst others. Accordingly, we are releasing this data to further research in multiple disciplines related to understanding customer product experiences. Specifically, this dataset was constructed to represent a sample of customer evaluations and opinions, variation in the perception of a product across geographical regions, and promotional intent or bias in reviews.

Over 130+ million customer reviews are available to researchers as part of this release. The data is available in TSV files in the amazon-reviews-pds S3 bucket in AWS US East Region. Each line in the data files corresponds to an individual review (tab delimited, with no quote and escape characters).

Each Dataset contains the following columns:

- marketplace: 2 letter country code of the marketplace where the review was written.
- customer_id: Random identifier that can be used to aggregate reviews written by a single author.
- review_id: The unique ID of the review.
- product_id: The unique Product ID the review pertains to. In the multilingual dataset the reviews for the same product in different countries can be grouped by the same product_id.
- product_parent: Random identifier that can be used to aggregate reviews for the same product.
- product_title: Title of the product.
- product_category: Broad product category that can be used to group reviews (also used to group the dataset into coherent parts).
- star_rating: The 1-5 star rating of the review.
- helpful_votes: Number of helpful votes.
- total_votes: Number of total votes the review received.
- vine: Review was written as part of the Vine program.
- verified_purchase: The review is on a verified purchase.
- review_headline: The title of the review.
- review_body: The review text.
- review_date: The date the review was written.
```

- **许可**：无已知许可
- **版本**：0.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 2302401

- **特征**：

```json
{
    "marketplace": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "customer_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
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
    "product_parent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_category": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "star_rating": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "helpful_votes": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "total_votes": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "vine": {
        "num_classes": 2,
        "names": [
            "N",
            "Y"
        ],
        "id": null,
        "_type": "ClassLabel"
    },
    "verified_purchase": {
        "num_classes": 2,
        "names": [
            "N",
            "Y"
        ],
        "id": null,
        "_type": "ClassLabel"
    },
    "review_headline": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_body": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_date": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## Pet_Products_v1_00

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:amazon_us_reviews/Pet_Products_v1_00')
```

- **说明**：

```
Amazon Customer Reviews (a.k.a. Product Reviews) is one of Amazons iconic products. In a period of over two decades since the first review in 1995, millions of Amazon customers have contributed over a hundred million reviews to express opinions and describe their experiences regarding products on the Amazon.com website. This makes Amazon Customer Reviews a rich source of information for academic researchers in the fields of Natural Language Processing (NLP), Information Retrieval (IR), and Machine Learning (ML), amongst others. Accordingly, we are releasing this data to further research in multiple disciplines related to understanding customer product experiences. Specifically, this dataset was constructed to represent a sample of customer evaluations and opinions, variation in the perception of a product across geographical regions, and promotional intent or bias in reviews.

Over 130+ million customer reviews are available to researchers as part of this release. The data is available in TSV files in the amazon-reviews-pds S3 bucket in AWS US East Region. Each line in the data files corresponds to an individual review (tab delimited, with no quote and escape characters).

Each Dataset contains the following columns:

- marketplace: 2 letter country code of the marketplace where the review was written.
- customer_id: Random identifier that can be used to aggregate reviews written by a single author.
- review_id: The unique ID of the review.
- product_id: The unique Product ID the review pertains to. In the multilingual dataset the reviews for the same product in different countries can be grouped by the same product_id.
- product_parent: Random identifier that can be used to aggregate reviews for the same product.
- product_title: Title of the product.
- product_category: Broad product category that can be used to group reviews (also used to group the dataset into coherent parts).
- star_rating: The 1-5 star rating of the review.
- helpful_votes: Number of helpful votes.
- total_votes: Number of total votes the review received.
- vine: Review was written as part of the Vine program.
- verified_purchase: The review is on a verified purchase.
- review_headline: The title of the review.
- review_body: The review text.
- review_date: The date the review was written.
```

- **许可**：无已知许可
- **版本**：0.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 2643619

- **特征**：

```json
{
    "marketplace": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "customer_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
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
    "product_parent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_category": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "star_rating": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "helpful_votes": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "total_votes": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "vine": {
        "num_classes": 2,
        "names": [
            "N",
            "Y"
        ],
        "id": null,
        "_type": "ClassLabel"
    },
    "verified_purchase": {
        "num_classes": 2,
        "names": [
            "N",
            "Y"
        ],
        "id": null,
        "_type": "ClassLabel"
    },
    "review_headline": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_body": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_date": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## Video_DVD_v1_00

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:amazon_us_reviews/Video_DVD_v1_00')
```

- **说明**：

```
Amazon Customer Reviews (a.k.a. Product Reviews) is one of Amazons iconic products. In a period of over two decades since the first review in 1995, millions of Amazon customers have contributed over a hundred million reviews to express opinions and describe their experiences regarding products on the Amazon.com website. This makes Amazon Customer Reviews a rich source of information for academic researchers in the fields of Natural Language Processing (NLP), Information Retrieval (IR), and Machine Learning (ML), amongst others. Accordingly, we are releasing this data to further research in multiple disciplines related to understanding customer product experiences. Specifically, this dataset was constructed to represent a sample of customer evaluations and opinions, variation in the perception of a product across geographical regions, and promotional intent or bias in reviews.

Over 130+ million customer reviews are available to researchers as part of this release. The data is available in TSV files in the amazon-reviews-pds S3 bucket in AWS US East Region. Each line in the data files corresponds to an individual review (tab delimited, with no quote and escape characters).

Each Dataset contains the following columns:

- marketplace: 2 letter country code of the marketplace where the review was written.
- customer_id: Random identifier that can be used to aggregate reviews written by a single author.
- review_id: The unique ID of the review.
- product_id: The unique Product ID the review pertains to. In the multilingual dataset the reviews for the same product in different countries can be grouped by the same product_id.
- product_parent: Random identifier that can be used to aggregate reviews for the same product.
- product_title: Title of the product.
- product_category: Broad product category that can be used to group reviews (also used to group the dataset into coherent parts).
- star_rating: The 1-5 star rating of the review.
- helpful_votes: Number of helpful votes.
- total_votes: Number of total votes the review received.
- vine: Review was written as part of the Vine program.
- verified_purchase: The review is on a verified purchase.
- review_headline: The title of the review.
- review_body: The review text.
- review_date: The date the review was written.
```

- **许可**：无已知许可
- **版本**：0.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 5069140

- **特征**：

```json
{
    "marketplace": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "customer_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
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
    "product_parent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_category": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "star_rating": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "helpful_votes": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "total_votes": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "vine": {
        "num_classes": 2,
        "names": [
            "N",
            "Y"
        ],
        "id": null,
        "_type": "ClassLabel"
    },
    "verified_purchase": {
        "num_classes": 2,
        "names": [
            "N",
            "Y"
        ],
        "id": null,
        "_type": "ClassLabel"
    },
    "review_headline": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_body": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_date": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## Apparel_v1_00

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:amazon_us_reviews/Apparel_v1_00')
```

- **说明**：

```
Amazon Customer Reviews (a.k.a. Product Reviews) is one of Amazons iconic products. In a period of over two decades since the first review in 1995, millions of Amazon customers have contributed over a hundred million reviews to express opinions and describe their experiences regarding products on the Amazon.com website. This makes Amazon Customer Reviews a rich source of information for academic researchers in the fields of Natural Language Processing (NLP), Information Retrieval (IR), and Machine Learning (ML), amongst others. Accordingly, we are releasing this data to further research in multiple disciplines related to understanding customer product experiences. Specifically, this dataset was constructed to represent a sample of customer evaluations and opinions, variation in the perception of a product across geographical regions, and promotional intent or bias in reviews.

Over 130+ million customer reviews are available to researchers as part of this release. The data is available in TSV files in the amazon-reviews-pds S3 bucket in AWS US East Region. Each line in the data files corresponds to an individual review (tab delimited, with no quote and escape characters).

Each Dataset contains the following columns:

- marketplace: 2 letter country code of the marketplace where the review was written.
- customer_id: Random identifier that can be used to aggregate reviews written by a single author.
- review_id: The unique ID of the review.
- product_id: The unique Product ID the review pertains to. In the multilingual dataset the reviews for the same product in different countries can be grouped by the same product_id.
- product_parent: Random identifier that can be used to aggregate reviews for the same product.
- product_title: Title of the product.
- product_category: Broad product category that can be used to group reviews (also used to group the dataset into coherent parts).
- star_rating: The 1-5 star rating of the review.
- helpful_votes: Number of helpful votes.
- total_votes: Number of total votes the review received.
- vine: Review was written as part of the Vine program.
- verified_purchase: The review is on a verified purchase.
- review_headline: The title of the review.
- review_body: The review text.
- review_date: The date the review was written.
```

- **许可**：无已知许可
- **版本**：0.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 5906333

- **特征**：

```json
{
    "marketplace": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "customer_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
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
    "product_parent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_category": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "star_rating": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "helpful_votes": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "total_votes": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "vine": {
        "num_classes": 2,
        "names": [
            "N",
            "Y"
        ],
        "id": null,
        "_type": "ClassLabel"
    },
    "verified_purchase": {
        "num_classes": 2,
        "names": [
            "N",
            "Y"
        ],
        "id": null,
        "_type": "ClassLabel"
    },
    "review_headline": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_body": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_date": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## PC_v1_00

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:amazon_us_reviews/PC_v1_00')
```

- **说明**：

```
Amazon Customer Reviews (a.k.a. Product Reviews) is one of Amazons iconic products. In a period of over two decades since the first review in 1995, millions of Amazon customers have contributed over a hundred million reviews to express opinions and describe their experiences regarding products on the Amazon.com website. This makes Amazon Customer Reviews a rich source of information for academic researchers in the fields of Natural Language Processing (NLP), Information Retrieval (IR), and Machine Learning (ML), amongst others. Accordingly, we are releasing this data to further research in multiple disciplines related to understanding customer product experiences. Specifically, this dataset was constructed to represent a sample of customer evaluations and opinions, variation in the perception of a product across geographical regions, and promotional intent or bias in reviews.

Over 130+ million customer reviews are available to researchers as part of this release. The data is available in TSV files in the amazon-reviews-pds S3 bucket in AWS US East Region. Each line in the data files corresponds to an individual review (tab delimited, with no quote and escape characters).

Each Dataset contains the following columns:

- marketplace: 2 letter country code of the marketplace where the review was written.
- customer_id: Random identifier that can be used to aggregate reviews written by a single author.
- review_id: The unique ID of the review.
- product_id: The unique Product ID the review pertains to. In the multilingual dataset the reviews for the same product in different countries can be grouped by the same product_id.
- product_parent: Random identifier that can be used to aggregate reviews for the same product.
- product_title: Title of the product.
- product_category: Broad product category that can be used to group reviews (also used to group the dataset into coherent parts).
- star_rating: The 1-5 star rating of the review.
- helpful_votes: Number of helpful votes.
- total_votes: Number of total votes the review received.
- vine: Review was written as part of the Vine program.
- verified_purchase: The review is on a verified purchase.
- review_headline: The title of the review.
- review_body: The review text.
- review_date: The date the review was written.
```

- **许可**：无已知许可
- **版本**：0.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 6908554

- **特征**：

```json
{
    "marketplace": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "customer_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
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
    "product_parent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_category": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "star_rating": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "helpful_votes": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "total_votes": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "vine": {
        "num_classes": 2,
        "names": [
            "N",
            "Y"
        ],
        "id": null,
        "_type": "ClassLabel"
    },
    "verified_purchase": {
        "num_classes": 2,
        "names": [
            "N",
            "Y"
        ],
        "id": null,
        "_type": "ClassLabel"
    },
    "review_headline": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_body": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_date": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## Tools_v1_00

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:amazon_us_reviews/Tools_v1_00')
```

- **说明**：

```
Amazon Customer Reviews (a.k.a. Product Reviews) is one of Amazons iconic products. In a period of over two decades since the first review in 1995, millions of Amazon customers have contributed over a hundred million reviews to express opinions and describe their experiences regarding products on the Amazon.com website. This makes Amazon Customer Reviews a rich source of information for academic researchers in the fields of Natural Language Processing (NLP), Information Retrieval (IR), and Machine Learning (ML), amongst others. Accordingly, we are releasing this data to further research in multiple disciplines related to understanding customer product experiences. Specifically, this dataset was constructed to represent a sample of customer evaluations and opinions, variation in the perception of a product across geographical regions, and promotional intent or bias in reviews.

Over 130+ million customer reviews are available to researchers as part of this release. The data is available in TSV files in the amazon-reviews-pds S3 bucket in AWS US East Region. Each line in the data files corresponds to an individual review (tab delimited, with no quote and escape characters).

Each Dataset contains the following columns:

- marketplace: 2 letter country code of the marketplace where the review was written.
- customer_id: Random identifier that can be used to aggregate reviews written by a single author.
- review_id: The unique ID of the review.
- product_id: The unique Product ID the review pertains to. In the multilingual dataset the reviews for the same product in different countries can be grouped by the same product_id.
- product_parent: Random identifier that can be used to aggregate reviews for the same product.
- product_title: Title of the product.
- product_category: Broad product category that can be used to group reviews (also used to group the dataset into coherent parts).
- star_rating: The 1-5 star rating of the review.
- helpful_votes: Number of helpful votes.
- total_votes: Number of total votes the review received.
- vine: Review was written as part of the Vine program.
- verified_purchase: The review is on a verified purchase.
- review_headline: The title of the review.
- review_body: The review text.
- review_date: The date the review was written.
```

- **许可**：无已知许可
- **版本**：0.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 1741100

- **特征**：

```json
{
    "marketplace": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "customer_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
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
    "product_parent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_category": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "star_rating": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "helpful_votes": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "total_votes": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "vine": {
        "num_classes": 2,
        "names": [
            "N",
            "Y"
        ],
        "id": null,
        "_type": "ClassLabel"
    },
    "verified_purchase": {
        "num_classes": 2,
        "names": [
            "N",
            "Y"
        ],
        "id": null,
        "_type": "ClassLabel"
    },
    "review_headline": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_body": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_date": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## Jewelry_v1_00

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:amazon_us_reviews/Jewelry_v1_00')
```

- **说明**：

```
Amazon Customer Reviews (a.k.a. Product Reviews) is one of Amazons iconic products. In a period of over two decades since the first review in 1995, millions of Amazon customers have contributed over a hundred million reviews to express opinions and describe their experiences regarding products on the Amazon.com website. This makes Amazon Customer Reviews a rich source of information for academic researchers in the fields of Natural Language Processing (NLP), Information Retrieval (IR), and Machine Learning (ML), amongst others. Accordingly, we are releasing this data to further research in multiple disciplines related to understanding customer product experiences. Specifically, this dataset was constructed to represent a sample of customer evaluations and opinions, variation in the perception of a product across geographical regions, and promotional intent or bias in reviews.

Over 130+ million customer reviews are available to researchers as part of this release. The data is available in TSV files in the amazon-reviews-pds S3 bucket in AWS US East Region. Each line in the data files corresponds to an individual review (tab delimited, with no quote and escape characters).

Each Dataset contains the following columns:

- marketplace: 2 letter country code of the marketplace where the review was written.
- customer_id: Random identifier that can be used to aggregate reviews written by a single author.
- review_id: The unique ID of the review.
- product_id: The unique Product ID the review pertains to. In the multilingual dataset the reviews for the same product in different countries can be grouped by the same product_id.
- product_parent: Random identifier that can be used to aggregate reviews for the same product.
- product_title: Title of the product.
- product_category: Broad product category that can be used to group reviews (also used to group the dataset into coherent parts).
- star_rating: The 1-5 star rating of the review.
- helpful_votes: Number of helpful votes.
- total_votes: Number of total votes the review received.
- vine: Review was written as part of the Vine program.
- verified_purchase: The review is on a verified purchase.
- review_headline: The title of the review.
- review_body: The review text.
- review_date: The date the review was written.
```

- **许可**：无已知许可
- **版本**：0.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 1767753

- **特征**：

```json
{
    "marketplace": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "customer_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
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
    "product_parent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_category": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "star_rating": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "helpful_votes": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "total_votes": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "vine": {
        "num_classes": 2,
        "names": [
            "N",
            "Y"
        ],
        "id": null,
        "_type": "ClassLabel"
    },
    "verified_purchase": {
        "num_classes": 2,
        "names": [
            "N",
            "Y"
        ],
        "id": null,
        "_type": "ClassLabel"
    },
    "review_headline": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_body": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_date": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## Baby_v1_00

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:amazon_us_reviews/Baby_v1_00')
```

- **说明**：

```
Amazon Customer Reviews (a.k.a. Product Reviews) is one of Amazons iconic products. In a period of over two decades since the first review in 1995, millions of Amazon customers have contributed over a hundred million reviews to express opinions and describe their experiences regarding products on the Amazon.com website. This makes Amazon Customer Reviews a rich source of information for academic researchers in the fields of Natural Language Processing (NLP), Information Retrieval (IR), and Machine Learning (ML), amongst others. Accordingly, we are releasing this data to further research in multiple disciplines related to understanding customer product experiences. Specifically, this dataset was constructed to represent a sample of customer evaluations and opinions, variation in the perception of a product across geographical regions, and promotional intent or bias in reviews.

Over 130+ million customer reviews are available to researchers as part of this release. The data is available in TSV files in the amazon-reviews-pds S3 bucket in AWS US East Region. Each line in the data files corresponds to an individual review (tab delimited, with no quote and escape characters).

Each Dataset contains the following columns:

- marketplace: 2 letter country code of the marketplace where the review was written.
- customer_id: Random identifier that can be used to aggregate reviews written by a single author.
- review_id: The unique ID of the review.
- product_id: The unique Product ID the review pertains to. In the multilingual dataset the reviews for the same product in different countries can be grouped by the same product_id.
- product_parent: Random identifier that can be used to aggregate reviews for the same product.
- product_title: Title of the product.
- product_category: Broad product category that can be used to group reviews (also used to group the dataset into coherent parts).
- star_rating: The 1-5 star rating of the review.
- helpful_votes: Number of helpful votes.
- total_votes: Number of total votes the review received.
- vine: Review was written as part of the Vine program.
- verified_purchase: The review is on a verified purchase.
- review_headline: The title of the review.
- review_body: The review text.
- review_date: The date the review was written.
```

- **许可**：无已知许可
- **版本**：0.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 1752932

- **特征**：

```json
{
    "marketplace": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "customer_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
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
    "product_parent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_category": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "star_rating": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "helpful_votes": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "total_votes": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "vine": {
        "num_classes": 2,
        "names": [
            "N",
            "Y"
        ],
        "id": null,
        "_type": "ClassLabel"
    },
    "verified_purchase": {
        "num_classes": 2,
        "names": [
            "N",
            "Y"
        ],
        "id": null,
        "_type": "ClassLabel"
    },
    "review_headline": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_body": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_date": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## Home_Improvement_v1_00

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:amazon_us_reviews/Home_Improvement_v1_00')
```

- **说明**：

```
Amazon Customer Reviews (a.k.a. Product Reviews) is one of Amazons iconic products. In a period of over two decades since the first review in 1995, millions of Amazon customers have contributed over a hundred million reviews to express opinions and describe their experiences regarding products on the Amazon.com website. This makes Amazon Customer Reviews a rich source of information for academic researchers in the fields of Natural Language Processing (NLP), Information Retrieval (IR), and Machine Learning (ML), amongst others. Accordingly, we are releasing this data to further research in multiple disciplines related to understanding customer product experiences. Specifically, this dataset was constructed to represent a sample of customer evaluations and opinions, variation in the perception of a product across geographical regions, and promotional intent or bias in reviews.

Over 130+ million customer reviews are available to researchers as part of this release. The data is available in TSV files in the amazon-reviews-pds S3 bucket in AWS US East Region. Each line in the data files corresponds to an individual review (tab delimited, with no quote and escape characters).

Each Dataset contains the following columns:

- marketplace: 2 letter country code of the marketplace where the review was written.
- customer_id: Random identifier that can be used to aggregate reviews written by a single author.
- review_id: The unique ID of the review.
- product_id: The unique Product ID the review pertains to. In the multilingual dataset the reviews for the same product in different countries can be grouped by the same product_id.
- product_parent: Random identifier that can be used to aggregate reviews for the same product.
- product_title: Title of the product.
- product_category: Broad product category that can be used to group reviews (also used to group the dataset into coherent parts).
- star_rating: The 1-5 star rating of the review.
- helpful_votes: Number of helpful votes.
- total_votes: Number of total votes the review received.
- vine: Review was written as part of the Vine program.
- verified_purchase: The review is on a verified purchase.
- review_headline: The title of the review.
- review_body: The review text.
- review_date: The date the review was written.
```

- **许可**：无已知许可
- **版本**：0.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 2634781

- **特征**：

```json
{
    "marketplace": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "customer_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
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
    "product_parent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_category": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "star_rating": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "helpful_votes": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "total_votes": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "vine": {
        "num_classes": 2,
        "names": [
            "N",
            "Y"
        ],
        "id": null,
        "_type": "ClassLabel"
    },
    "verified_purchase": {
        "num_classes": 2,
        "names": [
            "N",
            "Y"
        ],
        "id": null,
        "_type": "ClassLabel"
    },
    "review_headline": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_body": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_date": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## Camera_v1_00

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:amazon_us_reviews/Camera_v1_00')
```

- **说明**：

```
Amazon Customer Reviews (a.k.a. Product Reviews) is one of Amazons iconic products. In a period of over two decades since the first review in 1995, millions of Amazon customers have contributed over a hundred million reviews to express opinions and describe their experiences regarding products on the Amazon.com website. This makes Amazon Customer Reviews a rich source of information for academic researchers in the fields of Natural Language Processing (NLP), Information Retrieval (IR), and Machine Learning (ML), amongst others. Accordingly, we are releasing this data to further research in multiple disciplines related to understanding customer product experiences. Specifically, this dataset was constructed to represent a sample of customer evaluations and opinions, variation in the perception of a product across geographical regions, and promotional intent or bias in reviews.

Over 130+ million customer reviews are available to researchers as part of this release. The data is available in TSV files in the amazon-reviews-pds S3 bucket in AWS US East Region. Each line in the data files corresponds to an individual review (tab delimited, with no quote and escape characters).

Each Dataset contains the following columns:

- marketplace: 2 letter country code of the marketplace where the review was written.
- customer_id: Random identifier that can be used to aggregate reviews written by a single author.
- review_id: The unique ID of the review.
- product_id: The unique Product ID the review pertains to. In the multilingual dataset the reviews for the same product in different countries can be grouped by the same product_id.
- product_parent: Random identifier that can be used to aggregate reviews for the same product.
- product_title: Title of the product.
- product_category: Broad product category that can be used to group reviews (also used to group the dataset into coherent parts).
- star_rating: The 1-5 star rating of the review.
- helpful_votes: Number of helpful votes.
- total_votes: Number of total votes the review received.
- vine: Review was written as part of the Vine program.
- verified_purchase: The review is on a verified purchase.
- review_headline: The title of the review.
- review_body: The review text.
- review_date: The date the review was written.
```

- **许可**：无已知许可
- **版本**：0.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 1801974

- **特征**：

```json
{
    "marketplace": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "customer_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
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
    "product_parent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_category": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "star_rating": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "helpful_votes": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "total_votes": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "vine": {
        "num_classes": 2,
        "names": [
            "N",
            "Y"
        ],
        "id": null,
        "_type": "ClassLabel"
    },
    "verified_purchase": {
        "num_classes": 2,
        "names": [
            "N",
            "Y"
        ],
        "id": null,
        "_type": "ClassLabel"
    },
    "review_headline": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_body": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_date": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## Lawn_and_Garden_v1_00

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:amazon_us_reviews/Lawn_and_Garden_v1_00')
```

- **说明**：

```
Amazon Customer Reviews (a.k.a. Product Reviews) is one of Amazons iconic products. In a period of over two decades since the first review in 1995, millions of Amazon customers have contributed over a hundred million reviews to express opinions and describe their experiences regarding products on the Amazon.com website. This makes Amazon Customer Reviews a rich source of information for academic researchers in the fields of Natural Language Processing (NLP), Information Retrieval (IR), and Machine Learning (ML), amongst others. Accordingly, we are releasing this data to further research in multiple disciplines related to understanding customer product experiences. Specifically, this dataset was constructed to represent a sample of customer evaluations and opinions, variation in the perception of a product across geographical regions, and promotional intent or bias in reviews.

Over 130+ million customer reviews are available to researchers as part of this release. The data is available in TSV files in the amazon-reviews-pds S3 bucket in AWS US East Region. Each line in the data files corresponds to an individual review (tab delimited, with no quote and escape characters).

Each Dataset contains the following columns:

- marketplace: 2 letter country code of the marketplace where the review was written.
- customer_id: Random identifier that can be used to aggregate reviews written by a single author.
- review_id: The unique ID of the review.
- product_id: The unique Product ID the review pertains to. In the multilingual dataset the reviews for the same product in different countries can be grouped by the same product_id.
- product_parent: Random identifier that can be used to aggregate reviews for the same product.
- product_title: Title of the product.
- product_category: Broad product category that can be used to group reviews (also used to group the dataset into coherent parts).
- star_rating: The 1-5 star rating of the review.
- helpful_votes: Number of helpful votes.
- total_votes: Number of total votes the review received.
- vine: Review was written as part of the Vine program.
- verified_purchase: The review is on a verified purchase.
- review_headline: The title of the review.
- review_body: The review text.
- review_date: The date the review was written.
```

- **许可**：无已知许可
- **版本**：0.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 2557288

- **特征**：

```json
{
    "marketplace": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "customer_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
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
    "product_parent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_category": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "star_rating": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "helpful_votes": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "total_votes": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "vine": {
        "num_classes": 2,
        "names": [
            "N",
            "Y"
        ],
        "id": null,
        "_type": "ClassLabel"
    },
    "verified_purchase": {
        "num_classes": 2,
        "names": [
            "N",
            "Y"
        ],
        "id": null,
        "_type": "ClassLabel"
    },
    "review_headline": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_body": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_date": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## Office_Products_v1_00

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:amazon_us_reviews/Office_Products_v1_00')
```

- **说明**：

```
Amazon Customer Reviews (a.k.a. Product Reviews) is one of Amazons iconic products. In a period of over two decades since the first review in 1995, millions of Amazon customers have contributed over a hundred million reviews to express opinions and describe their experiences regarding products on the Amazon.com website. This makes Amazon Customer Reviews a rich source of information for academic researchers in the fields of Natural Language Processing (NLP), Information Retrieval (IR), and Machine Learning (ML), amongst others. Accordingly, we are releasing this data to further research in multiple disciplines related to understanding customer product experiences. Specifically, this dataset was constructed to represent a sample of customer evaluations and opinions, variation in the perception of a product across geographical regions, and promotional intent or bias in reviews.

Over 130+ million customer reviews are available to researchers as part of this release. The data is available in TSV files in the amazon-reviews-pds S3 bucket in AWS US East Region. Each line in the data files corresponds to an individual review (tab delimited, with no quote and escape characters).

Each Dataset contains the following columns:

- marketplace: 2 letter country code of the marketplace where the review was written.
- customer_id: Random identifier that can be used to aggregate reviews written by a single author.
- review_id: The unique ID of the review.
- product_id: The unique Product ID the review pertains to. In the multilingual dataset the reviews for the same product in different countries can be grouped by the same product_id.
- product_parent: Random identifier that can be used to aggregate reviews for the same product.
- product_title: Title of the product.
- product_category: Broad product category that can be used to group reviews (also used to group the dataset into coherent parts).
- star_rating: The 1-5 star rating of the review.
- helpful_votes: Number of helpful votes.
- total_votes: Number of total votes the review received.
- vine: Review was written as part of the Vine program.
- verified_purchase: The review is on a verified purchase.
- review_headline: The title of the review.
- review_body: The review text.
- review_date: The date the review was written.
```

- **许可**：无已知许可
- **版本**：0.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 2642434

- **特征**：

```json
{
    "marketplace": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "customer_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
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
    "product_parent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_category": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "star_rating": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "helpful_votes": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "total_votes": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "vine": {
        "num_classes": 2,
        "names": [
            "N",
            "Y"
        ],
        "id": null,
        "_type": "ClassLabel"
    },
    "verified_purchase": {
        "num_classes": 2,
        "names": [
            "N",
            "Y"
        ],
        "id": null,
        "_type": "ClassLabel"
    },
    "review_headline": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_body": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_date": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## Electronics_v1_00

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:amazon_us_reviews/Electronics_v1_00')
```

- **说明**：

```
Amazon Customer Reviews (a.k.a. Product Reviews) is one of Amazons iconic products. In a period of over two decades since the first review in 1995, millions of Amazon customers have contributed over a hundred million reviews to express opinions and describe their experiences regarding products on the Amazon.com website. This makes Amazon Customer Reviews a rich source of information for academic researchers in the fields of Natural Language Processing (NLP), Information Retrieval (IR), and Machine Learning (ML), amongst others. Accordingly, we are releasing this data to further research in multiple disciplines related to understanding customer product experiences. Specifically, this dataset was constructed to represent a sample of customer evaluations and opinions, variation in the perception of a product across geographical regions, and promotional intent or bias in reviews.

Over 130+ million customer reviews are available to researchers as part of this release. The data is available in TSV files in the amazon-reviews-pds S3 bucket in AWS US East Region. Each line in the data files corresponds to an individual review (tab delimited, with no quote and escape characters).

Each Dataset contains the following columns:

- marketplace: 2 letter country code of the marketplace where the review was written.
- customer_id: Random identifier that can be used to aggregate reviews written by a single author.
- review_id: The unique ID of the review.
- product_id: The unique Product ID the review pertains to. In the multilingual dataset the reviews for the same product in different countries can be grouped by the same product_id.
- product_parent: Random identifier that can be used to aggregate reviews for the same product.
- product_title: Title of the product.
- product_category: Broad product category that can be used to group reviews (also used to group the dataset into coherent parts).
- star_rating: The 1-5 star rating of the review.
- helpful_votes: Number of helpful votes.
- total_votes: Number of total votes the review received.
- vine: Review was written as part of the Vine program.
- verified_purchase: The review is on a verified purchase.
- review_headline: The title of the review.
- review_body: The review text.
- review_date: The date the review was written.
```

- **许可**：无已知许可
- **版本**：0.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 3093869

- **特征**：

```json
{
    "marketplace": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "customer_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
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
    "product_parent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_category": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "star_rating": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "helpful_votes": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "total_votes": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "vine": {
        "num_classes": 2,
        "names": [
            "N",
            "Y"
        ],
        "id": null,
        "_type": "ClassLabel"
    },
    "verified_purchase": {
        "num_classes": 2,
        "names": [
            "N",
            "Y"
        ],
        "id": null,
        "_type": "ClassLabel"
    },
    "review_headline": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_body": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_date": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## Automotive_v1_00

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:amazon_us_reviews/Automotive_v1_00')
```

- **说明**：

```
Amazon Customer Reviews (a.k.a. Product Reviews) is one of Amazons iconic products. In a period of over two decades since the first review in 1995, millions of Amazon customers have contributed over a hundred million reviews to express opinions and describe their experiences regarding products on the Amazon.com website. This makes Amazon Customer Reviews a rich source of information for academic researchers in the fields of Natural Language Processing (NLP), Information Retrieval (IR), and Machine Learning (ML), amongst others. Accordingly, we are releasing this data to further research in multiple disciplines related to understanding customer product experiences. Specifically, this dataset was constructed to represent a sample of customer evaluations and opinions, variation in the perception of a product across geographical regions, and promotional intent or bias in reviews.

Over 130+ million customer reviews are available to researchers as part of this release. The data is available in TSV files in the amazon-reviews-pds S3 bucket in AWS US East Region. Each line in the data files corresponds to an individual review (tab delimited, with no quote and escape characters).

Each Dataset contains the following columns:

- marketplace: 2 letter country code of the marketplace where the review was written.
- customer_id: Random identifier that can be used to aggregate reviews written by a single author.
- review_id: The unique ID of the review.
- product_id: The unique Product ID the review pertains to. In the multilingual dataset the reviews for the same product in different countries can be grouped by the same product_id.
- product_parent: Random identifier that can be used to aggregate reviews for the same product.
- product_title: Title of the product.
- product_category: Broad product category that can be used to group reviews (also used to group the dataset into coherent parts).
- star_rating: The 1-5 star rating of the review.
- helpful_votes: Number of helpful votes.
- total_votes: Number of total votes the review received.
- vine: Review was written as part of the Vine program.
- verified_purchase: The review is on a verified purchase.
- review_headline: The title of the review.
- review_body: The review text.
- review_date: The date the review was written.
```

- **许可**：无已知许可
- **版本**：0.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 3514942

- **特征**：

```json
{
    "marketplace": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "customer_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
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
    "product_parent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_category": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "star_rating": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "helpful_votes": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "total_votes": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "vine": {
        "num_classes": 2,
        "names": [
            "N",
            "Y"
        ],
        "id": null,
        "_type": "ClassLabel"
    },
    "verified_purchase": {
        "num_classes": 2,
        "names": [
            "N",
            "Y"
        ],
        "id": null,
        "_type": "ClassLabel"
    },
    "review_headline": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_body": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_date": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## Digital_Video_Download_v1_00

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:amazon_us_reviews/Digital_Video_Download_v1_00')
```

- **说明**：

```
Amazon Customer Reviews (a.k.a. Product Reviews) is one of Amazons iconic products. In a period of over two decades since the first review in 1995, millions of Amazon customers have contributed over a hundred million reviews to express opinions and describe their experiences regarding products on the Amazon.com website. This makes Amazon Customer Reviews a rich source of information for academic researchers in the fields of Natural Language Processing (NLP), Information Retrieval (IR), and Machine Learning (ML), amongst others. Accordingly, we are releasing this data to further research in multiple disciplines related to understanding customer product experiences. Specifically, this dataset was constructed to represent a sample of customer evaluations and opinions, variation in the perception of a product across geographical regions, and promotional intent or bias in reviews.

Over 130+ million customer reviews are available to researchers as part of this release. The data is available in TSV files in the amazon-reviews-pds S3 bucket in AWS US East Region. Each line in the data files corresponds to an individual review (tab delimited, with no quote and escape characters).

Each Dataset contains the following columns:

- marketplace: 2 letter country code of the marketplace where the review was written.
- customer_id: Random identifier that can be used to aggregate reviews written by a single author.
- review_id: The unique ID of the review.
- product_id: The unique Product ID the review pertains to. In the multilingual dataset the reviews for the same product in different countries can be grouped by the same product_id.
- product_parent: Random identifier that can be used to aggregate reviews for the same product.
- product_title: Title of the product.
- product_category: Broad product category that can be used to group reviews (also used to group the dataset into coherent parts).
- star_rating: The 1-5 star rating of the review.
- helpful_votes: Number of helpful votes.
- total_votes: Number of total votes the review received.
- vine: Review was written as part of the Vine program.
- verified_purchase: The review is on a verified purchase.
- review_headline: The title of the review.
- review_body: The review text.
- review_date: The date the review was written.
```

- **许可**：无已知许可
- **版本**：0.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 4057147

- **特征**：

```json
{
    "marketplace": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "customer_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
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
    "product_parent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_category": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "star_rating": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "helpful_votes": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "total_votes": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "vine": {
        "num_classes": 2,
        "names": [
            "N",
            "Y"
        ],
        "id": null,
        "_type": "ClassLabel"
    },
    "verified_purchase": {
        "num_classes": 2,
        "names": [
            "N",
            "Y"
        ],
        "id": null,
        "_type": "ClassLabel"
    },
    "review_headline": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_body": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_date": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## Mobile_Apps_v1_00

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:amazon_us_reviews/Mobile_Apps_v1_00')
```

- **说明**：

```
Amazon Customer Reviews (a.k.a. Product Reviews) is one of Amazons iconic products. In a period of over two decades since the first review in 1995, millions of Amazon customers have contributed over a hundred million reviews to express opinions and describe their experiences regarding products on the Amazon.com website. This makes Amazon Customer Reviews a rich source of information for academic researchers in the fields of Natural Language Processing (NLP), Information Retrieval (IR), and Machine Learning (ML), amongst others. Accordingly, we are releasing this data to further research in multiple disciplines related to understanding customer product experiences. Specifically, this dataset was constructed to represent a sample of customer evaluations and opinions, variation in the perception of a product across geographical regions, and promotional intent or bias in reviews.

Over 130+ million customer reviews are available to researchers as part of this release. The data is available in TSV files in the amazon-reviews-pds S3 bucket in AWS US East Region. Each line in the data files corresponds to an individual review (tab delimited, with no quote and escape characters).

Each Dataset contains the following columns:

- marketplace: 2 letter country code of the marketplace where the review was written.
- customer_id: Random identifier that can be used to aggregate reviews written by a single author.
- review_id: The unique ID of the review.
- product_id: The unique Product ID the review pertains to. In the multilingual dataset the reviews for the same product in different countries can be grouped by the same product_id.
- product_parent: Random identifier that can be used to aggregate reviews for the same product.
- product_title: Title of the product.
- product_category: Broad product category that can be used to group reviews (also used to group the dataset into coherent parts).
- star_rating: The 1-5 star rating of the review.
- helpful_votes: Number of helpful votes.
- total_votes: Number of total votes the review received.
- vine: Review was written as part of the Vine program.
- verified_purchase: The review is on a verified purchase.
- review_headline: The title of the review.
- review_body: The review text.
- review_date: The date the review was written.
```

- **许可**：无已知许可
- **版本**：0.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 5033376

- **特征**：

```json
{
    "marketplace": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "customer_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
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
    "product_parent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_category": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "star_rating": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "helpful_votes": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "total_votes": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "vine": {
        "num_classes": 2,
        "names": [
            "N",
            "Y"
        ],
        "id": null,
        "_type": "ClassLabel"
    },
    "verified_purchase": {
        "num_classes": 2,
        "names": [
            "N",
            "Y"
        ],
        "id": null,
        "_type": "ClassLabel"
    },
    "review_headline": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_body": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_date": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## Shoes_v1_00

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:amazon_us_reviews/Shoes_v1_00')
```

- **说明**：

```
Amazon Customer Reviews (a.k.a. Product Reviews) is one of Amazons iconic products. In a period of over two decades since the first review in 1995, millions of Amazon customers have contributed over a hundred million reviews to express opinions and describe their experiences regarding products on the Amazon.com website. This makes Amazon Customer Reviews a rich source of information for academic researchers in the fields of Natural Language Processing (NLP), Information Retrieval (IR), and Machine Learning (ML), amongst others. Accordingly, we are releasing this data to further research in multiple disciplines related to understanding customer product experiences. Specifically, this dataset was constructed to represent a sample of customer evaluations and opinions, variation in the perception of a product across geographical regions, and promotional intent or bias in reviews.

Over 130+ million customer reviews are available to researchers as part of this release. The data is available in TSV files in the amazon-reviews-pds S3 bucket in AWS US East Region. Each line in the data files corresponds to an individual review (tab delimited, with no quote and escape characters).

Each Dataset contains the following columns:

- marketplace: 2 letter country code of the marketplace where the review was written.
- customer_id: Random identifier that can be used to aggregate reviews written by a single author.
- review_id: The unique ID of the review.
- product_id: The unique Product ID the review pertains to. In the multilingual dataset the reviews for the same product in different countries can be grouped by the same product_id.
- product_parent: Random identifier that can be used to aggregate reviews for the same product.
- product_title: Title of the product.
- product_category: Broad product category that can be used to group reviews (also used to group the dataset into coherent parts).
- star_rating: The 1-5 star rating of the review.
- helpful_votes: Number of helpful votes.
- total_votes: Number of total votes the review received.
- vine: Review was written as part of the Vine program.
- verified_purchase: The review is on a verified purchase.
- review_headline: The title of the review.
- review_body: The review text.
- review_date: The date the review was written.
```

- **许可**：无已知许可
- **版本**：0.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 4366916

- **特征**：

```json
{
    "marketplace": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "customer_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
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
    "product_parent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_category": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "star_rating": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "helpful_votes": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "total_votes": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "vine": {
        "num_classes": 2,
        "names": [
            "N",
            "Y"
        ],
        "id": null,
        "_type": "ClassLabel"
    },
    "verified_purchase": {
        "num_classes": 2,
        "names": [
            "N",
            "Y"
        ],
        "id": null,
        "_type": "ClassLabel"
    },
    "review_headline": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_body": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_date": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## Toys_v1_00

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:amazon_us_reviews/Toys_v1_00')
```

- **说明**：

```
Amazon Customer Reviews (a.k.a. Product Reviews) is one of Amazons iconic products. In a period of over two decades since the first review in 1995, millions of Amazon customers have contributed over a hundred million reviews to express opinions and describe their experiences regarding products on the Amazon.com website. This makes Amazon Customer Reviews a rich source of information for academic researchers in the fields of Natural Language Processing (NLP), Information Retrieval (IR), and Machine Learning (ML), amongst others. Accordingly, we are releasing this data to further research in multiple disciplines related to understanding customer product experiences. Specifically, this dataset was constructed to represent a sample of customer evaluations and opinions, variation in the perception of a product across geographical regions, and promotional intent or bias in reviews.

Over 130+ million customer reviews are available to researchers as part of this release. The data is available in TSV files in the amazon-reviews-pds S3 bucket in AWS US East Region. Each line in the data files corresponds to an individual review (tab delimited, with no quote and escape characters).

Each Dataset contains the following columns:

- marketplace: 2 letter country code of the marketplace where the review was written.
- customer_id: Random identifier that can be used to aggregate reviews written by a single author.
- review_id: The unique ID of the review.
- product_id: The unique Product ID the review pertains to. In the multilingual dataset the reviews for the same product in different countries can be grouped by the same product_id.
- product_parent: Random identifier that can be used to aggregate reviews for the same product.
- product_title: Title of the product.
- product_category: Broad product category that can be used to group reviews (also used to group the dataset into coherent parts).
- star_rating: The 1-5 star rating of the review.
- helpful_votes: Number of helpful votes.
- total_votes: Number of total votes the review received.
- vine: Review was written as part of the Vine program.
- verified_purchase: The review is on a verified purchase.
- review_headline: The title of the review.
- review_body: The review text.
- review_date: The date the review was written.
```

- **许可**：无已知许可
- **版本**：0.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 4864249

- **特征**：

```json
{
    "marketplace": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "customer_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
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
    "product_parent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_category": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "star_rating": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "helpful_votes": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "total_votes": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "vine": {
        "num_classes": 2,
        "names": [
            "N",
            "Y"
        ],
        "id": null,
        "_type": "ClassLabel"
    },
    "verified_purchase": {
        "num_classes": 2,
        "names": [
            "N",
            "Y"
        ],
        "id": null,
        "_type": "ClassLabel"
    },
    "review_headline": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_body": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_date": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## Sports_v1_00

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:amazon_us_reviews/Sports_v1_00')
```

- **说明**：

```
Amazon Customer Reviews (a.k.a. Product Reviews) is one of Amazons iconic products. In a period of over two decades since the first review in 1995, millions of Amazon customers have contributed over a hundred million reviews to express opinions and describe their experiences regarding products on the Amazon.com website. This makes Amazon Customer Reviews a rich source of information for academic researchers in the fields of Natural Language Processing (NLP), Information Retrieval (IR), and Machine Learning (ML), amongst others. Accordingly, we are releasing this data to further research in multiple disciplines related to understanding customer product experiences. Specifically, this dataset was constructed to represent a sample of customer evaluations and opinions, variation in the perception of a product across geographical regions, and promotional intent or bias in reviews.

Over 130+ million customer reviews are available to researchers as part of this release. The data is available in TSV files in the amazon-reviews-pds S3 bucket in AWS US East Region. Each line in the data files corresponds to an individual review (tab delimited, with no quote and escape characters).

Each Dataset contains the following columns:

- marketplace: 2 letter country code of the marketplace where the review was written.
- customer_id: Random identifier that can be used to aggregate reviews written by a single author.
- review_id: The unique ID of the review.
- product_id: The unique Product ID the review pertains to. In the multilingual dataset the reviews for the same product in different countries can be grouped by the same product_id.
- product_parent: Random identifier that can be used to aggregate reviews for the same product.
- product_title: Title of the product.
- product_category: Broad product category that can be used to group reviews (also used to group the dataset into coherent parts).
- star_rating: The 1-5 star rating of the review.
- helpful_votes: Number of helpful votes.
- total_votes: Number of total votes the review received.
- vine: Review was written as part of the Vine program.
- verified_purchase: The review is on a verified purchase.
- review_headline: The title of the review.
- review_body: The review text.
- review_date: The date the review was written.
```

- **许可**：无已知许可
- **版本**：0.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 4850360

- **特征**：

```json
{
    "marketplace": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "customer_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
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
    "product_parent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_category": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "star_rating": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "helpful_votes": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "total_votes": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "vine": {
        "num_classes": 2,
        "names": [
            "N",
            "Y"
        ],
        "id": null,
        "_type": "ClassLabel"
    },
    "verified_purchase": {
        "num_classes": 2,
        "names": [
            "N",
            "Y"
        ],
        "id": null,
        "_type": "ClassLabel"
    },
    "review_headline": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_body": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_date": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## Kitchen_v1_00

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:amazon_us_reviews/Kitchen_v1_00')
```

- **说明**：

```
Amazon Customer Reviews (a.k.a. Product Reviews) is one of Amazons iconic products. In a period of over two decades since the first review in 1995, millions of Amazon customers have contributed over a hundred million reviews to express opinions and describe their experiences regarding products on the Amazon.com website. This makes Amazon Customer Reviews a rich source of information for academic researchers in the fields of Natural Language Processing (NLP), Information Retrieval (IR), and Machine Learning (ML), amongst others. Accordingly, we are releasing this data to further research in multiple disciplines related to understanding customer product experiences. Specifically, this dataset was constructed to represent a sample of customer evaluations and opinions, variation in the perception of a product across geographical regions, and promotional intent or bias in reviews.

Over 130+ million customer reviews are available to researchers as part of this release. The data is available in TSV files in the amazon-reviews-pds S3 bucket in AWS US East Region. Each line in the data files corresponds to an individual review (tab delimited, with no quote and escape characters).

Each Dataset contains the following columns:

- marketplace: 2 letter country code of the marketplace where the review was written.
- customer_id: Random identifier that can be used to aggregate reviews written by a single author.
- review_id: The unique ID of the review.
- product_id: The unique Product ID the review pertains to. In the multilingual dataset the reviews for the same product in different countries can be grouped by the same product_id.
- product_parent: Random identifier that can be used to aggregate reviews for the same product.
- product_title: Title of the product.
- product_category: Broad product category that can be used to group reviews (also used to group the dataset into coherent parts).
- star_rating: The 1-5 star rating of the review.
- helpful_votes: Number of helpful votes.
- total_votes: Number of total votes the review received.
- vine: Review was written as part of the Vine program.
- verified_purchase: The review is on a verified purchase.
- review_headline: The title of the review.
- review_body: The review text.
- review_date: The date the review was written.
```

- **许可**：无已知许可
- **版本**：0.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 4880466

- **特征**：

```json
{
    "marketplace": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "customer_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
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
    "product_parent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_category": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "star_rating": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "helpful_votes": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "total_votes": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "vine": {
        "num_classes": 2,
        "names": [
            "N",
            "Y"
        ],
        "id": null,
        "_type": "ClassLabel"
    },
    "verified_purchase": {
        "num_classes": 2,
        "names": [
            "N",
            "Y"
        ],
        "id": null,
        "_type": "ClassLabel"
    },
    "review_headline": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_body": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_date": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## Beauty_v1_00

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:amazon_us_reviews/Beauty_v1_00')
```

- **说明**：

```
Amazon Customer Reviews (a.k.a. Product Reviews) is one of Amazons iconic products. In a period of over two decades since the first review in 1995, millions of Amazon customers have contributed over a hundred million reviews to express opinions and describe their experiences regarding products on the Amazon.com website. This makes Amazon Customer Reviews a rich source of information for academic researchers in the fields of Natural Language Processing (NLP), Information Retrieval (IR), and Machine Learning (ML), amongst others. Accordingly, we are releasing this data to further research in multiple disciplines related to understanding customer product experiences. Specifically, this dataset was constructed to represent a sample of customer evaluations and opinions, variation in the perception of a product across geographical regions, and promotional intent or bias in reviews.

Over 130+ million customer reviews are available to researchers as part of this release. The data is available in TSV files in the amazon-reviews-pds S3 bucket in AWS US East Region. Each line in the data files corresponds to an individual review (tab delimited, with no quote and escape characters).

Each Dataset contains the following columns:

- marketplace: 2 letter country code of the marketplace where the review was written.
- customer_id: Random identifier that can be used to aggregate reviews written by a single author.
- review_id: The unique ID of the review.
- product_id: The unique Product ID the review pertains to. In the multilingual dataset the reviews for the same product in different countries can be grouped by the same product_id.
- product_parent: Random identifier that can be used to aggregate reviews for the same product.
- product_title: Title of the product.
- product_category: Broad product category that can be used to group reviews (also used to group the dataset into coherent parts).
- star_rating: The 1-5 star rating of the review.
- helpful_votes: Number of helpful votes.
- total_votes: Number of total votes the review received.
- vine: Review was written as part of the Vine program.
- verified_purchase: The review is on a verified purchase.
- review_headline: The title of the review.
- review_body: The review text.
- review_date: The date the review was written.
```

- **许可**：无已知许可
- **版本**：0.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 5115666

- **特征**：

```json
{
    "marketplace": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "customer_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
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
    "product_parent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_category": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "star_rating": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "helpful_votes": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "total_votes": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "vine": {
        "num_classes": 2,
        "names": [
            "N",
            "Y"
        ],
        "id": null,
        "_type": "ClassLabel"
    },
    "verified_purchase": {
        "num_classes": 2,
        "names": [
            "N",
            "Y"
        ],
        "id": null,
        "_type": "ClassLabel"
    },
    "review_headline": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_body": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_date": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## Music_v1_00

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:amazon_us_reviews/Music_v1_00')
```

- **说明**：

```
Amazon Customer Reviews (a.k.a. Product Reviews) is one of Amazons iconic products. In a period of over two decades since the first review in 1995, millions of Amazon customers have contributed over a hundred million reviews to express opinions and describe their experiences regarding products on the Amazon.com website. This makes Amazon Customer Reviews a rich source of information for academic researchers in the fields of Natural Language Processing (NLP), Information Retrieval (IR), and Machine Learning (ML), amongst others. Accordingly, we are releasing this data to further research in multiple disciplines related to understanding customer product experiences. Specifically, this dataset was constructed to represent a sample of customer evaluations and opinions, variation in the perception of a product across geographical regions, and promotional intent or bias in reviews.

Over 130+ million customer reviews are available to researchers as part of this release. The data is available in TSV files in the amazon-reviews-pds S3 bucket in AWS US East Region. Each line in the data files corresponds to an individual review (tab delimited, with no quote and escape characters).

Each Dataset contains the following columns:

- marketplace: 2 letter country code of the marketplace where the review was written.
- customer_id: Random identifier that can be used to aggregate reviews written by a single author.
- review_id: The unique ID of the review.
- product_id: The unique Product ID the review pertains to. In the multilingual dataset the reviews for the same product in different countries can be grouped by the same product_id.
- product_parent: Random identifier that can be used to aggregate reviews for the same product.
- product_title: Title of the product.
- product_category: Broad product category that can be used to group reviews (also used to group the dataset into coherent parts).
- star_rating: The 1-5 star rating of the review.
- helpful_votes: Number of helpful votes.
- total_votes: Number of total votes the review received.
- vine: Review was written as part of the Vine program.
- verified_purchase: The review is on a verified purchase.
- review_headline: The title of the review.
- review_body: The review text.
- review_date: The date the review was written.
```

- **许可**：无已知许可
- **版本**：0.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 4751577

- **特征**：

```json
{
    "marketplace": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "customer_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
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
    "product_parent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_category": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "star_rating": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "helpful_votes": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "total_votes": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "vine": {
        "num_classes": 2,
        "names": [
            "N",
            "Y"
        ],
        "id": null,
        "_type": "ClassLabel"
    },
    "verified_purchase": {
        "num_classes": 2,
        "names": [
            "N",
            "Y"
        ],
        "id": null,
        "_type": "ClassLabel"
    },
    "review_headline": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_body": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_date": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## Health_Personal_Care_v1_00

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:amazon_us_reviews/Health_Personal_Care_v1_00')
```

- **说明**：

```
Amazon Customer Reviews (a.k.a. Product Reviews) is one of Amazons iconic products. In a period of over two decades since the first review in 1995, millions of Amazon customers have contributed over a hundred million reviews to express opinions and describe their experiences regarding products on the Amazon.com website. This makes Amazon Customer Reviews a rich source of information for academic researchers in the fields of Natural Language Processing (NLP), Information Retrieval (IR), and Machine Learning (ML), amongst others. Accordingly, we are releasing this data to further research in multiple disciplines related to understanding customer product experiences. Specifically, this dataset was constructed to represent a sample of customer evaluations and opinions, variation in the perception of a product across geographical regions, and promotional intent or bias in reviews.

Over 130+ million customer reviews are available to researchers as part of this release. The data is available in TSV files in the amazon-reviews-pds S3 bucket in AWS US East Region. Each line in the data files corresponds to an individual review (tab delimited, with no quote and escape characters).

Each Dataset contains the following columns:

- marketplace: 2 letter country code of the marketplace where the review was written.
- customer_id: Random identifier that can be used to aggregate reviews written by a single author.
- review_id: The unique ID of the review.
- product_id: The unique Product ID the review pertains to. In the multilingual dataset the reviews for the same product in different countries can be grouped by the same product_id.
- product_parent: Random identifier that can be used to aggregate reviews for the same product.
- product_title: Title of the product.
- product_category: Broad product category that can be used to group reviews (also used to group the dataset into coherent parts).
- star_rating: The 1-5 star rating of the review.
- helpful_votes: Number of helpful votes.
- total_votes: Number of total votes the review received.
- vine: Review was written as part of the Vine program.
- verified_purchase: The review is on a verified purchase.
- review_headline: The title of the review.
- review_body: The review text.
- review_date: The date the review was written.
```

- **许可**：无已知许可
- **版本**：0.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 5331449

- **特征**：

```json
{
    "marketplace": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "customer_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
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
    "product_parent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_category": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "star_rating": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "helpful_votes": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "total_votes": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "vine": {
        "num_classes": 2,
        "names": [
            "N",
            "Y"
        ],
        "id": null,
        "_type": "ClassLabel"
    },
    "verified_purchase": {
        "num_classes": 2,
        "names": [
            "N",
            "Y"
        ],
        "id": null,
        "_type": "ClassLabel"
    },
    "review_headline": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_body": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_date": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## Digital_Ebook_Purchase_v1_01

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:amazon_us_reviews/Digital_Ebook_Purchase_v1_01')
```

- **说明**：

```
Amazon Customer Reviews (a.k.a. Product Reviews) is one of Amazons iconic products. In a period of over two decades since the first review in 1995, millions of Amazon customers have contributed over a hundred million reviews to express opinions and describe their experiences regarding products on the Amazon.com website. This makes Amazon Customer Reviews a rich source of information for academic researchers in the fields of Natural Language Processing (NLP), Information Retrieval (IR), and Machine Learning (ML), amongst others. Accordingly, we are releasing this data to further research in multiple disciplines related to understanding customer product experiences. Specifically, this dataset was constructed to represent a sample of customer evaluations and opinions, variation in the perception of a product across geographical regions, and promotional intent or bias in reviews.

Over 130+ million customer reviews are available to researchers as part of this release. The data is available in TSV files in the amazon-reviews-pds S3 bucket in AWS US East Region. Each line in the data files corresponds to an individual review (tab delimited, with no quote and escape characters).

Each Dataset contains the following columns:

- marketplace: 2 letter country code of the marketplace where the review was written.
- customer_id: Random identifier that can be used to aggregate reviews written by a single author.
- review_id: The unique ID of the review.
- product_id: The unique Product ID the review pertains to. In the multilingual dataset the reviews for the same product in different countries can be grouped by the same product_id.
- product_parent: Random identifier that can be used to aggregate reviews for the same product.
- product_title: Title of the product.
- product_category: Broad product category that can be used to group reviews (also used to group the dataset into coherent parts).
- star_rating: The 1-5 star rating of the review.
- helpful_votes: Number of helpful votes.
- total_votes: Number of total votes the review received.
- vine: Review was written as part of the Vine program.
- verified_purchase: The review is on a verified purchase.
- review_headline: The title of the review.
- review_body: The review text.
- review_date: The date the review was written.
```

- **许可**：无已知许可
- **版本**：0.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 5101693

- **特征**：

```json
{
    "marketplace": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "customer_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
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
    "product_parent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_category": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "star_rating": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "helpful_votes": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "total_votes": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "vine": {
        "num_classes": 2,
        "names": [
            "N",
            "Y"
        ],
        "id": null,
        "_type": "ClassLabel"
    },
    "verified_purchase": {
        "num_classes": 2,
        "names": [
            "N",
            "Y"
        ],
        "id": null,
        "_type": "ClassLabel"
    },
    "review_headline": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_body": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_date": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## Home_v1_00

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:amazon_us_reviews/Home_v1_00')
```

- **说明**：

```
Amazon Customer Reviews (a.k.a. Product Reviews) is one of Amazons iconic products. In a period of over two decades since the first review in 1995, millions of Amazon customers have contributed over a hundred million reviews to express opinions and describe their experiences regarding products on the Amazon.com website. This makes Amazon Customer Reviews a rich source of information for academic researchers in the fields of Natural Language Processing (NLP), Information Retrieval (IR), and Machine Learning (ML), amongst others. Accordingly, we are releasing this data to further research in multiple disciplines related to understanding customer product experiences. Specifically, this dataset was constructed to represent a sample of customer evaluations and opinions, variation in the perception of a product across geographical regions, and promotional intent or bias in reviews.

Over 130+ million customer reviews are available to researchers as part of this release. The data is available in TSV files in the amazon-reviews-pds S3 bucket in AWS US East Region. Each line in the data files corresponds to an individual review (tab delimited, with no quote and escape characters).

Each Dataset contains the following columns:

- marketplace: 2 letter country code of the marketplace where the review was written.
- customer_id: Random identifier that can be used to aggregate reviews written by a single author.
- review_id: The unique ID of the review.
- product_id: The unique Product ID the review pertains to. In the multilingual dataset the reviews for the same product in different countries can be grouped by the same product_id.
- product_parent: Random identifier that can be used to aggregate reviews for the same product.
- product_title: Title of the product.
- product_category: Broad product category that can be used to group reviews (also used to group the dataset into coherent parts).
- star_rating: The 1-5 star rating of the review.
- helpful_votes: Number of helpful votes.
- total_votes: Number of total votes the review received.
- vine: Review was written as part of the Vine program.
- verified_purchase: The review is on a verified purchase.
- review_headline: The title of the review.
- review_body: The review text.
- review_date: The date the review was written.
```

- **许可**：无已知许可
- **版本**：0.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 6221559

- **特征**：

```json
{
    "marketplace": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "customer_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
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
    "product_parent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_category": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "star_rating": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "helpful_votes": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "total_votes": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "vine": {
        "num_classes": 2,
        "names": [
            "N",
            "Y"
        ],
        "id": null,
        "_type": "ClassLabel"
    },
    "verified_purchase": {
        "num_classes": 2,
        "names": [
            "N",
            "Y"
        ],
        "id": null,
        "_type": "ClassLabel"
    },
    "review_headline": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_body": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_date": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## Wireless_v1_00

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:amazon_us_reviews/Wireless_v1_00')
```

- **说明**：

```
Amazon Customer Reviews (a.k.a. Product Reviews) is one of Amazons iconic products. In a period of over two decades since the first review in 1995, millions of Amazon customers have contributed over a hundred million reviews to express opinions and describe their experiences regarding products on the Amazon.com website. This makes Amazon Customer Reviews a rich source of information for academic researchers in the fields of Natural Language Processing (NLP), Information Retrieval (IR), and Machine Learning (ML), amongst others. Accordingly, we are releasing this data to further research in multiple disciplines related to understanding customer product experiences. Specifically, this dataset was constructed to represent a sample of customer evaluations and opinions, variation in the perception of a product across geographical regions, and promotional intent or bias in reviews.

Over 130+ million customer reviews are available to researchers as part of this release. The data is available in TSV files in the amazon-reviews-pds S3 bucket in AWS US East Region. Each line in the data files corresponds to an individual review (tab delimited, with no quote and escape characters).

Each Dataset contains the following columns:

- marketplace: 2 letter country code of the marketplace where the review was written.
- customer_id: Random identifier that can be used to aggregate reviews written by a single author.
- review_id: The unique ID of the review.
- product_id: The unique Product ID the review pertains to. In the multilingual dataset the reviews for the same product in different countries can be grouped by the same product_id.
- product_parent: Random identifier that can be used to aggregate reviews for the same product.
- product_title: Title of the product.
- product_category: Broad product category that can be used to group reviews (also used to group the dataset into coherent parts).
- star_rating: The 1-5 star rating of the review.
- helpful_votes: Number of helpful votes.
- total_votes: Number of total votes the review received.
- vine: Review was written as part of the Vine program.
- verified_purchase: The review is on a verified purchase.
- review_headline: The title of the review.
- review_body: The review text.
- review_date: The date the review was written.
```

- **许可**：无已知许可
- **版本**：0.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 9002021

- **特征**：

```json
{
    "marketplace": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "customer_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
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
    "product_parent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_category": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "star_rating": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "helpful_votes": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "total_votes": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "vine": {
        "num_classes": 2,
        "names": [
            "N",
            "Y"
        ],
        "id": null,
        "_type": "ClassLabel"
    },
    "verified_purchase": {
        "num_classes": 2,
        "names": [
            "N",
            "Y"
        ],
        "id": null,
        "_type": "ClassLabel"
    },
    "review_headline": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_body": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_date": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## Books_v1_00

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:amazon_us_reviews/Books_v1_00')
```

- **说明**：

```
Amazon Customer Reviews (a.k.a. Product Reviews) is one of Amazons iconic products. In a period of over two decades since the first review in 1995, millions of Amazon customers have contributed over a hundred million reviews to express opinions and describe their experiences regarding products on the Amazon.com website. This makes Amazon Customer Reviews a rich source of information for academic researchers in the fields of Natural Language Processing (NLP), Information Retrieval (IR), and Machine Learning (ML), amongst others. Accordingly, we are releasing this data to further research in multiple disciplines related to understanding customer product experiences. Specifically, this dataset was constructed to represent a sample of customer evaluations and opinions, variation in the perception of a product across geographical regions, and promotional intent or bias in reviews.

Over 130+ million customer reviews are available to researchers as part of this release. The data is available in TSV files in the amazon-reviews-pds S3 bucket in AWS US East Region. Each line in the data files corresponds to an individual review (tab delimited, with no quote and escape characters).

Each Dataset contains the following columns:

- marketplace: 2 letter country code of the marketplace where the review was written.
- customer_id: Random identifier that can be used to aggregate reviews written by a single author.
- review_id: The unique ID of the review.
- product_id: The unique Product ID the review pertains to. In the multilingual dataset the reviews for the same product in different countries can be grouped by the same product_id.
- product_parent: Random identifier that can be used to aggregate reviews for the same product.
- product_title: Title of the product.
- product_category: Broad product category that can be used to group reviews (also used to group the dataset into coherent parts).
- star_rating: The 1-5 star rating of the review.
- helpful_votes: Number of helpful votes.
- total_votes: Number of total votes the review received.
- vine: Review was written as part of the Vine program.
- verified_purchase: The review is on a verified purchase.
- review_headline: The title of the review.
- review_body: The review text.
- review_date: The date the review was written.
```

- **许可**：无已知许可
- **版本**：0.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 10319090

- **特征**：

```json
{
    "marketplace": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "customer_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
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
    "product_parent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_category": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "star_rating": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "helpful_votes": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "total_votes": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "vine": {
        "num_classes": 2,
        "names": [
            "N",
            "Y"
        ],
        "id": null,
        "_type": "ClassLabel"
    },
    "verified_purchase": {
        "num_classes": 2,
        "names": [
            "N",
            "Y"
        ],
        "id": null,
        "_type": "ClassLabel"
    },
    "review_headline": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_body": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_date": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```

## Digital_Ebook_Purchase_v1_00

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:amazon_us_reviews/Digital_Ebook_Purchase_v1_00')
```

- **说明**：

```
Amazon Customer Reviews (a.k.a. Product Reviews) is one of Amazons iconic products. In a period of over two decades since the first review in 1995, millions of Amazon customers have contributed over a hundred million reviews to express opinions and describe their experiences regarding products on the Amazon.com website. This makes Amazon Customer Reviews a rich source of information for academic researchers in the fields of Natural Language Processing (NLP), Information Retrieval (IR), and Machine Learning (ML), amongst others. Accordingly, we are releasing this data to further research in multiple disciplines related to understanding customer product experiences. Specifically, this dataset was constructed to represent a sample of customer evaluations and opinions, variation in the perception of a product across geographical regions, and promotional intent or bias in reviews.

Over 130+ million customer reviews are available to researchers as part of this release. The data is available in TSV files in the amazon-reviews-pds S3 bucket in AWS US East Region. Each line in the data files corresponds to an individual review (tab delimited, with no quote and escape characters).

Each Dataset contains the following columns:

- marketplace: 2 letter country code of the marketplace where the review was written.
- customer_id: Random identifier that can be used to aggregate reviews written by a single author.
- review_id: The unique ID of the review.
- product_id: The unique Product ID the review pertains to. In the multilingual dataset the reviews for the same product in different countries can be grouped by the same product_id.
- product_parent: Random identifier that can be used to aggregate reviews for the same product.
- product_title: Title of the product.
- product_category: Broad product category that can be used to group reviews (also used to group the dataset into coherent parts).
- star_rating: The 1-5 star rating of the review.
- helpful_votes: Number of helpful votes.
- total_votes: Number of total votes the review received.
- vine: Review was written as part of the Vine program.
- verified_purchase: The review is on a verified purchase.
- review_headline: The title of the review.
- review_body: The review text.
- review_date: The date the review was written.
```

- **许可**：无已知许可
- **版本**：0.1.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 12520722

- **特征**：

```json
{
    "marketplace": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "customer_id": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
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
    "product_parent": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "product_category": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "star_rating": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "helpful_votes": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "total_votes": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
    },
    "vine": {
        "num_classes": 2,
        "names": [
            "N",
            "Y"
        ],
        "id": null,
        "_type": "ClassLabel"
    },
    "verified_purchase": {
        "num_classes": 2,
        "names": [
            "N",
            "Y"
        ],
        "id": null,
        "_type": "ClassLabel"
    },
    "review_headline": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_body": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "review_date": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```
