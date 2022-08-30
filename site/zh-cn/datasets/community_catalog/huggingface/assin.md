# assin

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/assin)
- [Huggingface](https://huggingface.co/datasets/assin)

## full

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:assin/full')
```

- **说明**：

```
The ASSIN (Avaliação de Similaridade Semântica e INferência textual) corpus is a corpus annotated with pairs of sentences written in
Portuguese that is suitable for the  exploration of textual entailment and paraphrasing classifiers. The corpus contains pairs of sentences
extracted from news articles written in European Portuguese (EP) and Brazilian Portuguese (BP), obtained from Google News Portugal
and Brazil, respectively. To create the corpus, the authors started by collecting a set of news articles describing the
same event (one news article from Google News Portugal and another from Google News Brazil) from Google News.
Then, they employed Latent Dirichlet Allocation (LDA) models to retrieve pairs of similar sentences between sets of news
articles that were grouped together around the same topic. For that, two LDA models were trained (for EP and for BP)
on external and large-scale collections of unannotated news articles from Portuguese and Brazilian news providers, respectively.
Then, the authors defined a lower and upper threshold for the sentence similarity score of the retrieved pairs of sentences,
taking into account that high similarity scores correspond to sentences that contain almost the same content (paraphrase candidates),
and low similarity scores correspond to sentences that are very different in content from each other (no-relation candidates).
From the collection of pairs of sentences obtained at this stage, the authors performed some manual grammatical corrections
and discarded some of the pairs wrongly retrieved. Furthermore, from a preliminary analysis made to the retrieved sentence pairs
the authors noticed that the number of contradictions retrieved during the previous stage was very low. Additionally, they also
noticed that event though paraphrases are not very frequent, they occur with some frequency in news articles. Consequently,
in contrast with the majority of the currently available corpora for other languages, which consider as labels “neutral”, “entailment”
and “contradiction” for the task of RTE, the authors of the ASSIN corpus decided to use as labels “none”, “entailment” and “paraphrase”.
Finally, the manual annotation of pairs of sentences was performed by human annotators. At least four annotators were randomly
selected to annotate each pair of sentences, which is done in two steps: (i) assigning a semantic similarity label (a score between 1 and 5,
from unrelated to very similar); and (ii) providing an entailment label (one sentence entails the other, sentences are paraphrases,
or no relation). Sentence pairs where at least three annotators do not agree on the entailment label were considered controversial
and thus discarded from the gold standard annotations. The full dataset has 10,000 sentence pairs, half of which in Brazilian Portuguese
and half in European Portuguese. Either language variant has 2,500 pairs for training, 500 for validation and 2,000 for testing.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 4000
`'train'` | 5000
`'validation'` | 1000

- **特征**：

```json
{
    "sentence_pair_id": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "premise": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "hypothesis": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "relatedness_score": {
        "dtype": "float32",
        "id": null,
        "_type": "Value"
    },
    "entailment_judgment": {
        "num_classes": 3,
        "names": [
            "NONE",
            "ENTAILMENT",
            "PARAPHRASE"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```

## ptpt

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:assin/ptpt')
```

- **说明**：

```
The ASSIN (Avaliação de Similaridade Semântica e INferência textual) corpus is a corpus annotated with pairs of sentences written in
Portuguese that is suitable for the  exploration of textual entailment and paraphrasing classifiers. The corpus contains pairs of sentences
extracted from news articles written in European Portuguese (EP) and Brazilian Portuguese (BP), obtained from Google News Portugal
and Brazil, respectively. To create the corpus, the authors started by collecting a set of news articles describing the
same event (one news article from Google News Portugal and another from Google News Brazil) from Google News.
Then, they employed Latent Dirichlet Allocation (LDA) models to retrieve pairs of similar sentences between sets of news
articles that were grouped together around the same topic. For that, two LDA models were trained (for EP and for BP)
on external and large-scale collections of unannotated news articles from Portuguese and Brazilian news providers, respectively.
Then, the authors defined a lower and upper threshold for the sentence similarity score of the retrieved pairs of sentences,
taking into account that high similarity scores correspond to sentences that contain almost the same content (paraphrase candidates),
and low similarity scores correspond to sentences that are very different in content from each other (no-relation candidates).
From the collection of pairs of sentences obtained at this stage, the authors performed some manual grammatical corrections
and discarded some of the pairs wrongly retrieved. Furthermore, from a preliminary analysis made to the retrieved sentence pairs
the authors noticed that the number of contradictions retrieved during the previous stage was very low. Additionally, they also
noticed that event though paraphrases are not very frequent, they occur with some frequency in news articles. Consequently,
in contrast with the majority of the currently available corpora for other languages, which consider as labels “neutral”, “entailment”
and “contradiction” for the task of RTE, the authors of the ASSIN corpus decided to use as labels “none”, “entailment” and “paraphrase”.
Finally, the manual annotation of pairs of sentences was performed by human annotators. At least four annotators were randomly
selected to annotate each pair of sentences, which is done in two steps: (i) assigning a semantic similarity label (a score between 1 and 5,
from unrelated to very similar); and (ii) providing an entailment label (one sentence entails the other, sentences are paraphrases,
or no relation). Sentence pairs where at least three annotators do not agree on the entailment label were considered controversial
and thus discarded from the gold standard annotations. The full dataset has 10,000 sentence pairs, half of which in Brazilian Portuguese
and half in European Portuguese. Either language variant has 2,500 pairs for training, 500 for validation and 2,000 for testing.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 2000
`'train'` | 2500
`'validation'` | 500

- **特征**：

```json
{
    "sentence_pair_id": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "premise": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "hypothesis": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "relatedness_score": {
        "dtype": "float32",
        "id": null,
        "_type": "Value"
    },
    "entailment_judgment": {
        "num_classes": 3,
        "names": [
            "NONE",
            "ENTAILMENT",
            "PARAPHRASE"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```

## ptbr

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:assin/ptbr')
```

- **说明**：

```
The ASSIN (Avaliação de Similaridade Semântica e INferência textual) corpus is a corpus annotated with pairs of sentences written in
Portuguese that is suitable for the  exploration of textual entailment and paraphrasing classifiers. The corpus contains pairs of sentences
extracted from news articles written in European Portuguese (EP) and Brazilian Portuguese (BP), obtained from Google News Portugal
and Brazil, respectively. To create the corpus, the authors started by collecting a set of news articles describing the
same event (one news article from Google News Portugal and another from Google News Brazil) from Google News.
Then, they employed Latent Dirichlet Allocation (LDA) models to retrieve pairs of similar sentences between sets of news
articles that were grouped together around the same topic. For that, two LDA models were trained (for EP and for BP)
on external and large-scale collections of unannotated news articles from Portuguese and Brazilian news providers, respectively.
Then, the authors defined a lower and upper threshold for the sentence similarity score of the retrieved pairs of sentences,
taking into account that high similarity scores correspond to sentences that contain almost the same content (paraphrase candidates),
and low similarity scores correspond to sentences that are very different in content from each other (no-relation candidates).
From the collection of pairs of sentences obtained at this stage, the authors performed some manual grammatical corrections
and discarded some of the pairs wrongly retrieved. Furthermore, from a preliminary analysis made to the retrieved sentence pairs
the authors noticed that the number of contradictions retrieved during the previous stage was very low. Additionally, they also
noticed that event though paraphrases are not very frequent, they occur with some frequency in news articles. Consequently,
in contrast with the majority of the currently available corpora for other languages, which consider as labels “neutral”, “entailment”
and “contradiction” for the task of RTE, the authors of the ASSIN corpus decided to use as labels “none”, “entailment” and “paraphrase”.
Finally, the manual annotation of pairs of sentences was performed by human annotators. At least four annotators were randomly
selected to annotate each pair of sentences, which is done in two steps: (i) assigning a semantic similarity label (a score between 1 and 5,
from unrelated to very similar); and (ii) providing an entailment label (one sentence entails the other, sentences are paraphrases,
or no relation). Sentence pairs where at least three annotators do not agree on the entailment label were considered controversial
and thus discarded from the gold standard annotations. The full dataset has 10,000 sentence pairs, half of which in Brazilian Portuguese
and half in European Portuguese. Either language variant has 2,500 pairs for training, 500 for validation and 2,000 for testing.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'test'` | 2000
`'train'` | 2500
`'validation'` | 500

- **特征**：

```json
{
    "sentence_pair_id": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
    },
    "premise": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "hypothesis": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "relatedness_score": {
        "dtype": "float32",
        "id": null,
        "_type": "Value"
    },
    "entailment_judgment": {
        "num_classes": 3,
        "names": [
            "NONE",
            "ENTAILMENT",
            "PARAPHRASE"
        ],
        "names_file": null,
        "id": null,
        "_type": "ClassLabel"
    }
}
```
