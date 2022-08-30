# bookcorpus

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/bookcorpus)
- [Huggingface](https://huggingface.co/datasets/bookcorpus)

## plain_text

Use the following command to load this dataset in TFDS:

```python
ds = tfds.load('huggingface:bookcorpus/plain_text')
```

- **Description**:

```
Books are a rich source of both fine-grained information, how a character, an object or a scene looks like, as well as high-level semantics, what someone is thinking, feeling and how these states evolve through a story.This work aims to align books to their movie releases in order to providerich descriptive explanations for visual content that go semantically farbeyond the captions available in current datasets.
```

- **许可**：无已知许可
- **Version**: 1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 74004228

- **特征**：

```json
{
    "text": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```
