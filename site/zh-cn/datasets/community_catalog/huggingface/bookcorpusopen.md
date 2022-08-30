# bookcorpusopen

参考：

- [代码](https://github.com/huggingface/datasets/blob/master/datasets/bookcorpusopen)
- [Huggingface](https://huggingface.co/datasets/bookcorpusopen)

## plain_text

使用以下命令在 TFDS 中加载此数据集：

```python
ds = tfds.load('huggingface:bookcorpusopen/plain_text')
```

- **说明**：

```
Books are a rich source of both fine-grained information, how a character, an object or a scene looks like, as well as high-level semantics, what someone is thinking, feeling and how these states evolve through a story.
This version of bookcorpus has 17868 dataset items (books). Each item contains two fields: title and text. The title is the name of the book (just the file name) while text contains unprocessed book text. The bookcorpus has been prepared by Shawn Presser and is generously hosted by The-Eye. The-Eye is a non-profit, community driven platform dedicated to the archiving and long-term preservation of any and all data including but by no means limited to... websites, books, games, software, video, audio, other digital-obscura and ideas.
```

- **许可**：无已知许可
- **版本**：1.0.0
- **拆分**：

拆分 | 样本
:-- | --:
`'train'` | 17868

- **特征**：

```json
{
    "title": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    },
    "text": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
    }
}
```
