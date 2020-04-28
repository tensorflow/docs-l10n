# TensorFlow Docs Community Translations

## Issues

File community translation issues with the
[tensorflow/docs-l10](https://github.com/tensorflow/docs-l10n/issues) tracker.

For general documentation issues, use the tracker in the
[tensorflow/tensorflow](https://github.com/tensorflow/tensorflow/issues/new?template=20-documentation-issue.md)
repo.

# Community translations

Please read the
[community translations](https://www.tensorflow.org/community/contribute/docs#community_translations)
section in the
[TensorFlow docs contributor guide](https://www.tensorflow.org/community/contribute/docs).

Ask general questions on the
[docs@tensorflow.org list](https://groups.google.com/a/tensorflow.org/forum/#!forum/docs),
and there are a few
[language-specific docs lists](https://www.tensorflow.org/community/contribute/docs#community_translations)
to help coordinate communities. If a language is gaining momentum and
contributors think a new language-specific list would be useful, file
[a GitHub issue](https://github.com/tensorflow/docs-l10n/issues).

## Content

The source-of-truth for technical documentation is the
[site/en](https://github.com/tensorflow/docs/tree/master/site/en) directory in
the [tensorflow/docs](https://github.com/tensorflow/docs/tree/master/site/en)
repo. If, when translating, you find an issue in the source content, please send
a separate pull request that fixes the upstream content.

To view translated content on [tensorflow.org](https://www.tensorflow.org),
select the in-page language switcher or append `?hl=<lang>` to the URL. For
example, the English
[TensorFlow 2 quickstart tutorial](https://www.tensorflow.org/tutorials/quickstart/beginner?hl=en)
can be read in:

* Korean: https://www.tensorflow.org/tutorials/quickstart/beginner?hl=ko
* Russian: https://www.tensorflow.org/tutorials/quickstart/beginner?hl=ru
* Or any of the supported languages in [site/<lang>](https://github.com/tensorflow/docs-l10n/tree/master/site).

### Subsites

Subsites are doc collections for projects outside of the *core TensorFlow* in
the `tensorflow/docs` repo. These docs usually live with their project in a
separate GitHub repository. For example, the docs for
[tensorflow.org/federated](https://www.tensorflow.org/federated) live in the
[tensorflow/federated](https://github.com/tensorflow/federated/tree/master/docs)
GitHub repo. For translations, mirror this structure *within* this
`tensorflow/docs-l10n` repo (i.e. do not submit translation pull requests to the
subsite project repo).

### Do not translate

The following sections are *not* included in the community translations project.
TensorFlow.org does not translate API reference, and uses an internal system for
landing pages and release-sensitive documentation. Please do not translate the
following sections:

* Any `/images/` directories.
* Any `/r1/` directories (TensorFlow 1.x docs).
* The `/install/` directory.
* API reference including `/api_docs/` and `/versions/` directories.
* Navigation: `_book.yaml` and `_toc.yaml` files.
* Overview pages such as `_index.yaml`, `index.html`, and `index.md`.

## Style

Please follow the
[TensorFlow documentation style guide](https://www.tensorflow.org/community/contribute/docs_style)
and the
[Google developer docs style guide](https://developers.google.com/style/highlights),
when applicable. Additionally, language-specific style may be agreed upon by the
community—see the `README.md` file within a `site/<lang>/` directory. Community
proposals and communication can take place over pull request comments or
language-specific mailing lists (if available).

To reduce diff-churn on notebook pull requests and make reviews easier, please
use the [nbfmt](https://github.com/tensorflow/docs/blob/master/tools/nbfmt.py)
tool or download the notebook from
[Google Colab](https://colab.research.google.com/).

## License

[Apache License 2.0](LICENSE)
