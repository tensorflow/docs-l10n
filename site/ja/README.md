# コミュニティによる翻訳について

これらのドキュメントは私たちTensorFlowコミュニティが翻訳したものです。コミュニティによる
翻訳は**ベストエフォート**であるため、この翻訳が正確であることや[英語の公式ドキュメント](https://www.tensorflow.org/?hl=en)の
最新の状態を反映したものであることを保証することはできません。
この翻訳の品質を向上させるためのご意見をお持ちの方は、GitHub リポジトリ[tensorflow/docs-l10n](https://github.com/tensorflow/docs-l10n)にプルリクエストをお送りください。

翻訳やレビューに参加して頂ける方は以下のコミュニティにご連絡ください:

* Slack
  * Slack の #docs_translation チャンネルで議論をしています
  * [TensorFlow User Group の公式ページ](https://tfug.jp/)から参加可能です
* Google Groups
  * [docs@tensorflow.org](https://groups.google.com/a/tensorflow.org/forum/#!forum/docs)
  * [docs-ja@tensorflow.org](https://groups.google.com/a/tensorflow.org/forum/#!forum/docs-ja)

また、翻訳を行う際には [CONTRIBUTING.md](CONTRIBUTING.md) をお読みください。

## Community translations

Our TensorFlow community has translated these documents. Because community
translations are *best-effort*, there is no guarantee that this is an accurate
and up-to-date reflection of the
[official English documentation](https://www.tensorflow.org/?hl=en).
If you have suggestions to improve this translation, please send a pull request
to the [tensorflow/docs](https://github.com/tensorflow/docs) GitHub repository.
To volunteer to write or review community translations, contact the
[docs@tensorflow.org list](https://groups.google.com/a/tensorflow.org/forum/#!forum/docs).

Note: Please focus translation efforts on
[TensorFlow 2](https://www.tensorflow.org) in the
[site/en/](https://github.com/tensorflow/docs/tree/master/site/en/)
directory. TF 1.x community docs will no longer be updated as we prepare for the
2.0 release. See
[the announcement](https://groups.google.com/a/tensorflow.org/d/msg/docs/vO0gQnEXcSM/YK_ybv7tBQAJ).

## Do not translate

The following sections are *not* included in `site/ja` community translations
project. TensorFlow.org does not translate API reference, and uses an internal
system for landing pages and release-sensitive documentation. Please do not
translate the following sections:

* The `/install/` directory.
* API reference including `/api_docs/` and `/versions/` directories.
* Navigation: `_book.yaml` and `_toc.yaml` files.
* Overview pages such as `_index.yaml`, `index.html`, and `index.md`.

## Japanese translation guide

### Translation of technical words

Some technical words in English do not have a natural translation. Do _not_
translate the following words, use katakana otherwise:

*   (mini-) batch
*   estimator
*   eager execution
*   label
*   class
*   helper
*   hyperparameter
*   optimizer
*   one-hot encoding
*   epoch
*   callback
*   sequence
*   dictionary (in Python)
*   embedding
*   padding
*   unit
*   node
*   neuron
*   target
*   import
*   checkpoint
*   compile
*   dropout
*   penalty
*   scalar
*   tensor
*   decode
*   tuple
*   protocol buffer

### Additional Do not translate

We left the `community` directory untranslated because it is a contribution guideline for the global communities.
