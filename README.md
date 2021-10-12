# TensorFlow Docs Translations

This project contains translations of the technical content and Jupyter
notebooks published on [tensorflow.org](https://www.tensorflow.org/guide).

Please file issues under the *documentation* component of the
[TensorFlow issue tracker](https://github.com/tensorflow/tensorflow/issues/new?template=20-documentation-issue.md).
Questions about TensorFlow usage are better addressed on the
[TensorFlow Forum](https://discuss.tensorflow.org/).

## Contributing

Contributors are encouraged to use our GitLocalize project to submit pull
requests and reviews: https://gitlocalize.com/tensorflow/docs-l10n

General docs instructions are available in the
[TensorFlow docs contributor guide](https://www.tensorflow.org/community/contribute/docs).

Please sign a
[Contributor License Agreement](https://cla.developers.google.com/) (CLA) to
contribute to this Google open source project. Check
[your existing CLA](https://cla.developers.google.com/clas) and verify that
your [email is set on git commits](https://docs.github.com/en/github/setting-up-and-managing-your-github-user-account/setting-your-commit-email-address).

## Content

To view translated content on tensorflow.org, select the in-page language
switcher or append `?hl=<lang>` to the URL. For example, the
[TensorFlow quickstart for beginners](https://www.tensorflow.org/tutorials/quickstart/beginner?hl=en)
tutorial is available in:

* Korean: https://www.tensorflow.org/tutorials/quickstart/beginner?hl=ko,
* Spanish: https://www.tensorflow.org/tutorials/quickstart/beginner?hl=es-419,
* Or any of the supported languages in [site/&lt;lang&gt;](./site/).

If a human-translation does not exist, some pages fall back to a *machine
translation* (MT). An MT page is indicated with a banner at the top of the page.
If the MT page is not useful or confusing, please click the *Switch to English*
button (and consider providing a human translation through the
[GitLocalize project](https://gitlocalize.com/tensorflow/docs-l10n)).

### Source

Source content is aggregated from multiple GitHub repos into the
[/site/en-snapshot/](./site/en-snapshot/) directory used for translations.
Translations are published to the website on a periodic basis (usually weekly or
bi-weekly). If you find an error in the source content, please submit a fix to
the [upstream repo](./site/en-snapshot/README.md) and *not* the
`/site/en-snapshot/` directory in this repo.

### Do not translate

Not all content on tensorflow.org is translated in this project (or at all).
Overview pages and navigation files are translated using another process.
tensorflow.org does not translate the API reference, old versions, images, or
time-sensitive sections like the
[installation instructions](https://www.tensorflow.org/install). Non-translated
pages are automatically filtered in the
[GitLocalize](https://gitlocalize.com/tensorflow/docs-l10n) interface.

## Style

The [TensorFlow docs notebook tools](https://github.com/tensorflow/docs/tree/master/tools/tensorflow_docs/tools)
are used for formatting and style consistency. This is integrated into the pull
request workflow and can be run locally.

Please follow the
[TensorFlow documentation style guide](https://www.tensorflow.org/community/contribute/docs_style)
and the
[Google developer docs style guide](https://developers.google.com/style/highlights),
when applicable.

## Languages

Official language support is determined by a number of factors including—but not
limited to—site metrics and demand, community support,
[English proficiency](https://en.wikipedia.org/wiki/EF_English_Proficiency_Index),
audience preference, and other indicators. Since each supported language incurs
a cost, unmaintained languages are removed. Support for new languages will be
announced on the [TensorFlow blog](https://blog.tensorflow.org/) or
[Twitter](https://twitter.com/TensorFlow).

The [community branch](https://github.com/tensorflow/docs-l10n/tree/community/site)
contains community contributed content for languages that are not officially
supported by the TensorFlow team. This is an unmaintained archive that you can
use for your own open source fork if your preferred language is not supported.
Please let us know if you maintain a language! These docs are not published to
tensorflow.org.

## License

[Apache License 2.0](LICENSE)
