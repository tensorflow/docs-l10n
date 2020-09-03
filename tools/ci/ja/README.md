# Proofreading tool for Japanese translation

This is a proofreading tool for Japanese translation.
We can check our documants following the configuration file: [`redpen-conf.xml`](redpen-conf.xml).
For example, checking notations and synonyms with [`terminologies.txt`](terminologies.txt).

## Requirements

- [RedPen](http://redpen.cc/)
  - We use RedPen to check markdown files
- [Jupyter](https://jupyter.org/)
  - We use Jupyter to convert notebooks to markdowns

## Basic usage

You can run the proofreading tool as below:

```shell script
./tools/ci/ja/bin/proofreading.sh <target directory or file>
```

For example:

```shell script
./tools/ci/ja/bin/proofreading.sh site/ja/guide
```

# Why use RedPen?

We are working on translation with more than one person.
So It is expected that a lot of orthographical variants will occur.
Redpen is a proofreading tool to help writing documents that need to adhere to a writing standard.
We can guarantee the quality of documents without lose writing speed while distributing translation tasks among multiple people.
RedPen officially support English and Japanese, but we can use some of the functions with another language.
