<!--* freshness: { owner: 'maringeo' reviewed: '2021-12-13' review_interval: '6 months'} *-->

# モデルの形式

[tfhub.dev](https://tfhub.dev) は、SavedModel、TF1 Hub 形式、TF.js、および TFLite のモデル形式をホストします。このページでは、各モデル形式の概要を説明します。

## TensorFlow 形式

[tfhub.dev](https://tfhub.dev) は SavedModel 形式と TF1 Hub 形式で TensorFlow モデルをホストします。可能な限り、使用廃止となった TF1 Hub 形式ではなく、標準の SavedModel 形式でモデルを使用することをお勧めします。

### SavedModel

SavedModel は、TensorFlow モデルを共有する際の推奨形式です。SavedModel 形式の詳細については、[TensorFlow SavedModel](https://www.tensorflow.org/guide/saved_model) ガイドをご覧ください。

[tfhub.dev の参照ページ](https://tfhub.dev/s?subtype=module,placeholder)で TF2 バージョンフィルタを使用するか、[こちらのリンク](https://tfhub.dev/s?subtype=module,placeholder&tf-version=tf2)に従って、tfhub.dev の SavedModel を参照できます。

この形式は TensorFlow のコアの一部であるため、tfhub.dev の SavedModel を `tensorflow_hub` ライブラリに依存せずに使用することができます。

SavedModel については、TF Hub でさらにご覧ください。

- [TF2 SavedModel を使用する](https://gitlocalize.com/repo/4592/ja/site/en-snapshot/hub/tf2_saved_model.md)
- [TF2 SavedModel をエクスポートする](https://gitlocalize.com/repo/4592/ja/site/en-snapshot/hub/exporting_tf2_saved_model.md)
- [TF2 SavedModel の TF1/TF2 互換性](https://gitlocalize.com/repo/4592/ja/site/en-snapshot/hub/model_compatibility.md)

### TF1 Hub 形式

TF1 Hub 形式は、TF Hub ライブラリで使用されるカスタムシリアル化形式です。TF1 Hub 形式は、TensorFlow 1 の SavedModel 形式と構文レベルで似ていますが（同じファイル名とプロトコルメッセージ）、セマンティックレベルでは、モジュールの再利用、合成および再トレーニングが可能という点で異なります（リソースイニシャライザのストレージが異なる、メタグラフのタグ規則が異なるなど）。ディスク上で区別するには、`tfhub_module.pb`ファイルの有無を確認することが最も簡単です。

[tfhub.dev の参照ページ](https://tfhub.dev/s?subtype=module,placeholder)で TF1 バージョンフィルタを使用するか、[こちらのリンク](https://tfhub.dev/s?subtype=module,placeholder&tf-version=tf1)に従って、tfhub.dev の TF1 Hub 形式のモデルを参照できます。

TF1 Hub 形式のモデルについては、さらに TF Hub でご覧ください。

- [TF1 Hub 形式モデルを使用する](https://gitlocalize.com/repo/4592/ja/site/en-snapshot/hub/tf1_hub_module.md)
- [TF1 Hub 形式でモデルをエクスポートする](https://gitlocalize.com/repo/4592/ja/site/en-snapshot/hub/exporting_hub_format.md)
- [TF1 Hub 形式の TF1/TF2 互換性](https://gitlocalize.com/repo/4592/ja/site/en-snapshot/hub/model_compatibility.md)

## TFLite 形式

TFLite 形式は、オンデバイス推論で使用されます。詳細は、[TFLite ドキュメント](https://www.tensorflow.org/lite)をご覧ください。

[tfhub.dev の参照ページ](https://tfhub.dev/s?subtype=module,placeholder)で TF Lite モデル形式フィルタを使用するか、[こちらのリンク](https://tfhub.dev/lite)に従って、tfhub.dev の TF Lite モデルを参照できます。

## TFJS 形式

TF.js 形式は、インブラウザ ML に使用されます。詳細については、[TF.js ドキュメント](https://www.tensorflow.org/js)をご覧ください。

[tfhub.dev の参照ページ](https://tfhub.dev/s?subtype=module,placeholder)で TF.js モデル形式フィルタを使用するか、[こちらのリンク](https://tfhub.dev/js)に従って、tfhub.dev の TF.js モデルを参照できます。
