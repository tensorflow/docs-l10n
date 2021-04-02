<!--* freshness: { owner: 'maringeo' reviewed: '2020-12-29' review_interval: '3 months' } *-->

# TF1/TF2 モデルの互換性

## TF Hub のモデル形式

TF Hub は TensorFlow プログラムで再読み込み、構築、再トレーニングできる再利用可能なモデル片を提供します。これらのモデル片には 2 種類の形式があります。

- 独自の [TF1 Hub 形式](https://www.tensorflow.org/hub/tf1_hub_module)。主に TF1（または TF2 の TF1 互換モード）で [hub.Module API](https://www.tensorflow.org/hub/api_docs/python/hub/Module) を介して使用されることを想定したものです。互換性に関する完全な情報は、[以下](#compatibility_of_hubmodule)をご覧ください。
- ネイティブな [TF2 SavedModel](https://www.tensorflow.org/hub/tf2_saved_model) 形式。主に TF2 で [hub.load](https://www.tensorflow.org/hub/api_docs/python/hub/load) および [hub.KerasLayer](https://www.tensorflow.org/hub/api_docs/python/hub/KerasLayer) API を介して使用されることを想定したものです。互換性に関する完全な情報は、[以下](#compatibility_of_tf2_savedmodel)をご覧ください。

モデル形式は、[tfhub.dev](https://tfhub.dev) のモデルページで確認できます。モデルの**読み込み/推論**、**微調整**、**作成**は、モデル形式に基づく TF1/2 ではサポートされていない場合があります。

## TF1 Hub 形式の互換性 {:#compatibility_of_hubmodule}

<table style="width: 100%;">
  <tr style="text-align: center">
    <col style="width: 20%">
    <col style="width: 40%">
    <col style="width: 40%">
    <td style="text-align: center; background-color: #D0D0D0">演算</td>
    <td style="text-align: center; background-color: #D0D0D0">TF1/ TF2 の TF1 互換モード <a href="#compatfootnote">[1]</a>
</td>
    <td style="text-align: center; background-color: #D0D0D0">TF2</td>
  </tr>
  <tr>
    <td>読み込み / 推論</td>
    <td>完全サポート（<a href="https://www.tensorflow.org/hub/tf1_hub_module#using_a_module">包括的な TF1 Hub 形式の読み込みガイド</a>）       <pre style="font-size: 12px;" lang="python">m = hub.Module(handle) outputs = m(inputs)</pre>
</td>
    <td>hub.load     <pre style="font-size: 12px;" lang="python">m = hub.load(handle) outputs = m.signatures["sig"](inputs)</pre>       または hub.KerasLayer        <pre style="font-size: 12px;" lang="python">m = hub.KerasLayer(handle, signature="sig") outputs = m(inputs)</pre> のいずれかを使用することをお勧めします。</td>
  </tr>
  <tr>
    <td>微調整</td>
    <td>完全サポート（<a href="https://www.tensorflow.org/hub/tf1_hub_module#for_consumers">包括的な TF1 Hub 形式の微調整ガイド</a>）     <pre style="font-size: 12px;" lang="python">m = hub.Module(handle,                trainable=True,                tags=["train"]*is_training) outputs = m(inputs)</pre>       <div style="font-style: italic; font-size: 14px">       注意: 個別の train グラフを必要としないモジュールには train タグがありません。       </div>
</td>
    <td style="text-align: center">未サポート</td>
  </tr>
  <tr>
    <td>作成</td>
    <td>完全サポート（<a href="https://www.tensorflow.org/hub/tf1_hub_module#general_approach">包括的な TF1 Hub 形式の作成ガイド</a>をご覧ください） <br> <div style="font-style: italic; font-size: 14px">       注意: TF1 Hub 形式は TF1 向けであり、TF2 では部分的にのみサポートされています。TF2 SavedModel の作成を検討してください。       </div>
</td>
    <td style="text-align: center">未サポート</td>
  </tr>
</table>

## TF2 SavedModel の互換性 {:#compatibility_of_tf2_savedmodel}

TF1.15 以前はサポートされていません。

<table style="width: 100%;">
  <tr style="text-align: center">
    <col style="width: 20%">
    <col style="width: 40%">
    <col style="width: 40%">
    <td style="text-align: center; background-color: #D0D0D0">演算</td>
    <td style="text-align: center; background-color: #D0D0D0">TF1.15/ TF2 の TF1 互換モード <a href="#compatfootnote">[1]</a>
</td>
    <td style="text-align: center; background-color: #D0D0D0">TF2</td>
  </tr>
  <tr>
    <td>読み込み / 推論</td>
    <td>       hub.load     <pre style="font-size: 12px;" lang="python">m = hub.load(handle) outputs = m(inputs)</pre>       または hub.KerasLayer       <pre style="font-size: 12px;" lang="python">m = hub.KerasLayer(handle) outputs = m(inputs)</pre> を使用してください</td>
    <td>完全サポート（<a href="https://www.tensorflow.org/hub/tf2_saved_model#using_savedmodels_from_tf_hub">包括的な TF2 SavedModel の読み込みガイド</a>）。hub.load     <pre style="font-size: 12px;" lang="python">m = hub.load(handle) outputs = m(inputs)</pre>       または hub.KerasLayer       <pre style="font-size: 12px;" lang="python">m = hub.KerasLayer(handle) outputs = m(inputs)</pre> のいずれかを使用してください。</td>
  </tr>
  <tr>
    <td>微調整</td>
    <td>Model.fit() でトレーニングされる場合、または <a href="https://www.tensorflow.org/guide/migrate#using_a_custom_model_fn">カスタム model_fn ガイド</a> に従って Model をラップする model_fn を持つ Estimator でトレーニングされる場合にtf.keras.Model で使用される hub.KerasLayer でサポートされます。       <br><div style="font-style: italic; font-size: 14px;">         注意: hub.KerasLayer は古い tf.compat.v1.layers または hub.Module API のようにグラフコレクションを<span style="font-weight: bold;">埋めません</span>。       </div>
</td>
    <td>完全サポート（<a href="https://www.tensorflow.org/hub/tf2_saved_model#for_savedmodel_consumers">包括的な TF2 SavedModel の微調整ガイド</a>）。hub.load:       <pre style="font-size: 12px;" lang="python">m = hub.load(handle) outputs = m(inputs, training=is_training)</pre>       または hub.KerasLayer:       <pre style="font-size: 12px;" lang="python">m =  hub.KerasLayer(handle, trainable=True) outputs = m(inputs)</pre> のいずれかを使用してください。</td>
  </tr>
  <tr>
    <td>作成</td>
    <td>TF2 API <a href="https://www.tensorflow.org/api_docs/python/tf/saved_model/save">       tf.saved_model.save()</a> は互換モード内から呼び出すことができます。</td>
   <td>完全サポート（<a href="https://www.tensorflow.org/hub/tf2_saved_model#creating_savedmodels_for_tf_hub">包括的な TF2 SavedModel の作成ガイド</a>）</td>
  </tr>
</table>

<p id="compatfootnote">[1] "TF2 の TF1 互換モード" とは、<a>TensorFlow 移行ガイド</a>に記載されているように TF2 を <code style="font-size: 12px;" lang="python">import tensorflow.compat.v1 as tf</code> でインポートして <code>tf.disable_v2_behavior()</code> を実行する場合の複合効果を指します。</p>
