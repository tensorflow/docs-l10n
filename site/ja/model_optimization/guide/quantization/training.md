# 量子化認識トレーニング

<sub>TensorFlow Model Optimization による管理</sub>

量子化には、ポストトレーニング量子化と調子化認識トレーニングの 2 つ形態があります。量子化認識トレーニングは通常、モデルの精度の観点からより優れていますが、比較的使いやすい[ポストトレーニング量子化](post_training.md)から始めると良いでしょう。

このページでは、ユースケースにどのように適合するかを判断できるように、量子化認識トレーニングの概要を説明します。

- 早速エンドツーエンドの例を確認するには、[量子化認識トレーニングの例](training_example.md)をご覧ください。
- ユースケースに合った API を素早く特定するには、[量子化認識トレーニングの総合ガイド](training_comprehensive_guide.md)をご覧ください。

## 概要

量子化認識トレーニングは、推論時の量子化をエミュレートし、下流のツールが実際に量子化されたモデルを生成するために使用するモデルを作成します。量子化モデルは低精度（32 ビット float ではなく 8 ビット）を使用するため、デプロイ中のメリットがあります。

### 量子化によるデプロイ

量子化によって、モデル圧縮とレイテンシ縮小による改善が得られます。API のデフォルトでは、モデルサイズは 4 倍縮小し、テスト対象のバックエンドの CPU レイテンシは通常 1.5～4 倍 に改善します。最終的に、レイテンシの改善は、[EdgeTPU](https://coral.ai/docs/edgetpu/benchmarks/) や NNAPI といった互換性のある機械学習アクセラレータで確認されるようになります。

このテクニックは、スピーチ、ビジョン、テキスト、および翻訳ユースケースの制作で使用されています。コードは現在、[これらのモデルのサブセット](#general-support-matrix)をサポートしています。

### 量子化による実験と関連ハードウェア

ユーザーは量子化パラメーター（ビット数など）や、ある程度の基盤のアルゴリズムを構成できます。API のデフォルトを変更した場合、バックエンドにデプロイするパスは現在サポートされていないことに注意してください。たとえば、TFLite 変換とカーネル実装は 8 ビット量子化のみをサポートしています。

この構成に固有の API は実験的 API であり、下位互換性はありません。

### API の互換性

ユーザーは、次の API を使って量子化を適用できます。

- モデルの構築: `tf.keras`（Sequential モデルと Functional モデルのみ）
- TensorFlow バージョン: TF 2.x（tf-nightly）
    - TF 2.X パッケージの `tf.compat.v1` はサポートされません。
- TensorFlow 実行モード: Eager execution

次の分野でのサポート追加が予定されています。

<!-- TODO(tfmot): file Github issues. -->

- モデルの構築: サブクラス化されたモデルのサポートがなぜ制限されているのか、またはサポートされていないかを説明
- 分散型トレーニング: `tf.distribute`

### 全般的なサポート状況

次の分野のサポートが提供されています。

- モデルのカバレッジ: [ホワイトリストに追加されたレイヤー](https://github.com/tensorflow/model-optimization/tree/master/tensorflow_model_optimization/python/core/quantization/keras/default_8bit/default_8bit_quantize_registry.py)を使用したモデル、Conv2D および DepthwiseConv2D レイヤーに従う BatchNormalization、および限られたケースでの `Concat`
    <!-- TODO(tfmot): add more details and ensure they are all correct. -->
- ハードウェアアクセラレータ: API のデフォルトは、EdgeTPU、NNAPI、TFLite バックエンドなどのアクセラレーションと互換しています。ロードマップの注意事項をご覧ください。
- 量子化によるデプロイ: テンソルごとの量子化ではなく、畳み込みレイヤーの軸ごとの量子化のみが現在サポートされています。

次の分野でのサポート追加が予定されています。

<!-- TODO(tfmot): file Github issue. Update as more functionality is added prior
to launch. -->

- モデルのカバレッジ: RNN/LSTM と一般的な Concat サポートを含めるように拡張。
- ハードウェアアクセラレーション: TFLite コンバータが完全な整数モデルを生成できるようにする。詳細は、[この課題](https://github.com/tensorflow/tensorflow/issues/38285)をご覧ください。
- 量子化ユースケースの実験:
    - Keras レイヤーに及ぶまたはトレーニングステップを必要とする量子化アルゴリズムによる実験
    - API の安定化

## 結果

### ツールによる画像分類

<figure>
  <table>
    <tr>
      <th>Model</th>
      <th>Non-quantized Top-1 Accuracy </th>
      <th>8-bit Quantized Accuracy </th>
    </tr>
    <tr>
      <td>MobilenetV1 224</td>
      <td>71.03%</td>
      <td>71.06%</td>
    </tr>
    <tr>
      <td>Resnet v1 50</td>
      <td>76.3%</td>
      <td>76.1%</td>
    </tr>
    <tr>
      <td>MobilenetV2 224</td>
      <td>70.77%</td>
      <td>70.01%</td>
    </tr>
 </table>
</figure>

モデルは、Imagenet でテストされ、TensorFlow と TFLite で評価されました。

### テクニックの画像分類

<figure>
  <table>
    <tr>
      <th>Model</th>
      <th>Non-quantized Top-1 Accuracy </th>
      <th>8-Bit Quantized Accuracy </th>
    <tr>
      <td>Nasnet-Mobile</td>
      <td>74%</td>
      <td>73%</td>
    </tr>
    <tr>
      <td>Resnet-v2 50</td>
      <td>75.6%</td>
      <td>75%</td>
    </tr>
 </table>
</figure>

モデルは、Imagenet でテストされ、TensorFlow と TFLite で評価されました。

## 例

[量子化認識トレーニングの例](training_example.md)のほかに、次の例をご覧ください。

- 量子化を使用した MNIST 手書き数字の分類タスクの CNN モデル: [コード](https://github.com/tensorflow/model-optimization/blob/master/tensorflow_model_optimization/python/core/quantization/keras/quantize_functional_test.py)

同様の背景については、「*Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference*」[論文](https://arxiv.org/abs/1712.05877)をご覧ください。この論文では、このツールが使用するいくつかの概念を紹介します。実装はまったく同じではなく、このツールで使用される追加の概念があります（軸ごとの量子化など）。
