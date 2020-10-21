# モデル最適化

Edge devices often have limited memory or computational power. Various optimizations can be applied to models so that they can be run within these constraints. In addition, some optimizations allow the use of specialized hardware for accelerated inference.

TensorFlow Lite と [TensorFlow モデル最適化ツールキット](https://www.tensorflow.org/model_optimization)は推論の最適化の複雑さを最小限に抑えるツールを提供します。

It's recommended that you consider model optimization during your application development process. This document outlines some best practices for optimizing TensorFlow models for deployment to edge hardware.

## モデルを最適化する理由

There are several main ways model optimization can help with application development.

### サイズ縮小

一部の形式の最適化は、モデルのサイズを縮小するために使用できます。小さいモデルには次の利点があります。

- **Smaller storage size:** Smaller models occupy less storage space on your users' devices. For example, an Android app using a smaller model will take up less storage space on a user's mobile device.
- **Smaller download size:** Smaller models require less time and bandwidth to download to users' devices.
- **Less memory usage:** Smaller models use less RAM when they are run, which frees up memory for other parts of your application to use, and can translate to better performance and stability.

量子化により、これらのすべてのケースでモデルのサイズを縮小できますが、精度が低下する可能性があります。プルーニングとクラスタリングは、モデルをより簡単に圧縮できるようにすることで、ダウンロード用のモデルのサイズを縮小します。

### レイテンシ短縮

*Latency* is the amount of time it takes to run a single inference with a given model. Some forms of optimization can reduce the amount of computation required to run inference using a model, resulting in lower latency. Latency can also have an impact on power consumption.

Currently, quantization can be used to reduce latency by simplifying the calculations that occur during inference, potentially at the expense of some accuracy.

### アクセラレータの互換性

Some hardware accelerators, such as the [Edge TPU](https://cloud.google.com/edge-tpu/), can run inference extremely fast with models that have been correctly optimized.

一般に、これらの種類のデバイスでは、モデルを特定の方法で量子化する必要があります。要件についての詳細は、各ハードウェアアクセラレータのドキュメントをご覧ください。

## トレードオフ

Optimizations can potentially result in changes in model accuracy, which must be considered during the application development process.

精度の変更は、最適化される個々のモデルに依存するため、事前に予測することは困難です。一般に、サイズまたはレイテンシが最適化されたモデルでは、精度がわずかに低下します。アプリに応じては、これによりユーザーエクスペリエンスが影響される場合があります。まれに、特定のモデルでは最適化プロセスの結果として精度がやや向上する場合があります。

## 最適化の種類

TensorFlow Lite は現在、量子化、プルーニング、クラスタリングによる最適化をサポートしています。

These are part of the [TensorFlow Model Optimization Toolkit](https://www.tensorflow.org/model_optimization), which provides resources for model optimization techniques that are compatible with TensorFlow Lite.

### 量子化

[Quantization](https://www.tensorflow.org/model_optimization/guide/quantization/post_training) works by reducing the precision of the numbers used to represent a model's parameters, which by default are 32-bit floating point numbers. This results in a smaller model size and faster computation.

TensorFlow Lite で使用できる量子化の種類は次のとおりです。

手法 | データ要件 | サイズ縮小 | 精度 | サポートされているハードウェア
--- | --- | --- | --- | ---
[トレーニング後の float16 量子化](post_training_float16_quant.ipynb) | データなし | 50% 以下 | 精度低下（ごくわずか） | CPU、GPU
[トレーニング後のダイナミックレンジ量子化](post_training_quant.ipynb) | データなし | 75% 以下 | 精度低下 | CPU、GPU (Android)
[トレーニング後の整数量子化](post_training_integer_quant.ipynb) | ラベルなしの代表的なサンプル | 75% 以下 | 精度低下（少量） | CPU, GPU (Android)、EdgeTPU、Hexagon DSP
[量子化認識トレーニング](http://www.tensorflow.org/model_optimization/guide/quantization/training) | ラベル付けされたトレーニングデータ | 75% 以下 | 精度低下（ごく少量） | CPU, GPU (Android)、EdgeTPU、Hexagon DSP

Below are the latency and accuracy results for post-training quantization and quantization-aware training on a few models. All latency numbers are measured on Pixel 2 devices using a single big core CPU. As the toolkit improves, so will the numbers here:

<figure>
  <table>
    <tr>
      <th>モデル</th>
      <th>トップ 1 の精度（オリジナル）</th>
      <th>トップ 1 の精度（トレーニング後の量子化）</th>
      <th>トップ1の精度（量子化認識トレーニング）</th>
      <th>レイテンシ (オリジナル) (ms)</th>
      <th>レイテンシ (トレーニング後の量子化) (ms)</th>
      <th>レイテンシ (量子化認識トレーニング) (ms)</th>
      <th>サイズ (オリジナル) (MB)</th>
      <th> サイズ (最適化) (MB)</th>
    </tr> <tr>
<td>Mobilenet-v1-1-224</td>
<td>0.709</td>
<td>0.657</td>
<td>0.70</td>
      <td>124</td>
<td>112</td>
<td>64</td>
<td>16.9</td>
<td>4.3</td>
</tr>
    <tr>
<td>Mobilenet-v2-1-224</td>
<td>0.719</td>
<td>0.637</td>
<td>0.709</td>
      <td>89</td>
<td>98</td>
<td>54</td>
<td>14</td>
<td>3.6</td>
</tr>
   <tr>
<td>Inception_v3</td>
<td>0.78</td>
<td>0.772</td>
<td>0.775</td>
      <td>1130</td>
<td>845</td>
<td>543</td>
<td>95.7</td>
<td>23.9</td>
</tr>
   <tr>
<td>Resnet_v2_101</td>
<td>0.770</td>
<td>0.768</td>
<td>N/A</td>
      <td>3973</td>
<td>2868</td>
<td>N/A</td>
<td>178.3</td>
<td>44.9</td>
</tr>
 </table>
  <figcaption>     <b>Table 1</b> Benefits of model quantization for select CNN models   </figcaption>
</figure>

### プルーニング

[Pruning](https://www.tensorflow.org/model_optimization/guide/pruning) works by removing parameters within a model that have only a minor impact on its predictions. Pruned models are the same size on disk, and have the same runtime latency, but can be compressed more effectively. This makes pruning a useful technique for reducing model download size.

今後、TensorFlow Lite ではプルーニングされたモデルのレイテンシが低減される予定です。

### クラスタリング

[クラスタリング](https://www.tensorflow.org/model_optimization/guide/clustering)はモデル内の各レイヤーの重みを事前定義された数のクラスタにグループ化し、個々のクラスタに属する重みの重心値を共有します。これにより、モデル内の一意の重み値の数が減り、複雑さが軽減されます。

その結果、クラスタ化されたモデルをより効果的に圧縮でき、プルーニングと同様にデプロイメントにおける利点を提供します。

## 開発ワークフロー

まず、[ホステッドモデル](../guide/hosted_models.md)のモデルがアプリで機能することを確認します。機能しない場合は、[トレーニング後の量子化ツール](post_training_quantization.md)から始めることをお勧めします。これはトレーニングデータを必要としないため、幅広く適用できます。

精度とレイテンシが不十分な場合やハードウェアアクセラレータのサポートが重要な場合は、[量子化認識トレーニング](https://www.tensorflow.org/model_optimization/guide/quantization/training){:.external}が最適なオプションです。[TensorFlow モデル最適化ツールキット](https://www.tensorflow.org/model_optimization)で追加の最適化手法をご覧ください。

モデルサイズをさらに縮小する場合は、モデルを量子化する前に、[プルーニング](#pruning)や[クラスタリング](#clustering)をお試しください。
