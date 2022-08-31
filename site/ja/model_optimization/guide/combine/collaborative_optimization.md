# 協調最適化

<sub>保守担当: Arm ML Tooling</sub>

このドキュメントでは、さまざまな手法を組み合わせて機械学習モデルを最適化してデプロイするための実験的な API について概説します。

## Overview

協調最適化は、最適な推論速度、モデルサイズ、精度などのバランスをデプロイメントの際に達成するモデルを作成するためのさまざまな手法を含む包括的なプロセスです。

協調最適化では、累積された最適化効果を達成するために、個々の技術を重ねて適用します。次の最適化のさまざまな組み合わせが可能です。

- [重みプルーニング](https://medium.com/tensorflow/tensorflow-model-optimization-toolkit-pruning-api-42cac9157a6a)

- [Weight clustering](https://blog.tensorflow.org/2020/08/tensorflow-model-optimization-toolkit-weight-clustering-api.html)

- Quantization

    - [Post-training quantization](https://medium.com/tensorflow/tensorflow-model-optimization-toolkit-post-training-integer-quantization-b4964a1ea9ba)
    - [量子化認識トレーニング](https://blog.tensorflow.org/2020/04/quantization-aware-training-with-tensorflow-model-optimization-toolkit.html)（QAT）

通常、これらの手法を組み合わせて適用すると前に適用された手法の結果が破壊され、すべてを同時に適用することにより全体的な利点が損なわれるという問題が発生します。たとえば、クラスタリングでは、プルーニング API によるスパース性は保持されません。この問題を解決するために、次の実験的な協調最適化手法を紹介します。

- [スパース性を保持するクラスタリング](https://www.tensorflow.org/model_optimization/guide/combine/sparse_clustering_example)
- [スパース性を保持する量子化認識トレーニング](https://www.tensorflow.org/model_optimization/guide/combine/pqat_example)（PQAT）
- [クラスタリングを保持する量子化認識トレーニング](https://www.tensorflow.org/model_optimization/guide/combine/cqat_example)（CQAT）
- [スパース性とクラスタリングを保持する量子化認識トレーニング](https://www.tensorflow.org/model_optimization/guide/combine/pcqat_example)

これらは、機械学習モデルを圧縮し、推論時にハードウェアアクセラレーションを利用するために使用できるいくつかのデプロイメントのパスを提供します。次の図は、いくつかのデプロイメントパスを示しています。希望するデプロイメント特性を持つモデルを検索できます。リーフノードは tflite 形式のデプロイメント対応モデルで、部分的または完全に量子化されています。緑の塗りつぶしは、再トレーニング/微調整が必要なステップを示し、赤い破線の境界線は、協調最適化ステップを強調しています。特定のノードでモデルを取得するために使用される手法は、対応するラベルに示されています。

![協調最適化](images/collaborative_optimization.png "collaborative optimization")

上の図では、直接的な量子化のみ（ポストトレーニングまたは QAT）のデプロイメントパスは省略されています。

上記のデプロイメントツリーの第 3 レベルでは完全に最適化されたモデルに到達しますが、他のレベルの最適化でも、必要とされる推論のレイテンシ―と精度のトレードオフを達成できる可能性があります。その場合、さらなる最適化は必要ありません。推奨されるトレーニングプロセスは、ターゲットのデプロイメントシナリオに適用可能なデプロイメントツリーのレベルを繰り返し適用し、モデルが推論のレイテンシ―の要件を満たしているかどうかを確認することです。満たしていない場合は、対応する協調最適化手法を使用してモデルをさらに圧縮し、必要に応じてモデルが完全に最適化 (プルーニング、クラスタリング、量子化) されるまで繰り返します。

次の図は、協調最適化パイプラインを通るサンプル重みカーネル密度プロットを示しています。

![協調最適化密度プロット](images/collaborative_optimization_dist.png "collaborative optimization density plot")

その結果、トレーニング時に指定されたターゲットのスパース性に応じて、一意の値の数が減り、スパースな重みの数が大幅に増える量子化されたデプロイメントモデルが得られます。大幅なモデル圧縮の利点に加えて、特定のハードウェアサポートにより、これらのスパースなクラスタⒿリングされたモデルを利用して、推論のレイテンシ―を大幅に短縮できます。

## 結果

以下は、PQAT と CQAT の協調最適化パスを実験したときに得られた精度と圧縮の結果です。

### スパース性を保持する量子化認識トレーニング （PQAT）

<figure>
<table class="tableizer-table">
<tr class="tableizer-firstrow"><th>Model</th><th>Items</th><th>Baseline</th><th>Pruned Model (50% sparsity)</th><th>QAT Model</th><th>PQAT Model</th></tr>
 <tr><td>DS-CNN-L</td><td>FP32 Top1 Accuracy</td><td><b>95.23%</b></td><td>94.80%</td><td>(Fake INT8) 94.721%</td><td>(Fake INT8) 94.128%</td></tr>
 <tr><td> </td><td>INT8 full integer quantization</td><td>94.48%</td><td><b>93.80%</b></td><td>94.72%</td><td><b>94.13%</b></td></tr>
 <tr><td> </td><td>Compression</td><td>528,128 → 434,879 (17.66%)</td><td>528,128 → 334,154 (36.73%)</td><td>512,224 → 403,261 (21.27%)</td><td>512,032 → 303,997 (40.63%)</td></tr>
 <tr><td>Mobilenet_v1-224</td><td>FP32 Top 1 Accuracy</td><td><b>70.99%</b></td><td>70.11%</td><td>(Fake INT8) 70.67%</td><td>(Fake INT8) 70.29%</td></tr>
 <tr><td> </td><td>INT8 full integer quantization</td><td>69.37%</td><td><b>67.82%</b></td><td>70.67%</td><td><b>70.29%</b></td></tr>
 <tr><td> </td><td>Compression</td><td>4,665,520 → 3,880,331 (16.83%)</td><td>4,665,520 → 2,939,734 (37.00%)</td><td>4,569,416 → 3,808,781 (16.65%)</td><td>4,569,416 → 2,869,600 (37.20%)</td></tr>
</table>
</figure>

### クラスタリングを保持する量子化認識トレーニング （CQAT）

<figure>
<table class="tableizer-table">
<tr class="tableizer-firstrow"><th>Model</th><th>Items</th><th>Baseline</th><th>Clustered Model</th><th>QAT Model</th><th>CQAT Model</th></tr>
 <tr><td>Mobilenet_v1 on CIFAR-10</td><td>FP32 Top1 Accuracy</td><td><b>94.88%</b></td><td>94.48%</td><td>(Fake INT8) 94.80%</td><td>(Fake INT8) 94.60%</td></tr>
 <tr><td> </td><td>INT8 full integer quantization</td><td>94.65%</td><td><b>94.41%</b></td><td>94.77%</td><td><b>94.52%</b></td></tr>
 <tr><td> </td><td>Size</td><td>3.00 MB</td><td>2.00 MB</td><td>2.84 MB</td><td>1.94 MB</td></tr>
 <tr><td>Mobilenet_v1 on ImageNet</td><td>FP32 Top 1 Accuracy</td><td><b>71.07%</b></td><td>65.30%</td><td>(Fake INT8) 70.39%</td><td>(Fake INT8) 65.35%</td></tr>
 <tr><td> </td><td>INT8 full integer quantization</td><td>69.34%</td><td><b>60.60%</b></td><td>70.35%</td><td><b>65.42%</b></td></tr>
 <tr><td> </td><td>Compression</td><td>4,665,568 → 3,886,277 (16.7%)</td><td>4,665,568 → 3,035,752 (34.9%)</td><td>4,569,416 → 3,804,871 (16.7%)</td><td>4,569,472 → 2,912,655 (36.25%)</td></tr>
</table>
</figure>

### チャネルごとにクラスタリングされたモデルの CQAT および PCQAT の結果

以下の結果は、[チャネルごとのクラスタリング](https://www.tensorflow.org/model_optimization/guide/clustering)の手法で得られたものです。モデルの畳み込み層がチャネルごとにクラスタリングされている場合、モデルの精度が高くなることを示しています。モデルに多くの畳み込み層がある場合は、チャネルごとにクラスタリングすることをお勧めします。圧縮率は同じままですが、モデルの精度は高くなります。モデル最適化パイプラインは、この実験では「クラスタリング-&gt;クラスタリングを保持する QAT-&gt;ポストトレーニング量子化、int8」です。

<figure>
<table  class="tableizer-table">
<tr class="tableizer-firstrow"><th>Model</th><th>Clustered -> CQAT, int8 quantized</th><th>Clustered per channel -> CQAT, int8 quantized</th>
 <tr><td>DS-CNN-L</td><td>95.949%</td><td> 96.44%</td></tr>
 <tr><td>MobileNet-V2</td><td>71.538%</td><td>72.638%</td></tr>
 <tr><td>MobileNet-V2 (pruned)</td><td>71.45%</td><td>71.901%</td></tr>
</table>
</figure>

## 例

ここで説明する協調最適化手法のエンドツーエンドの例については、[CQAT](https://www.tensorflow.org/model_optimization/guide/combine/cqat_example)、[PQAT](https://www.tensorflow.org/model_optimization/guide/combine/pqat_example)、[スパース性を維持するクラスタリング](https://www.tensorflow.org/model_optimization/guide/combine/sparse_clustering_example)、および [PCQAT](https://www.tensorflow.org/model_optimization/guide/combine/pcqat_example) ノートブックの例を参照してください。
