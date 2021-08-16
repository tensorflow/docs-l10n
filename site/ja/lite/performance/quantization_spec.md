# TensorFlow Lite 8 ビット量子化の仕様

次のドキュメントでは、TensorFlow Lite の 8 ビット量子化スキームの仕様を説明します。これは、ハードウェア開発者が量子化された TensorFlow Lite モデルを使った推論のハードウェアサポートを提供できるように支援することを目的としています。

## 仕様の要約

私たちは仕様を提供しており、その仕様に従う場合にのみある程度の動作保証を提供することができます。また、ハードウェアにはそれぞれの環境設定や制限があり、仕様を実装する際には、そのわずかな違いによって、1 ビットまで正確な実装を得られないことがあることも理解しています。ほとんどの場合は許容範囲内に収まるかもしれませんが（また、私たちが知り得る範囲で複数のモデルから収集した演算ごとの許容値を含む一連のテストも提供します）、機械学習（また通例ディープラーニング）の特性により、確固たる保証を提供することはできません。

8 ビット量子化は、次の方程式により、浮動小数点値を概算します。

$$real_value = (int8_value - zero_point) \times scale$$

軸単位（畳み込み演算では「チャンネル単位」）またはテンソルの重み単位は、ゼロ点が 0 に等しい  `[-127, 127]` <br>の範囲の `int8` の 2 の補数によって表現されます。テンソルごとのアクティベーション/入力は、ゼロ点が `[-128, 127]` の範囲となる `[-128, 127]` 範囲の `int8` の 2 の補数によって表現されます。

特定の演算にはほかの例外があり、それについては以下の方に記載されています。

注意: 以前、量子化ツールではテンソル単位の非対称 `uint8` 量子化を使用していました。現在は、参照カーネル、および 8 ビット量子化に最適化されたカーネルでこの仕様が使用されます。

## 署名付き整数と署名無し整数

TensorFlow Lite 量子化は主に、8 ビットの `int8` 量子化に対するツールとカーネルを優先します。これは、0 に等しいゼロ点で対称的な量子化を便宜的に表現するためです。また、多数のバックエンドには、`int8xint8` 累積に追加の最適化を使用しています。

## 軸単位とテンソル単位

テンソル単位の量子化とは、全テンソルごとに 1 つのスケールやゼロ点があることを意味します。軸単位の量子化は、`quantized_dimension` のスライスごとに 1 つのスケールやゼロ点があるということです。量子化された次元は、スケールとゼロ点が対応するテンソルの形状の次元を指定します。たとえば、`dims=[4, 3, 2, 1]` で、量子パラメータが `scale=[1.0, 2.0, 3.0]`、`zero_point=[1, 2, 3]`、`quantization_dimension=1` のテンソル `t` は、`t` の第 2 次元で量子化されます。

```
t[:, 0, :, :] will have scale[0]=1.0, zero_point[0]=1
t[:, 1, :, :] will have scale[1]=2.0, zero_point[1]=2
t[:, 2, :, :] will have scale[2]=3.0, zero_point[2]=3
```

通常、`quantized_dimension` は、畳み込みの重みの `output_channel` ですが、理論的に、カーネル実装の各ドット積に対応する次元である可能性もあるため、パフォーマンスに影響を与えることなく、より高い量子化の粒度を得ることができます。これは、精度を大きく改善させることができます。

TFLite は増え続ける演算で軸単位をサポートしています。このドキュメントを執筆した時点では、Conv2d と DepthwiseConv2d がサポートされています。

## 対称と非対称

アクティベーションは非対称で、署名付きの `int8` の `[-128, 127]` の範囲内にゼロ点を持つことができます。多くのアクティベーションはもともと非対称であり、ゼロ点は、バイナリビットの精度を効果的に得る上で比較的安価な方法です。アクティベーションは重み定数によってのみ乗算されるため、 一定のゼロ点値を大きく最適化できます。

重みは対照的で、ゼロ点は強制的に 0 となります。重み値は、動的な入力とアクティベーション値によって乗算されます。つまり、重みのゼロ点をアクティベーションで乗算するという回避できないランタイムコストが伴ってしまいますが、ゼロ点を 0 にすることで、このコストを回避できます。

数学の説明: [arXiv:1712.05877](https://arxiv.org/abs/1712.05877) のセクション 2.3 に似ていますが、スケール値を軸単位にできるという違いがあります。次のように一般化して言えます。

$A$ は、量子化されたアクティベーションの $m X n$ の行列です。<br> $B$ は、量子化された重みの $n X p$ の行列です。<br> $A$ の $j$ 行目である $a_j$ を $B$ の $k$ 列目である $b_k$ で乗算してみましょう。両方の長さは $n$　です。量子化された整数値とゼロ点の値は、それぞれ $q_a$, $z_a$ と $q_b$, $z_b$ になります。

$$a_j \cdot b_k = \sum_{i=0}^{n} a_{j}^{(i)} b_{k}^{(i)} = \sum_{i=0}^{n} (q_{a}^{(i)} - z_a) (q_{b}^{(i)} - z_b) = \sum_{i=0}^{n} q_{a}^{(i)} q_{b}^{(i)} - \sum_{i=0}^{n} q_{a}^{(i)} z_b - \sum_{i=0}^{n} q_{b}^{(i)} z_a + \sum_{i=0}^{n} z_a z_b$$

<!-- Don't change these `\\(` `\\)` to `$`. mathjax fails here with `$`-->

(\sum_{i=0}^{n} q_{a}^{(i)} q_{b}^{(i)}) 項は、入力値と重み値のドット積を実行しているため、回避できません。

$$\sum_{i=0}^{n} q_{b}^{(i)} z_a$$ と $$\sum_{i=0}^{n} z_a z_b$$ の項は、推論の呼び出しごとに同一のままとなる定数で構成されているため、事前に計算することが可能です。

(\sum_{i=0}^{n} q_{a}^{(i)} z_b) 項は、推論ごとにアクティベーションが変化するため、推論ごとに計算する必要があります。重みを対照的にすることで、この項のタスクを排除することができます。

## int8 量子化演算子の仕様

次に、int8 tflite カーネルの量子化要件を記述します。

```
ADD
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Input 1:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor

AVERAGE_POOL_2D
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  restriction: Input and outputs must all have same scale/zero_point

CONCATENATION
  Input ...:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  restriction: Input and outputs must all have same scale/zero_point

CONV_2D
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Input 1 (Weight):
    data_type  : int8
    range      : [-127, 127]
    granularity: per-axis (dim = 0)
    restriction: zero_point = 0
  Input 2 (Bias):
    data_type  : int32
    range      : [int32_min, int32_max]
    granularity: per-axis
    restriction: (scale, zero_point) = (input0_scale * input1_scale[...], 0)
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor

DEPTHWISE_CONV_2D
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Input 1 (Weight):
    data_type  : int8
    range      : [-127, 127]
    granularity: per-axis (dim = 3)
    restriction: zero_point = 0
  Input 2 (Bias):
    data_type  : int32
    range      : [int32_min, int32_max]
    granularity: per-axis
    restriction: (scale, zero_point) = (input0_scale * input1_scale[...], 0)
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor

FULLY_CONNECTED
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Input 1 (Weight):
    data_type  : int8
    range      : [-127, 127]
    granularity: per-tensor
    restriction: zero_point = 0
  Input 2 (Bias):
    data_type  : int32
    range      : [int32_min, int32_max]
    granularity: per-tensor
    restriction: (scale, zero_point) = (input0_scale * input1_scale[...], 0)
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor

L2_NORMALIZATION
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
    restriction: (scale, zero_point) = (1.0 / 128.0, 0)

LOGISTIC
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
    restriction: (scale, zero_point) = (1.0 / 256.0, -128)

MAX_POOL_2D
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  restriction: Input and outputs must all have same scale/zero_point

MUL
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Input 1:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor

RESHAPE
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  restriction: Input and outputs must all have same scale/zero_point

RESIZE_BILINEAR
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  restriction: Input and outputs must all have same scale/zero_point

SOFTMAX
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
    restriction: (scale, zero_point) = (1.0 / 256.0, -128)

SPACE_TO_DEPTH
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  restriction: Input and outputs must all have same scale/zero_point

TANH
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
    restriction: (scale, zero_point) = (1.0 / 128.0, 0)

PAD
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  restriction: Input and outputs must all have same scale/zero_point

GATHER
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  restriction: Input and outputs must all have same scale/zero_point

BATCH_TO_SPACE_ND
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  restriction: Input and outputs must all have same scale/zero_point

SPACE_TO_BATCH_ND
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  restriction: Input and outputs must all have same scale/zero_point

TRANSPOSE
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  restriction: Input and outputs must all have same scale/zero_point

MEAN
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor

SUB
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Input 1:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor

SUM
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor

SQUEEZE
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  restriction: Input and outputs must all have same scale/zero_point

LOG_SOFTMAX
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
    restriction: (scale, zero_point) = (16.0 / 256.0, 127)

MAXIMUM
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  restriction: Input and outputs must all have same scale/zero_point

ARG_MAX
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor

MINIMUM
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  restriction: Input and outputs must all have same scale/zero_point

LESS
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Input 1:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor

PADV2
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  restriction: Input and outputs must all have same scale/zero_point

GREATER
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Input 1:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor

GREATER_EQUAL
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Input 1:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor

LESS_EQUAL
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Input 1:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor

SLICE
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  restriction: Input and outputs must all have same scale/zero_point

EQUAL
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Input 1:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor

NOT_EQUAL
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Input 1:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor

SHAPE
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor

QUANTIZE (Requantization)
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
```

## 参考資料

[arXiv:1712.05877](https://arxiv.org/abs/1712.05877)
