# TensorFlow Data Validation: データの確認と分析

データが TFX のパイプラインに投入されると、データの分析と変換に TFX コンポーネント使用できるようになります。これらのツールは、モデルをトレーニングする前であっても使用することが可能です。

データを分析および変換する理由はいくつもあります。

- To find problems in your data. Common problems include:
    - Missing data, such as features with empty values.
    - Labels treated as features, so that your model gets to peek at the right answer during training.
    - Features with values outside the range you expect.
    - Data anomalies.
    - 転移学習で作成されたモデルには、トレーニングデータと一致しない前処理があります。
- To engineer more effective feature sets. For example, you can identify:
    - 特に有用な特徴量
    - Redundant features.
    - Features that vary so widely in scale that they may slow learning.
    - Features with little or no unique predictive information.

TFX tools can both help find data bugs, and help with feature engineering.

## TensorFlow Data Validation

- [Overview](#overview)
- [Schema Based Example Validation](#schema_based_example_validation)
- [Training-Serving Skew Detection](#skewdetect)
- [Drift Detection](#drift_detection)

### 概要

TensorFlow Data Validation identifies anomalies in training and serving data, and can automatically create a schema by examining the data. The component can be configured to detect different classes of anomalies in the data. It can

1. Perform validity checks by comparing data statistics against a schema that codifies expectations of the user.
2. Detect training-serving skew by comparing examples in training and serving data.
3. Detect data drift by looking at a series of data.

We document each of these functionalities independently:

- [Schema Based Example Validation](#schema_based_example_validation)
- [Training-Serving Skew Detection](#skewdetect)
- [Drift Detection](#drift_detection)

### スキーマに基づく Example の検証

TensorFlow Data Validation identifies any anomalies in the input data by comparing data statistics against a schema. The schema codifies properties which the input data is expected to satisfy, such as data types or categorical values, and can be modified or replaced by the user.

Tensorflow Data Validation は通常、TFX パイプラインのコンテキストで (i) ExampleGen から取得されたすべての分割、(ii) Transform により使用されるすべての変換前データ、および (iii) Transform により生成されたすべての変換後データに対して複数回呼び出されます 。Transform (ii-iii) のコンテキストで呼び出される場合、統計オプションとスキーマベースの制約は、[`stats_options_updater_fn`](tft.md) を定義することで設定できます。これは、非構造化データ (テキストデータの特徴量など) を検証するときに特に役立ちます。例については、[ユーザーコード](https://github.com/tensorflow/tfx/blob/master/tfx/examples/bert/mrpc/bert_mrpc_utils.py)を参照してください。

#### スキーマの高度な機能

This section covers more advanced schema configuration that can help with special setups.

##### スパースな特徴量

Encoding sparse features in Examples usually introduces multiple Features that are expected to have the same valency for all Examples. For example the sparse feature:

<pre><code>
WeightedCategories = [('CategoryA', 0.3), ('CategoryX', 0.7)]
</code></pre>

would be encoded using separate Features for index and value:

<pre><code>
WeightedCategoriesIndex = ['CategoryA', 'CategoryX']
WeightedCategoriesValue = [0.3, 0.7]
</code></pre>

with the restriction that the valency of the index and value feature should match for all Examples. This restriction can be made explicit in the schema by defining a sparse_feature:

<pre><code class="lang-proto">
sparse_feature {
  name: 'WeightedCategories'
  index_feature { name: 'WeightedCategoriesIndex' }
  value_feature { name: 'WeightedCategoriesValue' }
}
</code></pre>

スパースな特徴量の定義では、スキーマに存在する特徴量から、インデックスに1つ以上、値に1つを指定する必要があります。スパースな特徴量を明示的に指定すると、TFDV は参照されている特徴量の値がすべて一致することを確認できます。

いくつかのユースケースでは特徴量間に値に関する同様の制限が導入されることがありますが、必ずしもスパースな特徴量としてエンコードする必要はありません。そのようなケースでスパースな特徴量を用いることは妨げにはなりませんが、理想的ではありません。

##### スキーマの環境設定

データの検証において、デフォルトではパイプラインのすべての Example が単一のスキーマに準拠していると想定しています。場合によっては、スキーマにバリエーションが必要になるケースがあります。たとえば、ラベルとして用いられる特徴量はトレーニング時には与えられ (そして検証される必要があり) ますが、本番環境では与えられません。環境設定、特に`default_environment()`、`in_environment()`、`not_in_environment()` はこのような要求を満たすために利用できます。

例として、'LABEL' という特徴量はトレーニング時には必要なものの、本番環境では欠損していることが期待される場合を考察しましょう。これは次のようにして表現できます。

- スキーマで異なる 2 つの環境 ["SERVING", "TRAINING"] を定義し、'LABEL' を "TRAINING" 環境にだけ関連付けます。
- Associate the training data with environment "TRAINING" and the serving data with environment "SERVING".

##### スキーマ生成

入力データのスキーマは TensorFlow [Schema](https://github.com/tensorflow/metadata/blob/master/tensorflow_metadata/proto/v0/schema.proto) のインスタンスとして指定できます。

スキーマを初めから手動で構成する代わりに、TensorFlow Data Validation の自動スキーマ構成機能を利用することができます。具体的には、パイプラインで利用可能なトレーニングデータをもとに計算された統計量に基づいて、TensorFlow Data Validation が自動的に構築したスキーマを最初のバージョンとして利用することができます。ユーザーは単に、自動的に生成されたスキーマを確認し、必要であれば修正し、バージョン管理システムに登録して、さらなる検証のためにパイプラインに明示的に組み込むだけが必要となります。

TFDV includes `infer_schema()` to generate a schema automatically.  For example:

```python
schema = tfdv.infer_schema(statistics=train_stats)
tfdv.display_schema(schema=schema)
```

これは次のルールに従って自動的なスキーマ生成を実行します。

- スキーマがすでに自動生成されている場合、それがそのまま使われます。

- それ以外の場合、TensorFlow Data Validation は利用可能なデータの統計量を確認し、データにあったスキーマを計算します。

*Note: The auto-generated schema is best-effort and only tries to infer basic properties of the data. It is expected that users review and modify it as needed.*

### Training-Serving Skew Detection<a name="skewdetect"></a>

#### 概要

TensorFlow Data Validation can detect distribution skew between training and serving data. Distribution skew occurs when the distribution of feature values for training data is significantly different from serving data. One of the key causes for distribution skew is using either a completely different corpus for training data generation to overcome lack of initial data in the desired corpus. Another reason is a faulty sampling mechanism that only chooses a subsample of the serving data to train on.

##### シナリオの例

Note: For instance, in order to compensate for an underrepresented slice of data, if a biased sampling is used without upweighting the downsampled examples appropriately, the distribution of feature values between training and serving data gets artificially skewed.

See the [TensorFlow Data Validation Get Started Guide](https://www.tensorflow.org/tfx/data_validation/get_started#checking_data_skew_and_drift) for information about configuring training-serving skew detection.

### Drift Detection

Drift detection is supported between consecutive spans of data (i.e., between span N and span N+1), such as between different days of training data. We express drift in terms of [L-infinity distance](https://en.wikipedia.org/wiki/Chebyshev_distance) for categorical features and approximate [Jensen-Shannon divergence](https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence) for numeric features. You can set the threshold distance so that you receive warnings when the drift is higher than is acceptable. Setting the correct distance is typically an iterative process requiring domain knowledge and experimentation.

See the [TensorFlow Data Validation Get Started Guide](https://www.tensorflow.org/tfx/data_validation/get_started#checking_data_skew_and_drift) for information about configuring drift detection.

## データの確認のための可視化の利用

TensorFlow Data Validation は特徴量の分布を可視化するためのツールも提供しています。[Facets](https://pair-code.github.io/facets/) を使って Jupyter ノートブック上でこれらの分布を確認することで、データに関する一般的な問題を確認することができます。

![Feature stats](images/feature_stats.png)

### 疑わしい分布の識別

You can identify common bugs in your data by using a Facets Overview display to look for suspicious distributions of feature values.

#### 不均衡データ

An unbalanced feature is a feature for which one value predominates. Unbalanced features can occur naturally, but if a feature always has the same value you may have a data bug. To detect unbalanced features in a Facets Overview, choose "Non-uniformity" from the "Sort by" dropdown.

The most unbalanced features will be listed at the top of each feature-type list. For example, the following screenshot shows one feature that is all zeros, and a second that is highly unbalanced, at the top of the "Numeric Features" list:

![Visualization of unbalanced data](images/unbalanced.png)

#### 一様分布に従うデータ

一様分布に従う特徴量はすべての取りうる値が同じ程度の頻度で出現する特徴量です。不均衡データと同様に、この分布は自然に生じます。しかし、データのバグによっても引き起こされます。

To detect uniformly distributed features in a Facets Overview, choose "Non- uniformity" from the "Sort by" dropdown and check the "Reverse order" checkbox:

![Histogram of uniform data](images/uniform.png)

String data is represented using bar charts if there are 20 or fewer unique values, and as a cumulative distribution graph if there are more than 20 unique values. So for string data, uniform distributions can appear as either flat bar graphs like the one above or straight lines like the one below:

![Line graph: cumulative distribution of uniform data](images/uniform_cumulative.png)

##### 一様分布を生成しうるバグの例

Here are some common bugs that can produce uniformly distributed data:

- Using strings to represent non-string data types such as dates. For example, you will have many unique values for a datetime feature with representations like "2017-03-01-11-45-03". Unique values will be distributed uniformly.

- Including indices like "row number" as features. Here again you have many unique values.

#### データの欠損

To check whether a feature is missing values entirely:

1. "Sort by" ドロップダウンから "Amount missing/zero" を選択します。
2. "Reverse order" チェックボックスをオンにします。
3. "missing" 列を見て、特徴量に含まれる欠損値の割合を確認します。

A data bug can also cause incomplete feature values. For example you may expect a feature's value list to always have three elements and discover that sometimes it only has one. To check for incomplete values or other cases where feature value lists don't have the expected number of elements:

1. 右側の "Chart to show" ドロップダウンから、"Value list length" を選択します。

2. Look at the chart to the right of each feature row. The chart shows the range of value list lengths for the feature. For example, the highlighted row in the screenshot below shows a feature that has some zero-length value lists:

![Facets Overview display with feature with zero-length feature value lists](images/zero_length.png)

#### 特徴量間のスケールの大きな違い

If your features vary widely in scale, then the model may have difficulties learning. For example, if some features vary from 0 to 1 and others vary from 0 to 1,000,000,000, you have a big difference in scale. Compare the "max" and "min" columns across features to find widely varying scales.

Consider normalizing feature values to reduce these wide variations.

#### 無効な値のあるラベル

TensorFlow's Estimators have restrictions on the type of data they accept as labels. For example, binary classifiers typically only work with {0, 1} labels.

Review the label values in the Facets Overview and make sure they conform to the [requirements of Estimators](https://github.com/tensorflow/docs/blob/master/site/en/r1/guide/feature_columns.md).
