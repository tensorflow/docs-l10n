# Google Cloud による ML のデータ前処理

このチュートリアルでは、[TensorFlow Transform](https://github.com/tensorflow/transform){: .external}（`tf.Transform` ライブラリ）を使用して、機械学習（ML）用のデータの前処理を実装する方法を説明します。TensorFlow の `tf.Transform` ライブラリでは、データ前処理パイプラインを通じて、インスタンスレベル変換とフルパスデータ変換の両方を定義することができます。これらのパイプラインは [Apache Beam](https://beam.apache.org/){: .external} で効率よく実行され、予測中に、モデルが提供されるときと同じ変換を適用できるようにする TensorFlow グラフを副産物として作成します。

このチュートリアルでは、[Dataflow](https://cloud.google.com/dataflow/docs){: .external } を Apache Beam のランナーとして使用するエンドツーエンドの例を示します。[BigQuery](https://cloud.google.com/bigquery/docs){: .external }、Dataflow、[Vertex AI](https://cloud.google.com/vertex-ai/docs/start/introduction-unified-platform){: .external }、および TensorFlow [Keras](https://www.tensorflow.org/guide/keras/overview) API に精通していることが前提です。また、[Vertex AI Workbench](https://cloud.google.com/vertex-ai/docs/workbench/introduction){: .external } などでの Jupyter ノートブックの使用経験をお持ちであることも前提としています。

このチュートリアルではまた、[ML のデータ前処理: オプションと推奨事項](../../guide/tft_bestpractices)で説明されている前処理タイプの概念、その課題、および Google Cloud のオプションに精通していることも前提としています。

## 目標

- `tf.Transform` ライブラリを使用して Apache Beam パイプラインを実装すること。
- Dataflow でパイプラインを実行すること。
- `tf.Transform` ライブラリを使用して TensorFlow モデルを実装すること。
- そのモデルを予測用にトレーニングして使用すること。

## コスト

このチュートリアルでは、Google Cloud の以下の課金コンポーネントを使用します。

- [Vertex AI](https://cloud.google.com/vertex-ai/pricing){: .external}
- [Cloud Storage](https://cloud.google.com/storage/pricing){: .external}
- [BigQuery](https://cloud.google.com/bigquery/pricing){: .external}
- [Dataflow](https://cloud.google.com/dataflow/pricing){: .external}

<!-- This doc uses plain text cost information because the pricing calculator is pre-configured -->

すべてのリソースを 1 日中使用することを前提に、このチュートリアルを実行するためのコストを見積もるには、事前構成済みの[料金計算ツール](/products/calculator/#id=fad4d8-dd68-45b8-954e-5a56a5d148){: .external }を使用してください。

## 始める前に

1. Google Cloud コンソールのプロジェクト選択ページで、[Google Cloud プロジェクトを作成](https://cloud.google.com/resource-manager/docs/creating-managing-projects)するか選択します。

注意: この手順で作成するリソースを維持する予定がない場合は、既存のプロジェクトを選択するのではなく、プロジェクトを作成してください。以下の手順を完了したら、プロジェクトを削除して、プロジェクトに関連するすべてのリソースを削除できます。

[プロジェクトセレクタに移動](https://console.cloud.google.com/projectselector2/home/dashboard){: class="button button-primary" target="console" track-type="solution" track-name="consoleLink" track-metadata-position="body" }

1. Cloud プロジェクトの課金が有効であることを確認します。[プロジェクトの課金が有効になっているかをチェックする](https://cloud.google.com/billing/docs/how-to/verify-billing-enabled)方法をご覧ください。

2. Dataflow、Vertex AI、および Notebooks API を有効にします。[API を有効化](https://console.cloud.google.com/flows/enableapi?apiid=dataflow,aiplatform.googleapis.com,notebooks.googleapis.com){: class="button button-primary" target="console" track-type="solution" track-name="consoleLink" track-metadata-position="body" }

## このソリューションの Jupyter ノートブック

以下の Jupyter ノートブックには、実装例が示されています。

- [Notebook 1](https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/blogs/babyweight_tft/babyweight_tft_keras_.ipynb){: .external } は、データの前処理に対応しています。詳細は、後の「[Apache Beam パイプラインの実装](#implement-the-apache-beam-pipeline)」で説明されています。
- [Notebook 2](https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/blogs/babyweight_tft/babyweight_tft_keras_02.ipynb){: .external } は、モデルのトレーニングに対応しています。詳細は、後の「[TensorFlow モデルの実装](#implement-the-tensorflow-model)」セクションで説明されています。

以下のセクションでは、これらのノートブックをクローンした後でノートブックを実行し、実装例がどのように動作するかを学習します。

## ユーザー管理のノートブックインスタンスを起動する

1. Google Cloud コンソールで、**Vertex AI Workbench** ページに移動します。

    [Workbench に移動](https://console.cloud.google.com/ai-platform/notebooks/list/instances){: class="button button-primary" target="console" track-type="solution" track-name="consoleLink" track-metadata-position="body" }

2. **ユーザー管理のノートブック**タブで、**+新しいノートブック**をクリックします。

3. インスタンスタイプに、**GPU を使用しない TensorFlow Enterprise 2.8（LTS）**を選択します。

4. **作成**をクリックします。

ノートブックを作成したら、JupyterLab へのプロキシが初期化を完了するのを待ちます。準備が完了したら、ノートブック名の横に **Open JupyterLab** が表示されます。

## ノートブックをクローンする

1. **ユーザー管理のノートブック**タブのノートブック名の横に表示される **Open JupyterLab** をクリックします。JupyterLab インターフェースが新しいタブに開きます。

    JupyterLab に **Build Recommended** ダイアログが表示される場合は、**Cancel** をクリックして、提案されたビルドを拒否します。

2. **Launcher** タブで、**Terminal** をクリックします。

3. ターミナルウィンドウで、以下のようにしてノートブックをクローンします。

    ```sh
    git clone https://github.com/GoogleCloudPlatform/training-data-analyst
    ```

## Apache Beam パイプラインを実装する

このセクションと次のセクションの[Dataflow でパイプラインを実行する](#run-the-pipeline-in-dataflow){: track-type="solution" track-name="internalLink" track-metadata-position="body" }では、Notebook 1 の概要とコンテキストを説明します。このノートブックには、`tf.Transform` ライブラリを使用してデータを前処理する方法を説明した実用的な例が含まれます。この例では、様々な入力に基づいて胎児の体重を予測するために使用される Natality データセットが使用されています。データは、BigQuery による公開 [natality](https://console.cloud.google.com/bigquery?p=bigquery-public-data&d=samples&t=natality&page=table&_ga=2.267763789.2122871960.1676620306-376763843.1676620306){: target="console" track-type="solution" track-name="consoleLink" track-metadata-position="body" } テーブルに保存されます。

### Notebook 1 を実行する

1. JupyterLab インターフェースで **File &gt; Open from path** をクリックし、次のパスを入力します。

    ```sh
    training-data-analyst/blogs/babyweight_tft/babyweight_tft_keras_01.ipynb
    ```

2. **Edit &gt; Clear all outputs** をクリックします。

3. **Install required packages** セクションで最初のセルを実行して、`pip install apache-beam` コマンドを実行します。

    出力の最後に、以下が表示されます。

    ```none{:.devsite-disable-click-to-copy}
    Successfully installed ...
    ```

    出力の依存関係エラーは無視してかまいません。まだ、カーネルを再起動する必要はありません。

4. 2 つ目のセルを実行して、`pip install tensorflow-transform` コマンドを実行します。出力の最後に、以下が表示されます。

    ```none{:.devsite-disable-click-to-copy}
    Successfully installed ...
    Note: you may need to restart the kernel to use updated packages.
    ```

    出力の依存関係エラーは無視してかまいません。

5. **Kernel &gt; Restart Kernel** をクリックします。

6. **Confirm the installed packages** と **Create setup.py to install packages to Dataflow containers** セクションでセルを実行します。

7. **Set global flags** セクションの `PROJECT` と `BUCKET` の横の `your-project` を Cloud プロジェクト ID に置換し、セルを実行します。

8. 残りのすべてのセルをノートブックの最後まで実行します。各セルで行う内容については、ノートブックの指示をご覧ください。

### パイプラインの概要

ノートブックの例では、Dataflow が `tf.Transform` パイプラインを大規模に実行し、データの準備と変換アーティファクトの生成が行われます。この記事の後のセクションで、パイプラインの各ステップを実行する関数が説明されていますが、パイプライン全体のステップは以下のとおりです。

1. BigQuery からトレーニングデータを読み取ります。
2. `tf.Transform` ライブラリを使用して、トレーニングデータの分析と変換を行います。
3. 変換されたトレーニングデータを、[TFRecord](https://www.tensorflow.org/tutorials/load_data/tfrecord){: target="external" class="external" track-type="solution" track-name="externalLink" track-metadata-position="body" } 形式で Cloud Storage に書き込みます。
4. BigQuery から評価データを読み取ります。
5. ステップ 2 で生成された `transform_fn` グラフを使用して、評価データを変換します。
6. 変換されたトレーニングデータを、TFRecord 形式で Cloud Storage に書き込みます。
7. 後でモデルの作成とエクスポートに使用される変換アーティファクトを Cloud Storage に書き込みます。

以下の例は、上記のパイプライン全体の Python コードを示します。各ステップの説明とコードリストは、この後のセクションに示されています。

```py{:.devsite-disable-click-to-copy}
def run_transformation_pipeline(args):

    pipeline_options = beam.pipeline.PipelineOptions(flags=[], **args)

    runner = args['runner']
    data_size = args['data_size']
    transformed_data_location = args['transformed_data_location']
    transform_artefact_location = args['transform_artefact_location']
    temporary_dir = args['temporary_dir']
    debug = args['debug']

    # Instantiate the pipeline
    with beam.Pipeline(runner, options=pipeline_options) as pipeline:
        with impl.Context(temporary_dir):

            # Preprocess train data
            step = 'train'
            # Read raw train data from BigQuery
            raw_train_dataset = read_from_bq(pipeline, step, data_size)
            # Analyze and transform raw_train_dataset
            transformed_train_dataset, transform_fn = analyze_and_transform(raw_train_dataset, step)
            # Write transformed train data to sink as tfrecords
            write_tfrecords(transformed_train_dataset, transformed_data_location, step)

            # Preprocess evaluation data
            step = 'eval'
            # Read raw eval data from BigQuery
            raw_eval_dataset = read_from_bq(pipeline, step, data_size)
            # Transform eval data based on produced transform_fn
            transformed_eval_dataset = transform(raw_eval_dataset, transform_fn, step)
            # Write transformed eval data to sink as tfrecords
            write_tfrecords(transformed_eval_dataset, transformed_data_location, step)

            # Write transformation artefacts
            write_transform_artefacts(transform_fn, transform_artefact_location)

            # (Optional) for debugging, write transformed data as text
            step = 'debug'
            # Write transformed train data as text if debug enabled
            if debug == True:
                write_text(transformed_train_dataset, transformed_data_location, step)
```

### BigQuery から未加工のトレーニングデータを読み取る{: id="read_raw_training_data"}

最初のステップは、`read_from_bq` 関数を使用して、BigQuery から未加工のトレーニングデータを読み取ることです。この関数は、BigQuery から抽出される `raw_dataset` オブジェクトを返します。`data_size` 値を渡し、`train` または `eval` の `step` 値を渡します。BigQuery ソースクエリは、以下の例に示すように、`get_source_query` 関数を使って構築されます。

```py{:.devsite-disable-click-to-copy}
def read_from_bq(pipeline, step, data_size):

    source_query = get_source_query(step, data_size)
    raw_data = (
        pipeline
        | '{} - Read Data from BigQuery'.format(step) >> beam.io.Read(
                           beam.io.BigQuerySource(query=source_query, use_standard_sql=True))
        | '{} - Clean up Data'.format(step) >> beam.Map(prep_bq_row)
    )

    raw_metadata = create_raw_metadata()
    raw_dataset = (raw_data, raw_metadata)
    return raw_dataset
```

`tf.Transform` 前処理を実行する前に、Map、Filter、Group、Window 処理などの一般的な Apache Beam ベースの処理を実行する必要がある場合があります。この例では、コード内の `beam.Map(prep_bq_row)` メソッドによって BigQuery から読み取られたレコードがクリーニングされています。この `prep_bq_row` はカスタム関数です。個のカスタム関数は、カテゴリカル特徴量の数値コードを人間が解読的できるラベルに変換するものです。

また、`tf.Transform` ライブラリを使用して BigQuery から抽出された `raw_data` オブジェクトの解析と変換を行うには、`raw_dataset` オブジェクトを使用する必要があります。これは、`raw_data` と `raw_metadata` オブジェクトのタプルです。`raw_metadata` オブジェクトは、以下のように、`create_raw_metadata` 関数を使って作成されます。

```py{:.devsite-disable-click-to-copy}
CATEGORICAL_FEATURE_NAMES = ['is_male', 'mother_race']
NUMERIC_FEATURE_NAMES = ['mother_age', 'plurality', 'gestation_weeks']
TARGET_FEATURE_NAME = 'weight_pounds'

def create_raw_metadata():

    feature_spec = dict(
        [(name, tf.io.FixedLenFeature([], tf.string)) for name in CATEGORICAL_FEATURE_NAMES] +
        [(name, tf.io.FixedLenFeature([], tf.float32)) for name in NUMERIC_FEATURE_NAMES] +
        [(TARGET_FEATURE_NAME, tf.io.FixedLenFeature([], tf.float32))])

    raw_metadata = dataset_metadata.DatasetMetadata(
        schema_utils.schema_from_feature_spec(feature_spec))

    return raw_metadata
```

ノートブックでこのメソッドを定義するセルの直後に続くセルを実行すると、`raw_metadata.schema` オブジェクトのコンテンツが表示されます。以下の列が含まれています。

- `gestation_weeks`（型: `FLOAT`）
- `is_male`（型: `BYTES`）
- `mother_age`（型: `FLOAT`）
- `mother_race`（型: `BYTES`）
- `plurality`（型: `FLOAT`）
- `weight_pounds`（型: `FLOAT`）

### 未加工のトレーニングデータを変換する

トレーニングデータの未加工の入力特徴量を ML 用に準備するために、一般的な前処理変換を適用するとします。以下の表に示されるように、これらの変換には、フルパスとインスタンスレベルの演算が含まれます。

<table>
<thead>
  <tr>
    <th>入力特徴量</th>
    <th>変換</th>
    <th>必要な統計</th>
    <th>タイプ</th>
    <th>出力特徴量</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td><code>weight_pound</code></td>
    <td>なし</td>
    <td>なし</td>
    <td>該当なし</td>
    <td><code>weight_pound</code></td>
  </tr>
  <tr>
    <td><code>mother_age</code></td>
    <td>正規化</td>
    <td>mean、var</td>
    <td>フルパス</td>
    <td><code>mother_age_normalized</code></td>
  </tr>
  <tr>
    <td><code>mother_age</code></td>
    <td>等サイズのバケット化</td>
    <td>quantiles</td>
    <td>フルパス</td>
    <td><code>mother_age_bucketized</code></td>
  </tr>
  <tr>
    <td><code>mother_age</code></td>
    <td>ログの計算</td>
    <td>なし</td>
    <td>インスタンスレベル</td>
    <td>
        <code>mother_age_log</code>
    </td>
  </tr>
  <tr>
    <td><code>plurality</code></td>
    <td>単胎か多胎か</td>
    <td>なし</td>
    <td>インスタンスレベル</td>
    <td><code>is_multiple</code></td>
  </tr>
  <tr>
    <td><code>is_multiple</code></td>
    <td>名義から数値インデックスへの変換</td>
    <td>vocab</td>
    <td>フルパス</td>
    <td><code>is_multiple_index</code></td>
  </tr>
  <tr>
    <td><code>gestation_weeks</code></td>
    <td>0 から 1 の間でのスケーリング</td>
    <td>min、max</td>
    <td>フルパス</td>
    <td><code>gestation_weeks_scaled</code></td>
  </tr>
  <tr>
    <td><code>mother_race</code></td>
    <td>名義から数値インデックスへの変換</td>
    <td>vocab</td>
    <td>フルパス</td>
    <td><code>mother_race_index</code></td>
  </tr>
  <tr>
    <td><code>is_male</code></td>
    <td>名義から数値インデックスへの変換</td>
    <td>vocab</td>
    <td>フルパス</td>
    <td><code>is_male_index</code></td>
  </tr>
</tbody>
</table>

これらの変換は `preprocess_fn` 関数に実装されており、テンソル（`input_features`のディクショナリを取って、処理済みの特徴量（`output_features`）のディクショナリを返します。

以下のコードは、`tf.Transform` フルパス変換 API（`tft.` 接頭辞を使用）と TensorFlow インスタンスレベル演算（`tf.` 接頭辞を使用）による `preprocess_fn` 関数の実装を示します。

```py{:.devsite-disable-click-to-copy}
def preprocess_fn(input_features):

    output_features = {}

    # target feature
    output_features['weight_pounds'] = input_features['weight_pounds']

    # normalization
    output_features['mother_age_normalized'] = tft.scale_to_z_score(input_features['mother_age'])

    # scaling
    output_features['gestation_weeks_scaled'] =  tft.scale_to_0_1(input_features['gestation_weeks'])

    # bucketization based on quantiles
    output_features['mother_age_bucketized'] = tft.bucketize(input_features['mother_age'], num_buckets=5)

    # you can compute new features based on custom formulas
    output_features['mother_age_log'] = tf.math.log(input_features['mother_age'])

    # or create flags/indicators
    is_multiple = tf.as_string(input_features['plurality'] > tf.constant(1.0))

    # convert categorical features to indexed vocab
    output_features['mother_race_index'] = tft.compute_and_apply_vocabulary(input_features['mother_race'], vocab_filename='mother_race')
    output_features['is_male_index'] = tft.compute_and_apply_vocabulary(input_features['is_male'], vocab_filename='is_male')
    output_features['is_multiple_index'] = tft.compute_and_apply_vocabulary(is_multiple, vocab_filename='is_multiple')

    return output_features
```

`tf.Transform` [フレームワーク](https://github.com/tensorflow/transform){: .external }には、前の例の他に、以下の表に記載の変換を含む変換があります。

<table>
<thead>
  <tr>
  <th>変換</th>
  <th>適用先</th>
  <th>説明</th>
  </tr>
</thead>
<tbody>
    <tr>
    <td><code>scale_by_min_max</code></td>
    <td>数値特徴量</td>
    <td>数値カラムを [<code>output_min</code>,       <code>output_max</code>] の範囲にスケーリングします。</td>
  </tr>
  <tr>
    <td><code>scale_to_0_1</code></td>
    <td>数値特徴量</td>
    <td>[<code>0</code>,<code>1</code>] の範囲にスケーリングされた入力カラムを返します。</td>
  </tr>
  <tr>
    <td><code>scale_to_z_score</code></td>
    <td>数値特徴量</td>
    <td>平均 0 と分散 1 で標準化されたカラムを返します。</td>
  </tr>
  <tr>
    <td><code>tfidf</code></td>
    <td>テキスト特徴量</td>
    <td> <i>x</i> の用語を用語頻度 * 逆文書頻度にマッピングします。</td>
  </tr>
  <tr>
    <td><code>compute_and_apply_vocabulary</code></td>
    <td>カテゴリカル特徴量</td>
    <td>カテゴリカル特徴量の語彙を生成し、それをこの語彙を持つ整数にマッピングします。</td>
  </tr>
  <tr>
    <td><code>ngrams</code></td>
    <td>テキスト特徴量</td>
    <td>n-gram の <code>SparseTensor</code> を作成します。</td>
  </tr>
  <tr>
    <td><code>hash_strings</code></td>
    <td>カテゴリカル特徴量</td>
    <td>文字列をバケットにハッシュ化します。</td>
  </tr>
  <tr>
    <td><code>pca</code></td>
    <td>数値特徴量</td>
    <td>バイアス付きの共分散を使ってデータセットの PCA を計算します。</td>
  </tr>
  <tr>
    <td><code>bucketize</code></td>
    <td>数値特徴量</td>
    <td>等サイズ（変位値ベース）のバケット化カラムを返します。バケットインデックスは各入力に割り当てられます。</td>
  </tr>
</tbody>
</table>

`preprocess_fn` 関数で実装された変換を、パイプラインの前のステップで生成された `raw_train_dataset` オブジェクトに適用するには、`AnalyzeAndTransformDataset` メソッドを使用します。このメソッドは入力として `raw_dataset` オブジェクトを取り、`preprocess_fn` 関数を適用して `transformed_dataset` オブジェクトと the `transform_fn` グラフを生成します。以下は、この処理を説明するコードです。

```py{:.devsite-disable-click-to-copy}
def analyze_and_transform(raw_dataset, step):

    transformed_dataset, transform_fn = (
        raw_dataset
        | '{} - Analyze & Transform'.format(step) >> tft_beam.AnalyzeAndTransformDataset(
            preprocess_fn, output_record_batches=True)
    )

    return transformed_dataset, transform_fn
```

変換は、解析と変換の 2 つのフェーズで未加工のデータに対して適用されます。この記事の後にある図 3 には、`AnalyzeAndTransformDataset` メソッドが `AnalyzeDataset` メソッドと `TransformDataset` メソッドに分解されて示されています。

#### 解析フェーズ

解析フェーズでは、変換に必要な統計を計算するために、未加工のデータがフルパスプロセスで解析されます。これには、平均、分散、最小値、最大値、変分位、語彙の計算が含まれます。解析プロセスは未加工のデータセット（未加工のデータと未加工のメタデータ）を使用して、以下の 2 つの出力を生成します。

- `transform_fn`: 解析フェーズで計算された統計とインスタンスレベル演算の変換ロジック（統計を使用）を含むを TensorFlow グラフです。後の「[グラフを保存する](#save_the_graph)」{: track-type="solution" track-name="internalLink" track-metadata-position="body" }で説明されるとおり、`transform_fn` グラフはモデル `serving_fn` 関数にアタッチするために保存されます。こうすることで、同じ変換をオンライン予測データポイントに適用できるようになります。
- `transform_metadata`: 変換後のデータに期待されるスキーマを説明するオブジェクトです。

解析フェーズは、以下の図 1 で説明されています。

<figure id="tf-transform-analyze-phase">
  <img src="images/data-preprocessing-for-ml-with-tf-transform-tf-transform-analyze-phase.svg"
    alt="The tf.Transform analyze phase.">
  <figcaption><b>Figure 1.</b> The <code>tf.Transform</code> analyze phase.</figcaption>
</figure>

`tf.Transform` [アナライザ](https://github.com/tensorflow/transform/blob/master/tensorflow_transform/beam/analyzer_impls.py){: target="github" class="external" track-type="solution" track-name="gitHubLink" track-metadata-position="body" } には、`min`、`max`、`sum`、`size`、`mean`、`var`、`covariance`、`quantiles`、`vocabulary`、および `pca` が含まれます。

#### 変換フェーズ

変換フェーズでは、変換済みのトレーニングデータが生成するために、解析フェーズで生成された `transform_fn` グラフを使用して、インスタンスレベルのプロセスで未加工のトレーニングデータが変換されます。変換済みのトレーニングデータは変換済みのメタデータ（解析フェーズで生成）と合わせて、`transformed_train_dataset` データセットが生成されます。

変換フェーズは、以下の図 2 で説明されています。

<figure id="tf-transform-transform-phase">
<img src="images/data-preprocessing-for-ml-with-tf-transform-tf-transform-transform-phase.svg"
alt="The tf.Transform transform phase.">
<figcaption><b>Figure 2.</b> The <code>tf.Transform</code> transform phase.</figcaption>
</figure>

特徴量を前処理するには、`preprocess_fn` 関数の実装で、必要な `tensorflow_transform` 変換（コード内で `tft` としてインポート）を呼び出します。たとえば、`tft.scale_to_z_score` 演算を呼び出すと、`tf.Transform` ライブラリはこの関数呼び出しを平均値と分散のアナライザーに解釈してから、変換フェーズで数値特徴量を正規化するために、これらの統計を適用します。これは、`AnalyzeAndTransformDataset(preprocess_fn)` メソッドの呼び出しによって、自動的に実行されます。

この呼び出しによって生成される `transformed_metadata.schema` エンティティには、以下のカラムが含まれます。

- `gestation_weeks_scaled`（型: `FLOAT`）
- `is_male_index`（型: `INT`、is_categorical: `True`）
- `is_multiple_index`（型: `INT`、is_categorical: `True`）
- `mother_age_bucketized`（型: `INT`、is_categorical: `True`）
- `mother_age_log`（型: `FLOAT`）
- `mother_age_normalized`（型: `FLOAT`）
- `mother_race_index`（型: `INT`、is_categorical: `True`）
- `weight_pounds`（型: `FLOAT`）

このシリーズのパート 1 の「[処理演算](data-preprocessing-for-ml-with-tf-transform-pt1#preprocessing_operations)」で説明されているように、特徴量変換は、カテゴリカル特徴量を数値表現に変換します。変換が完了すると、カテゴリカル特徴量は整数値で表されます。`transformed_metadata.schema` エンティティでは、`INT` 型カラムの `is_categorical` フラグによって、カラムがカテゴリカル特徴量であるか真の数値特徴量であるかが示されます。

### 変換済みのトレーニングデータを書き込む{: id="step_3_write_transformed_training_data"}

解析フェーズと変換フェーズと通じて、`preprocess_fn` 関数によるトレーニングデータの前処理が完了したら、そのデータをシンクに書き込んで、TensorFlow モデルのトレーニングに使用できるようにします。Dataflow を使って Apache Beam パイプラインを実行する場合、このシンクは Cloud Storage です。そうでない場合は、ローカルディスクとなります。データを固定幅で書式設定された CSV ファイルとして書き込むことも可能ですが、TensorFlow データセットでは、TFRecord 形式が推奨されるファイル形式です。これは、レコード指向の単純なバイナリ形式で、`tf.train.Example` プロトコルバッファーメッセージで構成されています。

それぞれの `tf.train.Example` レコードには、1 つ以上の特徴量が含まれます。これらは、トレーニングでモデルにフィードされるときに、テンソルに変換されます。以下のコードでは、変換されたデータセットを指定された場所にある TFRecord に書き込んでいます。

```py{:.devsite-disable-click-to-copy}
def write_tfrecords(transformed_dataset, location, step):
    from tfx_bsl.coders import example_coder

    transformed_data, transformed_metadata = transformed_dataset
    (
        transformed_data
        | '{} - Encode Transformed Data'.format(step) >> beam.FlatMapTuple(
                            lambda batch, _: example_coder.RecordBatchToExamples(batch))
        | '{} - Write Transformed Data'.format(step) >> beam.io.WriteToTFRecord(
                            file_path_prefix=os.path.join(location,'{}'.format(step)),
                            file_name_suffix='.tfrecords')
    )
```

### 評価データを読み取り、変換して書き込む

トレーニングデータを変換して `transform_fn` グラフを生成したら、それを使用して評価データに変換できます。まず、前の「[BigQuery から未加工のトレーニングデータを読み取る](#read-raw-training-data-from-bigquery)」{: track-type="solution" track-name="internalLink" track-metadata-position="body" } で説明した `read_from_bq` 関数を使用して、`step` の `eval` 値を渡して BigQuery から評価データを読み取ってクリーニングします。次に、以下のコードを使用して、未加工の評価データセット（`raw_dataset`）を期待される変換済みのフォーマット（`transformed_dataset`）に変換します。

```py{:.devsite-disable-click-to-copy}
def transform(raw_dataset, transform_fn, step):

    transformed_dataset = (
        (raw_dataset, transform_fn)
        | '{} - Transform'.format(step) >> tft_beam.TransformDataset(output_record_batches=True)
    )

    return transformed_dataset
```

評価データを変換する際は、`transform_fn` グラフのロジックとトレーニングデータの解析フェーズで計算された統計を使って、インスタンスレベルの演算のみが使用されます。言い換えると、評価データをフルパスで解析して、評価データの数値特徴量の z スコア正規化用に平均値や分散などの新しい統計を計算することはありません。代わりに、トレーニングデータからの計算済みの統計を使用して、インスタンスレベルで評価データを変換します。

したがって、トレーニングデータのコンテキストで `AnalyzeAndTransform` メソッドを使用して統計を計算し、データを変換します。同時に、トレーニングデータで計算された統計を使用してデータを変換するだけのために、評価データの変換のコンテキストで `TransformDataset` メソッドを使用します。

次に、トレーニングプロセス中に TensorFlow モデルを評価するためのデータを TFRecord 形式でシンク（ランナーに応じて、Cloud Storage かローカルディスク）に書き込みます。これを行うには、「[変換済みのトレーニングデータを書き込む](#step_3_write_transformed_training_data){: track-type="solution" track-name="internalLink" track-metadata-position="body" }」で説明された `write_tfrecords` 関数を使用します。以下の図 3 は、トレーニングデータの解析フェーズで生成された `transform_fn` グラフが評価データを変換するために使用される方法を示しています。

<figure id="transform-eval-data-using-transform-fn">
  <img src="images/data-preprocessing-for-ml-with-tf-transform-transforming-eval-data-using-transform_fn.svg"
    alt="Transforming evaluation data using the transform_fn graph.">
  <figcaption><b>Figure 3.</b> Transforming evaluation data using the <code>transform_fn</code> graph.</figcaption>
</figure>

### グラフを保存する

`tf.Transform` 前処理パイプラインの最後のステップは、アーティファクトの保存です。これには、トレーニングデータの解析フェーズで生成された `transform_fn` グラフが含まれます。アーティファクトを保存するコードは、以下の `write_transform_artefacts` 関数で示されます。

```py{:.devsite-disable-click-to-copy}
def write_transform_artefacts(transform_fn, location):

    (
        transform_fn
        | 'Write Transform Artifacts' >> transform_fn_io.WriteTransformFn(location)
    )
```

これらのアーティファクトは、後で、モデルをトレーニングしてサービング用にエクスポートするために使用されます。次のセクションで説明されるように、以下のアーティファクトも生成されます。

- `saved_model.pb`: 変換ロジック（`transform_fn` グラフ）を含む TensorFlow グラフを表現します。これは、未加工のデータポイントを変換済みのフォーマットに変換するために、モデルサービングインターフェースにアタッチされます。
- `variables`: トレーニングデータの解析フェーズ中に計算され、`saved_model.pb` アーティファクトの変換ロジックで使用される統計が含まれます。
- `assets`: `compute_and_apply_vocabulary` メソッドで処理されたカテゴリカル特徴量ごとの語彙ファイルが含まれます。未加工の入力名義値を数値インデックスに変換するためにサービング中に使用されます。
- `transformed_metadata`: 変換データのスキーマを記述した `schema.json` ファイルを含むディクショナリです。

## Dataflow でパイプラインを実行する{:#run_the_pipeline_in_dataflow}

`tf.Transform` パイプラインを定義したら、Dataflow を使ってこのパイプラインを実行します。以下の図 4 は、例で記述された `tf.Transform` パイプラインの Dataflow 実行グラフを示します。

<figure id="dataflow-execution-graph">
<img src="images/data-preprocessing-for-ml-with-tf-transform-dataflow-execution-graph.png"
alt="Dataflow execution graph of the tf.Transform pipeline." class="screenshot">
<figcaption><b>Figure 4.</b> Dataflow execution graph
of the <code>tf.Transform</code> pipeline.</figcaption>
</figure>

Dataflow パイプラインを実行して、トレーニングデータと評価データを前処理したら、ノートブックの最後のセルを実行して、Cloud Storage 内の生成されたオブジェクトを調べることができます。このセクションのコードスニペットは、その結果を示しており、<var><code>YOUR_BUCKET_NAME</code></var> は Cloud Storage バケットの名前です。

変換済みのトレーニングデータと評価データは、TFRecord 形式で以下の場所に保存されます。

```none{:.devsite-disable-click-to-copy}
gs://YOUR_BUCKET_NAME/babyweight_tft/transformed
```

変換アーティファクトは、以下の場所に生成されます。

```none{:.devsite-disable-click-to-copy}
gs://YOUR_BUCKET_NAME/babyweight_tft/transform
```

以下のリストはパイプラインの出力で、生成されたデータオブジェクトとアーティファクトを示します。

```none{:.devsite-disable-click-to-copy}
transformed data:
gs://YOUR_BUCKET_NAME/babyweight_tft/transformed/eval-00000-of-00001.tfrecords
gs://YOUR_BUCKET_NAME/babyweight_tft/transformed/train-00000-of-00002.tfrecords
gs://YOUR_BUCKET_NAME/babyweight_tft/transformed/train-00001-of-00002.tfrecords

transformed metadata:
gs://YOUR_BUCKET_NAME/babyweight_tft/transform/transformed_metadata/
gs://YOUR_BUCKET_NAME/babyweight_tft/transform/transformed_metadata/asset_map
gs://YOUR_BUCKET_NAME/babyweight_tft/transform/transformed_metadata/schema.pbtxt

transform artefact:
gs://YOUR_BUCKET_NAME/babyweight_tft/transform/transform_fn/
gs://YOUR_BUCKET_NAME/babyweight_tft/transform/transform_fn/saved_model.pb
gs://YOUR_BUCKET_NAME/babyweight_tft/transform/transform_fn/assets/
gs://YOUR_BUCKET_NAME/babyweight_tft/transform/transform_fn/variables/

transform assets:
gs://YOUR_BUCKET_NAME/babyweight_tft/transform/transform_fn/assets/
gs://YOUR_BUCKET_NAME/babyweight_tft/transform/transform_fn/assets/is_male
gs://YOUR_BUCKET_NAME/babyweight_tft/transform/transform_fn/assets/is_multiple
gs://YOUR_BUCKET_NAME/babyweight_tft/transform/transform_fn/assets/mother_race
```

## TensorFlow モデルを実装する{: id="implementing_the_tensorflow_model"}

このセクションと次の「[モデルをトレーニングして予測に使用する](#train_and_use_the_model_for_predictions){: track-type="solution" track-name="internalLink" track-metadata-position="body" }」セクションでは、Notebook 2 の概要とコンテキストを説明します。このノートブックには、胎児の体重を予測する ML モデルの例が含まれます。この例では、Keras API を使って TensorFlow モデルが実装されています。モデルでは、前述の `tf.Transform` 前処理パイプラインで生成されたデータとアーティファクトが使用されます。

### Notebook 2 を実行する

1. JupyterLab インターフェースで **File &gt; Open from path** をクリックし、次のパスを入力します。

    ```sh
    training-data-analyst/blogs/babyweight_tft/babyweight_tft_keras_02.ipynb
    ```

2. **Edit &gt; Clear all outputs** をクリックします。

3. **Install required packages** セクションで最初のセルを実行して、`pip install tensorflow-transform` コマンドを実行します。

    出力の最後に、以下が表示されます。

    ```none{:.devsite-disable-click-to-copy}
    Successfully installed ...
    Note: you may need to restart the kernel to use updated packages.
    ```

    出力の依存関係エラーは無視してかまいません。

4. **Kernel** メニューで **Restart Kernel** を選択します。

5. **Confirm the installed packages** と **Create setup.py to install packages to Dataflow containers** セクションでセルを実行します。

6. **Set global flags** セクションの `PROJECT` と `BUCKET` の横の <code>your-project</code> を Cloud プロジェクト ID に置換し、セルを実行します。

7. 残りのすべてのセルをノートブックの最後まで実行します。各セルで行う内容については、ノートブックの指示をご覧ください。

### モデル作成の概要

以下は、モデルの作成ステップです。

1. `transformed_metadata` ディレクトリに保存されているスキーマ情報を使用して、特徴量カラムを作成します。
2. Keras API でこの特徴量カラムをモデルへの入力として使用して、ワイドアンドディープモデルを作成します。
3. 変換アーティファクトを使用してトレーニングデータと評価データを読み取って解析する `tfrecords_input_fn` 関数を作成します。
4. モデルをトレーニングして評価します。
5. `transform_fn` グラフがアタッチされた `serving_fn` 関数を定義して、トレーニング済みのモデルをエクスポートします。
6. [`saved_model_cli`](https://www.tensorflow.org/guide/saved_model) ツールを使用して、エクスポートされたモデルを検査します。
7. エクスポートされたモデルを予測に使用します。

この記事ではモデルの構築方法については説明しないため、モデルの構築方法やトレーニング方法の詳細には触れていませんが、以下のセクションでは、`tf.Transform` プロセスで生成される `transform_metadata` ディレクトリに保存された情報が、モデルの特徴量カラムを作成する際にどのように使用されるかについて示しています。この記事ではまた、同様に `tf.Transform` プロセスで生成される `transform_fn` グラフが、モデルがサービング用にエクスポートされる際にどのように `serving_fn` 関数で使用されているかについても説明しています。

### モデルのトレーニングで生成済みの変換アーティファクトを使用する

TensorFlow モデルをトレーニングする場合、前のデータ処理ステップで生成される変換済みの `train` オブジェクトと `eval` オブジェクトを使用します。これらのオブジェクトは TFRecord 形式で保存された共有ファイルです。前のステップで生成された`transformed_metadata` ディレクトリのスキーマ情報は、トレーニングと評価でモデルにフィードされるデータ（`tf.train.Example` オブジェクト）を解析する際に役立ちます。

#### データを解析する

モデルにトレーニングデータと評価データをフィードするファイルを TFRecord 形式で読み取るため、ファイル内の各 `tf.train.Example` オブジェクトを解析して特徴量（テンソル）のディクショナリを作成する必要があります。こうすることで、特徴量が特徴量カラムを使用してモデルの入力レイヤーにマッピングされることを保証できます。つまり、モデルのトレーニングと評価のインターフェースとしての役割を果たします。データを解析するには、前のステップで生成されたアーティファクトから作成される `TFTransformOutput` オブジェクトを使用します。

1. 前の前処理ステップで生成して保存したアーティファクトから、`TFTransformOutput` オブジェクトを作成します。これは、「[グラフを保存する](#save_the_graph){: track-type="solution" track-name="internalLink" track-metadata-position="body" }」セクションで説明されています。

    ```py
    tf_transform_output = tft.TFTransformOutput(TRANSFORM_ARTEFACTS_DIR)
    ```

2. `TFTransformOutput` オブジェクトから `feature_spec` オブジェクトを抽出します。

    ```py
    tf_transform_output.transformed_feature_spec()
    ```

3. `tfrecords_input_fn` 関数と同様に、`feature_spec` オブジェクトを使用して、`tf.train.Example` オブジェクトに含まれる特徴量を指定します。

    ```py
    def tfrecords_input_fn(files_name_pattern, batch_size=512):

        tf_transform_output = tft.TFTransformOutput(TRANSFORM_ARTEFACTS_DIR)
        TARGET_FEATURE_NAME = 'weight_pounds'

        batched_dataset = tf.data.experimental.make_batched_features_dataset(
            file_pattern=files_name_pattern,
            batch_size=batch_size,
            features=tf_transform_output.transformed_feature_spec(),
            reader=tf.data.TFRecordDataset,
            label_key=TARGET_FEATURE_NAME,
            shuffle=True).prefetch(tf.data.experimental.AUTOTUNE)

        return batched_dataset
    ```

#### 特徴量カラムを作成する

このパイプラインは、モデルがトレーニングと評価に期待する変換済みデータのスキーマを記述したスキーマ情報を `transformed_metadata` ディレクトリに生成します。このスキーマには、以下のような特徴量名とデータ型が含まれます。

- `gestation_weeks_scaled`（型: `FLOAT`）
- `is_male_index`（型: `INT`、is_categorical: `True`）
- `is_multiple_index`（型: `INT`、is_categorical: `True`）
- `mother_age_bucketized`（型: `INT`、is_categorical: `True`）
- `mother_age_log`（型: `FLOAT`）
- `mother_age_normalized`（型: `FLOAT`）
- `mother_race_index`（型: `INT`、is_categorical: `True`）
- `weight_pounds`（型: `FLOAT`）

この情報を表示するには、以下のコマンドを使用します。

```sh
transformed_metadata = tft.TFTransformOutput(TRANSFORM_ARTEFACTS_DIR).transformed_metadata
transformed_metadata.schema
```

以下のコードは、特徴量名を使用して特徴量カラムを作成する方法を示します。

```py
def create_wide_and_deep_feature_columns():

    deep_feature_columns = []
    wide_feature_columns = []
    inputs = {}
    categorical_columns = {}

    # Select features you've checked from the metadata
    # Categorical features are associated with the vocabulary size (starting from 0)
    numeric_features = ['mother_age_log', 'mother_age_normalized', 'gestation_weeks_scaled']
    categorical_features = [('is_male_index', 1), ('is_multiple_index', 1),
                            ('mother_age_bucketized', 4), ('mother_race_index', 10)]

    for feature in numeric_features:
        deep_feature_columns.append(tf.feature_column.numeric_column(feature))
        inputs[feature] = layers.Input(shape=(), name=feature, dtype='float32')

    for feature, vocab_size in categorical_features:
        categorical_columns[feature] = (
            tf.feature_column.categorical_column_with_identity(feature, num_buckets=vocab_size+1))
        wide_feature_columns.append(tf.feature_column.indicator_column(categorical_columns[feature]))
        inputs[feature] = layers.Input(shape=(), name=feature, dtype='int64')

    mother_race_X_mother_age_bucketized = tf.feature_column.crossed_column(
        [categorical_columns['mother_age_bucketized'],
         categorical_columns['mother_race_index']],  55)
    wide_feature_columns.append(tf.feature_column.indicator_column(mother_race_X_mother_age_bucketized))

    mother_race_X_mother_age_bucketized_embedded = tf.feature_column.embedding_column(
        mother_race_X_mother_age_bucketized, 5)
    deep_feature_columns.append(mother_race_X_mother_age_bucketized_embedded)

    return wide_feature_columns, deep_feature_columns, inputs
```

このコードは、数値特徴量の `tf.feature_column.numeric_column` カラムと、カテゴリカル特徴量の `tf.feature_column.categorical_column_with_identity` カラムを作成します。

また、このシリーズのパート 1 の「[オプション C: TensorFlow](/architecture/data-preprocessing-for-ml-with-tf-transform-pt1#option_c_tensorflow){: track-type="solution" track-name="internalLink" track-metadata-position="body" }」で説明されているように、拡張特徴量カラムも作成できます。このシリーズで使用される例では、`tf.feature_column.crossed_column` 特徴量カラムを使用して、`mother_race` と `mother_age_bucketized` の特徴量を掛け合わせることで、`mother_race_X_mother_age_bucketized` という新しい特徴量が作成されます。掛け合わせられた特徴量の低次元で密な表現は、`tf.feature_column.embedding_column` 特徴量カラムを使用して作成されます。

以下の図 5 では、変換済みのデータと、TensorFlow モデルの定義とトレーニングに、変換済みのメタデータがどのように使用されるかを示します。

<figure id="training-tf-with-transformed-data">
<img src="images/data-preprocessing-for-ml-with-tf-transform-training-tf-model-with-transformed-data.svg"
alt="Training the TensorFlow model with transformed data.">
<figcaption><b>Figure 5.</b> Training the TensorFlow model with
the transformed data.</figcaption>
</figure>

### 予測のサービング用のモデルをエクスポートする

Keras API で TensorFlow モデルをトレーニングしたら、新しいデータポイントを予測用にサービングできるように、トレーニング済みのモデルを SavedModel オブジェクトとしてエクスポートします。モデルをエクスポートする際は、サービング中に期待される入力特徴量スキーマとしてインターフェースを定義しなければなりません。この入力特徴量スキーマは、以下のコードに示されるように、`serving_fn` 関数に定義されます。

```py{:.devsite-disable-click-to-copy}
def export_serving_model(model, output_dir):

    tf_transform_output = tft.TFTransformOutput(TRANSFORM_ARTEFACTS_DIR)
    # The layer has to be saved to the model for Keras tracking purposes.
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serveing_fn(uid, is_male, mother_race, mother_age, plurality, gestation_weeks):
        features = {
            'is_male': is_male,
            'mother_race': mother_race,
            'mother_age': mother_age,
            'plurality': plurality,
            'gestation_weeks': gestation_weeks
        }
        transformed_features = model.tft_layer(features)
        outputs = model(transformed_features)
        # The prediction results have multiple elements in general.
        # But we need only the first element in our case.
        outputs = tf.map_fn(lambda item: item[0], outputs)

        return {'uid': uid, 'weight': outputs}

    concrete_serving_fn = serveing_fn.get_concrete_function(
        tf.TensorSpec(shape=[None], dtype=tf.string, name='uid'),
        tf.TensorSpec(shape=[None], dtype=tf.string, name='is_male'),
        tf.TensorSpec(shape=[None], dtype=tf.string, name='mother_race'),
        tf.TensorSpec(shape=[None], dtype=tf.float32, name='mother_age'),
        tf.TensorSpec(shape=[None], dtype=tf.float32, name='plurality'),
        tf.TensorSpec(shape=[None], dtype=tf.float32, name='gestation_weeks')
    )
    signatures = {'serving_default': concrete_serving_fn}

    model.save(output_dir, save_format='tf', signatures=signatures)
```

サービング中、モデルは未加工のデータポイント（変換前の未加工の特徴量）を期待します。したがって、`serving_fn` 関数は未加工の特徴量を受け取って、それらを Python ディクショナリとして `features` オブジェクトに保存します。ただし、前述のとおり、トレーニング済みのモデルは、変換済みスキーマのデータポイントを期待します。未加工の特徴量を、モデルインターフェースが期待する `transformed_features` オブジェクトに変換するには、以下のステップで、保存した `transform_fn` グラフを `features` オブジェクトに適用します。

1. 前の前処理ステップで生成されて保存されたアーティファクトから `TFTransformOutput` オブジェクトを作成します。

    ```py
    tf_transform_output = tft.TFTransformOutput(TRANSFORM_ARTEFACTS_DIR)
    ```

2. `TFTransformOutput` オブジェクトから `TransformFeaturesLayer` オブジェクトを作成します。

    ```py
    model.tft_layer = tf_transform_output.transform_features_layer()
    ```

3. `TransformFeaturesLayer` オブジェクトを使用して、`transform_fn` グラフを適用します。

    ```py
    transformed_features = model.tft_layer(features)
    ```

以下の図 6 は、サービングするためのモデルをエクスポートするための最後のステップを説明しています。

<figure id="exporting-model-for-serving-with-transform_fn">
  <img src="images/data-preprocessing-for-ml-with-tf-transform-exporting-model-for-serving-with-transform_fn.svg"
    alt="Exporting the model for serving with the transform_fn graph attached.">
  <figcaption><b>Figure 6.</b> Exporting the model for serving with the
    <code>transform_fn</code> graph attached.</figcaption>
</figure>

## モデルをトレーニングして予測に使用する

ノートブックのセルを実行すると、ローカルでモデルをトレーニングできます。コードをパッケージ化して、Vertex AI Training で大規模にモデルをトレーニングする例については、Google Cloud [cloudml-samples](https://github.com/GoogleCloudPlatform/cloudml-samples){: .external } GitHub リポジトリのサンプルとガイドをご覧ください。

`saved_model_cli` ツールで、エクスポートされた SavedModel オブジェクトを検査すると、以下の例で示されるとおり、シグネチャ定義 `signature_def` の `inputs` 要素に未加工の特徴量が含まれているのがわかります。

```py{:.devsite-disable-click-to-copy}
signature_def['serving_default']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['gestation_weeks'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1)
        name: serving_default_gestation_weeks:0
    inputs['is_male'] tensor_info:
        dtype: DT_STRING
        shape: (-1)
        name: serving_default_is_male:0
    inputs['mother_age'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1)
        name: serving_default_mother_age:0
    inputs['mother_race'] tensor_info:
        dtype: DT_STRING
        shape: (-1)
        name: serving_default_mother_race:0
    inputs['plurality'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1)
        name: serving_default_plurality:0
    inputs['uid'] tensor_info:
        dtype: DT_STRING
        shape: (-1)
        name: serving_default_uid:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['uid'] tensor_info:
        dtype: DT_STRING
        shape: (-1)
        name: StatefulPartitionedCall_6:0
    outputs['weight'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1)
        name: StatefulPartitionedCall_6:1
  Method name is: tensorflow/serving/predict
```

ノートブックの残りのセルでは、エクスポートされたモデルをローカル予測に使用する方法と、Vertex AI Prediction を使ってモデルをマイクロサービスとしてデプロイする方法が説明されています。いずれのケースでも、入力（サンプル）データポイントが未加工のスキーマであることを強調しておくことが重要です。

## クリーンアップ

このチュートリアルで使用したリソースに対し、Google Cloud アカウントに追加料金が発生しないように、リソースを含むプロジェクトを削除します。

### プロジェクトを削除する

  <aside class="caution"><strong>警告</strong>: プロジェクトを削除すると、以下のことが発生します。 <ul> <li> <strong>プロジェクト内のすべてが削除されます。</strong> このチュートリアルで既存のプロジェクトを使用した場合、それを削除すると、プロジェクトで行った他の作業もすべて削除されます。 </li> <li> <strong>カスタムプロジェクト ID が失われます。</strong> このプロジェクトを作成した時に、将来的に使用しようと考えているカスタムプロジェクト ID を作成した可能性があります。そのプロジェクト ID を使用する <code translate="no" dir="ltr">appspot.com</code> などの URL を保持するには、プロジェクト全体ではなく、プロジェクト内の該当リソースを削除してください。 </li> </ul> <p> 複数のチュートリアルとクイックスタートを使用する予定がある場合は、プロジェクトを再利用することで、プロジェクトのクォータ制限を超えてしまうことを回避できます。 </p></aside>


1. Google Cloud コンソールで、**Manage resources** ページに移動します。

    [Manage resources に移動](https://console.cloud.google.com/iam-admin/projects){: class="button button-primary" target="console" track-type="solution" track-name="consoleLink" track-metadata-position="body" }

2. プロジェクトのリストで、削除するプロジェクトを選択し、**Delete** をクリックします。

3. ダイアログにプロジェクト ID を入力し、**Shut down** をクリックしてプロジェクトを削除します。

## 次のステップ

- Google Cloud で機械学習向けにデータを前処理する際の概念、課題、およびオプションについては、このシリーズのパート 1「[ML 向けのデータ前処理: オプションと推奨事項](../guide/tft_bestpractices)」をご覧ください。
- Dataflow での tf.Transform パイプラインの実装、パッケージ化、および実行方法については、[国税調査データセットによる収入予測](https://github.com/GoogleCloudPlatform/cloudml-samples/tree/master/census/tftransformestimator){: .external }の例をご覧ください。
- [TensorFlow on Google Cloud](https://www.coursera.org/specializations/machine-learning-tensorflow-gcp){: .external } で ML に関する Coursera の専門講座を受講してください。
- [Rules of ML](https://developers.google.com/machine-learning/guides/rules-of-ml/){: .external } で ML エンジニアリングのベストプラクティスをご覧ください。
- その他のリファレンスアーキテクチャ、ダイアグラム、およびベストプラクティスについては、[Cloud Architecture Center](https://cloud.google.com/architecture) をご覧ください。
