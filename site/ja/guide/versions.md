# TensorFlow バージョンの互換性

このドキュメントは、異なる TensorFlow バージョン間で（コードまたはデータのいずれかに対する）下位互換性を必要としているユーザー、および互換性を維持しながら TensorFlow を変更する必要がある開発者を対象としています。

## セマンティック バージョニング 2.0

TensorFlow は公開 API について、セマンティック バージョニング 2.0（[semver](http://semver.org)）を採用しています。TensorFlow の各リリースバージョンは、`メジャー.マイナー.パッチ` の形式になっています。たとえば、TensorFlow バージョン 1.2.3 の場合は `メジャー`バージョンは 1、`マイナー`バージョンは 2、`パッチ`バージョンは 3 です。各番号に対する変更には、以下のような意味があります。

- **メジャー**：互換性のない変更である可能性があります。以前のメジャーリリースで機能していたコードとデータは、必ずしも新しいリリースで機能するとは限りません。ただし、既存の TensorFlow グラフとチェックポイントは新しいリリースに移行できる場合があります。データの互換性に関する詳細は、[グラフとチェックポイントの互換性](#compatibility_of_graphs_and_checkpoints)を参照してください。

- **マイナー**：下位互換性のある機能、速度の改善などです。以前のマイナーリリースで機能し、*かつ*実験的ではない公開 API のみを利用していたコードやデータは、変更なしで引き続き機能します。公開 API と非公開 API の詳細については、[互換対象](#what_is_covered)を参照してください。

- **パッチ**：下位互換性のあるバグ修正です。

たとえば、リリース 1.0.0 ではリリース 0.12.1 とは下位*互換性のない*変更が取り込まれました。しかし、リリース 1.1.1 はリリース 1.0.0 と下位*互換性がありました*。<a name="what_is_covered"></a>

## 互換対象

TensorFlow の公開 API のみが、マイナーバージョンとパッチバージョン間で下位互換性があります。公開 API には以下が含まれます。

- `tensorflow` モジュールとそのサブモジュールでドキュメント化されているすべての [Python](../api_docs/python) 関数とクラス。ただし、以下を除きます。

    - 非公開シンボル：名前が `_` で始まる任意の関数、クラスなど。
    - 実験的なシンボルと `tf.contrib` シンボル。詳細は、[以下](#not_covered)を参照してください。

    `examples/` および `tools/` ディレクトリ内のコードは `tensorflow` Python モジュール経由では到達不可能であり、互換性の保証対象からは外れていることに注意してください。

    シンボルが `tensorflow` Python モジュールかそのサブモジュールから利用できるものの、ドキュメント化されていない場合は公開 API の一部とは**見なされません**。

- 互換 API（Pythonでは `tf.compat` モジュール）。メジャーバージョンでは、新しいメジャーバージョンへの移行を支援するユーティリティと追加のエンドポイントがリリースされる場合があります。これらの API シンボルは非推奨であり、サポートされません（つまり、機能を追加せず、脆弱性を修正する以外のバグを修正しません）。ただし、互換性の保証対象にはなります。

- [C API](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/c/c_api.h)。

- 以下のプロトコル バッファ ファイル。

    - [`attr_value`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/attr_value.proto)
    - [`config`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/protobuf/config.proto)
    - [`event`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/util/event.proto)
    - [`graph`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/graph.proto)
    - [`op_def`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/op_def.proto)
    - [`reader_base`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/reader_base.proto)
    - [`summary`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto)
    - [`tensor`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/tensor.proto)
    - [`tensor_shape`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/tensor_shape.proto)
    - [`types`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/types.proto)

<a name="not_covered"></a>

## 互換*対象外*

TensorFlow の一部の構成要素は、いつでも下位互換性のない方法で変更できます。以下が含まれます。

- **実験的な API**：開発を容易にするため、実験的であると明確に指定されている一部の API シンボルは互換性の保証対象から除外されています。特に、以下はすべての互換性保証の対象から外されています。

    - `tf.contrib` モジュールまたはそのサブモジュール内のシンボル。
    - 名前に `experimental` または `Experimental` を含むシンボル（モジュール、関数、引数、プロパティ、クラス、または定数）。
    - 完全修飾名に、それ自体が実験的なモジュールまたはクラスを含むシンボル。`experimental` と呼ばれる任意のプロトコルバッファのフィールドとサブメッセージが含まれます。

- **その他の言語**：以下のような、Python や C 以外の言語で書かれた TensorFlow API。

    - [C++](../install/lang_c.md) （[`tensorflow/cc`](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/cc) 内のヘッダーファイル経由で公開されているもの）。
    - [Java](../install/lang_java.md)
    - [Go](../install/lang_go.md)
    - [JavaScript](https://www.tensorflow.org/js)

- **複合ops（オプス）の詳細：** Python 内の多くのパブリック関数はグラフ内で複数のプリミティブops（オプス）に展開され、これらの詳細はディスクに `GraphDef` として保存されているグラフの要素になります。特に、グラフ間の正確な一致をチェックする回帰テストは、グラフの動作を変更する必要がなく、既存のチェックポイントが引き続き機能する場合であっても、マイナーリリース間で中断される可能性があります。

- **浮動小数点数値の詳細：** ops（オプス）によって計算された特定の浮動小数点値は、常に変更される可能性があります。ユーザーは計算された特定のビットではなく、おおよその精度と数値的安定性のみを頼りにする必要があります。マイナーリリースおよびパッチリリースで数式を変更すると、同等またはより正確な精度が得られるはずです。ただし、機械学習では特定の数式の精度を向上させると、システム全体の精度が低下する場合があります。

- **乱数：** 計算された特定の乱数は常に変更される可能性があります。ユーザーは計算された特定のビットではなく、おおよその正しい分布と統計的強度のみを頼りにする必要があります。詳細については、[乱数生成](random_numbers.ipynb)ガイドを参照してください。

- **分散化された Tensorflow でのバージョンの差異：** 単一のクラスターでバージョンの異なる 2 つの TensorFlow を実行することはできません。ワイヤープロトコルの下位互換性については保証されていません。

- **バグ：** 現在の実装に明らかな不具合がある場合（すなわち、ドキュメントと矛盾している場合、またはバグが原因で既知の明確に意図された動作が適切に実装されていない場合）、下位互換性のない動作（ただし、APIではない）を変更する場合があります。たとえば、オプティマイザが既知の最適化アルゴリズムを実装することを明言しているにもかかわらず、バグが原因でそのアルゴリズムに対応できていない場合は、オプティマイザを修正します。修正によっては、統一化のために間違った動作に依存しているコードが廃止される可能性があります。このような変更については、リリースノートで説明します。

- **未使用の API：**（GitHub 検索で TensorFlow の使用箇所を調査することにより）使用されている記録が見つからない API に下位互換性のない変更を行う場合があります。このような変更を行う前には、[発表用のメーリングリスト](https://groups.google.com/a/tensorflow.org/forum/#!forum/announce)で変更を行う意向を発表し、（適切な場合に）廃止への対処方法を説明し、コミュニティにフィードバックを提供する機会を与えるために 2 週間待ちます。

- **エラー動作：**エラーを非エラー動作に置き換える場合があります。たとえば、エラーが記録されている場合でも、エラーを発生させる代わりに結果を計算する関数が変更される場合があります。また、エラーメッセージの本文を変更する場合があります。さらに、特定のエラー条件に対する例外タイプがドキュメントで指定されていない限り、エラーのタイプが変更される場合があります。

<a name="compatibility_of_graphs_and_checkpoints"></a>

## SavedModel、グラフ、チェックポイントの互換性

SavedModel は、TensorFlow プログラムでの使用に推奨されるシリアライズ形式です。SavedModel には、`GraphDefs` としてエンコードされる 1 つ以上のグラフとチェックポイントの 2 つの要素が含まれています。グラフは実行されるops（オプス）のデータフローを表し、チェックポイントにはグラフ内の変数の保存されたテンソル値が含まれています。

多くの TensorFlow ユーザーは SavedModel を作成し、TensorFlow の新しいリリースでそれらを読み込んで実行します。[semver](https://semver.org) に準拠し、あるバージョンの TensorFlow で記述された SavedModel は、同じメジャーリリースを持つ TensorFlow の新しいバージョンで読み込み、評価できます。

*サポート対象*の SavedModel については、追加の保証を行っています。TensorFlow メジャーバージョン `N` で**非推奨ではなく、実験的ではなく、互換性のない API のみ**を使用して作成された SavedModel を、<em data-md-type="raw_html">バージョン `N` のサポート対象 SavedModel</em> と呼びます。TensorFlow メジャーバージョン `N` のサポート対象となるすべての SavedModel は TensorFlow メジャーバージョン `N+1` で読み込み、実行できます。ただし、このようなモデルの構築や変更に必要な機能は利用できなくなっている場合があるため、この保証は未変更の SavedModel のみに適用されます。

シリアライズされたファイルを長期間使用できるよう、可能な限り下位互換性を維持するよう努めます。

### GraphDef の互換性

グラフは `GraphDef` プロトコル バッファ経由でシリアライズされます。グラフへの下位互換性のない変更を容易にするため、各 `GraphDef` には TensorFlow のバージョンとは別のバージョン番号が割り振られています。たとえば、`GraphDef` バージョン 17 では、`reciprocal` に変わり、`inv` 演算が非推奨になりました。セマンティックは以下のとおりです。

- TensorFlow の各バージョンは、`GraphDef` のバージョン間隔をサポートします。この間隔はパッチリリース全体で一定であり、マイナーリリース全体でのみ増加します。`GraphDef` バージョンのサポートは、TensorFlow のメジャーリリース時のみ廃止されます（また、SavedModel で保証されているバージョンのサポートのみに対応しています）。

- 新しく作成されたグラフには、最新の `GraphDef` バージョン番号が割り当てられます。

- ある TensorFlow のバージョンが `GraphDef` バージョンのグラフをサポートしている場合、それを生成するのに使用された TensorFlow バージョンと同じ動作で読み込まれ、評価されます（上記で概説されている浮動小数点値の詳細と乱数は対象外）。この場合、TensorFlow のメジャーバージョンは問いません。特に、TensorFlow のあるバージョンのチェックポイントファイルと互換性のある GraphDef（SavedModel の場合など）は、GraphDef が<br>サポートされている限り、以降のバージョンのチェックポイントとの互換性が維持されます。

    これは、GraphDefs（および SavedModels）でシリアライズされたグラフのみに適用されます。チェックポイントを読み取る*コード*は、異なるバージョンの TensorFlow を実行する同じコードが生成したチェックポイントを読み取ることができない場合があります。

- `GraphDef` の*上限*がある（マイナー）リリースで X に増える場合、6 か月以上過ぎてから *下限*が X に増えます。以下に例を示します（ここでは仮想的なバージョン番号を使用しています）。

    - TensorFlow 1.2 は `GraphDef` バージョン 4 から 7 をサポートしていました。
    - TensorFlow 1.3 は `GraphDef` バージョン 8 を追加し、バージョン 4 から 8 をサポートしていました。
    - 6 カ月が過ぎた後、TensorFlow 2.0.0 ではバージョン 8 のみが残され、バージョン 4 から 7 のサポートが終了する可能性があります。

    TensorFlow のメジャーバージョンは通常 6 か月以上の間隔で公開されるため、上記のサポート対象の SavedModels に対する保証は、GraphDefs に対する 6 か月の保証よりもはるかに強力です。

さらに、ある `GraphDef` バージョンのサポートが終了した場合は、より新しいサポート対象の `GraphDef` バージョンにグラフを自動的に変換するツールの提供を試みます。

## TensorFlow を拡張する際のグラフとチェックポイントの互換性

このセクションは、ops（オプス）の追加、ops（オプス）の削除、または既存ops（オプス）の機能変更など、`GraphDef` 形式に互換性のない変更を行う場合のみ関係します。ほとんどのユーザーは、これまでのセクションの内容で十分に対応できます。

<a id="backward_forward"></a>

### 下位互換性および部分的な上位互換性

当社のバージョン管理スキームには 3 つの要件があります。

- **下位互換性**。旧バージョンの TensorFlow で作成されたグラフとチェックポイントの読み込みをサポートするものです。
- **上位互換性**。グラフまたはチェックポイントのプロデューサーがコンシューマーより先に新バージョンの TensorFlow にアップグレードされるシナリオをサポートするものです。
- 互換性のない方法で TensorFlow の進化を可能にすること。たとえば、ops（オプス）の削除、属性の追加、属性の削除などがあります。

`GraphDef` のバージョン構造は TensorFlow のバージョンとは独立しており、`GraphDef` の形式に対する下位互換性のない変更は依然としてセマンティック バージョニングによる制限を受けることに注意してください。つまり、機能を削除または変更できるのは TensorFlow の`メジャー`バージョン間のみです（`1.7` から `2.0` など）。また、パッチ リリース内では上位互換性が強制されます（`1.x.1` から `1.x.2` など）。

下位互換性と上位互換性を実現し、形式の変更を強制するタイミングを把握するため、グラフとチェックポイントには生成された時間を表すメタデータがあります。以下のセクションでは、TensorFlow の実装と `GraphDef` のバージョンを進化させるためのガイドラインについて詳述します。

### 独立したデータ バージョン スキーム

グラフとチェックポイントにはさまざまなデータバージョンがあります。2 つのデータ形式は、互いに異なる速度で、TensorFlow とも異なる速度で進化します。どちらのバージョン管理システムも [`core/public/version.h`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/public/version.h) で定義されています。新しいバージョンが追加されると、変更内容と日付を詳述した注釈がヘッダーに追加されます。

### データ、プロデューサー、コンシューマー

次の種類のデータバージョン情報を区別します。

- **プロデューサー**：データを生成するバイナリです。プロデューサーには、バージョン（`producer`）と互換性のある最小コンシューマー バージョン（`min_consumer`）があります。
- **コンシューマー**：データを消費するバイナリです。コンシューマーには、バージョン（`consumer`）と互換性のある最小プロデューサー バージョン（`min_producer`）があります。

バージョン管理対象の各データには、データを作成した`producer`、互換性のある `min_consumer`、および許可されていない `bad_consumers` バージョンのリストを記録する [`VersionDef バージョン`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/versions.proto)フィールドがあります。

デフォルトではプロデューサーが何らかのデータを作成した場合、そのデータはプロデューサーの `producer` バージョンと `min_consumer` バージョンを継承します。`bad_consumers` は、特定のコンシューマー バージョンがバグを含んでおり、回避すべきであることが分かっている場合に設定できます。コンシューマーは、次のすべての条件を満たす場合にデータ片を受け付けます。

- `consumer` &gt;= データの `min_consumer`
- データの `producer` &gt;= コンシューマーの `min_producer`
- `consumer` がデータの `bad_consumers` に含まれない

プロデューサーとコンシューマーはどちらも同じ TensorFlow コードベースに由来するため、[`core/public/version.h`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/public/version.h) には、コンテキストに応じて `producer` または `consumer` として扱われるメインデータのバージョンと、`min_consumer` および `min_producer` の両方（それぞれプロデューサーとコンシューマーで必要）が含まれます。具体的には以下のとおりです。

- `GraphDef` バージョンの場合は、`TF_GRAPH_DEF_VERSION`、`TF_GRAPH_DEF_VERSION_MIN_CONSUMER`、`TF_GRAPH_DEF_VERSION_MIN_PRODUCER` が含まれます。
- チェックポイントバージョンの場合は、`TF_CHECKPOINT_VERSION`、`TF_CHECKPOINT_VERSION_MIN_CONSUMER`、`TF_CHECKPOINT_VERSION_MIN_PRODUCER` が含まれます。

### デフォルトの新しい属性を既存の演算に追加する

以下のガイダンスに従うと、一連のops（オプス）が変更されていない場合に限り、上位互換性が確保されます。

1. 上位互換性が必要な場合は、`SavedModelBuilder` クラスの `tf.saved_model.SavedModelBuilder.add_meta_graph_and_variables` メソッドと `tf.saved_model.SavedModelBuilder.add_meta_graph` メソッドのいずれか、または `tf.estimator.Estimator.export_saved_model` を使用してモデルをエクスポートする際に、`strip_default_attrs` を `True` に設定します。
2. これにより、モデルの生成/エクスポート時にデフォルト値の属性が取り除かれます。その結果、デフォルト値が使用されている場合にエクスポートされた `tf.MetaGraphDef` に新しい演算属性が含まれなくなります。
3. このコントロールにより、古いコンシューマー（トレーニングバイナリより遅れるバイナリを提供している等）がモデルの読み込みを続行し、モデル提供の中断を防げるようになる場合があります。

### GraphDef バージョンの進化

このセクションでは、このバージョン管理手法を使用して `GraphDef` 形式にさまざまな種類の変更を加える方法について説明します。

#### 演算を追加する

新しい演算をコンシューマーとプロデューサーの両方に同時に追加し、 `GraphDef` のバージョンを変更しないようにしてください。既存のプロデューサースクリプトが新しい機能を不意に使用することはないため、この種の変更には自動的に下位互換性が確保され、上位互換性の計画には影響しません。

#### 演算を追加し、既存の Python ラッパーを切り替えて使用する

1. 新しいコンシューマーの機能を実装し、`GraphDef` のバージョンを上げます。
2. 以前は機能しなかったラッパーが新機能のみを使用できるようにすることが可能な場合、ラッパーをすぐに更新できます。
3. Python ラッパーを変更して新機能を使用するようにします。`min_consumer` の値は増やさないでください。この演算を使用しないモデルは壊れないためです。

#### 演算の機能を削除または制限する

1. すべてのプロデューサースクリプト（TensorFlow自体ではない）を修正し、禁止された演算や機能を使用しないようにします。
2. `GraphDef` のバージョンを上げ、新バージョン以降の GraphDefs で削除された演算や機能を禁止する新しいコンシューマー機能を実装します。可能であれば、TensorFlow が禁止された機能を使用して `GraphDefs` を生成するのを阻止してください。そのためには、[`REGISTER_OP(...).Deprecated(deprecated_at_version, message)`](https://github.com/tensorflow/tensorflow/blob/b289bc7a50fc0254970c60aaeba01c33de61a728/tensorflow/core/ops/array_ops.cc#L1009) を追加します。
3. 下位互換性を確保するためにメジャーリリースを待ちます。
4. `min_producer`を (2) の GraphDef バージョンに増やし、機能を完全に削除します。

#### 演算の機能を変更する

1. `SomethingV2` などの名前を持つ類似した新しい演算を追加し、それを追加して既存の Python ラッパーがそれを使用するように切り替えるプロセスを実行します。上位互換性を確保するには、Python ラッパーを変更する際に [compat.py](https://www.tensorflow.org/code/tensorflow/python/compat/compat.py) で提案されているチェックを使用します。
2. 古い演算を削除します（下位互換性のためにメジャーバージョンが変更された場合にのみ実行できます）。
3. `min_consumer` を増やし、古い演算を持つコンシューマーを除外し、古い演算を `SomethingV2` のエイリアスとして再度追加し、既存の Python ラッパーがそれを使用するようにプロセスを切り替えます。
4. `SomethingV2` を削除するプロセスを実行します。

#### 単一の安全でないコンシューマーバージョンを禁止する

1. `GraphDef` のバージョン番号を上げ、すべての新しい GraphDefs に対して不具合のあるバージョンを `bad_consumers` に追加します。可能であれば、特定の演算やそれに類するものを含む GraphDefs のみに `bad_consumers` を追加してください。
2. 既存のコンシューマーに不具合のあるバージョンがある場合、できるだけ早くそれらを排除してください。
