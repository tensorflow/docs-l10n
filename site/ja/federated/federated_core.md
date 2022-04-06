# フェデレーテッドコア

このドキュメントでは、[フェデレーテッドラーニング](federated_learning.md)の基盤として機能する TFF のコアレイヤーと、可能性のある将来の非学習型フェデレーテッドアルゴリズムを説明します。

フェデレーテッドコアの簡単な説明について、以下のチュートリアルをお読みください。このチュートリアルでは、例を使っていくらかの基本概念を紹介し、単純なフェデレーテッドアベレージングあっるごリズムの構造を、手順を追って実演しています。

- [カスタムフェデレーテッドアルゴリズム、パート 1: フェデレーテッドコアの基礎](tutorials/custom_federated_algorithms_1.ipynb)

- [カスタムフェデレーテッドアルゴリズム、パート 2: フェデレーテッドアベレージングの実装](tutorials/custom_federated_algorithms_2.ipynb)

また、[フェデレーテッドラーニング](federated_learning.md)と、[画像分類](tutorials/federated_learning_for_image_classification.ipynb)および[テキスト生成](tutorials/federated_learning_for_text_generation.ipynb)に関する関連チュートリアルにも慣れておくことをお勧めします。フェデレーテッドラーニングに Federated Core API（FC API）を使用すると、このレイヤーの設計で行ういくつかの選択に関する重要なコンテキストを得ることができます。

## 概要

### 目標、用途、およびスコープ

フェデレーテッドコア（FC）は、分散計算、つまり、それぞれがローカルで重要な処理を行い、作業のやり取りをネットワークで行う複数のコンピュータ（携帯電話、タブレット、組み込みデバイス、デスクトップコンピュータ、センサー、データベースサーバーなど）を使用する計算を実装するためのプログラミング環境として最もよく理解されています。

「*分散*」という言葉は非常に一般的で、TFF は、存在するあらゆる分散アルゴリズムをターゲットしてはいないため、一般性に劣る「*フェデレーテッドコンピュテーション*」という言葉で、子のフレームワークで表現できるアルゴリズムの種類を説明しています。

全く正式に*フェデレーテッドコンピュテーション*という言葉を定義するのは、このドキュメントの趣旨から外れてしまいますが、新しい分散型学習アルゴリズムを説明する[研究発表](https://arxiv.org/pdf/1602.05629.pdf)で、疑似コードで表現されたアルゴリズムの種類と考えるとよいでしょう。

FC の目標は、要約すると、疑似コード*ではなく*、多様なターゲット環境で実行可能なプログラムロジックの同様にコンパクトな表現を、同様の疑似コードのようなレベルの抽象化で実現することです。

FC が表現するように設計されているアルゴリズムの種類の主な決定的な特性は、システムの要素のアクションが集合的に記述されていることです。したがって、ローカルでデータを変換する*各デバイス*おと、その結果を*ブロードキャスト*、*収集*、または*集計*する中央コーディネータによって調整するデバイスについて言及する傾向にあります。

TFF は、単純な*クライアントサーバー*アーキテクチャを超えられるように設計されてはいますが、集合処理の概念を基本としています。これは、フェデレーテッドラーニングという、クライアントデバイスの管理下のままとなり、プライバシーの理由で中央ロケーションに簡単にはダウンロードされない潜在的に機密なデータでの計算をサポートするようにもともと設計された技術が TFF の起源であるためです。このようなシステムの各クライアントは、システムによってデータと処理能力を結果の計算に使用しますが（一般的に、すべての構成要素の値として期待する結果）、各クライアントのプライバシーと匿名性の保護にも努めています。

したがって、分散計算向けのほとんどのフレームワークは個々の構成要素の観点、つまりポイントツーポイントのメッセージ交換のレベルで処理を表現するように設計されており、構成要素のローカルの状態の相互依存は受信メッセージと送信メッセージによって変化しますが、TFF<br>のフェデレーテッドコアは、*グローバル*システム全体の観点（[MapReduce](https://research.google/pubs/pub62/) などに類似）でシステムの動作を説明するように設計されています。

結果として、汎用の分散フレームワークでは、*send* や *receive* といった演算をビルディングブロックとして提供することがありますが、FC は、単純な分散型プロトコルをカプセル化する `tff.federated_sum`、`tff.federated_reduce`、または `tff.federated_broadcast` などのビルディングブロックを提供しています。

## 言語

### Python インターフェース

TFF uses an internal language to represent federated computations, the syntax of which is defined by the serializable representation in [computation.proto](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/proto/v0/computation.proto). Users of FC API generally won't need to interact with this language directly, though. Rather, we provide a Python API (the `tff` namespace) that wraps arounds it as a way to define computations.

具体的には、TFF はデコレートされた関数の本文をトレースして TFF の言語でシリアル化表現を生成する `tff.federated_computation` といった Python 関数デコレータを提供しています。`tff.federated_computation` でデコレートされた関数はそういったシリアル化表現のキャリアとして機能し、別の計算の本文にビルディングブロックとして組み込み、呼び出し時にオンデマンドで実行することができます。

次は、一例です。その他の例は[カスタムアルゴリズム](tutorials/custom_federated_algorithms_1.ipynb)チュートリアルをご覧ください。

```python
@tff.federated_computation(tff.type_at_clients(tf.float32))
def get_average_temperature(sensor_readings):
  return tff.federated_mean(sensor_readings)
```

非 Eager の TensorFlow に慣れているユーザーは、このアプローチが TensorFlow グラフを定義する Python コードのセクションで `tf.add` または `tf.reduce_sum` などの関数路使用する Python コードの書き方に類似していることに気づくでしょう。コードが技術的に Python で表現されているとはいえ、その目的は、TensorFlow ランタイムが内部的に実行できる、Python コードではなく、グラフである、根底の `tf.Graph` のシリアル化可能表現を構築することにあります。同様に、 *フェデレーテッド演算*を `get_average_temperature` が表現するフェデレーテッドコンピュテーションに挿入するとして、`tff.federated_mean` を捉えることができます。

FC が言語を定義する理由の一部は、上述のように、フェデレーテッドコンピュテーションが分散化された集合的な動作を指定するため、そのロジックがローカルではないという事実に関係しています。 たとえば、TFF はネットワーク内のさまざまな場所に存在する可能性のある演算子、入力、および出力を提供します。

これには、分散の概念を捉えた言語と型システムが必要です。

### 型システム

フェデレーテッドコアには、次の型カテゴリがあります。これらの型を説明するために、型コンストラクタを示し、コンパクトな表記を紹介します。これは、計算と演算子の型をわかりやすく説明しています。

まず、既存の主要言語に見られる型カテゴリに類似するカテゴリから説明します。

- **テンソル型**（`tff.TensorType`）。TensorFlow と同様に、`dtype` と `shape` があります。唯一の違いは、この型のオブジェクトは、TensorFlow 演算の出力を表す Python の `tf.Tensor` インスタンスに限られず、たとえば分散集約プロトコルの出力として生成されるデータのユニットを含むことがあるというところです。そのため、TFF テンソル型は単に、Python または TensorFlow のそのような型の具体的な物理表現の抽象バージョンです。

    TFF's `TensorTypes` can be stricter in their (static) treatment of shapes than TensorFlow. For example, TFF's typesystem treats a tensor with unknown rank as assignable *from* any other tensor of the same `dtype`, but not assignable *to* any tensor with fixed rank. This treatment prevents certain runtime failures (e.g., attempting to reshape a tensor of unknown rank into a shape with incorrect number of elements), at the cost of greater strictness in what computations TFF accepts as valid.

    テンソル型のコンパクト表記は、`dtype` または `dtype[shape]` です。たとえば、`int32` と `int32[10]` は、それぞれ整数と int ベクトルの型です。

- **シーケンス型**（`tff.SequenceType`）。TensorFlow の `tf.data.Dataset` という具象概念に相当する TFF の抽象型です。シーケンスの要素は順次消費され、複雑な型を含めることができます。

    シーケンス型のコンパクト表記は `T*` で、`T` は要素の型を指します。たとえば、`int32*` は、整数のシーケンスです。

- **名前付きタプル型**（`tff.StructType`）。名前がついているか否かにかかわらず、事前に定義された数の、具体的な型を持つ*要素*を持つ、タプルおよびディクショナリのような構造を構築する TFF の方法です。TFF の名前付きタプルの概念は、Python の引数タプルと同等の抽象型、つまり、すべてではなく一部が名前付きで、一部が定位置にある要素のコレクションを含む点が重要です。

    名前付きタプルのコンパクト表記は `<n_1=T_1, ..., n_k=T_k>` で、`n_k` はオプションの要素名、`T_k` は要素の型です。たとえば、`<int32,int32>` は名前付きでない整数ペアのコンパクト表記で、`<X=float32,Y=float32>` は、平面の点を表す名前付きの `X` と `Y` の浮動小数点数のコンパクト表記です。タプルはネストされるだけでなく、ほかの型と混在することができます。たとえば、`<X=float32,Y=float32>*` は、点のシーケンスのコンパクト表記です。

- **関数型**（`tff.FunctionType`）。TFF は関数型プログラミングフレームワークで、関数は[第一級の値](https://en.wikipedia.org/wiki/First-class_citizen)として扱われます。関数には最大 1 つの引数があり、ちょうど 1 つの結果を返します。

    関数型のコンパクト表記は `(T -> U)` で、`T` は引数の型、`U` は結果の型であるか、引数がない場合は `( -> U)` です（ただし、引数無し関数は、ほぼ Python レベルでのみ存在する縮退した概念です）。たとえば、`(int32* -> int32)` は、整数のシーケンスと単一の整数値に縮小する関数の種類の表記です。

次の型は、TFF 計算の分散型システム概念を解決します。これらの概念は TFF 固有のものである傾向にあるため、説明や例がさらに必要な場合は、[カスタムアルゴリズム](tutorials/custom_federated_algorithms_1.ipynb)チュートリアルを参照することをお勧めします。

- **配置型**。この型はパブリック API ではなく、この型の定数として捉えることのできる `tff.SERVER` と `tff.CLIENTS` という 2 つのリテラルの形態で公開されています。現在は内部的に使用されますが、将来のリリースでパブリック API に導入される予定です。この型のコンパクト表記は、`placement` です。

    *placement* は、特定の役割を果たすシステム構成要素の集合を表します。初期のリリースは、クライアントサーバーの計算をターゲットとしており、*クライアント*と*サーバー*の 2 つの構成要素グループがあります（サーバーはシングルトングループとして考えることができます）。ただし、より精巧なアーキテクチャでは、様々な種類の集計を実施するか、サーバーまたはクライアントのいずれかが使用する以外のデータ圧縮/解凍を使用する、マルチティアシステムの中間アグリゲーターなどの役割があります。

    placement の表記を定義するのは、主に、*フェデレーテッド型*を定義するための基盤とするのが目的です。

- **フェデレーテッド型**（`tff.FederatedType`）。フェデレーテッド型の値は、特定の placement（`tff.SERVER` または `tff.CLIENTS` など）によって定義されるシステム構成要素のグループがホストする値です。フェデレーテッド型は *placement* 値（したがって[依存型](https://en.wikipedia.org/wiki/Dependent_type)）、*構成メンバー*の型（各構成要素がローカルにどの種のコンテンツをホストしているか）、およびすべての構成要素が同じ項目をローカルにホストしているかを指定する追加のビット `all_equal` によって定義されています。

    型 `T` の項目（メンバー要素）を含み、それぞれがグループ（placement）`G` によってホストされている値のフェデレーテッド型のコンパクト表記は、`all_equal` ビットが設定されている `T@G` または設定されていない `{T}@G` です。

    次に例を示します。

    - `{ int32}@CLIENTS` は、クライアントデバイスごとに潜在的に異なる一連の整数値で構成される*フェデレーテッド型の値*を表します。ネットワークの複数の場所に現れるデータの複数の項目を含む単一の*フェデレーテッド型の値*について言及しているところに注意してください。これは、「ネットワーク」次元を持つある種のテンソルとして考えることもできます。ただし、TFF ではフェデレーテッド型の値のメンバー要素に[ランダムにアクセス](https://en.wikipedia.org/wiki/Random_access)することができないため、完全に類比できるわけではありません。

    - `{<X=float32,Y=float32>*}@CLIENTS` は、クライアントデバイス当たり 1 つのシーケンスとして、`XY` 座標の複数のシーケンスから成る、*フェデレーテッドデータセット*の値を表します。

    - `<weights=float32[10,5],bias=float32[5]>@SERVER` は、サーバーの重みとバイアステンソルの名前付きタプルを表します。波括弧を使用していないため、これは、<code>all_equal</code> ビットが設定されていることを示します。つまり、単一のタプルのみがあるということです（この値をホストしているクラスタ内に存在するサーバーレプリカの数に関係ありません）。

### ビルディングブロック

フェデレーテッドコアの言語は、[ラムダ計算](https://en.wikipedia.org/wiki/Lambda_calculus)に要素をいくつか追加した形態の言語です。

パブリック API で現在公開されている次のプログラミング抽象を提供しています。

- **TensorFlow** 計算（`tff.tf_computation`）。`tff.tf_computation` デコレータを使用して、TFF で再利用可能なコンポーネントとしてラッピングされている TensorFlow コードのセクションです。常に関数型があり、TensorFlow の関数とは異なって、構造化パラメータを取り、シーケンス型の構造結果を返すことができます。

    次に、`tf.data.Dataset.reduce` 演算子を使用して整数の和を計算する `(int32* -> int)` 型の TF 計算の一例を示します。

    ```python
    @tff.tf_computation(tff.SequenceType(tf.int32))
    def add_up_integers(x):
      return x.reduce(np.int32(0), lambda x, y: x + y)
    ```

- **組み込み関数**または*フェデレーテッド演算子*（`tff.federated_...`）。FC API のバルクを構成する `tff.federated_sum` や `tff.federated_broadcast` などの関数のライブラリです。このほとんどのバルクは、TFF と使用するための分散型通信演算子を表します。

    これらは、[組み込み関数](https://en.wikipedia.org/wiki/Intrinsic_function)とある程度同様に、TFF が理解し、より低レベルのコードにコンパイルされるオープンエンドの拡張可能な演算子セットであるため、*組み込み関数*と呼んでいます。

    これらのほとんどの演算子には、フェデレーテッド型のパラメータと結果があり、ほとんどが多様なデータに適用できるテンプレートです。

    たとえば、`tff.federated_broadcast` は、関数型 `T@SERVER -> T@CLIENTS` のテンプレート演算子として考えることができます。

- **ラムダ式**（`tff.federated_computation`）。TFF のラムダ式は、Python の `lambda` または `def` に相当します。パラメータ名、およびこのパラメータへの参照を含む本文（式）で構成されています。

    Python コードでは、Python 関数を `tff.federated_computation` でデコレートし、引数を定義することで作成されます。

    次は、前述のラムダ式の例です。

    ```python
    @tff.federated_computation(tff.type_at_clients(tf.float32))
    def get_average_temperature(sensor_readings):
      return tff.federated_mean(sensor_readings)
    ```

- **配置リテラル**。現時点では、`tff.SERVER` と `tff.CLIENTS` のみが、単純なクライアントサーバー計算を定義することができます。

- **関数呼び出し**（`__call__`）。関数型のあるものは、標準的な Python `__call__` 構文を使って呼び出すことができます。呼び出しは式であり、呼び出される関数の結果の型と同じ型です。

    次に例を示します。

    - `add_up_integers(x)` は、前述で引数 `x` に定義した TensorFlow 計算の呼び出しを表します。この式の型は `int32` です。

    - `tff.federated_mean(sensor_readings)` は、`sensor_readings` のフェデレーテッドアベレージング演算子の呼び出しを表します。この式の型は `float32@SERVER` です（上記の例のコンテキストを前提とした場合）。

- **タプル**を形成し、その要素を**選択**します。`tff.federated_computation` でデコレートされた関数の本文に現れるフォーム `[x, y]`、`x[y]`、または `x.y` の Paython 式です。
