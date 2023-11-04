# 連合プログラム

このドキュメントは、連合プログラムの概念の概要に関心のある方を対象としています。TensorFlow Federated の知識、特にその型システムの知識を前提としています。

連合プログラムに関するその他の情報は、以下をご覧ください。

- [API ドキュメント](https://www.tensorflow.org/federated/api_docs/python/tff/program)
- [例](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/program)
- [開発者ガイド](https://github.com/tensorflow/federated/blob/main/docs/program/guide.md)

[TOC]

## 連合プログラムとは？

**連合プログラム**とは、連合環境で計算やその他の処理ロジックを実行するプログラムを指します。

より具体的には、**連合プログラム**は以下のように動作します。

- [計算](#computations)を実行します。
- これには[プログラムロジック](#program-logic)を使用します。
- [プラットフォーム固有のコンポーネント](#platform-specific-components)
- および[プラットフォームに依存しないコンポーネント](#platform-agnostic-components)が含まれます。
- [プログラム](#program)が設定する特定の[パラメータ](#parameters)
- および[顧客](#customer)が設定する[パラメータ](#parameters)があります。
- このパラメータは[顧客](#customer)が[プログラム](#program)を実行したときに設定されます。
- [プラットフォームストレージ](#platform storage)では以下の目的でデータを[マテリアライズ](#materialize)できます。
    - Python ロジックで使用する
    - [フォールトトレランス](#fault tolerance)を実装する
- また、データを[顧客ストレージ](#customer storage)に[リリース](#release)することもできます。

これらの[概念](#concepts)と抽象を定義することで、連合プログラムの[コンポーネント](#components)の関係を説明し、様々な[ロール](#roles)がこれらのコンポーネントを所有して記述することが可能になります。このように切り離すことで、開発者は他の連合プログラムと共有されているコンポーネントを使用して、連合プログラムを作成することができるため、通常、多数の様々なプラットフォームで同じプログラムを実行することが可能になります。

TFF の連合プログラムライブラリ（[tff.program](https://www.tensorflow.org/federated/api_docs/python/tff/program)）は、連合プログラムの作成に必要な抽象を定義し、[プラットフォームに依存しないコンポーネント](#platform-agnostic-components)を提供します。

## コンポーネント

TFF の連合プログラムライブラリの**コンポーネント**は、異なる[ロール](#roles)が所有し、記述できるように設計されています。

注意: これはコンポーネントの高レベルの概要です。特定の API のドキュメントについては、[tff.program](https://www.tensorflow.org/federated/api_docs/python/tff/program) をご覧ください。

### プログラム

**プログラム**は、以下を行う Python バイナリです。

1. [パラメータ](#parameters)を定義します（フラグなど）。
2. [プラットフォーム固有のコンポーネント](#platform-specific-components)と[プラットフォームに依存しないコンポーネント](#platform-agnostic-components)を構築します。
3. 連合のコンテキストで[プログラムロジック](#program_logic)を使って[計算](#computations)を実行します。

以下に例を示します。

```python
# Parameters set by the customer.
flags.DEFINE_string('output_dir', None, 'The output path.')

def main() -> None:

  # Parameters set by the program.
  total_rounds = 10
  num_clients = 3

  # Construct the platform-specific components.
  context = tff.program.NativeFederatedContext(...)
  data_source = tff.program.DatasetDataSource(...)

  # Construct the platform-agnostic components.
  summary_dir = os.path.join(FLAGS.output_dir, 'summary')
  metrics_manager = tff.program.GroupingReleaseManager([
      tff.program.LoggingReleaseManager(),
      tff.program.TensorBoardReleaseManager(summary_dir),
  ])
  program_state_dir = os.path.join(..., 'program_state')
  program_state_manager = tff.program.FileProgramStateManager(program_state_dir)

  # Define the computations.
  initialize = ...
  train = ...

  # Execute the computations using program logic.
  tff.framework.set_default_context(context)
  asyncio.run(
      train_federated_model(
          initialize=initialize,
          train=train,
          data_source=data_source,
          total_rounds=total_rounds,
          num_clients=num_clients,
          metrics_manager=metrics_manager,
          program_state_manager=program_state_manager,
      )
  )
```

### パラメータ

**パラメータ**は[プログラム](#program)への入力です。これらの入力は、フラグとして公開される場合は[顧客](#customer)が設定しますが、プログラムによって設定される場合もあります。上記の例では、`output_dir` が[顧客](#customer)によって設定されるパラメータで、`total_rounds` と `num_clients` がプログラムによって設定されるパラメータです。

### プラットフォーム固有のコンポーネント

**プラットフォーム固有のコンポーネント**とは、TFF の連合プログラムライブラリで定義されている抽象インターフェースを実装する[プラットフォーム](#platform)が提供するコンポーネントを指します。

### プラットフォームに依存しないコンポーネント

**プラットフォームに依存しないコンポーネント**とは、TFF の連合プログラムライブラリで定義されている抽象インターフェースを実装する[ライブラリ](#library)（TFF など）が提供するコンポーネントを指します。

### 計算

**計算**は、抽象インターフェース [`tff.Computation`](https://www.tensorflow.org/federated/api_docs/python/tff/Computation) の実装です。

たとえば、TFF プラットフォームでは、[`tff.tf_computation`](https://www.tensorflow.org/federated/api_docs/python/tff/tf_computation) や [`tff.federated_computation`](https://www.tensorflow.org/federated/api_docs/python/tff/federated_computation) デコレータを使用して、[`tff.framework.ConcreteComputation`](https://www.tensorflow.org/federated/api_docs/python/tff/framework/ConcreteComputation) を作成できます。

詳細は、[計算のライフサイクル](https://github.com/tensorflow/federated/blob/main/docs/design/life_of_a_computation.md)をご覧ください。

### プログラムロジック

**プログラムロジック**は、以下を入力として取る Python 関数です。

- [顧客](#customer)と[プログラム](#program)によって設定される[パラメータ](#parameters)
- [プラットフォーム固有のコンポーネント](#platform-specific-components)
- [プラットフォームに依存しないコンポーネント](#platform-agnostic-components)
- [計算](#computations)

そして、何らかの演算を実行します。これには、通常以下が含まれます。

- [計算](#computations)の実行
- Python ロジックの実行
- 以下の目的による[プラットフォームストレージ](#platform storage)でのデータの[マテリアライズ](#materialize):
    - Python ロジックで使用する
    - [フォールトトレランス](#fault tolerance)を実装する

また、何らかの出力を生成します。これには、通常以下が含まれます。

- データを[顧客ストレージ](#customer storage)に[指標](#metrics)として[リリース](#release)する

以下に例を示します。

```python
async def program_logic(
    initialize: tff.Computation,
    train: tff.Computation,
    data_source: tff.program.FederatedDataSource,
    total_rounds: int,
    num_clients: int,
    metrics_manager: tff.program.ReleaseManager[
        tff.program.ReleasableStructure, int
    ],
) -> None:
  state = initialize()
  start_round = 1

  data_iterator = data_source.iterator()
  for round_number in range(1, total_rounds + 1):
    train_data = data_iterator.select(num_clients)
    state, metrics = train(state, train_data)

    _, metrics_type = train.type_signature.result
    metrics_manager.release(metrics, metrics_type, round_number)
```

## ロール

連合プログラムに関しては、定義しておくと有用な**ロール**が 3 つあります。[顧客](#customer)、[プラットフォーム](#platform)、[ライブラリ](#library)です。各ロールは、連合プログラムの作成に使用される一部の[コンポーネント](#components)を所有し、記述しますが、単一のエンティティまたはグループが複数のロールを実行することも可能です。

### 顧客

**顧客** は通常以下を実行できます。

- [顧客ストレージ](#customer-storage)の所有
- [プログラム](#program)の起動

ただし、以下を行う場合もあります。

- [プログラム](#program)の記述
- [プラットフォーム](#platform)のすべての機能

### プラットフォーム

**プラットフォーム** は通常以下を実行できます。

- [プラットフォームストレージ](#platform-storage)の所有
- [プラットフォーム固有のコンポーネント](#platform-specific-components)の記述

ただし、以下を行う場合もあります。

- [プログラム](#program)の記述
- [ライブラリ](#library)のすべての機能

### ライブラリ

**ライブラリ**は通常以下を実行できます。

- [プラットフォームに依存しないコンポーネント](#platform-agnostic-components)の記述
- [計算](#computations)の記述
- [プログラムロジック](#program-logic)の記述

## 概念

連合プログラムに関しては、定義しておくと有用な**概念**がいくつかあります。

### 顧客ストレージ

**顧客ストレージ**は、[顧客](#customer)が読み書きアクセスを持ち、[プラットフォーム](#platform)が書き込みアクセスを持つストレージです。

### プラットフォームストレージ

**プラットフォームストレージ**は、[プラットフォーム](#platform)のみが読み書きアクセスを持つストレージです。

### リリース

値を**リリース**すると、[顧客ストレージ](#customer-storage)がその値を使用できるようになります（ダッシュボードに当たりを公開する、値をログに記録する、値をディスクに書き込むなど）。

### マテリアライズ

値参照を**マテリアライズ**すると、[プログラム](#program)が参照された値を使用できるようになります。通常、値参照のマテリアライズには、値を[リリース](#release)するか、[プログラムロジック](#program-logic)を[フォールトトレランス](#fault-tolerance)にする必要があります。

### フォールトトレランス

**フォールトトレランス**は、計算を実行する際にエラーから回復するための[プログラムロジック](#program-logic)です。たとえば、100 ラウンド中、最初の 90 ラウンドでトレーニングに成功した後でエラーが発生した場合、プログラムロジックは 91 ラウンドからトレーニング再開できますか？それとも、1 ラウンドからやり直す必要がありますか？
