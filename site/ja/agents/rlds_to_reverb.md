# TF-Agents の RLDS to Reverb util

[RLDS](https://github.com/google-research/rlds) to [Reverb](https://github.com/deepmind/reverb) util は、RLDS からエピソードを読み取り、それをトラジェクトリに変換して Reverb にプッシュする、TF Agents に含まれるツールです。

### RLDS データセット

RLDS（強化学習データセット）は、強化学習（RL）、デモからの学習、オフライン RL、または模倣学習などの順次意思決定の文脈で、エピソードデータを格納し、取得し、操作するためのツールのエコシステムです。

各ステップには以下のフィールドがあります（ステップメタデータの追加フィールドがある場合もあります）。例として、D4RL データセット [half-cheetah/v0-expert](https://www.tensorflow.org/datasets/catalog/d4rl_mujoco_halfcheetah#d4rl_mujoco_halfcheetahv0-expert_default_config) の仕様を使用することにします。

- **'action'**: `TensorSpec(shape = (6,), dtype = tf.float32, name = None)`

- **'discount'**: `TensorSpec(shape = (), dtype = tf.float32, name = None)`

- **'is_first'**: `TensorSpec(shape = (), dtype = tf.bool, name = None)`

- **'is_last'**: `TensorSpec(shape = (), dtype = tf.bool, name = None)`

- **'is_terminal'**: `TensorSpec(shape = (), dtype = tf.bool, name = None)`

- **'observation'**: `TensorSpec(shape = (17,), dtype = tf.float32, name = None)`

- **'reward'**: `TensorSpec(shape = (), dtype = tf.float32, name = None)}, TensorShape([]))`

## RLDS to TF-Agents util の API

### データセットからトラジェクトリ仕様を作成する

Reverb サーバーと Reverb Replay Buffer を初期化するためのデータ仕様を作成します。

```
def create_trajectory_data_spec(rlds_data: tf.data.Dataset) -> trajectory.Trajectory:
```

入力として提供される `rlds_data` を使用して作成できる、対応するトラジェクトリデータセットのデータ仕様を作成します。このデータ仕様は、Reverb サーバーと Reverb Replay Buffer の初期化に必要です。

**引数**:

- `rlds_data`: RLDS データセットは RLDS エピソードの `tf.data.Dataset` で、各エピソードにはRLDS ステップの `tf.data.Dataset` と、オプションとしてエピソードのメタデータが含まれます。RLDS ステップは、`is_first`、`is_last`、`observation`、`action`、`reward`、`is_terminal`、および `discount`（さらに場合によってはステップメタデータ）を含むテンソルのディクショナリです。

**戻り値**:

- 入力として指定される `rlds_data` を使ってトラジェクトリを作成するために使用できるトラジェクトリ仕様。

**エラー**:

- `ValueError`: `rlds_data` に RLDS ステップが存在しない場合に発生します。

### RLDS データを TF Agents トラジェクトリに変換する

RLDS データをトラジェクトリのデータセットに変換します。現在、2 ステップのトラジェクトリへの変換のみをサポートしています。

```
def convert_rlds_to_trajectories(rlds_data: tf.data.Dataset,
    policy_info_fn: _PolicyFnType = None) -> tf.data.Dataset:
```

指定された `rlds_data` をフラット化してバッチに変換し、隣接する RLDS ステップのオーバーラップしているペアのタプルに変換することで、TF Agents トラジェクトリのデータセットに変換します。

RLDS データは、最後のエピソードの最後のステップを使って作成されるトラジェクトリに有効な次のステップ型があることを保証するために、型 `first` のステップで最後をパディングされています。

**引数**:

- `rlds_data`: RLDS データセットは RLDS エピソードの `tf.data.Dataset` で、各エピソードには RLDS ステップの `tf.data.Dataset`（とオプションとしてエピソードのメタデータ）が含まれます。RLDS ステップは、`is_first`、`is_last`、`observation`、`action`、`reward`、`is_terminal`、および `discount`（さらにオプションのステップメタデータ）を含むテンソルのディクショナリです。
- `policy_info_fn`: TF-Agents トラジェクトリを生成する際に使用される policy.info を作成するオプションの関数。

**戻り値**:

- `tf.data.Dataset` 型のデータセット。この要素は、`rlds_data` に指定された RLDS ステップに対応する TF Agents トラジェクトリです。

**エラー**:

- `ValueError`: `rlds_data` に RLDS ステップが存在しない場合に発生します。

- `InvalidArgumentError`: 指定された RLDS データセットに以下のようなエピソードが含まれる場合に発生します。

    - 誤って終了している。つまり、最後のステップで終了していない。
    - 終端が誤っている。つまり、終端のステップが最後のステップではない。
    - 誤って開始している。つまり、最後のステップの後に最初のステップがない。最後のエピソードの最後のステップは関数によって処理されるため、ユーザーが、最後のエピソードの最後のステップの後に最初のステップがあることを確認する必要はありません。

### RLDS データを Reverb にプッシュする

RLDS データを TF Agents トラジェクトリとして Reverb サーバーにプッシュします。Reverb オブザーバーは、インターフェースを呼び出す前にインスタンス化されており、パラメータとして指定されている必要があります。

```
def push_rlds_to_reverb(rlds_data: tf.data.Dataset, reverb_observer: Union[
    reverb_utils.ReverbAddEpisodeObserver,
    reverb_utils.ReverbAddTrajectoryObserver],
    policy_info_fn: _PolicyFnType = None) -> int:
```

指定された `rlds_data` を TF Angets トラジェクトリに変換してから、`reverb_observer` を使用して Reverb サーバーにプッシュします。

`reverb_observer` を作成するために再生バッファと Reverb サーバーをインスタンス化する際のデータ仕様は、`rlds_data` のデータ仕様に一致する必要があることに注意してください。

**引数**:

- `rlds_data`: RLDS データセットは RLDS エピソードの `tf.data.Dataset` で、各エピソードには RLDS ステップの `tf.data.Dataset`（とオプションとしてエピソードのメタデータ）が含まれます。RLDS ステップは、`is_first`、`is_last`、`observation`、`action`、`reward`、`is_terminal`、および `discount`（さらにオプションのステップメタデータ）を含むテンソルのディクショナリです。
- `reverb_observer`: トラジェクトリデータを Reverb に書き込むための Reverb オブザーバー。
- `policy_info_fn`: TF-Agents トラジェクトリを生成する際に使用される policy.info を作成するオプションの関数。

**戻り値**:

- RLDS に正しくプッシュされたトラジェクトリ数を表す `int`。

**エラー**:

- `ValueError`: `rlds_data` に RLDS ステップが存在しない場合に発生します。

- `ValueError`: `reverb_observer` を作成するために再生バッファと Reverb サーバーを初期化する際のデータ仕様が `rlds_data` を使用して作成できるトラジェクトリデータセットのデータ仕様に一致しない場合に発生します。

- `InvalidArgumentError`: 指定された RLDS データセットに以下のようなエピソードが含まれる場合に発生します。

    - 誤って終了している。つまり、最後のステップで終了していない。
    - 終端が誤っている。つまり、終端のステップが最後のステップではない。
    - 誤って開始している。つまり、最後のステップの後に最初のステップがない。最後のエピソードの最後のステップは関数によって処理されるため、ユーザーが、最後のエピソードの最後のステップの後に最初のステップがあることを確認する必要はありません。

## RLDS ステップとTF Agents トラジェクトリのマッピング

以下のシーケンスは、時間ステップ t、t+1、および t+2 における RLDS ステップです。各ステップには、観測（o）、アクション（a）、報酬（r）、およびディスカウント（d）が含まれます。同じステップの要素は、括弧でグループ化されます。

```
(o_t, a_t, r_t, d_t), (o_t+1, a_t+1, r_t+1, d_t+1), (o_t+2, a_t+2, r_t+2, d_t+2)
```

RLDS

- `o_t` は、時間 t における観測に対応します。

- `a_t` は、時間 t におけるアクションに対応します。

- `r_t` は、観測 `o_t` で実行したアクションに対して受け取る報酬に対応します。

- `d_t` は、報酬 `r_t` に適用されるディスカウントに対応します。

```
Step 1 =  o_0, a_0, r_0, d_0, is_first = true, is_last = false, is_terminal = false
```

```
Step 2 =  o_1, a_1, r_1,d_1, is_first = False, is_last = false, is_terminal = false
```

…

```
Step n =  o_t, a_t, r_t, d_t, is_first = False, is_last = false, is_terminal = false
```

```
Step n+1 =   o_t+1, a_t+1, r_t+1, d_t+1, is_first = False, is_last = true, is_terminal = false
```

`is_terminal = True` である場合、観測は最終状態に対応するため、報酬、ディスカウント、およびアクションは無意味になります。環境によっては、最終観測も無意味になる可能性があります。

エピソードが `is_terminal = False` のステップで終了する場合、そのエピソードは切り捨てられています。この場合、環境によっては、報酬とディスカウントも空である可能性があります。

![RLDS ステップから TF-Agents トラジェクトリ](images/rlds/rlds_step_to_trajectory.png)

### 変換プロセス

#### データセットをフラット化する

RLDS データセットはエピソードのデータセットであり、これが RLDS ステップのデータセットになります。初めにステップのデータセットにフラット化されます。

![フラット化された RLDS](images/rlds/flatten_rlds.png)

#### 隣接するステップのオーバーラップするペアを作成する

RLDS データセットがフラット化されたら、次はバッチ処理されて、隣接する RLDS ステップのオーバーラップするペアのデータセットに変換されます。

![RLDS からオーバーラップするペア](images/rlds/rlds_to_pairs.png)

#### TF-Agents トラジェクトリに変換する

次に、このデータセットは TF-Agents トラジェクトリに変換されます。

![RLDS ペアから TF-Agents トラジェクトリ](images/rlds/pairs_to_trajectories.png)
