# TF-Agents의 RLDS to Reverb util

[RLDS](https://github.com/google-research/rlds) to [Reverb](https://github.com/deepmind/reverb) util은 TF-Agents에서 RLDS로부터 에피소드를 읽고 이 에피소드들을 궤적으로 변환하여 Reverb로 푸시하는 도구입니다.

### RLDS 데이터세트

RLDS(Reinforcement Learning Datasets)는 강화 학습(RL), 데모 학습, 오프라인 RL, 모방 학습을 포함하여 순차적 의사 결정의 맥락에서 에피소드 데이터를 저장, 검색, 조작하는 도구 생태계입니다.

각 단계에는 아래와 같은 필드(경우에 따라, 단계 메타데이터에 대한 추가 필드)가 있습니다. 예를 들어, D4RL 데이터세트 [half-cheetah/v0-expert](https://www.tensorflow.org/datasets/catalog/d4rl_mujoco_halfcheetah#d4rl_mujoco_halfcheetahv0-expert_default_config)의 사양을 사용해 보겠습니다.

- **'action'**: `TensorSpec(shape = (6,), dtype = tf.float32, name = None)`

- **'discount'**: `TensorSpec(shape = (), dtype = tf.float32, name = None)`

- **'is_first'**: `TensorSpec(shape = (), dtype = tf.bool, name = None)`

- **'is_last'**: `TensorSpec(shape = (), dtype = tf.bool, name = None)`

- **'is_terminal'**: `TensorSpec(shape = (), dtype = tf.bool, name = None)`

- **'observation'**: `TensorSpec(shape = (17,), dtype = tf.float32, name = None)`

- **'reward'**: `TensorSpec(shape = (), dtype = tf.float32, name = None)}, TensorShape([]))`

## RLDS to TF-Agents util의 API

### 데이터세트에서 궤적 사양 만들기

Reverb 서버 및 Reverb Replay Buffer 초기화를 위해 데이터 사양을 생성합니다.

```
def create_trajectory_data_spec(rlds_data: tf.data.Dataset) -> trajectory.Trajectory:
```

입력된 `rlds_data`를 사용하여 생성할 수 있는 해당 궤적 데이터세트에 대한 데이터 사양을 생성합니다. 이 데이터 사양은 Reverb 서버 및 Reverb Replay Buffer를 초기화하는 데 필요합니다.

**인수**:

- `rlds_data`: RLDS 데이터세트는 RLDS 에피소드의 `tf.data.Dataset`이며, 여기서 각 에피소드에는 RLDS 단계의 `tf.data.Dataset`와 (선택적으로) 에피소드 메타데이터가 포함됩니다. RLDS 단계는 `is_first`, `is_last`, `observation`, `action`, `reward`, `is_terminal`, `discount` (경우에 따라, 단계 메타데이터)를 포함하는 텐서의 사전입니다.

**반환**:

- 입력된 `rlds_data`를 사용하여 궤적 데이터세트를 만드는 데 사용할 수 있는 궤적 사양입니다.

**발생**:

- `ValueError`: `rlds_data`에 RLDS 단계가 없는 경우.

### RLDS 데이터를 TF Agents 궤적으로 변환

RLDS 데이터를 궤적 데이터세트로 변환합니다. 현재 2단계 궤적 변환만 지원합니다.

```
def convert_rlds_to_trajectories(rlds_data: tf.data.Dataset,
    policy_info_fn: _PolicyFnType = None) -> tf.data.Dataset:
```

제공된 `rlds_data`를 병합하고 배치로 변환한 다음 인접한 RLDS 단계에서 겹치는 쌍의 튜플로 변환하여 TF Agents 궤적의 데이터세트로 변환합니다.

RLDS 데이터는 끝부분에 단계 유형 `first`로 되어 있어 마지막 에피소드의 마지막 단계를 사용하여 생성된 궤적에는 유효한 다음 단계 유형이 있습니다.

**인수**:

- `rlds_data`: RLDS 데이터세트는 RLDS 에피소드의 `tf.data.Dataset`이며, 여기서 각 에피소드에는 RLDS 단계의 `tf.data.Dataset` (및 선택적으로 에피소드 메타데이터)이 포함됩니다. RLDS 단계는 `is_first`, `is_last`, `observation`, `action`, `reward`, `is_terminal`, `discount` (및 선택적으로 단계 메타데이터)를 포함하는 텐서의 사전입니다.
- `policy_info_fn`: TF-Agent 궤적을 생성하는 동안 사용할 policy.info를 일부 생성하는 선택적 기능입니다.

**반환**:

- `tf.data.Dataset` 유형의 데이터세트로, 이 데이터세트의 요소는 `rlds_data`에서 제공된 RLDS 단계에 해당하는 TF Agents 궤적입니다.

**발생**:

- `ValueError`: `rlds_data`에 RLDS 단계가 없는 경우.

- `InvalidArgumentError`: 제공된 RLDS 데이터세트에 다음과 같은 에피소드가 있는 경우:

    - 잘못 종료(마지막 단계에서 끝나지 않는 경우)
    - 잘못 종료(종료 단계가 마지막 단계가 아닌 경우)
    - 잘못 시작(마지막 단계 다음에 첫 번째 단계가 오지 않는 경우. 마지막 에피소드의 마지막 단계는 함수에서 처리되며 사용자는 마지막 에피소드의 마지막 단계 다음에 첫 번째 단계가 오는지 확인할 필요가 없음)

### RLDS 데이터를 Reverb로 푸시

RLDS 데이터를 TF Agents 궤적으로 Reverb 서버에 푸시합니다. Reverb 관찰자는 인터페이스를 호출하기 전에 인스턴스화하고 매개변수로 제공되어야 합니다.

```
def push_rlds_to_reverb(rlds_data: tf.data.Dataset, reverb_observer: Union[
    reverb_utils.ReverbAddEpisodeObserver,
    reverb_utils.ReverbAddTrajectoryObserver],
    policy_info_fn: _PolicyFnType = None) -> int:
```

`reverb_observer`를 TF Agents 궤적으로 변환한 후 이를 사용해 Reverb 서버에 제공된 `rlds_data`를 푸시합니다.

`reverb_observer` 생성을 위한 재현 버퍼 및 리버브 서버를 초기화하는 데 사용되는 데이터 사양은 `rlds_data`의 데이터 사양과 일치해야 합니다.

**인수**:

- `rlds_data`: RLDS 데이터세트는 RLDS 에피소드의 `tf.data.Dataset`이며, 여기서 각 에피소드에는 RLDS 단계의 `tf.data.Dataset` (및 선택적으로 에피소드 메타데이터)이 포함됩니다. RLDS 단계는 `is_first`, `is_last`, `observation`, `action`, `reward`, `is_terminal`, `discount` (및 선택적으로 단계 메타데이터)를 포함하는 텐서의 사전입니다.
- `reverb_observer`: Reverb에 궤적 데이터를 쓰기 위한 Reverb 관찰자.
- `policy_info_fn`: TF-Agent 궤적을 생성하는 동안 사용할 policy.info를 일부 생성하는 선택적 기능.

**반환**:

- `int`는 RLDS에 성공적으로 푸시된 궤적의 수를 나타냅니다.

**발생**:

- `ValueError`: `rlds_data`에 RLDS 단계가 없는 경우.

- `ValueError`: `reverb_observer`생성을 위한 재현 버퍼 및 리버브 서버 초기화에 사용된 데이터 사양은 `rlds_data`를 사용하여 생성할 수 있는 궤적 데이터세트의 데이터 사양과 일치하지 않습니다.

- `InvalidArgumentError`: 제공된 RLDS 데이터세트에 다음과 같은 에피소드가 있는 경우:

    - 잘못 종료(마지막 단계에서 끝나지 않음)
    - 잘못 종료(종료 단계가 마지막 단계가 아닌 경우)
    - 잘못 시작(마지막 단계 다음에 첫 번째 단계가 오지 않는 경우. 마지막 에피소드의 마지막 단계는 함수에서 처리되며 사용자는 마지막 에피소드의 마지막 단계 다음에 첫 번째 단계가 오는지 확인할 필요가 없음)

## RLDS 단계가 TF Agents 궤적에 매핑되는 방법

다음 시퀀스는 시간 단계 t, t+1, t+2에 대한 RLDS 단계입니다. 각 단계에는 관찰(o), 행동(a), 보상(r), 감가율(d)이 포함됩니다. 같은 단계의 요소는 괄호 안에 그룹화됩니다.

```
(o_t, a_t, r_t, d_t), (o_t+1, a_t+1, r_t+1, d_t+1), (o_t+2, a_t+2, r_t+2, d_t+2)
```

RLDS에서

- `o_t`는 시간 t에서의 관찰에 해당합니다.

- `a_t`는 시간 t에서의 행동에 해당합니다.

- `r_t`는 관찰 `o_t`에서 수행한 행동에 대해 받은 보상에 해당합니다.

- `d_t`는 보상 `r_t`에 적용된 감가율에 해당합니다.

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

`is_terminal = True`인 경우, 관찰은 최종 상태에 해당하므로 보상, 감가율, 행동은 의미가 없습니다. 환경에 따라 최종 관찰도 의미가 없을 수 있습니다.

에피소드가 `is_terminal = False`인 단계에서 끝나면 이 에피소드가 중단되었음을 의미합니다. 이런 경우, 환경에 따라 행동, 보상, 감가율도 비어있을 수 있습니다.

![RLDS step to TF-Agents trajectory](images/rlds/rlds_step_to_trajectory.png)

### 변환 프로세스

#### 데이터세트 병합

RLDS 데이터 세트는 순차적으로 RLDS 단계의 데이터세트가 되는 에피소드의 데이터세트이며, 단계의 데이터세트로 우선 병합됩니다.

![Flatten RLDS](images/rlds/flatten_rlds.png)

#### 인접한 단계의 겹치는 쌍 만들기

그런 다음 병합된 RLDS 데이터세트를 일괄 처리하고 인접한 RLDS 단계에서 겹치는 쌍의 데이터세트로 변환합니다.

![RLDS to overlapping pairs](images/rlds/rlds_to_pairs.png)

#### TF 에이전트 궤적으로 변환

데이터세트는 그런 다음 TF-Agents 궤적으로 변환됩니다.

![RLDS pairs to TF-Agents trajectories](images/rlds/pairs_to_trajectories.png)
