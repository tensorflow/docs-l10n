# Utilitário RLDS para Reverb no TF-Agents

O utilitário [RLDS](https://github.com/google-research/rlds) para [Reverb](https://github.com/deepmind/reverb) é uma ferramenta no TF Agents para ler os episódios do RLDS, transformá-los em trajetórias e enviá-los ao Reverb.

### Dataset RLDS

O RLDS (Datasets de Aprendizado por Reforço) é um ecossistema de ferramentas para armazenar, recuperar e manipular dados episódicos no contexto da Tomada de Decisão Sequencial, incluindo o Aprendizado por Reforço (RL), o Aprendizado por Demonstração, o RL Offline ou o Aprendizado por Imitação.

Cada passo tem os campos abaixo (e, às vezes, campos adicionais para metadados de passos). Como um exemplo, usamos as especificações do dataset D4RL [half-cheetah/v0-expert](https://www.tensorflow.org/datasets/catalog/d4rl_mujoco_halfcheetah#d4rl_mujoco_halfcheetahv0-expert_default_config)

- **'action'**: `TensorSpec(shape = (6,), dtype = tf.float32, name = None)`

- **'discount'**: `TensorSpec(shape = (), dtype = tf.float32, name = None)`

- **'is_first'**: `TensorSpec(shape = (), dtype = tf.bool, name = None)`

- **'is_last'**: `TensorSpec(shape = (), dtype = tf.bool, name = None)`

- **'is_terminal'**: `TensorSpec(shape = (), dtype = tf.bool, name = None)`

- **'observation'**: `TensorSpec(shape = (17,), dtype = tf.float32, name = None)`

- **'reward'**: `TensorSpec(shape = (), dtype = tf.float32, name = None)}, TensorShape([]))`

## API dos utilitários de RLDS para TF-Agents

### Crie uma especificação de trajetória a partir de um dataset

Cria especificações de dados para inicializar o servidor Reverb e o Buffer de Replay do Reverb.

```
def create_trajectory_data_spec(rlds_data: tf.data.Dataset) -> trajectory.Trajectory:
```

Cria a especificação de dados para o dataset de trajetória correspondente que pode ser criado usando o `rlds_data ` fornecido como entrada. Essa especificação de dados é necessária para inicializar um servidor Reverb e Buffer de Replay do Reverb.

**Argumentos**:

- `rlds_data`: um dataset RLDS é um `tf.data.Dataset` de episódios RLDS, em que cada episódio contém um `tf.data.Dataset` de passos RLDS e, opcionalmente, metadados de episódios. Um passo RLDS é um dicionário de tensores que contêm `is_first`, `is_last`, `observation`, `action`, `reward`, `is_terminal` e `discount` (e, às vezes, metadados de passos).

**Retorna**:

- Uma especificação de trajetória que pode ser usada para criar um dataset de trajetória com o `rlds_data` fornecido como entrada.

**Exceções**:

- `ValueError`: se não existir nenhum passo RLDS em `rlds_data`.

### Converta dados RLDS para trajetórias do TF Agents

Converte os dados RLDS para um dataset de trajetórias. No momento, só há compatibilidade com a conversão para uma trajetória de dois passos.

```
def convert_rlds_to_trajectories(rlds_data: tf.data.Dataset,
    policy_info_fn: _PolicyFnType = None) -> tf.data.Dataset:
```

Converte o `rlds_data` fornecido para um dataset de trajetórias do TF Agents com o flattening e a conversão em lotes e, depois, tuplas com pares sobrepostos de passos RLDS adjacentes.

Os dados RLDS são preenchidos no final com um passo do tipo `first` para garantir que a trajetória criada usando o último passo do último episódio tenha um próximo tipo de passo válido.

**Argumentos**:

- `rlds_data`: um dataset RLDS é um `tf.data.Dataset` de episódios RLDS, em que cada episódio contém um `tf.data.Dataset` de passos RLDS e, opcionalmente, metadados de episódios. Um passo RLDS é um dicionário de tensores que contêm `is_first`, `is_last`, `observation`, `action`, `reward`, `is_terminal` e `discount` (e metadados de passos).
- `policy_info_fn`: uma função opcional para criar algumas policy.info que serão usadas ao gerar trajetórias do TF-Agents.

**Retorna**:

- Um dataset do tipo `tf.data.Dataset`, cujos elementos são trajetórias do TF Agents correspondentes aos passos RLDS fornecidos em `rlds_data`.

**Exceções**:

- `ValueError`: se não existir nenhum passo RLDS em `rlds_data`.

- `InvalidArgumentError`: se o dataset RLDS fornecido tiver episódios que:

    - Terminam incorretamente, ou seja, não terminam no último passo.
    - Terminam incorretamente, ou seja, um passo terminal não é o último.
    - Começam incorretamente, ou seja, um último passo não é seguido pelo primeiro. Observe que o último passo do último episódio é tratado na função, e o usuário não precisa verificar se o último passo do último episódio é seguido por um primeiro.

### Envie dados RLDS para Reverb

Envia os dados RLDS para o servidor Reverb como trajetórias do TF Agents. O observador Reverb precisa ser instanciado antes de chamar a interface e fornecido como um parâmetro.

```
def push_rlds_to_reverb(rlds_data: tf.data.Dataset, reverb_observer: Union[
    reverb_utils.ReverbAddEpisodeObserver,
    reverb_utils.ReverbAddTrajectoryObserver],
    policy_info_fn: _PolicyFnType = None) -> int:
```

Envia o `rlds_data` fornecido para o servidor usando o `reverb_observer` após a conversão para trajetórias do TF Agents.

Observe que as especificações de dados usadas para inicializar o buffer de replay e o servidor do reverb para criar o `reverb_observer` precisam corresponder às especificações de dados para `rlds_data`.

**Argumentos**:

- `rlds_data`: um dataset RLDS é um `tf.data.Dataset` de episódios RLDS, em que cada episódio contém um `tf.data.Dataset` de passos RLDS e, opcionalmente, metadados de episódios. Um passo RLDS é um dicionário de tensores que contêm `is_first`, `is_last`, `observation`, `action`, `reward`, `is_terminal` e `discount` (e metadados de passos).
- `reverb_observer`: um observador Reverb para escrever dados de trajetórias para Reverb.
- `policy_info_fn`: uma função opcional para criar algumas policy.info que serão usadas ao gerar trajetórias do TF-Agents.

**Retorna**:

- Um `int` que representa o número de trajetórias enviadas a RLDS com êxito.

**Exceções**:

- `ValueError`: se não existir nenhum passo RLDS em `rlds_data`.

- `ValueError`: se as especificações de dados usadas para inicializar o buffer de replay e o servidor do reverb para criar o `reverb_observer` não corresponderem às especificações de dados para o dataset de trajetória que pode ser criado usando `rlds_data`.

- `InvalidArgumentError`: se o dataset RLDS fornecido tiver episódios que:

    - Terminam incorretamente, ou seja, não terminam no último passo.
    - Terminam incorretamente, ou seja, um passo terminal não é o último.
    - Começam incorretamente, ou seja, um último passo não é seguido pelo primeiro. Observe que o último passo do último episódio é tratado na função, e o usuário não precisa verificar se o último passo do último episódio é seguido por um primeiro.

## Como os passos RLDS mapeiam a trajetórias do TF Agents

A sequência a seguir são passos RLDS em timesteps t, t+1 e t+2. Cada passo contém uma observação (o), ação (a), recompensa (r) e desconto (d). Os elementos do mesmo passo são agrupados em parênteses.

```
(o_t, a_t, r_t, d_t), (o_t+1, a_t+1, r_t+1, d_t+1), (o_t+2, a_t+2, r_t+2, d_t+2)
```

No RLDS,

- `o_t` corresponde à observação no tempo t

- `a_t` corresponde à ação no tempo t

- `r_t` corresponde à recompensa recebida por realizar a ação na observação `o_t`

- `d_t` corresponde ao desconto aplicado à recompensa `r_t`

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

Quando `is_terminal = True`, a observação corresponde a um estado final, então a recompensa, o desconto e a ação são insignificantes. Dependendo do ambiente, a observação final também pode ser insignificante.

Se um episódio terminar em um passo onde `is_terminal = False`, significa que esse episódio foi truncado. Nesse caso, dependendo do ambiente, a ação, a recompensa e o desconto podem estar vazios também.

![Passos RLDS à trajetória do TF-Agents](images/rlds/rlds_step_to_trajectory.png)

### Processo de conversão

#### Realize o flattening do dataset

O dataset RLDS é composto por episódios que são, por sua vez, datasets de passos RLDS. É primeiro aplicado o flattening para um dataset de passos.

![Flattening do RLDS](images/rlds/flatten_rlds.png)

#### Crie pares sobrepostos de passos adjacentes

Após o flattening, o dataset RLDS é dividido em lotes e convertido para um dataset de pares sobrepostos de passos RLDS adjacentes.

![RLDS para pares sobrepostos](images/rlds/rlds_to_pairs.png)

#### Converta para trajetórias do TF-Agents

Em seguida, o dataset é convertido para trajetórias do TF-Agents.

![Pares RLDS para trajetórias do TF-Agents](images/rlds/pairs_to_trajectories.png)
