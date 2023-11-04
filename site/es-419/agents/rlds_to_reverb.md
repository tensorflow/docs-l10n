# RLDS to Reverb util en TF-Agents

[RLDS](https://github.com/google-research/rlds) to [Reverb](https://github.com/deepmind/reverb) util es una herramienta de TF Agents que permite leer los episodios de RLDS, transformarlos en trayectorias y enviarlos a Reverb.

### Conjunto de datos de RLDS

RLDS (Conjuntos de Datos de Aprendizaje por Refuerzo) es un ecosistema de herramientas que sirve para almacenar, recuperar y manipular datos episódicos en el contexto de la Toma de Decisiones Secuenciales, como el Aprendizaje por Refuerzo (RL), el Aprendizaje a partir de demostraciones, el RL fuera de línea o el Aprendizaje por imitación.

Cada paso cuenta con los siguientes campos (y, en algunos casos, con campos adicionales para metadatos del paso). A mod de ejemplo, se usan las especificaciones del conjunto de datos D4RL [half-cheetah/v0-expert](https://www.tensorflow.org/datasets/catalog/d4rl_mujoco_halfcheetah#d4rl_mujoco_halfcheetahv0-expert_default_config)

- **'action'**: `TensorSpec(shape = (6,), dtype = tf.float32, name = None)`

- **'discount'**: `TensorSpec(shape = (), dtype = tf.float32, name = None)`

- **'is_first'**: `TensorSpec(shape = (), dtype = tf.bool, name = None)`

- **'is_last'**: `TensorSpec(shape = (), dtype = tf.bool, name = None)`

- **'is_terminal'**: `TensorSpec(shape = (), dtype = tf.bool, name = None)`

- **'observation'**: `TensorSpec(shape = (17,), dtype = tf.float32, name = None)`

- **'reward'**: `TensorSpec(shape = (), dtype = tf.float32, name = None)}, TensorShape([]))`

## API de RLDS to TF-Agents utils

### Cómo crear especificación de trayectoria desde un conjunto de datos

Crea especificación de datos para inicializar el servidor de Reverb y Reverb Replay Buffer.

```
def create_trajectory_data_spec(rlds_data: tf.data.Dataset) -> trajectory.Trajectory:
```

Crea una especificación de datos para el conjunto de datos de trayectoria correspondiente que se puede crear mediante el uso de `rlds_data` que se proporciona como entrada. Esta especificación de datos es necesaria para inicializar un servidor de Reverb y Reverb Replay Buffer.

**Argumentos**:

- `rlds_data`: un conjunto de datos de RLDS es un `tf.data.Dataset` de episodios de RLDS, donde cada episodio contiene un `tf.data.Dataset` de pasos de RLDS y, de forma opcional, metadatos de episodios. Un paso de RLDS es un diccionario de tensores que contiene `is_first`, `is_last`, `observation`, `action`, `reward`, `is_terminal` y `discount` (y, en algunos casos, metadatos del paso).

**Devuelve**:

- Una especificación de trayectoria que se puede usar para crear un conjunto de datos de trayectoria con `rlds_data` como entrada.

**Genera**:

- `ValueError`: si no existen los pasos RLDS en `rlds_data`.

### Cómo convertir datos RLDS en trayectorias de TF Agents

Convierte los datos RLDS en un conjunto de datos de trayectorias. Actualmente, solo se admite la conversión a una trayectoria de dos pasos.

```
def convert_rlds_to_trajectories(rlds_data: tf.data.Dataset,
    policy_info_fn: _PolicyFnType = None) -> tf.data.Dataset:
```

Convierte los `rlds_data` proporcionados en un conjunto de datos de trayectorias de TF Agents al aplanarlos y convertirlos en lotes y luego en tuplas de pares superpuestos de pasos RLDS adyacentes.

Los datos del RLDS se amortiguan al final con un paso de tipo `first` para garantizar que la trayectoria creada con el último paso del último episodio tenga un tipo de paso siguiente válido.

**Argumentos**:

- `rlds_data`: un conjunto de datos de RLDS es un `tf.data.Dataset` de episodios de RLDS, donde cada episodio contiene un `tf.data.Dataset` de pasos de RLDS y, de forma opcional, metadatos de episodios. Un paso de RLDS es un diccionario de tensores que contiene `is_first`, `is_last`, `observation`, `action`, `reward`, `is_terminal` y `discount` (y, opcionalmente, metadatos del paso).
- `policy_info_fn`: una función opcional para crear policy.info que se usa para generar trayectorias de TF-Agents.

**Devuelve**:

- Un tipo de conjunto de datos `tf.data.Dataset`, cuyos elementos son trayectorias de TF Agents correspondientes a los pasos RLDS que se proporcionan en `rlds_data`.

**Genera**:

- `ValueError`: si no existen los pasos RLDS en `rlds_data`.

- `InvalidArgumentError`: si el conjunto de datos RLDS proporcionado tiene episodios con las siguientes características:

    - Que terminan incorrectamente, es decir, que no terminan en el último paso.
    - Que finalizan incorrectamente, es decir, con un paso final que no es el último paso.
    - Que comienzan incorrectamente, es decir, un último paso no va seguido de un primer paso. Tenga en cuenta que la función se encarga del último paso del último episodio y que el usuario no necesita asegurarse de que el último paso del último episodio venga seguido de un primer paso.

### Cómo enviar datos RLDS a Reverb

Envía los datos de RLDS al servidor de Reverb como trayectorias de TF Agents. El observador de reverberación deberá instanciarse antes de llamar a la interfaz y proporcionarse como parámetro.

```
def push_rlds_to_reverb(rlds_data: tf.data.Dataset, reverb_observer: Union[
    reverb_utils.ReverbAddEpisodeObserver,
    reverb_utils.ReverbAddTrajectoryObserver],
    policy_info_fn: _PolicyFnType = None) -> int:
```

Envía el `rlds_data` proporcionado al servidor de Reverb mediante el uso de `reverb_observer` tras convertirlo en trayectorias de TF Agents.

Tenga en cuenta que la especificación de datos que se usa para inicializar el búfer de repetición y el servidor de reverberación para crear el `reverb_observer` debe coincidir con la especificación de los datos para `rlds_data`.

**Argumentos**:

- `rlds_data`: un conjunto de datos de RLDS es un `tf.data.Dataset` de episodios de RLDS, donde cada episodio contiene un `tf.data.Dataset` de pasos de RLDS y, de forma opcional, metadatos de episodios. Un paso de RLDS es un diccionario de tensores que contiene `is_first`, `is_last`, `observation`, `action`, `reward`, `is_terminal` y `discount` (y, opcionalmente, metadatos del paso).
- `reverb_observer`: un observador de reverberación para grabar datos de trayectorias en Reverb.
- `policy_info_fn`: una función opcional para crear policy.info que se usa para generar trayectorias de TF-Agents.

**Devuelve**:

- Un `int` que representa la cantidad de trayectorias que se enviaron correctamente a RLDS.

**Genera**:

- `ValueError`: si no existen los pasos RLDS en `rlds_data`.

- `ValueError`: si la especificación de datos que se usa para inicializar el búfer de repetición y el servidor de reverberación para crear el `reverb_observer` no coincide con la especificación de los datos para el conjunto de datos de trayectoria que se puede crear mediante el uso de `rlds_data`.

- `InvalidArgumentError`: si el conjunto de datos RLDS proporcionado tiene episodios con las siguientes características:

    - Que terminan incorrectamente, es decir, que no terminan en el último paso.
    - Que finalizan incorrectamente, es decir, con un paso final que no es el último paso.
    - Que comienzan incorrectamente, es decir, un último paso no va seguido de un primer paso. Tenga en cuenta que la función se encarga del último paso del último episodio y que el usuario no necesita asegurarse de que el último paso del último episodio venga seguido de un primer paso.

## Cómo se corresponden los pasos de RLDS con las trayectorias de TF Agents

La siguiente secuencia corresponde a pasos de RLDS en los pasos del tiempo t, t+1 and t+2. Cada paso contiene una observación (o), una acción (a), una recompensa (r) y un descuento (d). Los elementos de un mismo paso se agrupan entre paréntesis.

```
(o_t, a_t, r_t, d_t), (o_t+1, a_t+1, r_t+1, d_t+1), (o_t+2, a_t+2, r_t+2, d_t+2)
```

En RLDS,

- `o_t` corresponde a la observación a tiempo t

- `a_t` corresponde con la acción a tiempo t

- `r_t` corresponde con la recompensa recibida por haber ejecutado la acción en observación `o_t`

- `d_t` corresponde con el descuento que se aplica a la recompensa `r_t`

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

Cuando `is_terminal = True`, la observación corresponde a un estado final, por lo que la recompensa, el descuento y la acción carecen de sentido. En función del entorno, la observación final también puede carecer de sentido.

Si un episodio termina en un paso donde `is_terminal = False`, significa que este episodio ha sido truncado. En ese caso, en función del entorno, la acción, la recompensa y el descuento también podrían estar vacíos.

![Paso RLDS a trayectoria TF-Agents](images/rlds/rlds_step_to_trajectory.png)

### Proceso de conversión

#### Aplanar el conjunto de datos

El conjunto de datos RLDS es un conjunto de datos de episodios que son a su vez conjuntos de datos de pasos de RLDS. Primero se aplana a un conjunto de datos de pasos.

![RLDS aplanado](images/rlds/flatten_rlds.png)

#### Crear pares superpuestos de pasos adyacentes

El conjunto de datos de RLDS aplanado luego se divide en lotes y se convierte en un conjunto de datos de pares superpuestos de pasos RLDS adyacentes.

![RLDS en pares superpuestos](images/rlds/rlds_to_pairs.png)

#### Convertir en trayectorias de TF-Agents

Luego, el conjunto de datos se convierte en trayectorias de TF Agents.

![Pares RLDS en trayectorias de TF-Agents](images/rlds/pairs_to_trajectories.png)
