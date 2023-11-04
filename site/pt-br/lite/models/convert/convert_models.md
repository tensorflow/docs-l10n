# Converta modelos do TensorFlow

Esta página descreve como converter um modelo do TensorFlow para um do TensorFlow Lite (um formato [FlatBuffer](https://google.github.io/flatbuffers/) otimizado, indicado pela extensão de arquivo `.tflite`) usando o conversor do TensorFlow Lite.

Observação: este guia pressupõe que você [tenha instalado o TensorFlow 2.x](https://www.tensorflow.org/install/pip#tensorflow-2-packages-are-available) e treinado modelos no TensorFlow 2.x. Caso o seu modelo tenha sido treinado no TensorFlow 1.x, considere [migrar para o TensorFlow 2.x](https://www.tensorflow.org/guide/migrate/tflite). Para identificar a versão do TensorFlow instalada, execute `print(tf.__version__)`.

## Workflow de conversão

O diagrama abaixo ilustra o workflow geral para converter um modelo:

![TFLite converter workflow](../../images/convert/convert.png)

**Figura 1.** Workflow do conversor

É possível converter um modelo por uma das seguintes opções:

1. [API do Python](#python_api) (***opção recomendada***): permite integrar a conversão ao seu pipeline de desenvolvimento, aplicar otimizações, adicionar metadados e realizar diversas outras tarefas que simplificam o processo de conversão.
2. [Linha de comando](#cmdline): tem suporte somente à conversão básica de modelos.

Observação: caso haja problemas durante a conversão do modelo, crie um [issue no GitHub](https://github.com/tensorflow/tensorflow/issues/new?template=60-tflite-converter-issue.md).

## API do Python <a name="python_api"></a>

*Código auxiliar: para saber mais sobre a API do conversor do TensorFlow Lite, execute `print(help(tf.lite.TFLiteConverter))`.*

Converta um modelo do TensorFlow usando [`tf.lite.TFLiteConverter`](https://www.tensorflow.org/api_docs/python/tf/lite/TFLiteConverter). Um modelo do TensorFlow é armazenado usando o formato SavedModel e é gerado usando as APIs de alto nível `tf.keras.*` (um modelo do Keras) ou as APIs de baixo nível `tf.*` (a partir das quais você gera funções concretas). Consequentemente, você tem as três opções abaixo (confira os exemplos nas próximas seções):

- `tf.lite.TFLiteConverter.from_saved_model()` (**opção recomendada**): converte um [SavedModel](https://www.tensorflow.org/guide/saved_model).
- `tf.lite.TFLiteConverter.from_keras_model()`: converte um modelo do [Keras](https://www.tensorflow.org/guide/keras/overview).
- `tf.lite.TFLiteConverter.from_concrete_functions()`: converte [funções concretas](https://www.tensorflow.org/guide/intro_to_graphs).

### Converta um SavedModel (recomendado) <a name="saved_model"></a>

O exemplo abaixo mostra como converter um [SavedModel](https://www.tensorflow.org/guide/saved_model) em um modelo do TensorFlow Lite.

```python
import tensorflow as tf

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir) # path to the SavedModel directory
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)
```

### Converta um modelo do Keras <a name="keras"></a>

O exemplo abaixo mostra como converter um modelo do [Keras](https://www.tensorflow.org/guide/keras/overview) em um modelo do TensorFlow Lite.

```python
import tensorflow as tf

# Create a model using high-level tf.keras.* APIs
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1]),
    tf.keras.layers.Dense(units=16, activation='relu'),
    tf.keras.layers.Dense(units=1)
])
model.compile(optimizer='sgd', loss='mean_squared_error') # compile the model
model.fit(x=[-1, 0, 1], y=[-3, -1, 1], epochs=5) # train the model
# (to generate a SavedModel) tf.saved_model.save(model, "saved_model_keras_dir")

# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)
```

### Converta funções concretas <a name="concrete_function"></a>

O exemplo abaixo mostra como converter [funções concretas](https://www.tensorflow.org/guide/intro_to_graphs) em um modelo do TensorFlow Lite.

```python
import tensorflow as tf

# Create a model using low-level tf.* APIs
class Squared(tf.Module):
  @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.float32)])
  def __call__(self, x):
    return tf.square(x)
model = Squared()
# (ro run your model) result = Squared(5.0) # This prints "25.0"
# (to generate a SavedModel) tf.saved_model.save(model, "saved_model_tf_dir")
concrete_func = model.__call__.get_concrete_function()

# Convert the model.

converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func],
                                                            model)
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)
```

### Outros recursos

- Aplique [otimizações](../../performance/model_optimization.md). Uma otimização usada com frequência é a [quantização pós-treinamento](../../performance/post_training_quantization.md), que pode reduzir a latência e o tamanho do modelo, com perda mínima da exatidão.

- Adicione [metadados](metadata.md), que facilitam a criação de código encapsulador para plataformas específicas ao implantar modelos em dispositivos.

### Erros de conversão

Veja abaixo os erros de conversão comuns e suas respectivas soluções:

- Erro: `Some ops are not supported by the native TFLite runtime, you can enable TF kernels fallback using TF Select. See instructions: https://www.tensorflow.org/lite/guide/ops_select. TF Select ops: ..., .., ...` (Algumas operações não têm suporte do runtime nativo do TFLite. Você pode ativar o fallback para kernels do TF usando TF Select. Confira as instruções: https://www.tensorflow.org/lite/guide/ops_select. Operações específicas do TF: ..., .., ...)

    Solução: esse erro ocorre quando seu modelo usa operações do TF que não têm uma implementação correspondente no TF Lite. Para resolver esse problema, basta [usar a operação do TF no modelo do TF Lite](../../guide/ops_select.md) (recomendado). Se você quiser gerar um modelo somente com operações do TF Lite, pode adicionar uma solicitação para a operação do TF Lite ausente no [issue 21526 do GitHub](https://github.com/tensorflow/tensorflow/issues/21526) (deixe um comentário caso sua solicitação ainda não tenha sido mencionada) ou [pode criar a operação do TF Lite](../../guide/ops_custom#create_and_register_the_operator) por conta própria.

- Erro: `.. is neither a custom op nor a flex op` (... não é uma operação personalizada nem uma operação flex).

    Solução: se essa operação do TF:

    - Tiver suporte no TF: o erro ocorre porque a operação do TF está ausente na [lista de permissão](../../guide/op_select_allowlist.md) (uma lista completa das operações do TF com suporte no TF Lite). Você pode resolver da seguinte forma:

        1. [Adicione as operações ausentes à lista de permissão](../../guide/op_select_allowlist.md#add_tensorflow_core_operators_to_the_allowed_list).
        2. [Converta o modelo do TF em um modelo do TF Lite e execute a inferência](../../guide/ops_select.md).

    - Não tiver suporte no TF: o erro ocorre porque o TF Lite não conhece o operador do TF personalizado definido por você. É possível resolver da seguinte forma:

        1. [Crie a operação do TF](https://www.tensorflow.org/guide/create_op).
        2. [Converta o modelo do TF em um do TF Lite](../../guide/op_select_allowlist.md#users_defined_operators).
        3. [Crie e a operação do TF Lite](../../guide/ops_custom.md#create_and_register_the_operator) e execute a inferência fazendo sua vinculação ao runtime do TF Lite.

## Ferramenta de linha de comando <a name="cmdline"></a>

**Observação:** é altamente recomendável usar a [API do Python](#python_api) indicada acima, se possível.

Se você [tiver instalado o TensorFlow 2.x via pip](https://www.tensorflow.org/install/pip), use o comando `tflite_convert`. Para ver todos os sinalizadores disponíveis, use o seguinte comando:

```sh
$ tflite_convert --help

`--output_file`. Type: string. Full path of the output file.
`--saved_model_dir`. Type: string. Full path to the SavedModel directory.
`--keras_model_file`. Type: string. Full path to the Keras H5 model file.
`--enable_v1_converter`. Type: bool. (default False) Enables the converter and flags used in TF 1.x instead of TF 2.x.

You are required to provide the `--output_file` flag and either the `--saved_model_dir` or `--keras_model_file` flag.
```

Se você tiver baixado o [código-fonte do TensorFlow 2.x](https://www.tensorflow.org/install/source) e quiser executar o conversor a partir dele sem compilar e instalar o pacote, pode substituir '`tflite_convert`' por '`bazel run tensorflow/lite/python:tflite_convert --`' no comando.

### Como converter um SavedModel <a name="cmdline_saved_model"></a>

```sh
tflite_convert \
  --saved_model_dir=/tmp/mobilenet_saved_model \
  --output_file=/tmp/mobilenet.tflite
```

### Como converter um modelo H5 do Keras <a name="cmdline_keras_model"></a>

```sh
tflite_convert \
  --keras_model_file=/tmp/mobilenet_keras_model.h5 \
  --output_file=/tmp/mobilenet.tflite
```

## Próximos passos

Use o [interpretador do TensorFlow Lite](../../guide/inference.md) para executar a inferência em um dispositivo cliente (como um dispositivo móvel ou embarcado, por exemplo).
