# Conversão de RNN do TensorFlow para o TensorFlow Lite

## Visão geral

O TensorFlow Lite oferece suporte à conversão de modelos de RNN do TensorFlow em operações de LSTM combinadas do TensorFlow Lite. As operações combinadas existem para maximizar o desempenho das implementações de kernel subjacentes, bem como para fornecer uma interface de alto nível para definir transformações complexas, como quantização.

Como existem diversas variantes das APIs de RNN no TensorFlow, temos duas estratégias:

1. Oferecer **suporte nativo às APIs padrão de RNN do TensorFlow**, como LSTM do Keras. Essa é a opção recomendada.
2. Oferecer uma **interface** **para a infraestrutura de conversão para ** **implementações de RNN** **definidas pelo usuário** a fim de incluir e converter para o TensorFlow Lite. Fornecemos alguns exemplos prontos dessa conversão usando [LSTMCellSimple](https://github.com/tensorflow/tensorflow/blob/82abf0dbf316526cd718ae8cd7b11cfcb805805e/tensorflow/compiler/mlir/lite/transforms/prepare_composite_functions_tf.cc#L130) do Lingvo e as interfaces de RNN [LayerNormalizedLSTMCellSimple](https://github.com/tensorflow/tensorflow/blob/c11d5d8881fd927165eeb09fd524a80ebaf009f2/tensorflow/compiler/mlir/lite/transforms/prepare_composite_functions_tf.cc#L137).

## API de conversão

Esse recurso faz parte do TensorFlow versão 2.3 e também está disponível via pip [tf-nightly](https://pypi.org/project/tf-nightly/) ou pelo head.

Essa funcionalidade de conversão está disponível ao converter para o TensorFlow Lite usando um SavedModel ou um modelo do Keras diretamente. Veja os exemplos de uso.

### Usando um SavedModel

<a id="from_saved_model"></a>

```
# build a saved model. Here concrete_function is the exported function
# corresponding to the TensorFlow model containing one or more
# Keras LSTM layers.
saved_model, saved_model_dir = build_saved_model_lstm(...)
saved_model.save(saved_model_dir, save_format="tf", signatures=concrete_func)

# Convert the model.
converter = TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()
```

### Usando um modelo do Keras

```
# build a Keras model
keras_model = build_keras_lstm(...)

# Convert the model.
converter = TFLiteConverter.from_keras_model(keras_model)
tflite_model = converter.convert()

```

## Exemplo

O [Colab](https://colab.research.google.com/github/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/experimental_new_converter/Keras_LSTM_fusion_Codelab.ipynb) LSTM do Keras para TensorFlow Lite ilustra um exemplo completo usando o interpretador do TensorFlow Lite.

## APIs de RNN do TensorFlow com suporte

<a id="rnn_apis"></a>

### Conversão de LSTM do Keras (recomendado)

Temos suporte integrado à conversão de LSTM do Keras para o TensorFlow Lite. Confira os detalhes de funcionamento na [interface de LSTM do Keras](https://github.com/tensorflow/tensorflow/blob/35a3ab91b42503776f428bda574b74b9a99cd110/tensorflow/python/keras/layers/recurrent_v2.py#L1238)<span style="text-decoration:space;"> </span>e sua lógica de conversão [aqui](https://github.com/tensorflow/tensorflow/blob/35a3ab91b42503776f428bda574b74b9a99cd110/tensorflow/compiler/mlir/lite/utils/lstm_utils.cc#L627).

Também é importante ressaltar o contrato da LSTM do TensorFlow Lite em relação à definição de operações do Keras:

1. A dimensão 0 do tensor **input** é o tamanho do lote.
2. A dimensão 0 do tensor **recurrent_weight** é o número de saídas.
3. Os tensores **weight** e **recurrent_kernel** são transpostos.
4. Os tensores weight transposto, recurrent_kernel transposto e **bias** são divididos em 4 tensores de tamanho igual ao longo da dimensão 0. Eles correspondem a **input gate, forget gate, cell e output gate**.

#### Variantes de LSTM do Keras

##### Time major

Os usuários podem optar por usar ou não time-major. A LSTM do Keras adiciona um atributo time-major aos atributos da definição da função. Para LSTM de sequência unidirecional, podemos simplesmente mapear para [atributo time major](https://github.com/tensorflow/tensorflow/blob/35a3ab91b42503776f428bda574b74b9a99cd110/tensorflow/compiler/mlir/lite/ir/tfl_ops.td#L3902) da unidirecional_sequence_lstm.

##### LSTM bidirecional

A LSTM bidirecional pode ser implementada com duas camadas LSTM do Keras, uma para frente e uma para trás. Confira alguns exemplos [aqui](https://github.com/tensorflow/tensorflow/blob/35a3ab91b42503776f428bda574b74b9a99cd110/tensorflow/python/keras/layers/wrappers.py#L382). Quando há o atributo go_backward, reconhecemos como LSTM para trás e agrupamos a LSTM para trás e para frente. **Isso ainda será desenvolvido.** No momento, são criadas duas operações UnidirectionalSequenceLSTM no modelo do TensorFlow Lite.

### Exemplos de conversão de LSTM definida pelo usuário

O TensorFlow Lite também conta com uma forma de converter implementações de LSTM definidas pelo usuário. Aqui, usamos a LSTM do Lingvo como exemplo de implementação. Confira mais detalhes na [interface lingvo.LSTMCellSimple](https://github.com/tensorflow/lingvo/blob/91a4609dbc2579748a95110eda59c66d17c594c5/lingvo/core/rnn_cell.py#L228) e sua lógica de conversão [aqui](https://github.com/tensorflow/tensorflow/blob/82abf0dbf316526cd718ae8cd7b11cfcb805805e/tensorflow/compiler/mlir/lite/transforms/prepare_composite_functions_tf.cc#L130). Também oferecemos um exemplo de outra definição de LSTM do Lingvo na [interface lingvo.LayerNormalizedLSTMCellSimple](https://github.com/tensorflow/lingvo/blob/91a4609dbc2579748a95110eda59c66d17c594c5/lingvo/core/rnn_cell.py#L1173) e sua lógica de conversão [aqui](https://github.com/tensorflow/tensorflow/blob/c11d5d8881fd927165eeb09fd524a80ebaf009f2/tensorflow/compiler/mlir/lite/transforms/prepare_composite_functions_tf.cc#L137).

## “Traga sua própria RNN do TensorFlow” para o TensorFlow Lite

Se a interface da RNN de um usuário for diferente das interfaces padrão com suporte, existem algumas opções:

**Opção 1:** escreva o código adaptador no Python do TensorFlow para adaptar a interface da RNN à interface da RNN do Keras. Portanto, é preciso ter uma tf.function com a [anotação tf_implements](https://github.com/tensorflow/community/pull/113) na função da interface da RNN gerada que seja idêntica à gerada pela camada de LSTM do Keras. Em seguida, a mesma API de conversão usada para a LSTM do Keras funcionará.

**Opção 2:** se não for possível seguir a instrução acima (por exemplo, se a LSTM do Keras não tiver alguma funcionalidade exposta atualmente pela operação de LSTM combinada do TensorFlow Lite, como normalização de camadas), é preciso estender o conversor do TensorFlow Lite: basta escrever um código de conversão personalizado e incluí-lo no passo da MLIR prepare-composite-functions (preparar funções compostas) [aqui](https://github.com/tensorflow/tensorflow/blob/c11d5d8881fd927165eeb09fd524a80ebaf009f2/tensorflow/compiler/mlir/lite/transforms/prepare_composite_functions_tf.cc#L115). A interface da função deve ser tratada como um contrato de API e deve conter os argumentos necessários para converter para operações de LSTM combinadas do TensorFlow Lite, como entrada, bias, pesos, projeção, normalização de camadas, etc. É preferível que os tensores passados como argumentos para essa função tenham um posto conhecido (por exemplo, RankedTensorType na MLIR). Dessa forma, fica muito mais fácil escrever código de conversão que possa presumir que esses tensores sejam RankedTensorType, além de ser possível transformá-los em tensores com posto correspondentes aos operandos da operação combinada do TensorFlow Lite.

Um exemplo completo desse workflow de conversão é o LSTMCellSimple do Lingvo para conversão para o TensorFlow Lite.

O LSTMCellSimple do Lingvo está definido [aqui](https://github.com/tensorflow/lingvo/blob/91a4609dbc2579748a95110eda59c66d17c594c5/lingvo/core/rnn_cell.py#L228). Os modelos treinados com essa célula de LSTM podem ser convertidos para o TensorFlow Lite da seguinte forma:

1. Encapsule todos os usos de LSTMCellSimple em uma tf.function com a anotação tf_implements que seja rotulada dessa forma (por exemplo, lingvo.LSTMCellSimple seria um bom nome de anotação nesse caso). A tf.function gerada precisa coincidir com a interface da função esperada pelo código de conversão. Isso é um contrato entre o autor do modelo que adiciona a anotação e o código de conversão.

2. Estenda o passo prepare-composite-functions para incluir uma operação composta personalizada para a conversão em operação de LSTM combinada do TensorFlow Lite. Confira o código de conversão [LSTMCellSimple](https://github.com/tensorflow/tensorflow/blob/82abf0dbf316526cd718ae8cd7b11cfcb805805e/tensorflow/compiler/mlir/lite/transforms/prepare_composite_functions_tf.cc#L130).

    Contrato de conversão:

3. Os tensores **weight** e **projection** são transpostos.

4. **{input, recurrent}** de **{cell, input gate, forget gate, output gate}** são extraídos por meio do fatiamento do tensor de pesos transposto.

5. **{bias}** de **{cell, input gate, forget gate, output gate}** é extraído pelo fatiamento do tensor bias.

6. **projection** é extraída pelo fatiamento do tensor de projeção transposto.

7. Uma conversão similar é escrita para [LayerNormalizedLSTMCellSimple](https://github.com/tensorflow/tensorflow/blob/c11d5d8881fd927165eeb09fd524a80ebaf009f2/tensorflow/compiler/mlir/lite/transforms/prepare_composite_functions_tf.cc#L137).

8. É possível reutilizar o restante da infraestrutura de conversão do TensorFlow Lite, incluindo todos os [passos da MLIR](https://github.com/tensorflow/tensorflow/blob/35a3ab91b42503776f428bda574b74b9a99cd110/tensorflow/compiler/mlir/lite/tf_tfl_passes.cc#L57), bem como a exportação final para Flatbuffer do TensorFlow Lite.

## Limitações/problemas conhecidos

1. No momento, há suporte somente à conversão de LSTM stateless do Keras (comportamento padrão do Keras). A conversão de LSTM stateful do Keras ainda será desenvolvida.
2. Ainda é possível modelar uma camada de LSTM stateful do Keras usando a camada de LSTM stateless do Keras subjacente e gerenciando o estado explicitamente no programa do usuário. Um programa do TensorFlow como esse ainda pode ser convertido para o TensorFlow Lite usando o recurso descrito aqui.
3. No momento, a LSTM bidirecional é modelada como duas operações UnidirectionalSequenceLSTM no TensorFlow Lite. Isso será substituído por uma única operação BidirectionalSequenceLSTM.
