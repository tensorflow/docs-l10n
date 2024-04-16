# Conversión de TensorFlow RNN a TensorFlow Lite

## Visión general

TensorFlow Lite admite la conversión de modelos RNN de TensorFlow a las operaciones LSTM fusionadas de TensorFlow Lite. Las operaciones fusionadas existen para maximizar el rendimiento de sus implementaciones de kernel subyacentes, así como para proporcionar una interfaz de alto nivel para definir transformaciones complejas como la cuantización.

Dado que existen muchas variantes de API RNN en TensorFlow, nuestro enfoque ha sido doble:

1. Proporcionar **soporte nativo para APIs RNN estándar de TensorFlow** como Keras LSTM. Esta es la opción recomendada.
2. Proporcionar una **interfaz** **a la infraestructura de conversión para** **definidas por el usuario** **implementaciones de RNN** para conectarlas y convertirlas a TensorFlow Lite. Damos un par de ejemplos listos para implementar de dicha conversión usando las interfaces RNN [LSTMCellSimple](https://github.com/tensorflow/tensorflow/blob/82abf0dbf316526cd718ae8cd7b11cfcb805805e/tensorflow/compiler/mlir/lite/transforms/prepare_composite_functions_tf.cc#L130) y [LayerNormalizedLSTMCellSimple](https://github.com/tensorflow/tensorflow/blob/c11d5d8881fd927165eeb09fd524a80ebaf009f2/tensorflow/compiler/mlir/lite/transforms/prepare_composite_functions_tf.cc#L137) de lingvo.

## API de conversión

La función forma parte de la versión 2.3 de TensorFlow. También está disponible a través del pip [tf-nightly](https://pypi.org/project/tf-nightly/) o desde la cabecera.

Esta funcionalidad de conversión está disponible cuando se convierte a TensorFlow Lite a través de un SavedModel o desde el modelo Keras directamente. Vea ejemplos de uso.

### Del modelo guardado

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

### Del modelo Keras

```
# build a Keras model
keras_model = build_keras_lstm(...)

# Convert the model.
converter = TFLiteConverter.from_keras_model(keras_model)
tflite_model = converter.convert()

```

## Ejemplo

La [Colab](https://colab.research.google.com/github/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/experimental_new_converter/Keras_LSTM_fusion_Codelab.ipynb) de Keras LSTM a TensorFlow Lite ilustra el uso de principio a fin con el intérprete TensorFlow Lite.

## APIs RNNs de TensorFlow compatibles

<a id="rnn_apis"></a>

### Conversión Keras LSTM (recomendada)

Apoyamos la conversión lista para implementar de Keras LSTM a TensorFlow Lite. Para más detalles sobre cómo funciona, consulte la interfaz [Keras LSTM](https://github.com/tensorflow/tensorflow/blob/35a3ab91b42503776f428bda574b74b9a99cd110/tensorflow/python/keras/layers/recurrent_v2.py#L1238)<span style="text-decoration:space;"> </span>y la lógica de conversión [aquí](https://github.com/tensorflow/tensorflow/blob/35a3ab91b42503776f428bda574b74b9a99cd110/tensorflow/compiler/mlir/lite/utils/lstm_utils.cc#L627).

También es importante destacar el contrato LSTM de TensorFlow Lite con respecto a la definición de operación de Keras:

1. La dimensión 0 del tensor de **input** es el tamaño del lote.
2. La dimensión 0 del tensor **recurrent_weight** es el número de salidas.
3. Los tensores **weight** y **recurrent_kernel** se transponen.
4. Los tensores recurrent_kernel y **bias** de ponderación transpuesta, se descomponen en 4 tensores de igual tamaño a lo largo de la dimensión 0. Éstos corresponden a **input gate, forget gate, cell y output gate**.

#### Variantes de Keras LSTM

##### Time major

Los usuarios pueden seleccionar time-major o no time-major. Keras LSTM añade un atributo time-major en la función def attributes. Para la secuencia unidireccional LSTM, podemos simplemente mapear al atributo [time major](https://github.com/tensorflow/tensorflow/blob/35a3ab91b42503776f428bda574b74b9a99cd110/tensorflow/compiler/mlir/lite/ir/tfl_ops.td#L3902) de unidirecional_sequence_lstm.

##### LSTM bidireccional

La LSTM bidireccional se puede implementar con dos capas LSTM de Keras, una para adelante y otra para atrás, vea los ejemplos [aquí](https://github.com/tensorflow/tensorflow/blob/35a3ab91b42503776f428bda574b74b9a99cd110/tensorflow/python/keras/layers/wrappers.py#L382). Una vez que vemos el atributo go_backward, lo reconocemos como LSTM hacia atrás, entonces agrupamos las LSTM hacia delante y hacia atrás. **Esto aún está por desarrollarse.** Actualmente, esto crea dos operaciones UnidirectionalSequenceLSTM en el modelo TensorFlow Lite.

### Ejemplos de conversión LSTM definidos por el usuario

TensorFlow Lite también ofrece una forma de convertir implementaciones de LSTM definidas por el usuario. Aquí se usa el LSTM de Lingvo como ejemplo de cómo se puede implementar. Para más detalles, consulte la interfaz [lingvo.LSTMCellSimple](https://github.com/tensorflow/lingvo/blob/91a4609dbc2579748a95110eda59c66d17c594c5/lingvo/core/rnn_cell.py#L228) y la lógica de conversión [aquí](https://github.com/tensorflow/tensorflow/blob/82abf0dbf316526cd718ae8cd7b11cfcb805805e/tensorflow/compiler/mlir/lite/transforms/prepare_composite_functions_tf.cc#L130). También proporcionamos un ejemplo para otra de las definiciones LSTM de Lingvo en [lingvo.LayerNormalizedLSTMCellSimple interface](https://github.com/tensorflow/lingvo/blob/91a4609dbc2579748a95110eda59c66d17c594c5/lingvo/core/rnn_cell.py#L1173) y su lógica de conversión [aquí](https://github.com/tensorflow/tensorflow/blob/c11d5d8881fd927165eeb09fd524a80ebaf009f2/tensorflow/compiler/mlir/lite/transforms/prepare_composite_functions_tf.cc#L137).

## "Aporte su propia RNN de TensorFlow" a TensorFlow Lite

Si la interfaz RNN de un usuario es diferente de las soportadas por los estándares, existen un par de opciones:

**Opción 1:** Escribir código adaptador en Python de TensorFlow para adaptar la interfaz RNN a la interfaz RNN de Keras. Esto significa una tf.function con [tf_implements annotation](https://github.com/tensorflow/community/pull/113) en la función de la interfaz RNN generada que sea idéntica a la generada por la capa LSTM de Keras. Después de esto, funcionará la misma API de conversión usada para Keras LSTM.

**Opción 2:** Si lo anterior no es posible (por ejemplo, al LSTM de Keras le falta alguna funcionalidad que actualmente está expuesta por la op LSTM fusionada de TensorFlow Lite como la normalización de capas), entonces extienda el conversor de TensorFlow Lite escribiendo código de conversión personalizado e insértelo en la pasada MLIR de prepare-composite-functions [aquí](https://github.com/tensorflow/tensorflow/blob/c11d5d8881fd927165eeb09fd524a80ebaf009f2/tensorflow/compiler/mlir/lite/transforms/prepare_composite_functions_tf.cc#L115). La interfaz de la función debe tratarse como un contrato de API y debe contener los argumentos necesarios para convertirlas a operaciones LSTM de TensorFlow Lite fusionadas, es decir, entrada, sesgo, ponderaciones, proyección, normalización de capas, etc. Sería preferible que los tensores pasados como argumentos a esta función tuvieran un rango conocido (es decir, RankedTensorType en MLIR). Esto facilita mucho la escritura de código de conversión que pueda asumir estos tensores como RankedTensorType y ayude a transformarlos en tensores clasificados correspondientes a los operandos del operador TensorFlow Lite fusionado.

Un ejemplo completo de este flujo de conversión es la conversión de LSTMCellSimple a TensorFlow Lite de Lingvo.

En Lingvo, se define [aquí](https://github.com/tensorflow/lingvo/blob/91a4609dbc2579748a95110eda59c66d17c594c5/lingvo/core/rnn_cell.py#L228) la LSTMCellSimple. Los modelos entrenados con esta célula LSTM pueden convertirse a TensorFlow Lite de la siguiente manera:

1. Encapsule todos los usos de LSTMCellSimple en una tf.function con una anotación tf_implements que esté etiquetada como tal (por ejemplo, un buen nombre de anotación sería lingvo.LSTMCellSimple). Asegúrese de que la función tf.function que se genera coincide con la interfaz de la función esperada en el código de conversión. Se trata de un contrato entre el autor del modelo que añade la anotación y el código de conversión.

2. Amplíe la pasada prepare-composite-functions para incorporar una conversión de op compuesta personalizada a op LSTM fusionada de TensorFlow Lite. Ver el código de conversión [LSTMCellSimple](https://github.com/tensorflow/tensorflow/blob/82abf0dbf316526cd718ae8cd7b11cfcb805805e/tensorflow/compiler/mlir/lite/transforms/prepare_composite_functions_tf.cc#L130).

    El contrato de conversión:

3. Los tensores **Weight** y **proyection** se transponen.

4. De las **{input, recurrent}** a **{cell, input gate, forget gate, output gate}** se extraen seccionando el tensor de ponderación transpuesto.

5. De los **{bias}** a **{cell, input gate, forget gate, output gate}** se extraen seccionando el tensor de sesgo.

6. La **projection** se extrae seccionando el tensor de proyección transpuesto.

7. Una conversión similar se escribe para [LayerNormalizedLSTMCellSimple](https://github.com/tensorflow/tensorflow/blob/c11d5d8881fd927165eeb09fd524a80ebaf009f2/tensorflow/compiler/mlir/lite/transforms/prepare_composite_functions_tf.cc#L137).

8. El resto de la infraestructura de conversión de TensorFlow Lite, incluyendo todas [las pasadas MLIR](https://github.com/tensorflow/tensorflow/blob/35a3ab91b42503776f428bda574b74b9a99cd110/tensorflow/compiler/mlir/lite/tf_tfl_passes.cc#L57) definidas así como la exportación final a flatbuffer de TensorFlow Lite pueden ser reutilizadas.

## Problemas/limitaciones conocidos

1. Actualmente sólo hay soporte para convertir Keras LSTM sin estado (comportamiento predeterminado en Keras). Se está trabajando en la conversión de Keras LSTM con estado.
2. Todavía es posible modelar una capa Keras LSTM con estado usando la capa Keras LSTM sin estado subyacente y administrando el estado explícitamente en el programa de usuario. Tal programa TensorFlow todavía se puede convertir a TensorFlow Lite usando la característica que se describe aquí.
3. El LSTM bidireccional se modela actualmente como dos operaciones UnidirectionalSequenceLSTM en TensorFlow Lite. Esto se reemplazará por una única op BidirectionalSequenceLSTM.
