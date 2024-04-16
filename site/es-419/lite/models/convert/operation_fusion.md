# Fusión de operaciones TensorFlow

## Visión general

Esta página describe el diseño y los pasos necesarios para convertir operaciones compuestas en TensorFlow a operaciones fusionadas en TensorFlow Lite. Esta infraestructura es de propósito general y soporta la conversión de cualquier operación compuesta en TensorFlow a una operación fusionada correspondiente en TensorFlow Lite.

Un ejemplo de uso de esta infraestructura es la fusión de operaciones TensorFlow RNN a TensorFlow Lite, como se detalla [aquí](https://www.tensorflow.org/lite/models/convert/rnn).

### Qué son las operaciones fusionadas

![imagen](../../images/convert/op_fusion_banner.jpg)

Las operaciones TensorFlow pueden ser ops primitivas, por ejemplo [tf.add](https://www.tensorflow.org/api_docs/python/tf/math/add) o pueden estar compuestas a partir de otras operaciones primitivas, por ejemplo [tf.einsum](https://www.tensorflow.org/api_docs/python/tf/einsum). Una operación primitiva se muestra como un único nodo en el grafo TensorFlow mientras que.una operación compuesta es una colección de nodos en el grafo TensorFlow. Ejecutar una operación compuesta es equivalente a ejecutar cada una de sus operaciones primitivas constituyentes.

Una operación fusionada corresponde a una operación única que subsume todo el cálculo realizado por cada operación primitiva dentro de la operación compuesta correspondiente.

### Beneficios de las operaciones fusionadas

Las operaciones fusionadas existen para maximizar el rendimiento de sus implementaciones del kernel subyacente, optimizando el cálculo global y reduciendo la huella de memoria. Esto es muy valioso, especialmente para cargas de trabajo de inferencia de baja latencia y plataformas móviles con recursos limitados.

Las operaciones fusionadas también ofrecen una interfaz de alto nivel para definir transformaciones complejas como la cuantización, que de otro modo serían inviables o muy difíciles de realizar a un nivel más granular.

TensorFlow Lite tiene muchos casos de operaciones fusionadas por las razones expuestas anteriormente. Estas operaciones fusionadas corresponden típicamente a operaciones compuestas en el programa TensorFlow fuente. Ejemplos de operaciones compuestas en TensorFlow que se implementan como una única operación fusionada en TensorFlow Lite incluyen varias operaciones RNN como la secuencia unidireccional y bidireccional LSTM, convolución (conv2d, bias add, relu), totalmente conectado (matmul, bias add, relu) y más. En TensorFlow Lite, la cuantización LSTM sólo se implementa actualmente en las operaciones LSTM fusionadas.

### Problemas con las operaciones fusionadas

Convertir operaciones compuestas de TensorFlow en operaciones fusionadas en TensorFlow Lite es un reto difícil. Esto se debe a que:

1. Las operaciones compuestas se representan en el grafo TensorFlow como un conjunto de operaciones primitivas sin un límite bien definido. Puede resultar muy difícil identificar (por ejemplo, mediante la concordancia de patrones) el subgrafo correspondiente a una operación compuesta de este tipo.

2. Puede haber más de una implementación de TensorFlow que apunte a un operario de TensorFlow Lite fusionado. Por ejemplo, hay muchas implementaciones de LSTM en TensorFlow (Keras, Babelfish/lingvo, etc.) y cada una de ellas se compone de diferentes operaciones primitivas, pero aun así todas podrían convertirse en la misma operación LSTM fusionada en TensorFlow Lite.

Por ello, la conversión de operaciones fusionadas ha resultado ser todo un reto.

## Conversión de una op compuesta a una operación personalizada TFLite (recomendado)

### Encapsular la operación compuesta en una `tf.function`

En muchos casos, alguna parte del modelo puede mapearse a una única operación en TFLite. Esto puede ayudar al rendimiento cuando se escribe una implementación optimizada para operaciones específicas. Para poder crear una operación fusionada en TFLite, identifique la parte del grafo que representa una operación fusionada y encapsúlela en una `tf.function` con atributo "experimental_implements" a una `tf.function`, que tiene valor de atributo `tfl_fusable_op` con valor `true`. Si la operación personalizada toma atributos, páselos como parte del mismo "experimental_implements".

Ejemplo:

```python
def get_implements_signature():
  implements_signature = [
    # 'name' will be used as a name for the operation.
    'name: "my_custom_fused_op"',
    # attr "tfl_fusable_op" is required to be set with true value.
    'attr {key: "tfl_fusable_op" value { b: true } }',
    # Example attribute "example_option" that the op accepts.
    'attr {key: "example_option" value { i: %d } }' % 10
  ]
  return ' '.join(implements_signature)

@tf.function(experimental_implements=get_implements_signature())
def my_custom_fused_op(input_1, input_2):
  # An empty function that represents pre/post processing example that
  # is not represented as part of the Tensorflow graph.
  output_1 = tf.constant(0.0, dtype=tf.float32, name='first_output')
  output_2 = tf.constant(0.0, dtype=tf.float32, name='second_output')
  return output_1, output_2

class TestModel(tf.Module):
  def __init__(self):
    super(TestModel, self).__init__()
    self.conv_1 = tf.keras.layers.Conv2D(filters=1, kernel_size=(3, 3))
    self.conv_2 = tf.keras.layers.Conv2D(filters=1, kernel_size=(3, 3))

  @tf.function(input_signature=[
      tf.TensorSpec(shape=[1, 28, 28, 3], dtype=tf.float32),
      tf.TensorSpec(shape=[1, 28, 28, 3], dtype=tf.float32),
  ])
  def simple_eval(self, input_a, input_b):
    return my_custom_fused_op(self.conv_1(input_a), self.conv_2(input_b))
```

Tenga en cuenta que no necesita configurar `allow_custom_ops` en el conversor ya que el atributo `tfl_fusable_op` ya lo implica.

### Implementar op y registro personalizados con el intérprete TFLite

Implemente su operación fusionada como una operación TFLite personalizada - vea las [instrucciones](https://www.tensorflow.org/lite/guide/ops_custom).

Tenga en cuenta que, el nombre con el que registrar la op debe ser similar al nombre especificado en el atributo `name` de la firma implements.

Un ejemplo para la op del ejemplo es

```c++
  TfLiteRegistration reg = {};
  // This name must match the name specified in the implements signature.
  static constexpr char kOpName[] = "my_custom_fused_op";
  reg.custom_name = kOpName;
  reg.prepare = [](TfLiteContext* context, TfLiteNode* node) -> TfLiteStatus {
    // Add your code.
    return kTfLiteOk;
  };
  reg.invoke = [](TfLiteContext* context, TfLiteNode* node) -> TfLiteStatus {
    // Add your code.
    return kTfLiteOk;
  };
  reg.builtin_code = kTfLiteCustom;
  resolver->AddCustom(kOpName, &reg);
```

## Convertir de operación compuesta a fusionada (Avanzado)

Más abajo se muestra la arquitectura general para convertir las operaciones compuestas de TensorFlow en operaciones fusionadas de TensorFlow Lite:

![imagen](../../images/convert/op_fusion.png)

### Encapsular la operación compuesta en una `tf.function`

En el código fuente del modelo TensorFlow, identifique y abstraiga la operación compuesta en una `tf.function` con la anotación de función [experimental_implements](https://github.com/tensorflow/tensorflow/blob/c11d5d8881fd927165eeb09fd524a80ebaf009f2/tensorflow/python/eager/def_function.py#L470). Véase un ejemplo de [búsqueda de incorporaciones](#composing_ops). La función define la interfaz y sus argumentos deben usarse para implementar la lógica de conversión.

### Escribir el código de conversión

El código de conversión se escribe por la interfaz de la función con la anotación `implements`. Puede ver un ejemplo de fusión para la [búsqueda de incorporaciones](#fusion_code). Conceptualmente, el código de conversión reemplaza la implementación compuesta de esta interfaz por la fusionada.

En la pasada preparar-componer-funciones, introduzca su [código de conversión](https://github.com/tensorflow/tensorflow/blob/c11d5d8881fd927165eeb09fd524a80ebaf009f2/tensorflow/compiler/mlir/lite/transforms/prepare_composite_functions_tf.cc#L115).

En usos más avanzados, se pueden implementar transformaciones complejas de los operarios de la operación compuesta para derivar los operarios de la operación fusionada. Véase el código de transformación de las [LSTM Keras](https://github.com/tensorflow/tensorflow/blob/1099faa8d6a941ef44d09ed8c372ff0ffda94112/tensorflow/compiler/mlir/lite/utils/lstm_utils.cc#L627). como ejemplo.

### Convertir a TensorFlow Lite

Use la API [TFLiteConverter.from_saved_model](https://www.tensorflow.org/api_docs/python/tf/lite/TFLiteConverter#from_saved_model) para convertir a TensorFlow Lite.

## En el fondo

<a id="under_the_hood"></a>

Seguidamente describimos los detalles de alto nivel del diseño general en la conversión a operaciones fusionadas en TensorFlow Lite.

### Componer operaciones en TensorFlow

<a id="composing_ops"></a>

El uso de `tf.function` con el atributo de función [experimental_implements](https://github.com/tensorflow/tensorflow/blob/c11d5d8881fd927165eeb09fd524a80ebaf009f2/tensorflow/python/eager/def_function.py#L470) permite a los usuarios componer explícitamente nuevas operaciones utilizando operaciones primitivas de TensorFlow y especificar la interfaz que implementa la operación compuesta resultante. Esto es muy útil ya que proporciona:

1. Un límite bien definido para la operación compuesta en el grafo TensorFlow subyacente.
2. Especifique explícitamente la interfaz que implementa esta operación. Los argumentos de `tf.function` corresponden a los argumentos de esta interfaz.

Como ejemplo, consideremos una operación compuesta definida para implementar la búsqueda de incorporaciones. Esto mapea a una operación fusionada en TensorFlow Lite.

```python
  @tf.function(
        experimental_implements="embedding_lookup")
    def EmbFprop(embs, ids_vec):
      """Embedding forward prop.

      Effectively, it computes:
        num = size of ids_vec
        rets = zeros([num, embedding dim])
        for i in range(num):
          rets[i, :] = embs[ids_vec[i], :]
        return rets

      Args:
        embs: The embedding matrix.
        ids_vec: A vector of int32 embedding ids.

      Returns:
        The result of embedding lookups. A matrix of shape
        [num ids in ids_vec, embedding dims].
      """
      num = tf.shape(ids_vec)[0]
      rets = inplace_ops.empty([num] + emb_shape_suf, py_utils.FPropDtype(p))

      def EmbFpropLoop(i, embs, ids_vec, rets):
        # row_id = ids_vec[i]
        row_id = tf.gather(ids_vec, i)
        # row = embs[row_id]
        row = tf.reshape(tf.gather(embs, row_id), [1] + emb_shape_suf)
        # rets[i] = row
        rets = inplace_ops.alias_inplace_update(rets, [i], row)
        return embs, ids_vec, rets

      _, _, rets = functional_ops.For(
          start=0,
          limit=num,
          delta=1,
          inputs=[embs, ids_vec, rets],
          body=EmbFpropLoop,
          rewrite_with_while=compiled)
      if len(weight_shape) > 2:
        rets = tf.reshape(rets, [num, symbolic.ToStatic(p.embedding_dim)])
      return rets
```

Al hacer que los modelos usen operaciones compuestas a través de `tf.function` como se ilustra más arriba, se hace posible construir una infraestructura general para **identificar y convertir** tales operaciones en operaciones fusionadas de TensorFlow Lite.

### Extender el convertidor TensorFlow Lite

El convertidor TensorFlow Lite que se lanzó a principios de este año sólo soportaba la importación de modelos TensorFlow como un grafo con todas las variables repuestas con sus correspondientes valores constantes. Esto no funciona para la fusión de operaciones, ya que dichos grafos tienen todas las funciones incorporadas para que las variables puedan convertirse en constantes.

Para aprovechar la función `tf.function` con la función `experimental_implements` durante el proceso de conversión, es necesario conservar las funciones hasta más adelante en el proceso de conversión.

Como tal, implementamos un nuevo flujo de trabajo de importación y conversión de modelos TensorFlow en el convertidor para apoyar el caso de uso de fusión de operaciones compuestas. En concreto, las nuevas funciones añadidas son:

1. Importar modelos TensorFlow [guardados en MLIR](https://github.com/tensorflow/tensorflow/blob/1099faa8d6a941ef44d09ed8c372ff0ffda94112/tensorflow/compiler/mlir/tensorflow/translate/import_model.cc#L3748)
2. [fusionar operaciones compuestas](https://github.com/tensorflow/tensorflow/blob/1099faa8d6a941ef44d09ed8c372ff0ffda94112/tensorflow/compiler/mlir/lite/transforms/prepare_composite_functions_tf.cc#L103)
3. [análisis de mutabilidad variable](https://github.com/tensorflow/tensorflow/blob/1099faa8d6a941ef44d09ed8c372ff0ffda94112/tensorflow/compiler/mlir/tensorflow/transforms/optimize_global_tensors.cc#L43)
4. [congelar todas las variables de sólo lectura](https://github.com/tensorflow/tensorflow/blob/1099faa8d6a941ef44d09ed8c372ff0ffda94112/tensorflow/compiler/mlir/tensorflow/transforms/freeze_global_tensors.cc#L44)

Esto nos permite realizar la fusión de operaciones usando las funciones que representan las operaciones compuestas antes de la incorporación de funciones y la congelación de variables.

### Implementar la fusión de operaciones

Veamos con más detalle el pase de fusión de operaciones. Este pase hace lo siguiente:

1. Recorrir en bucle todas las funciones del módulo MLIR.
2. Si una función tiene el atributo tf._implements, basándose en el valor del atributo, llama a la utilidad de fusión de operaciones adecuada.
3. La utilidad de fusión de operaciones actúa sobre los operandos y atributos de la función (que sirven de interfaz para la conversión) y sustituye el cuerpo de la función por un cuerpo de función equivalente que contiene la operación fusionada.
4. En muchos casos, el cuerpo repuesto contendrá operaciones distintas de la operación fusionada. Corresponden a algunas transformaciones estáticas sobre los operandos de la función para obtener los operandos de la operación fusionada. Dado que todos estos cómputos pueden plegarse de forma constante, no estarían presentes en el flatbuffer exportado, donde sólo existiría la operación fusionada.

Este es un fragmento de código del pase que muestra el flujo de trabajo principal:

```
void PrepareCompositeFunctionsPass::ConvertTFImplements(FuncOp func,
                                                        StringAttr attr) {
  if (attr.getValue() == "embedding_lookup") {
    func.eraseBody();
    func.addEntryBlock();
    // Convert the composite embedding_lookup function body to a
    // TFLite fused embedding_lookup op.
    ConvertEmbeddedLookupFunc convert_embedded_lookup(func);
    if (failed(convert_embedded_lookup.VerifySignature())) {
      return signalPassFailure();
    }
    convert_embedded_lookup.RewriteFunc();
  } else if (attr.getValue() == mlir::TFL::kKerasLstm) {
     func.eraseBody();
     func.addEntryBlock();
     OpBuilder builder(func.getBody());
     if (failed(ConvertKerasLSTMLayer(func, &builder))) {
       return signalPassFailure();
     }
  } else if (.....) /* Other fusions can plug in here */
}
```

Aquí tiene un fragmento de código que muestra cómo mapear esta operación compuesta a una operación fusionada en TensorFlow Lite aprovechando la función como interfaz de conversión.

<a id="fusion_code"></a>

```c++
void RewriteFunc() {
    Value lookup = func_.getArgument(1);
    Value value = func_.getArgument(0);
    auto output_type = func_.getType().getResult(0);

    OpBuilder builder(func_.getBody());
    auto op = builder.create<mlir::TFL::EmbeddingLookupOp>(
        func_.getLoc(), output_type, lookup, value);

    builder.create<mlir::ReturnOp>(func_.getLoc(), op.getResult());
  }
```
