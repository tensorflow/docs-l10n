# Combinação de operações do TensorFlow

## Visão geral

Esta etapa descreve o design e as etapas necessárias para converter operações compostas no TensorFlow em fused operations (operações combinadas) no TensorFlow Lite. Essa infraestrutura tem uma finalidade geral e oferece suporte à conversão de qualquer operação composta no TensorFlow em uma operação combinada correspondente no TensorFlow Lite.

Um exemplo de uso dessa infraestrutura é a combinação de operações de RNN do TensorFlow no TensorFlow Lite, conforme detalhado [aqui](https://www.tensorflow.org/lite/models/convert/rnn).

### O que são operações combinadas

![drawing](../../images/convert/op_fusion_banner.jpg)

As operações do TensorFlow podem ser primitivas, como [tf.add](https://www.tensorflow.org/api_docs/python/tf/math/add), ou podem ser compostas a partir de outras operações primitivas, como [tf.einsum](https://www.tensorflow.org/api_docs/python/tf/einsum). Uma operação primitiva é exibida como um único nó no grafo do TensorFlow, enquanto uma operação composta é uma coleção de nós no grafo do TensorFlow. Executar uma operação composta é equivalente a executar cada uma de suas operações primitivas.

Uma operação combinada corresponde a uma única operação que agrupa toda a computação realizada por cada operação primitiva dentro da operação composta correspondente.

### Vantagens das operações combinadas

As operações combinadas existem para maximizar o desempenho das implementações de kernel subjacentes por meio da otimização das computações gerais e da redução do volume de memória. Isso traz muitas vantagens, especialmente para cargas de trabalho de inferência de baixa latência e para plataformas móveis com restrição de recursos.

As operações combinadas também oferecem uma interface de alto nível para definir transformações complexas, como quantização, que seriam impraticáveis ou muito difíceis de fazer em um nível mais granular.

O TensorFlow Lite tem diversas instâncias de operações combinadas pelos motivos indicados acima. Geralmente, essas operações combinadas correspondem a operações compostas no programa fonte do TensorFlow. Exemplos de operações compostas no TensorFlow que são implementadas como uma única operação combinada no TensorFlow Lite incluem diversas operações de RNN, como LSTM de sequência unidirecional ou bidirecional, convolução (conv2d, bias add, relu), totalmente conectadas (matmul, bias add, relu) e muitas outras. No TensorFlow Lite, a quantização de LSTM só é implementada nas operações de LSTM combinadas, no momento.

### Desafios das operações combinadas

Converter operações compostas do TensorFlow em operações combinadas do TensorFlow Lite é um problema difícil, pois:

1. As operações compostas são representadas no grafo do TensorFlow como um conjunto de operações primitivas sem uma fronteira bem definida. Pode ser desafiador identificar (por exemplo, pela correspondência de padrões) o subgrafo correspondente a uma operação composta.

2. Pode haver mais de uma implementação do TensorFlow usando uma operação combinada do TensorFlow Lite. Por exemplo: existem diversas implementações de LSTM no TensorFlow (Keras, Babelfish/lingvo, etc.), e cada uma delas é composta por diferentes operações primitivas, mas todas podem ser convertidas na mesma operação de LSTM combinada no TensorFlow Lite.

Portanto, a conversão de operações combinadas se mostrou bastante desafiadora.

## Como converter uma operação composta em uma operação personalizada do TF Lite (recomendado)

### Encapsule a operação composta em uma `tf.function`

Em diversos casos, uma parte do modelo pode ser mapeada em uma única operação no TF Lite, o que pode ajudar com o desempenho ao criar uma implementação otimizada de operações específicas. Para poder criar uma operação combinada no TF Lite, identifique a parte do grafo que representa uma operação combinada e encapsule-a em uma `tf.function` com o atributo "experimental_implements" para uma `tf.function`, que tem o atributo `tfl_fusable_op` com valor `true`. Se a operação personalizada aceitar atributos, passe-os como parte do mesmo "experimental_implements".

Exemplo:

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

Você não precisa definir `allow_custom_ops` no conversor, pois o atributo `tfl_fusable_op` já pressupõe isso.

### Implemente a operação personalizada e registre no interpretador do TFLite

Implemente a operação combinada como uma operação personalizada do TF Lite – confira as [instruções](https://www.tensorflow.org/lite/guide/ops_custom).

O nome para registrar a operação deve ser similar ao nome especificado no atributo `name` na assinatura implements.

Veja um exemplo dessa operação:

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

## Como converter uma operação composta em uma combinada (avançado)

Veja abaixo a arquitetura geral para converter operações compostas do TensorFlow em operações combinadas do TensorFlow Lite.

![drawing](../../images/convert/op_fusion.png)

### Encapsule a operação composta em uma `tf.function`

No código-fonte do modelo do TensorFlow, identifique e abstraia a operação personalizada em uma `tf.function` com a anotação de função [experimental_implements](https://github.com/tensorflow/tensorflow/blob/c11d5d8881fd927165eeb09fd524a80ebaf009f2/tensorflow/python/eager/def_function.py#L470). Confira um exemplo de [pesquisa de embedding](#composing_ops). A função define a interface, e seus argumentos devem ser usados para implementar a lógica de conversão.

### Escreva o código de conversão

O código de conversão é escrito para cada interface da função com a anotação `implements`. Confira um exemplo de combinação para [pesquisa de embedding](#fusion_code). Conceitualmente, o código de conversão substitui a implementação composta dessa interface pela combinada.

Na etapa prepare-composite-functions "preparar funções compostas", adicione seu [código de conversão](https://github.com/tensorflow/tensorflow/blob/c11d5d8881fd927165eeb09fd524a80ebaf009f2/tensorflow/compiler/mlir/lite/transforms/prepare_composite_functions_tf.cc#L115).

Em casos mais avançados, é possível implementar transformações complexas dos operandos da operação composta para derivar os operandos da operação combinada. Confira o código de conversão de [LSTM do Keras](https://github.com/tensorflow/tensorflow/blob/1099faa8d6a941ef44d09ed8c372ff0ffda94112/tensorflow/compiler/mlir/lite/utils/lstm_utils.cc#L627) como exemplo.

### Converta para o TensorFlow Lite

Use a API [TFLiteConverter.from_saved_model](https://www.tensorflow.org/api_docs/python/tf/lite/TFLiteConverter#from_saved_model) para converter para o TensorFlow Lite.

## Nos bastidores

<a id="under_the_hood"></a>

Agora vamos descrever os detalhes de alto nível do design geral da conversão em operações combinadas no TensorFlow Lite.

### Como compor operações no TensorFlow

<a id="composing_ops"></a>

Ao usar `tf.function` com o atributo de função [experimental_implements](https://github.com/tensorflow/tensorflow/blob/c11d5d8881fd927165eeb09fd524a80ebaf009f2/tensorflow/python/eager/def_function.py#L470), os usuários podem compor explicitamente novas operações usando as operações primitivas do TensorFlow e especificar a interface que a operação composta relevante implementa. Isso é muito útil, pois:

1. Fornece uma fronteira bem definida para a operação composta no grafo do TensorFlow subjacente.
2. Especifica explicitamente a interface que essa operação implementa. Os argumentos de `tf.function` correspondem aos argumentos dessa interface.

Para fins de exemplo, vamos considerar uma operação composta definida para implementar pesquisa de embedding. Ela é mapeada para uma operação combinada no TensorFlow Lite.

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

Ao fazer os modelos usarem operações compostas via `tf.function` conforme ilustrado acima, é possível criar uma estrutura geral para **identificar e converter** essas operações em operações combinadas do TensorFlow Lite.

### Como estender o conversor do TensorFlow Lite

O conversor do TensorFlow Lite lançado anteriormente neste ano só oferecia suporte à importação de modelos do TensorFlow como um grafo com todas as variáveis substituídas por seus valores constantes correspondentes, o que não funciona para a combinação de operações, já que esses grafos têm todas as funções embutidas para que as variáveis possam ser transformadas em constantes.

Para poder usar `tf.function` com o recurso `experimental_implements` durante o processo de conversão, as funções precisam ser preservadas até as etapas posteriores do processo de conversão.

Dessa forma, implementamos um novo workflow de importação e conversão de modelos do TensorFlow no conversor para oferecer suporte à combinação de operações compostas. Especificamente, os novos recursos adicionados são:

1. Importação de [SavedModels do TensorFlow na MLIR](https://github.com/tensorflow/tensorflow/blob/1099faa8d6a941ef44d09ed8c372ff0ffda94112/tensorflow/compiler/mlir/tensorflow/translate/import_model.cc#L3748)
2. [Combinação de operações compostas](https://github.com/tensorflow/tensorflow/blob/1099faa8d6a941ef44d09ed8c372ff0ffda94112/tensorflow/compiler/mlir/lite/transforms/prepare_composite_functions_tf.cc#L103)
3. [Análise de mutabilidade variável](https://github.com/tensorflow/tensorflow/blob/1099faa8d6a941ef44d09ed8c372ff0ffda94112/tensorflow/compiler/mlir/tensorflow/transforms/optimize_global_tensors.cc#L43)
4. [Congelamento de todas as variáveis somente leitura](https://github.com/tensorflow/tensorflow/blob/1099faa8d6a941ef44d09ed8c372ff0ffda94112/tensorflow/compiler/mlir/tensorflow/transforms/freeze_global_tensors.cc#L44)

Dessa forma, podemos fazer a combinação de operações usando as funções que representam as operações compostas antes de as funções serem embutidas e de as variáveis serem congeladas.

### Como implementar a combinação de operações

Vamos avaliar a etapa de combinação de operações com maiores detalhes. Essa etapa faz o seguinte:

1. Percorre em um loop todas as funções no módulo MLIR.
2. Se uma função tiver o atributo tf._implements – com base no valor do atributo, chama o utilitário de combinação de operações apropriado.
3. O utilitário de combinação de operações trabalha com os operandos e atributos da função (que servem como a interface para a conversão) e substitui o corpo da função por um corpo de função equivalente que contém a operação combinada.
4. Em diversos casos, o corpo substituído conterá operações diferentes da operação combinada, que correspondem a algumas transformações estáticas dos operandos da função para obter os operandos da operação combinada. Como essas computações podem sofrer constant-folding, não estariam presentes no Flatbuffer exportado, em que somente a operação combinada existiria.

Aqui está o trecho de código da etapa, mostrando o workflow principal:

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

Aqui está o trecho de código mostrando o mapeamento dessa operação composta para uma operação combinada no TensorFlow Lite usando a função como uma interface de conversão.

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
