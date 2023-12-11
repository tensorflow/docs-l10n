# Quantização pós-treinamento

A quantização pós-treinamento é uma técnica de conversão que pode reduzir o tamanho do modelo e, ao mesmo tempo, melhorar a latência do acelerador de hardware e da CPU, com pouca degradação da exatidão do modelo. Você pode quantizar um modelo float já treinado do TensorFlow ao convertê-lo para o formato TensorFlow Lite usando o [Conversor do TensorFlow Lite](../models/convert/).

Observação: os procedimentos nesta página exigem o TensorFlow 1.15 ou mais recente.

### Métodos de otimização

Há várias opções de quantização pós-treinamento disponíveis. Veja esta tabela com um resumo das alternativas e dos benefícios que elas oferecem:

Técnica | Benefícios | Hardware
--- | --- | ---
Intervalo dinâmico | 4x menor, speedup de 2-3x | CPU
: quantização         :                           :                  : |  |
Números inteiros | 4x menor, speedup de 3x+ | CPU, Edge TPU,
: quantização         :                           : Microcontroladores : |  |
Quantização float16 | 2x menor, GPU | CPU, GPU
:                      : aceleração              :                  : |  |

A seguinte árvore de decisão pode ajudar a determinar qual método de quantização pós-treinamento é o melhor para seu caso de uso:

![opções de otimização pós-treinamento](images/optimization.jpg)

### Quantização de intervalo dinâmico

A quantização de intervalo dinâmico é um ponto de partida recomendado, porque possibilita um uso menor de memória e computações mais rápidas sem precisar fornecer um dataset representativo para fazer calibração. Esse tipo de quantização quantiza estaticamente somente os pesos, de ponto flutuante para inteiro no momento da conversão, o que proporciona a precisão de 8 bits:

<pre>
import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
&lt;b&gt;converter.optimizations = [tf.lite.Optimize.DEFAULT]&lt;/b&gt;
tflite_quant_model = converter.convert()
</pre>

Para reduzir a latência ainda mais durante a inferência, os operadores de "intervalo dinâmico" quantizam as ativações dinamicamente com base no intervalo até 8 bits e realizam computações com pesos e ativações de 8 bits. Essa otimização fornece latências próximas às inferências inteiramente de ponto fixo. No entanto, as saídas ainda são armazenadas usando o ponto flutuante, então a maior velocidade das operações de intervalo dinâmico é menor do que uma computação inteira de ponto fixo.

### Quantização de números inteiros

Você pode obter ainda mais melhorias na latência, reduções no pico de uso da memória e compatibilidade com dispositivos ou aceleradores de hardware somente números inteiros ao garantir que toda a matemática do modelo seja quantizada em números inteiros.

Para a quantização de números inteiros, você precisa calibrar ou estimar o intervalo, ou seja, (min, max) de todos os tensores de ponto flutuante no modelo. Ao contrário dos tensores constantes, como pesos e biases, os tensores variáveis, como entrada e saída do modelo e ativações (saídas de camadas intermediárias), não podem ser calibrados a menos que sejam realizados alguns ciclos de inferência. Como resultado, o conversor exige um dataset representativo para calibrá-los. Esse dataset pode ser um subset pequeno (cerca de 100 a 500 amostras) dos dados de treinamento ou validação. Consulte a função `representative_dataset()` abaixo.

A partir da versão 2.7 do TensorFlow, você pode especificar o dataset representativo através de uma [assinatura](../guide/signatures.ipynb), como neste exemplo:

<pre>
def representative_dataset():
  for data in dataset:
    yield {
      "image": data.image,
      "bias": data.bias,
    }
</pre>

Se houver mais de uma assinatura no modelo do TensorFlow, você pode especificar o dataset múltiplo ao especificar as chaves de assinatura:

<pre>
def representative_dataset():
  # Feed data set for the "encode" signature.
  for data in encode_signature_dataset:
    yield (
      "encode", {
        "image": data.image,
        "bias": data.bias,
      }
    )

  # Feed data set for the "decode" signature.
  for data in decode_signature_dataset:
    yield (
      "decode", {
        "image": data.image,
        "hint": data.hint,
      },
    )
</pre>

Você pode gerar o dataset representativo ao fornecer uma lista de tensores de entrada:

<pre>
def representative_dataset():
  for data in tf.data.Dataset.from_tensor_slices((images)).batch(1).take(100):
    yield [tf.dtypes.cast(data, tf.float32)]
</pre>

Desde a versão 2.7 do TensorFlow, recomendamos usar uma abordagem baseada na assinatura em vez de baseada na lista de tensores de entrada, porque a ordem dos tensores pode ser facilmente invertida.

Para fins de teste, você pode usar um dataset falso da seguinte maneira:

<pre>
def representative_dataset():
    for _ in range(100):
      data = np.random.rand(1, 244, 244, 3)
      yield [data.astype(np.float32)]
 </pre>

#### Números inteiros com fallback de float (usando a entrada/saída de float padrão)

Para fazer a quantização de números inteiros de um modelo, mas usar operadores float quando eles não tiverem uma implementação de números inteiros (para garantir que a conversão ocorra sem problemas), use as seguintes etapas:

<pre>
import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
&lt;b&gt;converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset&lt;/b&gt;
tflite_quant_model = converter.convert()
</pre>

Observação: esse `tflite_quant_model` não será compatível com dispositivos somente números inteiros (como microcontroladores de 8 bits) e aceleradores (como o Coral Edge TPU), porque a entrada e a saída ainda permanecem em float para ter a mesma interface que o modelo somente float original.

#### Somente números inteiros

*A criação de modelos somente números inteiros é um caso de uso comum no [TensorFlow Lite para microcontroladores](https://www.tensorflow.org/lite/microcontrollers) e [Coral Edge TPUs](https://coral.ai/).*

Observação: a partir do 2.3.0, oferecemos suporte aos atributos `inference_input_type` e `inference_output_type`.

Além disso, para garantir a compatibilidade com dispositivos somente números inteiros (como microcontroladores de 8 bits) e aceleradores (como o Coral Edge TPU), você pode aplicar a quantização de números inteiros a todas as operações, incluindo a entrada e a saída, ao seguir estes passos:

<pre>
import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
&lt;b&gt;converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]&lt;/b&gt;
&lt;b&gt;converter.inference_input_type = tf.int8&lt;/b&gt;  # or tf.uint8
&lt;b&gt;converter.inference_output_type = tf.int8&lt;/b&gt;  # or tf.uint8
tflite_quant_model = converter.convert()
</pre>

### Quantização float16

Você pode reduzir o tamanho de um modelo de ponto flutuante ao quantizar os pesos em float16, o padrão IEEE para números de ponto flutuante de 16 bits. Para ativar a quantização float16 dos pesos, siga estas etapas:

<pre>
import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
&lt;b&gt;converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]&lt;/b&gt;
tflite_quant_model = converter.convert()
</pre>

Estas são as vantagens da quantização float16:

- Ela reduz o tamanho do modelo pela metade (já que todos os pesos ficam a metade do tamanho original).
- Ela causa perda mínima de exatidão.
- Ela é compatível com alguns delegados (por exemplo, o delegado de GPU) que podem operar diretamente nos dados float16, resultando em uma execução mais rápida que as computações float32.

Estas são as desvantagens da quantização float16:

- Ela não reduz tanto a latência como a quantização de matemática de ponto fixo.
- Por padrão, um modelo quantizado float16 "desquantizará" os valores dos pesos para float32 quando executado na CPU. (Observe que o delegado de GPU não realizará essa desquantização, já que pode operar em dados float16.)

### Somente números inteiros: ativações de 16 bits com pesos de 8 bits (experimental)

Esse é um esquema de quantização experimental. É semelhante ao esquema "somente números inteiros", mas as ativações são quantizadas com base no intervalo até 16 bits, os pesos são quantizados em números inteiros de 18 bits e o bias é quantizado em números inteiros de 64 bits. Daqui em diante, isso será chamado de quantização 16x8.

O principal benefício dessa quantização é que ela pode melhorar a exatidão significativamente, mas só aumentar levemente o tamanho do modelo.

<pre>
import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.representative_dataset = representative_dataset
&lt;b&gt;converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]&lt;/b&gt;
tflite_quant_model = converter.convert()
</pre>

Se a quantização 16x8 não for compatível com alguns operadores no modelo, ele ainda poderá ser quantizado, mas os operadores incompatíveis são mantidos em float. A seguinte opção deve ser adicionada a target_spec para permitir isso.

<pre>
import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.representative_dataset = representative_dataset
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8,
&lt;b&gt;tf.lite.OpsSet.TFLITE_BUILTINS&lt;/b&gt;]
tflite_quant_model = converter.convert()
</pre>

Exemplos de casos de uso em que esse esquema de quantização oferece melhorias na exatidão:

- super-resolução,
- processamento de sinais de áudio, como cancelamento de ruído e beamforming,
- remoção de ruído de imagens,
- reconstrução em HDR a partir de uma única imagem.

As desvantagens dessa quantização são:

- No momento, a inferência é perceptivelmente mais lenta que os números inteiros de 8 bits devido à ausência de implementação de kernels otimizados.
- Atualmente, é incompatível com os delegados do TFLite acelerados de hardware existentes.

Observação: esse é um recurso experimental.

Encontre um tutorial para esse modelo quantizado [aqui](post_training_integer_quant_16x8.ipynb).

### Exatidão do modelo

Como os pesos são quantizados após o treinamento, pode haver uma perda na exatidão, principalmente para redes menores. Modelos totalmente quantizados pré-treinados são fornecidos para redes específicas no [TensorFlow Hub](https://tfhub.dev/s?deployment-format=lite&q=quantized){:.external}. É importante conferir a exatidão do modelo quantizado para verificar se qualquer degradação está dentro dos limites aceitáveis. Há ferramentas para avaliar a [exatidão de modelos do TensorFlow Lite](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/evaluation/tasks){:.external}.

Como alternativa, se a queda na exatidão for muito grande, considere usar o [treinamento consciente de quantização](https://www.tensorflow.org/model_optimization/guide/quantization/training). No entanto, para isso, é necessário fazer modificações durante o treinamento do modelo para adicionar nós de quantização falsos, enquanto as técnicas de quantização pós-treinamento nesta página usam um modelo pré-treinado existente.

### Representação para tensores quantizados

A quantização de 8 bits aproxima os valores de ponto flutuante usando a seguinte fórmula.

$$real_value = (int8_value - zero_point) \times scale$$

A representação tem duas partes:

- Os pesos por eixo (ou seja, por canal) ou por tensor são representados por dois valores complementares int8 no intervalo [-127, 127] com um ponto zero igual a 0.

- As ativações/entradas por tensor são representadas por dois valores complementares int8 no intervalo [-128, 127], com um ponto zero no intervalo [-128, 127].

Para uma visão detalhada do nosso esquema de quantização, confira nossa [especificação de quantização](./quantization_spec). Os fornecedores de hardware que quiserem se conectar à interface de delegados do TensorFlow Lite são incentivados a implementar o esquema de quantização descrito aqui.
