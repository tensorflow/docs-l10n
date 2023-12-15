# Quantização pós-treinamento

A quantização pós-treinamento inclui técnicas gerais para reduzir a latência do acelerador de hardware e da CPU, o processamento, a energia e o tamanho do modelo com pouca degradação da exatidão do modelo. Essas técnicas podem ser executadas em um modelo float do TensorFlow já treinado e aplicadas durante a conversão para o TensorFlow Lite. Elas estão disponíveis como opções no [conversor do TensorFlow Lite](https://www.tensorflow.org/lite/convert/).

Para pular direto para os exemplos completos, veja os seguintes tutoriais:

- [Quantização de intervalo dinâmico pós-treinamento](https://www.tensorflow.org/lite/performance/post_training_quant)
- [Quantização de números inteiros pós-treinamento](https://www.tensorflow.org/lite/performance/post_training_integer_quant)
- [Quantização float16 pós-treinamento](https://www.tensorflow.org/lite/performance/post_training_float16_quant)

## Quantização de pesos

Os pesos podem ser convertidos em tipos com precisão reduzida, como floats de 16 bits ou números inteiros de 8 bits. Geralmente, recomendamos floats de 16 bits para a aceleração de GPU e números inteiros de 8 bits para a execução da CPU.

Por exemplo, especifique a quantização de pesos de números inteiros de 8 bits desta forma:

```
import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant_model = converter.convert()
```

Durante a inferência, as partes mais criticamente intensas são computadas com 8 bits em vez de ponto flutuante. Há overhead de desempenho no tempo de inferência, relativo à quantização de ambos os pesos e as ativações abaixo.

Para mais informações, consulte o guia de [quantização pós-treinamento](https://www.tensorflow.org/lite/performance/post_training_quantization) do TensorFlow Lite.

## Quantização de números inteiros de pesos e ativações

Melhore a latência, o processamento e o uso de energia, além de obter acesso aos aceleradores de hardware somente números inteiros, ao assegurar a quantização de ambos os pesos e as ativações. Isso exige um pequeno dataset representativo.

```
import tensorflow as tf

def representative_dataset_gen():
  for _ in range(num_calibration_steps):
    # Get sample input data as a numpy array in a method of your choosing.
    yield [input]

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
tflite_quant_model = converter.convert()
```

O modelo resultante ainda aceitará entradas e a saídas float por conveniência.

Para mais informações, consulte o guia de [quantização pós-treinamento](https://www.tensorflow.org/lite/performance/post_training_quantization#full_integer_quantization_of_weights_and_activations) do TensorFlow Lite.
