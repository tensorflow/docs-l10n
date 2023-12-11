# Otimização de modelo

Os dispositivos de borda geralmente têm memória ou poder computacional limitado. Várias otimizações podem ser aplicadas aos modelos para que eles possam ser executados nessas restrições. Além disso, algumas otimizações permitem o uso de hardware especializado para a inferência acelerada.

O TensorFlow Lite e o [Kit de ferramentas para otimização de modelo do TensorFlow](https://www.tensorflow.org/model_optimization) fornece ferramentas para reduzir a complexidade de otimizar a inferência.

É recomendável considerar a otimização do modelo durante o processo de desenvolvimento do seu aplicativo. Este documento descreve algumas práticas recomendadas para otimizar modelos do TensorFlow para a implantação de hardware de borda.

## Por que os modelos devem ser otimizados

A otimização de modelo pode ajudar com o desenvolvimento de aplicativos de várias maneiras.

### Redução de tamanho

Algumas formas de otimização podem ser usadas para reduzir o tamanho de um modelo. Modelos menores têm os seguintes benefícios:

- **Menor tamanho de armazenamento:** modelos menores ocupam menos espaço de armazenamento nos dispositivos dos seus usuários. Por exemplo, um aplicativo Android usando um modelo menor ocupará menos espaço de armazenamento no dispositivo móvel do usuário.
- **Menor tamanho de download:** modelos menores exigem menos tempo e largura de banda para serem baixados nos dispositivos dos usuários.
- **Menos uso da memória:** modelos menores usam menos RAM quando são executados, o que libera a memória para que seja usada por outras partes do seu aplicativo e pode resultar em melhor desempenho e estabilidade.

A quantização pode reduzir o tamanho de um modelo em todos esses casos, possivelmente à custa de um pouco de exatidão. O pruning e o clustering podem reduzir o tamanho de um modelo para download ao torná-lo mais facilmente compressível.

### Redução de latência

A *latência* é a quantidade de tempo que leva para executar uma única inferência com um determinado modelo. Algumas formas de otimização podem reduzir a quantidade de computação necessária para realizar a inferência usando um modelo, resultando em menor latência. A latência também pode ter um impacto no consumo de energia.

No momento, a quantização pode ser usada para reduzir a latência ao simplificar os cálculos que ocorrem durante a inferência, possivelmente à custa de um pouco de exatidão.

### Compatibilidade com os aceleradores

Alguns aceleradores de hardware, como o [Edge TPU](https://cloud.google.com/edge-tpu/), podem realizar a inferência de forma extremamente rápida com modelos que foram otimizados de maneira correta.

Geralmente, esses tipos de dispositivos exigem que os modelos sejam quantizados de maneira específica. Veja a documentação de cada acelerador de hardware para saber mais sobre os requisitos deles.

## Trade-offs

As otimizações podem resultar possivelmente em mudanças na exatidão do modelo, o que precisa ser considerado durante o processo de desenvolvimento do aplicativo.

As mudanças na exatidão dependem do modelo individual que está sendo otimizado e são difíceis de prever com antecedência. Geralmente, os modelos otimizados para o tamanho ou a latência perdem um pouco da exatidão. Dependendo do seu aplicativo, isso pode afetar ou não a experiência dos seus usuários. Em casos raros, alguns modelos podem ganhar um pouco de exatidão como resultado do processo de otimização.

## Tipos de otimização

Atualmente, o TensorFlow Lite aceita a otimização por quantização, pruning e clustering.

Eles fazem parte do [Kit de ferramentas para a otimização de modelo do TensorFlow](https://www.tensorflow.org/model_optimization), que fornece recursos para técnicas de otimização de modelo compatíveis com o TensorFlow Lite.

### Quantização

A [quantização](https://www.tensorflow.org/model_optimization/guide/quantization/post_training) funciona ao reduzir a precisão dos números usados para representar os parâmetros de um modelo, que, por padrão, são números de ponto flutuante de 32 bits. Isso resulta em um menor tamanho de modelo e uma computação mais rápida.

Os seguintes tipos de quantização estão disponíveis no TensorFlow Lite:

Técnica | Requisitos de dados | Redução de tamanho | Exatidão | Hardware compatível
--- | --- | --- | --- | ---
[Quantização float16 pós-treinamento](post_training_float16_quant.ipynb) | Nenhum dado | Até 50% | Perda de exatidão insignificante | CPU, GPU
[Quantização de intervalo dinâmico pós-treinamento](post_training_quant.ipynb) | Nenhum dado | Até 75% | Perda de exatidão mínima | CPU, GPU (Android)
[Quantização de números inteiros pós-treinamento](post_training_integer_quant.ipynb) | Amostra representativa não rotulada | Até 75% | Perda de exatidão pequena | CPU, GPU (Android), EdgeTPU, DSP Hexagon
[Treinamento consciente de quantização](http://www.tensorflow.org/model_optimization/guide/quantization/training) | Dados de treinamento rotulados | Até 75% | Perda de exatidão mínima | CPU, GPU (Android), EdgeTPU, DSP Hexagon

A seguinte árvore de decisão ajuda você a selecionar os esquemas de quantização que talvez queira usar para seu modelo, simplesmente com base no tamanho e na exatidão esperados do modelo.

![Árvore de decisão de quantização](images/quantization_decision_tree.png)

Confira abaixo os resultados de latência e exatidão para a quantização pós-treinamento e o treinamento consciente de quantização em alguns modelos. Todos os números de latência são medidos em dispositivos Pixel 2 usando uma única CPU big core. Conforme o kit de ferramentas melhorar, os números aqui também vão:

<figure>
  <table>
    <tr>
      <th>Model</th>
      <th>Top-1 Accuracy (Original) </th>
      <th>Top-1 Accuracy (Post Training Quantized) </th>
      <th>Top-1 Accuracy (Quantization Aware Training) </th>
      <th>Latency (Original) (ms) </th>
      <th>Latency (Post Training Quantized) (ms) </th>
      <th>Latency (Quantization Aware Training) (ms) </th>
      <th> Size (Original) (MB)</th>
      <th> Size (Optimized) (MB)</th>
    </tr> <tr><td>Mobilenet-v1-1-224</td><td>0.709</td><td>0.657</td><td>0.70</td>
      <td>124</td><td>112</td><td>64</td><td>16.9</td><td>4.3</td></tr>
    <tr><td>Mobilenet-v2-1-224</td><td>0.719</td><td>0.637</td><td>0.709</td>
      <td>89</td><td>98</td><td>54</td><td>14</td><td>3.6</td></tr>
   <tr><td>Inception_v3</td><td>0.78</td><td>0.772</td><td>0.775</td>
      <td>1130</td><td>845</td><td>543</td><td>95.7</td><td>23.9</td></tr>
   <tr><td>Resnet_v2_101</td><td>0.770</td><td>0.768</td><td>N/A</td>
      <td>3973</td><td>2868</td><td>N/A</td><td>178.3</td><td>44.9</td></tr>
 </table>
  <figcaption>
    <b>Table 1</b> Benefits of model quantization for select CNN models
  </figcaption>
</figure>

### Quantização de números inteiros com ativações int16 e pesos int8

A [quantização com ativações int16](https://www.tensorflow.org/model_optimization/guide/quantization/post_training) é um esquema de quantização de números inteiros com ativações em int16 e pesos em int8. Esse modo pode melhorar a exatidão do modelo quantizado em comparação com o esquema de quantização de números inteiros com as ativações e os pesos em int8, mantendo um tamanho de modelo semelhante. É recomendado quando as ativações são sensíveis à quantização.

<i>OBSERVAÇÃO:</i> no momento, somente implementações de kernels de referência não otimizados estão disponíveis no TFLite para esse esquema de quantização. Então, por padrão, o desempenho será lento em comparação aos kernels int8. Os benefícios completos desse modo podem ser acessados por hardware especializado ou software personalizado.

Confira os resultados de exatidão para alguns modelos que se beneficiam desse modo.

<figure>
  <table>
    <tr>
      <th>Model</th>
      <th>Accuracy metric type </th>
      <th>Accuracy (float32 activations) </th>
      <th>Accuracy (int8 activations) </th>
      <th>Accuracy (int16 activations) </th>
    </tr> <tr><td>Wav2letter</td><td>WER</td><td>6.7%</td><td>7.7%</td>
      <td>7.2%</td></tr>
    <tr><td>DeepSpeech 0.5.1 (unrolled)</td><td>CER</td><td>6.13%</td><td>43.67%</td>
      <td>6.52%</td></tr>
    <tr><td>YoloV3</td><td>mAP(IOU=0.5)</td><td>0.577</td><td>0.563</td>
      <td>0.574</td></tr>
    <tr><td>MobileNetV1</td><td>Top-1 Accuracy</td><td>0.7062</td><td>0.694</td>
      <td>0.6936</td></tr>
    <tr><td>MobileNetV2</td><td>Top-1 Accuracy</td><td>0.718</td><td>0.7126</td>
      <td>0.7137</td></tr>
    <tr><td>MobileBert</td><td>F1(Exact match)</td><td>88.81(81.23)</td><td>2.08(0)</td>
      <td>88.73(81.15)</td></tr>
 </table>
  <figcaption>
    <b>Table 2</b> Benefits of model quantization with int16 activations
  </figcaption>
</figure>

### Pruning

O [pruning](https://www.tensorflow.org/model_optimization/guide/pruning) atua removendo os parâmetros de um modelo que só têm um impacto menor nas suas previsões. Os modelos que passaram pelo pruning têm o mesmo tamanho em disco e a mesma latência de runtime, mas podem ser comprimidos com mais eficiência. Isso faz com que o pruning seja uma técnica útil para reduzir o tamanho de download do modelo.

No futuro, o TensorFlow Lite proporcionará a redução da latência para modelos após o pruning.

### Clustering

O [clustering](https://www.tensorflow.org/model_optimization/guide/clustering) atua agrupando os pesos de cada camada de um modelo em um número predefinido de clusters e, depois, compartilhando os valores de centroides para os pesos que pertencem a cada cluster individual. Isso reduz o número de valores de pesos únicos em um modelo, diminuindo sua complexidade.

Como resultado, os modelos que passaram pelo clustering podem ser comprimidos com mais eficiência, oferecendo benefícios semelhantes ao pruning para a implantação.

## Fluxo de trabalho de desenvolvimento

Como ponto de partida, verifique se os [modelos hospedados](../guide/hosted_models.md) funcionam para seu aplicativo. Caso contrário, recomendamos que os usuários comecem com a [ferramenta de quantização pós-treinamento](post_training_quantization.md), já que é amplamente aplicável e não exige dados de treinamento.

Para os casos em que os alvos de exatidão e latência não forem atingidos ou o suporte ao acelerador de hardware for importante, o [treinamento consciente de quantização](https://www.tensorflow.org/model_optimization/guide/quantization/training){:.external} é a melhor opção. Veja as técnicas de otimização adicionais no [Kit de ferramentas para a otimização de modelo do TensorFlow](https://www.tensorflow.org/model_optimization).

Se você quiser reduzir ainda mais o tamanho do modelo, experimente o [pruning](#pruning) e/ou [clustering](#clustering) antes da quantização.
