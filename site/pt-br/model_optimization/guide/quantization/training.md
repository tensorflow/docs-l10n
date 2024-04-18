# Treinamento consciente de quantização

<sub>Manutenção pela Otimização de modelos do TensorFlow</sub>

Há duas formas de quantização: quantização pós-treinamento e treinamento consciente de quantização. Comece com a [quantização pós-treinamento](post_training.md), já que é mais fácil de usar, embora o treinamento consciente de quantização geralmente seja melhor para a exatidão do modelo.

Esta página fornece uma visão geral do treinamento consciente de quantização para ajudar você a determinar como ele se adequa ao seu caso de uso.

- Para ir direto para um exemplo completo, veja o [exemplo de treinamento consciente de quantização](training_example.ipynb).
- Para encontrar rapidamente as APIs necessárias para seu caso de uso, veja o [guia completo de treinamento consciente de quantização](training_comprehensive_guide.ipynb).

## Visão geral

O treinamento consciente de quantização emula a quantização de tempo de inferência, criando um modelo que faz o downstream das ferramentas que usará para produzir modelos realmente quantizados. Os modelos quantizados usam uma precisão inferior (por exemplo, float de 8 bits em vez de 32 bits), levando a benefícios durante a implantação.

### Implante com a quantização

A quantização traz melhorias através da compressão do modelo e da redução da latência. Com os padrões da API, o tamanho do modelo encolhe em 4x, e geralmente vemos melhorias de 1,5 a 4x na latência da CPU nos back-ends testados. Por fim, as melhorias na latência podem ser vistas em aceleradores de aprendizado de máquina compatíveis, como [EdgeTPU](https://coral.ai/docs/edgetpu/benchmarks/) e NNAPI.

A técnica é usada na produção de casos de uso de fala, visão, texto e tradução. No momento, o código é compatível com um [subconjunto desses modelos](#general-support-matrix).

### Experimente com a quantização e o hardware associado

Os usuários podem configurar os parâmetros da quantização (por exemplo, o número de bits) e, até certo ponto, os algoritmos subjacentes. Observe que, com essas mudanças dos padrões da API, atualmente não há caminho compatível para a implantação em um back-end. Por exemplo, a conversão para o TFLite e as implementações de kernel só são compatíveis com a quantização de 8 bits.

As APIs específicas à essa configuração são experimentais e não estão sujeitas à compatibilidade com versões anteriores.

### Compatibilidade de APIs

Os usuários podem aplicar a quantização com as seguintes APIs:

- Criação de modelos: `tf.keras` com apenas os modelos sequenciais e funcionais.
- Versões do TensorFlow: TF 2.x para tf-nightly.
    - `tf.compat.v1` com um pacote do TF 2.X não é compatível.
- Modo de execução do TensorFlow: eager execution

Planejamos adicionar suporte nas seguintes áreas:

<!-- TODO(tfmot): file Github issues. -->

- Criação de modelos: esclarecer como os modelos com subclasses não são compatíveis ou têm suporte limitado
- Treinamento distribuído: `tf.distribute`

### Matriz de suporte geral

O suporte está disponível nas seguintes áreas:

- Cobertura de modelos: modelos que usam [camadas permitidas](https://github.com/tensorflow/model-optimization/tree/master/tensorflow_model_optimization/python/core/quantization/keras/default_8bit/default_8bit_quantize_registry.py), BatchNormalization quando segue camadas Conv2D e DepthwiseConv2D e, em casos limitados, `Concat`.
    <!-- TODO(tfmot): add more details and ensure they are all correct. -->
- Aceleração de hardware: os padrões da nossa API são compatíveis com os back-ends do EdgeTPU, NNAPI e TFLite, entre outros. Veja a ressalva no plano.
- Implante com a quantização: só é compatível a quantização por eixo para camadas convolucionais, e não a quantização por tensor.

Planejamos adicionar suporte nas seguintes áreas:

<!-- TODO(tfmot): file Github issue. Update as more functionality is added prior
to launch. -->

- Cobertura de modelos: ampliada para incluir o suporte a RNN/LSTMs e à Concat em geral.
- Aceleração de hardware: garanta que o conversor do TFLite possa produzir modelos de números inteiros. Veja [este issue](https://github.com/tensorflow/tensorflow/issues/38285) para mais detalhes.
- Casos de uso para experimentar com a quantização:
    - Experimentar com algoritmos de quantização que abrangem camadas do Keras ou exigem o passo de treinamento.
    - Estabilizar APIs.

## Resultados

### Classificação de imagens com ferramentas

<figure>
  <table>
    <tr>
      <th>Model</th>
      <th>Non-quantized Top-1 Accuracy </th>
      <th>8-bit Quantized Accuracy </th>
    </tr>
    <tr>
      <td>MobilenetV1 224</td>
      <td>71.03%</td>
      <td>71.06%</td>
    </tr>
    <tr>
      <td>Resnet v1 50</td>
      <td>76.3%</td>
      <td>76.1%</td>
    </tr>
    <tr>
      <td>MobilenetV2 224</td>
      <td>70.77%</td>
      <td>70.01%</td>
    </tr>
 </table>
</figure>

Os modelos foram testados com o Imagenet e avaliados no TensorFlow e no TFLite.

### Classificação de imagens para técnica

<figure>
  <table>
    <tr>
      <th>Model</th>
      <th>Non-quantized Top-1 Accuracy </th>
      <th>8-Bit Quantized Accuracy </th>
    <tr>
      <td>Nasnet-Mobile</td>
      <td>74%</td>
      <td>73%</td>
    </tr>
    <tr>
      <td>Resnet-v2 50</td>
      <td>75.6%</td>
      <td>75%</td>
    </tr>
 </table>
</figure>

Os modelos foram testados com o Imagenet e avaliados no TensorFlow e no TFLite.

## Exemplos

Além do [exemplo de treinamento consciente de quantização](training_example.ipynb), confira estes exemplos:

- Modelo de CNN na tarefa de classificação de dígitos escritos à mão do MNIST com quantização: [código](https://github.com/tensorflow/model-optimization/blob/master/tensorflow_model_optimization/python/core/quantization/keras/quantize_functional_test.py)

Para contexto sobre algo semelhante, veja o [artigo](https://arxiv.org/abs/1712.05877) *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference* (Quantização e treinamento de redes neurais para inferência somente aritmética de números inteiros eficiente). Esse artigo apresenta alguns conceitos usados por essa ferramenta. A implementação não é exatamente igual, e a ferramenta utiliza conceitos adicionais (por exemplo, quantização por eixo).
