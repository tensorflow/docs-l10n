# Atualizações da API <a name="api_updates"></a>

Esta página fornece informações sobre as atualizações feita na [API do Python](index.md) `tf.lite.TFLiteConverter` no TensorFlow 2.x.

Observação: se alguma das alterações gerar preocupações, crie um [issue no GitHub](https://github.com/tensorflow/tensorflow/issues/new?template=60-tflite-converter-issue.md).

- TensorFlow 2.3

    - Suporte ao tipo de saída/entrada inteiro (antes, só havia suporte a ponto flutuante) para modelos quantizados com inteiros usando os novos atributos `inference_input_type` e `inference_output_type`. Confira este [exemplo de uso](../../performance/post_training_quantization.md#integer_only).
    - Suporte à conversão e ao redimensionamento de modelos com dimensões dinâmicas.
    - Inclusão de um novo modo de quantização experimental com ativações de 16 bits e pesos de 8 bits.

- TensorFlow 2.2

    - Por padrão, use a [conversão baseada em MLIR](https://mlir.llvm.org/), uma tecnologia de compilação de ponta do Google para aprendizado de máquina que permite fazer a conversão de novas classes de modelos, incluindo Mask R-CNN, Mobile BERT, etc., além de oferecer suporte a modelos com fluxo de controle funcional.

- TensorFlow 2.0 versus TensorFlow 1.x

    - O atributo `target_ops` foi renomeado para `target_spec.supported_ops`
    - Os seguintes atributos foram removidos:
        - *Quantização*: `inference_type`, `quantized_input_stats`, `post_training_quantize`, `default_ranges_stats`, `reorder_across_fake_quant`, `change_concat_input_ranges`, `get_input_arrays()`. Em vez disso, há suporte ao [treinamento com reconhecimento de quantização](https://www.tensorflow.org/model_optimization/guide/quantization/training) por meio da API `tf.keras`, e a [quantização pós-treinamento](../../performance/post_training_quantization.md) usa menos atributos.
        - *Visualização*: `output_format`, `dump_graphviz_dir`, `dump_graphviz_video`. Em vez disso, a estratégia recomendável para visualizar um modelo do TensorFlow lite é o uso de [visualize.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/tools/visualize.py).
        - *Grafos congelados*: `drop_control_dependency`, pois não há suporte a grafos congelados no TensorFlow 2.x.
    - Foram removidas outras APIs de conversão, como `tf.lite.toco_convert` e `tf.lite.TocoConverter`.
    - Foram removidas outras APIs relacionadas, como `tf.lite.OpHint` e `tf.lite.constants` (os tipos `tf.lite.constants.*` foram mapeados para tipos de dados `tf.*` do TensorFlow para reduzir a duplicação).
