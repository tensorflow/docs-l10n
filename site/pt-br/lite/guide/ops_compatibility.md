# Compatibilidade de operadores do TensorFlow Lite e TensorFlow

Os operadores de aprendizado de máquina (ML) que você usa em seu modelo podem impactar o processo de conversão de um modelo do TensorFlow para o formato do TensorFlow Lite. O conversor do TF Lite tem suporte a um número limitado de operações do TensorFlow usadas em modelos comuns de inferência, ou seja, nem todo modelo pode ser convertido diretamente. A ferramenta de conversão permite incluir operadores adicionais, mas, para converter um modelo dessa forma, é preciso modificar o ambiente de runtime do TensorFlow Lite que você usa para executar seu modelo, o que pode limitar a capacidade de usar as opções padrão de implantação do runtime, como os [serviços do Google Play](../android/play_services).

O conversor do TensorFlow Lite foi criado para analisar a estrutura do modelo e aplicar otimizações a fim de torná-lo compatível com os operações com suporte nativo. Por exemplo: dependendo dos operadores de ML em seu modelo, o conversor poderá [eliminar ou fundir](../models/convert/operation_fusion) esses operadores para mapeá-los para suas contrapartes no TensorFlow Lite.

Mesmo para as operações com suporte, às vezes são esperados padrões de uso específicos por questões de desempenho. A melhor forma de entender como criar um modelo do TensorFlow que possa ser usado no TensorFlow Lite é considerar cuidadosamente como as operações serão convertidas e otimizadas, além das limitações decorrentes desse processo.

## Operadores com suporte

Os operadores integrados do TensorFlow Lite são um subconjunto dos operadores que fazem parte da biblioteca principal do TensorFlow. Seu modelo do TensorFlow também pode incluir operadores personalizados, como operadores compostos ou novos operadores definidos por você. O diagrama abaixo mostra as relações entre esses operadores.

![TensorFlow operators](../images/convert/tf_operators_relationships.png)

De todos esses operadores de modelos de ML, existem 3 tipos de modelos com suporte a esse processo de conversão:

1. Modelos com apenas operadores integrados do TensorFlow Lite (**recomendado**).
2. Modelos com operadores integrados e operadores core específicos do TensorFlow.
3. Modelos com os operadores integrados, operadores core do TensorFlow e/ou operadores personalizados.

Se o seu modelo tiver apenas operações com suporte nativo do TensorFlow Lite, você não precisa de nenhum sinalizador adicional para convertê-lo. Essa é a forma recomendada, pois esse tipo de modelo terá uma conversão tranquila e é mais simples de otimizar e executar utilizando o runtime padrão do TensorFlow Lite. Além disso, você tem mais opções de implantação do modelo, como os [serviços do Google Play](../android/play_services). Comece conferindo o [guia do conversor do TensorFlow Lite](../models/convert/convert_models). Veja a lista de operadores integrados na [página de operações do TensorFlow Lite](https://www.tensorflow.org/mlir/tfl_ops).

Se você precisar incluir operações específicas do TensorFlow da biblioteca core, precisa especificar na conversão e garantir que o runtime inclua essas operações. Confira mais detalhes no tópico [operadores específicos do TensorFlow](ops_select.md).

Sempre que possível, evite a última opção, a de incluir operadores personalizados em seu modelo convertido. Os [operadores personalizados](https://www.tensorflow.org/guide/create_op) são operadores criados pela combinação de diversos operadores primitivos core do TensorFlow ou pela definição de um operador totalmente novo. Quando operadores personalizados são convertidos, podem aumentar o tamanho do modelo de forma geral devido à inclusão de dependências fora da biblioteca integrada do TensorFlow Lite. Se as operações personalizadas não forem criadas para implantação em dispositivos móveis ou outros dispositivos, o desempenho pode piorar ao implantar em dispositivos com restrição de recursos em comparação a um ambiente de servidor. Por fim, assim como ao incluir os operadores específicos core do TensorFlow, é preciso [modificar o ambiente de runtime](ops_custom#create_and_register_the_operator) ao incluir os operadores personalizados, o que limita o uso de serviços padrão do runtime, como os [serviços do Google Play](../android/play_services).

## Tipos permitidos

A maioria das operações do TensorFlow Lite tem como objetivo inferência em ponto flutuante (`float32`) e também quantizada (`uint8` e `int8`), mas diversas outras operações ainda não contam com esse objetivo para outros tipos, como `tf.float16` e strings.

Além do uso de uma versão diferente das operações, as outras diferenças entre os modelos de ponto flutuante e quantizados é a forma como são convertidos. A conversão quantizada requer informações de intervalo dinâmico para os tensores, o que exige uma "quantização falsa" durante o treinamento do modelo, obtendo as informações de intervalo por um dataset de calibração ou fazendo a estimativa do intervalo em tempo real. Confira mais detalhes em [quantização](../performance/model_optimization.md).

## Conversões diretas, constant-folding e fusão

Diversas operações do TensorFlow podem ser processadas pelo TensorFlow Lite, mesmo que não tenham um equivalente direto. Esse é o caso de operações que podem ser simplesmente removidas do grafo (`tf.identity`), substituídas por tensores (`tf.placeholder`) ou fundidas em operações mais complexas (`tf.nn.bias_add`). Porém, algumas operações com suporte podem ser removidas por um desses processos, às vezes.

Confira abaixo uma lista não exaustiva de operações do TensorFlow que geralmente são removidas do grafo:

- `tf.add`
- `tf.debugging.check_numerics`
- `tf.constant`
- `tf.div`
- `tf.divide`
- `tf.fake_quant_with_min_max_args`
- `tf.fake_quant_with_min_max_vars`
- `tf.identity`
- `tf.maximum`
- `tf.minimum`
- `tf.multiply`
- `tf.no_op`
- `tf.placeholder`
- `tf.placeholder_with_default`
- `tf.realdiv`
- `tf.reduce_max`
- `tf.reduce_min`
- `tf.reduce_sum`
- `tf.rsqrt`
- `tf.shape`
- `tf.sqrt`
- `tf.square`
- `tf.subtract`
- `tf.tile`
- `tf.nn.batch_norm_with_global_normalization`
- `tf.nn.bias_add`
- `tf.nn.fused_batch_norm`
- `tf.nn.relu`
- `tf.nn.relu6`

Observação: diversas dessas operações não têm equivalentes no TensorFlow Lite, e o modelo correspondente não poderá ser convertido se elas não puderem ser eliminadas ou fundidas.

## Operações experimentais

As operações do TensorFlow Lite abaixo estão presentes, mas não estão prontas para modelos personalizados:

- `CALL`
- `CONCAT_EMBEDDINGS`
- `CUSTOM`
- `EMBEDDING_LOOKUP_SPARSE`
- `HASHTABLE_LOOKUP`
- `LSH_PROJECTION`
- `SKIP_GRAM`
- `SVDF`
