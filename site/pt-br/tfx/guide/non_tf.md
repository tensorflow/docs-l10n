# Usando outros frameworks de ML no TFX

O TFX como plataforma é neutro em relação a frameworks e pode ser usado com outros frameworks de ML, por exemplo, JAX, scikit-learn.

Para desenvolvedores de modelo, isto significa que eles não precisam reescrever o código do modelo implementado em outro framework de ML, mas podem reutilizar a maior parte do código de treinamento como está no TFX e se beneficiar de outros recursos do TFX e recursos oferecidos pelo ecossistema do TensorFlow.

O SDK do pipeline do TFX e a maioria dos módulos do TFX, por exemplo, orquestrador do pipeline, não têm nenhuma dependência direta no TensorFlow, mas possui alguns aspectos que são orientados para o TensorFlow, como formatos de dados. Levando em consideração as necessidades de um framework de modelagem específico, um pipeline TFX poderia ser usado para treinar modelos em qualquer outro framework de ML baseado no Python. Isto inclui o Scikit-learn, o XGBoost e o PyTorch, entre outros. Algumas questões a serem consideradas ao usar os componentes padrão do TFX com outros frameworks incluem:

- O **ExampleGen** gera [tf.train.Example](https://www.tensorflow.org/tutorials/load_data/tfrecord) em arquivos TFRecord. É uma representação genérica para dados de treinamento, e os componentes downstream usam [TFXIO](https://github.com/tensorflow/community/blob/master/rfcs/20191017-tfx-standardized-inputs.md) para lê-los como Arrow/RecordBatch na memória, que pode ser posteriormente convertido em `tf.dataset`, `Tensors` ou outros formatos. Estão sendo considerados formatos de payload/arquivo diferentes de tf.train.Example/TFRecord, mas para os usuários do TFXIO, isto deve ser uma caixa preta.
- O **Transform** pode ser usado para gerar exemplos de treinamento transformados, independentemente do framework usado no treinamento, mas se o formato do modelo não for `saved_model`, os usuários não poderão incorporar o grafo de transformação ao modelo. Neste caso, a previsão do modelo precisa usar características transformadas em vez de características brutas, e os usuários poderão executar a transformação como uma etapa de pré-processamento antes de chamar a previsão do modelo durante o serviço.
- O **O Trainer** suporta o [GenericTraining](https://www.tensorflow.org/tfx/guide/trainer#generic_trainer) para que os usuários possam treinar seus modelos usando qualquer framework de ML.
- Por padrão, o **Evaluator** suporta apenas `saved_model`, mas os usuários podem fornecer uma UDF que gera previsões para a avaliação de modelos.

O treinamento de um modelo num framework que não se baseia em Python exigirá o isolamento de um componente de treinamento personalizado num container Docker, como parte de um pipeline que esteja sendo executado num ambiente de container, como o Kubernetes.

## JAX

O [JAX](https://github.com/google/jax) é Autograd e XLA, combinados para a pesquisa de aprendizado de máquina de alto desempenho. O [Flax](https://github.com/google/flax) é uma biblioteca de redes neurais e ecossistema para o JAX, projetada para oferecer flexibilidade.

Com o [jax2tf](https://github.com/google/jax/tree/main/jax/experimental/jax2tf), podemos converter modelos JAX/Flax treinados no formato `saved_model`, que pode ser usado tranquilamente no TFX com treinamento genérico e avaliação de modelos. Para mais detalhes, veja este [exemplo](https://github.com/tensorflow/tfx/blob/master/tfx/examples/penguin/penguin_utils_flax_experimental.py).

## scikit-learn

O [Scikit-learn](https://scikit-learn.org/stable/) é uma biblioteca de aprendizado de máquina para a linguagem de programação Python. Temos um [exemplo](https://github.com/tensorflow/tfx-addons/tree/main/examples/sklearn_penguins) e2e com treinamento e avaliação customizados em TFX-Addons.
