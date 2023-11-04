# Melhorando a qualidade do modelo com o TensorFlow Model Analysis

## Introdução

À medida em que você ajusta seu modelo durante o desenvolvimento, você precisa verificar se suas alterações estão melhorando seu modelo. Apenas verificar a exatidão pode não ser suficiente. Por exemplo, se você tiver um classificador para um problema no qual 95% de suas instâncias são positivas, você poderá melhorar a exatidão simplesmente prevendo sempre o positivo, mas não terá um classificador muito robusto.

## Visão geral

O objetivo do TensorFlow Model Analysis é fornecer um mecanismo para avaliação de modelos no TFX. A TensorFlow Model Analysis permite realizar avaliações de modelo no pipeline do TFX e visualizar métricas e gráficos resultantes num notebook Jupyter. Especificamente, ele pode fornecer:

- [Métricas](../model_analysis/metrics) computadas sobre todo o dataset de treinamento e validação, bem como avaliações do dia seguinte
- Acompanhamento de métricas ao longo do tempo
- Desempenho de qualidade do modelo em diferentes fatias de características
- [Validação do modelo](../model_analysis/model_validations) para garantir que o modelo mantenha um desempenho consistente

## Próximos passos

Veja nosso [tutorial TFMA](../tutorials/model_analysis/tfma_basic).

Confira nossa página no [github](https://github.com/tensorflow/model-analysis) para detalhes sobre as [métricas e gráficos](../model_analysis/metrics) suportados e [visualizações](../model_analysis/visualizations) de notebook associadas.

Consulte os guias de [instalação](../model_analysis/install) e [introdução](../model_analysis/get_started) para informações e exemplos sobre como [configurar](../model_analysis/setup) um pipeline independente. Lembre-se de que o TFMA também é usado no componente [Evaluator](evaluator.md) do TFX, portanto, esses recursos também serão úteis para começar no TFX.
