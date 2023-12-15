# Otimize ainda mais

Quando os modelos pré-otimizados e as ferramentas de pós-treinamento não atendem ao seu caso de uso, a próxima etapa é tentar diferentes ferramentas de tempo de treinamento.

As ferramentas de tempo de treinamento pegam carona com a função de perda do modelo nos dados de treinamento, de modo que o modelo consiga se "adaptar" às mudanças provocadas pela técnica de otimização.

O ponto de partida para usar nossas APIs de treinamento é um script de treinamento do Keras, que pode ser inicializado opcionalmente a partir de um modelo do Keras pré-treinado para ajuste adicional.

Ferramentas de tempo de treinamento disponíveis para você testar:

- [Pruning de peso](./pruning/)
- [Quantização](./quantization/training)
- [Clustering de peso](./clustering/)
- [Otimização colaborativa](./combine/collaborative_optimization)
