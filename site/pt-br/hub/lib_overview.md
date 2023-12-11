# Visão geral da biblioteca do TensorFlow Hub

Com a biblioteca [`tensorflow_hub`](https://github.com/tensorflow/hub), você pode baixar e reutilizar modelos treinados em seu programa do TensorFlow com uma quantidade mínima de código. A principal forma de carregar um modelo treinado é usando a API `hub.KerasLayer`.

```python
import tensorflow_hub as hub

embed = hub.KerasLayer("https://tfhub.dev/google/nnlm-en-dim128/2")
embeddings = embed(["A long sentence.", "single-word", "http://example.com"])
print(embeddings.shape, embeddings.dtype)
```

**Observação:** esta documentação usa identificadores de URL TFhub.dev nos exemplos. Confira mais informações sobre outros tipos de identificador válidos [aqui](tf2_saved_model.md#model_handles).

## Definição do local de cache para downloads

Por padrão, `tensorflow_hub` use um diretório temporário para todo o sistema para fazer o cache de modelos baixados e descompactados. Confira opções de uso de outros locais, possivelmente mais persistentes, em [Como fazer cache](caching.md).

## Estabilidade da API

Embora esperamos evitar mudanças que causem problemas em códigos existentes, este projeto ainda está em pleno desenvolvimento, e ainda não há garantias de uma API ou formato de modelo estáveis.

## Equidade

Em aprendizado de máquina, a [equidade](http://ml-fairness.com) é uma consideração [importante](https://research.googleblog.com/2016/10/equality-of-opportunity-in-machine.html). Vários modelos pré-treinados são treinados com datasets grandes. Ao reutilizar qualquer modelo, é importante ter em mente com quais dados o modelo foi treinado (e se há algum bias existente) e como eles podem impactar seu uso do modelo.

## Segurança

Como os modelos contêm grafos arbitrários do TensorFlow, podem ser considerados programas. O artigo [Como usar o TensorFlow com segurança](https://github.com/tensorflow/tensorflow/blob/master/SECURITY.md) descreve as implicações de segurança ao referenciar um modelo de uma fonte não confiável.

## Próximos passos

- [Usando a biblioteca](tf2_saved_model.md)
- [SavedModels reutilizáveis](reusable_saved_models.md)
