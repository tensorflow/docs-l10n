# Formatos de modelos

[tfhub.dev](https://tfhub.dev) hospeda os seguintes formatos de modelos: SavedModel do TF2, TF1 Hub, TF.js e TFLite. Esta página apresenta uma visão geral de cada formato de modelos.

Conteúdo publicado em tfhub.dev pode ser espelhado automaticamente para outros hubs de modelos, desde que siga um formato especificado e seja permitido pelos nossos termos e condições (https://tfhub.dev/terms). Confira mais detalhes em nossa [documentação de publicação](publish.md). Caso opte por não fazer o espelhamento, confira nossa [documentação de contribuição](contribute_a_model.md).

## Formatos do TensorFlow

[tfhub.dev](https://tfhub.dev) hospeda modelos do TensorFlow no formato SavedModel do TF2 e no formato TF1 Hub. Recomendamos usar modelos no formato padronizado SavedModel do TF2 em vez do formato descontinuado TF1 Hub quando possível.

### SavedModel

SavedModel do TF2 é o formato recomendado para compartilhar modelos do TensorFlow. Saiba mais sobre esse formato no guia de [SavedModel do TensorFlow](https://www.tensorflow.org/guide/saved_model).

Você pode pesquisar SavedModels em tfhub.dev usando o filtro de versão TF2 na [página de pesquisa em tfhub.dev](https://tfhub.dev/s?subtype=module,placeholder) ou acessando [este link](https://tfhub.dev/s?subtype=module,placeholder&tf-version=tf2).

Você pode usar os SavedModels em tfhub.dev sem depender da biblioteca `tensorflow_hub`, já que esse formato faz parte do core do TensorFlow.

Saiba mais sobre SavedModels no TF Hub:

- [Usando SavedModels do TF2](tf2_saved_model.md)
- [Exportando um SavedModel do TF2](exporting_tf2_saved_model.md)
- [Compatibilidade de SavedModels do TF2 no TF1/TF2](model_compatibility.md)

### Formato TF1 Hub

O formato TF1 Hub é um formato de serialização personalizado usado pela biblioteca do TF Hub e é similar ao formato SavedModel do TensorFlow 1 em um nível sintático (mesmos nomes de arquivo e mensagens de protocolo), mas é diferente semanticamente para permitir a reutilização de modelos, a composição e o retreinamento (por exemplo, armazenamento diferente de inicializadores de recursos e convenções de etiquetas diferentes para metagrafos). A maneira mais fácil de diferenciá-los no disco é a presença ou ausência do arquivo `tfhub_module.pb`.

Você pode pesquisar modelos com o formato TF1 Hub em tfhub.dev usando o filtro de versão TF1 na [página de pesquisa em tfhub.dev](https://tfhub.dev/s?subtype=module,placeholder) ou acessando [este link](https://tfhub.dev/s?subtype=module,placeholder&tf-version=tf1).

Saiba mais sobre modelos no formato TF1 Hub no TF Hub:

- [Usando modelos no formato TF1 Hub](tf1_hub_module.md)
- [Exportando um modelo no formato TF1 Hub](exporting_hub_format.md)
- [Compatibilidade do formato TF1 Hub no TF1/TF2](model_compatibility.md)

## Formato TFLite

O formato TFLite é usado para inferência em dispositivos. Saiba mais na [documentação do TFLite](https://www.tensorflow.org/lite).

Você pode pesquisar modelos do TFLite em tfhub.dev usando o filtro de formato de modelo TF Lite na [página de pesquisa em tfhub.dev](https://tfhub.dev/s?subtype=module,placeholder) ou acessando [este link](https://tfhub.dev/lite).

## Formato TFJS

O formato TF.js é usado para aprendizado de máquina em navegadores. Saiba mais na [documentação do TF.js](https://www.tensorflow.org/js).

Você pode pesquisar modelos do TF.js em tfhub.dev usando o filtro de formato de modelo TF.js na [página de pesquisa em tfhub.dev](https://tfhub.dev/s?subtype=module,placeholder) ou acessando [este link](https://tfhub.dev/js).
