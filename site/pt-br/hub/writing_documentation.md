# Como escrever a documentação

Para contribuir com o tfhub.dev, é preciso fornecer a documentação no formato Markdown. Confira a visão geral do processo de como contribuir com modelos para o tfhub.dev no guia [Como contribuir com um modelo](contribute_a_model.md).

**Observação:** o termo "publicador" é usado em todo este documento e refere-se ao proprietário registrado de um modelo hospedado em tfhub.dev.

## Tipos de documentação Markdown

Existem três tipos de documentação Markdown usadas em tfhub.dev:

- Markdown de publicador – contém informações sobre um publicador ([confira a sintaxe de marcação](#publisher))
- Markdown de modelo – contém informações sobre um modelo específico e como usá-lo ([confira a sintaxe de marcação](#model))
- Markdown de coleção – contém informações sobre uma coleção de modelos definidos por um publicador ([confira a sintaxe de marcação](#collection))

## Organização do conteúdo

A organização de conteúdo abaixo é necessária ao contribuir com o repositório do [GitHub do TensorFlow](https://github.com/tensorflow/tfhub.dev):

- o diretório de cada publicador fica no diretório `assets/docs`
- o diretório de cada publicador contém os diretórios opcionais `models` (modelos) e `collections` (coleções)
- cada modelo deve ter seu próprio diretório dentro de `assets/docs/<publisher_name>/models`
- cada coleção deve ter seu próprio diretório dentro de `assets/docs/<publisher_name>/collections`

As marcações de publicador não são versionadas, enquanto as de modelos têm diferentes versões. Cada versão de modelo requer um arquivo Markdown separado com nome referente à sua versão (por exemplo, 1.md, 2.md). As coleções são versionadas, mas há suporte somente a uma versão (1.md).

Todas as versões de um determinado modelo devem estar localizadas no diretório de modelos.

Veja abaixo um exemplo de como o conteúdo Markdown é organizado:

```
assets/docs
├── <publisher_name_a>
│   ├── <publisher_name_a>.md  -> Documentation of the publisher.
│   └── models
│       └── <model_name>       -> Model name with slashes encoded as sub-path.
│           ├── 1.md           -> Documentation of the model version 1.
│           └── 2.md           -> Documentation of the model version 2.
├── <publisher_name_b>
│   ├── <publisher_name_b>.md  -> Documentation of the publisher.
│   ├── models
│   │   └── ...
│   └── collections
│       └── <collection_name>
│           └── 1.md           -> Documentation for the collection.
├── <publisher_name_c>
│   └── ...
└── ...
```

## Formato de marcação de publicador {:#publisher}

A documentação do publicador é declarada com o mesmo tipo de arquivos Markdown que os modelos, mas com pequenas diferenças sintáticas.

O local correto do arquivo de publicador no repositório do TensorFlow Hub é: [tfhub.dev/assets/docs](https://github.com/tensorflow/tfhub.dev/tree/master/assets/docs)/&lt;ID_de_publicador&gt;/&lt;ID_de_publicador.md&gt;

Confira um exemplo mínimo de documentação para o publicador "vtab":

```markdown
# Publisher vtab
Visual Task Adaptation Benchmark

[![Icon URL]](https://storage.googleapis.com/vtab/vtab_logo_120.png)

## VTAB
The Visual Task Adaptation Benchmark (VTAB) is a diverse, realistic and
challenging benchmark to evaluate image representations.
```

O exemplo acima especifica o ID de publicador, o nome do publicador, o caminho do ícone a ser usado e uma documentação de marcação maior com formato livre. É importante saber que o ID de publicador deve conter somente letras minúsculas, números e hifens.

### Diretrizes de nome de publicador

O nome de publicador pode ser seu nome de usuário do GitHub ou o nome da organização do GitHub que você gerencia.

## Formato de marcação da página de modelos {:#model}

A documentação de modelos é um arquivo Markdown com uma sintaxe complementar. Confira abaixo um exemplo mínimo ou [um exemplo de arquivo Markdown mais realista](https://github.com/tensorflow/tfhub.dev/blob/master/examples/docs/tf2_model_example.md).

### Exemplo de documentação

Uma documentação de modelos de alta qualidade contém trechos de código, informações de como o modelo foi treinado e seu uso pretendido. Você também deve usar as propriedades de metadados específicas a modelos [explicadas abaixo](#metadata) para que os usuários consigam encontrar seus modelos em tfhub.dev mais rápido.

```markdown
# Module google/text-embedding-model/1

Simple one sentence description.

<!-- asset-path: https://path/to/text-embedding-model/model.tar.gz -->
<!-- task: text-embedding -->
<!-- fine-tunable: true -->
<!-- format: saved_model_2 -->

## Overview

Here we give more information about the model including how it was trained,
expected use cases, and code snippets demonstrating how to use the model:

```
Code snippet demonstrating use (e.g. for a TF model using the tensorflow_hub library)

import tensorflow_hub as hub

model = hub.KerasLayer(<model name>)
inputs = ...
output = model(inputs)
```
```

### Implantações de modelos e agrupamento de implantações

tfhub.dev permite a publicação de implantações de TF.js, TFLite e Coral de um SavedModel do TensorFlow.

A primeira linha do arquivo Markdown deve especificar o tipo de formato:

- `# Module publisher/model/version` para SavedModels
- `# Tfjs publisher/model/version` para implantações de TF.js
- `# Lite publisher/model/version` para implantações de Lite
- `# Coral publisher/model/version` para implantações de Coral

É uma boa ideia que esses diferentes formatos do mesmo modelo conceitual sejam exibidos na mesma página de modelos em tfhub.dev. Para associar uma determinada implantação de TF.js, TFLite ou Coral a um modelo no formato SavedModel do TensorFlow, especifique a tag parent-model (modelo pai):

```markdown
<!-- parent-model: publisher/model/version -->
```

Às vezes, você vai querer publicar uma ou mais implantações sem um SavedModel do TensorFlow. Nesse caso, você precisará criar um modelo temporário e especificar seu identificador na tag `parent-model`. O Markdown temporário é idêntico ao Markdown de modelo do TensorFlow, exceto pela primeira linha: `# Placeholder publisher/model/version`, além de não precisar da propriedade `asset-path` (caminho dos ativos).

### Propriedades de metadados específicos ao Markdown de modelos {:#metadata}

Os arquivos Markdown podem conter propriedades de metadados, que são usadas para fornecer filtros e marcas para ajudar os usuários a encontrar seu modelo. Os atributos de metadados estão incluídos nos comentários do Markdown após a descrição curta do arquivo Markdown. Por exemplo:

```markdown
# Module google/universal-sentence-encoder/1
Encoder of greater-than-word length text trained on a variety of data.

<!-- task: text-embedding -->
...
```

Há suporte às seguintes propriedades de metadados:

- `format` (formato) – para modelos do TensorFlow: formato do modelo do TensorFlow Hub. Os valores válidos são `hub` quando o modelo é exportado no [formato legado TF1 Hub](exporting_hub_format.md) ou `saved_model_2` quando o modelo é exportado no formato [SavedModel do TF2](exporting_tf2_saved_model.md).
- `asset-path` (caminho dos ativos) – o caminho remoto dos ativos do modelo a carregar que pode ser lido globalmente, como um bucket do Google Cloud Storage. Deve ser possível buscar a URL no mesmo arquivo robots.txt (por esse motivo, não há suporte a "https://github.com/.*/releases/download/.*", pois é proibido por https://github.com/robots.txt). Veja mais informações sobre o tipo de arquivo e conteúdo esperados [abaixo](#model-specific-asset-content).
- `parent-model` (modelo pai) – para modelos do TF.js/TFLite/Coral: identificador do SavedModel/Temporário subjacente
- `fine-tunable` (permite ajustes finos) – booleano, define se o usuário pode fazer ajustes finos no modelo.
- `task` (tarefa) – domínio do problema, por exemplo, "text-embedding" (embedding de texto). Todos os valores permitidos são definidos em [task.yaml](https://github.com/tensorflow/tfhub.dev/blob/master/tags/task.yaml).
- `dataset` – dataset com o qual o modelo foi treinado, por exemplo, "wikipedia". Todos os valores permitidos são definidos em [dataset.yaml](https://github.com/tensorflow/tfhub.dev/blob/master/tags/dataset.yaml).
- `network-architecture` (arquitetura da rede) – arquitetura da rede na qual o modelo é baseado, por exemplo, "mobilenet-v3". Todos os valores permitidos são definidos em [network_architecture.yaml](https://github.com/tensorflow/tfhub.dev/blob/master/tags/network_architecture.yaml).
- `language` (idioma) – código do idioma em que um modelo de texto foi treinado, por exemplo, "en". Todos os valores permitidos são definidos em [language.yaml](https://github.com/tensorflow/tfhub.dev/blob/master/tags/language.yaml).
- `license` (licença) – licença aplicável ao modelo, por exemplo, "mit". A licença presumida padrão é [Licença Apache 2.0](https://opensource.org/licenses/Apache-2.0). Todos os valores permitidos são definidos em [license.yaml](https://github.com/tensorflow/tfhub.dev/blob/master/tags/license.yaml). É importante saber que a licença `custom` (personalizada) requer considerações especiais caso a caso.
- `colab` – URL em HTTPS de um notebook que demonstra como o modelo pode ser usado ou treinado ([exemplo](https://colab.sandbox.google.com/github/tensorflow/docs/blob/master/g3doc/en/hub/tutorials/bigbigan_with_tf_hub.ipynb) para [bigbigan-resnet50](https://tfhub.dev/deepmind/bigbigan-resnet50/1)). Deve levar a `colab.research.google.com`. É importante saber que notebooks Jupyter hospedados no GitHub podem ser acessados via `https://colab.research.google.com/github/ORGANIZATION/PROJECT/ blob/master/.../my_notebook.ipynb`.
- `demo` – URL em HTTPS para um site que demonstra como o modelo do TF.js pode ser usado ([exemplo](https://teachablemachine.withgoogle.com/train/pose) para [posenet](https://tfhub.dev/tensorflow/tfjs-model/posenet/mobilenet/float/075/1/default/1)).
- `interactive-visualizer` (visualizador interativo) – nome do visualizador que deve ser embutido na página de modelos, por exemplo, "vision" (visão). Ao exibir um visualizador, os usuários podem explorar as previsões do modelo interativamente. Todos os valores permitidos são definidos em [interactive_visualizer.yaml](https://github.com/tensorflow/tfhub.dev/blob/master/tags/interactive_visualizer.yaml).

Os tipos de documentação Markdown têm suporte a diferentes propriedades de metadados obrigatórias e opcionais.

Tipo | Obrigatória | Opcional
--- | --- | ---
Publicador |  |
Coleção | task (tarefa) | dataset, language (idioma)
:             :                          : network-architecture                : |  |
Temporário | task (tarefa) | dataset, fine-tunable (permite ajustes finos)
:             :                          : interactive-visualizer, language,   : |  |
:             :                          : license, network-architecture       : |  |
SavedModel | asset-path (caminho dos ativos), task (tarefa) | colab, dataset
:             : fine-tunable, format     : interactive-visualizer, language,   : |  |
:             :                          : license, network-architecture       : |  |
Tfjs | asset-path (caminho dos ativos), parent-model (modelo pai) | colab, demo, interactive-visualizer (visualizador interativo)
Lite | asset-path (caminho dos ativos), parent-model (modelo pai) | colab, interactive-visualizer (visualizador interativo)
Coral | asset-path (caminho dos ativos), parent-model (modelo pai) | colab, interactive-visualizer (visualizador interativo)

### Conteúdo de ativos específicos ao modelo

Dependendo do tipo de modelo, os tipos de arquivo e conteúdo abaixo são esperados:

- SavedModel: arquivo tar.gz contendo conteúdo como:

```
saved_model.tar.gz
├── assets/            # Optional.
├── assets.extra/      # Optional.
├── variables/
│     ├── variables.data-?????-of-?????
│     └──  variables.index
├── saved_model.pb
├── keras_metadata.pb  # Optional, only required for Keras models.
└── tfhub_module.pb    # Optional, only required for TF1 models.
```

- TF.js: arquivo tar.gz contendo conteúdo como:

```
tf_js_model.tar.gz
├── group*
├── *.json
├── *.txt
└── *.pb
```

- TFLite: arquivo .tflite
- Coral: arquivo .tflite

Para arquivos tar.gz: supondo que os arquivos do seu modelo estejam no diretório `my_model` (por exemplo, `my_model/saved_model.pb` para SavedModels ou `my_model/model.json` para modelos do TF.js), você pode criar um arquivo tar.gz válido usado a ferramenta [tar](https://www.gnu.org/software/tar/manual/tar.html) via `cd my_model && tar -czvf ../model.tar.gz *`.

Geralmente, todos os arquivos e diretórios (sejam compactados ou descompactados) precisam começar com uma letra, então pontos não são um prefixo válido de nomes de arquivos e diretórios.

## Formato de marcação da página de coleção {:#collection}

As coleções são um recurso de tfhub.dev que permite aos publicadores agrupar modelos relacionados para melhorar a experiência de busca dos usuários.

Confira a [lista de todas as coleções](https://tfhub.dev/s?subtype=model-family) em tfhub.dev.

O local correto do arquivo de coleção no repositório [github.com/tensorflow/tfhub.dev](https://github.com/tensorflow/tfhub.dev) é [assets/docs](https://github.com/tensorflow/tfhub.dev/tree/master/assets/docs)/<b>&lt;nome_do_publicador&gt;</b>/collections/<b>&lt;nome_da_coleção&gt;</b>/<b>1</b>.md

Veja um exemplo mínimo que iria para /<b>vtab</b>/collections/<b>benchmark</b>/<b>1</b>.md. Observe que o nome da coleção na primeira linha não inclui a parte `collections/`, que é incluída no caminho do arquivo.

```markdown
# Collection vtab/benchmark/1
Collection of visual representations that have been evaluated on the VTAB
benchmark.

<!-- task: image-feature-vector -->

## Overview
This is the list of visual representations in TensorFlow Hub that have been
evaluated on VTAB. Results can be seen in
[google-research.github.io/task_adaptation/](https://google-research.github.io/task_adaptation/)

#### Models
|                   |
|-------------------|
| [vtab/sup-100/1](https://tfhub.dev/vtab/sup-100/1)   |
| [vtab/rotation/1](https://tfhub.dev/vtab/rotation/1) |
|------------------------------------------------------|
```

Esse exemplo especifica o nome da coleção, uma descrição curta (com uma frase), os metadados de domínio do problema e a documentação de Markdown em formato livre.
