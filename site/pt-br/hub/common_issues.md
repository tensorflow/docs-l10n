# Problemas comuns

Caso seu problema não esteja indicado aqui, pesquise nos [issues do GitHub](https://github.com/tensorflow/hub/issues) antes de abrir um novo issue.

**Observação:** esta documentação usa identificadores de URL TFhub.dev nos exemplos. Confira mais informações sobre outros tipos de identificador válidos [aqui](tf2_saved_model.md#model_handles).

## TypeError: 'AutoTrackable' object is not callable (objeto 'AutoTrackable' não é chamável)

```python
# BAD: Raises error
embed = hub.load('https://tfhub.dev/google/nnlm-en-dim128/1')
embed(['my text', 'batch'])
```

Este erro é gerado com frequência ao carregar modelos com o formato TF1 Hub com a API `hub.load()` no TF2. Adicionar a assinatura correta deve corrigir esse problema. Leia o [guia de migração do TF-Hub para o TF2](migration_tf2.md) para ver mais detalhes sobre como migrar para o TF2 e usar os modelos com o formato TF1 Hub no TF2.

```python

embed = hub.load('https://tfhub.dev/google/nnlm-en-dim128/1')
embed.signatures['default'](['my text', 'batch'])
```

## Não é possível baixar um módulo

Ao usar um modelo de uma URL, diversos erros podem surgir devido à pilha da rede. Geralmente, é um problema específico da máquina que está executando o código, e não um problema com a biblioteca. Veja uma lista de erros comuns:

- **"EOF occurred in violation of protocol"** (EOF ocorreu em violação do protocolo) – Geralmente, esse problema é gerado quando a versão do Python instalada não tem suporte aos requisitos de TLS do servidor que está hospedando o modelo. Notavelmente, sabe-se que o Python 2.7.5 não consegue resolver módulos do domínio tfhub.dev. **CORREÇÃO**: atualize para uma versão mais recente do Python.

- **"cannot verify tfhub.dev's certificate"** (Não foi possível verificar o certificado de tfhub.dev) – Geralmente, esse problema é gerado se alguma coisa na rede estiver tentando atuar como o gTLD (domínio genérico de nível superior) de dev. Antes de .dev ser usado como gTLD, às vezes os desenvolvedores e frameworks usavam nomes .dev para ajudar a testar o código. **CORREÇÃO:** identifique e reconfigure o software que intercepta a resolução de nomes para o domínio ".dev".

- Falhas ao gravar no diretório de cache `/tmp/tfhub_modules` (ou similar): confira o que é e como alterar o local em [Como fazer cache](caching.md).

Se você encontrar os erros acima e as correções indicadas não funcionarem, pode tentar baixar manualmente um módulo simulando o protocolo, acrescentando `?tf-hub-format=compressed` à URL para baixar o arquivo tar compactado que precisa ser descompactado manualmente em um arquivo local. Em seguida, o caminho do arquivo local pode ser usado no lugar da URL. Veja um exemplo rápido:

```bash
# Create a folder for the TF hub module.
$ mkdir /tmp/moduleA
# Download the module, and uncompress it to the destination folder. You might want to do this manually.
$ curl -L "https://tfhub.dev/google/universal-sentence-encoder/2?tf-hub-format=compressed" | tar -zxvC /tmp/moduleA
# Test to make sure it works.
$ python
> import tensorflow_hub as hub
> hub.Module("/tmp/moduleA")
```

## Como executar a inferência em um módulo pré-inicializado

Se você estiver escrevendo um programa do Python que aplique um módulo diversas vezes aos dados de entrada, pode seguir as seguintes receitas (observação: para atender a solicitações em serviços de produção, considere o [TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving) ou outras soluções sem Python que sejam escaláveis).

Supondo que o caso de uso do seu modelo seja **inicialização** e **solicitações** subsequentes (por exemplo, Django, Flask, servidor HTTP personalizado, etc.), você pode considerar o serviço da seguinte forma:

### SavedModels do TF2

- Na parte de inicialização:
    - Carregue o modelo do TF2.0.

```python
import tensorflow_hub as hub

embedding_fn = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
```

- Na parte de solicitação:
    - Use a função de embedding para executar a inferência.

```python
embedding_fn(["Hello world"])
```

Essa chamada a uma tf.function é otimizada para ter bom desempenho. Confira o [guia da tf.function](https://www.tensorflow.org/guide/function).

### Módulos do TF1 Hub

- Na parte de inicialização:
    - Construa o grafo com um **placeholder** – ponto de entrada do grafo.
    - Inicialize a sessão.

```python
import tensorflow as tf
import tensorflow_hub as hub

# Create graph and finalize (finalizing optional but recommended).
g = tf.Graph()
with g.as_default():
  # We will be feeding 1D tensors of text into the graph.
  text_input = tf.placeholder(dtype=tf.string, shape=[None])
  embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")
  embedded_text = embed(text_input)
  init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
g.finalize()

# Create session and initialize.
session = tf.Session(graph=g)
session.run(init_op)
```

- Na parte de solicitação:
    - Use a sessão para alimentar o grafo com dados por meio do placeholder.

```python
result = session.run(embedded_text, feed_dict={text_input: ["Hello world"]})
```

## Não é possível alterar o dtype de um modelo (por exemplo, de float32 para bfloat16)

Os SavedModels do TensorFlow (compartilhados no TF Hub ou de alguma outra forma) contêm operações que funcionam em tipos de dados fixos (geralmente, float32 para os pesos e ativações intermediárias de redes neurais). Essas operações não podem ser alteradas após carregar o SavedModel (mas os publicadores de modelos podem optar por publicar modelos diferentes com tipos de dados diferentes).

## Atualize a versão de um modelo

Os metadados da documentação de versões do modelo podem ser atualizados. Entretanto, os ativos (arquivos do modelo) são imutáveis. Se você quiser alterar os ativos do modelo, pode publicar uma versão mais recente. É uma prática recomendada atualizar a documentação com um registro de alterações descrevendo o que mudou entre as versões.
