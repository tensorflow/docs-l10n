# Como fazer cache de downloads de modelos do TF Hub

## Visão geral

No momento, a biblioteca `tensorflow_hub` tem suporte a apenas dois modos de download de modelos. Por padrão, um modelo é baixado como um arquivo compactado, e é feito cache dele no disco. A segunda forma é ler os modelos diretamente do armazenamento remoto e gravar no TensorFlow. De toda forma, as chamadas a funções do `tensorflow_hub` no código do Python podem e devem continuar usando as URLs canônicas tfhub.dev dos modelos, e pode ser feito sua portabilidade entre sistemas, além de terem uma documentação navegável. No caso raro em que o código do usuário precise do local real do sistema de arquivos (após baixar e descompactar ou após resolver um identificador de modelo para um caminho de sistema de arquivos), esse local pode ser obtido pela função `hub.resolve(handle)`.

### Como fazer cache de downloads compactados

Por padrão, a biblioteca `tensorflow_hub` faz o cache dos modelos no sistema de arquivos quando eles são baixados de tfhub.dev (ou de outros [sites de hospedagem](hosting.md)) e descompactados. Esse modo é recomendado para a maioria dos ambientes, exceto se o espaço em disco for escasso, mas a largura de banda e latência da rede forem excelentes.

O local padrão de downloads é um diretório temporário, mas pode ser personalizado definindo a variável de ambiente `TFHUB_CACHE_DIR` (o que é recomendando) ou passando o sinalizador de linha de comando `--tfhub_cache_dir`. O local padrão de cache`/tmp/tfhub_modules` (ou o resultado de `os.path.join(tempfile.gettempdir(), "tfhub_modules")`) deve funcionar na maioria dos casos.

Usuários que prefiram um cache persistente entre as reinicializações do sistema podem definir `TFHUB_CACHE_DIR` como um local em seu diretório base. Por exemplo, um usuário do shell bash em um sistema Linux pode adicionar a `~/.bashrc` uma linha como a abaixo:

```bash
export TFHUB_CACHE_DIR=$HOME/.cache/tfhub_modules
```

...reiniciar o shell, e então esse local será usado. Ao utilizar um local persistente, lembre-se de que não há limpeza automática.

### Leitura de um armazenamento remoto

Os usuários podem instruir a biblioteca `tensorflow_hub` a ler os modelos diretamente do armazenamento interno (Google Cloud Storage) em vez de baixar os modelos localmente, usando:

```shell
os.environ["TFHUB_MODEL_LOAD_FORMAT"] = "UNCOMPRESSED"
```

ou definindo o sinalizador de linha de comando `--tfhub_model_load_format` como `UNCOMPRESSED` (não compactado). Desta forma, não é necessário um diretório para cache, o que é bastante útil em ambientes com pouco espaço em disco, mas uma conexão de Internet rápida.

### Como executar em TPUs em notebooks do Colab

Em [colab.research.google.com](https://colab.research.google.com), baixar modelos compactados entrará em conflito com o runtime da TPU, pois a carga de trabalho de computação é delegada para outra máquina que não tem acesso ao local do cache por padrão. Nesta situação, há duas soluções alternativas:

#### 1) Usar um bucket do GCS ao qual o worker da TPU tenha acesso

A solução mais fácil é instruir a biblioteca `tensorflow_hub` a ler os modelos do bucket do GCS para o TF Hub, conforme explicado acima. Usuários com seu próprio bucket do GCS podem especificar um diretório em seu bucket como o local do cache com código como este:

```python
import os
os.environ["TFHUB_CACHE_DIR"] = "gs://my-bucket/tfhub-modules-cache"
```

...antes de chamar a biblioteca `tensorflow_hub`.

#### 2) Redirecionar todas as leituras pelo host do Colab

Outra solução alternativa é redirecionar todas as leituras (até mesmo de variáveis grandes) pelo host do Colab:

```python
load_options =
tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')
reloaded_model = hub.load("https://tfhub.dev/...", options=load_options)
```

**Observação:** confira mais informações sobre identificadores válidos [aqui](tf2_saved_model.md#model_handles).
