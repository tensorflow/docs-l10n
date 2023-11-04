# Protocolo de hospedagem de modelos

Este documento descreve as convenções de URL ao hospedar todos os tipos de modelos em [tfhub.dev](https://tfhub.dev) – modelos do  TFJS, TF Lite e TensorFlow. Descreve também o protocolo baseado em HTTP(S) implementado pela biblioteca `tensorflow_hub` para carregar modelos do TensorFlow de [tfhub.dev](https://tfhub.dev) e serviços compatíveis nos programas do TensorFlow.

Sua principal característica é usar a mesma URL no código para carregar um modelo e no navegador para consultar a documentação do modelo.

## Conversões gerais de URL

[tfhub.dev](https://tfhub.dev) tem suporte aos seguintes formatos de URL:

- Os publicadores do TF Hub seguem `https://tfhub.dev/<publisher>`
- As coleções do TF Hub seguem `https://tfhub.dev/<publisher>/collection/<collection_name>`
- Os modelos do TF Hub têm URL versionada `https://tfhub.dev/<publisher>/<model_name>/<version>` e URL não versionada `https://tfhub.dev/<publisher>/<model_name>` resolvida para a versão mais recente do modelo.

Os modelos do TF Hub podem ser baixados como ativos compactados acrescentando os parâmetros de URL à URL do modelo em [tfhub.dev](https://tfhub.dev). Porém, os parâmetros de URL necessários dependem do tipo de modelo:

- Modelos do TensorFlow (formatos SavedModel e TF1 Hub): acrescente `?tf-hub-format=compressed` à URL do modelo do TensorFlow.
- Modelos do TFJS: acrescente `?tfjs-format=compressed` à URL do modelo do TFJS para baixar o arquivo compactado ou `/model.json?tfjs-format=file` para ler, se for um armazenamento remoto.
- Modelos do TF Lite: acrescente `?lite-format=tflite` à URL do modelo do TF Lite.

Por exemplo:

<table style="width: 100%;">
  <tr style="text-align: center">
    <col style="width: 10%">
    <col style="width: 20%">
    <col style="width: 15%">
    <col style="width: 30%">
    <col style="width: 25%">
    <td style="text-align: center; background-color: #D0D0D0">Tipo</td>
    <td style="text-align: center; background-color: #D0D0D0">URL do modelo</td>
    <td style="text-align: center; background-color: #D0D0D0">Tipo de download</td>
    <td style="text-align: center; background-color: #D0D0D0">Parâmetro de URL</td>
    <td style="text-align: center; background-color: #D0D0D0">URL para download</td>
  </tr>
  <tr>
    <td>TensorFlow (formatos SavedModel, TF1 Hub)</td>
    <td>https://tfhub.dev/google/spice/2</td>
    <td>.tar.gz</td>
    <td>?tf-hub-format=compressed</td>
    <td>https://tfhub.dev/google/spice/2?tf-hub-format=compressed</td>
  </tr>
  <tr>
    <td>TF Lite</td>
    <td>https://tfhub.dev/google/lite-model/spice/1</td>
    <td>.tflite</td>
    <td>?lite-format=tflite</td>
    <td>https://tfhub.dev/google/lite-model/spice/1?lite-format=tflite</td>
  </tr>
  <tr>
    <td>TF.js</td>
    <td>https://tfhub.dev/google/tfjs-model/spice/2/default/1</td>
    <td>.tar.gz</td>
    <td>?tfjs-format=compressed</td>
    <td>https://tfhub.dev/google/tfjs-model/spice/2/default/1?tfjs-format=compressed</td>
  </tr>
</table>

Além disso, alguns modelos também são hospedados em um formato que pode ler lido diretamente de armazenamentos remotos sem precisar baixá-los. Isso é particularmente útil quando não há um armazenamento local disponível, como ao executar um modelo do TF.js model no navegador ou ao carregar um SavedModel no [Colab](https://colab.research.google.com/). Lembre-se de que ler modelos hospedados remotamente sem baixá-los localmente pode aumentar a latência.

<table style="width: 100%;">
  <tr style="text-align: center">
    <col style="width: 10%">
    <col style="width: 20%">
    <col style="width: 15%">
    <col style="width: 30%">
    <col style="width: 25%">
    <td style="text-align: center; background-color: #D0D0D0">Tipo</td>
    <td style="text-align: center; background-color: #D0D0D0">URL do modelo</td>
    <td style="text-align: center; background-color: #D0D0D0">Tipo de resposta</td>
    <td style="text-align: center; background-color: #D0D0D0">Parâmetro de URL</td>
    <td style="text-align: center; background-color: #D0D0D0">URL para solicitação</td>
  </tr>
  <tr>
    <td>TensorFlow (formatos SavedModel, TF1 Hub)</td>
    <td>https://tfhub.dev/google/spice/2</td>
    <td>String (caminho para pasta do GCS onde o modelo descompactado é armazenado)</td>
    <td>?tf-hub-format=uncompressed</td>
    <td>https://tfhub.dev/google/spice/2?tf-hub-format=uncompressed</td>
  </tr>
  <tr>
    <td>TF.js</td>
    <td>https://tfhub.dev/google/tfjs-model/spice/2/default/1</td>
    <td>.json</td>
    <td>?tfjs-format=file</td>
    <td>https://tfhub.dev/google/tfjs-model/spice/2/default/1/model.json?tfjs-format=file</td>
  </tr>
</table>

## Protocolo da biblioteca tensorflow_hub

Esta seção descreve como hospedamos modelos em [tfhub.dev](https://tfhub.dev) para uso com a biblioteca tensorflow_hub. Se você quiser hospedar seu próprio repositório de modelos de modo que funcione com a biblioteca tensorflow_hub, seu serviço de distribuição HTTP(S) deve fornecer uma implementação deste protocolo.

Observação; esta seção não aborda a hospedagem de modelos do TF Lite e TFJS, já que eles não são baixados pela biblioteca `tensorflow_hub`. Confira mais informações sobre como hospedar esses tipos de modelos [acima](#general-url-conventions).

### Hospedagem compactada

Os modelos são armazenados em [tfhub.dev](https://tfhub.dev) como arquivos tar.gz compactados. Por padrão, a biblioteca tensorflow_hub baixa automaticamente o modelo compactado. Eles também podem ser baixados manualmente acrescentando-se `?tf-hub-format=compressed` à URL do modelo. Por exemplo:

```shell
wget https://tfhub.dev/tensorflow/albert_en_xxlarge/1?tf-hub-format=compressed
```

A raiz do arquivo é a raiz do diretório do modelo e deve conter um SavedModel, como neste exemplo:

```shell
# Create a compressed model from a SavedModel directory.
$ tar -cz -f model.tar.gz --owner=0 --group=0 -C /tmp/export-model/ .

# Inspect files inside a compressed model
$ tar -tf model.tar.gz
./
./variables/
./variables/variables.data-00000-of-00001
./variables/variables.index
./assets/
./saved_model.pb
```

Arquivos tarball para uso com o formato legado TF1 Hub também conterão um arquivo <code>./tfhub_module.pb</code>.

Quando uma das APIs de carregamento de modelos da biblioteca `tensorflow_hub` é invocada ([hub.KerasLayer](https://www.tensorflow.org/hub/api_docs/python/hub/KerasLayer), [hub.load](https://www.tensorflow.org/hub/api_docs/python/hub/load), etc), a biblioteca baixa o modelo, descompacta-o e faz o cache localmente. A biblioteca `tensorflow_hub` espera que as URLs do modelo estejam versionadas e que o conteúdo do modelo de uma determinada versão seja imutável para que possa fazer o cache indefinidamente. Saiba mais sobre [como fazer cache de modelos](caching.md).

![](https://raw.githubusercontent.com/tensorflow/hub/master/docs/images/library_download_cache.png)

### Hospedagem descompactada

Quando a variável de ambiente `TFHUB_MODEL_LOAD_FORMAT` ou o sinalizador de linha de comando `--tfhub_model_load_format` é definido como `UNCOMPRESSED` (não comprimido), o modelo é lido diretamente do armazenamento remoto (Google Cloud Storage) em vez de ser baixado e descompactado localmente. Quando esse comportamento é ativado, a biblioteca acrescenta `?tf-hub-format=uncompressed` à URL do modelo. Essa solicitação retorna o caminho da pasta no GCS que contém os arquivos descompactados do modelo. Como exemplo, <br> `https://tfhub.dev/google/spice/2?tf-hub-format=uncompressed` <br> retorna <br> `gs://tfhub-modules/google/spice/2/uncompressed` no corpo da resposta 303. Em seguida, a biblioteca lê o modelo no destino do GCS.
