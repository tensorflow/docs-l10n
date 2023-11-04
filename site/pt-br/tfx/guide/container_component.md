# Criando componentes baseados em container

Os componentes baseados em container fornecem flexibilidade para integrar código escrito em qualquer linguagem em seu pipeline, desde que você possa executar esse código num container do Docker.

Se você é novato em pipelines TFX, [aprenda mais sobre os principais conceitos dos pipelines TFX](understanding_tfx_pipelines) antes de continuar.

## Criando um componente baseado em container

Os componentes baseados em container são apoiados por programas de linha de comando em containers. Se você já possui uma imagem de container, pode usar o TFX para criar um componente a partir dela usando a [função `create_container_component`](https://github.com/tensorflow/tfx/blob/master/tfx/dsl/component/experimental/container_component.py){: .external } para declarar entradas e saídas. Parâmetros da função:

- **name:** o nome do componente.
- **inputs:** um dicionário que mapeia nomes de entrada para tipos. <strong>outputs</strong>: um dicionário que mapeia nomes de saída para tipos. <strong>parameters</strong>: um dicionário que mapeia nomes de parâmetros para tipos.
- **image:** nome da imagem do container e, opcionalmente, tag da imagem.
- **command:** linha de comando do ponto de entrada do container. Não executável dentro de um shell. A linha de comando pode usar placeholders que são substituídos no momento da compilação pelo input, output ou parameter. Os placeholders podem ser importados de [`tfx.dsl.component.experimental.placeholders`](https://github.com/tensorflow/tfx/blob/master/tfx/dsl/component/experimental/placeholders.py) {: .external }. Observe que os modelos Jinja não são suportados.

**Valor de retorno:** uma classe Component herdada de base_component.BaseComponent que pode ser instanciada e usada dentro do pipeline.

### Placeholders

Para um componente que possui entradas ou saídas, o `command` geralmente precisa ter placeholders que sejam substituídos por dados reais em tempo de execução. Vários placeholders são fornecidos para essa finalidade:

- `InputValuePlaceholder`: um placeholder para o valor do artefato de entrada. Em tempo de execução, este placeholder é substituído pela representação em string do valor do artefato.

- `InputUriPlaceholder`: um placeholder para o URI do argumento do artefato de entrada. Em tempo de execução, este placeholder é substituído pela URI dos dados do artefato de entrada.

- `OutputUriPlaceholder`: um placeholder para o URI do argumento do artefato de saída. Em tempo de execução, este placeholder é substituído pela URI onde o componente deve armazenar os dados do artefato de saída.

Saiba mais sobre [Placeholders de linha de comando do componente TFX](https://github.com/tensorflow/tfx/blob/master/tfx/dsl/component/experimental/placeholders.py){: .external }.

### Exemplo de componente baseado em container

O código a seguir é um exemplo de um componente não python que baixa, transforma e carrega os dados:

```python
import tfx.v1 as tfx

grep_component = tfx.dsl.components.create_container_component(
    name='FilterWithGrep',
    inputs={
        'text': tfx.standard_artifacts.ExternalArtifact,
    },
    outputs={
        'filtered_text': tfx.standard_artifacts.ExternalArtifact,
    },
    parameters={
        'pattern': str,
    },
    # The component code uses gsutil to upload the data to Google Cloud Storage, so the
    # container image needs to have gsutil installed and configured.
    image='google/cloud-sdk:278.0.0',
    command=[
        'sh', '-exc',
        '''
          pattern="$1"
          text_uri="$3"/data  # Adding suffix, because currently the URI are "directories". This will be fixed soon.
          text_path=$(mktemp)
          filtered_text_uri="$5"/data  # Adding suffix, because currently the URI are "directories". This will be fixed soon.
          filtered_text_path=$(mktemp)

          # Getting data into the container
          gsutil cp "$text_uri" "$text_path"

          # Running the main code
          grep "$pattern" "$text_path" >"$filtered_text_path"

          # Getting data out of the container
          gsutil cp "$filtered_text_path" "$filtered_text_uri"
        ''',
        '--pattern', tfx.dsl.placeholders.InputValuePlaceholder('pattern'),
        '--text', tfx.dsl.placeholders.InputUriPlaceholder('text'),
        '--filtered-text', tfx.dsl.placeholders.OutputUriPlaceholder('filtered_text'),
    ],
)
```
