# Cómo compilar componentes basados ​​en contenedores

Los componentes basados ​​en contenedores ofrecen la flexibilidad de integrar código escrito en cualquier idioma en su canalización, siempre que pueda ejecutar ese código en un contenedor Docker.

Si no tiene experiencia con las canalizaciones de TFX, [obtenga más información sobre los conceptos básicos de las canalizaciones de TFX](understanding_tfx_pipelines).

## Cómo crear un componente basado ​​en contenedores

Los componentes basados ​​en contenedores están respaldados por programas de línea de comandos en contenedores. Si ya tiene una imagen de contenedor, puede usar TFX para crear un componente a partir de ella con la [función `create_container_component`](https://github.com/tensorflow/tfx/blob/master/tfx/dsl/component/experimental/container_component.py) {: .external } para declarar entradas y salidas. Parámetros de función:

- **nombre:** El nombre del componente.
- **entradas:** un diccionario que asigna nombres de entradas a tipos. salidas: un diccionario que asigna nombres de salida a tipos. parámetros: un diccionario que asigna nombres de parámetros a tipos.
- **imagen:** nombre de la imagen del contenedor y, opcionalmente, etiqueta de imagen.
- **comando:** línea de comando del punto de entrada del contenedor. No se ejecuta dentro de un shell. La línea de comando puede usar objetos marcadores de posición que se reemplazan en el momento de la compilación por la entrada, la salida o el parámetro. Los objetos de marcador de posición se pueden importar desde [`tfx.dsl.component.experimental.placeholders`](https://github.com/tensorflow/tfx/blob/master/tfx/dsl/component/experimental/placeholders.py) {: .external }. Tenga en cuenta que las plantillas de Jinja no son compatibles.

**Valor de retorno:** una clase de componente que hereda de base_component.BaseComponent y que se puede crear una instancia y usar dentro de la canalización.

### Marcadores de posición

Para un componente que tiene entradas o salidas, el `command` a menudo necesita marcadores de posición que se reemplazan con datos reales en tiempo de ejecución. Para ello se proporcionan varios marcadores de posición:

- `InputValuePlaceholder`: un marcador de posición para el valor del artefacto de entrada. En tiempo de ejecución, este marcador de posición se reemplaza con la representación de cadena del valor del artefacto.

- `InputUriPlaceholder`: un marcador de posición para el URI del argumento del artefacto de entrada. En tiempo de ejecución, este marcador de posición se reemplaza con el URI de los datos del artefacto de entrada.

- `OutputUriPlaceholder`: un marcador de posición para el URI del argumento del artefacto de salida. En tiempo de ejecución, este marcador de posición se reemplaza con el URI donde el componente debe almacenar los datos del artefacto de salida.

Obtenga más información sobre los [marcadores de posición de la línea de comandos del componente de TFX](https://github.com/tensorflow/tfx/blob/master/tfx/dsl/component/experimental/placeholders.py) {: .external }.

### Ejemplo de componente basado en contenedores

El siguiente es un ejemplo de un componente que no es de Python que descarga, transforma y carga los datos:

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
