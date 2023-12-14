# Apache Beam y TFX

[Apache Beam](https://beam.apache.org/) ofrece un marco para ejecutar trabajos de procesamiento de datos por lotes y en streaming que se ejecutan en una variedad de motores de ejecución. Varias de las bibliotecas de TFX usan Beam para ejecutar tareas, lo que permite un alto grado de escalabilidad entre clústeres informáticos. Beam es compatible con diversos motores de ejecución o "ejecutores", incluido un ejecutor directo que se ejecuta en un único nodo informático y es muy útil para desarrollo, pruebas o implementaciones pequeñas. Beam proporciona una capa de abstracción que permite que TFX se ejecute en cualquier ejecutor compatible sin modificaciones de código. TFX usa la API de Beam para Python, por lo que está limitado a los ejecutores compatibles con la API de Python.

## Implementación y escalabilidad

A medida que aumentan los requisitos de carga de trabajo, Beam puede escalar a implementaciones muy grandes en grandes clústeres informáticos. Esto solo está limitado por la escalabilidad del ejecutor subyacente. Los ejecutores en implementaciones grandes generalmente se implementan en un sistema de orquestación de contenedores como Kubernetes o Apache Mesos para automatizar la implementación, el escalado y la gestión de aplicaciones.

Consulte la documentación de [Apache Beam](https://beam.apache.org/) para obtener más información sobre Apache Beam.

Para los usuarios de Google Cloud, se recomienda el uso de [Dataflow](https://cloud.google.com/dataflow), que ofrece una plataforma sin servidor y rentable a través de escalado automático de recursos, reajuste dinámico del trabajo, una sólida integración con otros servicios de Google Cloud, seguridad integrada y monitoreo.

## Código Python personalizado y dependencias

Una complejidad notable del uso de Beam en una canalización de TFX es la gestión de código personalizado y/o las dependencias necesarias de módulos adicionales de Python. A continuación, se muestran algunos ejemplos de cuándo esto podría ser un problema:

- preprocessing_fn debe hacer referencia al módulo de Python del propio usuario
- un extractor personalizado para el componente evaluador
- módulos personalizados que se subclasifican a partir de un componente de TFX

TFX depende del soporte de Beam para la [Gestión de dependencias de canalización de Python](https://beam.apache.org/documentation/sdks/python-pipeline-dependencies/) con el fin de gestionar las dependencias de Python. Actualmente hay dos formas de gestionar esto:

1. Proporcionar código Python y dependencias como paquete fuente
2. [Solo Dataflow] Usar una imagen de contenedor como trabajador

Estos se analizan a continuación.

### Cómo proporcionar código Python y dependencias como un paquete fuente

Esto se recomienda para usuarios que cumplan los siguientes requisitos:

1. Estén familiarizados con el empaquetado de Python y
2. solo usen el código fuente de Python (es decir, no utilice módulos C ni bibliotecas compartidas).

Siga una de las rutas en [Gestión de dependencias de canalización de Python](https://beam.apache.org/documentation/sdks/python-pipeline-dependencies/) para proporcionar esto con uno de los siguientes beam_pipeline_args:

- --setup_file
- --extra_package
- --requirements_file

Aviso: En cualquiera de los casos anteriores, asegúrese de que la misma versión de `tfx` aparezca como dependencia.

### [Solo Dataflow] Cómo usar una imagen de contenedor para un trabajador

TFX 0.26.0 y versiones posteriores ofrecen compatibilidad experimental para el uso de [imágenes de contenedor personalizadas](https://beam.apache.org/documentation/runtime/environments/#customizing-container-images) para trabajadores de Dataflow.

Para poder usar esto, tiene que seguir estos pasos:

- Crear una imagen de Docker que tenga `tfx` y el código personalizado y las dependencias de los usuarios preinstalados.
    - Para los usuarios que (1) usan `tfx>=0.26` y (2) usan python 3.7 puedan desarrollar sus canalizaciones, la forma más sencilla de hacerlo es extender la versión correspondiente de la imagen oficial `tensorflow/tfx`:

```Dockerfile
# You can use a build-arg to dynamically pass in the
# version of TFX being used to your Dockerfile.

ARG TFX_VERSION
FROM tensorflow/tfx:${TFX_VERSION}
# COPY your code and dependencies in
```

- Inserte la imagen creada en un registro de imágenes contenedor al que pueda acceder desde el proyecto usado por Dataflow.
    - Los usuarios de Google Cloud pueden optar por el uso de [Cloud Build](https://cloud.google.com/cloud-build/docs/quickstart-build), que automatiza muy bien los pasos anteriores.
- Proporcione los siguientes `beam_pipeline_args`:

```python
beam_pipeline_args.extend([
    '--runner=DataflowRunner',
    '--project={project-id}',
    '--worker_harness_container_image={image-ref}',
    '--experiments=use_runner_v2',
])
```

**TODO(b/171733562): elimine use_runner_v2 una vez que sea el predeterminado para Dataflow.**

**TODO(b/179738639): cree documentación sobre cómo probar un contenedor personalizado a nivel local después de https://issues.apache.org/jira/browse/BEAM-5440.**

## Argumentos de canalización de Beam

Varios componentes de TFX dependen de Beam para el procesamiento de datos distribuidos. Se configuran con `beam_pipeline_args`, que se especifica durante la creación de la canalización:

```python
my_pipeline = Pipeline(
    ...,
    beam_pipeline_args=[...])
```

TFX 0.30 y sus versiones posteriores agregan una interfaz, `with_beam_pipeline_args`, para extender los argumentos de Beam a nivel de canalización por componente:

```python
example_gen = CsvExampleGen(input_base=data_root).with_beam_pipeline_args([...])
```
