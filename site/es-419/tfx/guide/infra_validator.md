# El componente de canalización InfraValidator TFX

InfraValidator es un componente de TFX que se usa como capa de alerta temprana antes de llevar un modelo a producción. El nombre "infra" validador proviene del hecho de que valida el modelo en el modelo real que sirve "infraestructura". Si [Evaluator](evaluator.md) debe garantizar el rendimiento del modelo, InfraValidator debe garantizar que el modelo sea mecánicamente correcto y evitar que se impulsen modelos defectuosos.

## ¿Cómo funciona?

InfraValidator toma el modelo, inicia un servidor de modelos en un espacio aislado con el modelo y comprueba si se puede cargar correctamente y, opcionalmente, consultar. El resultado de la validación de infraestructura se va a generar en la salida `blessing` de la misma manera que lo hace [Evaluator](evaluator.md).

InfraValidator se centra en la compatibilidad entre el sistema binario del servidor de modelos (por ejemplo, [TensorFlow Serving](serving.md)) y el modelo que se va a implementar. A pesar del nombre "infra" validador, el **usuario es responsable** de configurar el entorno correctamente, y el infra validador solo interactúa con el servidor de modelos en el entorno configurado por el usuario para ver si funciona bien. Si este entorno se configura correctamente, la validación de la infraestructura o un error de validación nos indicarán si el modelo se puede usar o no en el entorno de producción. Esto implica, entre otros aspectos, lo siguiente:

1. InfraValidator usa el mismo sistema binario del servidor de modelos que se usará en producción. Este es el nivel mínimo al que debe converger el entorno de validación de infraestructura.
2. InfraValidator usa los mismos recursos (por ejemplo, cantidad de asignación y tipo de CPU, memoria y aceleradores) que se usarán en producción.
3. InfraValidator usa la misma configuración de servidor de modelos que se usará en producción.

Dependiendo de la situación, los usuarios pueden elegir hasta qué punto InfraValidator debe ser idéntico al entorno de producción. Técnicamente, se puede validar la infraestructura de un modelo en un entorno Docker local y luego se puede servir en un entorno completamente diferente (por ejemplo, un clúster de Kubernetes) sin problemas. Sin embargo, InfraValidator no habrá comprobado esta divergencia.

### Modo de operación

En función de la configuración, la validación de infraestructura se lleva a cabo en uno de los siguientes modos:

- Modo `LOAD_ONLY`: comprueba si el modelo se cargó correctamente en la infraestructura de servicio o no. **O**
- Modo `LOAD_AND_QUERY`: Modo `LOAD_ONLY` más el envío de algunas solicitudes de muestra para comprobar si el modelo es capaz de servir inferencias. A InfraValidator no le importa si la predicción fue correcta o no. Solo le importa si la solicitud tuvo éxito o no.

## ¿Como lo uso?

Por lo general, InfraValidator se define junto a un componente Evaluator y su salida se envía a un Pusher. Si InfraValidator falla, el modelo no se envía.

```python
evaluator = Evaluator(
    model=trainer.outputs['model'],
    examples=example_gen.outputs['examples'],
    baseline_model=model_resolver.outputs['model'],
    eval_config=tfx.proto.EvalConfig(...)
)

infra_validator = InfraValidator(
    model=trainer.outputs['model'],
    serving_spec=tfx.proto.ServingSpec(...)
)

pusher = Pusher(
    model=trainer.outputs['model'],
    model_blessing=evaluator.outputs['blessing'],
    infra_blessing=infra_validator.outputs['blessing'],
    push_destination=tfx.proto.PushDestination(...)
)
```

### Cómo configurar un componente InfraValidator

Hay tres tipos de protos para configurar InfraValidator.

#### `ServingSpec`

`ServingSpec` es la configuración más importante para InfraValidator. Define lo siguiente:

- <u>qué</u> tipo de servidor de modelos ejecutar
- <u>dónde</u> ejecutarlo

Para los tipos de servidores de modelos (denominados binario de servicio) admitimos lo que sigue:

- [TensorFlow Serving](serving.md)

Nota: InfraValidator permite especificar varias versiones del mismo tipo de servidor de modelos para actualizar la versión del servidor de modelos sin perjudicar la compatibilidad del modelo. Por ejemplo, el usuario puede probar la imagen `tensorflow/serving` con la versión `2.1.0` y `latest`, para asegurarse de que el modelo también será compatible con la última versión `tensorflow/serving`.

Actualmente se admiten las siguientes plataformas de servicio:

- Docker local (Docker debe instalarse con anticipación)
- Kubernetes (compatibilidad limitada solo para KubeflowDagRunner)

La elección de binario de servicio y plataforma de servicio se hace mediante la especificación de un bloque [`oneof`](https://developers.google.com/protocol-buffers/docs/proto3#oneof) de `ServingSpec`. Por ejemplo, para usar el binario TensorFlow Serving que se ejecuta en el clúster de Kubernetes, se deben establecer los campos `tensorflow_serving` y `kubernetes`.

```python
infra_validator=InfraValidator(
    model=trainer.outputs['model'],
    serving_spec=tfx.proto.ServingSpec(
        tensorflow_serving=tfx.proto.TensorFlowServing(
            tags=['latest']
        ),
        kubernetes=tfx.proto.KubernetesConfig()
    )
)
```

Para configurar aún más `ServingSpec`, consulte la [definición de protobuf](https://github.com/tensorflow/tfx/blob/master/tfx/proto/infra_validator.proto).

#### `ValidationSpec`

Configuración opcional para ajustar los criterios de validación de infraestructura o el flujo de trabajo.

```python
infra_validator=InfraValidator(
    model=trainer.outputs['model'],
    serving_spec=tfx.proto.ServingSpec(...),
    validation_spec=tfx.proto.ValidationSpec(
        # How much time to wait for model to load before automatically making
        # validation fail.
        max_loading_time_seconds=60,
        # How many times to retry if infra validation fails.
        num_tries=3
    )
)
```

Todos los campos de ValidationSpec tienen un valor predeterminado válido. Consulte más detalles en la [definición de protobuf](https://github.com/tensorflow/tfx/blob/master/tfx/proto/infra_validator.proto).

#### `RequestSpec`

Configuración opcional para especificar cómo generar solicitudes de muestra cuando se ejecuta la validación de infraestructura en modo `LOAD_AND_QUERY`. Para usar el modo `LOAD_AND_QUERY`, se deben especificar tanto las propiedades de ejecución `request_spec` como el canal de entrada `examples` en la definición del componente.

```python
infra_validator = InfraValidator(
    model=trainer.outputs['model'],
    # This is the source for the data that will be used to build a request.
    examples=example_gen.outputs['examples'],
    serving_spec=tfx.proto.ServingSpec(
        # Depending on what kind of model server you're using, RequestSpec
        # should specify the compatible one.
        tensorflow_serving=tfx.proto.TensorFlowServing(tags=['latest']),
        local_docker=tfx.proto.LocalDockerConfig(),
    ),
    request_spec=tfx.proto.RequestSpec(
        # InfraValidator will look at how "classification" signature is defined
        # in the model, and automatically convert some samples from `examples`
        # artifact to prediction RPC requests.
        tensorflow_serving=tfx.proto.TensorFlowServingRequestSpec(
            signature_names=['classification']
        ),
        num_examples=10  # How many requests to make.
    )
)
```

### Cómo producir un SavedModel con preparación

(Desde la versión 0.30.0)

Como InfraValidator valida el modelo con solicitudes reales, puede reutilizar fácilmente estas solicitudes de validación como [solicitudes de preparación](https://www.tensorflow.org/tfx/serving/saved_model_warmup) de un SavedModel. InfraValidator ofrece una opción (`RequestSpec.make_warmup`) para exportar un modelo guardado con preparación.

```python
infra_validator = InfraValidator(
    ...,
    request_spec=tfx.proto.RequestSpec(..., make_warmup=True)
)
```

Luego, el artefacto `InfraBlessing` de salida contendrá un SavedModel con preparación y también puede ser enviado por [Pusher](pusher.md), al igual que el artefacto `Model`.

## Limitaciones

El InfraValidator actual aún no está completo y tiene algunas limitaciones.

- Solo se puede validar el formato del modelo [SavedModel](/guide/saved_model) de TensorFlow.

- Al ejecutar TFX en Kubernetes, `KubeflowDagRunner` debe ejecutar la canalización dentro de Kubeflow Pipelines. El servidor de modelos se iniciará en el mismo clúster de Kubernetes y en el mismo espacio de nombres que usa Kubeflow.

- InfraValidator se centra principalmente en implementaciones en [TensorFlow Serving](serving.md) y, aunque todavía es útil, es menos preciso para implementaciones en [TensorFlow Lite](/lite) y [TensorFlow.js](/js) u otros marcos de inferencia.

- En el modo `LOAD_AND_QUERY` hay una compatibilidad limitada con la firma del método [Predict](/versions/r1.15/api_docs/python/tf/saved_model/predict_signature_def) (que es el único método exportable en TensorFlow 2). InfraValidator requiere que la firma Predict consuma un [`tf.Example`](/tutorials/load_data/tfrecord#tfexample) serializado como única entrada.

    ```python
    @tf.function
    def parse_and_run(serialized_example):
      features = tf.io.parse_example(serialized_example, FEATURES)
      return model(features)

    model.save('path/to/save', signatures={
      # This exports "Predict" method signature under name "serving_default".
      'serving_default': parse_and_run.get_concrete_function(
          tf.TensorSpec(shape=[None], dtype=tf.string, name='examples'))
    })
    ```

    - Consulte un código de muestra de [ejemplo Penguin](https://github.com/tensorflow/tfx/blob/master/tfx/examples/penguin/penguin_pipeline_local_infraval.py) para ver cómo interactúa esta firma con otros componentes en TFX.
