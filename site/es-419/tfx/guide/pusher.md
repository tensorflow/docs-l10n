# El componente de canalización Pusher TFX

El componente Pusher se usa para insertar un modelo validado en un [objetivo de implementación](index.md#deployment_targets) durante el entrenamiento o reentrenamiento del modelo. Antes de la implementación, Pusher depende de una o más aprobaciones de otros componentes de validación para decidir si inserta o no el modelo.

- [Evaluator](evaluator) aprueba el modelo si el nuevo modelo entrenado es "lo suficientemente bueno" como para insertarlo en producción.
- (Opcional pero recomendado) [InfraValidator](infra_validator) aprueba el modelo si el modelo se puede usar mecánicamente en un entorno de producción.

Un componente Pusher consume un modelo entrenado en formato [SavedModel](/guide/saved_model) y produce el mismo SavedModel, junto con los metadatos de control de versiones.

## Cómo usar el componente Pusher

Un componente de canalización Pusher suele ser muy fácil de implementar y requiere muy poca personalización, ya que el componente Pusher TFX hace la mayor parte del trabajo. El código típico se ve así:

```python
pusher = Pusher(
  model=trainer.outputs['model'],
  model_blessing=evaluator.outputs['blessing'],
  infra_blessing=infra_validator.outputs['blessing'],
  push_destination=tfx.proto.PushDestination(
    filesystem=tfx.proto.PushDestination.Filesystem(
        base_directory=serving_model_dir)
  )
)
```

### Cómo insertar un modelo producido a partir de InfraValidator

(Desde la versión 0.30.0)

InfraValidator también puede producir un artefacto `InfraBlessing` que contiene un [modelo con preparación](infra_validator#producing_a_savedmodel_with_warmup), y Pusher puede impulsarlo como un artefacto `Model`.

```python
infra_validator = InfraValidator(
    ...,
    # make_warmup=True will produce a model with warmup requests in its
    # 'blessing' output.
    request_spec=tfx.proto.RequestSpec(..., make_warmup=True)
)

pusher = Pusher(
    # Push model from 'infra_blessing' input.
    infra_blessing=infra_validator.outputs['blessing'],
    push_destination=tfx.proto.PushDestination(...)
)
```

Hay más detalles disponibles en la [referencia de la API de Pusher](https://www.tensorflow.org/tfx/api_docs/python/tfx/v1/components/Pusher).
