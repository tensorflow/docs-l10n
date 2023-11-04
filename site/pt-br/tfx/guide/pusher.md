# O componente do pipeline Pusher TFX

O componente Pusher é usado para enviar um modelo validado para um [destino de implantação](index.md#deployment_targets) durante o treinamento ou retreinamento do modelo. Antes da implantação, o Pusher depende de uma ou mais autorizações (blessings) de outros componentes de validação para decidir se deve ou não enviar o modelo.

- O [Evaluator](evaluator) autoriza (blesses) o modelo se o novo modelo treinado for "bom o suficiente" para ser colocado em produção.
- (Opcional, mas recomendado) O [InfraValidator](infra_validator) autoriza o modelo se o modelo puder ser atendido mecanicamente num ambiente de produção.

Um componente Pusher consome um modelo treinado no formato [SavedModel](/guide/saved_model) e produz o mesmo SavedModel, juntamente com metadados de controle de versão.

## Usando o componente Pusher

Um componente de pipeline Pusher normalmente é muito fácil de implantar e requer pouca personalização, já que todo o trabalho é feito pelo componente Pusher TFX. O código típico é assim:

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

### Enviando um modelo produzido a partir do InfraValidator.

(Da versão 0.30.0)

O InfraValidator também pode produzir o artefato `InfraBlessing` contendo um [modelo com warmup](infra_validator#producing_a_savedmodel_with_warmup), e o Pusher pode enviá-lo como um artefato `Model`.

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

Mais detalhes estão disponíveis na [Referência da API Pusher](https://www.tensorflow.org/tfx/api_docs/python/tfx/v1/components/Pusher).
